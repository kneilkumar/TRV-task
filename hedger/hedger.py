#!/usr/bin/env python3
"""Hedger: keeps the desk's position on EACH instrument (AAH6/AAM6/AAU6)
below a dollar-denominated cap, independently per instrument.

Policy, settled per design session (supersedes the old single-feed,
ticks-through-book-ramp version -- see hedger_old.py for history):

  - Three instruments, three fully independent flatteners. No cross-
    instrument netting via rel/driver-equivalent -- each feed's position is
    tracked and flattened on its own book.
  - Position is dollar-denominated: |position * price| compared against a
    tunable cap (HEDGE_CAP_USD, default $10,000), not a raw lot count --
    keeps AAH6/AAM6/AAU6 on a consistent risk footing despite trading at
    different price levels.
  - Reconstructs desk position purely by watching ex.md.<FEED>.* fill
    messages and checking whether the quoter's or taker's sender shows up
    as either party to an execution. No cooperation needed from those
    processes for position tracking itself.
  - We already know "the currently correct price" -- same fair-value
    signal the quoter's own PGYB_CROSS logic uses (cEst for piggybacks,
    fv(AAH6) directly for the driver) -- so hedging isn't a blind sweep
    through the book. Read via the DESK_STATE KV bucket, published by the
    quoter on every recompute (i.e. every driver tick). This throws away
    nothing: the hedger prices its flatten order right at the level the
    quoter already knows is correct, not some generic ticks-through-bbo
    ramp.
  - Urgency no longer controls price aggression via a blind ramp. It
    controls how far PAST the known-correct price to reach, as a small
    escalating cushion -- start right at the known-correct price; the
    longer/further we stay pinned over cap, ratchet a few extra ticks
    through the book on each retry to guarantee the sweep actually clears,
    rather than assuming one perfectly-priced F always fully fills.
  - On tripping the cap: publish a pause flag to the HEDGE_STATE KV bucket
    (separate bucket from DESK_STATE -- two writers on one bucket risks
    clobbering each other's fields; quoter writes/owns DESK_STATE, hedger
    writes/owns HEDGE_STATE) so the quoter stops firing new PGYB_CROSS
    orders on that instrument while we work it down. Otherwise the quoter
    keeps adding to the very pile we're trying to clear -- sitting tight
    alone does nothing since sitting tight doesn't change desk position,
    it just stops it from getting worse while the hedger fights entropy.
  - Target on trip is a full flatten down to HEDGE_TARGET_USD (default
    $5,000, tunable, independent from the $10,000 trip cap) -- not just
    back under the cap -- so there's real headroom before the next
    quoter cross can re-trip it. Bringing it back to just under cap would
    thrash pause/unpause on the very next fill.
  - Reacts per-fill AND on a periodic backstop tick (every
    HEDGE_BACKSTOP_MS, default 500ms) per instrument, since dollar
    exposure can also worsen from price drift alone with lot count
    unchanged, and a per-fill-only reaction could miss/undercount if a
    fill event is ever dropped or arrives out of the expected shape.
"""
import asyncio
import os
import random
import time

import nats

NATS_URL = os.environ.get("NATS_URL", "nats://127.0.0.1:4222")
SENDER = os.environ.get("HEDGER_SENDER", "PYHGR001")
QUOTER_SENDER = os.environ.get("SENDER", "QUOTE01")       # matches quoter's $SENDER
TAKER_SENDER = os.environ.get("TAKER_SENDER", "PYTKR001")

DRIVER_FEED = os.environ.get("HEDGE_DRIVER_FEED", "AAH6")
PIGGYBACK_FEEDS = os.environ.get("HEDGE_PIGGYBACK_FEEDS", "AAM6,AAU6").split(",")
ALL_FEEDS = [DRIVER_FEED] + [f for f in PIGGYBACK_FEEDS if f]

# --- Risk policy knobs (tune freely) ---------------------------------------
# Dollar-denominated, independent per instrument. |position * known_price|
# is what's compared against these -- NOT raw lot count.
HEDGE_CAP_USD = float(os.environ.get("HEDGE_CAP_USD", "10000"))       # trip point
HEDGE_TARGET_USD = float(os.environ.get("HEDGE_TARGET_USD", "0"))  # flatten-down-to point

# Escalation ramp: on repeated attempts against the same excess (first F
# didn't fully clear it), reach a few extra ticks PAST the known-correct
# price each retry, up to a cap -- guarantees eventual clearance without
# treating "pay up a lot" as the default first move.
ESCALATION_TICKS_PER_RETRY = float(os.environ.get("HEDGE_ESCALATION_TICKS_PER_RETRY", "2"))
MAX_ESCALATION_TICKS = float(os.environ.get("HEDGE_MAX_ESCALATION_TICKS", "10"))

BACKSTOP_INTERVAL_S = float(os.environ.get("HEDGE_BACKSTOP_MS", "500")) / 1000.0

DESK_STATE_BUCKET = os.environ.get("DESK_STATE_BUCKET", "DESK_STATE")
HEDGE_STATE_BUCKET = os.environ.get("HEDGE_STATE_BUCKET", "HEDGE_STATE")


def opposite(side):
    return "S" if side == "B" else "B"


def parse_kv(value: str) -> dict:
    """Space-separated key=value pairs, same shape as EX_META (PROTOCOL.md)
    and used for DESK_STATE/HEDGE_STATE too."""
    out = {}
    for kv in value.split():
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        out[k] = v
    return out


class InstrumentMeta:
    def __init__(self):
        self.ticksize = 1
        self.ref_price = 0
        self.band = None
        self.pos_lim = None  # exchange's own position_limit, sanity backstop only


async def fetch_meta(js, feed: str) -> InstrumentMeta:
    m = InstrumentMeta()
    try:
        kv = await js.key_value("EX_META")
        entry = await kv.get(feed)
        d = parse_kv(entry.value.decode())
        m.ticksize = int(d.get("ticksize", 1))
        m.ref_price = int(d.get("ref_price", 0))
        m.band = int(d["band"]) if "band" in d else None
        m.pos_lim = int(d["position_limit"]) if "position_limit" in d else None
        print(f"[hedger] EX_META {feed}: ticksize={m.ticksize} ref_price={m.ref_price} "
              f"band={m.band} position_limit={m.pos_lim}", flush=True)
    except Exception as e:
        print(f"[hedger][WARN] failed to read EX_META for {feed}: {e!r}; "
              f"falling back to ticksize=1, band/position_limit unclamped", flush=True)
    return m


class InstrumentHedger:
    """Independent per-instrument position tracker + flattener. is_driver
    controls which DESK_STATE field is treated as the known-correct price:
    driver uses 'fv' directly, piggybacks use 'cEst' (the quoter's
    round(fv(driver)+rel) -- the LP's implied fresh center, same value the
    quoter's own PGYB_CROSS trusts as truth)."""

    def __init__(self, nc, js, feed: str, is_driver: bool):
        self.nc = nc
        self.js = js
        self.feed = feed
        self.is_driver = is_driver
        self.meta = InstrumentMeta()

        self.oid = random.randint(0, 80_000_000)

        # Desk position (quoter + taker), reconstructed from fills.
        self.desk_position = 0

        # Hedger's own fills are counted into desk_position too (unlike the
        # old file, which excluded them) -- the risk being managed is total
        # desk exposure including whatever the hedger itself has already
        # done; excluding hedger fills risks under-counting and re-firing
        # against a position that's already been partly corrected.
        self.own_fills = 0

        # Latest known-correct price for this feed, from DESK_STATE
        # (published by the quoter on every recompute).
        self.known_price = None  # float or None if not yet published

        # Retry/escalation state: tracks how many consecutive hedge attempts
        # have fired without the position clearing back under target, so
        # ticks-past-known-price can ratchet up rather than resetting to
        # "start at known price" every single retry while still pinned.
        self.consecutive_attempts = 0

        self.send_lock = asyncio.Lock()
        self.hedging_in_flight = False

        self.paused = False  # mirrors what we've published to HEDGE_STATE

    def next_oid(self):
        self.oid += 1
        return f"{self.oid:08d}"

    def notional(self) -> float:
        if self.known_price is None:
            return 0.0
        return abs(self.desk_position) * abs(self.known_price)

    # --- position reconstruction --------------------------------------

    def on_desk_fill(self, side: str, vol: int):
        signed = vol if side == "B" else -vol
        self.desk_position += signed

    def on_own_fill(self, side: str, vol: int):
        signed = vol if side == "B" else -vol
        self.desk_position += signed
        self.own_fills += 1

    # --- DESK_STATE (read: quoter's published cEst/fv) ------------------

    def apply_desk_state(self, fields: dict):
        key = "fv" if self.is_driver else "cEst"
        if key in fields:
            try:
                self.known_price = float(fields[key])
            except ValueError:
                pass

    # --- HEDGE_STATE (write: our own pause flag) -------------------------

    async def publish_pause(self, paused: bool):
        if paused == self.paused:
            return  # no-op, avoid redundant KV writes every tick
        self.paused = paused
        try:
            kv = await self.js.key_value(HEDGE_STATE_BUCKET)
            body = f"paused={1 if paused else 0} ts={time.time_ns()}"
            await kv.put(self.feed, body.encode())
            print(f"[hedger] {self.feed} HEDGE_STATE paused={paused}", flush=True)
        except Exception as e:
            print(f"[hedger][WARN] {self.feed} failed to publish HEDGE_STATE: {e!r}", flush=True)

    # --- hedging policy ---------------------------------------------------

    async def evaluate(self):
        """Called on every fill and every backstop tick. Decides whether to
        trip/clear the pause flag and whether to fire a flatten order."""
        if self.known_price is None:
            return  # nothing to price against yet, quoter hasn't published

        notional = self.notional()

        if not self.paused and notional > HEDGE_CAP_USD:
            self.consecutive_attempts = 0
            await self.publish_pause(True)

        if self.paused:
            target_notional = HEDGE_TARGET_USD
            if notional <= target_notional:
                # Back under target -- clear to flat behavior, unpause.
                self.consecutive_attempts = 0
                await self.publish_pause(False)
                return
            await self.fire_flatten(target_notional)

    async def fire_flatten(self, target_notional: float):
        if self.hedging_in_flight:
            return  # one F in flight at a time per instrument; let it land first

        pos = self.desk_position
        if pos == 0 or self.known_price in (None, 0):
            return

        # How many lots to shed to bring notional down to target_notional,
        # preserving current sign (we're reducing magnitude, not flipping).
        target_lots = target_notional / abs(self.known_price)
        target_pos = target_lots if pos > 0 else -target_lots
        excess = pos - target_pos  # positive -> too long, sell |excess|
        side = "S" if excess > 0 else "B"
        size = int(round(abs(excess)))
        if size <= 0:
            return
        if self.meta.pos_lim is not None:
            size = min(size, self.meta.pos_lim)

        tick = self.meta.ticksize if self.meta.ticksize > 0 else 1
        escalation = min(self.consecutive_attempts * ESCALATION_TICKS_PER_RETRY,
                          MAX_ESCALATION_TICKS)
        # Reach PAST the known-correct price by the escalation amount, in
        # the direction that guarantees the fill (through the book, not
        # away from it) -- selling reaches down, buying reaches up.
        price = self.known_price - escalation * tick if side == "S" \
            else self.known_price + escalation * tick
        price = int(round(price))

        if self.meta.band is not None:
            lo, hi = self.meta.ref_price - self.meta.band, self.meta.ref_price + self.meta.band
            clamped = max(lo, min(hi, price))
            if clamped != price:
                print(f"[hedger] {self.feed} price {price} clamped to band [{lo},{hi}] -> {clamped}",
                      flush=True)
            price = clamped

        self.consecutive_attempts += 1
        await self.send_hedge_order(side, size, price, pos, escalation)

    async def send_hedge_order(self, side, size, price, pos_before, escalation_ticks):
        async with self.send_lock:
            self.hedging_in_flight = True
            oid = self.next_oid()
            body = f"{SENDER} A {self.feed} {oid} {side} {size} {price} F"
            print(f"[hedger] {self.feed} pos={pos_before} known_price={self.known_price:.1f} "
                  f"-> F {side} {size} @ {price} (escalation_ticks={escalation_ticks:.1f}, "
                  f"attempt={self.consecutive_attempts})", flush=True)
            try:
                await self.nc.publish(f"ex.req.{SENDER}", body.encode())
                await self.nc.flush()
            finally:
                self.hedging_in_flight = False


class Hedger:
    """Owns all three per-instrument trackers and the shared subscriptions/
    dispatch. Each feed's InstrumentHedger is fully independent -- no
    cross-instrument netting."""

    def __init__(self, nc, js):
        self.nc = nc
        self.js = js
        self.instruments = {}  # feed -> InstrumentHedger

    async def setup(self):
        for feed in ALL_FEEDS:
            is_driver = (feed == DRIVER_FEED)
            ih = InstrumentHedger(self.nc, self.js, feed, is_driver)
            ih.meta = await fetch_meta(self.js, feed)
            self.instruments[feed] = ih

    async def on_md(self, msg):
        f = msg.data.decode().split()
        if len(f) < 8 or f[1] != "E":
            return
        subj_parts = msg.subject.split(".")
        if len(subj_parts) < 3:
            return
        feed = subj_parts[2]
        ih = self.instruments.get(feed)
        if ih is None:
            return

        incoming, resting = f[2], f[3]
        vol = int(f[4])
        in_sender = incoming.split(":", 1)[0]
        rs_sender = resting.split(":", 1)[0]
        aggressor_side = f[7] if len(f) > 7 and f[7] else None
        if aggressor_side is None:
            return

        for sender, tag in ((in_sender, "in"), (rs_sender, "rs")):
            side = aggressor_side if tag == "in" else opposite(aggressor_side)
            if sender in (QUOTER_SENDER, TAKER_SENDER):
                ih.on_desk_fill(side, vol)
            elif sender == SENDER:
                ih.on_own_fill(side, vol)

        await ih.evaluate()

    async def on_desk_state(self, entry):
        """KV watch callback for DESK_STATE. entry.key is the feed,
        entry.value is the quoter's published 'cEst=... fv=... rel=... ts=...'
        blob for that instrument."""
        feed = entry.key
        ih = self.instruments.get(feed)
        if ih is None or entry.value is None:
            return
        fields = parse_kv(entry.value.decode())
        ih.apply_desk_state(fields)
        await ih.evaluate()

    async def backstop_loop(self):
        """Periodic re-check per instrument, independent of fill/KV events
        -- catches dollar-exposure drift from price movement alone (lot
        count unchanged, notional worsening) and guards against any missed
        or malformed fill/KV message."""
        while True:
            await asyncio.sleep(BACKSTOP_INTERVAL_S)
            for ih in self.instruments.values():
                await ih.evaluate()


DESK_STATE_RETRY_INITIAL_S = float(os.environ.get("HEDGE_DESK_STATE_RETRY_INITIAL_S", "0.5"))
DESK_STATE_RETRY_MAX_S = float(os.environ.get("HEDGE_DESK_STATE_RETRY_MAX_S", "20.0"))


async def watch_desk_state(js, hedger: Hedger):
    """Long-running KV watcher for DESK_STATE. Runs as its own task since
    nats.py's KV watch is itself an async iterator, not a plain callback
    subscription like ex.md.*.

    Retries forever with exponential backoff, both on the initial bucket
    lookup and if an established watch ever drops. Without this, a one-time
    startup race (hedger comes up before the quoter has created/attached to
    DESK_STATE) permanently blinds the hedger to pricing for the rest of the
    run -- known_price never gets set, so evaluate() early-returns and the
    hedger silently never flattens. DESK_STATE existing eventually (even
    seconds later, once the quoter's own coordStatus check succeeds) should
    still let the hedger pick it up, not require a restart."""
    backoff = DESK_STATE_RETRY_INITIAL_S
    while True:
        try:
            kv = await js.key_value(DESK_STATE_BUCKET)
            watcher = await kv.watch(">", include_history=True)
            print(f"[hedger] DESK_STATE watch established", flush=True)
            backoff = DESK_STATE_RETRY_INITIAL_S  # reset once we're actually watching
            async for entry in watcher:
                if entry is None:
                    continue
                await hedger.on_desk_state(entry)
            # Watcher's async iterator ended on its own (e.g. connection
            # drop) -- fall through to retry rather than exiting the task.
            print(f"[hedger][WARN] DESK_STATE watch ended, retrying", flush=True)
        except Exception as e:
            print(f"[hedger][WARN] DESK_STATE watch failed: {e!r}; retrying in {backoff:.1f}s",
                  flush=True)
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, DESK_STATE_RETRY_MAX_S)


async def ensure_hedge_state_bucket(js):
    """HEDGE_STATE is owned/written by us (unlike EX_META/DESK_STATE, which
    are created elsewhere) -- js.key_value() only looks up an existing
    bucket, it doesn't create one, so without this every publish_pause call
    fails with BucketNotFoundError until something creates it. Called once
    at startup, before any fills/backstop ticks can fire a publish_pause."""
    try:
        await js.key_value(HEDGE_STATE_BUCKET)
        print(f"[hedger] {HEDGE_STATE_BUCKET} bucket already exists", flush=True)
    except Exception:
        try:
            await js.create_key_value(bucket=HEDGE_STATE_BUCKET)
            print(f"[hedger] created {HEDGE_STATE_BUCKET} bucket", flush=True)
        except Exception as e:
            # Another hedger instance may have created it between our lookup
            # and this call -- treat "already exists" as success, only warn
            # on a genuine failure.
            print(f"[hedger][WARN] create_key_value({HEDGE_STATE_BUCKET}) failed: {e!r} "
                  f"(may already exist from a concurrent startup)", flush=True)


async def main():
    nc = await nats.connect(NATS_URL)
    js = nc.jetstream()
    print(f"[hedger] connected to {NATS_URL} as {SENDER}", flush=True)

    await ensure_hedge_state_bucket(js)

    hedger = Hedger(nc, js)
    await hedger.setup()

    for feed in ALL_FEEDS:
        await nc.subscribe(f"ex.md.{feed}.*", cb=hedger.on_md)

    print(f"[hedger] {SENDER} watching {ALL_FEEDS}, hedging desk position from "
          f"quoter={QUOTER_SENDER} taker={TAKER_SENDER} "
          f"(cap=${HEDGE_CAP_USD:.0f}, target=${HEDGE_TARGET_USD:.0f}, "
          f"backstop_every={BACKSTOP_INTERVAL_S}s)", flush=True)

    asyncio.create_task(watch_desk_state(js, hedger))
    asyncio.create_task(hedger.backstop_loop())

    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
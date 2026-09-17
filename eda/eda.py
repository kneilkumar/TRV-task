#!/usr/bin/env python3
"""
eda_market.py — exploratory data analysis on the exchange sim capture.

Inputs
------
  <data-dir>/bbo_data.csv (OPTIONAL) : interleaved, tag-prefixed rows, from
      a separate full-traffic exchange capture -- the quoter/hedger seats
      themselves never write this file, so it will simply be absent in a
      quoter-only run. Row shapes:
      BBO,ex.bbo.<FEED>,<ts>,<FEED>,<bid_px>,<bid_vol>,<ask_px>,<ask_vol>
      MD,ex.md.<FEED>.<SENDER>,<ts>,A,<SENDER>:<orderid>,<B|S>,<volume>,<price>
      MD,ex.md.<FEED>.<SENDER>,<ts>,C,<SENDER>:<orderid>
      MD,ex.md.<FEED>.<SENDER>,<ts>,{E|T},<incoming>,<resting>,<volume>,<price>,<matchid>,<aggressorSide>
      FILL,<ts>,<feed>,<side>,<price>,<volume>,<positionBefore>,<positionAfter>,<pnlDelta>,<realizedPnL>,<pnlSinceRecalib>
      PNL,<ts>,<feed>,<position>,<avgEntryPrice>,<realizedPnL>,<pnlSinceRecalib>,<rollingMean>,<rollingStd>,<zscore>,<ref_price>,<restingBidPx>,<restingAskPx>
        (older captures may omit ref_price -- this script accepts both the
         12-field and 13-field PNL shape and treats a missing ref_price as NaN)
      When this file is missing, every plot that depends solely on it
      (mid-vs-ref, other-party order postings, executed-trade plots,
      half-life, returns ACF, cross-feed cointegration) is skipped; the
      own-PnL, counterparty, and discovery-based analyses below still run
      off pnl.csv / discovery.csv alone.

  <pnl-csv> (optional, default <data-dir>/pnl.csv) : header'd own-strategy
      log written by the quoter (writePnlRow): ts,event_type,feed,side,
      fill_price,fill_volume,counterparty,counterparty_role,position,
      avg_entry_price,mid,ref_price,realized_pnl,unrealized_pnl,total_pnl,
      pnl_since_recalib
        (older captures may lack the counterparty,counterparty_role columns
         right after fill_volume -- see the own-PnL/counterparty section
         below; counterparty-specific analysis is skipped gracefully then.)

  <discovery-csv> (optional, default <data-dir>/discovery.csv) : header'd
      log written by the quoter's discovery pass: ts,event,feed_or_pair,
      detail, where `detail` is a ";"-separated key=value string whose keys
      depend on `event`:
        PAIR_RELATED       feed_or_pair=<A>-<B>   detail: r2,lag,aLeadsB
        DRIVER_IDENTIFIED  feed_or_pair=<feed>    detail: mover,frac
        PIGGYBACK_IDENTIFIED feed_or_pair=<feed>  detail: driver
        REL_TRUSTED        feed_or_pair=<feed>    detail: driver,rel,sdTicks
        AMBIGUOUS_CLUSTER  feed_or_pair=<f1;f2;..> (no detail)
      This is the primary window onto *other* participants' behavior (the
      driver-moving "mover" bot and the fixed-offset background LP), since
      the quoter never trades the driver itself.

What this produces (per discovered feed, in --out-dir)
-------------------------------------------------------
  <feed>_mid_vs_ref.png            [needs bbo_data.csv] mid vs. ref price
  <feed>_order_price_hist.png      [needs bbo_data.csv] histogram of other
                                    parties' order prices (rel. to mid), by side
  <feed>_order_volume_hist.png     [needs bbo_data.csv] other parties' order
                                    volumes
  <feed>_order_frequency.png       [needs bbo_data.csv] order-posting frequency
  <feed>_order_arrival_acf.png     [needs bbo_data.csv] order inter-arrival ACF
  <feed>_trade_volume_hist.png     [needs bbo_data.csv] executed trade volumes
  <feed>_trade_price_vs_mid.png    [needs bbo_data.csv] trade prices vs. mid,
                                    colored by aggressor side
  <feed>_trade_frequency.png       [needs bbo_data.csv] trade frequency
  <feed>_mover_sweep.png           [needs bbo_data.csv] mid price with mover
                                    trades marked and sweep windows shaded
  <feed>_returns_acf.png           [needs bbo_data.csv] mid-price return ACF
  <feed>_own_pnl_over_time.png     realized/unrealized/total PnL over time,
                                    from OUR OWN pnl.csv log
  <feed>_counterparty_over_time.png  cumulative volume filled BY each
                                    counterparty who has fulfilled our
                                    resting orders (PASSIVE role), over time
  <feed>_counterparty_breakdown.png  total volume traded with each
                                    counterparty, split by whether we were
                                    the aggressor or they were (PASSIVE)

Cross-feed / other-trader outputs
-----------------------------------
  cointegration_spreads.png        [needs bbo_data.csv] spread(s) for any
                                    cointegrated pair/triplet found
  mover_dominance.png              from discovery.csv: each identified
                                    driver feed's dominant sender ("mover")
                                    and its trade-count dominance fraction
  rel_discovery_timeline.png       from discovery.csv: recovered driver/
                                    piggyback offset (rel) at the moment it
                                    became trusted, one line per piggyback
  summary.txt                      half-life, ACF findings, cointegration
                                    test results, sweep event counts, own
                                    PnL, counterparty fulfillment, and
                                    discovery.csv-derived mover/piggyback/
                                    rel-recovery stats

Uniform-grid resampling note
-----------------------------
Two different resampling concepts are used and they are NOT the same:
  1. Plot-only downsampling (--plot-resample-threshold / --resample-rule):
     applied purely so large scatter/line plots stay legible and fast to
     render. Only kicks in above the row-count threshold.
  2. Statistical resampling (half-life, cointegration, returns/ACF): ALWAYS
     resampled onto a uniform time grid at --resample-rule regardless of row
     count, because these tests require evenly spaced observations -- an
     irregularly-sampled series (BBO ticks arrive whenever the book moves)
     would silently bias autocorrelation/half-life/cointegration estimates.
"""

import argparse
import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is required: pip install matplotlib --break-system-packages", file=sys.stderr)
    raise

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import coint, acf
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    from statsmodels.graphics.tsaplots import plot_acf
except ImportError:
    print("statsmodels is required: pip install statsmodels --break-system-packages", file=sys.stderr)
    raise

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# Parsing bbo_data.csv
# ============================================================================

def _num_or_nan(s):
    return np.nan if s == "-" else float(s)


def _sender_of(party_tag):
    """'SENDER:ORDERID' -> 'SENDER'."""
    colon = party_tag.find(":")
    return party_tag if colon == -1 else party_tag[:colon]


def _feed_from_subject(subject):
    # ex.md.<FEED>.<SENDER> or ex.bbo.<FEED>
    parts = subject.split(".")
    return parts[2] if len(parts) >= 3 else None


def parse_bbo_data(path):
    """Single pass over bbo_data.csv, bucketing rows by tag/type.

    Returns a dict of DataFrames:
      bbo, order_add, order_cancel, trade, fill (own), pnl_snap (own)
    """
    bbo_rows, add_rows, cancel_rows, trade_rows, fill_rows, pnl_rows = [], [], [], [], [], []
    skipped = {"BBO": 0, "MD_A": 0, "MD_C": 0, "MD_ET": 0, "MD_other": 0, "FILL": 0, "PNL": 0}

    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(",")
            tag = parts[0]

            if tag == "BBO":
                if len(parts) != 8:
                    skipped["BBO"] += 1
                    continue
                try:
                    bbo_rows.append({
                        "ts": int(parts[2]), "feed": parts[3],
                        "bid_px": _num_or_nan(parts[4]), "bid_vol": _num_or_nan(parts[5]),
                        "ask_px": _num_or_nan(parts[6]), "ask_vol": _num_or_nan(parts[7]),
                    })
                except ValueError:
                    skipped["BBO"] += 1

            elif tag == "MD":
                if len(parts) < 4:
                    skipped["MD_other"] += 1
                    continue
                subject, ts_s, mtype = parts[1], parts[2], parts[3]
                feed = _feed_from_subject(subject)
                try:
                    ts = int(ts_s)
                except ValueError:
                    skipped["MD_other"] += 1
                    continue

                if mtype == "A" and len(parts) == 8:
                    try:
                        add_rows.append({
                            "ts": ts, "feed": feed, "sender": _sender_of(parts[4]),
                            "order_id": parts[4], "side": parts[5],
                            "volume": float(parts[6]), "price": float(parts[7]),
                        })
                    except ValueError:
                        skipped["MD_A"] += 1

                elif mtype == "C" and len(parts) == 5:
                    cancel_rows.append({
                        "ts": ts, "feed": feed, "sender": _sender_of(parts[4]),
                        "order_id": parts[4],
                    })

                elif mtype in ("E", "T") and len(parts) == 10:
                    try:
                        trade_rows.append({
                            "ts": ts, "feed": feed, "type": mtype,
                            "incoming": parts[4], "resting": parts[5],
                            "incoming_sender": _sender_of(parts[4]),
                            "resting_sender": _sender_of(parts[5]),
                            "volume": float(parts[6]), "price": float(parts[7]),
                            "matchid": parts[8], "aggressor_side": parts[9],
                        })
                    except ValueError:
                        skipped["MD_ET"] += 1
                else:
                    skipped["MD_other"] += 1

            elif tag == "FILL":
                if len(parts) != 11:
                    skipped["FILL"] += 1
                    continue
                try:
                    fill_rows.append({
                        "ts": int(parts[1]), "feed": parts[2], "side": parts[3],
                        "price": float(parts[4]), "volume": float(parts[5]),
                        "position_before": float(parts[6]), "position_after": float(parts[7]),
                        "pnl_delta": float(parts[8]), "realized_pnl": float(parts[9]),
                        "pnl_since_recalib": float(parts[10]),
                    })
                except ValueError:
                    skipped["FILL"] += 1

            elif tag == "PNL":
                # Accept both the 12-field (no ref_price) and 13-field
                # (with ref_price) shapes for backward compatibility.
                if len(parts) == 13:
                    ref_price = _num_or_nan(parts[10])
                    resting_bid, resting_ask = parts[11], parts[12]
                elif len(parts) == 12:
                    ref_price = np.nan
                    resting_bid, resting_ask = parts[10], parts[11]
                else:
                    skipped["PNL"] += 1
                    continue
                try:
                    pnl_rows.append({
                        "ts": int(parts[1]), "feed": parts[2], "position": float(parts[3]),
                        "avg_entry_price": float(parts[4]), "realized_pnl": float(parts[5]),
                        "pnl_since_recalib": float(parts[6]), "rolling_mean": float(parts[7]),
                        "rolling_std": float(parts[8]), "zscore": float(parts[9]),
                        "ref_price": ref_price,
                        "resting_bid_px": _num_or_nan(resting_bid),
                        "resting_ask_px": _num_or_nan(resting_ask),
                    })
                except ValueError:
                    skipped["PNL"] += 1
            # any other tag: ignore

    out = {}
    schemas = {
        "bbo": ["ts", "feed", "bid_px", "bid_vol", "ask_px", "ask_vol"],
        "order_add": ["ts", "feed", "sender", "order_id", "side", "volume", "price"],
        "order_cancel": ["ts", "feed", "sender", "order_id"],
        "trade": ["ts", "feed", "type", "incoming", "resting", "incoming_sender",
                  "resting_sender", "volume", "price", "matchid", "aggressor_side"],
        "fill": ["ts", "feed", "side", "price", "volume", "position_before",
                 "position_after", "pnl_delta", "realized_pnl", "pnl_since_recalib"],
        "pnl_snap": ["ts", "feed", "position", "avg_entry_price", "realized_pnl",
                     "pnl_since_recalib", "rolling_mean", "rolling_std", "zscore",
                     "ref_price", "resting_bid_px", "resting_ask_px"],
    }
    for name, rows in [("bbo", bbo_rows), ("order_add", add_rows), ("order_cancel", cancel_rows),
                        ("trade", trade_rows), ("fill", fill_rows), ("pnl_snap", pnl_rows)]:
        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=schemas[name])
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"], unit="ns")
            df.sort_values("ts", inplace=True)
            df.reset_index(drop=True, inplace=True)
        out[name] = df

    if not out["bbo"].empty:
        out["bbo"]["mid"] = (out["bbo"]["bid_px"] + out["bbo"]["ask_px"]) / 2.0

    if not out["trade"].empty:
        # E and T are two addressings of the same underlying trade (one per
        # party's subject) -- dedupe by (feed, matchid) so volume/price
        # stats aren't double-counted.
        out["trade"] = out["trade"].drop_duplicates(subset=["feed", "matchid"]).reset_index(drop=True)

    total_skipped = sum(skipped.values())
    if total_skipped:
        print(f"  [parse warning] skipped malformed rows: {skipped}")

    return out


def load_pnl_csv(path):
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], unit="ns")
    df.sort_values("ts", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _parse_detail(detail):
    """'driver=BOND1;rel=12.5;sdTicks=0.8' -> {'driver': 'BOND1', 'rel':
    12.5, 'sdTicks': 0.8}. Numeric-looking values are coerced to float;
    everything else stays a string. Missing/empty detail -> {}.
    """
    out = {}
    if not isinstance(detail, str) or not detail:
        return out
    for kv in detail.split(";"):
        if not kv or "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


def load_discovery_csv(path):
    """Loads the quoter's discovery.csv (ts,event,feed_or_pair,detail) and
    expands `detail` into per-event-type columns. This is the file's only
    record of *other* participants' behavior -- the mover bot's identity/
    dominance and the background LP's recovered fixed offset -- since the
    quoter itself never trades the driver leg.
    """
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    # ts,event,feed_or_pair,detail -- detail itself may legitimately contain
    # commas? No (it's ';'-joined key=value), but a header-less file with a
    # variable field count elsewhere would break pd.read_csv, so read raw.
    rows = []
    with open(path, "r") as f:
        header = f.readline()  # "ts,event,feed_or_pair,detail"
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(",", 3)
            if len(parts) < 3:
                continue
            ts, event, feed_or_pair = parts[0], parts[1], parts[2]
            detail = parts[3] if len(parts) == 4 else ""
            try:
                ts = int(ts)
            except ValueError:
                continue
            row = {"ts": ts, "event": event, "feed_or_pair": feed_or_pair, "detail": detail}
            row.update(_parse_detail(detail))
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], unit="ns")
    df.sort_values("ts", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ============================================================================
# Resampling helpers
# ============================================================================

def uniform_mid_series(bbo_feed_df, rule):
    """Always-uniform-grid mid series for statistical use (half-life,
    cointegration, returns/ACF). Forward-filled; leading NaNs (before the
    first BBO update) are dropped.
    """
    if bbo_feed_df.empty:
        return pd.Series(dtype=float)
    s = bbo_feed_df.set_index("ts")["mid"].resample(rule).last().ffill()
    return s.dropna()


def downsample_for_plot(df, ts_col, threshold, rule):
    """Only downsamples if the row count exceeds threshold -- for plot
    legibility/speed, not for statistical validity. Returns df unchanged if
    small enough.
    """
    if len(df) <= threshold:
        return df
    return df.set_index(ts_col).resample(rule).last().dropna(how="all").reset_index()


# ============================================================================
# 1. Mid vs reference price
# ============================================================================

def plot_mid_vs_ref(feed, bbo_feed_df, ref_series, out_path, plot_threshold, plot_rule):
    if bbo_feed_df.empty:
        print(f"  [{feed}] no BBO data -- skipping mid-vs-ref plot")
        return

    plot_df = downsample_for_plot(bbo_feed_df[["ts", "mid"]], "ts", plot_threshold, plot_rule)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(plot_df["ts"], plot_df["mid"], color="tab:blue", linewidth=0.9, label="mid price")
    if ref_series is not None and not ref_series.empty:
        ax.plot(ref_series.index, ref_series.values, color="tab:red", linewidth=1.2,
                linestyle="--", label="ref_price")
    ax.set_title(f"{feed}: mid price vs. reference price")
    ax.set_xlabel("time")
    ax.set_ylabel("price")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [{feed}] wrote {out_path}")


# ============================================================================
# 2. Other parties' order postings
# ============================================================================

def analyze_order_postings(feed, order_add_df, mid_uniform, out_dir, own_senders,
                            plot_threshold, plot_rule, acf_lags):
    orders = order_add_df[order_add_df["feed"] == feed].copy()
    if own_senders:
        orders = orders[~orders["sender"].isin(own_senders)]
    if orders.empty:
        print(f"  [{feed}] no (other-party) order-add rows -- skipping order posting analysis")
        return {}

    # Price relative to contemporaneous mid -- normalizes for the fact that
    # mid itself drifts, so a raw price histogram would just look like a
    # smeared-out copy of the mid-price distribution.
    if not mid_uniform.empty:
        mid_at = pd.merge_asof(orders.sort_values("ts"), mid_uniform.rename("mid").reset_index(),
                                on="ts", direction="backward")
        orders["price_vs_mid"] = orders["price"].values - mid_at["mid"].values
    else:
        orders["price_vs_mid"] = np.nan

    # --- histogram: order price relative to mid, by side ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for side, color in [("B", "tab:green"), ("S", "tab:red")]:
        vals = orders.loc[orders["side"] == side, "price_vs_mid"].dropna()
        if len(vals):
            ax.hist(vals, bins=50, alpha=0.6, label=f"side={side}", color=color)
    ax.set_title(f"{feed}: other-party order price (relative to mid)")
    ax.set_xlabel("price - contemporaneous mid")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    p1 = os.path.join(out_dir, f"{feed}_order_price_hist.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # --- histogram: order volume ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(orders["volume"].dropna(), bins=50, color="tab:purple", alpha=0.8)
    ax.set_title(f"{feed}: other-party order volume distribution")
    ax.set_xlabel("volume")
    ax.set_ylabel("count")
    fig.tight_layout()
    p2 = os.path.join(out_dir, f"{feed}_order_volume_hist.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    # --- frequency plot: order postings per time bucket ---
    counts = orders.set_index("ts").resample(plot_rule).size()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(counts.index, counts.values, color="tab:orange", linewidth=0.9)
    ax.set_title(f"{feed}: order-posting frequency ({plot_rule} buckets)")
    ax.set_xlabel("time")
    ax.set_ylabel("orders posted")
    fig.tight_layout()
    p3 = os.path.join(out_dir, f"{feed}_order_frequency.png")
    fig.savefig(p3, dpi=150)
    plt.close(fig)

    # --- autocorrelation of inter-arrival times ---
    inter_arrival = orders["ts"].diff().dt.total_seconds().dropna()
    stats = {}
    if len(inter_arrival) > acf_lags + 1:
        acf_vals = acf(inter_arrival, nlags=acf_lags, fft=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_acf(inter_arrival, lags=acf_lags, ax=ax, title=f"{feed}: order inter-arrival-time ACF")
        fig.tight_layout()
        p4 = os.path.join(out_dir, f"{feed}_order_arrival_acf.png")
        fig.savefig(p4, dpi=150)
        plt.close(fig)
        # crude significance band: +/- 1.96/sqrt(N), same convention plot_acf uses
        band = 1.96 / np.sqrt(len(inter_arrival))
        sig_lags = [i for i, v in enumerate(acf_vals[1:], start=1) if abs(v) > band]
        stats["order_arrival_acf_significant_lags"] = sig_lags
        stats["order_arrival_mean_gap_sec"] = float(inter_arrival.mean())
    else:
        print(f"  [{feed}] too few order-add rows for ACF ({len(inter_arrival)})")

    print(f"  [{feed}] wrote order posting plots ({len(orders)} orders analyzed)")
    return stats


# ============================================================================
# 3. Other parties' executed trades + mover/sweep detection
# ============================================================================

def analyze_trades(feed, trade_df, mid_uniform, out_dir, own_senders,
                    plot_threshold, plot_rule, mover_pattern,
                    sweep_window, sweep_volume_multiple):
    trades = trade_df[trade_df["feed"] == feed].copy()
    if own_senders:
        trades = trades[~trades["incoming_sender"].isin(own_senders) &
                         ~trades["resting_sender"].isin(own_senders)]
    if trades.empty:
        print(f"  [{feed}] no (other-party) trade rows -- skipping trade analysis")
        return {}

    # --- histogram: trade volume ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(trades["volume"].dropna(), bins=50, color="tab:blue", alpha=0.8)
    ax.set_title(f"{feed}: executed trade volume distribution")
    ax.set_xlabel("volume")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{feed}_trade_volume_hist.png"), dpi=150)
    plt.close(fig)

    # --- trade price over time vs mid, colored by aggressor side ---
    plot_trades = downsample_for_plot(trades[["ts", "price", "aggressor_side"]], "ts",
                                       plot_threshold, plot_rule)
    fig, ax = plt.subplots(figsize=(14, 6))
    if not mid_uniform.empty:
        ax.plot(mid_uniform.index, mid_uniform.values, color="tab:gray", linewidth=0.7,
                alpha=0.6, label="mid")
    for side, color, label in [("B", "tab:green", "aggressor bought"), ("S", "tab:red", "aggressor sold")]:
        sub = plot_trades[plot_trades["aggressor_side"] == side]
        ax.scatter(sub["ts"], sub["price"], s=8, color=color, alpha=0.6, label=label)
    ax.set_title(f"{feed}: executed trades vs. mid")
    ax.set_xlabel("time")
    ax.set_ylabel("price")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{feed}_trade_price_vs_mid.png"), dpi=150)
    plt.close(fig)

    # --- trade frequency over time ---
    counts = trades.set_index("ts").resample(plot_rule).size()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(counts.index, counts.values, color="tab:brown", linewidth=0.9)
    ax.set_title(f"{feed}: trade frequency ({plot_rule} buckets)")
    ax.set_xlabel("time")
    ax.set_ylabel("trades executed")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{feed}_trade_frequency.png"), dpi=150)
    plt.close(fig)

    # --- mover identification + sweep detection ---
    #
    # A mover-only volume baseline breaks down when the mover trades rarely
    # except during the sweep itself (no "normal" mover activity to compare
    # against) -- so sweeps are detected from ALL trade volume plus a
    # concurrent price displacement check, and mover involvement is then
    # measured within each detected window separately. This also catches
    # sweeps caused by a party that doesn't match --mover-pattern.
    is_mover_incoming = trades["incoming_sender"].str.contains(mover_pattern, case=False, na=False)
    is_mover_resting = trades["resting_sender"].str.contains(mover_pattern, case=False, na=False)
    mover_trades = trades[is_mover_incoming | is_mover_resting].copy()

    sweep_events = pd.DataFrame(columns=["start", "end", "peak_volume", "price_move", "mover_volume_share"])

    all_vol = trades.set_index("ts")["volume"].resample(sweep_window).sum()
    if len(all_vol) >= 8 and not mid_uniform.empty:
        baseline = all_vol.rolling(30, min_periods=5).median()
        baseline = baseline.replace(0, np.nan).fillna(all_vol[all_vol > 0].median() if (all_vol > 0).any() else 1.0)
        volume_burst = all_vol > (baseline * sweep_volume_multiple)

        # Intra-bucket high-low range, not a lag-1 diff of bucket-end
        # values -- a diff would miss a spike that decays back down before
        # the bucket closes, which is exactly the "sweep then decay" shape
        # observed in the earlier AAH6 spread-spike analysis.
        agg = mid_uniform.resample(sweep_window).agg(["max", "min"]).reindex(all_vol.index)
        price_range = (agg["max"] - agg["min"]).fillna(0.0)
        typical_range = price_range.replace(0, np.nan).median()
        move_threshold = 3.0 * (typical_range if pd.notna(typical_range) and typical_range > 0 else price_range.std())

        flagged = (volume_burst & (price_range > move_threshold)).fillna(False)

        if flagged.any():
            mover_vol_by_bucket = mover_trades.set_index("ts")["volume"].resample(sweep_window).sum().reindex(
                all_vol.index, fill_value=0.0)
            grp = (flagged != flagged.shift()).cumsum()
            for _, idxs in flagged[flagged].groupby(grp[flagged]).groups.items():
                block_vol = all_vol.loc[idxs]
                block_mover_vol = mover_vol_by_bucket.loc[idxs]
                block_range = price_range.loc[idxs]
                mover_share = float(block_mover_vol.sum() / block_vol.sum()) if block_vol.sum() > 0 else 0.0
                sweep_events.loc[len(sweep_events)] = [
                    idxs.min(), idxs.max(), float(block_vol.max()),
                    float(block_range.max()), mover_share,
                ]

    if not mover_trades.empty or not sweep_events.empty:
        fig, ax = plt.subplots(figsize=(14, 6))
        if not mid_uniform.empty:
            ax.plot(mid_uniform.index, mid_uniform.values, color="tab:gray", linewidth=0.8, label="mid")
        if not mover_trades.empty:
            ax.scatter(mover_trades["ts"], mover_trades["price"], s=14, color="tab:orange",
                       label=f"'{mover_pattern}' trades", zorder=3)
        for _, ev in sweep_events.iterrows():
            color = "red" if ev["mover_volume_share"] >= 0.5 else "purple"
            ax.axvspan(ev["start"], ev["end"], color=color, alpha=0.2)
        n_mover_dominant = int((sweep_events["mover_volume_share"] >= 0.5).sum()) if not sweep_events.empty else 0
        ax.set_title(f"{feed}: mover activity and detected sweep windows "
                     f"({len(sweep_events)} total, {n_mover_dominant} mover-dominant [red])")
        ax.set_xlabel("time")
        ax.set_ylabel("price")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{feed}_mover_sweep.png"), dpi=150)
        plt.close(fig)
        print(f"  [{feed}] mover trades: {len(mover_trades)}, sweep events: {len(sweep_events)} "
              f"({n_mover_dominant} mover-dominant)")
    else:
        print(f"  [{feed}] no mover trades and no sweep events detected")

    print(f"  [{feed}] wrote trade analysis plots ({len(trades)} trades analyzed)")
    return {
        "n_trades": len(trades),
        "n_mover_trades": len(mover_trades),
        "n_sweep_events": len(sweep_events),
        "sweep_events": sweep_events,
    }


# ============================================================================
# 4. Half-life of mean reversion
# ============================================================================

def compute_half_life(mid_series):
    """Standard OU discretization: regress delta_y on lagged level (with
    intercept). beta<0 => mean-reverting; half-life = -ln(2)/beta.
    beta>=0 => no mean reversion detected (e.g. a random walk), returns None.
    """
    if len(mid_series) < 30:
        return {"half_life_sec": None, "beta": None, "pvalue": None, "note": "insufficient data"}

    y = mid_series.values
    dy = np.diff(y)
    y_lag = y[:-1]
    X = sm.add_constant(y_lag)
    model = sm.OLS(dy, X).fit()
    beta = model.params[1]
    pvalue = model.pvalues[1]

    # convert regression steps -> seconds using the series' own sampling
    # interval (uniform grid, so this is constant)
    dt_sec = (mid_series.index[1] - mid_series.index[0]).total_seconds()

    if beta < 0:
        half_life_steps = -np.log(2) / beta
        half_life_sec = half_life_steps * dt_sec
        note = "mean-reverting"
    else:
        half_life_sec = None
        note = "no mean reversion detected (beta >= 0, consistent with a random walk)"

    return {"half_life_sec": half_life_sec, "beta": float(beta), "pvalue": float(pvalue), "note": note}


# ============================================================================
# 5. Returns + autocorrelation
# ============================================================================

def analyze_returns(feed, mid_series, out_dir, acf_lags):
    if len(mid_series) < acf_lags + 2:
        print(f"  [{feed}] too few points for returns ACF")
        return {}

    returns = mid_series.pct_change().dropna()
    acf_vals = acf(returns, nlags=acf_lags, fft=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_acf(returns, lags=acf_lags, ax=ax, title=f"{feed}: mid-price return ACF")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{feed}_returns_acf.png"), dpi=150)
    plt.close(fig)

    band = 1.96 / np.sqrt(len(returns))
    sig_lags = [i for i, v in enumerate(acf_vals[1:], start=1) if abs(v) > band]
    print(f"  [{feed}] wrote returns ACF ({len(returns)} return obs, "
          f"significant lags: {sig_lags if sig_lags else 'none'})")
    return {
        "n_returns": len(returns),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "significant_acf_lags": sig_lags,
    }


# ============================================================================
# 6. Cointegration across feeds
# ============================================================================

def run_cointegration_tests(mid_series_by_feed, out_dir, significance):
    feeds = sorted(mid_series_by_feed.keys())
    results = {"pairwise": [], "johansen": None}

    if len(feeds) < 2:
        return results

    # Align all series on a common index (inner join) so tests compare like
    # with like -- feeds may have different start/end times.
    aligned = pd.concat(
        {f: mid_series_by_feed[f] for f in feeds}, axis=1, join="inner"
    ).dropna()
    if len(aligned) < 30:
        print("  not enough overlapping data across feeds for cointegration tests")
        return results

    # --- pairwise Engle-Granger ---
    for f1, f2 in itertools.combinations(feeds, 2):
        print("samples required: ", len(aligned[f1]))
        score, pvalue, _ = coint(aligned[f1], aligned[f2])
        # hedge ratio via OLS of f1 on f2 (for spread = f1 - beta*f2)
        beta = sm.OLS(aligned[f1], sm.add_constant(aligned[f2])).fit().params.iloc[1]
        results["pairwise"].append({
            "pair": (f1, f2), "pvalue": float(pvalue), "score": float(score),
            "hedge_ratio": float(beta), "cointegrated": bool(pvalue < significance),
        })

    # --- Johansen (multivariate, covers the triplet case) ---
    try:
        joh = coint_johansen(aligned.values, det_order=0, k_ar_diff=1)
        trace_stats = joh.lr1
        crit_90_95_99 = joh.cvt  # critical values at 90/95/99%
        n_coint_95 = int(np.sum(trace_stats > crit_90_95_99[:, 1]))  # count relations significant at 95%
        results["johansen"] = {
            "feeds": feeds,
            "trace_stats": trace_stats.tolist(),
            "critical_values_95": crit_90_95_99[:, 1].tolist(),
            "n_cointegrating_relations_95pct": n_coint_95,
            "eigenvectors": joh.evec.tolist(),
        }
    except Exception as e:
        print(f"  Johansen test failed: {e}")

    # --- plot any cointegrated pair's spread, plus the top Johansen combo if found ---
    plotted_any = False
    fig, ax = plt.subplots(figsize=(14, 6))
    for r in results["pairwise"]:
        if r["cointegrated"]:
            f1, f2 = r["pair"]
            spread = aligned[f1] - r["hedge_ratio"] * aligned[f2]
            ax.plot(spread.index, spread.values,
                    label=f"{f1} - {r['hedge_ratio']:.3f}*{f2} (p={r['pvalue']:.4f})")
            plotted_any = True

    if results["johansen"] and results["johansen"]["n_cointegrating_relations_95pct"] > 0:
        w = np.array(results["johansen"]["eigenvectors"])[:, 0]  # first cointegrating vector
        combo = aligned.values @ w
        result = adfuller(combo)
        print("--- Augmented Dickey-Fuller Test Results ---")
        print(f"ADF Statistic:       {result[0]:.4f}")
        print(f"p-value:             {result[1]:.4f}")
        print(f"Lags Used:           {result[2]}")
        print(f"Observations Used:   {result[3]}")
        print("Critical Values:")
        for key, value in result[4].items():
            print(f"   {key}: {value:.4f}")
        ax.plot(aligned.index, combo, color="black", linestyle="--",
                label=f"Johansen combo ({'+'.join(feeds)})")
        plotted_any = True

    if plotted_any:
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_title("Cointegrated spread(s)")
        ax.set_xlabel("time")
        ax.set_ylabel("spread")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "cointegration_spreads.png"), dpi=150)
        print(f"  wrote {os.path.join(out_dir, 'cointegration_spreads.png')}")
    plt.close(fig)

    return results


# ============================================================================
# 6b. Other-trader behavior, from discovery.csv
# ============================================================================

def analyze_discovery(discovery_df, out_dir):
    """Everything we know about *other* participants comes from what the
    quoter's own discovery pass figured out, since it never trades the
    driver leg directly:
      - DRIVER_IDENTIFIED: which sender dominates a driver feed's trade
        flow (the "mover" bot) and how dominant it is
      - PIGGYBACK_IDENTIFIED: which feeds track which driver
      - REL_TRUSTED: the fixed offset recovered against the background LP,
        and when it became trustworthy
      - PAIR_RELATED / AMBIGUOUS_CLUSTER: raw co-movement clustering, incl.
        clusters where driver identification is still ambiguous
    """
    if discovery_df.empty:
        print("  no discovery.csv rows -- skipping other-trader behavior analysis")
        return {}

    stats = {}

    drivers = discovery_df[discovery_df["event"] == "DRIVER_IDENTIFIED"].copy()
    piggybacks = discovery_df[discovery_df["event"] == "PIGGYBACK_IDENTIFIED"].copy()
    rel_trusted = discovery_df[discovery_df["event"] == "REL_TRUSTED"].copy()
    ambiguous = discovery_df[discovery_df["event"] == "AMBIGUOUS_CLUSTER"]
    pair_related = discovery_df[discovery_df["event"] == "PAIR_RELATED"]

    # --- mover dominance per driver feed (first identification only -- the
    # mover's identity for a given driver doesn't change once found) ---
    if not drivers.empty and {"mover", "frac"}.issubset(drivers.columns):
        first_seen = drivers.drop_duplicates(subset=["feed_or_pair"], keep="first")
        first_seen = first_seen.sort_values("frac", ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(first_seen))
        ax.bar(x, first_seen["frac"], color="tab:orange", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{r.feed_or_pair}\n(mover={r.mover})" for r in first_seen.itertuples()],
            rotation=0, ha="center", fontsize=8)
        ax.axhline(cfg_dominance_floor_hint(), color="gray", linewidth=0.8, linestyle=":")
        ax.set_ylim(0, 1.0)
        ax.set_title("Identified driver feeds: dominant sender ('mover') trade-count share")
        ax.set_ylabel("fraction of trades from the dominant sender")
        fig.tight_layout()
        p = os.path.join(out_dir, "mover_dominance.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  wrote {p} ({len(first_seen)} driver feed(s) identified)")
        stats["drivers"] = list(zip(first_seen["feed_or_pair"], first_seen["mover"], first_seen["frac"]))

    # --- rel recovery timeline: recovered offset at the moment it was
    # first trusted, per piggyback, plus how it may have been re-confirmed
    # (refreshed) later in the run ---
    if not rel_trusted.empty and {"driver", "rel"}.issubset(rel_trusted.columns):
        fig, ax = plt.subplots(figsize=(12, 6))
        cmap = plt.get_cmap("tab10")
        for i, (piggyback, grp) in enumerate(rel_trusted.groupby("feed_or_pair")):
            grp = grp.sort_values("ts")
            ax.step(grp["ts"], grp["rel"], where="post", marker="o", markersize=4,
                    color=cmap(i % 10), label=f"{piggyback} (driver={grp['driver'].iloc[0]})")
        ax.set_title("Recovered driver/piggyback offset (rel) once trusted, over time")
        ax.set_xlabel("time")
        ax.set_ylabel("rel (driver fair value - piggyback mid)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(out_dir, "rel_discovery_timeline.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  wrote {p} ({rel_trusted['feed_or_pair'].nunique()} piggyback(s) with a trusted rel)")
        stats["rel_by_piggyback"] = {
            pb: {"driver": grp.sort_values("ts")["driver"].iloc[-1],
                 "final_rel": float(grp.sort_values("ts")["rel"].iloc[-1]),
                 "n_confirmations": len(grp)}
            for pb, grp in rel_trusted.groupby("feed_or_pair")
        }

    stats["n_pairs_related"] = len(pair_related)
    stats["n_drivers_identified"] = drivers["feed_or_pair"].nunique() if not drivers.empty else 0
    stats["piggybacks"] = list(zip(piggybacks["feed_or_pair"], piggybacks.get("driver", pd.Series(dtype=str))))
    stats["n_ambiguous_clusters"] = len(ambiguous)

    print(f"  discovery.csv: {stats['n_drivers_identified']} driver(s), "
          f"{len(piggybacks)} piggyback(s), {len(rel_trusted)} rel-trust event(s), "
          f"{len(ambiguous)} ambiguous cluster event(s)")
    return stats


def cfg_dominance_floor_hint():
    """Reference line only -- mirrors the quoter's cfg::kDominanceFloor
    (0.6) so the mover_dominance.png plot shows identified drivers against
    the threshold that qualified them. Not read from the C++ source; update
    here if that tunable changes."""
    return 0.6


# ============================================================================
# 7. Own PnL over time + counterparty fulfillment
# ============================================================================

def analyze_own_pnl(feed, pnl_csv_df, out_dir, plot_threshold, plot_rule, top_n_counterparties):
    """Uses OUR OWN pnl.csv log (not bbo_data.csv) to show:
      1. realized/unrealized/total PnL over time for this feed
      2. who has been fulfilling our resting orders over time (counterparty
         fulfillment, from rows where counterparty_role == PASSIVE, i.e.
         someone else hit our resting order)
      3. an overall counterparty breakdown (volume traded with each party,
         split by whether we were the aggressor or they were)

    Gracefully degrades if pnl_csv_df doesn't have the newer
    counterparty/counterparty_role columns (older capture) -- still plots
    PnL over time, just skips the counterparty-specific plots.
    """
    df = pnl_csv_df[pnl_csv_df["feed"] == feed].copy() if not pnl_csv_df.empty else pd.DataFrame()
    if df.empty:
        print(f"  [{feed}] no pnl.csv rows -- skipping own-PnL analysis")
        return {}

    stats = {}

    # --- 1. PnL over time (every row, FILL or SNAPSHOT, carries a PnL mark) ---
    if {"realized_pnl", "unrealized_pnl", "total_pnl"}.issubset(df.columns):
        plot_df = downsample_for_plot(
            df[["ts", "realized_pnl", "unrealized_pnl", "total_pnl"]], "ts",
            plot_threshold, plot_rule)
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(plot_df["ts"], plot_df["realized_pnl"], color="tab:blue", linewidth=1.0, label="realized PnL")
        ax.plot(plot_df["ts"], plot_df["unrealized_pnl"], color="tab:orange", linewidth=1.0,
                linestyle="--", label="unrealized PnL")
        ax.plot(plot_df["ts"], plot_df["total_pnl"], color="tab:red", linewidth=1.4, label="total PnL")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_title(f"{feed}: own PnL over time (from pnl.csv)")
        ax.set_xlabel("time")
        ax.set_ylabel("PnL")
        ax.legend(fontsize=9)
        fig.tight_layout()
        p = os.path.join(out_dir, f"{feed}_own_pnl_over_time.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  [{feed}] wrote {p}")
        stats["final_realized_pnl"] = float(df["realized_pnl"].iloc[-1])
        stats["final_total_pnl"] = float(df["total_pnl"].iloc[-1])
    else:
        print(f"  [{feed}] pnl.csv missing realized/unrealized/total_pnl columns -- skipping PnL-over-time plot")

    # --- counterparty analysis needs the newer columns; skip gracefully if absent ---
    if not {"counterparty", "counterparty_role", "event_type", "fill_volume"}.issubset(df.columns):
        print(f"  [{feed}] pnl.csv has no counterparty/counterparty_role columns "
              f"(older capture?) -- skipping counterparty fulfillment analysis")
        return stats

    fills = df[(df["event_type"] == "FILL") & (df["counterparty"].notna()) &
               (df["counterparty"] != "-")].copy()
    if fills.empty:
        print(f"  [{feed}] no FILL rows with counterparty info -- skipping counterparty analysis")
        return stats

    passive_fills = fills[fills["counterparty_role"] == "PASSIVE"].sort_values("ts")  # they filled us
    aggressor_fills = fills[fills["counterparty_role"] == "AGGRESSOR"].sort_values("ts")  # we filled them

    # --- 2. who's been fulfilling my orders, over time ---
    # (cumulative volume filled BY each counterparty, PASSIVE role only --
    # this is specifically "who fulfills my orders", not "who I trade with"
    # in general, which the breakdown plot below covers separately)
    if not passive_fills.empty:
        top_cps = (passive_fills.groupby("counterparty")["fill_volume"].sum()
                   .sort_values(ascending=False))
        top_names = top_cps.head(top_n_counterparties).index.tolist()

        fig, ax = plt.subplots(figsize=(14, 6))
        cmap = plt.get_cmap("tab10")
        for i, cp in enumerate(top_names):
            sub = passive_fills[passive_fills["counterparty"] == cp].set_index("ts")["fill_volume"]
            cum = sub.cumsum()
            ax.step(cum.index, cum.values, where="post", label=cp, color=cmap(i % 10), linewidth=1.3)

        # bucket anyone outside the top N into a single "other" line so the
        # legend doesn't explode with long-tail counterparties
        other_names = [c for c in top_cps.index if c not in top_names]
        if other_names:
            other = passive_fills[passive_fills["counterparty"].isin(other_names)].set_index("ts")["fill_volume"]
            cum_other = other.cumsum()
            ax.step(cum_other.index, cum_other.values, where="post", label="other",
                    color="gray", linewidth=1.0, linestyle=":")

        ax.set_title(f"{feed}: cumulative volume filled by each counterparty "
                     f"(who's fulfilling my resting orders)")
        ax.set_xlabel("time")
        ax.set_ylabel("cumulative volume filled")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        p = os.path.join(out_dir, f"{feed}_counterparty_over_time.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  [{feed}] wrote {p} ({len(passive_fills)} passive fills, "
              f"{len(top_cps)} distinct counterparties)")
        stats["top_fulfilling_counterparties"] = list(top_cps.head(top_n_counterparties).items())
    else:
        print(f"  [{feed}] no PASSIVE fills (nobody has hit our resting orders) -- "
              f"skipping counterparty-over-time plot")

    # --- 3. overall breakdown: volume traded with each counterparty, by role ---
    role_volume = (fills.groupby(["counterparty", "counterparty_role"])["fill_volume"]
                   .sum().unstack(fill_value=0.0))
    for col in ("PASSIVE", "AGGRESSOR"):
        if col not in role_volume.columns:
            role_volume[col] = 0.0
    role_volume["total"] = role_volume["PASSIVE"] + role_volume["AGGRESSOR"]
    role_volume = role_volume.sort_values("total", ascending=False).head(top_n_counterparties)

    if not role_volume.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(role_volume))
        ax.bar(x, role_volume["PASSIVE"], label="they filled us (PASSIVE)", color="tab:green")
        ax.bar(x, role_volume["AGGRESSOR"], bottom=role_volume["PASSIVE"],
               label="we filled them (AGGRESSOR)", color="tab:red")
        ax.set_xticks(x)
        ax.set_xticklabels(role_volume.index, rotation=45, ha="right")
        ax.set_title(f"{feed}: total volume traded per counterparty, by role")
        ax.set_ylabel("volume")
        ax.legend(fontsize=9)
        fig.tight_layout()
        p = os.path.join(out_dir, f"{feed}_counterparty_breakdown.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  [{feed}] wrote {p}")

    stats["n_fills_with_counterparty"] = len(fills)
    stats["n_passive_fills"] = len(passive_fills)
    stats["n_aggressor_fills"] = len(aggressor_fills)
    stats["counterparty_breakdown"] = role_volume

    return stats


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--pnl-csv", default=None, help="default: <data-dir>/pnl.csv")
    p.add_argument("--discovery-csv", default=None, help="default: <data-dir>/discovery.csv")
    p.add_argument("--out-dir", default="eda_out")
    p.add_argument("--feed", default=None, help="restrict to one feed (default: all discovered)")
    p.add_argument("--resample-rule", default="500ms", help="uniform grid for stats (default 500ms)")
    p.add_argument("--plot-resample-threshold", type=int, default=50000,
                   help="downsample plots above this many raw rows (default 50000)")
    p.add_argument("--own-senders", default="", help="comma-separated sender IDs to exclude as 'own'")
    p.add_argument("--mover-pattern", default="MOVER", help="substring to match mover sender IDs")
    p.add_argument("--sweep-window", default="1s", help="rolling window for sweep volume detection")
    p.add_argument("--sweep-volume-multiple", type=float, default=4.0)
    p.add_argument("--acf-lags", type=int, default=40)
    p.add_argument("--coint-significance", type=float, default=0.05)
    p.add_argument("--top-n-counterparties", type=int, default=10,
                   help="how many distinct counterparties to break out individually (default 10)")
    args = p.parse_args()

    bbo_path = os.path.join(args.data_dir, "bbo_data.csv")
    pnl_csv_path = args.pnl_csv or os.path.join(args.data_dir, "pnl.csv")
    discovery_csv_path = args.discovery_csv or os.path.join(args.data_dir, "discovery.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    own_senders = [s.strip() for s in args.own_senders.split(",") if s.strip()]

    # bbo_data.csv is a separate full-traffic exchange capture -- the
    # quoter/hedger seats never write it themselves, so a quoter-only run
    # will legitimately lack this file. Degrade gracefully rather than
    # exiting: everything downstream that needs it (mid-vs-ref, other-party
    # order/trade plots, half-life, returns ACF, cointegration) is skipped,
    # but the own-PnL, counterparty, and discovery.csv analyses still run.
    have_bbo = os.path.exists(bbo_path)
    if have_bbo:
        print(f"loading {bbo_path} ...")
        parsed = parse_bbo_data(bbo_path)
        bbo_df, add_df, cancel_df, trade_df = (
            parsed["bbo"], parsed["order_add"], parsed["order_cancel"], parsed["trade"])
        print(f"  BBO={len(bbo_df)}, order_add={len(add_df)}, order_cancel={len(cancel_df)}, "
              f"trade(deduped)={len(trade_df)}, own_fill={len(parsed['fill'])}, "
              f"own_pnl_snap={len(parsed['pnl_snap'])}")
        if bbo_df.empty:
            print("  no BBO rows in bbo_data.csv -- treating as absent")
            have_bbo = False
    if not have_bbo:
        print(f"  {bbo_path} not found/empty -- this is expected for a quoter-only run "
              f"(it only writes pnl.csv/discovery.csv); skipping all bbo_data.csv-only analyses")
        parsed = {"bbo": pd.DataFrame(), "order_add": pd.DataFrame(), "order_cancel": pd.DataFrame(),
                  "trade": pd.DataFrame(), "fill": pd.DataFrame(), "pnl_snap": pd.DataFrame()}
        bbo_df, add_df, cancel_df, trade_df = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    pnl_csv_df = load_pnl_csv(pnl_csv_path)
    if not pnl_csv_df.empty:
        has_cp = {"counterparty", "counterparty_role"}.issubset(pnl_csv_df.columns)
        print(f"  loaded {pnl_csv_path}: {len(pnl_csv_df)} rows "
              f"(counterparty columns: {'yes' if has_cp else 'no -- older schema'})")
    else:
        print(f"  {pnl_csv_path} not found/empty -- will use ref_price from bbo_data.csv PNL rows if present, "
              f"and own-PnL/counterparty analysis will be skipped")

    discovery_df = load_discovery_csv(discovery_csv_path)
    if not discovery_df.empty:
        print(f"  loaded {discovery_csv_path}: {len(discovery_df)} rows, "
              f"events: {sorted(discovery_df['event'].unique().tolist())}")
    else:
        print(f"  {discovery_csv_path} not found/empty -- other-trader (mover/rel) analysis will be skipped")

    if not have_bbo and pnl_csv_df.empty and discovery_df.empty:
        print("error: none of bbo_data.csv, pnl.csv, discovery.csv have any data -- nothing to analyze",
              file=sys.stderr)
        sys.exit(1)

    # Feeds come from whichever inputs are actually present -- bbo_data.csv
    # if we have it (it's the richest source), else pnl.csv.
    if args.feed is not None:
        feeds = [args.feed]
    elif have_bbo:
        feeds = sorted(bbo_df["feed"].unique())
    elif not pnl_csv_df.empty and "feed" in pnl_csv_df.columns:
        feeds = sorted(pnl_csv_df["feed"].unique())
    else:
        feeds = []
    print(f"discovered feeds: {feeds}")

    mid_series_by_feed = {}
    summary = []

    for feed in feeds:
        print(f"\n=== {feed} ===")
        order_stats, trade_stats, ret_stats = {}, {}, {}
        hl = {"half_life_sec": None, "beta": None, "pvalue": None, "note": "no bbo_data.csv -- not computed"}

        if have_bbo:
            bbo_feed = bbo_df[bbo_df["feed"] == feed].reset_index(drop=True)
            if bbo_feed.empty:
                print(f"  [{feed}] no BBO rows for this feed -- skipping bbo-dependent analyses")
            else:
                mid_u = uniform_mid_series(bbo_feed, args.resample_rule)
                mid_series_by_feed[feed] = mid_u

                # --- ref_price series: prefer dedicated pnl.csv, fall back
                # to bbo_data.csv's own PNL rows ---
                ref_series = pd.Series(dtype=float)
                if not pnl_csv_df.empty and "ref_price" in pnl_csv_df.columns:
                    rp = pnl_csv_df[pnl_csv_df["feed"] == feed].set_index("ts")["ref_price"]
                    if not rp.empty:
                        ref_series = rp.resample(args.resample_rule).last().ffill()
                if ref_series.empty and not parsed["pnl_snap"].empty:
                    rp2 = parsed["pnl_snap"]
                    rp2 = rp2[(rp2["feed"] == feed) & rp2["ref_price"].notna()].set_index("ts")["ref_price"]
                    if not rp2.empty:
                        ref_series = rp2.resample(args.resample_rule).last().ffill()

                # 1. mid vs ref
                plot_mid_vs_ref(feed, bbo_feed, ref_series,
                                os.path.join(args.out_dir, f"{feed}_mid_vs_ref.png"),
                                args.plot_resample_threshold, args.resample_rule)

                # 2. order postings
                order_stats = analyze_order_postings(feed, add_df, mid_u, args.out_dir, own_senders,
                                                      args.plot_resample_threshold, args.resample_rule,
                                                      args.acf_lags)

                # 3. trades + mover/sweep
                trade_stats = analyze_trades(feed, trade_df, mid_u, args.out_dir, own_senders,
                                              args.plot_resample_threshold, args.resample_rule,
                                              args.mover_pattern, args.sweep_window, args.sweep_volume_multiple)

                # 4. half-life
                hl = compute_half_life(mid_u)
                print(f"  [{feed}] half-life: {hl}")

                # 5. returns + ACF
                ret_stats = analyze_returns(feed, mid_u, args.out_dir, args.acf_lags)

        # 7. own PnL + counterparty fulfillment (from pnl.csv) -- always
        # runs, independent of bbo_data.csv
        own_pnl_stats = analyze_own_pnl(feed, pnl_csv_df, args.out_dir,
                                         args.plot_resample_threshold, args.resample_rule,
                                         args.top_n_counterparties)

        summary.append({
            "feed": feed, "half_life": hl, "order_stats": order_stats,
            "trade_stats": trade_stats, "return_stats": ret_stats,
            "own_pnl_stats": own_pnl_stats,
        })

    # 6. cointegration across feeds (needs bbo_data.csv-derived mid series)
    coint_results = {"pairwise": [], "johansen": None}
    if mid_series_by_feed:
        print("\n=== Cross-feed cointegration ===")
        coint_results = run_cointegration_tests(mid_series_by_feed, args.out_dir, args.coint_significance)

    # 6b. other-trader behavior (mover identity/dominance, recovered LP
    # offsets) -- from discovery.csv, independent of bbo_data.csv
    print("\n=== Other-trader behavior (discovery.csv) ===")
    discovery_stats = analyze_discovery(discovery_df, args.out_dir)

    # --- write summary.txt ---
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Market EDA summary\n" + "=" * 60 + "\n\n")
        for s in summary:
            f.write(f"--- {s['feed']} ---\n")
            hl = s["half_life"]
            if hl["half_life_sec"] is not None:
                f.write(f"  half-life: {hl['half_life_sec']:.2f}s (beta={hl['beta']:.5f}, p={hl['pvalue']:.4f})\n")
            else:
                f.write(f"  half-life: {hl['note']} (beta={hl['beta']})\n")
            if s["order_stats"]:
                f.write(f"  order arrival ACF significant lags: "
                        f"{s['order_stats'].get('order_arrival_acf_significant_lags', 'n/a')}\n")
                f.write(f"  mean order inter-arrival: "
                        f"{s['order_stats'].get('order_arrival_mean_gap_sec', float('nan')):.4f}s\n")
            if s["trade_stats"]:
                ts_ = s["trade_stats"]
                f.write(f"  trades: {ts_.get('n_trades', 0)}, mover trades: {ts_.get('n_mover_trades', 0)}, "
                        f"sweep events: {ts_.get('n_sweep_events', 0)}\n")
            if s["return_stats"]:
                rs = s["return_stats"]
                f.write(f"  returns: n={rs.get('n_returns')}, mean={rs.get('return_mean'):.6f}, "
                        f"std={rs.get('return_std'):.6f}, significant ACF lags: "
                        f"{rs.get('significant_acf_lags', 'none')}\n")
            ops = s.get("own_pnl_stats") or {}
            if ops:
                if "final_total_pnl" in ops:
                    f.write(f"  own PnL: final realized={ops.get('final_realized_pnl'):.2f}, "
                            f"final total={ops.get('final_total_pnl'):.2f}\n")
                if "n_fills_with_counterparty" in ops:
                    f.write(f"  fills with known counterparty: {ops['n_fills_with_counterparty']} "
                            f"(passive/they-filled-us: {ops['n_passive_fills']}, "
                            f"aggressor/we-filled-them: {ops['n_aggressor_fills']})\n")
                if ops.get("top_fulfilling_counterparties"):
                    f.write("  top counterparties fulfilling our orders (by volume):\n")
                    for name, vol in ops["top_fulfilling_counterparties"]:
                        f.write(f"    {name}: {vol:.0f}\n")
            f.write("\n")

        if coint_results["pairwise"] or coint_results["johansen"]:
            f.write("--- Cross-feed cointegration (Engle-Granger, pairwise) ---\n")
            for r in coint_results["pairwise"]:
                f.write(f"  {r['pair'][0]} vs {r['pair'][1]}: p={r['pvalue']:.4f}, "
                        f"hedge_ratio={r['hedge_ratio']:.4f}, cointegrated={r['cointegrated']}\n")
            if coint_results["johansen"]:
                j = coint_results["johansen"]
                f.write(f"\n--- Johansen test ({'+'.join(j['feeds'])}) ---\n")
                f.write(f"  trace stats: {[f'{v:.3f}' for v in j['trace_stats']]}\n")
                f.write(f"  95% critical values: {[f'{v:.3f}' for v in j['critical_values_95']]}\n")
                f.write(f"  cointegrating relations at 95%: {j['n_cointegrating_relations_95pct']}\n")
                if j["n_cointegrating_relations_95pct"] > 0:
                    f.write(f"  first cointegrating vector (weights on {j['feeds']}): "
                            f"{[f'{v:.4f}' for v in np.array(j['eigenvectors'])[:, 0]]}\n")
            f.write("\n")

        if discovery_stats:
            f.write("--- Other-trader behavior (discovery.csv) ---\n")
            f.write(f"  related pairs found: {discovery_stats.get('n_pairs_related', 0)}, "
                    f"ambiguous clusters: {discovery_stats.get('n_ambiguous_clusters', 0)}\n")
            if discovery_stats.get("drivers"):
                f.write("  identified drivers (feed: dominant sender, dominance fraction):\n")
                for feed_name, mover, frac in discovery_stats["drivers"]:
                    f.write(f"    {feed_name}: mover={mover}, frac={frac:.3f}\n")
            if discovery_stats.get("piggybacks"):
                f.write("  identified piggybacks (feed: driver):\n")
                for pb, driver in discovery_stats["piggybacks"]:
                    f.write(f"    {pb}: driver={driver}\n")
            if discovery_stats.get("rel_by_piggyback"):
                f.write("  recovered driver/piggyback offsets (final rel, once trusted):\n")
                for pb, info in discovery_stats["rel_by_piggyback"].items():
                    f.write(f"    {pb} (driver={info['driver']}): rel={info['final_rel']:.4f}, "
                            f"confirmations={info['n_confirmations']}\n")
            f.write("\n")

    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
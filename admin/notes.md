# NOTES

## Design decisions

- **Front-month (AAH6) traded via mean reversion.** The mover bot's target
  selection is a symmetric bounded random walk (target clipped around a
  fixed center, direction a fair coin flip, step size independent of
  direction), which guarantees the front-month price is positive-recurrent
  around a known mean. This made mean reversion a natural, verifiable
  strategy rather than a directional bet: entries are armed against the
  mover's sweep, exits ladder back toward the estimated true mean, gated by
  a z-score entry threshold.
- **Back-month instruments (piggybacks) traded via stale-ladder crossing.**
  A back-month LP bot reprices its ladder around the front-month's fair
  value plus a fixed offset, but only reposts when the rounded center
  crosses an integer boundary. Between the front-month moving and the LP's
  repost landing, the LP's resting ladder is briefly stale relative to the
  freshly-implied center. The quoter crosses that stale side (fill-and-kill
  only, no resting orders) once it's identified as mispriced.
- **Clustering assumption:** front-month/back-month clusters are discovered
  at runtime (lagged correlation between instruments), with the assumption
  that each cluster contains exactly one front-month instrument (identified
  by dominant single-aggressor volume share) and one or more back-months.
- **Taker was disabled.** The provided taker took directional bets on inherently non-directional contracts.  
  So running it would only add noise/risk without
  a plausible source of profit.  The contract that was given in the codebase mismatched the bot contracts.  Silencing it was a deliberate scope
  decision, not an oversight.
- **Hedger is a separate process**, independent per instrument (no
  cross-instrument netting), using a dollar-notional risk cap so
  differently-priced instruments sit on a consistent risk footing. It
  reconstructs desk position purely from watching fill messages (no
  cooperation needed from the quoter/taker for position tracking).
- **Quoter/hedger coordination uses two separate KV buckets** (DESK_STATE,
  written by the quoter; HEDGE_STATE, written by the hedger) rather than one
  shared blob. This was specifically to avoid the two processes clobbering
  each other's fields on a read-modify-write, or one process missing a
  field update because it partially overwrote the other's data.
- **Re-deriving bot parameters/behaviours**   I wanted to re-derive the bots parameters/behaviours
  because I don't how many other bots there would be in grading, which is why I wanted to constantly rediscover the parameters and information about the system to trade on it.  

## Assumptions

- Treated the underlying data-generating process as stationary: half-life
  is a single fixed constant estimated once from a captured sample, not
  recalibrated live. Given more time, incremental live recalibration would
  be the natural next step rather than trusting a frozen estimate for the
  full grading run.
- Assumed bot behaviour (mover walk mechanics, LP requote pattern, taker
  behaviour) observed in the sim environment will hold in the grading
  environment. Nothing in the strategy re-derives these mechanics live
  beyond the warmup clustering/half-life pass.
- Assumed each discovered cluster contains exactly one front-month
  instrument (single dominant aggressor) — clusters with no dominant
  instrument are left untraded rather than guessed at.


## Hurdles

- **Rel-recovery and crossing wasted significant time.** An earlier design
  tried to recover each piggyback's price offset from the LP's own
  requote bursts in order to cross profitably, but this required winning a
  timing race against causality itself (NATS gives no cross-subscriber
  delivery-order guarantee, so the LP's decision-time fair value and a
  live-read fair value aren't reconcilable). The follow-up attempt at a
  pure market-making/crossing strategy also didn't work well.  In hindsight
  the strategy itself was poorly specified (the profit mechanism for
  "cross now, unwind later" stemmed from a misunderstanding from the previous attempts observations), and
  crossing as a concept wasn't well understood at the time. Mean reversion
  ended up being a strategy that was actually possible to reason about
  correctly, which is a large part of why it was adopted over crossing.
- **Learning NATS and C++ simultaneously was a real cost.** This was a
  first C++ project, so basic patterns (pointer-to-pointer out-params,
  manual resource cleanup, no exceptions, blocking vs. async request
  patterns) all had to be learned alongside the actual trading logic that wasn't just look at the data and find the strat.  Actually emulating the work of a trading desk of trying to manage latency, orders, execution timing, stale orders and getting it work all at once was really tough, especially in a language I didn't know anything about. 
- **Working with an AI coding assistant had its own failure mode.**
  Imprecise descriptions of intent led to the assistant producing code that
  didn't match what was actually needed, which sometimes wasn't caught
  until later. Combined with usage limits being hit mid-session, this led
  to periods of being stuck with partially-broken C++ and no easy way to
  get it fixed quickly.  It a process hurdle as much as a technical one.
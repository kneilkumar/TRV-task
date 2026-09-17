#!/usr/bin/env python3
"""
analyze_pnl.py — diagnose where the quoter is losing PnL.

Reads the two logs the strategy writes:
  - data/bbo_data.csv : interleaved BBO / MD / FILL / PNL rows (no header,
    tag-prefixed, variable column count per tag — see skeleton.cpp's
    writeRow / applyFill / main loop for the exact formats this parses)
  - data/pnl.csv       : dedicated PnL log with a proper header
    (ts,event_type,feed,side,fill_price,fill_volume,position,
     avg_entry_price,mid,realized_pnl,unrealized_pnl,total_pnl,
     pnl_since_recalib)

Main question this script answers: WHERE is realized PnL actually dropping,
and does it line up with (a) fair-value lag during trends (adverse
selection) or (b) sweep events the 250ms requote cycle was too slow to
react to?

Usage:
    python3 analyze_pnl.py --data-dir data/ --out-dir analysis_out/
    python3 analyze_pnl.py --data-dir data/ --feed AAH6 --sweep-window-ms 500

Outputs (per feed, written to --out-dir):
    <feed>_pnl_vs_fairvalue.png   -- the main diagnostic plot requested
    <feed>_loss_events.csv        -- table of realized-PnL-drop events with
                                      trend/sweep classification
    summary.txt                   -- overall verdict across all feeds
"""

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")  # headless: we only ever save PNGs
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is required: pip install matplotlib --break-system-packages",
          file=sys.stderr)
    raise


# ============================================================================
# Loading & parsing
# ============================================================================

# bbo_data.csv is NOT a normal fixed-column CSV -- each tag has its own
# shape, and pandas.read_csv can't handle ragged rows directly. We read it
# line-by-line and bucket rows by tag instead.
def load_bbo_data(path):
    """Returns dict of DataFrames: {'bbo': ..., 'fill': ..., 'pnl_snap': ...}
    Ignores MD rows (order/execution wire messages) -- not needed for this
    analysis and their column count varies by message type.
    """
    bbo_rows = []
    fill_rows = []
    pnl_rows = []

    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(",")
            tag = parts[0]

            if tag == "BBO":
                # BBO,ex.bbo.<FEED>,<ts>,<FEED>,<bid_px>,<bid_vol>,<ask_px>,<ask_vol>
                if len(parts) != 8:
                    continue
                try:
                    bbo_rows.append({
                        "ts": int(parts[2]),
                        "feed": parts[3],
                        "bid_px": _num_or_nan(parts[4]),
                        "bid_vol": _num_or_nan(parts[5]),
                        "ask_px": _num_or_nan(parts[6]),
                        "ask_vol": _num_or_nan(parts[7]),
                    })
                except ValueError:
                    continue

            elif tag == "FILL":
                # FILL,<ts>,<feed>,<side>,<price>,<volume>,<posBefore>,
                #      <posAfter>,<pnlDelta>,<realizedPnL>,<pnlSinceRecalib>
                if len(parts) != 11:
                    continue
                try:
                    fill_rows.append({
                        "ts": int(parts[1]),
                        "feed": parts[2],
                        "side": parts[3],
                        "price": float(parts[4]),
                        "volume": float(parts[5]),
                        "position_before": float(parts[6]),
                        "position_after": float(parts[7]),
                        "pnl_delta": float(parts[8]),
                        "realized_pnl": float(parts[9]),
                        "pnl_since_recalib": float(parts[10]),
                    })
                except ValueError:
                    continue

            elif tag == "PNL":
                # PNL,<ts>,<feed>,<position>,<avgEntryPrice>,<realizedPnL>,
                #     <pnlSinceRecalib>,<rollingMean>,<rollingStd>,<zscore>,
                #     <restingBidPx>,<restingAskPx>
                if len(parts) != 12:
                    continue
                try:
                    pnl_rows.append({
                        "ts": int(parts[1]),
                        "feed": parts[2],
                        "position": float(parts[3]),
                        "avg_entry_price": float(parts[4]),
                        "realized_pnl": float(parts[5]),
                        "pnl_since_recalib": float(parts[6]),
                        "rolling_mean": float(parts[7]),
                        "rolling_std": float(parts[8]),
                        "zscore": float(parts[9]),
                        "resting_bid_px": _num_or_nan(parts[10]),
                        "resting_ask_px": _num_or_nan(parts[11]),
                    })
                except ValueError:
                    continue
            # MD and anything else: skip.

    bbo_df = pd.DataFrame(bbo_rows)
    fill_df = pd.DataFrame(fill_rows)
    pnl_df = pd.DataFrame(pnl_rows)

    for df in (bbo_df, fill_df, pnl_df):
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"], unit="ns")
            df.sort_values("ts", inplace=True)
            df.reset_index(drop=True, inplace=True)

    if not bbo_df.empty:
        bbo_df["mid"] = (bbo_df["bid_px"] + bbo_df["ask_px"]) / 2.0

    return {"bbo": bbo_df, "fill": fill_df, "pnl_snap": pnl_df}


def _num_or_nan(s):
    if s == "-":
        return np.nan
    return float(s)


def load_pnl_csv(path):
    """Load the dedicated pnl.csv (proper header, FILL/SNAPSHOT rows)."""
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], unit="ns")
    df.sort_values("ts", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ============================================================================
# Sweep / trend detection
# ============================================================================

@dataclass
class SweepEvent:
    ts: pd.Timestamp
    spread_before: float
    spread_peak: float


def detect_sweeps(bbo_feed_df, spread_jump_multiple=3.0, resample_rule="100ms"):
    """Flags timestamps where spread jumped sharply above its recent local
    baseline -- the same heuristic used earlier to spot the AAH6 spike.
    Works off a resampled spread series so it's robust to uneven BBO update
    frequency. Returns a DataFrame of candidate sweep timestamps.
    """
    if bbo_feed_df.empty:
        return pd.DataFrame(columns=["ts", "spread"])

    s = bbo_feed_df.set_index("ts")
    spread = (s["ask_px"] - s["bid_px"]).resample(resample_rule).last().ffill()
    if spread.empty:
        return pd.DataFrame(columns=["ts", "spread"])

    baseline = spread.rolling(300, min_periods=10).median()  # ~30s at 100ms buckets
    jump = spread > (baseline * spread_jump_multiple)
    jump = jump.fillna(False)

    sweeps = spread[jump].reset_index()
    sweeps.columns = ["ts", "spread"]
    return sweeps


def detect_trends(bbo_feed_df, pnl_feed_df, trend_window="5s", z_threshold=1.5):
    """Flags timestamps where mid was trending away from rollingMean -- i.e.
    fair value (rollingMean) lagging the market -- using the zscore already
    computed by the strategy itself (mid vs rolling mean, in rolling-std
    units). |zscore| above threshold = fair value is lagging price.
    """
    if pnl_feed_df.empty:
        return pd.DataFrame(columns=["ts", "zscore"])

    lagging = pnl_feed_df[pnl_feed_df["zscore"].abs() >= z_threshold]
    return lagging[["ts", "zscore", "rolling_mean"]].reset_index(drop=True)


def classify_loss_events(fill_feed_df, sweeps_df, trends_df, window_ms=500):
    """For each fill that realized a loss (pnl_delta < 0), check whether a
    sweep or a fair-value-lag (trend) event happened within `window_ms`
    beforehand. Returns fill rows augmented with near_sweep / near_trend_lag
    boolean columns, restricted to losing fills.
    """
    losses = fill_feed_df[fill_feed_df["pnl_delta"] < 0].copy()
    if losses.empty:
        losses["near_sweep"] = pd.Series(dtype=bool)
        losses["near_trend_lag"] = pd.Series(dtype=bool)
        return losses

    window = pd.Timedelta(milliseconds=window_ms)

    def _near_any(ts, events_df, col):
        if events_df.empty:
            return False
        diffs = (events_df[col] - ts).abs()
        return bool((diffs <= window).any())

    losses["near_sweep"] = losses["ts"].apply(lambda t: _near_any(t, sweeps_df, "ts"))
    losses["near_trend_lag"] = losses["ts"].apply(lambda t: _near_any(t, trends_df, "ts"))
    return losses


# ============================================================================
# Plotting
# ============================================================================

def plot_pnl_vs_fairvalue(feed, bbo_feed_df, pnl_feed_df, sweeps_df, out_path):
    """The main diagnostic plot requested: realized PnL over time (from PNL
    snapshot rows) against rollingMean and raw BBO mid, on twin y-axes, with
    detected sweep timestamps marked. Visual read:
      - realized PnL drops that line up with mid pulling away from
        rollingMean (fair value lagging) -> adverse-selection-during-trends
      - realized PnL drops clustered tightly at sweep markers, with mid and
        rollingMean otherwise close together -> 250ms too slow for sweeps
    """
    if pnl_feed_df.empty or bbo_feed_df.empty:
        print(f"  [{feed}] skipping plot: missing PNL or BBO data")
        return

    fig, ax1 = plt.subplots(figsize=(14, 7))

    ax1.plot(bbo_feed_df["ts"], bbo_feed_df["mid"], color="tab:gray",
             linewidth=0.8, alpha=0.7, label="raw BBO mid")
    ax1.plot(pnl_feed_df["ts"], pnl_feed_df["rolling_mean"], color="tab:blue",
              linewidth=1.2, label="rollingMean (fair value)")
    ax1.set_ylabel("price")
    ax1.set_xlabel("time")

    ax2 = ax1.twinx()
    ax2.plot(pnl_feed_df["ts"], pnl_feed_df["realized_pnl"], color="tab:red",
              linewidth=1.5, label="realized PnL")
    ax2.axhline(0, color="tab:red", linewidth=0.5, linestyle=":")
    ax2.set_ylabel("realized PnL", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    if not sweeps_df.empty:
        for i, ts in enumerate(sweeps_df["ts"]):
            ax1.axvline(ts, color="tab:orange", linewidth=0.8, alpha=0.5,
                        label="detected sweep" if i == 0 else None)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    ax1.set_title(f"{feed}: realized PnL vs. fair value (rollingMean) vs. raw mid")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [{feed}] wrote {out_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data",
                         help="Directory containing bbo_data.csv and pnl.csv (default: data/)")
    parser.add_argument("--out-dir", default="analysis_out",
                         help="Directory to write plots/tables to (default: analysis_out/)")
    parser.add_argument("--feed", default=None,
                         help="Restrict analysis to a single feed (default: all feeds found)")
    parser.add_argument("--sweep-window-ms", type=int, default=500,
                         help="Window (ms) around a fill to check for a nearby sweep/trend event (default: 500)")
    parser.add_argument("--sweep-spread-multiple", type=float, default=3.0,
                         help="Spread jump multiple over rolling median to flag as a sweep (default: 3.0)")
    parser.add_argument("--trend-zscore", type=float, default=1.5,
                         help="|zscore| threshold above which fair value is considered 'lagging' (default: 1.5)")
    args = parser.parse_args()

    bbo_path = os.path.join(args.data_dir, "bbo_data.csv")
    pnl_path = os.path.join(args.data_dir, "pnl.csv")
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(bbo_path):
        print(f"error: {bbo_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"loading {bbo_path} ...")
    parsed = load_bbo_data(bbo_path)
    bbo_df, fill_df, pnl_snap_df = parsed["bbo"], parsed["fill"], parsed["pnl_snap"]
    print(f"  BBO rows: {len(bbo_df)}, FILL rows: {len(fill_df)}, PNL snapshot rows: {len(pnl_snap_df)}")

    # pnl.csv (dedicated log) is optional -- older runs / this exact skeleton
    # revision may not have it yet. Used only as a supplementary cross-check
    # (it carries unrealized + total PnL, which bbo_data.csv's PNL rows don't).
    pnl_csv_df = load_pnl_csv(pnl_path)
    if not pnl_csv_df.empty:
        print(f"  pnl.csv rows: {len(pnl_csv_df)} (has unrealized/total PnL -- used as cross-check)")
    else:
        print(f"  {pnl_path} not found or empty -- proceeding with bbo_data.csv PNL rows only")

    if pnl_snap_df.empty:
        print("error: no PNL snapshot rows found in bbo_data.csv -- nothing to analyze", file=sys.stderr)
        sys.exit(1)

    feeds = sorted(pnl_snap_df["feed"].unique()) if args.feed is None else [args.feed]

    summary_lines = []

    for feed in feeds:
        print(f"\n=== {feed} ===")
        bbo_feed = bbo_df[bbo_df["feed"] == feed].reset_index(drop=True) if not bbo_df.empty else pd.DataFrame()
        fill_feed = fill_df[fill_df["feed"] == feed].reset_index(drop=True) if not fill_df.empty else pd.DataFrame()
        pnl_feed = pnl_snap_df[pnl_snap_df["feed"] == feed].reset_index(drop=True)

        if bbo_feed.empty:
            print(f"  [{feed}] no BBO rows -- skipping")
            continue

        sweeps_df = detect_sweeps(bbo_feed, spread_jump_multiple=args.sweep_spread_multiple)
        trends_df = detect_trends(bbo_feed, pnl_feed, z_threshold=args.trend_zscore)
        print(f"  detected {len(sweeps_df)} candidate sweep event(s), "
              f"{len(trends_df)} fair-value-lag snapshot(s) (|z|>={args.trend_zscore})")

        # --- main requested plot ---
        plot_path = os.path.join(args.out_dir, f"{feed}_pnl_vs_fairvalue.png")
        plot_pnl_vs_fairvalue(feed, bbo_feed, pnl_feed, sweeps_df, plot_path)

        # --- loss event classification ---
        if fill_feed.empty:
            print(f"  [{feed}] no FILL rows -- can't classify individual loss events")
            summary_lines.append(f"{feed}: no fills recorded, plot only.")
            continue

        losses = classify_loss_events(fill_feed, sweeps_df, trends_df,
                                       window_ms=args.sweep_window_ms)
        loss_csv_path = os.path.join(args.out_dir, f"{feed}_loss_events.csv")
        losses.to_csv(loss_csv_path, index=False)
        print(f"  [{feed}] wrote {loss_csv_path} ({len(losses)} losing fill(s))")

        total_loss_pnl = losses["pnl_delta"].sum() if not losses.empty else 0.0
        n_losses = len(losses)
        n_near_sweep = int(losses["near_sweep"].sum()) if n_losses else 0
        n_near_trend = int(losses["near_trend_lag"].sum()) if n_losses else 0
        n_both = int((losses["near_sweep"] & losses["near_trend_lag"]).sum()) if n_losses else 0

        pnl_sweep_only = losses.loc[losses["near_sweep"] & ~losses["near_trend_lag"], "pnl_delta"].sum() if n_losses else 0.0
        pnl_trend_only = losses.loc[losses["near_trend_lag"] & ~losses["near_sweep"], "pnl_delta"].sum() if n_losses else 0.0
        pnl_both = losses.loc[losses["near_sweep"] & losses["near_trend_lag"], "pnl_delta"].sum() if n_losses else 0.0
        pnl_neither = losses.loc[~losses["near_sweep"] & ~losses["near_trend_lag"], "pnl_delta"].sum() if n_losses else 0.0

        print(f"  losing fills: {n_losses}, total realized loss: {total_loss_pnl:.2f}")
        print(f"    near sweep only:      {int((losses['near_sweep'] & ~losses['near_trend_lag']).sum()) if n_losses else 0} fills, pnl {pnl_sweep_only:.2f}")
        print(f"    near trend-lag only:  {int((losses['near_trend_lag'] & ~losses['near_sweep']).sum()) if n_losses else 0} fills, pnl {pnl_trend_only:.2f}")
        print(f"    near both:            {n_both} fills, pnl {pnl_both:.2f}")
        print(f"    near neither:         {n_losses - n_near_sweep - n_near_trend + n_both} fills, pnl {pnl_neither:.2f}")

        if abs(pnl_trend_only) > abs(pnl_sweep_only) * 1.5:
            verdict = "trend/adverse-selection dominant (fair value lagging price)"
        elif abs(pnl_sweep_only) > abs(pnl_trend_only) * 1.5:
            verdict = "sweep-reaction-speed dominant (250ms too slow for sweeps)"
        else:
            verdict = "mixed / inconclusive -- both mechanisms contribute comparably"

        summary_lines.append(
            f"{feed}: {n_losses} losing fills, total realized loss {total_loss_pnl:.2f}. "
            f"sweep-only={pnl_sweep_only:.2f}, trend-only={pnl_trend_only:.2f}, "
            f"both={pnl_both:.2f}, neither={pnl_neither:.2f}. Verdict: {verdict}"
        )

    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("PnL loss diagnosis summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"sweep_window_ms={args.sweep_window_ms}, "
                f"sweep_spread_multiple={args.sweep_spread_multiple}, "
                f"trend_zscore={args.trend_zscore}\n\n")
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nwrote {summary_path}")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
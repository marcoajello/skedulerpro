"""Phase 1 screener: rank funds by discount z-score as of the latest date.

Usage: python -m src.screener [--min-adv 250000] [--top 25]
Writes reports/screen_<date>.csv and prints the ranked table.
"""
from __future__ import annotations

import argparse
import pathlib

import pandas as pd

from . import data_store, signals

REPORTS = pathlib.Path(__file__).resolve().parent.parent / "reports"


def run_screen(panel: pd.DataFrame, min_adv: float = 250_000.0) -> pd.DataFrame:
    sig = signals.compute_signals(panel)
    last_date = sig["date"].max()
    cur = sig[sig["date"] == last_date].copy()

    # Trailing 12m cash yield from actual distributions (not a quoted yield);
    # funds with high yield + falling NAV get the ROC-suspect flag below.
    yr_ago = last_date - pd.Timedelta(days=365)
    tr12 = sig[sig["date"] > yr_ago].groupby("ticker").agg(
        dist_12m=("dist", "sum"), nav_start=("nav", "first"), nav_end=("nav", "last"))
    cur = cur.merge(tr12, on="ticker", how="left")
    cur["yield_12m"] = cur["dist_12m"] / cur["price"]
    cur["nav_change_12m"] = cur["nav_end"] / cur["nav_start"] - 1.0
    # ROC proxy: paying out a lot while NAV erodes by a comparable amount.
    # True ROC classification needs 19a-1 notices (not in free data) - this
    # flag marks funds where the "yield" may just be your capital returning.
    cur["roc_suspect"] = (cur["yield_12m"] > 0.06) & (cur["nav_change_12m"] < -0.05)

    cur["liquid"] = cur["adv_dollar"] >= min_adv
    screened = (cur[cur["liquid"] & cur["z_long"].notna()]
                .sort_values("z_long")
                [["ticker", "date", "price", "nav", "discount", "z_long", "z_short",
                  "adv_dollar", "yield_12m", "nav_change_12m", "roc_suspect"]]
                .reset_index(drop=True))
    return screened


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-adv", type=float, default=250_000.0)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    conn = data_store.connect()
    panel = data_store.load_panel(conn)
    if panel.empty:
        raise SystemExit("Empty store - run `python -m src.build_dataset` first "
                         "(and `python -m src.verify_sources` before that).")
    screened = run_screen(panel, min_adv=args.min_adv)
    last_date = screened["date"].max()

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"screen_{last_date.date().isoformat()}.csv"
    screened.to_csv(out, index=False)

    show = screened.head(args.top).copy()
    for c, fmt in (("discount", "{:.1%}"), ("z_long", "{:+.2f}"), ("z_short", "{:+.2f}"),
                   ("adv_dollar", "{:,.0f}"), ("yield_12m", "{:.1%}"),
                   ("nav_change_12m", "{:+.1%}")):
        show[c] = show[c].map(lambda v, f=fmt: f.format(v) if pd.notna(v) else "-")
    print(f"Cheapest by 1y discount z-score, {last_date.date()} "
          f"(liquidity >= ${args.min_adv:,.0f}/day):\n")
    print(show.drop(columns=["date"]).to_string(index=False))
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()

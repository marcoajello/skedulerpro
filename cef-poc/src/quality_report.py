"""Data-quality report: coverage, gaps, stale NAVs, dropped funds.

Usage: python -m src.quality_report
Writes reports/data_quality.md. The backtest should only be trusted if this
report looks sane (enough funds, enough history, few stale NAVs).
"""
from __future__ import annotations

import datetime as dt
import pathlib

import pandas as pd

from . import data_store

REPORTS = pathlib.Path(__file__).resolve().parent.parent / "reports"


def build_report(conn) -> str:
    funds = data_store.load_funds(conn)
    panel = data_store.load_panel(conn, ok_only=True)
    lines = [f"# Data quality report — {dt.date.today().isoformat()}", ""]

    if funds.empty:
        return "\n".join(lines + ["No funds in store. Run build_dataset first.", ""])

    by_status = funds[funds["status"] != "benchmark"]["status"].value_counts()
    lines += ["## Universe", "",
              f"- funds attempted: {int(by_status.sum())}",
              *[f"- {k}: {v}" for k, v in by_status.items()], ""]
    dropped = funds[~funds["status"].isin(["ok", "benchmark"])]
    if not dropped.empty:
        lines += ["### Dropped funds (with reason)", "",
                  *[f"- {r.ticker}: {r.status} — {r.note}" for r in dropped.itertuples()], ""]

    if panel.empty:
        return "\n".join(lines + ["No daily data stored.", ""])

    per = panel.groupby("ticker").agg(
        first=("date", "min"), last=("date", "max"), days=("date", "count"))
    per["years"] = (per["last"] - per["first"]).dt.days / 365.25

    # Gap detection: worst run of missing weekdays inside each fund's range.
    gaps = {}
    for t, g in panel.groupby("ticker"):
        d = g["date"].sort_values()
        diff = d.diff().dt.days.dropna()
        gaps[t] = int(diff.max()) if len(diff) else 0
    per["max_gap_days"] = pd.Series(gaps)

    # Stale NAV detection: NAV unchanged for many consecutive days is a
    # symptom of a dead/misquoted NAV feed (real NAVs move ~daily).
    stale = {}
    for t, g in panel.groupby("ticker"):
        nav = g.sort_values("date")["nav"]
        runs = (nav.diff() != 0).cumsum()
        stale[t] = int(nav.groupby(runs).size().max())
    per["max_stale_nav_run"] = pd.Series(stale)

    lines += [
        "## Coverage (funds with status=ok)", "",
        f"- funds: {len(per)}",
        f"- median history: {per['years'].median():.1f} years "
        f"(min {per['years'].min():.1f}, max {per['years'].max():.1f})",
        f"- funds with >=3y history: {(per['years'] >= 3).sum()}",
        f"- panel rows: {len(panel):,}",
        f"- date range: {panel['date'].min().date()} -> {panel['date'].max().date()}",
        "",
        "## Suspect data", "",
        f"- funds with a gap > 7 calendar days: "
        f"{(per['max_gap_days'] > 7).sum()} "
        f"({sorted(per[per['max_gap_days'] > 7].index.tolist())[:15]})",
        f"- funds with NAV unchanged >5 consecutive sessions: "
        f"{(per['max_stale_nav_run'] > 5).sum()} "
        f"({sorted(per[per['max_stale_nav_run'] > 5].index.tolist())[:15]})",
        "",
        "## Per-fund detail", "",
        per.sort_values("years").to_string(),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    conn = data_store.connect()
    report = build_report(conn)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "data_quality.md"
    out.write_text(report)
    print(report[:2500])
    print(f"[written] {out}")


if __name__ == "__main__":
    main()

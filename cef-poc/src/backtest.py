"""Phase 2: honest backtest of discount z-score mean reversion.

Design choices (all aimed at not fooling ourselves):
- Trailing-only stats: z-scores at date t use data <= t (see signals.py).
- Execution lag: a signal observed at close of day t is executed at the
  close of day t+1. No same-day fills on information you didn't have.
- Per-trade decomposition: net return is split into (1) discount change,
  (2) NAV price return, (3) distribution income, minus costs, so we can see
  whether P&L actually comes from the thesis (discount compression) or just
  from NAV beta.
- Costs: configurable one-way cost in bps applied twice (entry+exit).
  Default 30bps one-way ~ half-spread + commission for a liquid CEF; thin
  CEFs are excluded by the ADV filter, not modeled.
- Benchmarks matched per trade over the identical window: equal-weight
  universe buy&hold, and SPY buy&hold.
- Out-of-sample: trades are split at the sample midpoint by entry date and
  both halves reported separately.
- Sensitivity: a threshold x horizon grid is always reported, so a result
  that only works at one magic parameter is visible immediately.

Usage: python -m src.backtest [--threshold -1.5] [--horizons 63 126] ...
Writes reports/backtest_report.md and charts to reports/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import pathlib

import numpy as np
import pandas as pd

from . import data_store, signals

REPORTS = pathlib.Path(__file__).resolve().parent.parent / "reports"


# ---------------------------------------------------------------- trades

def find_trades(sig: pd.DataFrame, threshold: float, horizon: int,
                exit_z: float = 0.0, min_adv: float = 250_000.0,
                cost_bps_oneway: float = 30.0) -> pd.DataFrame:
    """Event-driven simulation, one fund at a time, no overlapping trades
    per fund. Returns one row per completed trade with full decomposition.
    Trades still open when data ends are dropped and counted (censored)."""
    trades, censored = [], 0
    for tkr, g in sig.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        i = 1  # start at 1 so 'crossed below today' is well-defined
        while i < len(g) - 1:
            row, prev = g.loc[i], g.loc[i - 1]
            entered = (
                pd.notna(row.z_long) and row.z_long <= threshold
                and (pd.isna(prev.z_long) or prev.z_long > threshold)  # fresh cross
                and pd.notna(row.adv_dollar) and row.adv_dollar >= min_adv
            )
            if not entered:
                i += 1
                continue
            e = i + 1  # execute next close
            entry = g.loc[e]
            if pd.isna(entry.price) or pd.isna(entry.nav):
                i += 1
                continue
            # exit: z reverts to exit_z (executed next close) or horizon end
            x, reason = None, None
            for j in range(e + 1, min(e + horizon, len(g) - 1) + 1):
                zj = g.loc[j, "z_long"]
                if pd.notna(zj) and zj >= exit_z:
                    x, reason = min(j + 1, len(g) - 1), "revert"
                    break
            if x is None:
                if e + horizon <= len(g) - 1:
                    x, reason = e + horizon, "horizon"
                else:
                    censored += 1
                    break  # not enough future data; stop scanning this fund
            exit_ = g.loc[x]
            dist = g.loc[e + 1:x, "dist"].fillna(0.0).sum()  # ex-dates after entry, through exit
            cost = 2 * cost_bps_oneway / 1e4

            gross = (exit_.price - entry.price + dist) / entry.price
            r_nav = exit_.nav / entry.nav - 1.0                      # NAV price return
            d0, d1 = entry.discount, exit_.discount
            r_disc = (1.0 + d1) / (1.0 + d0) - 1.0                   # discount change
            income = dist / entry.price
            trades.append(dict(
                ticker=tkr, entry_date=entry.date, exit_date=exit_.date,
                days_held=int(x - e), signal_z=row.z_long, entry_z=entry.z_long,
                exit_z=exit_.z_long, entry_discount=d0, exit_discount=d1,
                gross_ret=gross, net_ret=gross - cost, nav_component=r_nav,
                discount_component=r_disc, income_component=income, cost=cost,
                exit_reason=reason,
            ))
            i = x + 1  # next possible entry only after this trade closes
    df = pd.DataFrame(trades)
    df.attrs["censored"] = censored
    return df


def attach_benchmarks(trades: pd.DataFrame, panel: pd.DataFrame,
                      spy: pd.DataFrame | None) -> pd.DataFrame:
    """Per trade, the buy&hold return of (a) EW universe, (b) SPY over the
    identical entry->exit window."""
    if trades.empty:
        return trades
    ew = signals.equal_weight_index(panel)
    spy_tri = signals.total_return_index(spy) if spy is not None and len(spy) else None

    def window_ret(idx: pd.Series, a, b):
        w = idx.loc[(idx.index >= a) & (idx.index <= b)]
        return w.iloc[-1] / w.iloc[0] - 1.0 if len(w) >= 2 else np.nan

    trades = trades.copy()
    trades["ew_bench_ret"] = [window_ret(ew, a, b) for a, b in
                              zip(trades.entry_date, trades.exit_date)]
    trades["spy_bench_ret"] = ([window_ret(spy_tri, a, b) for a, b in
                                zip(trades.entry_date, trades.exit_date)]
                               if spy_tri is not None else np.nan)
    trades["excess_vs_ew"] = trades.net_ret - trades.ew_bench_ret
    trades["excess_vs_spy"] = trades.net_ret - trades.spy_bench_ret
    return trades


# ---------------------------------------------------------------- stats

def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0}
    t = trades
    return {
        "n": len(t),
        "hit_rate": float((t.net_ret > 0).mean()),
        "mean_net": float(t.net_ret.mean()),
        "median_net": float(t.net_ret.median()),
        "p10": float(t.net_ret.quantile(0.10)),
        "p90": float(t.net_ret.quantile(0.90)),
        "worst": float(t.net_ret.min()),
        "mean_days": float(t.days_held.mean()),
        "mean_discount_comp": float(t.discount_component.mean()),
        "mean_nav_comp": float(t.nav_component.mean()),
        "mean_income_comp": float(t.income_component.mean()),
        "mean_excess_vs_ew": float(t.excess_vs_ew.mean()),
        "hit_vs_ew": float((t.excess_vs_ew > 0).mean()),
        "mean_excess_vs_spy": float(t.excess_vs_spy.mean()) if t.excess_vs_spy.notna().any() else np.nan,
        "hit_vs_spy": float((t.excess_vs_spy > 0).mean()) if t.excess_vs_spy.notna().any() else np.nan,
    }


def strategy_equity_curve(trades: pd.DataFrame, sig: pd.DataFrame) -> pd.Series:
    """Daily equity of 'equal weight across all open trades, cash when none'.
    Crude but sufficient for a drawdown estimate."""
    if trades.empty:
        return pd.Series(dtype=float)
    px = sig.pivot_table(index="date", columns="ticker", values="price")
    dv = sig.pivot_table(index="date", columns="ticker", values="dist")
    tri = (px + dv.fillna(0.0)) / px.shift(1)  # 1 + daily total return
    dates = px.index.sort_values()
    daily = pd.Series(0.0, index=dates)
    counts = pd.Series(0, index=dates)
    for tr in trades.itertuples():
        mask = (dates > tr.entry_date) & (dates <= tr.exit_date)
        r = tri.loc[mask, tr.ticker] - 1.0
        daily.loc[mask] = daily.loc[mask].add(r.fillna(0.0), fill_value=0.0)
        counts.loc[mask] += 1
    port = daily.where(counts == 0, daily / counts.replace(0, np.nan)).fillna(0.0)
    return (1.0 + port).cumprod()


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    return float((equity / equity.cummax() - 1.0).min())


# ---------------------------------------------------------------- report

def fmt_summary(s: dict) -> str:
    if s.get("n", 0) == 0:
        return "    no trades\n"
    def pct(x):
        return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.2%}"
    return "\n".join([
        f"- trades: {s['n']}, avg hold {s['mean_days']:.0f} trading days",
        f"- hit rate (net>0): {s['hit_rate']:.0%}",
        f"- net per trade: mean {pct(s['mean_net'])}, median {pct(s['median_net'])}, "
        f"p10 {pct(s['p10'])}, p90 {pct(s['p90'])}, worst {pct(s['worst'])}",
        f"- decomposition (mean per trade): discount {pct(s['mean_discount_comp'])}, "
        f"NAV {pct(s['mean_nav_comp'])}, income {pct(s['mean_income_comp'])}",
        f"- vs EW universe: mean excess {pct(s['mean_excess_vs_ew'])}, "
        f"beats it {s['hit_vs_ew']:.0%} of trades",
        f"- vs SPY: mean excess {pct(s['mean_excess_vs_spy'])}, "
        f"beats it {pct(s['hit_vs_spy']) if isinstance(s['hit_vs_spy'], float) and not np.isnan(s['hit_vs_spy']) else 'n/a'} of trades",
    ]) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=-1.5)
    ap.add_argument("--horizons", type=int, nargs="+", default=[63, 126])
    ap.add_argument("--exit-z", type=float, default=0.0)
    ap.add_argument("--min-adv", type=float, default=250_000.0)
    ap.add_argument("--cost-bps", type=float, default=30.0,
                    help="one-way cost in bps (slippage+commission)")
    ap.add_argument("--sens-thresholds", type=float, nargs="+",
                    default=[-1.0, -1.5, -2.0])
    args = ap.parse_args()

    conn = data_store.connect()
    panel = data_store.load_panel(conn)
    if panel.empty:
        raise SystemExit("Empty store - run verify_sources, then build_dataset first.")
    spy = data_store.load_panel(conn, tickers=["SPY"], ok_only=False)

    sig = signals.compute_signals(panel)
    lines = [f"# Backtest report — {dt.date.today().isoformat()}", "",
             f"Universe: {panel.ticker.nunique()} funds, "
             f"{panel.date.min().date()} -> {panel.date.max().date()}. "
             f"Costs: {args.cost_bps:.0f}bps one-way. Entry: 1y z-score crosses "
             f"<= threshold + ADV >= ${args.min_adv:,.0f}. Exit: z >= "
             f"{args.exit_z} or horizon. Signals trail-only; fills lag one day.", ""]

    charts = []
    for hz in args.horizons:
        trades = find_trades(sig, args.threshold, hz, args.exit_z,
                             args.min_adv, args.cost_bps)
        trades = attach_benchmarks(trades, panel, spy)
        lines += [f"## Horizon {hz}d, threshold {args.threshold}", "",
                  fmt_summary(summarize(trades))]
        if not trades.empty:
            mid = trades.entry_date.sort_values().iloc[len(trades) // 2]
            lines += [f"### In-sample half (entries < {mid.date()})", "",
                      fmt_summary(summarize(trades[trades.entry_date < mid])),
                      f"### Out-of-sample half (entries >= {mid.date()})", "",
                      fmt_summary(summarize(trades[trades.entry_date >= mid]))]
            eq = strategy_equity_curve(trades, sig)
            lines += [f"- strategy max drawdown (EW open trades): {max_drawdown(eq):.1%}",
                      f"- censored (still-open, dropped) trades: {trades.attrs.get('censored', 0)}", ""]
            charts.append((hz, trades, eq))
        trades.to_csv(REPORTS / f"trades_h{hz}.csv", index=False)

    # Sensitivity grid: does this only "work" at one magic threshold?
    lines += ["## Sensitivity: mean net excess vs EW universe (per trade)", "",
              "| threshold \\ horizon | " + " | ".join(f"{h}d" for h in args.horizons) + " |",
              "|---|" + "---|" * len(args.horizons)]
    for thr in args.sens_thresholds:
        cells = []
        for hz in args.horizons:
            tr = attach_benchmarks(
                find_trades(sig, thr, hz, args.exit_z, args.min_adv, args.cost_bps),
                panel, spy)
            s = summarize(tr)
            cells.append("no trades" if s["n"] == 0
                         else f"{s['mean_excess_vs_ew']:+.2%} (n={s['n']})")
        lines.append(f"| {thr} | " + " | ".join(cells) + " |")
    lines.append("")

    # Charts
    REPORTS.mkdir(parents=True, exist_ok=True)
    if charts:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(charts), figsize=(6 * len(charts), 4))
        for ax, (hz, trades, _) in zip(np.atleast_1d(axes), charts):
            ax.hist(trades.net_ret, bins=40)
            ax.axvline(0, color="k", lw=1)
            ax.set_title(f"Net return per trade, {hz}d horizon (n={len(trades)})")
        fig.tight_layout()
        fig.savefig(REPORTS / "net_return_hist.png", dpi=120)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        for hz, _, eq in charts:
            if len(eq):
                ax2.plot(eq.index, eq.values, label=f"strategy {hz}d")
        ew = signals.equal_weight_index(panel)
        ax2.plot(ew.index, ew / ew.iloc[0], label="EW universe", alpha=0.7)
        ax2.legend(); ax2.set_title("Equity curves")
        fig2.tight_layout()
        fig2.savefig(REPORTS / "equity_curves.png", dpi=120)
        lines += ["![net return histogram](net_return_hist.png)",
                  "![equity curves](equity_curves.png)", ""]

    out = REPORTS / "backtest_report.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"[written] {out}")


if __name__ == "__main__":
    main()

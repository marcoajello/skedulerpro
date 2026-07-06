"""Core signal math: discount, trailing z-scores, liquidity.

All statistics are TRAILING (rolling windows ending at the observation
date), so a value at date t uses only data <= t. That property is what the
no-lookahead test in tests/test_math.py asserts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WIN_LONG = 252     # ~1 trading year
WIN_SHORT = 126    # ~6 months
MIN_LONG = 189     # require 75% of the window before emitting a z-score
MIN_SHORT = 95
ADV_WIN = 60
MIN_STD = 5e-4     # 0.05% discount std floor: below this the fund's
                   # discount is effectively constant and z is meaningless


def compute_signals(panel: pd.DataFrame) -> pd.DataFrame:
    """panel: long df [ticker, date, price, nav, volume, dist] (nav required).

    Adds: discount, z_long, z_short, adv_dollar. Rows keep panel order
    within ticker (sorted by date).
    """
    out = []
    for _, g in panel.sort_values(["ticker", "date"]).groupby("ticker"):
        g = g.copy()
        g["discount"] = g["price"] / g["nav"] - 1.0
        for col, win, minp in (("z_long", WIN_LONG, MIN_LONG),
                               ("z_short", WIN_SHORT, MIN_SHORT)):
            m = g["discount"].rolling(win, min_periods=minp).mean()
            s = g["discount"].rolling(win, min_periods=minp).std()
            g[col] = (g["discount"] - m) / s.where(s > MIN_STD, np.nan)
        g["adv_dollar"] = (g["price"] * g["volume"]).rolling(
            ADV_WIN, min_periods=20).mean()
        out.append(g)
    return pd.concat(out, ignore_index=True)


def total_return_index(g: pd.DataFrame) -> pd.Series:
    """Per-fund daily total-return index from unadjusted price + cash dists.

    r_t = (P_t + D_t) / P_{t-1} - 1  (distribution credited on ex-date).
    """
    g = g.sort_values("date")
    r = (g["price"] + g["dist"].fillna(0.0)) / g["price"].shift(1) - 1.0
    idx = (1.0 + r.fillna(0.0)).cumprod()
    return pd.Series(idx.values, index=pd.DatetimeIndex(g["date"]))


def equal_weight_index(panel: pd.DataFrame) -> pd.Series:
    """Equal-weight daily-rebalanced total-return index across all funds.

    This is benchmark (a): 'just buy the whole universe and hold'. Any
    strategy that can't beat this isn't extracting value from the signal.
    """
    rets = {}
    for t, g in panel.groupby("ticker"):
        tri = total_return_index(g)
        rets[t] = tri.pct_change()
    mat = pd.DataFrame(rets).sort_index()
    ew = mat.mean(axis=1, skipna=True).fillna(0.0)
    return (1.0 + ew).cumprod()

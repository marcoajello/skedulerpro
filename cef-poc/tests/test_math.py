"""Unit tests for the POC math on SYNTHETIC fixtures.

These fixtures are deliberately artificial series with known dynamics.
They exist to prove the pipeline's arithmetic (z-scores, no-lookahead,
return decomposition, trade lifecycle) — they are NOT market data and no
number from here appears in any findings report.

Fixture funds:
- MRV : NAV flat at 10; discount sits near -5% (small deterministic wiggle),
        then shocks to -12% for 30 days and linearly recovers. A z-score
        strategy SHOULD profit here, purely from the discount component.
- TRAP: same, but the widening to -12% is PERMANENT (the value-trap /
        "NAV falls to meet price" family). The strategy should roughly
        break even minus costs, with ~zero discount component at exit.
- CONST: discount pinned at exactly -15%. "Always cheap" — must produce
        NO z-score (std floor) and therefore no trades.

Run: python -m pytest tests/ -q
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src import data_store, signals
from src.backtest import attach_benchmarks, find_trades
from src.screener import run_screen

N_DAYS = 1400
SHOCK_START, SHOCK_LEN, RECOVER_LEN = 800, 30, 60


def _discount_path(permanent: bool) -> np.ndarray:
    t = np.arange(N_DAYS)
    base = -0.05 + 0.005 * np.sin(2 * np.pi * t / 21)  # small wiggle -> real std
    d = base.copy()
    lo = -0.12
    d[SHOCK_START:SHOCK_START + SHOCK_LEN] = lo
    if permanent:
        d[SHOCK_START + SHOCK_LEN:] = lo + 0.005 * np.sin(
            2 * np.pi * t[SHOCK_START + SHOCK_LEN:] / 21)
    else:
        rec = np.linspace(lo, -0.05, RECOVER_LEN)
        d[SHOCK_START + SHOCK_LEN:SHOCK_START + SHOCK_LEN + RECOVER_LEN] = rec
        d[SHOCK_START + SHOCK_LEN + RECOVER_LEN:] = base[SHOCK_START + SHOCK_LEN + RECOVER_LEN:]
    return d


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    dates = pd.bdate_range("2018-01-02", periods=N_DAYS)
    frames = []
    specs = {
        "MRV": _discount_path(permanent=False),
        "TRAP": _discount_path(permanent=True),
        "CONST": np.full(N_DAYS, -0.15),
    }
    for tkr, disc in specs.items():
        nav = np.full(N_DAYS, 10.0)
        price = nav * (1.0 + disc)
        dist = np.zeros(N_DAYS)
        if tkr == "MRV":
            dist[::63] = 0.15  # quarterly 15c so income component is exercised
        frames.append(pd.DataFrame({
            "ticker": tkr, "date": dates, "price": price, "nav": nav,
            "volume": 100_000.0, "dist": dist,
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def sig(panel):
    return signals.compute_signals(panel)


def test_no_lookahead(panel):
    """Signals at date t must be identical whether or not future data exists."""
    full = signals.compute_signals(panel)
    cutoff = panel["date"].sort_values().unique()[1000]
    trunc = signals.compute_signals(panel[panel["date"] <= cutoff])
    cols = ["ticker", "date", "z_long", "z_short", "adv_dollar"]
    a = full[full["date"] <= cutoff][cols].reset_index(drop=True)
    b = trunc[cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_constant_discount_has_no_z(sig):
    """'Always at -15%' is not a signal: std floor must yield NaN z."""
    assert sig.loc[sig.ticker == "CONST", "z_long"].isna().all()


def test_z_triggers_on_shock(sig):
    g = sig[sig.ticker == "MRV"].reset_index(drop=True)
    pre = g.loc[SHOCK_START - 10, "z_long"]
    shock = g.loc[SHOCK_START + 5, "z_long"]
    assert abs(pre) < 1.5
    assert shock < -3.0


def test_screener_ranks_shocked_fund_first(panel):
    # Screen as of a date inside the shock window
    dates = panel["date"].sort_values().unique()
    cut = panel[panel["date"] <= dates[SHOCK_START + 10]]
    screened = run_screen(cut, min_adv=250_000.0)
    assert screened.iloc[0]["ticker"] in ("MRV", "TRAP")  # both shocked & cheap
    assert "CONST" not in screened["ticker"].tolist()     # no z -> excluded


def test_mean_reversion_trade_profits_from_discount(sig, panel):
    trades = attach_benchmarks(
        find_trades(sig, threshold=-1.5, horizon=126, cost_bps_oneway=30.0),
        panel, spy=None)
    mrv = trades[trades.ticker == "MRV"]
    assert len(mrv) >= 1
    tr = mrv.iloc[0]
    assert tr.discount_component > 0.04          # thesis P&L: compression
    assert abs(tr.nav_component) < 1e-9          # NAV was flat by design
    assert tr.net_ret > 0.03
    assert tr.exit_reason == "revert"


def test_value_trap_trade_is_flat_or_negative(sig, panel):
    trades = find_trades(sig, threshold=-1.5, horizon=63, cost_bps_oneway=30.0)
    trap = trades[trades.ticker == "TRAP"]
    assert len(trap) >= 1
    tr = trap.iloc[0]
    assert tr.exit_reason == "horizon"           # never reverts
    assert abs(tr.discount_component) < 0.02     # no compression happened
    assert tr.net_ret < 0.01                     # ~zero minus costs


def test_decomposition_identity(sig, panel):
    """(1 + gross - income) == (1 + nav_comp) * (1 + discount_comp) exactly."""
    trades = find_trades(sig, threshold=-1.5, horizon=126)
    assert len(trades) >= 2
    lhs = 1.0 + trades.gross_ret - trades.income_component
    rhs = (1.0 + trades.nav_component) * (1.0 + trades.discount_component)
    assert np.allclose(lhs, rhs, atol=1e-12)


def test_execution_lag(sig):
    """Entry must be at the close AFTER the signal day, never same-day."""
    trades = find_trades(sig, threshold=-1.5, horizon=126)
    g = sig[sig.ticker == "MRV"].reset_index(drop=True)
    for tr in trades[trades.ticker == "MRV"].itertuples():
        sig_dates = g.loc[g.z_long <= -1.5, "date"]
        assert (tr.entry_date > sig_dates.min())  # entered after first signal day


def test_store_roundtrip(tmp_path, panel):
    conn = data_store.connect(tmp_path / "t.db")
    for t in panel.ticker.unique():
        data_store.upsert_fund(conn, t, name=t, category="test", nav_ticker=f"X{t}",
                               status="ok", note=None, updated_at="now")
    n = data_store.upsert_daily(conn, panel)
    assert n == len(panel)
    back = data_store.load_panel(conn)
    assert len(back) == len(panel)
    assert set(back.ticker) == set(panel.ticker)
    # idempotent upsert
    data_store.upsert_daily(conn, panel)
    assert len(data_store.load_panel(conn)) == len(panel)

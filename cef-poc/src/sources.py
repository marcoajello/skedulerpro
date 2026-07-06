"""Data-source adapters for the CEF POC.

Source strategy (each claim below must be re-verified by verify_sources.py
before the pipeline is trusted — run it first):

1. CEFConnect (Nuveen, free, unofficial JSON API)
   - ``/api/v3/DailyPricing`` is the endpoint behind their daily-pricing
     screen and is expected to return the full current CEF universe with
     ticker, name, category, price, NAV, discount, volume.
   - ``/api/v3/pricinghistory/{ticker}/{range}`` backs the fund detail
     charts (price + NAV time series, shallow history).
   - Role here: define the UNIVERSE and cross-check recent NAVs.
   - Unofficial API: shapes can change without notice, hence the runtime
     verification step instead of trusting this docstring.

2. Yahoo Finance via yfinance (free)
   - Market price history: plain ticker (unadjusted close + dividends).
   - NAV history: X-prefixed pseudo-ticker (XADX for ADX). Coverage varies
     per fund; funds whose NAV series is missing or inconsistent are
     dropped and reported, not silently kept.
   - Role here: the deep daily price + NAV history for the backtest.

3. Nasdaq Data Link / CEFData (CEF Advisors) - the purpose-built paid feed
   (150+ fields, daily history to ~2012, pre-computed z-scores). Not used
   in the POC; the free sample is only evaluated for the scale-up note.

Fallback universe: ``SEED_UNIVERSE`` below - a hand-written list of large,
liquid, long-lived CEFs. Every entry is validated empirically at fetch time
(no price or no NAV -> dropped and reported), so a wrong entry degrades to
a dropped row, never to bad data.
"""
from __future__ import annotations

import time

import pandas as pd
import requests

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")

CEFCONNECT_DAILY = "https://www.cefconnect.com/api/v3/DailyPricing"
CEFCONNECT_HISTORY = "https://www.cefconnect.com/api/v3/pricinghistory/{ticker}/{rng}"

# Bootstrap-only list of well-known CEF tickers, used when CEFConnect is
# unreachable. NOT authoritative: written from memory, validated at fetch
# time against real quote data. Prefer the live CEFConnect universe.
SEED_UNIVERSE = [
    # Equity / general
    "ADX", "PEO", "TY", "GAM", "USA", "ASA", "RVT", "RMT", "GDV", "GAB",
    "GGT", "CII", "BDJ", "BOE", "BGY", "NFJ", "SPXX",
    # Sector equity
    "BST", "BSTZ", "BME", "BMEZ", "HQH", "HQL", "THQ", "THW",
    "UTF", "UTG", "RQI", "RNP", "RFI",
    # Eaton Vance / covered call
    "EVT", "ETY", "ETG", "ETO", "EOS", "EOI",
    # Taxable fixed income / multi-asset
    "PDI", "PDO", "PTY", "PCN", "PHK", "PFN", "PFL", "GOF",
    "BCAT", "ECAT", "JPC",
    # Municipal
    "NVG", "NZF", "NEA", "NAD", "NXP",
]


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "application/json",
                      "Referer": "https://www.cefconnect.com/"})
    return s


def fetch_cefconnect_universe(timeout: int = 30) -> pd.DataFrame:
    """Full current CEF universe from CEFConnect. Raises on any failure;
    callers decide whether to fall back to SEED_UNIVERSE."""
    r = _session().get(CEFCONNECT_DAILY, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data)
    if df.empty or "Ticker" not in df.columns:
        raise ValueError(f"Unexpected DailyPricing shape: {list(df.columns)[:20]}")
    return df


def fetch_cefconnect_history(ticker: str, rng: str = "1Y",
                             timeout: int = 30) -> pd.DataFrame:
    """Recent price+NAV history for one fund (used to cross-check Yahoo NAV)."""
    url = CEFCONNECT_HISTORY.format(ticker=ticker, rng=rng)
    r = _session().get(url, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    # Shape observed historically: {"Data": [{"DataDateJs": ..., "Price": ..., "NAV": ...}]}
    rows = payload.get("Data", payload) if isinstance(payload, dict) else payload
    return pd.DataFrame(rows)


def fetch_yf_history(ticker: str, period: str = "max") -> pd.DataFrame | None:
    """Daily unadjusted OHLCV + dividends for a ticker; None if no data.

    auto_adjust=False on purpose: we need the raw close and the explicit
    Dividends column so the backtest can account for distributions itself
    instead of relying on Yahoo's adjustment math.
    """
    import yfinance as yf  # local import: keep module importable offline

    h = yf.Ticker(ticker).history(period=period, auto_adjust=False, actions=True)
    if h is None or h.empty:
        return None
    h = h.copy()
    h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
    return h


def fetch_fund_series(ticker: str, nav_prefix: str = "X",
                      pause: float = 0.5) -> tuple[pd.DataFrame | None, str, str]:
    """Fetch merged price+NAV+dist daily series for one fund from Yahoo.

    Returns (df, status, note); df has columns
    ticker, date, price, nav, volume, dist and is None unless status == 'ok'.
    """
    px = fetch_yf_history(ticker)
    time.sleep(pause)
    if px is None:
        return None, "no_price", "Yahoo returned no price history"
    nav_ticker = f"{nav_prefix}{ticker}"
    nav = fetch_yf_history(nav_ticker)
    time.sleep(pause)
    if nav is None:
        return None, "no_nav", f"Yahoo returned no history for NAV ticker {nav_ticker}"

    df = pd.DataFrame({
        "price": px["Close"],
        "volume": px["Volume"],
        "dist": px.get("Dividends", pd.Series(0.0, index=px.index)),
    })
    df = df.join(nav["Close"].rename("nav"), how="inner")
    df = df.dropna(subset=["price", "nav"])
    if len(df) < 60:
        return None, "bad_overlap", f"only {len(df)} overlapping price/NAV days"

    # Sanity: premium/discount should live in a plausible band. A NAV series
    # that is really some other instrument shows up here immediately.
    prem = df["price"] / df["nav"] - 1.0
    frac_implausible = float((prem.abs() > 0.60).mean())
    if frac_implausible > 0.02:
        return None, "bad_overlap", (
            f"{frac_implausible:.0%} of days have |premium|>60% - "
            f"{nav_ticker} likely not this fund's NAV")

    out = df.reset_index().rename(columns={"index": "date", "Date": "date"})
    out["ticker"] = ticker
    return out[["ticker", "date", "price", "nav", "volume", "dist"]], "ok", nav_ticker

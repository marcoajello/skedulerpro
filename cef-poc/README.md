# CEF Discount-Arbitrage POC

Tests one question before any capital moves: **does buying CEFs whose
discount is unusually wide *versus their own history* (discount z-score)
produce an edge on a 3–6 month horizon, net of costs, out of sample?**
A null result is a valid outcome. No broker integration, no live orders.

## Status (2026-07-06)

**Phase 0 is code-complete but data-blocked in the sandboxed build
environment.** Every external financial-data host is denied by that
environment's egress policy — see
`reports/source_verification_2026-07-06.md`, which is machine-generated
evidence, not an assumption. The math pipeline is fully built and verified
against synthetic fixtures (9 passing tests). **No market data has been
fetched and no findings exist yet.** Nothing in this repo is a backtest
result.

### To unblock (either option)

1. **Run locally** (simplest): clone, `pip install -r requirements.txt`,
   then follow "How to run" below. Any normal internet connection works.
2. **Re-run in Claude Code on the web**: edit the environment's network
   policy (see [docs](https://code.claude.com/docs/en/claude-code-on-the-web))
   to allow these hosts, then re-run the same commands:
   - `www.cefconnect.com` (universe + NAV cross-check)
   - `query1.finance.yahoo.com`, `query2.finance.yahoo.com`, `fc.yahoo.com`
     (yfinance price + NAV history)
   - `data.nasdaq.com` (optional; only for the paid-feed evaluation)

## How to run (in order — each step gates the next)

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # verify the math (synthetic fixtures)
python -m src.verify_sources        # Phase 0 gate: probe real sources, write evidence report
python -m src.build_dataset         # fetch universe + price/NAV -> data/cef.db + quality report
python -m src.screener              # Phase 1: today's cheapest-by-z-score screen
python -m src.backtest              # Phase 2: backtest + decomposition + verdict inputs
```

`verify_sources` exits non-zero and tells you to stop if no price+NAV path
works. `build_dataset` writes `reports/data_quality.md` (coverage, gaps,
stale NAVs, dropped funds) — read it before trusting the backtest.

## Data-source strategy and why

| Source | Role | Cost | Verified? |
|---|---|---|---|
| CEFConnect (unofficial JSON API) | fund universe; recent NAV cross-check | free | blocked in sandbox — endpoints coded, verified at runtime by `verify_sources` |
| Yahoo Finance via yfinance | deep daily history: price (ticker) + NAV (`X`-prefixed pseudo-ticker, e.g. `XADX`) | free | blocked in sandbox — coverage probe built in |
| CEFData / CEF Advisors via Nasdaq Data Link | purpose-built feed, 150+ fields, daily to ~2012, pre-computed z-scores | paid (free sample) | reachability probe only; needs API key |

Chosen POC path: **CEFConnect for the universe + Yahoo X-tickers for
history**, because it's the only free combination that can plausibly give
daily price *and* NAV for 100+ funds over 3+ years. Known risks, handled
in code rather than assumed away:

- **X-ticker coverage is uneven.** Funds with no NAV series are dropped
  and *reported* (`funds.status`), never silently kept.
- **Wrong-instrument NAV series.** Each fund's merged series is sanity
  checked (≥60 overlapping days; >2% of days with |premium|>60% ⇒ reject).
- **Unofficial APIs drift.** `verify_sources` re-checks shapes at runtime
  instead of trusting documentation (or this README).

**Free vs paid:** if the free path's quality report shows poor NAV
coverage (<100 clean funds) or the POC shows an edge worth scaling, the
CEFData/Nasdaq Data Link feed is the upgrade path — it removes the
X-ticker fragility and adds distribution/ROC classification (19a-1-based),
which the free path can only proxy. Evaluate its free sample before
paying; pricing must be checked live (was not reachable from the sandbox).

## What's built

```
src/
  verify_sources.py  Phase 0 gate: probes each source, writes evidence report
  sources.py         adapters: CEFConnect universe/history, Yahoo price+NAV,
                     seed universe fallback (validated at fetch time)
  build_dataset.py   universe -> per-fund price/NAV/dists -> SQLite (incremental)
  data_store.py      SQLite cache (funds, daily, meta)
  quality_report.py  coverage, gaps, stale-NAV runs, dropped funds
  signals.py         discount, 1y/6m trailing z-scores (trailing-only), ADV
  screener.py        Phase 1: ranked screen + ROC-suspect flag
  backtest.py        Phase 2: event-driven trades, next-day fills, costs,
                     return decomposition (discount / NAV / income),
                     EW-universe + SPY benchmarks per trade, in/out-of-sample
                     split, threshold x horizon sensitivity grid, charts
tests/test_math.py   synthetic-fixture proofs: no-lookahead, decomposition
                     identity, value-trap flatness, std-floor ("always cheap
                     isn't cheap"), execution lag, store roundtrip
```

### Honesty guarantees baked into the design

- z-scores use trailing windows only; a test asserts signals are identical
  when future data is truncated (no lookahead).
- Fills happen at the close *after* the signal day.
- A fund whose discount never varies (std floor) produces no signal — a
  permanent -15% discount is not "cheap".
- Per-trade decomposition separates thesis P&L (discount change) from NAV
  beta and income, so "the discount closed because NAV fell" is visible.
- The sensitivity grid (thresholds × horizons) is always printed, so a
  result that only works at one magic parameter self-identifies.
- Trades still open at data end are dropped and counted, not marked-to-hope.

## Phase plan

- [x] Phase 0a — pipeline + verification tooling (this commit)
- [ ] Phase 0b — run `verify_sources` + `build_dataset` with network access;
      review `data_quality.md`; go/no-go on data
- [ ] Phase 1 — screener run on real data
- [ ] Phase 2 — backtest; written verdict incl. the null case
- [ ] Phase 3 — forward paper-signal tracker (only if Phase 2 shows an edge)

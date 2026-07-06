"""Phase 0: build the local cached dataset (universe -> price+NAV -> SQLite).

Usage:
    python -m src.build_dataset [--max-funds N] [--seed-only] [--pause 0.5]

Flow:
1. Universe: CEFConnect DailyPricing (full CEF list). Falls back to the
   hand-written SEED_UNIVERSE (clearly reported) if unreachable.
2. Per fund: Yahoo price history + X-prefixed NAV history, merged, sanity
   checked (overlap length, plausible premium band). Failures are recorded
   in the funds table with a status, never silently dropped.
3. Everything lands in data/cef.db; re-runs are incremental upserts.
4. Finishes by writing the data-quality report (quality_report.py).
"""
from __future__ import annotations

import argparse
import datetime as dt

from . import data_store, quality_report, sources


def get_universe(seed_only: bool) -> tuple[list[dict], str]:
    if not seed_only:
        try:
            df = sources.fetch_cefconnect_universe()
            cols = {c.lower(): c for c in df.columns}
            tick = cols.get("ticker")
            name = cols.get("name") or cols.get("fundname")
            cat = cols.get("category") or cols.get("categoryname")
            funds = [{"ticker": str(r[tick]).strip().upper(),
                      "name": r.get(name) if name else None,
                      "category": r.get(cat) if cat else None}
                     for _, r in df.iterrows()]
            return funds, "cefconnect"
        except Exception as e:  # noqa: BLE001
            print(f"[universe] CEFConnect unavailable ({type(e).__name__}: {e}); "
                  f"falling back to seed list")
    return [{"ticker": t, "name": None, "category": None}
            for t in sources.SEED_UNIVERSE], "seed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-funds", type=int, default=250)
    ap.add_argument("--seed-only", action="store_true",
                    help="skip CEFConnect, use the built-in seed universe")
    ap.add_argument("--pause", type=float, default=0.5,
                    help="seconds between Yahoo requests (be polite)")
    ap.add_argument("--benchmark", default="SPY",
                    help="also fetch this benchmark ticker (price only)")
    args = ap.parse_args()

    conn = data_store.connect()
    universe, src = get_universe(args.seed_only)
    universe = universe[: args.max_funds]
    print(f"[universe] {len(universe)} funds from source={src}")
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('universe_source', ?)", (src,))

    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    n_ok = 0
    for i, f in enumerate(universe, 1):
        t = f["ticker"]
        try:
            df, status, note = sources.fetch_fund_series(t, pause=args.pause)
        except Exception as e:  # noqa: BLE001
            df, status, note = None, "error", f"{type(e).__name__}: {e}"
        data_store.upsert_fund(conn, t, name=f.get("name"), category=f.get("category"),
                               nav_ticker=note if status == "ok" else None,
                               status=status, note=None if status == "ok" else note,
                               updated_at=now)
        if status == "ok" and df is not None:
            data_store.upsert_daily(conn, df)
            n_ok += 1
        print(f"[{i}/{len(universe)}] {t}: {status}"
              + (f" ({len(df)} days)" if df is not None else f" - {note}"))

    # Benchmark (price + dividends only; stored with nav = NULL)
    if args.benchmark:
        px = sources.fetch_yf_history(args.benchmark)
        if px is not None:
            bdf = px.reset_index().rename(columns={px.index.name or "index": "date"})
            bdf["ticker"] = args.benchmark
            bdf["price"], bdf["nav"] = bdf["Close"], None
            bdf["volume"], bdf["dist"] = bdf["Volume"], bdf.get("Dividends", 0.0)
            data_store.upsert_daily(conn, bdf[["ticker", "date", "price", "nav", "volume", "dist"]])
            data_store.upsert_fund(conn, args.benchmark, name="benchmark",
                                   category="benchmark", nav_ticker=None,
                                   status="benchmark", note=None, updated_at=now)
            print(f"[benchmark] {args.benchmark}: {len(bdf)} days")

    print(f"\n[done] {n_ok}/{len(universe)} funds stored ok -> {data_store.DB_PATH}")
    quality_report.main()
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

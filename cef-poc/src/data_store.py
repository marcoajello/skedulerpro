"""SQLite-backed local cache for CEF price/NAV/distribution history.

Why SQLite over parquet: the dataset is small (a few hundred funds x a few
thousand days), we want incremental upserts as funds are fetched one at a
time, and a single-file DB with a primary key gives us dedup for free.
Parquet would win for columnar analytics at much larger scale.
"""
from __future__ import annotations

import pathlib
import sqlite3

import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "cef.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS funds (
    ticker      TEXT PRIMARY KEY,
    name        TEXT,
    category    TEXT,
    nav_ticker  TEXT,          -- e.g. XADX for ADX; NULL if no NAV series found
    status      TEXT,          -- ok | no_price | no_nav | bad_overlap
    note        TEXT,
    updated_at  TEXT
);
CREATE TABLE IF NOT EXISTS daily (
    ticker  TEXT NOT NULL,
    date    TEXT NOT NULL,     -- YYYY-MM-DD
    price   REAL,              -- unadjusted market close
    nav     REAL,              -- NAV per share (from NAV pseudo-ticker close)
    volume  REAL,              -- share volume
    dist    REAL,              -- cash distribution ex- this date (per share)
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def connect(db_path: pathlib.Path | str = DB_PATH) -> sqlite3.Connection:
    db_path = pathlib.Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def upsert_fund(conn: sqlite3.Connection, ticker: str, **fields) -> None:
    cols = ["name", "category", "nav_ticker", "status", "note", "updated_at"]
    vals = [fields.get(c) for c in cols]
    conn.execute(
        f"INSERT INTO funds (ticker, {', '.join(cols)}) VALUES (?,?,?,?,?,?,?) "
        f"ON CONFLICT(ticker) DO UPDATE SET "
        + ", ".join(f"{c}=excluded.{c}" for c in cols),
        [ticker] + vals,
    )
    conn.commit()


def upsert_daily(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """df columns: ticker, date (datetime or str), price, nav, volume, dist."""
    if df.empty:
        return 0
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    rows = df[["ticker", "date", "price", "nav", "volume", "dist"]].itertuples(index=False)
    conn.executemany(
        "INSERT INTO daily (ticker, date, price, nav, volume, dist) "
        "VALUES (?,?,?,?,?,?) "
        "ON CONFLICT(ticker, date) DO UPDATE SET "
        "price=excluded.price, nav=excluded.nav, "
        "volume=excluded.volume, dist=excluded.dist",
        [tuple(None if pd.isna(v) else v for v in r) for r in rows],
    )
    conn.commit()
    return len(df)


def load_panel(conn: sqlite3.Connection, tickers: list[str] | None = None,
               ok_only: bool = True) -> pd.DataFrame:
    """Long-format panel: ticker, date, price, nav, volume, dist."""
    q = "SELECT d.ticker, d.date, d.price, d.nav, d.volume, d.dist FROM daily d"
    params: list = []
    if ok_only:
        q += " JOIN funds f ON f.ticker = d.ticker AND f.status = 'ok'"
    if tickers:
        q += f" WHERE d.ticker IN ({','.join('?' * len(tickers))})"
        params = list(tickers)
    df = pd.read_sql_query(q, conn, params=params or None)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_funds(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM funds", conn)

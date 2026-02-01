import sqlite3
import pandas as pd

def connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def init_db(con: sqlite3.Connection) -> None:
    con.execute("""
    CREATE TABLE IF NOT EXISTS candles (
        pair TEXT NOT NULL,
        ts TEXT NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume REAL,
        PRIMARY KEY(pair, ts)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pair TEXT NOT NULL,
        ts TEXT NOT NULL,
        p_up REAL NOT NULL,
        model_version TEXT NOT NULL,
        UNIQUE(pair, ts)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pair TEXT NOT NULL,
        entry_ts TEXT NOT NULL,
        side TEXT NOT NULL,
        entry_price REAL NOT NULL,
        size_units REAL NOT NULL,
        sl_price REAL NOT NULL,
        tp_price REAL NOT NULL,
        exit_ts TEXT,
        exit_price REAL,
        pnl REAL,
        reason TEXT
    );
    """)
    con.commit()

def upsert_candles(con: sqlite3.Connection, pair: str, df: pd.DataFrame) -> None:
    rows = []
    for ts, r in df.iterrows():
        rows.append((
            pair, str(ts),
            float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]),
            None if pd.isna(r.get("volume")) else float(r.get("volume"))
        ))

    con.executemany("""
        INSERT INTO candles(pair, ts, open, high, low, close, volume)
        VALUES(?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(pair, ts) DO UPDATE SET
          open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close, volume=excluded.volume;
    """, rows)
    con.commit()

def load_candles(con: sqlite3.Connection, pair: str, limit: int = 8000) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT ts, open, high, low, close, volume FROM candles WHERE pair=? ORDER BY ts ASC",
        con, params=(pair,)
    )
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts")
    if len(df) > limit:
        df = df.iloc[-limit:]
    return df

def save_signal(con: sqlite3.Connection, pair: str, ts: str, p_up: float, model_version: str) -> None:
    con.execute("""
        INSERT INTO signals(pair, ts, p_up, model_version)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(pair, ts) DO UPDATE SET p_up=excluded.p_up, model_version=excluded.model_version;
    """, (pair, ts, float(p_up), model_version))
    con.commit()

def insert_trade_open(con: sqlite3.Connection, pair: str, entry_ts: str, side: str,
                      entry_price: float, size_units: float, sl_price: float, tp_price: float) -> int:
    cur = con.cursor()
    cur.execute("""
        INSERT INTO trades(pair, entry_ts, side, entry_price, size_units, sl_price, tp_price)
        VALUES(?, ?, ?, ?, ?, ?, ?)
    """, (pair, entry_ts, side, float(entry_price), float(size_units), float(sl_price), float(tp_price)))
    con.commit()
    return int(cur.lastrowid)

def close_trade(con: sqlite3.Connection, trade_id: int, exit_ts: str, exit_price: float, pnl: float, reason: str) -> None:
    con.execute("""
        UPDATE trades
        SET exit_ts=?, exit_price=?, pnl=?, reason=?
        WHERE id=?
    """, (exit_ts, float(exit_price), float(pnl), reason, int(trade_id)))
    con.commit()

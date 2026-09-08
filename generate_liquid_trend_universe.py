import pandas as pd
import numpy as np
from datetime import datetime
import os

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

PARQUET_PATH = "nse_6yr_historical.parquet"
OUTPUT_CSV = "liquid_trend_universe.csv"
OUTPUT_TV_TXT = "tradingview_liquid_trend_universe.txt"

MIN_HISTORY_DAYS = 200
MEDIAN_TURNOVER_WINDOW = 50
MEDIAN_TURNOVER_MIN = 15_00_00_000  # 15 crore

EMA_SHORT = 20
EMA_MID = 50
EMA_LONG = 200

# Adjust these patterns if your symbol naming differs
# Assumptions:
# - Mainboard equities: simple symbols like "RELIANCE", "INFY", no special suffixes
# - Exclude: indices, ETFs, SME, BE/BZ series, etc.
EXCLUDE_PATTERNS = [
    "NIFTY", "SENSEX", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY",
    "ETF", "ETN", "INDEX", "ICE", "BSE", "NSE",
    "-EQ", "-BE", "-BZ", "-SM", "-ST", "-GR", "-IP"
]

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def is_mainboard_equity(symbol: str) -> bool:
    s = str(symbol).upper()
    # Basic heuristic: simple alphanumeric, no dashes, no known index/ETF keywords
    if any(p in s for p in EXCLUDE_PATTERNS):
        return False
    if "-" in s or "_" in s:
        return False
    # You can refine this further if you maintain a separate master list
    return True

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # df must have columns: ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
    df = df.sort_values(["symbol", "date"]).copy()

    # Turnover
    df["turnover"] = df["close"] * df["volume"]

    # Group by symbol
    out_list = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < MIN_HISTORY_DAYS:
            continue
        if not is_mainboard_equity(sym):
            continue

        # EMAs
        g["ema20"] = g["close"].ewm(span=EMA_SHORT, adjust=False).mean()
        g["ema50"] = g["close"].ewm(span=EMA_MID, adjust=False).mean()
        g["ema200"] = g["close"].ewm(span=EMA_LONG, adjust=False).mean()

        # Median turnover over last 50 days (rolling median)
        g["median_turnover_50d"] = (
            g["turnover"]
            .rolling(window=MEDIAN_TURNOVER_WINDOW, min_periods=MEDIAN_TURNOVER_WINDOW)
            .median()
        )

        out_list.append(g)

    if not out_list:
        return pd.DataFrame()

    out = pd.concat(out_list, ignore_index=True)
    return out

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Latest date per symbol
    last = df.groupby("symbol", sort=False).apply(
        lambda x: x.loc[x["date"].idxmax()]
    ).reset_index(drop=True)

    f = last.copy()

    # Filter conditions
    f = f[f["median_turnover_50d"] >= MEDIAN_TURNOVER_MIN]
    f = f[f["close"] > f["ema50"]]
    f = f[f["close"] > f["ema200"]]
    f = f[f["ema50"] > f["ema200"]]
    f = f[f["ema20"] > f["ema50"]]

    # Diagnostic columns
    f["close_vs_ema50_pct"] = ((f["close"] - f["ema50"]) / f["ema50"]) * 100
    f["close_vs_ema200_pct"] = ((f["close"] - f["ema200"]) / f["ema200"]) * 100
    f["median_turnover_50d_cr"] = f["median_turnover_50d"] / 1_00_00_000
    f["trend_stack"] = "Close > EMA20 > EMA50 > EMA200"

    # Select final columns
    out_cols = [
        "symbol",
        "date",
        "close",
        "median_turnover_50d_cr",
        "ema20",
        "ema50",
        "ema200",
        "close_vs_ema50_pct",
        "close_vs_ema200_pct",
        "trend_stack"
    ]
    f = f[out_cols].rename(columns={"date": "as_of_date"})
    f = f.sort_values(["symbol"]).reset_index(drop=True)
    return f

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("Loading parquet...")
    df = pd.read_parquet(PARQUET_PATH)

    # Ensure expected columns exist
    required_cols = {"symbol", "date", "open", "high", "low", "close", "volume"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns in parquet: {missing}")

    # Convert date to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    print("Computing indicators...")
    ind = compute_indicators(df)
    if ind.empty:
        print("No symbols passed basic history / equity filter.")
        return

    print("Applying filters...")
    filtered = apply_filters(ind)

    if filtered.empty:
        print("No symbols passed all filters.")
        # Still write empty outputs
        filtered.to_csv(OUTPUT_CSV, index=False)
        with open(OUTPUT_TV_TXT, "w") as f:
            pass
        return

    # Write full CSV
    filtered.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(filtered)} symbols to {OUTPUT_CSV}")

    # Write TradingView symbol list (one symbol per line)
    symbols = filtered["symbol"].dropna().unique().tolist()
    with open(OUTPUT_TV_TXT, "w") as f:
        for s in symbols:
            f.write(str(s) + "\n")
    print(f"Wrote {len(symbols)} symbols to {OUTPUT_TV_TXT}")

if __name__ == "__main__":
    main()

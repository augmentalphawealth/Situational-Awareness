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

# Exclude patterns to approximate "NSE mainboard equity only"
EXCLUDE_SYMBOL_PATTERNS = [
    "NIFTY", "SENSEX", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY",
    "ETF", "ETN", "INDEX", "ICE", "BSE", "NSE",
]

EXCLUDE_ISIN_PREFIX = [
    # Add prefixes here if you know specific non-equity ISIN patterns
]

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def is_mainboard_equity(row: pd.Series) -> bool:
    symbol = str(row.get("Symbol", "")).upper()
    company_name = str(row.get("Company_Name", "")).upper()
    isin = str(row.get("ISIN", "")).upper()

    if any(p in symbol for p in EXCLUDE_SYMBOL_PATTERNS):
        return False
    if any(p in company_name for p in EXCLUDE_SYMBOL_PATTERNS):
        return False

    if any(isin.startswith(p) for p in EXCLUDE_ISIN_PREFIX):
        return False

    if "-" in symbol or "_" in symbol:
        return False

    return True

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Columns expected:
    # ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol',
    #  'Instrument_Token', 'ISIN', 'Company_ID', 'Company_Name',
    #  'Adjustment_Source', 'Turnover']

    df = df.sort_values(["Symbol", "Date"]).copy()

    if "Turnover" not in df.columns:
        df["Turnover"] = df["Close"] * df["Volume"]

    out_list = []
    for sym, g in df.groupby("Symbol", sort=False):
        g = g.sort_values("Date").reset_index(drop=True)
        if len(g) < MIN_HISTORY_DAYS:
            continue

        if not is_mainboard_equity(g.iloc[0]):
            continue

        g["ema20"] = g["Close"].ewm(span=EMA_SHORT, adjust=False).mean()
        g["ema50"] = g["Close"].ewm(span=EMA_MID, adjust=False).mean()
        g["ema200"] = g["Close"].ewm(span=EMA_LONG, adjust=False).mean()

        g["median_turnover_50d"] = (
            g["Turnover"]
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

    # Ensure 'Symbol' column exists; if groupby renamed it, fix it
    if "Symbol" not in df.columns and "symbol" in df.columns:
        df = df.rename(columns={"symbol": "Symbol"})

    # Latest date per symbol
    # Use sort + drop_duplicates instead of groupby().apply() to avoid index issues
    df_sorted = df.sort_values("Date", ascending=False)
    last = df_sorted.drop_duplicates(subset=["Symbol"], keep="first").copy()

    f = last.copy()

    # Filter conditions
    f = f[f["median_turnover_50d"] >= MEDIAN_TURNOVER_MIN]
    f = f[f["Close"] > f["ema50"]]
    f = f[f["Close"] > f["ema200"]]
    f = f[f["ema50"] > f["ema200"]]
    f = f[f["ema20"] > f["ema50"]]

    # Diagnostic columns
    f["close_vs_ema50_pct"] = ((f["Close"] - f["ema50"]) / f["ema50"]) * 100
    f["close_vs_ema200_pct"] = ((f["Close"] - f["ema200"]) / f["ema200"]) * 100
    f["median_turnover_50d_cr"] = f["median_turnover_50d"] / 1_00_00_000
    f["trend_stack"] = "Close > EMA20 > EMA50 > EMA200"

    # Ensure Symbol column name is correct before selecting out_cols
    if "Symbol" not in f.columns:
        # Fallback: use the first column as symbol if only one left
        possible_sym_cols = [c for c in f.columns if "symbol" in c.lower() or c.lower() == "symbol"]
        if possible_sym_cols:
            f = f.rename(columns={possible_sym_cols[0]: "Symbol"})

    out_cols = [
        "Symbol",
        "Date",
        "Close",
        "median_turnover_50d_cr",
        "ema20",
        "ema50",
        "ema200",
        "close_vs_ema50_pct",
        "close_vs_ema200_pct",
        "trend_stack",
        "Company_Name",
        "ISIN",
    ]
    # Only keep columns that exist
    out_cols = [c for c in out_cols if c in f.columns]

    f = f[out_cols].rename(
        columns={
            "Symbol": "symbol",
            "Date": "as_of_date",
            "Close": "close",
            "Company_Name": "company_name",
            "ISIN": "isin",
        }
    )
    f = f.sort_values(["symbol"]).reset_index(drop=True)
    return f

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("Loading parquet...")
    df = pd.read_parquet(PARQUET_PATH)

    required_cols = {
        "Date", "Open", "High", "Low", "Close", "Volume", "Symbol", "Turnover"
    }
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns in parquet: {missing}")

    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])

    print("Computing indicators...")
    ind = compute_indicators(df)
    if ind.empty:
        print("No symbols passed basic history / equity filter.")
        pd.DataFrame().to_csv(OUTPUT_CSV, index=False)
        with open(OUTPUT_TV_TXT, "w") as f:
            pass
        return

    print("Applying filters...")
    filtered = apply_filters(ind)

    if filtered.empty:
        print("No symbols passed all filters.")
        filtered.to_csv(OUTPUT_CSV, index=False)
        with open(OUTPUT_TV_TXT, "w") as f:
            pass
        return

    filtered.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(filtered)} symbols to {OUTPUT_CSV}")

    symbols = filtered["symbol"].dropna().unique().tolist()
    with open(OUTPUT_TV_TXT, "w") as f:
        for s in symbols:
            f.write(str(s) + "\n")
    print(f"Wrote {len(symbols)} symbols to {OUTPUT_TV_TXT}")

if __name__ == "__main__":
    main()

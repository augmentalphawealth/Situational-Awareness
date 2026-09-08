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
# Adjust if you see unwanted symbols getting included/excluded.
EXCLUDE_SYMBOL_PATTERNS = [
    "NIFTY", "SENSEX", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY",
    "ETF", "ETN", "INDEX", "ICE", "BSE", "NSE",
]

EXCLUDE_ISIN_PREFIX = [
    "INE999",  # example placeholder; extend if you know specific non-equity prefixes
]

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def is_mainboard_equity(row: pd.Series) -> bool:
    symbol = str(row.get("Symbol", "")).upper()
    company_name = str(row.get("Company_Name", "")).upper()
    isin = str(row.get("ISIN", "")).upper()

    # Exclude obvious non-equities / indices / ETFs by symbol
    if any(p in symbol for p in EXCLUDE_SYMBOL_PATTERNS):
        return False
    if any(p in company_name for p in EXCLUDE_SYMBOL_PATTERNS):
        return False

    # Exclude by ISIN prefix if configured
    if any(isin.startswith(p) for p in EXCLUDE_ISIN_PREFIX):
        return False

    # Basic sanity: symbol should be alphanumeric, no dashes
    if "-" in symbol or "_" in symbol:
        return False

    return True

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Expected columns (based on your diagnostic):
    # ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol',
    #  'Instrument_Token', 'ISIN', 'Company_ID', 'Company_Name',
    #  'Adjustment_Source', 'Turnover']

    df = df.sort_values(["Symbol", "Date"]).copy()

    # Use existing Turnover column if available, else compute
    if "Turnover" not in df.columns:
        df["Turnover"] = df["Close"] * df["Volume"]

    out_list = []
    for sym, g in df.groupby("Symbol", sort=False):
        g = g.sort_values("Date").reset_index(drop=True)
        if len(g) < MIN_HISTORY_DAYS:
            continue

        # Use first row to decide if this is a mainboard equity
        if not is_mainboard_equity(g.iloc[0]):
            continue

        # EMAs on Close
        g["ema20"] = g["Close"].ewm(span=EMA_SHORT, adjust=False).mean()
        g["ema50"] = g["Close"].ewm(span=EMA_MID, adjust=False).mean()
        g["ema200"] = g["Close"].ewm(span=EMA_LONG, adjust=False).mean()

        # Median turnover over last 50 days (rolling median)
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

    # Latest date per symbol
    last = df.groupby("Symbol", sort=False).apply(
        lambda x: x.loc[x["Date"].idxmax()]
    ).reset_index(drop=True)

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

    # Select final columns
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

    # Ensure expected columns exist (based on your diagnostic output)
    required_cols = {
        "Date", "Open", "High", "Low", "Close", "Volume", "Symbol", "Turnover"
    }
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns in parquet: {missing}")

    # Convert Date to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])

    print("Computing indicators...")
    ind = compute_indicators(df)
    if ind.empty:
        print("No symbols passed basic history / equity filter.")
        # Still write empty outputs
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

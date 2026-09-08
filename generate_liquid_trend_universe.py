from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PARQUET = Path("nse_6yr_historical.parquet")

OUTPUT_CSV = Path("liquid_trend_universe.csv")
OUTPUT_TRADINGVIEW_TXT = Path("tradingview_liquid_trend_universe.txt")

MIN_HISTORY_DAYS = 200

EMA_20_PERIOD = 20
EMA_50_PERIOD = 50
EMA_200_PERIOD = 200

MEDIAN_TURNOVER_WINDOW = 50
MIN_MEDIAN_TURNOVER_RS = 15_00_00_000  # Rs 15 crore


# ============================================================
# OUTPUT HELPERS
# ============================================================

OUTPUT_COLUMNS = [
    "symbol",
    "company_name",
    "isin",
    "as_of_date",
    "close",
    "median_turnover_50d_cr",
    "ema20",
    "ema50",
    "ema200",
    "close_vs_ema20_pct",
    "close_vs_ema50_pct",
    "close_vs_ema200_pct",
    "trend_stack",
]


def write_empty_outputs() -> None:
    pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(OUTPUT_CSV, index=False)
    OUTPUT_TRADINGVIEW_TXT.write_text("", encoding="utf-8")


def print_stage(label: str, df: pd.DataFrame) -> None:
    print(f"{label}: {len(df):,} symbols")


# ============================================================
# DATA PREPARATION
# ============================================================

def load_and_prepare_data() -> pd.DataFrame:
    print("Loading parquet...")
    df = pd.read_parquet(INPUT_PARQUET).copy()

    required_columns = {
        "Date",
        "Close",
        "Volume",
        "Symbol",
        "Turnover",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df["Turnover"] = pd.to_numeric(df["Turnover"], errors="coerce")

    # If Turnover is absent/invalid on any row, calculate it from Close x Volume.
    df["Turnover"] = df["Turnover"].where(
        df["Turnover"].notna() & (df["Turnover"] > 0),
        df["Close"] * df["Volume"],
    )

    # This script assumes the Parquet itself is your NSE mainboard EQ universe.
    # Therefore, no custom ETF / SME / index / symbol-name exclusions are applied.
    df = df.dropna(subset=["Date", "Symbol", "Close", "Volume", "Turnover"]).copy()

    df = df[
        (df["Symbol"] != "")
        & (df["Symbol"] != "NAN")
        & (df["Close"] > 0)
        & (df["Volume"] >= 0)
        & (df["Turnover"] > 0)
    ].copy()

    # Avoid duplicate daily rows for the same symbol.
    df = (
        df.sort_values(["Symbol", "Date"])
        .drop_duplicates(subset=["Symbol", "Date"], keep="last")
        .copy()
    )

    print(f"Usable OHLCV rows: {len(df):,}")
    print(f"Symbols in Parquet universe: {df['Symbol'].nunique():,}")

    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Symbol", "Date"]).copy()

    history_count = df.groupby("Symbol")["Date"].transform("count")
    df = df[history_count >= MIN_HISTORY_DAYS].copy()

    print(f"Symbols with at least {MIN_HISTORY_DAYS} trading days: {df['Symbol'].nunique():,}")

    if df.empty:
        return df

    print("Calculating EMA20, EMA50, EMA200 and 50-day median turnover...")

    df["ema20"] = (
        df.groupby("Symbol")["Close"]
        .transform(
            lambda series: series.ewm(
                span=EMA_20_PERIOD,
                adjust=False,
                min_periods=EMA_20_PERIOD,
            ).mean()
        )
    )

    df["ema50"] = (
        df.groupby("Symbol")["Close"]
        .transform(
            lambda series: series.ewm(
                span=EMA_50_PERIOD,
                adjust=False,
                min_periods=EMA_50_PERIOD,
            ).mean()
        )
    )

    df["ema200"] = (
        df.groupby("Symbol")["Close"]
        .transform(
            lambda series: series.ewm(
                span=EMA_200_PERIOD,
                adjust=False,
                min_periods=EMA_200_PERIOD,
            ).mean()
        )
    )

    df["median_turnover_50d"] = (
        df.groupby("Symbol")["Turnover"]
        .transform(
            lambda series: series.rolling(
                window=MEDIAN_TURNOVER_WINDOW,
                min_periods=MEDIAN_TURNOVER_WINDOW,
            ).median()
        )
    )

    return df


# ============================================================
# FILTERING
# ============================================================

def get_common_latest_date_data(df: pd.DataFrame) -> pd.DataFrame:
    latest_per_symbol = (
        df.sort_values(["Symbol", "Date"])
        .drop_duplicates(subset=["Symbol"], keep="last")
        .copy()
    )

    print_stage("Symbols with a latest row", latest_per_symbol)

    common_latest_date = latest_per_symbol["Date"].max()

    # Important: only use symbols updated on the same final EOD date.
    # This prevents old / stale symbols from passing based on past prices.
    latest = latest_per_symbol[
        latest_per_symbol["Date"] == common_latest_date
    ].copy()

    print(f"Common latest EOD date: {common_latest_date.date()}")
    print_stage("Symbols available on the common latest date", latest)

    return latest


def apply_screen(latest: pd.DataFrame) -> pd.DataFrame:
    required_indicator_columns = [
        "ema20",
        "ema50",
        "ema200",
        "median_turnover_50d",
    ]

    stage = latest.dropna(subset=required_indicator_columns).copy()
    print_stage("Symbols ready with all indicators", stage)

    stage = stage[
        stage["median_turnover_50d"] >= MIN_MEDIAN_TURNOVER_RS
    ].copy()
    print_stage("After 50-day median turnover >= Rs 15 crore", stage)

    stage = stage[stage["Close"] > stage["ema50"]].copy()
    print_stage("After Close > EMA50", stage)

    stage = stage[stage["Close"] > stage["ema200"]].copy()
    print_stage("After Close > EMA200", stage)

    stage = stage[stage["ema50"] > stage["ema200"]].copy()
    print_stage("After EMA50 > EMA200", stage)

    stage = stage[stage["ema20"] > stage["ema50"]].copy()
    print_stage("After EMA20 > EMA50 (FINAL)", stage)

    return stage


def build_output(screened: pd.DataFrame) -> pd.DataFrame:
    if screened.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    result = screened.copy()

    result["median_turnover_50d_cr"] = (
        result["median_turnover_50d"] / 1_00_00_000
    )

    result["close_vs_ema20_pct"] = (
        (result["Close"] / result["ema20"] - 1) * 100
    )

    result["close_vs_ema50_pct"] = (
        (result["Close"] / result["ema50"] - 1) * 100
    )

    result["close_vs_ema200_pct"] = (
        (result["Close"] / result["ema200"] - 1) * 100
    )

    result["trend_stack"] = "Close > EMA20 > EMA50 > EMA200"

    optional_columns = {
        "Company_Name": "company_name",
        "ISIN": "isin",
    }

    for source_column in optional_columns:
        if source_column not in result.columns:
            result[source_column] = ""

    result = result.rename(
        columns={
            "Symbol": "symbol",
            "Company_Name": "company_name",
            "ISIN": "isin",
            "Date": "as_of_date",
            "Close": "close",
        }
    )

    result = result[OUTPUT_COLUMNS].copy()

    result = result.sort_values(
        by=["median_turnover_50d_cr", "close_vs_ema50_pct"],
        ascending=[False, True],
    ).reset_index(drop=True)

    result = result.round(
        {
            "close": 2,
            "median_turnover_50d_cr": 2,
            "ema20": 2,
            "ema50": 2,
            "ema200": 2,
            "close_vs_ema20_pct": 2,
            "close_vs_ema50_pct": 2,
            "close_vs_ema200_pct": 2,
        }
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    df = load_and_prepare_data()

    if df.empty:
        print("No usable data is available. Writing empty output files.")
        write_empty_outputs()
        return

    df = add_indicators(df)

    if df.empty:
        print("No symbols have at least 200 trading days. Writing empty output files.")
        write_empty_outputs()
        return

    latest = get_common_latest_date_data(df)

    if latest.empty:
        print("No symbols are available on the common latest EOD date.")
        write_empty_outputs()
        return

    print("\nApplying liquid-trend universe filters...")
    screened = apply_screen(latest)

    result = build_output(screened)

    result.to_csv(OUTPUT_CSV, index=False)

    # Prefix every symbol for unambiguous TradingView NSE import.
    tradingview_symbols = [
        f"NSE:{symbol}" for symbol in result["symbol"].dropna().unique()
    ]

    OUTPUT_TRADINGVIEW_TXT.write_text(
        "\n".join(tradingview_symbols) + ("\n" if tradingview_symbols else ""),
        encoding="utf-8",
    )

    print(f"\nFinal universe count: {len(result):,}")
    print(f"CSV output: {OUTPUT_CSV}")
    print(f"TradingView list output: {OUTPUT_TRADINGVIEW_TXT}")


if __name__ == "__main__":
    main()

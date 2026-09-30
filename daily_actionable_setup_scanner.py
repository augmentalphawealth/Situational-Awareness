from pathlib import Path

import pandas as pd

PARQUET_FILE = Path("nse_6yr_historical.parquet")
OUTPUT_FILE = Path("daily_actionable_setups.csv")
TRADINGVIEW_OUTPUT_FILE = Path("daily_actionable_tradingview_watchlist.txt")

SIX_MONTH_SESSIONS = 126
MIN_SIX_MONTH_RETURN_PCT = 30.0
MIN_SPIKE_DAYS = 3
SPIKE_MULTIPLE = 3.0
MAX_LAST_RANGE_PCT = 4.0
MAX_EMA20_DISTANCE_PCT = 7.0
TURNOVER_WINDOW = 50
MIN_MEDIAN_TURNOVER_RS = 5_00_00_000  # Rs 5 crore

REQUIRED_COLUMNS = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]


def score_higher_better(series: pd.Series) -> pd.Series:
    n = len(series)
    if n <= 1:
        return pd.Series(100.0, index=series.index)
    r = series.rank(method="average", ascending=True)
    return ((r - 1.0) / (n - 1.0) * 100.0).clip(0, 100)


def score_lower_better(series: pd.Series) -> pd.Series:
    n = len(series)
    if n <= 1:
        return pd.Series(100.0, index=series.index)
    r = series.rank(method="average", ascending=True)
    return (100.0 - ((r - 1.0) / (n - 1.0) * 100.0)).clip(0, 100)


def write_tradingview_watchlist(symbols) -> None:
    tradingview_symbols = [f"NSE:{symbol}" for symbol in symbols]
    TRADINGVIEW_OUTPUT_FILE.write_text(
        "\n".join(tradingview_symbols) + ("\n" if tradingview_symbols else ""),
        encoding="utf-8",
    )


def main() -> None:
    if not PARQUET_FILE.exists():
        raise SystemExit(f"Missing {PARQUET_FILE}")

    raw = pd.read_parquet(PARQUET_FILE)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    selected_columns = REQUIRED_COLUMNS + (["Turnover"] if "Turnover" in raw.columns else [])
    df = raw[selected_columns].copy()

    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df[
        (df["Symbol"] != "")
        & (df["Close"] > 0)
        & (df["Low"] > 0)
        & (df["Volume"] >= 0)
    ].copy()
    df = (
        df.sort_values(["Symbol", "Date"])
        .drop_duplicates(["Symbol", "Date"], keep="last")
        .copy()
    )

    calculated_turnover = df["Close"] * df["Volume"]
    if "Turnover" in df.columns:
        df["Turnover"] = pd.to_numeric(df["Turnover"], errors="coerce")
        df["Daily_Turnover"] = df["Turnover"].where(
            df["Turnover"].notna() & (df["Turnover"] > 0),
            calculated_turnover,
        )
    else:
        df["Daily_Turnover"] = calculated_turnover

    group = df.groupby("Symbol", group_keys=False)

    df["Median_Turnover_50D"] = group["Daily_Turnover"].transform(
        lambda s: s.rolling(TURNOVER_WINDOW, min_periods=TURNOVER_WINDOW).median()
    )

    df["EMA20"] = group["Close"].transform(
        lambda s: s.ewm(span=20, adjust=False, min_periods=20).mean()
    )
    df["EMA50"] = group["Close"].transform(
        lambda s: s.ewm(span=50, adjust=False, min_periods=50).mean()
    )
    df["EMA200"] = group["Close"].transform(
        lambda s: s.ewm(span=200, adjust=False, min_periods=200).mean()
    )

    df["Close_6M_Ago"] = group["Close"].shift(SIX_MONTH_SESSIONS)
    df["Return_6M_Pct"] = (df["Close"] / df["Close_6M_Ago"] - 1.0) * 100.0

    df["Prior50_Avg_Volume"] = group["Volume"].transform(
        lambda s: s.shift(1).rolling(50, min_periods=50).mean()
    )
    df["Volume_Ratio_50D"] = df["Volume"] / df["Prior50_Avg_Volume"]
    df["Volume_Spike"] = df["Volume"] > (SPIKE_MULTIPLE * df["Prior50_Avg_Volume"])
    df["Spike_Days_6M"] = group["Volume_Spike"].transform(
        lambda s: s.rolling(SIX_MONTH_SESSIONS, min_periods=SIX_MONTH_SESSIONS).sum()
    )

    df["Last_Range_Pct"] = (df["High"] / df["Low"] - 1.0) * 100.0
    df["EMA20_Distance_Pct"] = (df["Close"] / df["EMA20"] - 1.0).abs() * 100.0

    latest_date = df["Date"].max()
    latest = df[df["Date"] == latest_date].copy()

    eligible = latest[
        (latest["Median_Turnover_50D"] >= MIN_MEDIAN_TURNOVER_RS)
        & (latest["Return_6M_Pct"] >= MIN_SIX_MONTH_RETURN_PCT)
        & (latest["Spike_Days_6M"] >= MIN_SPIKE_DAYS)
        & (latest["Close"] > latest["EMA50"])
        & (latest["Close"] > latest["EMA200"])
        & (latest["EMA20"] > latest["EMA50"])
        & (latest["EMA50"] > latest["EMA200"])
        & (latest["Last_Range_Pct"] <= MAX_LAST_RANGE_PCT)
        & (latest["Volume"] < latest["Prior50_Avg_Volume"])
        & (latest["EMA20_Distance_Pct"] <= MAX_EMA20_DISTANCE_PCT)
    ].copy()

    eligible["Median_Turnover_50D_Cr"] = eligible["Median_Turnover_50D"] / 1_00_00_000

    columns = [
        "Date", "Overall_Rank", "Symbol", "Close", "Median_Turnover_50D_Cr",
        "Return_6M_Pct", "Momentum_Rank",
        "Spike_Days_6M", "HighVolume_Rank",
        "Last_Range_Pct", "Range_Rank",
        "Volume_Ratio_50D", "LastVolume_Rank", "Compression_Rank",
        "EMA20", "EMA50", "EMA200", "EMA20_Distance_Pct", "EMA20_Proximity_Rank",
        "Momentum_Score", "HighVolume_Score", "Compression_Score", "EMA20_Proximity_Score", "Overall_Score",
    ]

    if eligible.empty:
        pd.DataFrame(columns=columns).to_csv(OUTPUT_FILE, index=False)
        write_tradingview_watchlist([])
        print(f"{latest_date.date()}: no stocks passed all setup rules.")
        print(f"Saved: {OUTPUT_FILE}")
        print(f"Saved: {TRADINGVIEW_OUTPUT_FILE}")
        return

    eligible["Momentum_Score"] = score_higher_better(eligible["Return_6M_Pct"])
    eligible["HighVolume_Score"] = score_higher_better(eligible["Spike_Days_6M"])

    range_score = score_lower_better(eligible["Last_Range_Pct"])
    volume_score = score_lower_better(eligible["Volume_Ratio_50D"])
    eligible["Compression_Score"] = (range_score + volume_score) / 2.0
    eligible["EMA20_Proximity_Score"] = score_lower_better(eligible["EMA20_Distance_Pct"])

    eligible["Overall_Score"] = eligible[
        ["Momentum_Score", "HighVolume_Score", "Compression_Score", "EMA20_Proximity_Score"]
    ].mean(axis=1)

    eligible["Momentum_Rank"] = eligible["Return_6M_Pct"].rank(method="min", ascending=False).astype(int)
    eligible["HighVolume_Rank"] = eligible["Spike_Days_6M"].rank(method="min", ascending=False).astype(int)
    eligible["Range_Rank"] = eligible["Last_Range_Pct"].rank(method="min", ascending=True).astype(int)
    eligible["LastVolume_Rank"] = eligible["Volume_Ratio_50D"].rank(method="min", ascending=True).astype(int)
    eligible["Compression_Rank"] = eligible["Compression_Score"].rank(method="min", ascending=False).astype(int)
    eligible["EMA20_Proximity_Rank"] = eligible["EMA20_Distance_Pct"].rank(method="min", ascending=True).astype(int)

    eligible = eligible.sort_values(
        ["Overall_Score", "Last_Range_Pct", "Volume_Ratio_50D", "EMA20_Distance_Pct", "Return_6M_Pct"],
        ascending=[False, True, True, True, False],
    ).reset_index(drop=True)
    eligible["Overall_Rank"] = eligible.index + 1

    eligible["Spike_Days_6M"] = eligible["Spike_Days_6M"].astype(int)
    for col in [
        "Close", "Median_Turnover_50D_Cr", "Return_6M_Pct", "Last_Range_Pct", "Volume_Ratio_50D",
        "EMA20", "EMA50", "EMA200", "EMA20_Distance_Pct",
        "Momentum_Score", "HighVolume_Score", "Compression_Score",
        "EMA20_Proximity_Score", "Overall_Score",
    ]:
        eligible[col] = eligible[col].round(2)

    eligible[columns].to_csv(OUTPUT_FILE, index=False)
    write_tradingview_watchlist(eligible["Symbol"].dropna().astype(str).tolist())

    print(f"Actionable setup date: {latest_date.date()}")
    print(f"Liquidity gate: 50-day median daily turnover >= Rs 5 crore")
    print(f"Eligible stocks: {len(eligible)}")
    print(eligible[columns[:19]].head(25).to_string(index=False))
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Saved: {TRADINGVIEW_OUTPUT_FILE}")


if __name__ == "__main__":
    main()

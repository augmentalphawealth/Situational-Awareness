"""
ISOLATED BREAKOUT BACKTEST
==========================

This script is for RESEARCH ONLY.

It reads:
    ../nse_6yr_historical.parquet
    ../historical_breadth_regime_6yr.csv

It creates NEW output files inside the Research folder:
    research_breakout_summary.csv
    research_feature_buckets.csv
    research_daily_regime_scorecard.csv
    research_breakout_trades.csv

It DOES NOT edit:
    dashboard.py
    analyze_6yr_data.py
    fetch_6yr_history.py
    fetch_eod_6yr.py
    nse_6yr_historical.parquet
    historical_breadth_regime_6yr.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1. FILE LOCATIONS
# ============================================================
# This script is inside the Research folder.
# ".." means go one folder UP to the main repository folder.
RESEARCH_FOLDER = Path(__file__).resolve().parent
REPO_FOLDER = RESEARCH_FOLDER.parent

PARQUET_FILE = REPO_FOLDER / "nse_6yr_historical.parquet"
BREADTH_FILE = REPO_FOLDER / "historical_breadth_regime_6yr.csv"

OUTPUT_SUMMARY = RESEARCH_FOLDER / "research_breakout_summary.csv"
OUTPUT_BUCKETS = RESEARCH_FOLDER / "research_feature_buckets.csv"
OUTPUT_DAILY = RESEARCH_FOLDER / "research_daily_regime_scorecard.csv"
OUTPUT_TRADES = RESEARCH_FOLDER / "research_breakout_trades.csv"


# ============================================================
# 2. SETTINGS MATCHED TO YOUR TRADING STYLE
# ============================================================
# Breakout definition:
# - Close at a 20-day closing high
# - Volume at least 50% above 20-day average
# - Previous day had ATR contraction below 4%

STOP_LOSS_PCT = 7.0
TARGET_PCT = 25.0
HOLDING_DAYS = 30

print("=" * 70)
print("ISOLATED BREAKOUT BACKTEST")
print("=" * 70)
print("This script only reads existing files and writes new research CSVs.")
print()

if not PARQUET_FILE.exists():
    raise FileNotFoundError(
        f"Parquet file was not found: {PARQUET_FILE}\n"
        "Check that nse_6yr_historical.parquet exists in the main repo folder."
    )

if not BREADTH_FILE.exists():
    raise FileNotFoundError(
        f"Breadth CSV was not found: {BREADTH_FILE}\n"
        "Check that historical_breadth_regime_6yr.csv exists in the main repo folder."
    )


# ============================================================
# 3. LOAD DATA
# ============================================================
print("Step 1 of 6: Loading parquet and breadth CSV...")

prices = pd.read_parquet(PARQUET_FILE)
breadth = pd.read_csv(BREADTH_FILE)

prices["Date"] = pd.to_datetime(prices["Date"])
breadth["Date"] = pd.to_datetime(breadth["Date"])

required_price_columns = {"Date", "Symbol", "Open", "High", "Low", "Close", "Volume"}
missing_price_columns = required_price_columns - set(prices.columns)

if missing_price_columns:
    raise ValueError(
        "Parquet is missing required columns: "
        + ", ".join(sorted(missing_price_columns))
    )

prices = prices.sort_values(["Symbol", "Date"]).reset_index(drop=True)
breadth = breadth.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)

print(f"Loaded {len(prices):,} stock-day records.")
print(f"Loaded {len(breadth):,} breadth-day records.")
print()


# ============================================================
# 4. REBUILD EXISTING BREAKOUT LOGIC
# ============================================================
# This intentionally matches your current analyzer logic:
# ATR14 / Close < 4%
# Volume > 1.5 × 20-day average volume
# Close >= 20-day closing high
# Previous day was tight

print("Step 2 of 6: Rebuilding your current VCP breakout definition...")

prices["PrevClose"] = prices.groupby("Symbol")["Close"].shift(1)

prices["TR"] = np.maximum(
    prices["High"] - prices["Low"],
    np.maximum(
        (prices["High"] - prices["PrevClose"]).abs(),
        (prices["Low"] - prices["PrevClose"]).abs(),
    ),
)

prices["ATR14"] = prices.groupby("Symbol")["TR"].transform(
    lambda series: series.rolling(window=14, min_periods=5).mean()
)

prices["Volume20DAvg"] = prices.groupby("Symbol")["Volume"].transform(
    lambda series: series.rolling(window=20, min_periods=5).mean()
)

prices["Close20DHigh"] = prices.groupby("Symbol")["Close"].transform(
    lambda series: series.rolling(window=20, min_periods=20).max()
)

prices["VCPTightness"] = (prices["ATR14"] / prices["Close"]) < 0.04
prices["VolumeSurge"] = prices["Volume"] > (prices["Volume20DAvg"] * 1.5)
prices["PreviousDayTight"] = prices.groupby("Symbol")["VCPTightness"].shift(1)

prices["IsBreakout"] = (
    (prices["Close"] >= prices["Close20DHigh"])
    & prices["VolumeSurge"]
    & prices["PreviousDayTight"].fillna(False)
)

breakout_count = int(prices["IsBreakout"].sum())
print(f"Breakout events found: {breakout_count:,}")
print()


# ============================================================
# 5. CALCULATE FORWARD TRADE OUTCOMES
# ============================================================
# We check the next 30 trading rows for each stock.
# Target-first / stop-first is calculated properly:
# If both occur, the script checks which was reached first.

print("Step 3 of 6: Calculating 10/20/30-day returns and 7% stop / 25% target outcomes...")

records = []

for symbol, stock in prices.groupby("Symbol", sort=False):
    stock = stock.sort_values("Date").reset_index(drop=True)
    breakout_indexes = stock.index[stock["IsBreakout"]].tolist()

    for entry_index in breakout_indexes:
        entry = stock.iloc[entry_index]
        future = stock.iloc[entry_index + 1 : entry_index + 1 + HOLDING_DAYS]

        if future.empty:
            continue

        entry_price = float(entry["Close"])
        stop_price = entry_price * (1 - STOP_LOSS_PCT / 100)
        target_price = entry_price * (1 + TARGET_PCT / 100)

        future_10 = stock.iloc[entry_index + 10] if entry_index + 10 < len(stock) else None
        future_20 = stock.iloc[entry_index + 20] if entry_index + 20 < len(stock) else None
        future_30 = stock.iloc[entry_index + 30] if entry_index + 30 < len(stock) else None

        fwd_10 = (
            (float(future_10["Close"]) / entry_price - 1) * 100
            if future_10 is not None else np.nan
        )
        fwd_20 = (
            (float(future_20["Close"]) / entry_price - 1) * 100
            if future_20 is not None else np.nan
        )
        fwd_30 = (
            (float(future_30["Close"]) / entry_price - 1) * 100
            if future_30 is not None else np.nan
        )

        max_high = float(future["High"].max())
        min_low = float(future["Low"].min())

        target_day = None
        stop_day = None

        for day_number, (_, day) in enumerate(future.iterrows(), start=1):
            if target_day is None and float(day["High"]) >= target_price:
                target_day = day_number

            if stop_day is None and float(day["Low"]) <= stop_price:
                stop_day = day_number

            if target_day is not None or stop_day is not None:
                break

        if target_day is not None and (stop_day is None or target_day < stop_day):
            trade_result = "TARGET_25_BEFORE_STOP"
            trade_return = TARGET_PCT
        elif stop_day is not None and (target_day is None or stop_day < target_day):
            trade_result = "STOP_7_BEFORE_TARGET"
            trade_return = -STOP_LOSS_PCT
        else:
            trade_result = "NEITHER_IN_30D"
            exit_price = float(future.iloc[-1]["Close"])
            trade_return = (exit_price / entry_price - 1) * 100

        records.append(
            {
                "Date": entry["Date"],
                "Symbol": symbol,
                "EntryPrice": entry_price,
                "StopPrice7Pct": stop_price,
                "TargetPrice25Pct": target_price,
                "ForwardReturn10D": fwd_10,
                "ForwardReturn20D": fwd_20,
                "ForwardReturn30D": fwd_30,
                "MaxFavorableMove30D": (max_high / entry_price - 1) * 100,
                "MaxAdverseMove30D": (min_low / entry_price - 1) * 100,
                "TargetDay": target_day,
                "StopDay": stop_day,
                "TradeResult": trade_result,
                "TradeReturnRuleBased": trade_return,
            }
        )

trades = pd.DataFrame(records)

if trades.empty:
    raise ValueError(
        "No completed breakout trades were found. "
        "This can happen if the data has fewer than 30 future trading days."
    )

print(f"Completed breakout trades available for analysis: {len(trades):,}")
print()


# ============================================================
# 6. ADD SAME-DAY MARKET BREADTH CONDITIONS
# ============================================================
print("Step 4 of 6: Adding same-day breadth conditions to each breakout...")

candidate_breadth_columns = [
    "Date",
    "PctAbove20EMA",
    "PctAbove50EMA",
    "PctAbove200EMA",
    "Slope20EMA",
    "Slope50EMA",
    "Slope200EMA",
    "VolumeRatio",
    "Net52WHighLow",
    "Rolling3DUp4",
    "Rolling3DDown4",
    "Up251MCount",
    "Down251MCount",
    "T3Breakouts",
    "T3Wins",
    "LargePct20EMA",
    "LargePct50EMA",
    "MidPct20EMA",
    "MidPct50EMA",
    "SmallPct20EMA",
    "SmallPct50EMA",
    "MicroPct20EMA",
    "MicroPct50EMA",
]

available_breadth_columns = [
    column for column in candidate_breadth_columns if column in breadth.columns
]

breadth_for_merge = breadth[available_breadth_columns].copy()

if "T3Breakouts" in breadth_for_merge.columns and "T3Wins" in breadth_for_merge.columns:
    breadth_for_merge["FollowThroughRate"] = np.where(
        breadth_for_merge["T3Breakouts"] > 0,
        breadth_for_merge["T3Wins"] / breadth_for_merge["T3Breakouts"] * 100,
        np.nan,
    )

trades = trades.merge(breadth_for_merge, on="Date", how="left")


# ============================================================
# 7. CREATE SUMMARY OUTPUT
# ============================================================
print("Step 5 of 6: Creating research summary and feature-bucket tables...")

summary_rows = [
    {"Metric": "Total completed breakout trades", "Value": len(trades)},
    {
        "Metric": "Average 10-day return (%)",
        "Value": trades["ForwardReturn10D"].mean(),
    },
    {
        "Metric": "Median 10-day return (%)",
        "Value": trades["ForwardReturn10D"].median(),
    },
    {
        "Metric": "Average 20-day return (%)",
        "Value": trades["ForwardReturn20D"].mean(),
    },
    {
        "Metric": "Median 20-day return (%)",
        "Value": trades["ForwardReturn20D"].median(),
    },
    {
        "Metric": "Average 30-day return (%)",
        "Value": trades["ForwardReturn30D"].mean(),
    },
    {
        "Metric": "25% target reached before 7% stop (%)",
        "Value": (trades["TradeResult"] == "TARGET_25_BEFORE_STOP").mean() * 100,
    },
    {
        "Metric": "7% stop reached before 25% target (%)",
        "Value": (trades["TradeResult"] == "STOP_7_BEFORE_TARGET").mean() * 100,
    },
    {
        "Metric": "Neither target nor stop in 30 days (%)",
        "Value": (trades["TradeResult"] == "NEITHER_IN_30D").mean() * 100,
    },
    {
        "Metric": "Average rule-based trade return (%)",
        "Value": trades["TradeReturnRuleBased"].mean(),
    },
    {
        "Metric": "Median rule-based trade return (%)",
        "Value": trades["TradeReturnRuleBased"].median(),
    },
]

summary = pd.DataFrame(summary_rows)
summary["Value"] = summary["Value"].round(3)
summary.to_csv(OUTPUT_SUMMARY, index=False)


# ============================================================
# 8. TEST EACH MARKET FEATURE
# ============================================================
# Low / Medium / High means three equal-sized groups.
# Example:
# "High PctAbove20EMA" = breakouts taken on days with the strongest
# one-third of PctAbove20EMA readings in your actual history.

features_to_test = [
    "PctAbove20EMA",
    "PctAbove50EMA",
    "PctAbove200EMA",
    "Slope20EMA",
    "Slope50EMA",
    "Slope200EMA",
    "VolumeRatio",
    "Net52WHighLow",
    "Rolling3DUp4",
    "Rolling3DDown4",
    "Up251MCount",
    "Down251MCount",
    "FollowThroughRate",
    "LargePct20EMA",
    "LargePct50EMA",
    "MidPct20EMA",
    "MidPct50EMA",
    "SmallPct20EMA",
    "SmallPct50EMA",
    "MicroPct20EMA",
    "MicroPct50EMA",
]

bucket_rows = []

for feature in features_to_test:
    if feature not in trades.columns:
        continue

    valid = trades.dropna(subset=[feature]).copy()

    if len(valid) < 30:
        continue

    try:
        valid["Bucket"] = pd.qcut(
            valid[feature].rank(method="first"),
            q=3,
            labels=["Low", "Medium", "High"],
        )
    except ValueError:
        continue

    for bucket_name, group in valid.groupby("Bucket", observed=False):
        bucket_rows.append(
            {
                "Feature": feature,
                "Bucket": str(bucket_name),
                "BreakoutTrades": len(group),
                "Average10DayReturnPct": group["ForwardReturn10D"].mean(),
                "Average20DayReturnPct": group["ForwardReturn20D"].mean(),
                "Median20DayReturnPct": group["ForwardReturn20D"].median(),
                "Target25BeforeStopPct": (
                    group["TradeResult"] == "TARGET_25_BEFORE_STOP"
                ).mean() * 100,
                "Stop7BeforeTargetPct": (
                    group["TradeResult"] == "STOP_7_BEFORE_TARGET"
                ).mean() * 100,
                "AverageRuleBasedReturnPct": group["TradeReturnRuleBased"].mean(),
            }
        )

buckets = pd.DataFrame(bucket_rows)

if not buckets.empty:
    number_columns = [
        "Average10DayReturnPct",
        "Average20DayReturnPct",
        "Median20DayReturnPct",
        "Target25BeforeStopPct",
        "Stop7BeforeTargetPct",
        "AverageRuleBasedReturnPct",
    ]
    buckets[number_columns] = buckets[number_columns].round(3)
    buckets = buckets.sort_values(["Feature", "Bucket"])
    buckets.to_csv(OUTPUT_BUCKETS, index=False)
else:
    pd.DataFrame(
        columns=[
            "Feature",
            "Bucket",
            "BreakoutTrades",
            "Average10DayReturnPct",
            "Average20DayReturnPct",
            "Median20DayReturnPct",
            "Target25BeforeStopPct",
            "Stop7BeforeTargetPct",
            "AverageRuleBasedReturnPct",
        ]
    ).to_csv(OUTPUT_BUCKETS, index=False)


# ============================================================
# 9. CREATE DAILY SCORECARD
# ============================================================
daily_scorecard = (
    trades.groupby("Date")
    .agg(
        BreakoutsThatDay=("Symbol", "count"),
        Average10DayReturnPct=("ForwardReturn10D", "mean"),
        Average20DayReturnPct=("ForwardReturn20D", "mean"),
        Target25BeforeStopPct=(
            "TradeResult",
            lambda values: (values == "TARGET_25_BEFORE_STOP").mean() * 100,
        ),
        Stop7BeforeTargetPct=(
            "TradeResult",
            lambda values: (values == "STOP_7_BEFORE_TARGET").mean() * 100,
        ),
        AverageRuleBasedReturnPct=("TradeReturnRuleBased", "mean"),
    )
    .reset_index()
)

daily_scorecard = daily_scorecard.merge(breadth_for_merge, on="Date", how="left")
daily_scorecard = daily_scorecard.round(3)
daily_scorecard.to_csv(OUTPUT_DAILY, index=False)

trades = trades.sort_values(["Date", "Symbol"]).round(3)
trades.to_csv(OUTPUT_TRADES, index=False)


# ============================================================
# 10. FINISHED
# ============================================================
print()
print("=" * 70)
print("BACKTEST COMPLETE")
print("=" * 70)
print("New research-only files created:")
print(f"1. {OUTPUT_SUMMARY.name}")
print(f"2. {OUTPUT_BUCKETS.name}")
print(f"3. {OUTPUT_DAILY.name}")
print(f"4. {OUTPUT_TRADES.name}")
print()
print("These four files are inside the Research folder.")
print("Your existing dashboard and live data files were not changed.")

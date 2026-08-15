"""
BREAKOUT BACKTEST - RESEARCH ONLY

This script reads the two existing data files from the main repository folder:
    nse_6yr_historical.parquet
    historical_breadth_regime_6yr.csv

It creates four NEW CSV files inside the Research folder:
    research_breakout_summary.csv
    research_feature_buckets.csv
    research_daily_regime_scorecard.csv
    research_breakout_trades.csv

It does NOT edit your dashboard, analyzer, parquet, or live breadth files.

YOUR CORE TRADE RULE TESTED HERE:
- Entry: Existing VCP-style breakout definition
- Tightness: ATR14 / Close below 4%
- Stop-loss: Maximum 7%
- Minimum target: 15%
- Maximum holding time: 30 trading days
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS - YOUR TRADING RULES
# ============================================================

# Keep the existing VCP breakout/tightness definition at 4%.
VCP_TIGHTNESS_LIMIT = 0.04

# Your maximum acceptable stop-loss.
STOP_LOSS_PCT = 7.0

# Your minimum acceptable profit target.
MINIMUM_TARGET_PCT = 15.0

# Your maximum holding period in trading days.
MAX_HOLDING_DAYS = 30

# Extra milestones recorded for analysis.
EXTRA_TARGETS = [18.0, 25.0, 30.0]


# ============================================================
# FILE LOCATIONS
# ============================================================

# This script is inside the Research folder.
# parent means "go one level up to the main repo folder".
RESEARCH_FOLDER = Path(__file__).resolve().parent
REPO_FOLDER = RESEARCH_FOLDER.parent

PARQUET_FILE = REPO_FOLDER / "nse_6yr_historical.parquet"
BREADTH_FILE = REPO_FOLDER / "historical_breadth_regime_6yr.csv"

OUTPUT_SUMMARY = RESEARCH_FOLDER / "research_breakout_summary.csv"
OUTPUT_BUCKETS = RESEARCH_FOLDER / "research_feature_buckets.csv"
OUTPUT_DAILY = RESEARCH_FOLDER / "research_daily_regime_scorecard.csv"
OUTPUT_TRADES = RESEARCH_FOLDER / "research_breakout_trades.csv"


# ============================================================
# HELPER FUNCTION
# ============================================================

def first_hit_day(future_data, entry_price, target_percent, stop_percent):
    """
    Checks each future trading day in sequence.

    Returns:
    - SUCCESS if target is reached before stop
    - STOPPED_OUT if stop is reached before target
    - NO_DECISION if neither is hit during the maximum holding period
    """

    target_price = entry_price * (1 + target_percent / 100)
    stop_price = entry_price * (1 - stop_percent / 100)

    for day_number, (_, day) in enumerate(future_data.iterrows(), start=1):
        hit_target = float(day["High"]) >= target_price
        hit_stop = float(day["Low"]) <= stop_price

        # Daily OHLC data cannot prove the intraday order if BOTH occur.
        # Mark these rare cases separately instead of pretending to know.
        if hit_target and hit_stop:
            return "TARGET_AND_STOP_SAME_DAY", day_number

        if hit_target:
            return "SUCCESS", day_number

        if hit_stop:
            return "STOPPED_OUT", day_number

    return "NO_DECISION", np.nan


# ============================================================
# START
# ============================================================

print("=" * 72)
print("BREAKOUT BACKTEST - RESEARCH ONLY")
print("=" * 72)
print(f"VCP tightness filter: ATR14 / Close < {VCP_TIGHTNESS_LIMIT * 100:.0f}%")
print(f"Core trade rule: +{MINIMUM_TARGET_PCT:.0f}% before -{STOP_LOSS_PCT:.0f}%")
print(f"Maximum holding period: {MAX_HOLDING_DAYS} trading days")
print()

if not PARQUET_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find: {PARQUET_FILE.name}\n"
        "It must be in the main repository folder."
    )

if not BREADTH_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find: {BREADTH_FILE.name}\n"
        "It must be in the main repository folder."
    )


# ============================================================
# LOAD DATA
# ============================================================

print("Step 1 of 6: Loading data files...")

prices = pd.read_parquet(PARQUET_FILE)
breadth = pd.read_csv(BREADTH_FILE)

prices["Date"] = pd.to_datetime(prices["Date"])
breadth["Date"] = pd.to_datetime(breadth["Date"])

required_columns = {"Date", "Symbol", "Open", "High", "Low", "Close", "Volume"}
missing_columns = required_columns - set(prices.columns)

if missing_columns:
    raise ValueError(
        "The parquet file is missing these columns: "
        + ", ".join(sorted(missing_columns))
    )

prices = prices.sort_values(["Symbol", "Date"]).reset_index(drop=True)
breadth = breadth.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)

print(f"Loaded {len(prices):,} stock-day records.")
print(f"Loaded {len(breadth):,} breadth-day records.")
print()


# ============================================================
# REBUILD VCP BREAKOUT LOGIC
# ============================================================

print("Step 2 of 6: Identifying VCP-style breakout entries...")

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

prices["VCPTightness"] = (
    prices["ATR14"] / prices["Close"]
) < VCP_TIGHTNESS_LIMIT

prices["VolumeSurge"] = prices["Volume"] > (
    prices["Volume20DAvg"] * 1.5
)

prices["PreviousDayTight"] = prices.groupby("Symbol")[
    "VCPTightness"
].shift(1)

prices["IsBreakout"] = (
    (prices["Close"] >= prices["Close20DHigh"])
    & prices["VolumeSurge"]
    & prices["PreviousDayTight"].fillna(False)
)

print(f"Breakout entries found: {int(prices['IsBreakout'].sum()):,}")
print()


# ============================================================
# FORWARD TRADE OUTCOMES
# ============================================================

print("Step 3 of 6: Testing 15% target before 7% stop...")

trade_records = []

for symbol, stock_data in prices.groupby("Symbol", sort=False):
    stock_data = stock_data.sort_values("Date").reset_index(drop=True)

    breakout_indexes = stock_data.index[
        stock_data["IsBreakout"]
    ].tolist()

    for entry_index in breakout_indexes:
        entry = stock_data.iloc[entry_index]

        future = stock_data.iloc[
            entry_index + 1 : entry_index + 1 + MAX_HOLDING_DAYS
        ].copy()

        # Skip entries without enough future data.
        if len(future) < MAX_HOLDING_DAYS:
            continue

        entry_price = float(entry["Close"])
        stop_price = entry_price * (1 - STOP_LOSS_PCT / 100)
        target_15_price = entry_price * (
            1 + MINIMUM_TARGET_PCT / 100
        )

        core_result, core_day = first_hit_day(
            future_data=future,
            entry_price=entry_price,
            target_percent=MINIMUM_TARGET_PCT,
            stop_percent=STOP_LOSS_PCT,
        )

        future_10 = stock_data.iloc[entry_index + 10]
        future_20 = stock_data.iloc[entry_index + 20]
        future_30 = stock_data.iloc[entry_index + 30]

        max_high = float(future["High"].max())
        min_low = float(future["Low"].min())

        record = {
            "Date": entry["Date"],
            "Symbol": symbol,
            "EntryPrice": entry_price,
            "StopPrice7Pct": stop_price,
            "TargetPrice15Pct": target_15_price,
            "ForwardReturn10D": (
                float(future_10["Close"]) / entry_price - 1
            ) * 100,
            "ForwardReturn20D": (
                float(future_20["Close"]) / entry_price - 1
            ) * 100,
            "ForwardReturn30D": (
                float(future_30["Close"]) / entry_price - 1
            ) * 100,
            "MaxFavorableMove30D": (
                max_high / entry_price - 1
            ) * 100,
            "MaxAdverseMove30D": (
                min_low / entry_price - 1
            ) * 100,
            "CoreResult15Before7": core_result,
            "CoreDecisionDay": core_day,
        }

        # Record whether 18%, 25% and 30% were hit before the 7% stop.
        for extra_target in EXTRA_TARGETS:
            extra_result, extra_day = first_hit_day(
                future_data=future,
                entry_price=entry_price,
                target_percent=extra_target,
                stop_percent=STOP_LOSS_PCT,
            )

            target_label = str(int(extra_target))
            record[f"Target{target_label}BeforeStop7"] = extra_result
            record[f"Target{target_label}DecisionDay"] = extra_day

        # Rule-based return for the primary 15% / 7% test.
        if core_result == "SUCCESS":
            record["RuleBasedReturn15Before7"] = MINIMUM_TARGET_PCT
        elif core_result == "STOPPED_OUT":
            record["RuleBasedReturn15Before7"] = -STOP_LOSS_PCT
        else:
            record["RuleBasedReturn15Before7"] = (
                float(future.iloc[-1]["Close"]) / entry_price - 1
            ) * 100

        trade_records.append(record)

trades = pd.DataFrame(trade_records)

if trades.empty:
    raise ValueError(
        "No fully completed 30-trading-day breakout records were found."
    )

print(f"Completed trade records available: {len(trades):,}")
print()


# ============================================================
# ADD SAME-DAY BREADTH DATA
# ============================================================

print("Step 4 of 6: Adding market-breadth conditions...")

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
    "LargePct200EMA",
    "MidPct20EMA",
    "MidPct50EMA",
    "MidPct200EMA",
    "SmallPct20EMA",
    "SmallPct50EMA",
    "SmallPct200EMA",
    "MicroPct20EMA",
    "MicroPct50EMA",
    "MicroPct200EMA",
]

available_columns = [
    column
    for column in candidate_breadth_columns
    if column in breadth.columns
]

breadth_for_merge = breadth[available_columns].copy()

if (
    "T3Breakouts" in breadth_for_merge.columns
    and "T3Wins" in breadth_for_merge.columns
):
    breadth_for_merge["FollowThroughRate"] = np.where(
        breadth_for_merge["T3Breakouts"] > 0,
        breadth_for_merge["T3Wins"]
        / breadth_for_merge["T3Breakouts"]
        * 100,
        np.nan,
    )

trades = trades.merge(breadth_for_merge, on="Date", how="left")


# ============================================================
# CREATE OVERALL SUMMARY
# ============================================================

print("Step 5 of 6: Creating summary and feature tests...")

core_success = trades["CoreResult15Before7"] == "SUCCESS"
core_stop = trades["CoreResult15Before7"] == "STOPPED_OUT"
core_same_day = (
    trades["CoreResult15Before7"] == "TARGET_AND_STOP_SAME_DAY"
)
core_neither = trades["CoreResult15Before7"] == "NO_DECISION"

summary_rows = [
    {
        "Metric": "Completed breakout trades tested",
        "Value": len(trades),
    },
    {
        "Metric": "15% target before 7% stop (%)",
        "Value": core_success.mean() * 100,
    },
    {
        "Metric": "7% stop before 15% target (%)",
        "Value": core_stop.mean() * 100,
    },
    {
        "Metric": "15% target and 7% stop same day (%)",
        "Value": core_same_day.mean() * 100,
    },
    {
        "Metric": "Neither 15% target nor 7% stop in 30 days (%)",
        "Value": core_neither.mean() * 100,
    },
    {
        "Metric": "18% target before 7% stop (%)",
        "Value": (
            trades["Target18BeforeStop7"] == "SUCCESS"
        ).mean() * 100,
    },
    {
        "Metric": "25% target before 7% stop (%)",
        "Value": (
            trades["Target25BeforeStop7"] == "SUCCESS"
        ).mean() * 100,
    },
    {
        "Metric": "30% target before 7% stop (%)",
        "Value": (
            trades["Target30BeforeStop7"] == "SUCCESS"
        ).mean() * 100,
    },
    {
        "Metric": "Average 10-day close-to-close return (%)",
        "Value": trades["ForwardReturn10D"].mean(),
    },
    {
        "Metric": "Median 10-day close-to-close return (%)",
        "Value": trades["ForwardReturn10D"].median(),
    },
    {
        "Metric": "Average 20-day close-to-close return (%)",
        "Value": trades["ForwardReturn20D"].mean(),
    },
    {
        "Metric": "Median 20-day close-to-close return (%)",
        "Value": trades["ForwardReturn20D"].median(),
    },
    {
        "Metric": "Average 30-day close-to-close return (%)",
        "Value": trades["ForwardReturn30D"].mean(),
    },
    {
        "Metric": "Average rule-based return: 15% before 7% (%)",
        "Value": trades["RuleBasedReturn15Before7"].mean(),
    },
]

summary = pd.DataFrame(summary_rows)
summary["Value"] = summary["Value"].round(3)
summary.to_csv(OUTPUT_SUMMARY, index=False)


# ============================================================
# FEATURE-BUCKET TESTS
# ============================================================

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
    "LargePct200EMA",
    "MidPct20EMA",
    "MidPct50EMA",
    "MidPct200EMA",
    "SmallPct20EMA",
    "SmallPct50EMA",
    "SmallPct200EMA",
    "MicroPct20EMA",
    "MicroPct50EMA",
    "MicroPct200EMA",
]

bucket_rows = []

for feature in features_to_test:
    if feature not in trades.columns:
        continue

    valid_data = trades.dropna(subset=[feature]).copy()

    if len(valid_data) < 30:
        continue

    try:
        valid_data["Bucket"] = pd.qcut(
            valid_data[feature].rank(method="first"),
            q=3,
            labels=["Low", "Medium", "High"],
        )
    except ValueError:
        continue

    for bucket_name, group in valid_data.groupby(
        "Bucket", observed=False
    ):
        bucket_rows.append(
            {
                "Feature": feature,
                "Bucket": str(bucket_name),
                "BreakoutTrades": len(group),
                "Target15BeforeStop7Pct": (
                    group["CoreResult15Before7"] == "SUCCESS"
                ).mean()
                * 100,
                "Stop7BeforeTarget15Pct": (
                    group["CoreResult15Before7"] == "STOPPED_OUT"
                ).mean()
                * 100,
                "Target18BeforeStop7Pct": (
                    group["Target18BeforeStop7"] == "SUCCESS"
                ).mean()
                * 100,
                "Target25BeforeStop7Pct": (
                    group["Target25BeforeStop7"] == "SUCCESS"
                ).mean()
                * 100,
                "Target30BeforeStop7Pct": (
                    group["Target30BeforeStop7"] == "SUCCESS"
                ).mean()
                * 100,
                "Average10DayReturnPct": group[
                    "ForwardReturn10D"
                ].mean(),
                "Average20DayReturnPct": group[
                    "ForwardReturn20D"
                ].mean(),
                "AverageRuleBasedReturnPct": group[
                    "RuleBasedReturn15Before7"
                ].mean(),
            }
        )

buckets = pd.DataFrame(bucket_rows)

if buckets.empty:
    buckets = pd.DataFrame(
        columns=[
            "Feature",
            "Bucket",
            "BreakoutTrades",
            "Target15BeforeStop7Pct",
            "Stop7BeforeTarget15Pct",
            "Target18BeforeStop7Pct",
            "Target25BeforeStop7Pct",
            "Target30BeforeStop7Pct",
            "Average10DayReturnPct",
            "Average20DayReturnPct",
            "AverageRuleBasedReturnPct",
        ]
    )
else:
    numeric_columns = [
        "Target15BeforeStop7Pct",
        "Stop7BeforeTarget15Pct",
        "Target18BeforeStop7Pct",
        "Target25BeforeStop7Pct",
        "Target30BeforeStop7Pct",
        "Average10DayReturnPct",
        "Average20DayReturnPct",
        "AverageRuleBasedReturnPct",
    ]

    buckets[numeric_columns] = buckets[numeric_columns].round(3)
    buckets = buckets.sort_values(["Feature", "Bucket"])

buckets.to_csv(OUTPUT_BUCKETS, index=False)


# ============================================================
# DAILY REGIME SCORECARD
# ============================================================

daily_scorecard = (
    trades.groupby("Date")
    .agg(
        BreakoutsThatDay=("Symbol", "count"),
        Target15BeforeStop7Pct=(
            "CoreResult15Before7",
            lambda values: (values == "SUCCESS").mean() * 100,
        ),
        Stop7BeforeTarget15Pct=(
            "CoreResult15Before7",
            lambda values: (values == "STOPPED_OUT").mean() * 100,
        ),
        Target18BeforeStop7Pct=(
            "Target18BeforeStop7",
            lambda values: (values == "SUCCESS").mean() * 100,
        ),
        Target25BeforeStop7Pct=(
            "Target25BeforeStop7",
            lambda values: (values == "SUCCESS").mean() * 100,
        ),
        Target30BeforeStop7Pct=(
            "Target30BeforeStop7",
            lambda values: (values == "SUCCESS").mean() * 100,
        ),
        Average10DayReturnPct=("ForwardReturn10D", "mean"),
        Average20DayReturnPct=("ForwardReturn20D", "mean"),
        AverageRuleBasedReturnPct=(
            "RuleBasedReturn15Before7",
            "mean",
        ),
    )
    .reset_index()
)

daily_scorecard = daily_scorecard.merge(
    breadth_for_merge,
    on="Date",
    how="left",
)

daily_scorecard = daily_scorecard.round(3)
daily_scorecard.to_csv(OUTPUT_DAILY, index=False)


# ============================================================
# SAVE INDIVIDUAL TRADE RECORDS
# ============================================================

trades = trades.sort_values(["Date", "Symbol"]).round(3)
trades.to_csv(OUTPUT_TRADES, index=False)


# ============================================================
# FINISH
# ============================================================

print()
print("=" * 72)
print("BACKTEST COMPLETE")
print("=" * 72)
print("Created these new files inside the Research folder:")
print(f"1. {OUTPUT_SUMMARY.name}")
print(f"2. {OUTPUT_BUCKETS.name}")
print(f"3. {OUTPUT_DAILY.name}")
print(f"4. {OUTPUT_TRADES.name}")
print()
print("Your dashboard and existing live data files were not changed.")

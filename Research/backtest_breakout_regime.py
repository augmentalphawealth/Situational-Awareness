from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd


# ============================================================
# FILES AND TRADING RULES
# ============================================================

RESEARCH_FOLDER = Path(__file__).resolve().parent
REPO_FOLDER = RESEARCH_FOLDER.parent

PARQUET_FILE = REPO_FOLDER / "nse_6yr_historical.parquet"
BREADTH_FILE = REPO_FOLDER / "historical_breadth_regime_6yr.csv"

# Your existing VCP-type entry definition.
VCP_TIGHTNESS_LIMIT = 0.04

# Your trading objective.
STOP_LOSS_PCT = 7.0
TARGET_PCT = 15.0
MAX_HOLDING_DAYS = 30

# Research output files.
OUTPUT_AUDIT = RESEARCH_FOLDER / "research_input_audit.csv"
OUTPUT_TRADES = RESEARCH_FOLDER / "research_breakout_trades_enriched.csv"
OUTPUT_BUCKETS = RESEARCH_FOLDER / "research_feature_buckets.csv"
OUTPUT_RANKINGS = RESEARCH_FOLDER / "research_feature_rankings.csv"
OUTPUT_CANDIDATES = RESEARCH_FOLDER / "research_candidate_scores.csv"
OUTPUT_WALKFORWARD = RESEARCH_FOLDER / "research_walkforward_results.csv"
OUTPUT_DAILY_SCORE = RESEARCH_FOLDER / "research_best_composite_score_daily.csv"
OUTPUT_ZONES = RESEARCH_FOLDER / "research_score_action_zones.csv"
OUTPUT_SUMMARY = RESEARCH_FOLDER / "research_run_summary.csv"


# ============================================================
# COLUMN NAME MAP
# ============================================================
# LEFT SIDE: actual headings in historical_breadth_regime_6yr.csv
# RIGHT SIDE: simpler internal names used by this research code.

COLUMN_MAP = {
    "Total_Universe": "TotalUniverse",
    "Above_20_EMA": "Above20EMA",
    "Above_50_EMA": "Above50EMA",
    "Above_200_EMA": "Above200EMA",
    "Up_4_Count": "Up4Count",
    "Down_4_Count": "Down4Count",
    "Up_25_1M_Count": "Up251MCount",
    "Down_25_1M_Count": "Down251MCount",
    "New_52W_Highs": "New52WHighs",
    "New_52W_Lows": "New52WLows",
    "Total_Up_Volume": "TotalUpVolume",
    "Total_Down_Volume": "TotalDownVolume",
    "T3_Breakouts": "T3Breakouts",
    "T3_Wins": "T3Wins",
    "Pct_Above_20_EMA": "PctAbove20EMA",
    "Pct_Above_50_EMA": "PctAbove50EMA",
    "Pct_Above_200_EMA": "PctAbove200EMA",
    "Rolling_3D_Up_4": "Rolling3DUp4",
    "Rolling_3D_Down_4": "Rolling3DDown4",
    "Slope_20_EMA": "Slope20EMA",
    "Slope_50_EMA": "Slope50EMA",
    "Slope_200_EMA": "Slope200EMA",
    "Net_52W_High_Low": "Net52WHighLow",
    "Volume_Ratio": "VolumeRatio",
    "Large_Pct_20_EMA": "LargePct20EMA",
    "Large_Pct_50_EMA": "LargePct50EMA",
    "Large_Pct_200_EMA": "LargePct200EMA",
    "Mid_Pct_20_EMA": "MidPct20EMA",
    "Mid_Pct_50_EMA": "MidPct50EMA",
    "Mid_Pct_200_EMA": "MidPct200EMA",
    "Small_Pct_20_EMA": "SmallPct20EMA",
    "Small_Pct_50_EMA": "SmallPct50EMA",
    "Small_Pct_200_EMA": "SmallPct200EMA",
    "Micro_Pct_20_EMA": "MicroPct20EMA",
    "Micro_Pct_50_EMA": "MicroPct50EMA",
    "Micro_Pct_200_EMA": "MicroPct200EMA",
}

FEATURES = [
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

# These are negative when they rise.
DOWNSIDE_FEATURES = {
    "Rolling3DDown4",
    "Down251MCount",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_dates(series):
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def safe_csv(dataframe, file_path):
    dataframe.to_csv(file_path, index=False)


def percentile_score(series):
    return series.rank(pct=True, method="average") * 100


def first_hit(future_data, entry_price, target_pct, stop_pct):
    target_price = entry_price * (1 + target_pct / 100)
    stop_price = entry_price * (1 - stop_pct / 100)

    for day_number, (_, row) in enumerate(future_data.iterrows(), start=1):
        target_hit = float(row["High"]) >= target_price
        stop_hit = float(row["Low"]) <= stop_price

        # Daily data cannot establish which happened first if both
        # target and stop occurred on the same candle.
        if target_hit and stop_hit:
            return "AMBIGUOUS_SAME_DAY", day_number

        if target_hit:
            return "SUCCESS", day_number

        if stop_hit:
            return "STOPPED_OUT", day_number

    return "TIMEOUT", np.nan


# ============================================================
# LOAD DATA
# ============================================================

print("Loading parquet and breadth data...")

if not PARQUET_FILE.exists():
    raise FileNotFoundError(f"Parquet missing: {PARQUET_FILE}")

if not BREADTH_FILE.exists():
    raise FileNotFoundError(f"Breadth CSV missing: {BREADTH_FILE}")

prices = pd.read_parquet(PARQUET_FILE)
breadth = pd.read_csv(BREADTH_FILE)

# Remove accidental spaces from CSV headers, then rename them.
breadth.columns = breadth.columns.astype(str).str.strip()
breadth = breadth.rename(columns=COLUMN_MAP)

prices["Date"] = clean_dates(prices["Date"])
breadth["Date"] = clean_dates(breadth["Date"])

prices = prices.dropna(
    subset=["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]
).copy()

breadth = breadth.dropna(subset=["Date"]).copy()

prices = prices.sort_values(["Symbol", "Date"]).reset_index(drop=True)
breadth = breadth.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)

for column in breadth.columns:
    if column != "Date":
        breadth[column] = pd.to_numeric(
            breadth[column],
            errors="coerce",
        )

required_price_columns = {
    "Date",
    "Symbol",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
}

missing_price_columns = required_price_columns - set(prices.columns)

if missing_price_columns:
    raise ValueError(
        "Missing required parquet columns: "
        + ", ".join(sorted(missing_price_columns))
    )

print(f"Price rows loaded: {len(prices):,}")
print(f"Breadth rows loaded: {len(breadth):,}")
print()


# ============================================================
# BUILD VCP BREAKOUT SIGNALS
# ============================================================

print("Building VCP-style breakout entries...")

prices["PrevClose"] = prices.groupby("Symbol")["Close"].shift(1)

prices["TrueRange"] = np.maximum(
    prices["High"] - prices["Low"],
    np.maximum(
        (prices["High"] - prices["PrevClose"]).abs(),
        (prices["Low"] - prices["PrevClose"]).abs(),
    ),
)

prices["ATR14"] = prices.groupby("Symbol")["TrueRange"].transform(
    lambda series: series.rolling(window=14, min_periods=5).mean()
)

prices["Volume20Average"] = prices.groupby("Symbol")["Volume"].transform(
    lambda series: series.rolling(window=20, min_periods=5).mean()
)

prices["Close20DayHigh"] = prices.groupby("Symbol")["Close"].transform(
    lambda series: series.rolling(window=20, min_periods=20).max()
)

prices["VCPTightness"] = (
    prices["ATR14"] / prices["Close"]
) < VCP_TIGHTNESS_LIMIT

prices["VolumeSurge"] = prices["Volume"] > (
    prices["Volume20Average"] * 1.5
)

prices["PriorDayTight"] = prices.groupby("Symbol")[
    "VCPTightness"
].shift(1)

prices["IsBreakout"] = (
    (prices["Close"] >= prices["Close20DayHigh"])
    & prices["VolumeSurge"]
    & prices["PriorDayTight"].fillna(False)
)

print(f"Breakout entries found: {int(prices['IsBreakout'].sum()):,}")
print()


# ============================================================
# BACKTEST EVERY BREAKOUT
# ============================================================

print("Testing 15% target before 7% stop...")

trade_records = []

for symbol, stock_data in prices.groupby("Symbol", sort=False):
    stock_data = stock_data.sort_values("Date").reset_index(drop=True)

    breakout_indexes = stock_data.index[
        stock_data["IsBreakout"]
    ].tolist()

    for entry_index in breakout_indexes:
        future = stock_data.iloc[
            entry_index + 1 : entry_index + 1 + MAX_HOLDING_DAYS
        ].copy()

        # We need all 30 days to test a complete maximum-hold trade.
        if len(future) < MAX_HOLDING_DAYS:
            continue

        entry = stock_data.iloc[entry_index]
        entry_price = float(entry["Close"])

        core_result, core_day = first_hit(
            future_data=future,
            entry_price=entry_price,
            target_pct=TARGET_PCT,
            stop_pct=STOP_LOSS_PCT,
        )

        record = {
            "Date": entry["Date"],
            "Symbol": symbol,
            "EntryPrice": entry_price,
            "Result15Before7": core_result,
            "DecisionDay15Before7": core_day,
        }

        for target in [18, 25, 30]:
            result, day = first_hit(
                future_data=future,
                entry_price=entry_price,
                target_pct=target,
                stop_pct=STOP_LOSS_PCT,
            )

            record[f"Target{target}BeforeStop7"] = result
            record[f"Target{target}DecisionDay"] = day

        for holding_day in [5, 10, 15, 20, 30]:
            record[f"CloseReturn{holding_day}D"] = (
                float(stock_data.iloc[entry_index + holding_day]["Close"])
                / entry_price
                - 1
            ) * 100

        if core_result == "SUCCESS":
            record["RuleBasedReturn"] = TARGET_PCT
        elif core_result == "STOPPED_OUT":
            record["RuleBasedReturn"] = -STOP_LOSS_PCT
        else:
            record["RuleBasedReturn"] = (
                float(future.iloc[-1]["Close"]) / entry_price - 1
            ) * 100

        trade_records.append(record)

trades = pd.DataFrame(trade_records)

if trades.empty:
    raise ValueError("No completed breakout trades were created.")

print(f"Completed breakout trades: {len(trades):,}")
print()


# ============================================================
# ATTACH BREADTH DATA
# ============================================================

print("Attaching market-breadth data to each trade...")

if {"T3Breakouts", "T3Wins"}.issubset(breadth.columns):
    breadth["FollowThroughRate"] = np.where(
        breadth["T3Breakouts"] > 0,
        breadth["T3Wins"] / breadth["T3Breakouts"] * 100,
        np.nan,
    )

available_features = [
    feature for feature in FEATURES if feature in breadth.columns
]

breadth_for_merge = breadth[
    ["Date"] + available_features
].copy()

trades = trades.merge(
    breadth_for_merge,
    on="Date",
    how="left",
)

usable_features = [
    feature
    for feature in available_features
    if trades[feature].notna().sum() >= 100
]

merge_coverage = (
    trades[usable_features].notna().any(axis=1).mean() * 100
    if usable_features
    else 0
)

print(f"Available breadth features: {len(available_features)}")
print(f"Usable breadth features: {len(usable_features)}")
print(f"Breadth merge coverage: {merge_coverage:.2f}%")
print()


# ============================================================
# INPUT AUDIT
# ============================================================

audit_rows = [
    {"Item": "Price parquet rows", "Value": len(prices)},
    {"Item": "Breadth CSV rows", "Value": len(breadth)},
    {"Item": "Completed breakout trade records", "Value": len(trades)},
    {"Item": "Price data first date", "Value": prices["Date"].min()},
    {"Item": "Price data last date", "Value": prices["Date"].max()},
    {"Item": "Breadth data first date", "Value": breadth["Date"].min()},
    {"Item": "Breadth data last date", "Value": breadth["Date"].max()},
    {
        "Item": "Available breadth features",
        "Value": len(available_features),
    },
    {
        "Item": "Usable breadth features after merge",
        "Value": len(usable_features),
    },
    {
        "Item": "Breadth merge coverage percent",
        "Value": round(merge_coverage, 3),
    },
]

for feature in available_features:
    audit_rows.append(
        {
            "Item": f"Available rows for {feature}",
            "Value": int(trades[feature].notna().sum()),
        }
    )

audit = pd.DataFrame(audit_rows)
safe_csv(audit, OUTPUT_AUDIT)

trades.to_csv(OUTPUT_TRADES, index=False)


# ============================================================
# FEATURE BUCKET TESTS
# ============================================================

print("Ranking individual market features...")

bucket_rows = []
ranking_rows = []

for feature in usable_features:
    data = trades.dropna(subset=[feature]).copy()

    try:
        data["Bucket"] = pd.qcut(
            data[feature].rank(method="first"),
            q=3,
            labels=["Low", "Medium", "High"],
        )
    except ValueError:
        continue

    low_group = data[data["Bucket"] == "Low"]
    high_group = data[data["Bucket"] == "High"]

    low_success = (
        low_group["Result15Before7"] == "SUCCESS"
    ).mean() * 100

    high_success = (
        high_group["Result15Before7"] == "SUCCESS"
    ).mean() * 100

    low_rule_return = low_group["RuleBasedReturn"].mean()
    high_rule_return = high_group["RuleBasedReturn"].mean()

    # For downside counts, low values are better.
    if feature in DOWNSIDE_FEATURES:
        success_lift = low_success - high_success
        rule_return_lift = low_rule_return - high_rule_return
    else:
        success_lift = high_success - low_success
        rule_return_lift = high_rule_return - low_rule_return

    ranking_rows.append(
        {
            "Feature": feature,
            "Trades": len(data),
            "LowSuccess15Before7Pct": low_success,
            "HighSuccess15Before7Pct": high_success,
            "SuccessLiftPct": success_lift,
            "LowRuleBasedReturnPct": low_rule_return,
            "HighRuleBasedReturnPct": high_rule_return,
            "RuleReturnLiftPct": rule_return_lift,
        }
    )

    for bucket_name, group in data.groupby(
        "Bucket",
        observed=False,
    ):
        bucket_rows.append(
            {
                "Feature": feature,
                "Bucket": str(bucket_name),
                "BreakoutTrades": len(group),
                "Success15Before7Pct": (
                    group["Result15Before7"] == "SUCCESS"
                ).mean() * 100,
                "Stop7Before15Pct": (
                    group["Result15Before7"] == "STOPPED_OUT"
                ).mean() * 100,
                "Target18BeforeStop7Pct": (
                    group["Target18BeforeStop7"] == "SUCCESS"
                ).mean() * 100,
                "Target25BeforeStop7Pct": (
                    group["Target25BeforeStop7"] == "SUCCESS"
                ).mean() * 100,
                "Target30BeforeStop7Pct": (
                    group["Target30BeforeStop7"] == "SUCCESS"
                ).mean() * 100,
                "Average10DayReturnPct": group[
                    "CloseReturn10D"
                ].mean(),
                "Average20DayReturnPct": group[
                    "CloseReturn20D"
                ].mean(),
                "AverageRuleBasedReturnPct": group[
                    "RuleBasedReturn"
                ].mean(),
            }
        )

buckets = pd.DataFrame(bucket_rows)
rankings = pd.DataFrame(ranking_rows)

if buckets.empty:
    buckets = pd.DataFrame(
        columns=[
            "Feature",
            "Bucket",
            "BreakoutTrades",
            "Success15Before7Pct",
            "Stop7Before15Pct",
            "Target18BeforeStop7Pct",
            "Target25BeforeStop7Pct",
            "Target30BeforeStop7Pct",
            "Average10DayReturnPct",
            "Average20DayReturnPct",
            "AverageRuleBasedReturnPct",
        ]
    )

if rankings.empty:
    rankings = pd.DataFrame(
        columns=[
            "Feature",
            "Trades",
            "LowSuccess15Before7Pct",
            "HighSuccess15Before7Pct",
            "SuccessLiftPct",
            "LowRuleBasedReturnPct",
            "HighRuleBasedReturnPct",
            "RuleReturnLiftPct",
        ]
    )
else:
    rankings = rankings.sort_values(
        ["SuccessLiftPct", "RuleReturnLiftPct"],
        ascending=False,
    )

safe_csv(buckets.round(3), OUTPUT_BUCKETS)
safe_csv(rankings.round(3), OUTPUT_RANKINGS)


# ============================================================
# TEST COMPOSITE-SCORE MODELS
# ============================================================

print("Testing composite-score candidates...")

candidate_rows = []
walkforward_rows = []

if not rankings.empty:
    positive_features = rankings[
        rankings["SuccessLiftPct"] > 0
    ]["Feature"].tolist()

    top_features = positive_features[:8]

    candidate_models = {}

    for number_of_features in [3, 4, 5, 6]:
        if len(top_features) >= number_of_features:
            candidate_models[
                f"Top_{number_of_features}_Equal"
            ] = top_features[:number_of_features]

    for size in [3, 4, 5]:
        for combo in combinations(top_features[:7], size):
            model_name = "Mix_" + "_".join(combo)
            candidate_models[model_name] = list(combo)

    for model_name, model_features in candidate_models.items():
        if len(model_features) < 2:
            continue

        work = trades.copy()
        composite = pd.Series(0.0, index=work.index)

        for feature in model_features:
            feature_score = percentile_score(
                work[feature].fillna(work[feature].median())
            )

            if feature in DOWNSIDE_FEATURES:
                feature_score = 100 - feature_score

            composite = composite + feature_score

        work["CompositeScore"] = composite / len(model_features)

        for threshold in [50, 55, 60, 65, 70, 75, 80]:
            selected = work[
                work["CompositeScore"] >= threshold
            ].copy()

            if len(selected) < 100:
                continue

            candidate_rows.append(
                {
                    "Model": model_name,
                    "Features": " | ".join(model_features),
                    "Threshold": threshold,
                    "Trades": len(selected),
                    "Success15Before7Pct": (
                        selected["Result15Before7"] == "SUCCESS"
                    ).mean() * 100,
                    "Stop7Before15Pct": (
                        selected["Result15Before7"] == "STOPPED_OUT"
                    ).mean() * 100,
                    "AverageRuleBasedReturnPct": selected[
                        "RuleBasedReturn"
                    ].mean(),
                    "Average20DayReturnPct": selected[
                        "CloseReturn20D"
                    ].mean(),
                }
            )

            # Chronological walk-forward test:
            # score is calculated from data available on each entry date;
            # results are evaluated in later sequential periods.
            unique_dates = sorted(work["Date"].dropna().unique())

            if len(unique_dates) < 100:
                continue

            number_of_folds = 5
            fold_size = len(unique_dates) // number_of_folds

            for fold_number in range(1, number_of_folds):
                train_end = unique_dates[fold_size * fold_number - 1]

                test_end_index = min(
                    fold_size * (fold_number + 1) - 1,
                    len(unique_dates) - 1,
                )

                test_end = unique_dates[test_end_index]

                test_data = work[
                    (work["Date"] > train_end)
                    & (work["Date"] <= test_end)
                    & (work["CompositeScore"] >= threshold)
                ].copy()

                if len(test_data) < 50:
                    continue

                walkforward_rows.append(
                    {
                        "Model": model_name,
                        "Features": " | ".join(model_features),
                        "Threshold": threshold,
                        "Fold": fold_number,
                        "TrainEnd": train_end,
                        "TestEnd": test_end,
                        "Trades": len(test_data),
                        "Success15Before7Pct": (
                            test_data["Result15Before7"] == "SUCCESS"
                        ).mean() * 100,
                        "Stop7Before15Pct": (
                            test_data["Result15Before7"] == "STOPPED_OUT"
                        ).mean() * 100,
                        "AverageRuleBasedReturnPct": test_data[
                            "RuleBasedReturn"
                        ].mean(),
                        "Average20DayReturnPct": test_data[
                            "CloseReturn20D"
                        ].mean(),
                    }
                )

candidates = pd.DataFrame(candidate_rows)
walkforward = pd.DataFrame(walkforward_rows)

if candidates.empty:
    candidates = pd.DataFrame(
        columns=[
            "Model",
            "Features",
            "Threshold",
            "Trades",
            "Success15Before7Pct",
            "Stop7Before15Pct",
            "AverageRuleBasedReturnPct",
            "Average20DayReturnPct",
        ]
    )

if walkforward.empty:
    walkforward = pd.DataFrame(
        columns=[
            "Model",
            "Features",
            "Threshold",
            "Fold",
            "TrainEnd",
            "TestEnd",
            "Trades",
            "Success15Before7Pct",
            "Stop7Before15Pct",
            "AverageRuleBasedReturnPct",
            "Average20DayReturnPct",
        ]
    )

safe_csv(candidates.round(3), OUTPUT_CANDIDATES)
safe_csv(walkforward.round(3), OUTPUT_WALKFORWARD)


# ============================================================
# SELECT THE BEST WALK-FORWARD MODEL
# ============================================================

best_model_name = "No valid model"
best_features = []
best_threshold = np.nan

if not walkforward.empty:
    walkforward_summary = (
        walkforward.groupby(
            ["Model", "Features", "Threshold"],
            as_index=False,
        )
        .agg(
            Folds=("Fold", "nunique"),
            TotalTrades=("Trades", "sum"),
            AverageSuccessPct=("Success15Before7Pct", "mean"),
            AverageStopPct=("Stop7Before15Pct", "mean"),
            AverageRuleBasedReturnPct=(
                "AverageRuleBasedReturnPct",
                "mean",
            ),
            Average20DayReturnPct=(
                "Average20DayReturnPct",
                "mean",
            ),
        )
    )

    walkforward_summary = walkforward_summary[
        (walkforward_summary["Folds"] >= 2)
        & (walkforward_summary["TotalTrades"] >= 200)
    ].copy()

    if not walkforward_summary.empty:
        # Higher success and return is good.
        # Higher stop rate is bad.
        walkforward_summary["QualityScore"] = (
            walkforward_summary["AverageRuleBasedReturnPct"]
            + 0.05 * walkforward_summary["AverageSuccessPct"]
            - 0.03 * walkforward_summary["AverageStopPct"]
        )

        walkforward_summary = walkforward_summary.sort_values(
            "QualityScore",
            ascending=False,
        )

        best_row = walkforward_summary.iloc[0]

        best_model_name = best_row["Model"]
        best_features = best_row["Features"].split(" | ")
        best_threshold = best_row["Threshold"]

        safe_csv(
            walkforward_summary.round(3),
            OUTPUT_WALKFORWARD,
        )


# ============================================================
# DAILY COMPOSITE SCORE AND ACTION ZONES
# ============================================================

daily_score = breadth_for_merge.copy()

if best_features:
    score = pd.Series(0.0, index=daily_score.index)
    feature_count = 0

    for feature in best_features:
        if feature not in daily_score.columns:
            continue

        feature_score = percentile_score(
            daily_score[feature].fillna(
                daily_score[feature].median()
            )
        )

        if feature in DOWNSIDE_FEATURES:
            feature_score = 100 - feature_score

        score = score + feature_score
        feature_count += 1

    if feature_count > 0:
        daily_score["CompositeScore"] = score / feature_count
    else:
        daily_score["CompositeScore"] = np.nan
else:
    daily_score["CompositeScore"] = np.nan

daily_score["BestModel"] = best_model_name
daily_score["RecommendedThreshold"] = best_threshold

daily_score["ActionZone"] = pd.cut(
    daily_score["CompositeScore"],
    bins=[-1, 35, 50, 65, 80, 101],
    labels=[
        "Risk-off",
        "Defensive",
        "Selective",
        "Constructive",
        "Aggressive",
    ],
)

safe_csv(daily_score.round(3), OUTPUT_DAILY_SCORE)


# ============================================================
# ACTION-ZONE BACKTEST
# ============================================================

if best_features:
    trade_scores = trades.merge(
        daily_score[
            ["Date", "CompositeScore", "ActionZone"]
        ],
        on="Date",
        how="left",
    )

    zones = (
        trade_scores.groupby(
            "ActionZone",
            observed=False,
        )
        .agg(
            Trades=("Symbol", "count"),
            Success15Before7Pct=(
                "Result15Before7",
                lambda values: (values == "SUCCESS").mean() * 100,
            ),
            Stop7Before15Pct=(
                "Result15Before7",
                lambda values: (
                    values == "STOPPED_OUT"
                ).mean()
                * 100,
            ),
            AverageRuleBasedReturnPct=(
                "RuleBasedReturn",
                "mean",
            ),
            Average20DayReturnPct=(
                "CloseReturn20D",
                "mean",
            ),
        )
        .reset_index()
    )
else:
    zones = pd.DataFrame(
        columns=[
            "ActionZone",
            "Trades",
            "Success15Before7Pct",
            "Stop7Before15Pct",
            "AverageRuleBasedReturnPct",
            "Average20DayReturnPct",
        ]
    )

safe_csv(zones.round(3), OUTPUT_ZONES)


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = pd.DataFrame(
    [
        {
            "Metric": "Completed breakout trades",
            "Value": len(trades),
        },
        {
            "Metric": "Overall 15% before 7% success rate (%)",
            "Value": (
                trades["Result15Before7"] == "SUCCESS"
            ).mean()
            * 100,
        },
        {
            "Metric": "Overall 7% stop before 15% rate (%)",
            "Value": (
                trades["Result15Before7"] == "STOPPED_OUT"
            ).mean()
            * 100,
        },
        {
            "Metric": "Overall average rule-based return (%)",
            "Value": trades["RuleBasedReturn"].mean(),
        },
        {
            "Metric": "Available breadth features",
            "Value": len(available_features),
        },
        {
            "Metric": "Usable breadth features",
            "Value": len(usable_features),
        },
        {
            "Metric": "Breadth merge coverage (%)",
            "Value": merge_coverage,
        },
        {
            "Metric": "Best walk-forward model",
            "Value": best_model_name,
        },
        {
            "Metric": "Best model features",
            "Value": " | ".join(best_features),
        },
        {
            "Metric": "Best model score threshold",
            "Value": best_threshold,
        },
    ]
)

safe_csv(summary.round(3), OUTPUT_SUMMARY)

print()
print("=" * 70)
print("RESEARCH BACKTEST COMPLETED")
print("=" * 70)
print(f"Completed breakout trades: {len(trades):,}")
print(f"Usable breadth features: {len(usable_features)}")
print(f"Best composite model: {best_model_name}")
print(f"Recommended threshold: {best_threshold}")
print("All research CSVs have been updated in the Research folder.")

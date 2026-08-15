from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd


# ============================================================
# FILES AND CORE TRADE RULE
# ============================================================

RESEARCH_FOLDER = Path(__file__).resolve().parent
REPO_FOLDER = RESEARCH_FOLDER.parent

PARQUET_FILE = REPO_FOLDER / "nse_6yr_historical.parquet"
BREADTH_FILE = REPO_FOLDER / "historical_breadth_regime_6yr.csv"

VCP_TIGHTNESS_LIMIT = 0.04
STOP_LOSS_PCT = 7.0
TARGET_PCT = 15.0
MAX_HOLDING_DAYS = 30

OUTPUT_AUDIT = RESEARCH_FOLDER / "research_input_audit.csv"
OUTPUT_TRADES = RESEARCH_FOLDER / "research_breakout_trades_enriched.csv"
OUTPUT_BUCKETS = RESEARCH_FOLDER / "research_feature_buckets.csv"
OUTPUT_RANKINGS = RESEARCH_FOLDER / "research_feature_rankings.csv"
OUTPUT_CANDIDATES = RESEARCH_FOLDER / "research_candidate_scores.csv"
OUTPUT_WALKFORWARD = RESEARCH_FOLDER / "research_walkforward_results.csv"
OUTPUT_DAILY_SCORE = RESEARCH_FOLDER / "research_best_composite_score_daily.csv"
OUTPUT_ZONES = RESEARCH_FOLDER / "research_score_action_zones.csv"
OUTPUT_SUMMARY = RESEARCH_FOLDER / "research_run_summary.csv"

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

DOWNSIDE_FEATURES = {
    "Rolling3DDown4",
    "Down251MCount",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_dates(series):
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def first_hit(future_data, entry_price, target_pct, stop_pct):
    target_price = entry_price * (1 + target_pct / 100)
    stop_price = entry_price * (1 - stop_pct / 100)

    for day_number, (_, row) in enumerate(future_data.iterrows(), start=1):
        target_hit = float(row["High"]) >= target_price
        stop_hit = float(row["Low"]) <= stop_price

        if target_hit and stop_hit:
            return "AMBIGUOUS_SAME_DAY", day_number

        if target_hit:
            return "SUCCESS", day_number

        if stop_hit:
            return "STOPPED_OUT", day_number

    return "TIMEOUT", np.nan


def percentile_rank(series):
    return series.rank(pct=True, method="average") * 100


def safe_write_csv(dataframe, path):
    dataframe.to_csv(path, index=False)


# ============================================================
# LOAD AND VALIDATE INPUTS
# ============================================================

print("Loading inputs...")

if not PARQUET_FILE.exists():
    raise FileNotFoundError(f"Missing parquet file: {PARQUET_FILE}")

if not BREADTH_FILE.exists():
    raise FileNotFoundError(f"Missing breadth CSV: {BREADTH_FILE}")

prices = pd.read_parquet(PARQUET_FILE)
breadth = pd.read_csv(BREADTH_FILE)

prices["Date"] = clean_dates(prices["Date"])
breadth["Date"] = clean_dates(breadth["Date"])

prices = prices.dropna(subset=["Date", "Symbol", "Close", "High", "Low", "Volume"])
breadth = breadth.dropna(subset=["Date"])

prices = prices.sort_values(["Symbol", "Date"]).reset_index(drop=True)
breadth = breadth.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)

required_price_columns = {"Date", "Symbol", "Open", "High", "Low", "Close", "Volume"}
missing_price_columns = required_price_columns - set(prices.columns)

if missing_price_columns:
    raise ValueError(
        "Parquet is missing these columns: "
        + ", ".join(sorted(missing_price_columns))
    )


# ============================================================
# BUILD EXISTING VCP-STYLE BREAKOUT ENTRIES
# ============================================================

print("Building breakout entries...")

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

prices["PriorDayTight"] = prices.groupby("Symbol")["VCPTightness"].shift(1)

prices["IsBreakout"] = (
    (prices["Close"] >= prices["Close20DHigh"])
    & prices["VolumeSurge"]
    & prices["PriorDayTight"].fillna(False)
)


# ============================================================
# CREATE TRADES
# ============================================================

print("Calculating trade outcomes...")

trade_records = []

for symbol, stock in prices.groupby("Symbol", sort=False):
    stock = stock.sort_values("Date").reset_index(drop=True)

    breakout_indexes = stock.index[stock["IsBreakout"]].tolist()

    for entry_index in breakout_indexes:
        future = stock.iloc[
            entry_index + 1 : entry_index + 1 + MAX_HOLDING_DAYS
        ].copy()

        if len(future) < MAX_HOLDING_DAYS:
            continue

        entry = stock.iloc[entry_index]
        entry_price = float(entry["Close"])

        result_15, decision_day_15 = first_hit(
            future,
            entry_price,
            TARGET_PCT,
            STOP_LOSS_PCT,
        )

        record = {
            "Date": entry["Date"],
            "Symbol": symbol,
            "EntryPrice": entry_price,
            "Result15Before7": result_15,
            "DecisionDay15Before7": decision_day_15,
        }

        for target in [18, 25, 30]:
            result, day = first_hit(
                future,
                entry_price,
                target,
                STOP_LOSS_PCT,
            )

            record[f"Target{target}BeforeStop7"] = result
            record[f"Target{target}DecisionDay"] = day

        for holding_day in [5, 10, 15, 20, 30]:
            record[f"CloseReturn{holding_day}D"] = (
                float(stock.iloc[entry_index + holding_day]["Close"])
                / entry_price
                - 1
            ) * 100

        if result_15 == "SUCCESS":
            record["RuleBasedReturn"] = TARGET_PCT
        elif result_15 == "STOPPED_OUT":
            record["RuleBasedReturn"] = -STOP_LOSS_PCT
        else:
            record["RuleBasedReturn"] = (
                float(future.iloc[-1]["Close"]) / entry_price - 1
            ) * 100

        trade_records.append(record)

trades = pd.DataFrame(trade_records)

if trades.empty:
    raise ValueError("No completed 30-day breakout trade records were created.")


# ============================================================
# PREPARE AND MERGE BREADTH DATA
# ============================================================

print("Merging market-breadth features...")

if {"T3Breakouts", "T3Wins"}.issubset(breadth.columns):
    breadth["FollowThroughRate"] = np.where(
        breadth["T3Breakouts"] > 0,
        breadth["T3Wins"] / breadth["T3Breakouts"] * 100,
        np.nan,
    )

available_features = [
    feature
    for feature in FEATURES
    if feature in breadth.columns
]

breadth_for_merge = breadth[
    ["Date"] + available_features
].copy()

trades = trades.merge(
    breadth_for_merge,
    on="Date",
    how="left",
)

merge_coverage = (
    trades[available_features].notna().any(axis=1).mean() * 100
    if available_features
    else 0
)

usable_features = [
    feature
    for feature in available_features
    if trades[feature].notna().sum() >= 100
]


# ============================================================
# WRITE INPUT AUDIT FIRST
# ============================================================

audit_rows = [
    {
        "Item": "Price parquet rows",
        "Value": len(prices),
    },
    {
        "Item": "Breadth CSV rows",
        "Value": len(breadth),
    },
    {
        "Item": "Completed breakout trade records",
        "Value": len(trades),
    },
    {
        "Item": "Price data first date",
        "Value": prices["Date"].min(),
    },
    {
        "Item": "Price data last date",
        "Value": prices["Date"].max(),
    },
    {
        "Item": "Breadth data first date",
        "Value": breadth["Date"].min(),
    },
    {
        "Item": "Breadth data last date",
        "Value": breadth["Date"].max(),
    },
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
            "Item": f"Non-empty trade rows for {feature}",
            "Value": int(trades[feature].notna().sum()),
        }
    )

audit = pd.DataFrame(audit_rows)
safe_write_csv(audit, OUTPUT_AUDIT)

trades.to_csv(OUTPUT_TRADES, index=False)

print(f"Available features: {len(available_features)}")
print(f"Usable features: {len(usable_features)}")
print(f"Breadth merge coverage: {merge_coverage:.2f}%")


# ============================================================
# FEATURE BUCKETS AND FEATURE RANKINGS
# ============================================================

bucket_rows = []
ranking_rows = []

for feature in usable_features:
    valid = trades.dropna(subset=[feature]).copy()

    try:
        valid["Bucket"] = pd.qcut(
            valid[feature].rank(method="first"),
            q=3,
            labels=["Low", "Medium", "High"],
        )
    except ValueError:
        continue

    low_group = valid[valid["Bucket"] == "Low"]
    high_group = valid[valid["Bucket"] == "High"]

    high_success = (
        high_group["Result15Before7"] == "SUCCESS"
    ).mean() * 100

    low_success = (
        low_group["Result15Before7"] == "SUCCESS"
    ).mean() * 100

    if feature in DOWNSIDE_FEATURES:
        lift = low_success - high_success
    else:
        lift = high_success - low_success

    ranking_rows.append(
        {
            "Feature": feature,
            "Trades": len(valid),
            "LowSuccess15Before7Pct": low_success,
            "HighSuccess15Before7Pct": high_success,
            "SuccessLiftPct": lift,
            "LowRuleBasedReturnPct": low_group["RuleBasedReturn"].mean(),
            "HighRuleBasedReturnPct": high_group["RuleBasedReturn"].mean(),
        }
    )

    for bucket_name, group in valid.groupby("Bucket", observed=False):
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
                "Average10DayReturnPct": group["CloseReturn10D"].mean(),
                "Average20DayReturnPct": group["CloseReturn20D"].mean(),
                "AverageRuleBasedReturnPct": group["RuleBasedReturn"].mean(),
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
        ]
    )
else:
    rankings = rankings.sort_values(
        "SuccessLiftPct",
        ascending=False,
    )

safe_write_csv(buckets.round(3), OUTPUT_BUCKETS)
safe_write_csv(rankings.round(3), OUTPUT_RANKINGS)


# ============================================================
# BUILD CANDIDATE COMPOSITE SCORES
# ============================================================

print("Testing composite-score candidates...")

candidate_rows = []
walkforward_rows = []

if not rankings.empty:
    top_features = rankings["Feature"].head(8).tolist()

    candidate_mixes = {
        "Top_3_equal": top_features[:3],
        "Top_4_equal": top_features[:4],
        "Top_5_equal": top_features[:5],
        "Top_6_equal": top_features[:6],
    }

    for size in [3, 4, 5]:
        for combo in combinations(top_features[:7], size):
            name = "Mix_" + "_".join(combo)
            candidate_mixes[name] = list(combo)

    for model_name, model_features in candidate_mixes.items():
        model_features = [
            feature
            for feature in model_features
            if feature in usable_features
        ]

        if len(model_features) < 2:
            continue

        score = pd.Series(0.0, index=trades.index)

        for feature in model_features:
            normalized = percentile_rank(
                trades[feature].fillna(trades[feature].median())
            )

            if feature in DOWNSIDE_FEATURES:
                normalized = 100 - normalized

            score = score + normalized

        score = score / len(model_features)

        temporary = trades.copy()
        temporary["CompositeScore"] = score

        for threshold in [50, 55, 60, 65, 70, 75, 80]:
            selected = temporary[
                temporary["CompositeScore"] >= threshold
            ]

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

            unique_dates = sorted(temporary["Date"].dropna().unique())

            if len(unique_dates) < 100:
                continue

            split_count = 5
            split_size = len(unique_dates) // split_count

            for fold in range(1, split_count):
                train_end = unique_dates[split_size * fold - 1]
                test_end_index = min(
                    split_size * (fold + 1) - 1,
                    len(unique_dates) - 1,
                )
                test_end = unique_dates[test_end_index]

                test = temporary[
                    (temporary["Date"] > train_end)
                    & (temporary["Date"] <= test_end)
                    & (temporary["CompositeScore"] >= threshold)
                ]

                if len(test) < 50:
                    continue

                walkforward_rows.append(
                    {
                        "Model": model_name,
                        "Features": " | ".join(model_features),
                        "Threshold": threshold,
                        "Fold": fold,
                        "TrainEnd": train_end,
                        "TestEnd": test_end,
                        "Trades": len(test),
                        "Success15Before7Pct": (
                            test["Result15Before7"] == "SUCCESS"
                        ).mean() * 100,
                        "Stop7Before15Pct": (
                            test["Result15Before7"] == "STOPPED_OUT"
                        ).mean() * 100,
                        "AverageRuleBasedReturnPct": test[
                            "RuleBasedReturn"
                        ].mean(),
                        "Average20DayReturnPct": test[
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

safe_write_csv(candidates.round(3), OUTPUT_CANDIDATES)
safe_write_csv(walkforward.round(3), OUTPUT_WALKFORWARD)


# ============================================================
# CHOOSE BEST WALK-FORWARD MODEL
# ============================================================

best_model_name = "No valid model"
best_threshold = np.nan
best_features = []

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
            AverageRuleReturnPct=(
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
        walkforward_summary["QualityScore"] = (
            walkforward_summary["AverageRuleReturnPct"]
            + walkforward_summary["AverageSuccessPct"] * 0.05
            - walkforward_summary["AverageStopPct"] * 0.03
        )

        walkforward_summary = walkforward_summary.sort_values(
            "QualityScore",
            ascending=False,
        )

        best = walkforward_summary.iloc[0]
        best_model_name = best["Model"]
        best_threshold = best["Threshold"]
        best_features = best["Features"].split(" | ")

        safe_write_csv(
            walkforward_summary.round(3),
            OUTPUT_WALKFORWARD,
        )


# ============================================================
# DAILY FINAL SCORE AND ACTION ZONES
# ============================================================

daily_features = breadth_for_merge.copy()

if best_features:
    daily_score = pd.Series(
        0.0,
        index=daily_features.index,
    )

    valid_feature_count = 0

    for feature in best_features:
        if feature not in daily_features.columns:
            continue

        normalized = percentile_rank(
            daily_features[feature].fillna(
                daily_features[feature].median()
            )
        )

        if feature in DOWNSIDE_FEATURES:
            normalized = 100 - normalized

        daily_score = daily_score + normalized
        valid_feature_count += 1

    if valid_feature_count > 0:
        daily_features["CompositeScore"] = (
            daily_score / valid_feature_count
        )
    else:
        daily_features["CompositeScore"] = np.nan
else:
    daily_features["CompositeScore"] = np.nan

daily_features["BestModel"] = best_model_name
daily_features["RecommendedThreshold"] = best_threshold

daily_features["ActionZone"] = pd.cut(
    daily_features["CompositeScore"],
    bins=[-1, 35, 50, 65, 80, 101],
    labels=[
        "Risk-off",
        "Defensive",
        "Selective",
        "Constructive",
        "Aggressive",
    ],
)

safe_write_csv(
    daily_features.round(3),
    OUTPUT_DAILY_SCORE,
)

if best_features:
    trade_scores = trades.merge(
        daily_features[
            ["Date", "CompositeScore", "ActionZone"]
        ],
        on="Date",
        how="left",
    )

    zones = (
        trade_scores.groupby("ActionZone", observed=False)
        .agg(
            Trades=("Symbol", "count"),
            Success15Before7Pct=(
                "Result15Before7",
                lambda values: (
                    values == "SUCCESS"
                ).mean()
                * 100,
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

safe_write_csv(
    zones.round(3),
    OUTPUT_ZONES,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

summary_rows = [
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

summary = pd.DataFrame(summary_rows)
safe_write_csv(summary.round(3), OUTPUT_SUMMARY)

print()
print("RESEARCH BACKTEST COMPLETE")
print(f"Breakout trades: {len(trades):,}")
print(f"Usable breadth features: {len(usable_features)}")
print(f"Best model: {best_model_name}")
print("All outputs are saved in the Research folder.")

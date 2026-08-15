from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# FILE LOCATIONS
# ============================================================

RESEARCH_FOLDER = Path(__file__).resolve().parent

INPUT_FILE = RESEARCH_FOLDER / "research_breakout_trades_enriched.csv"

OUTPUT_SCORE_BANDS = RESEARCH_FOLDER / "validation_score_bands.csv"
OUTPUT_YEARLY = RESEARCH_FOLDER / "validation_score_yearly.csv"
OUTPUT_SUMMARY = RESEARCH_FOLDER / "validation_score_summary.csv"
OUTPUT_TRADE_SCORES = RESEARCH_FOLDER / "validation_trade_scores.csv"


# ============================================================
# SCORE DEFINITIONS
# ============================================================

SCORE_MODELS = {
    "Your_35_30_20_15": {
        "MidPct50EMA": 0.35,
        "SmallPct50EMA": 0.30,
        "LargePct50EMA": 0.20,
        "PctAbove200EMA": 0.15,
    },
    "Equal_25_Each": {
        "MidPct50EMA": 0.25,
        "SmallPct50EMA": 0.25,
        "LargePct50EMA": 0.25,
        "PctAbove200EMA": 0.25,
    },
    "Leadership_50_30_20": {
        "MidPct50EMA": 0.50,
        "SmallPct50EMA": 0.30,
        "LargePct50EMA": 0.20,
    },
    "WalkForward_Model": {
        "LargePct50EMA": 1 / 3,
        "PctAbove200EMA": 1 / 3,
        "SmallPct200EMA": 1 / 3,
    },
    "Mid50EMA_Only": {
        "MidPct50EMA": 1.00,
    },
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def percentile_score(series):
    """
    Turns a raw indicator into a 0-100 historical percentile score.
    Higher raw values become higher scores.
    """
    return series.rank(pct=True, method="average") * 100


def create_composite_score(data, weights):
    """
    Builds one 0-100 score using the supplied weights.
    Missing values are replaced by the historical median.
    """
    score = pd.Series(0.0, index=data.index)
    total_weight = 0.0

    for feature, weight in weights.items():
        if feature not in data.columns:
            continue

        values = data[feature].fillna(data[feature].median())
        score = score + percentile_score(values) * weight
        total_weight += weight

    if total_weight == 0:
        return pd.Series(np.nan, index=data.index)

    return score / total_weight


def add_outcome_columns(data):
    """
    Creates easy-to-read True/False outcome columns.
    """
    data["Success15"] = data["Result15Before7"] == "SUCCESS"
    data["Stop7"] = data["Result15Before7"] == "STOPPED_OUT"
    data["Success18"] = data["Target18BeforeStop7"] == "SUCCESS"
    data["Success25"] = data["Target25BeforeStop7"] == "SUCCESS"
    data["Success30"] = data["Target30BeforeStop7"] == "SUCCESS"

    return data


# ============================================================
# LOAD DATA
# ============================================================

print("Loading completed breakout research data...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "Cannot find research_breakout_trades_enriched.csv. "
        "Run the main backtest workflow first."
    )

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).copy()
df["Year"] = df["Date"].dt.year

required_columns = [
    "Result15Before7",
    "Target18BeforeStop7",
    "Target25BeforeStop7",
    "Target30BeforeStop7",
    "RuleBasedReturn",
    "CloseReturn10D",
    "CloseReturn20D",
    "CloseReturn30D",
]

missing = [column for column in required_columns if column not in df.columns]

if missing:
    raise ValueError(
        "Missing required backtest columns: " + ", ".join(missing)
    )

df = add_outcome_columns(df)

print(f"Loaded {len(df):,} completed breakout trades.")


# ============================================================
# TEST EACH SCORE MODEL
# ============================================================

score_band_rows = []
yearly_rows = []
summary_rows = []

for model_name, weights in SCORE_MODELS.items():
    print(f"Testing: {model_name}")

    model_features = list(weights.keys())

    missing_features = [
        feature for feature in model_features if feature not in df.columns
    ]

    if missing_features:
        print(
            f"Skipping {model_name}. Missing features: "
            + ", ".join(missing_features)
        )
        continue

    work = df.copy()

    work["CompositeScore"] = create_composite_score(
        work,
        weights,
    )

    # Five score ranges: 0-20, 20-40, 40-60, 60-80, 80-100.
    work["ScoreBand"] = pd.cut(
        work["CompositeScore"],
        bins=[-1, 20, 40, 60, 80, 101],
        labels=["0-20", "20-40", "40-60", "60-80", "80-100"],
    )

    # Overall score-model summary.
    high_score = work[work["CompositeScore"] >= 75].copy()
    low_score = work[work["CompositeScore"] < 40].copy()

    summary_rows.append(
        {
            "Model": model_name,
            "Weights": str(weights),
            "TotalTrades": len(work),
            "AllTradesSuccess15Pct": work["Success15"].mean() * 100,
            "AllTradesStop7Pct": work["Stop7"].mean() * 100,
            "AllTradesRuleReturnPct": work["RuleBasedReturn"].mean(),
            "Score75PlusTrades": len(high_score),
            "Score75PlusSuccess15Pct": (
                high_score["Success15"].mean() * 100
                if len(high_score) > 0 else np.nan
            ),
            "Score75PlusStop7Pct": (
                high_score["Stop7"].mean() * 100
                if len(high_score) > 0 else np.nan
            ),
            "Score75PlusRuleReturnPct": (
                high_score["RuleBasedReturn"].mean()
                if len(high_score) > 0 else np.nan
            ),
            "Below40Trades": len(low_score),
            "Below40Success15Pct": (
                low_score["Success15"].mean() * 100
                if len(low_score) > 0 else np.nan
            ),
            "Below40Stop7Pct": (
                low_score["Stop7"].mean() * 100
                if len(low_score) > 0 else np.nan
            ),
            "Below40RuleReturnPct": (
                low_score["RuleBasedReturn"].mean()
                if len(low_score) > 0 else np.nan
            ),
        }
    )

    # Score-band performance.
    for band_name, group in work.groupby("ScoreBand", observed=False):
        if len(group) == 0:
            continue

        score_band_rows.append(
            {
                "Model": model_name,
                "Weights": str(weights),
                "ScoreBand": str(band_name),
                "Trades": len(group),
                "Success15Before7Pct": group["Success15"].mean() * 100,
                "Stop7Before15Pct": group["Stop7"].mean() * 100,
                "Success18Before7Pct": group["Success18"].mean() * 100,
                "Success25Before7Pct": group["Success25"].mean() * 100,
                "Success30Before7Pct": group["Success30"].mean() * 100,
                "AverageRuleBasedReturnPct": group[
                    "RuleBasedReturn"
                ].mean(),
                "MedianRuleBasedReturnPct": group[
                    "RuleBasedReturn"
                ].median(),
                "Average10DayReturnPct": group[
                    "CloseReturn10D"
                ].mean(),
                "Average20DayReturnPct": group[
                    "CloseReturn20D"
                ].mean(),
                "Average30DayReturnPct": group[
                    "CloseReturn30D"
                ].mean(),
            }
        )

    # Year-by-year high-score validation.
    for year, group in work.groupby("Year"):
        high_score_year = group[group["CompositeScore"] >= 75].copy()

        if len(high_score_year) == 0:
            continue

        yearly_rows.append(
            {
                "Model": model_name,
                "Weights": str(weights),
                "Year": year,
                "AllTrades": len(group),
                "HighScoreTrades": len(high_score_year),
                "HighScoreSuccess15Before7Pct": (
                    high_score_year["Success15"].mean() * 100
                ),
                "HighScoreStop7Before15Pct": (
                    high_score_year["Stop7"].mean() * 100
                ),
                "HighScoreSuccess18Before7Pct": (
                    high_score_year["Success18"].mean() * 100
                ),
                "HighScoreSuccess25Before7Pct": (
                    high_score_year["Success25"].mean() * 100
                ),
                "HighScoreAverageRuleReturnPct": (
                    high_score_year["RuleBasedReturn"].mean()
                ),
                "HighScoreAverage20DayReturnPct": (
                    high_score_year["CloseReturn20D"].mean()
                ),
            }
        )

    # Save every trade with all candidate score values.
    df[f"Score_{model_name}"] = work["CompositeScore"]


# ============================================================
# SAVE RESULTS
# ============================================================

score_bands = pd.DataFrame(score_band_rows)
yearly = pd.DataFrame(yearly_rows)
summary = pd.DataFrame(summary_rows)

numeric_columns = [
    column for column in summary.columns
    if column not in ["Model", "Weights"]
]

summary[numeric_columns] = summary[numeric_columns].round(3)

for dataframe in [score_bands, yearly]:
    numeric_columns = dataframe.select_dtypes(
        include=[np.number]
    ).columns.tolist()
    dataframe[numeric_columns] = dataframe[numeric_columns].round(3)

summary = summary.sort_values(
    [
        "Score75PlusRuleReturnPct",
        "Score75PlusSuccess15Pct",
    ],
    ascending=False,
)

score_bands.to_csv(OUTPUT_SCORE_BANDS, index=False)
yearly.to_csv(OUTPUT_YEARLY, index=False)
summary.to_csv(OUTPUT_SUMMARY, index=False)
df.to_csv(OUTPUT_TRADE_SCORES, index=False)

print()
print("=" * 70)
print("FIXED-WEIGHT COMPOSITE VALIDATION COMPLETE")
print("=" * 70)
print("Created:")
print("1. validation_score_summary.csv")
print("2. validation_score_bands.csv")
print("3. validation_score_yearly.csv")
print("4. validation_trade_scores.csv")

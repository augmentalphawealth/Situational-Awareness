import os

import numpy as np
import pandas as pd


print("=========================================================")
print("  EOD BREADTH & REGIME ANALYSIS ENGINE (Calibrated v2.1) ")
print("=========================================================")

PARQUET_FILE = "nse_6yr_historical.parquet"
OUTPUT_CSV = "historical_breadth_regime_6yr.csv"
TRAILING_CACHE_FILE = "trailing_cache.parquet"

if not os.path.exists(PARQUET_FILE):
    print("❌ Parquet database not found! Run fetch_6yr_history.py first.")
    raise SystemExit(1)

df = pd.read_parquet(PARQUET_FILE)

required_columns = {
    "Symbol",
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    print(
        "❌ Parquet database is missing required columns: "
        f"{sorted(missing_columns)}"
    )
    raise SystemExit(1)

df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()

df = df.dropna(subset=["Symbol", "Date"])
df = df[df["Symbol"] != ""].copy()

if df.duplicated(["Symbol", "Date"]).any():
    print("❌ Parquet database contains duplicate Symbol/Date records.")
    raise SystemExit(1)

df = df.sort_values(by=["Symbol", "Date"]).reset_index(drop=True)

# ------------------------------------------------------------------
# 1. AGE TRACKING & TURNOVER
# ------------------------------------------------------------------
group = df.groupby("Symbol", group_keys=False)

df["History_Days"] = group.cumcount() + 1
df["Prior_History_Days"] = df["History_Days"] - 1

df["Daily_Turnover"] = df["Close"] * df["Volume"]

df["Prior_Turnover_20D_Avg"] = group["Daily_Turnover"].transform(
    lambda x: x.shift(1).rolling(20, min_periods=1).mean()
)

# ------------------------------------------------------------------
# 2. ACTIVE UNIVERSE GATES
# Mature stocks: prior 20D average turnover >= ₹5 crore.
# New listings: included from their second available session until
# they reach 20 prior-history days.
# ------------------------------------------------------------------
mature_valid = (
    (df["Prior_History_Days"] >= 20)
    & (df["Prior_Turnover_20D_Avg"] >= 50_000_000)
)

new_valid = (
    (df["Prior_History_Days"] >= 1)
    & (df["Prior_History_Days"] < 20)
)

df["Active_Universe"] = (
    (mature_valid | new_valid)
    & (df["Volume"] > 0)
)

df["Turnover_45d_Avg"] = group["Daily_Turnover"].transform(
    lambda x: x.shift(1).rolling(window=45, min_periods=1).mean()
)

df["Cap_Rank"] = df.groupby("Date")["Turnover_45d_Avg"].rank(
    ascending=False,
    method="min",
)

conditions = [
    df["Cap_Rank"] <= 100,
    (df["Cap_Rank"] > 100) & (df["Cap_Rank"] <= 250),
    (df["Cap_Rank"] > 250) & (df["Cap_Rank"] <= 500),
]

df["Liquidity_Category"] = np.select(
    conditions,
    [
        "Top 100 Liq",
        "Mid 150 Liq",
        "Lower 250 Liq",
    ],
    default="Micro Liq",
)

# ------------------------------------------------------------------
# 3. EMA CALCULATIONS
# ------------------------------------------------------------------
df["EMA_20"] = group["Close"].transform(
    lambda x: x.ewm(
        span=20,
        adjust=False,
        min_periods=1,
    ).mean()
)

df["EMA_50"] = group["Close"].transform(
    lambda x: x.ewm(
        span=50,
        adjust=False,
        min_periods=1,
    ).mean()
)

df["EMA_200"] = group["Close"].transform(
    lambda x: x.ewm(
        span=200,
        adjust=False,
        min_periods=1,
    ).mean()
)

df["Valid_20_EMA"] = (
    df["Active_Universe"]
    & (df["Prior_History_Days"] >= 20)
)

df["Valid_50_EMA"] = (
    df["Active_Universe"]
    & (df["Prior_History_Days"] >= 50)
)

df["Valid_200_EMA"] = (
    df["Active_Universe"]
    & (df["Prior_History_Days"] >= 200)
)

df["Above_20_EMA"] = (
    df["Valid_20_EMA"]
    & (df["Close"] > df["EMA_20"])
)

df["Above_50_EMA"] = (
    df["Valid_50_EMA"]
    & (df["Close"] > df["EMA_50"])
)

df["Above_200_EMA"] = (
    df["Valid_200_EMA"]
    & (df["Close"] > df["EMA_200"])
)

# ------------------------------------------------------------------
# 4. DAILY PERFORMANCE & HIGHS
# ------------------------------------------------------------------
df["Prev_Close"] = group["Close"].shift(1)
df["Has_Prior_Close"] = df["Prev_Close"].notna()

df["Daily_Pct"] = group["Close"].pct_change() * 100
df["Pct_1M"] = group["Close"].pct_change(periods=21) * 100

df["Gainer"] = (
    df["Active_Universe"]
    & df["Has_Prior_Close"]
    & (df["Daily_Pct"] > 0)
)

df["Loser"] = (
    df["Active_Universe"]
    & df["Has_Prior_Close"]
    & (df["Daily_Pct"] < 0)
)

df["Rolling_52W_High"] = group["High"].transform(
    lambda x: x.shift(1).rolling(
        window=252,
        min_periods=1,
    ).max()
)

df["Rolling_52W_Low"] = group["Low"].transform(
    lambda x: x.shift(1).rolling(
        window=252,
        min_periods=1,
    ).min()
)

df["Prior_ATH"] = group["High"].transform(
    lambda x: x.shift(1).cummax()
)

df["New_52W_High"] = (
    df["Active_Universe"]
    & (df["Prior_History_Days"] >= 252)
    & (df["Close"] >= df["Rolling_52W_High"])
)

df["New_52W_Low"] = (
    df["Active_Universe"]
    & (df["Prior_History_Days"] >= 252)
    & (df["Close"] <= df["Rolling_52W_Low"])
)

df["IPO_New_High"] = (
    df["Active_Universe"]
    & (df["Prior_History_Days"] >= 1)
    & (df["Prior_History_Days"] < 252)
    & (df["Close"] >= df["Prior_ATH"])
)

df["Up_4_Pct"] = (
    df["Active_Universe"]
    & (df["Daily_Pct"] >= 4.0)
)

df["Down_4_Pct"] = (
    df["Active_Universe"]
    & (df["Daily_Pct"] <= -4.0)
)

df["Up_25_1M"] = (
    df["Active_Universe"]
    & (df["Pct_1M"] >= 25.0)
)

df["Down_25_1M"] = (
    df["Active_Universe"]
    & (df["Pct_1M"] <= -25.0)
)

# ------------------------------------------------------------------
# 5. VOLATILITY & BREAKOUTS
# ------------------------------------------------------------------
hl = df["High"] - df["Low"]
hpc = (df["High"] - df["Prev_Close"]).abs()
lpc = (df["Low"] - df["Prev_Close"]).abs()

df["TR"] = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).fillna(hl)

df["ATR_14"] = group["TR"].transform(
    lambda x: x.ewm(
        alpha=1 / 14,
        min_periods=1,
        adjust=False,
    ).mean()
)

df["Vol_20D_Avg"] = group["Volume"].transform(
    lambda x: x.shift(1).rolling(
        20,
        min_periods=1,
    ).mean()
)

df["Max_20D_Prior"] = group["High"].transform(
    lambda x: x.shift(1).rolling(
        20,
        min_periods=1,
    ).max()
)

df["Listing_Day_High"] = group["High"].transform("first")

df["VCP_Tightness"] = (
    df["ATR_14"] / df["Close"]
) < 0.04

df["Volume_Surge"] = (
    df["Volume"]
    > (df["Vol_20D_Avg"] * 1.5)
)

prior_tight = (
    group["VCP_Tightness"]
    .shift(1)
    .astype(float)
    .fillna(0.0)
    .astype(bool)
)

df["Mature_Breakout"] = (
    (df["Prior_History_Days"] >= 20)
    & (df["Close"] > df["Max_20D_Prior"])
)

df["IPO_Breakout"] = (
    (df["Prior_History_Days"] >= 1)
    & (df["Prior_History_Days"] < 20)
    & (df["Close"] > df["Listing_Day_High"])
)

df["Is_Breakout"] = (
    df["Active_Universe"]
    & (df["Mature_Breakout"] | df["IPO_Breakout"])
    & df["Volume_Surge"]
    & prior_tight
)

df["Is_Breakout_3d_ago"] = (
    group["Is_Breakout"]
    .shift(3)
    .astype(float)
    .fillna(0.0)
    .astype(bool)
)

df["Close_3d_ago"] = group["Close"].shift(3)

df["Follow_Through_Win"] = (
    df["Is_Breakout_3d_ago"]
    & (df["Close"] > df["Close_3d_ago"])
)

df["Up_Volume"] = np.where(
    df["Gainer"],
    df["Daily_Turnover"],
    0,
)

df["Down_Volume"] = np.where(
    df["Loser"],
    df["Daily_Turnover"],
    0,
)


def get_liq_breadth(data, liq_name, prefix):
    liq_df = data[
        data["Liquidity_Category"] == liq_name
    ]

    aggregated = liq_df.groupby("Date").agg(
        Valid_20=("Valid_20_EMA", "sum"),
        Valid_50=("Valid_50_EMA", "sum"),
        Valid_200=("Valid_200_EMA", "sum"),
        Above_20=("Above_20_EMA", "sum"),
        Above_50=("Above_50_EMA", "sum"),
        Above_200=("Above_200_EMA", "sum"),
    ).reset_index()

    aggregated[f"{prefix}_Pct_20_EMA"] = np.where(
        aggregated["Valid_20"] > 0,
        (aggregated["Above_20"] / aggregated["Valid_20"]) * 100,
        np.nan,
    )

    aggregated[f"{prefix}_Pct_50_EMA"] = np.where(
        aggregated["Valid_50"] > 0,
        (aggregated["Above_50"] / aggregated["Valid_50"]) * 100,
        np.nan,
    )

    aggregated[f"{prefix}_Pct_200_EMA"] = np.where(
        aggregated["Valid_200"] > 0,
        (aggregated["Above_200"] / aggregated["Valid_200"]) * 100,
        np.nan,
    )

    return aggregated[
        [
            "Date",
            f"{prefix}_Pct_20_EMA",
            f"{prefix}_Pct_50_EMA",
            f"{prefix}_Pct_200_EMA",
        ]
    ]


large_breadth = get_liq_breadth(
    df,
    "Top 100 Liq",
    "Large",
)

mid_breadth = get_liq_breadth(
    df,
    "Mid 150 Liq",
    "Mid",
)

small_breadth = get_liq_breadth(
    df,
    "Lower 250 Liq",
    "Small",
)

micro_breadth = get_liq_breadth(
    df,
    "Micro Liq",
    "Micro",
)

# ------------------------------------------------------------------
# 6. OVERALL BREADTH AGGREGATION
# ------------------------------------------------------------------
overall_breadth = df.groupby("Date").agg(
    Total_Universe=("Active_Universe", "sum"),
    Valid_20=("Valid_20_EMA", "sum"),
    Valid_50=("Valid_50_EMA", "sum"),
    Valid_200=("Valid_200_EMA", "sum"),
    Advances=("Gainer", "sum"),
    Declines=("Loser", "sum"),
    Above_20_EMA=("Above_20_EMA", "sum"),
    Above_50_EMA=("Above_50_EMA", "sum"),
    Above_200_EMA=("Above_200_EMA", "sum"),
    Up_4_Count=("Up_4_Pct", "sum"),
    Down_4_Count=("Down_4_Pct", "sum"),
    Up_25_1M_Count=("Up_25_1M", "sum"),
    Down_25_1M_Count=("Down_25_1M", "sum"),
    New_52W_Highs=("New_52W_High", "sum"),
    New_52W_Lows=("New_52W_Low", "sum"),
    IPO_New_Highs=("IPO_New_High", "sum"),
    Total_Up_Volume=("Up_Volume", "sum"),
    Total_Down_Volume=("Down_Volume", "sum"),
    T3_Breakouts=("Is_Breakout_3d_ago", "sum"),
    T3_Wins=("Follow_Through_Win", "sum"),
).reset_index()

overall_breadth["Pct_Above_20_EMA"] = np.where(
    overall_breadth["Valid_20"] > 0,
    (
        overall_breadth["Above_20_EMA"]
        / overall_breadth["Valid_20"]
    ) * 100,
    np.nan,
)

overall_breadth["Pct_Above_50_EMA"] = np.where(
    overall_breadth["Valid_50"] > 0,
    (
        overall_breadth["Above_50_EMA"]
        / overall_breadth["Valid_50"]
    ) * 100,
    np.nan,
)

overall_breadth["Pct_Above_200_EMA"] = np.where(
    overall_breadth["Valid_200"] > 0,
    (
        overall_breadth["Above_200_EMA"]
        / overall_breadth["Valid_200"]
    ) * 100,
    np.nan,
)

overall_breadth["Rolling_3D_Up_4"] = (
    overall_breadth["Up_4_Count"]
    .rolling(window=3)
    .sum()
)

overall_breadth["Rolling_3D_Down_4"] = (
    overall_breadth["Down_4_Count"]
    .rolling(window=3)
    .sum()
)

overall_breadth["Net_52W_High_Low"] = (
    overall_breadth["New_52W_Highs"]
    - overall_breadth["New_52W_Lows"]
)

adv = overall_breadth["Advances"].astype(float)
dec = overall_breadth["Declines"].astype(float)
uv = overall_breadth["Total_Up_Volume"].astype(float)
dv = overall_breadth["Total_Down_Volume"].astype(float)

overall_breadth["Volume_Ratio"] = np.where(
    dv > 0,
    uv / dv,
    np.nan,
)

overall_breadth["AD_Spread"] = adv - dec

overall_breadth["MCO"] = (
    overall_breadth["AD_Spread"]
    .ewm(span=19, adjust=False)
    .mean()
    - overall_breadth["AD_Spread"]
    .ewm(span=39, adjust=False)
    .mean()
)

overall_breadth["TRIN"] = np.where(
    (adv > 0)
    & (dec > 0)
    & (uv > 0)
    & (dv > 0),
    (adv / dec) / (uv / dv),
    np.nan,
)

final_summary = (
    overall_breadth
    .merge(large_breadth, on="Date", how="left")
    .merge(mid_breadth, on="Date", how="left")
    .merge(small_breadth, on="Date", how="left")
    .merge(micro_breadth, on="Date", how="left")
)

# ------------------------------------------------------------------
# 7. CALIBRATED SCORING ENGINE (MAXIMUM 100)
# ------------------------------------------------------------------
df_score = final_summary.copy()

# C1: Core Breadth (0 to 25 points)
p_blend = (
    0.65 * df_score["Pct_Above_20_EMA"].fillna(0)
    + 0.35 * df_score["Pct_Above_50_EMA"].fillna(0)
)

c1_breadth = (p_blend / 100) * 25

# ------------------------------------------------------------------
# C2: Breakout Health (0 to 25 points)
#
# Daily raw T3_Breakouts and T3_Wins are preserved for dashboard display.
# Only the Composite Score uses five-trading-day sums to stabilize the
# sample size. This does not alter T+3 breakout definition or timing.
# ------------------------------------------------------------------
df_score["T3_Breakouts_5D"] = (
    df_score["T3_Breakouts"]
    .astype(float)
    .rolling(window=5, min_periods=1)
    .sum()
)

df_score["T3_Wins_5D"] = (
    df_score["T3_Wins"]
    .astype(float)
    .rolling(window=5, min_periods=1)
    .sum()
)

t3_b_5d = df_score["T3_Breakouts_5D"]
t3_w_5d = df_score["T3_Wins_5D"]

smoothed_rate = (
    (t3_w_5d + 5)
    / (t3_b_5d + 10)
)

ramp = np.clip(
    (smoothed_rate - 0.45) / 0.15,
    0,
    1,
)

confidence = np.clip(
    t3_b_5d / 10,
    0,
    1,
)

c2_breakout = ramp * 25 * confidence

# C3: Momentum Thrust (0 to 20 points)
net_4d = (
    df_score["Rolling_3D_Up_4"]
    - df_score["Rolling_3D_Down_4"]
)

net_1m = (
    df_score["Up_25_1M_Count"]
    - df_score["Down_25_1M_Count"]
)

rank_4d = (
    net_4d
    .rolling(126, min_periods=1)
    .apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False,
    )
)

rank_1m = (
    net_1m
    .rolling(126, min_periods=1)
    .apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False,
    )
)

c3_momentum = pd.Series(
    (rank_4d * 10)
    + (rank_1m * 10)
).fillna(0)

# C4: Volume and Net New Highs/Lows (0 to 20 points)
rank_vol = (
    df_score["Volume_Ratio"]
    .fillna(1)
    .rolling(126, min_periods=1)
    .apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False,
    )
)

rank_hl = (
    df_score["Net_52W_High_Low"]
    .rolling(126, min_periods=1)
    .apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False,
    )
)

c4_vol_hl = pd.Series(
    np.where(
        df_score["Volume_Ratio"].fillna(0) > 1.0,
        rank_vol * 10,
        0,
    )
    + np.where(
        df_score["Net_52W_High_Low"] > 0,
        rank_hl * 10,
        0,
    )
).fillna(0)

# C5: Structural Trend (0, 5, or 10 points)
p200 = df_score["Pct_Above_200_EMA"].fillna(0)

p200_slope = p200.diff(20).fillna(0)

c5_lt = np.where(
    (p200 > 50)
    & (p200_slope > 0),
    10,
    np.where(
        (p200 <= 50)
        & (p200_slope < 0),
        0,
        5,
    ),
)

# C6: Narrow-Rally Divergence Penalty (-15 to 0 points)
hunting_ground = (
    df_score["Small_Pct_50_EMA"].fillna(0)
    + df_score["Micro_Pct_50_EMA"].fillna(0)
) / 2

gap = (
    df_score["Large_Pct_50_EMA"].fillna(0)
    - hunting_ground
)

c6_penalty = -np.clip(
    (gap - 20) * 0.75,
    0,
    15,
)

# C7: Washout Recovery Bonus (0 or +15 points)
min_20d_p20 = (
    df_score["Pct_Above_20_EMA"]
    .rolling(20, min_periods=1)
    .min()
)

c7_bonus = np.where(
    (min_20d_p20 <= 10)
    & (p_blend >= 50),
    15,
    0,
)

# Final Composite Score (0 to 100)
raw_score = (
    pd.Series(c1_breadth).fillna(0)
    + pd.Series(c2_breakout).fillna(0)
    + c3_momentum.fillna(0)
    + c4_vol_hl.fillna(0)
    + pd.Series(c5_lt).fillna(0)
    + pd.Series(c6_penalty).fillna(0)
    + pd.Series(c7_bonus).fillna(0)
)

final_summary["T3_Breakouts_5D"] = (
    df_score["T3_Breakouts_5D"]
)

final_summary["T3_Wins_5D"] = (
    df_score["T3_Wins_5D"]
)

final_summary["Composite_Score"] = (
    raw_score
    .fillna(0)
    .clip(lower=0, upper=100)
    .round()
    .astype(int)
)

final_summary = final_summary.drop(
    columns=[
        "Valid_20",
        "Valid_50",
        "Valid_200",
    ]
)

final_summary.to_csv(
    OUTPUT_CSV,
    index=False,
)

cutoff_date = (
    df["Date"].max()
    - pd.Timedelta(days=450)
)

df[
    df["Date"] >= cutoff_date
].to_parquet(
    TRAILING_CACHE_FILE,
    index=False,
)

print(
    "✅ Output Generated Successfully. "
    "C2 uses rolling 5-day T+3 totals; "
    "daily T3 statistics remain unchanged for dashboard display."
)

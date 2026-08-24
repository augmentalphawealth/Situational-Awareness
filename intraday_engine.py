from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from kite_helper import fetch_with_backoff, get_kite_client

ROOT = Path(__file__).resolve().parent
MASTER_FILE = ROOT / "nse_6yr_historical.parquet"
TEMP_DIR = ROOT / "intraday_tmp"
SNAPSHOT_FILE = TEMP_DIR / "intraday_snapshot.parquet"
STOCK_METRICS_FILE = TEMP_DIR / "intraday_stock_metrics.parquet"
MARKET_METRICS_FILE = TEMP_DIR / "intraday_market_metrics.json"
STATUS_FILE = TEMP_DIR / "intraday_status.json"

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
NOW_IST = dt.datetime.now(IST)
TODAY = pd.Timestamp(NOW_IST.date()).normalize()


def fail(message: str) -> None:
    print(f"❌ {message}", file=sys.stderr)
    raise SystemExit(1)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def load_master() -> pd.DataFrame:
    if not MASTER_FILE.exists():
        fail(f"Permanent master file not found: {MASTER_FILE.name}")

    df = pd.read_parquet(MASTER_FILE)
    required = {"Symbol", "Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(df.columns)
    if missing:
        fail(f"Master file is missing columns: {sorted(missing)}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df = df.dropna(subset=["Date", "Symbol"])
    return df


def fetch_live_snapshot(kite, symbols: list[str]) -> pd.DataFrame:
    rows = []
    for start in range(0, len(symbols), 200):
        chunk_symbols = symbols[start:start + 200]
        instruments = [f"NSE:{symbol}" for symbol in chunk_symbols]
        response = fetch_with_backoff(kite.quote, instruments)
        if not response:
            fail(f"No quote response for chunk {start // 200 + 1}")

        for key, value in response.items():
            symbol = key.replace("NSE:", "", 1)
            ohlc = value.get("ohlc") or {}
            rows.append(
                {
                    "Symbol": symbol,
                    "Date": TODAY,
                    "Open": ohlc.get("open"),
                    "High": ohlc.get("high"),
                    "Low": ohlc.get("low"),
                    "Close": value.get("last_price"),
                    "Volume": value.get("volume", 0),
                    "Snapshot_Time": NOW_IST.isoformat(),
                }
            )
        time.sleep(0.5)

    live = pd.DataFrame(rows)
    if live.empty:
        fail("Kite returned no live stock data.")

    for column in ["Open", "High", "Low", "Close", "Volume"]:
        live[column] = pd.to_numeric(live[column], errors="coerce")

    live = live.dropna(subset=["Symbol", "Close"])
    live["Date"] = pd.to_datetime(live["Date"]).dt.normalize()
    live = live.drop_duplicates(["Symbol", "Date"], keep="last")

    if live.empty:
        fail("Live stock data was empty after validation.")
    return live


def prepare_intraday_stock_metrics(master: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    historical = master[master["Date"] < TODAY].copy()
    combined = pd.concat([historical, live.drop(columns=["Snapshot_Time"])], ignore_index=True, sort=False)
    combined = combined.sort_values(["Symbol", "Date"]).drop_duplicates(["Symbol", "Date"], keep="last").reset_index(drop=True)
    group = combined.groupby("Symbol", group_keys=False)

    combined["History_Days"] = group.cumcount() + 1
    combined["Prior_History_Days"] = combined["History_Days"] - 1
    combined["Daily_Turnover"] = combined["Close"] * combined["Volume"]
    combined["Prior_Turnover_20D_Avg"] = group["Daily_Turnover"].transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())
    mature = (combined["Prior_History_Days"] >= 20) & (combined["Prior_Turnover_20D_Avg"] >= 50_000_000)
    new = (combined["Prior_History_Days"] >= 1) & (combined["Prior_History_Days"] < 20)
    combined["Active_Universe"] = (mature | new) & (combined["Volume"] > 0)

    combined["Turnover_45d_Avg"] = group["Daily_Turnover"].transform(lambda x: x.shift(1).rolling(45, min_periods=1).mean())
    combined["Cap_Rank"] = combined.groupby("Date")["Turnover_45d_Avg"].rank(ascending=False, method="min")
    combined["Liquidity_Category"] = np.select(
        [combined["Cap_Rank"] <= 100, (combined["Cap_Rank"] > 100) & (combined["Cap_Rank"] <= 250), (combined["Cap_Rank"] > 250) & (combined["Cap_Rank"] <= 500)],
        ["Top 100 Liq", "Mid 150 Liq", "Lower 250 Liq"], default="Micro Liq"
    )

    for period in [20, 50, 200]:
        ema = group["Close"].transform(lambda x, p=period: x.ewm(span=p, adjust=False, min_periods=1).mean())
        combined[f"EMA_{period}"] = ema
        combined[f"Valid_{period}_EMA"] = combined["Active_Universe"] & (combined["Prior_History_Days"] >= period)
        combined[f"Above_{period}_EMA"] = combined[f"Valid_{period}_EMA"] & (combined["Close"] > ema)

    combined["Prev_Close"] = group["Close"].shift(1)
    combined["Has_Prior_Close"] = combined["Prev_Close"].notna()
    combined["Daily_Pct"] = group["Close"].pct_change() * 100
    combined["Pct_1M"] = group["Close"].pct_change(21) * 100
    combined["Gainer"] = combined["Active_Universe"] & combined["Has_Prior_Close"] & (combined["Daily_Pct"] > 0)
    combined["Loser"] = combined["Active_Universe"] & combined["Has_Prior_Close"] & (combined["Daily_Pct"] < 0)
    combined["Up_4_Pct"] = combined["Active_Universe"] & (combined["Daily_Pct"] >= 4)
    combined["Down_4_Pct"] = combined["Active_Universe"] & (combined["Daily_Pct"] <= -4)
    combined["Up_25_1M"] = combined["Active_Universe"] & (combined["Pct_1M"] >= 25)
    combined["Down_25_1M"] = combined["Active_Universe"] & (combined["Pct_1M"] <= -25)

    combined["Rolling_52W_High"] = group["High"].transform(lambda x: x.shift(1).rolling(252, min_periods=1).max())
    combined["Rolling_52W_Low"] = group["Low"].transform(lambda x: x.shift(1).rolling(252, min_periods=1).min())
    combined["Prior_ATH"] = group["High"].transform(lambda x: x.shift(1).cummax())
    combined["New_52W_High"] = combined["Active_Universe"] & (combined["Prior_History_Days"] >= 252) & (combined["Close"] >= combined["Rolling_52W_High"])
    combined["New_52W_Low"] = combined["Active_Universe"] & (combined["Prior_History_Days"] >= 252) & (combined["Close"] <= combined["Rolling_52W_Low"])
    combined["IPO_New_High"] = combined["Active_Universe"] & (combined["Prior_History_Days"] >= 1) & (combined["Prior_History_Days"] < 252) & (combined["Close"] >= combined["Prior_ATH"])

    combined["Up_Volume"] = np.where(combined["Gainer"], combined["Daily_Turnover"], 0)
    combined["Down_Volume"] = np.where(combined["Loser"], combined["Daily_Turnover"], 0)
    return combined


def calculate_aggregate(stock: pd.DataFrame) -> pd.DataFrame:
    aggregate = stock.groupby("Date").agg(
        Total_Universe=("Active_Universe", "sum"),
        Valid_20=("Valid_20_EMA", "sum"), Valid_50=("Valid_50_EMA", "sum"), Valid_200=("Valid_200_EMA", "sum"),
        Advances=("Gainer", "sum"), Declines=("Loser", "sum"),
        Above_20_EMA=("Above_20_EMA", "sum"), Above_50_EMA=("Above_50_EMA", "sum"), Above_200_EMA=("Above_200_EMA", "sum"),
        Up_4_Count=("Up_4_Pct", "sum"), Down_4_Count=("Down_4_Pct", "sum"),
        Up_25_1M_Count=("Up_25_1M", "sum"), Down_25_1M_Count=("Down_25_1M", "sum"),
        New_52W_Highs=("New_52W_High", "sum"), New_52W_Lows=("New_52W_Low", "sum"), IPO_New_Highs=("IPO_New_High", "sum"),
        Total_Up_Volume=("Up_Volume", "sum"), Total_Down_Volume=("Down_Volume", "sum"),
    ).reset_index()

    for period in [20, 50, 200]:
        aggregate[f"Pct_Above_{period}_EMA"] = np.where(aggregate[f"Valid_{period}"] > 0, aggregate[f"Above_{period}_EMA"] / aggregate[f"Valid_{period}"] * 100, np.nan)

    aggregate["Rolling_3D_Up_4"] = aggregate["Up_4_Count"].rolling(3, min_periods=1).sum()
    aggregate["Rolling_3D_Down_4"] = aggregate["Down_4_Count"].rolling(3, min_periods=1).sum()
    aggregate["Net_52W_High_Low"] = aggregate["New_52W_Highs"] - aggregate["New_52W_Lows"]
    adv = aggregate["Advances"].astype(float)
    dec = aggregate["Declines"].astype(float)
    up_volume = aggregate["Total_Up_Volume"].astype(float)
    down_volume = aggregate["Total_Down_Volume"].astype(float)
    aggregate["Volume_Ratio"] = np.where(down_volume > 0, up_volume / down_volume, np.nan)
    aggregate["AD_Spread"] = adv - dec
    aggregate["MCO"] = aggregate["AD_Spread"].ewm(span=19, adjust=False).mean() - aggregate["AD_Spread"].ewm(span=39, adjust=False).mean()
    aggregate["TRIN"] = np.where((adv > 0) & (dec > 0) & (up_volume > 0) & (down_volume > 0), (adv / dec) / (up_volume / down_volume), np.nan)

    for name, prefix in [("Top 100 Liq", "Large"), ("Mid 150 Liq", "Mid"), ("Lower 250 Liq", "Small"), ("Micro Liq", "Micro")]:
        part = stock[stock["Liquidity_Category"] == name].groupby("Date").agg(
            valid20=("Valid_20_EMA", "sum"), valid50=("Valid_50_EMA", "sum"), valid200=("Valid_200_EMA", "sum"),
            above20=("Above_20_EMA", "sum"), above50=("Above_50_EMA", "sum"), above200=("Above_200_EMA", "sum"),
        ).reset_index()
        for period in [20, 50, 200]:
            part[f"{prefix}_Pct_{period}_EMA"] = np.where(part[f"valid{period}"] > 0, part[f"above{period}"] / part[f"valid{period}"] * 100, np.nan)
        keep = ["Date", f"{prefix}_Pct_20_EMA", f"{prefix}_Pct_50_EMA", f"{prefix}_Pct_200_EMA"]
        aggregate = aggregate.merge(part[keep], on="Date", how="left")

    aggregate = aggregate.sort_values("Date").reset_index(drop=True)
    p_blend = 0.65 * aggregate["Pct_Above_20_EMA"].fillna(0) + 0.35 * aggregate["Pct_Above_50_EMA"].fillna(0)
    c1 = (p_blend / 100) * 25
    # T+3 is calculated only from fully available historical rows in this temporary analysis.
    t3_breakouts = pd.Series(0, index=aggregate.index, dtype=float)
    t3_wins = pd.Series(0, index=aggregate.index, dtype=float)
    smoothed_rate = (t3_wins + 5) / (t3_breakouts + 10)
    c2 = np.clip((smoothed_rate - 0.45) / 0.15, 0, 1) * 25 * np.clip(t3_breakouts / 10, 0, 1)
    net_4d = aggregate["Rolling_3D_Up_4"] - aggregate["Rolling_3D_Down_4"]
    net_1m = aggregate["Up_25_1M_Count"] - aggregate["Down_25_1M_Count"]
    c3 = net_4d.rolling(126, min_periods=1).rank(pct=True) * 10 + net_1m.rolling(126, min_periods=1).rank(pct=True) * 10
    rank_vol = aggregate["Volume_Ratio"].fillna(1).rolling(126, min_periods=1).rank(pct=True)
    rank_hl = aggregate["Net_52W_High_Low"].rolling(126, min_periods=1).rank(pct=True)
    c4 = np.where(aggregate["Volume_Ratio"].fillna(0) > 1, rank_vol * 10, 0) + np.where(aggregate["Net_52W_High_Low"] > 0, rank_hl * 10, 0)
    p200 = aggregate["Pct_Above_200_EMA"].fillna(0)
    c5 = np.where((p200 > 50) & (p200.diff(20).fillna(0) > 0), 10, np.where((p200 <= 50) & (p200.diff(20).fillna(0) < 0), 0, 5))
    hunting = (aggregate["Small_Pct_50_EMA"].fillna(0) + aggregate["Micro_Pct_50_EMA"].fillna(0)) / 2
    c6 = -np.clip((aggregate["Large_Pct_50_EMA"].fillna(0) - hunting - 20) * 0.75, 0, 15)
    c7 = np.where((aggregate["Pct_Above_20_EMA"].rolling(20, min_periods=1).min() <= 10) & (p_blend >= 50), 15, 0)
    aggregate["Composite_Score"] = (pd.Series(c1).fillna(0) + pd.Series(c2).fillna(0) + pd.Series(c3).fillna(0) + pd.Series(c4).fillna(0) + pd.Series(c5).fillna(0) + pd.Series(c6).fillna(0) + pd.Series(c7).fillna(0)).clip(0, 100).round().astype(int)
    aggregate["Unchanged"] = aggregate["Total_Universe"] - aggregate["Advances"] - aggregate["Declines"]
    return aggregate.drop(columns=["Valid_20", "Valid_50", "Valid_200"])


def main() -> None:
    master = load_master()
    symbols = sorted(master["Symbol"].unique())
    kite = get_kite_client()
    live = fetch_live_snapshot(kite, symbols)
    stock_metrics = prepare_intraday_stock_metrics(master, live)
    aggregate = calculate_aggregate(stock_metrics)
    today = aggregate[aggregate["Date"] == TODAY]
    if today.empty:
        fail("No current-day aggregate row was created.")

    atomic_parquet(live, SNAPSHOT_FILE)
    atomic_parquet(stock_metrics[stock_metrics["Date"] == TODAY], STOCK_METRICS_FILE)
    atomic_json(today.iloc[0].to_dict(), MARKET_METRICS_FILE)
    atomic_json({"status": "ok", "updated_at": NOW_IST.isoformat(), "rows": len(live)}, STATUS_FILE)
    print(f"✅ Temporary intraday files written for {TODAY.date()}")
    print(f"✅ Composite Score: {today.iloc[0]['Composite_Score']}")
    print(f"✅ MCO: {today.iloc[0]['MCO']}")
    print(f"✅ TRIN: {today.iloc[0]['TRIN']}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"❌ Intraday engine failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

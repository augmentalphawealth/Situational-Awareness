from __future__ import annotations

import datetime
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from kiteconnect import KiteConnect

ROOT = Path(__file__).resolve().parent
CACHE_FILE = ROOT / "trailing_cache.parquet"
AGGREGATE_FILE = ROOT / "historical_breadth_regime_6yr.csv"
LIVE_FILE = ROOT / "live_intraday.parquet"
DASHBOARD_FILE = ROOT / "dashboard_data.parquet"
BREADTH_FILE = ROOT / "live_intraday_breadth.csv"
SYNC_FILE = ROOT / "last_sync.txt"

API_KEY = os.environ.get("KITE_API_KEY", "").strip()
ACCESS_TOKEN = os.environ.get("KITE_ACCESS_TOKEN", "").strip()
IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
NOW_IST = datetime.datetime.now(IST)
TODAY = pd.Timestamp(NOW_IST.date()).normalize()


def fail(message: str) -> None:
    print(f"❌ {message}", file=sys.stderr)
    sys.exit(1)


def atomic_write(frame: pd.DataFrame, destination: Path, csv: bool = False) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    if csv:
        frame.to_csv(temporary, index=False)
    else:
        frame.to_parquet(temporary, index=False)
    temporary.replace(destination)


def connect_to_kite() -> KiteConnect:
    if not API_KEY:
        fail("KITE_API_KEY is missing.")
    if not ACCESS_TOKEN:
        fail("KITE_ACCESS_TOKEN is missing.")
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)
    return kite


def market_is_open_today(kite: KiteConnect) -> bool:
    try:
        response = kite.quote(["NSE:NIFTY 50"])
        data = response.get("NSE:NIFTY 50", {})
        last_trade_time = data.get("last_trade_time")
        if last_trade_time is None:
            print("⚠️ NIFTY 50 last_trade_time unavailable; continuing.")
            return True
        trade_date = last_trade_time.date() if hasattr(last_trade_time, "date") else pd.Timestamp(last_trade_time).date()
        if trade_date != TODAY.date():
            print(f"🛑 Market holiday guard: last trade was {trade_date}.")
            return False
        return True
    except Exception as exc:
        print(f"⚠️ Holiday guard failed; continuing: {type(exc).__name__}: {exc}")
        return True


def fetch_quotes(kite: KiteConnect, symbols: list[str]) -> pd.DataFrame:
    requests = [f"NSE:{symbol}" for symbol in symbols]
    chunks = [requests[i:i + 200] for i in range(0, len(requests), 200)]
    rows = []

    for number, chunk in enumerate(chunks, start=1):
        print(f"Fetching quote chunk {number}/{len(chunks)} ({len(chunk)} symbols)...")
        response = None
        for attempt in range(5):
            try:
                response = kite.quote(chunk)
                if response:
                    break
            except Exception as exc:
                text = str(exc)
                print(f"⚠️ Attempt {attempt + 1}/5 failed: {type(exc).__name__}: {text}")
                time.sleep(2 ** attempt if "429" in text or "403" in text else 1)
        if not response:
            print(f"⚠️ No response for chunk {number}; continuing.")
            continue

        for full_symbol, data in response.items():
            ohlc = data.get("ohlc") or {}
            price = data.get("last_price")
            if price is None:
                continue
            rows.append({
                "Date": TODAY,
                "Symbol": full_symbol.replace("NSE:", ""),
                "Open": ohlc.get("open"),
                "High": ohlc.get("high"),
                "Low": ohlc.get("low"),
                "Close": price,
                "Volume": data.get("volume", 0),
            })
        time.sleep(1.1)

    if not rows:
        fail("Kite returned no usable quote rows.")

    live = pd.DataFrame(rows)
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        live[column] = pd.to_numeric(live[column], errors="coerce")
    live = live.dropna(subset=["Symbol", "Close"])
    live["Date"] = pd.to_datetime(live["Date"]).dt.normalize()
    live = live.drop_duplicates(["Date", "Symbol"], keep="last")
    return live


def prepare_stock_data(historical: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    historical = historical.copy()
    historical["Date"] = pd.to_datetime(historical["Date"], errors="coerce").dt.normalize()
    historical["Symbol"] = historical["Symbol"].astype(str).str.strip()
    historical = historical.dropna(subset=["Date", "Symbol"])
    historical_without_today = historical[historical["Date"] != TODAY].copy()

    combined = pd.concat([historical_without_today, live], ignore_index=True, sort=False)
    combined = combined.sort_values(["Symbol", "Date"]).drop_duplicates(["Date", "Symbol"], keep="last").reset_index(drop=True)
    grouped = combined.groupby("Symbol", group_keys=False)

    combined["History_Days"] = grouped.cumcount() + 1
    combined["Prior_History_Days"] = combined["History_Days"] - 1
    combined["Daily_Turnover"] = combined["Close"] * combined["Volume"]
    combined["Prior_Turnover_20D_Avg"] = grouped["Daily_Turnover"].transform(lambda values: values.shift(1).rolling(20, min_periods=1).mean())

    mature = (combined["Prior_History_Days"] >= 20) & (combined["Prior_Turnover_20D_Avg"] >= 50_000_000)
    new = (combined["Prior_History_Days"] >= 1) & (combined["Prior_History_Days"] < 20)
    combined["Active_Universe"] = (mature | new) & (combined["Volume"] > 0)
    combined["Prev_Close"] = grouped["Close"].shift(1)
    combined["Daily_Pct"] = grouped["Close"].pct_change() * 100
    combined["Gainer"] = combined["Active_Universe"] & (combined["Daily_Pct"] > 0)
    combined["Loser"] = combined["Active_Universe"] & (combined["Daily_Pct"] < 0)

    for period in [20, 50, 200]:
        ema = grouped["Close"].transform(lambda values: values.ewm(span=period, adjust=False, min_periods=period).mean())
        combined[f"EMA_{period}"] = ema
        combined[f"Above_{period}_EMA"] = combined["Close"] > ema

    combined["Up_4_Pct"] = combined["Daily_Pct"] >= 4
    combined["Down_4_Pct"] = combined["Daily_Pct"] <= -4
    combined["Pct_1M"] = grouped["Close"].pct_change(21) * 100
    combined["Up_25_1M"] = combined["Pct_1M"] >= 25
    combined["Down_25_1M"] = combined["Pct_1M"] <= -25

    for period in [20, 50, 200]:
        combined[f"Above_{period}_EMA"] = combined[f"Above_{period}_EMA"].fillna(False)

    return combined


def previous_row(aggregate: pd.DataFrame, column: str, default=np.nan):
    if aggregate.empty or column not in aggregate.columns:
        return default
    values = aggregate[column].dropna()
    return values.iloc[-1] if not values.empty else default


def calculate_mco(aggregate_without_today: pd.DataFrame, advances: int, declines: int) -> float:
    if advances + declines == 0:
        return np.nan
    ad_diff = advances - declines
    prior = aggregate_without_today.copy()
    if not prior.empty and {"Advances", "Declines"}.issubset(prior.columns):
        prior_diff = pd.to_numeric(prior["Advances"], errors="coerce").fillna(0) - pd.to_numeric(prior["Declines"], errors="coerce").fillna(0)
        series = pd.concat([prior_diff, pd.Series([ad_diff])], ignore_index=True)
    else:
        series = pd.Series([ad_diff])
    return float(series.ewm(span=19, adjust=False).mean().iloc[-1] - series.ewm(span=39, adjust=False).mean().iloc[-1])


def create_aggregate_row(stock_data: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    today = stock_data[stock_data["Date"] == TODAY].copy()
    active = today[today["Active_Universe"] == True].copy()
    if active.empty:
        fail("No active current-day universe rows available.")

    advances = int(active["Gainer"].sum())
    declines = int(active["Loser"].sum())
    total = int(len(active))
    unchanged = max(total - advances - declines, 0)

    row = {
        "Date": TODAY,
        "Total_Universe": total,
        "Advances": advances,
        "Declines": declines,
        "Unchanged": unchanged,
        "Above_20_EMA": int(active["Above_20_EMA"].sum()),
        "Above_50_EMA": int(active["Above_50_EMA"].sum()),
        "Above_200_EMA": int(active["Above_200_EMA"].sum()),
        "Pct_Above_20_EMA": round(float(active["Above_20_EMA"].mean() * 100), 2),
        "Pct_Above_50_EMA": round(float(active["Above_50_EMA"].mean() * 100), 2),
        "Pct_Above_200_EMA": round(float(active["Above_200_EMA"].mean() * 100), 2),
        "Up_4_Count": int(active["Up_4_Pct"].sum()),
        "Down_4_Count": int(active["Down_4_Pct"].sum()),
        "Up_25_1M_Count": int(active["Up_25_1M"].sum()),
        "Down_25_1M_Count": int(active["Down_25_1M"].sum()),
    }

    up_volume = float(active.loc[active["Gainer"], "Volume"].sum())
    down_volume = float(active.loc[active["Loser"], "Volume"].sum())
    row["Total_Up_Volume"] = up_volume
    row["Total_Down_Volume"] = down_volume
    row["Volume_Ratio"] = round(up_volume / down_volume, 4) if down_volume > 0 else np.nan
    row["TRIN"] = round((advances / declines) / (up_volume / down_volume), 4) if declines > 0 and down_volume > 0 and up_volume > 0 else np.nan
    row["AD_Spread"] = advances - declines
    row["MCO"] = calculate_mco(previous, advances, declines)

    row["New_52W_Highs"] = np.nan
    row["New_52W_Lows"] = np.nan
    row["Net_52W_High_Low"] = np.nan
    row["IPO_New_Highs"] = np.nan
    row["Rolling_3D_Up_4"] = int(previous["Rolling_3D_Up_4"].tail(2).sum() + row["Up_4_Count"]) if "Rolling_3D_Up_4" in previous.columns else row["Up_4_Count"]
    row["Rolling_3D_Down_4"] = int(previous["Rolling_3D_Down_4"].tail(2).sum() + row["Down_4_Count"]) if "Rolling_3D_Down_4" in previous.columns else row["Down_4_Count"]

    # A T+3 result is only final after three later sessions. Keep the last completed value.
    row["T3_Breakouts"] = int(active["Up_4_Pct"].sum())
    row["T3_Wins"] = previous_row(previous, "T3_Wins", 0)

    # Composite score uses current breadth inputs and the prior EOD cohort fields.
    score = 0
    score += 25 if row["Pct_Above_20_EMA"] >= 50 else 0
    score += 25 if row["Pct_Above_50_EMA"] >= 50 else 0
    score += 25 if row["Pct_Above_200_EMA"] >= 50 else 0
    score += 25 if advances > declines else 0
    row["Composite_Score"] = score

    for column in previous.columns:
        if column not in row:
            row[column] = previous_row(previous, column, np.nan)

    return pd.DataFrame([row])


def main() -> None:
    print("Connecting to Zerodha API for live aggregate snapshot...")
    if not CACHE_FILE.exists():
        fail(f"Historical cache missing: {CACHE_FILE.name}")

    historical = pd.read_parquet(CACHE_FILE)
    required = {"Date", "Symbol", "Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(historical.columns)
    if missing:
        fail(f"Historical cache missing columns: {sorted(missing)}")

    symbols = sorted(historical["Symbol"].dropna().astype(str).str.strip().unique())
    kite = connect_to_kite()
    if not market_is_open_today(kite):
        return

    live = fetch_quotes(kite, symbols)
    prepared = prepare_stock_data(historical, live)

    aggregate = pd.read_csv(AGGREGATE_FILE) if AGGREGATE_FILE.exists() else pd.DataFrame()
    if not aggregate.empty and "Date" in aggregate.columns:
        aggregate["Date"] = pd.to_datetime(aggregate["Date"], errors="coerce").dt.normalize()
        aggregate = aggregate.dropna(subset=["Date"])
        previous = aggregate[aggregate["Date"] != TODAY].sort_values("Date").reset_index(drop=True)
    else:
        previous = pd.DataFrame()

    today_row = create_aggregate_row(prepared, previous)
    output = pd.concat([previous, today_row], ignore_index=True, sort=False).sort_values("Date").drop_duplicates("Date", keep="last")
    breadth = pd.DataFrame([{
        "Time": NOW_IST.strftime("%H:%M"),
        "Advances": int(today_row.iloc[0]["Advances"]),
        "Declines": int(today_row.iloc[0]["Declines"]),
        "Unchanged": int(today_row.iloc[0]["Unchanged"]),
        "Total_Universe": int(today_row.iloc[0]["Total_Universe"]),
        "Date": NOW_IST.strftime("%Y-%m-%d"),
    }])

    if breadth.iloc[0]["Advances"] == 0 and breadth.iloc[0]["Declines"] == 0:
        print("🛑 Zero advances and declines. No files changed.")
        return

    atomic_write(live, LIVE_FILE)
    atomic_write(prepared, DASHBOARD_FILE)
    atomic_write(output, AGGREGATE_FILE, csv=True)
    atomic_write(breadth, BREADTH_FILE, csv=True)
    SYNC_FILE.write_text(f"Today, {NOW_IST.strftime('%I:%M %p')} IST\n", encoding="utf-8")

    print(f"✅ Live rows written: {len(live)}")
    print(f"✅ Prepared dashboard rows written: {len(prepared)}")
    print(f"✅ Aggregate row written with Composite Score={today_row.iloc[0]['Composite_Score']}")
    print(f"✅ MCO={today_row.iloc[0]['MCO']} TRIN={today_row.iloc[0]['TRIN']}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"❌ Intraday update failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

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
    try:
        kite = KiteConnect(api_key=API_KEY)
        kite.set_access_token(ACCESS_TOKEN)
        return kite
    except Exception as exc:
        fail(f"Could not initialize Kite connection: {type(exc).__name__}: {exc}")


def market_is_open_today(kite: KiteConnect) -> bool:
    try:
        response = kite.quote(["NSE:NIFTY 50"])
        data = response.get("NSE:NIFTY 50", {})
        last_trade_time = data.get("last_trade_time")
        if last_trade_time is None:
            print("⚠️ NIFTY 50 last_trade_time unavailable; continuing.")
            return True
        trade_date = (
            last_trade_time.date()
            if hasattr(last_trade_time, "date")
            else pd.Timestamp(last_trade_time).date()
        )
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
    if live.empty:
        fail("All quote rows became invalid after cleaning.")
    return live


def prepare_stock_data(historical: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    historical["Date"] = pd.to_datetime(historical["Date"], errors="coerce").dt.normalize()
    historical["Symbol"] = historical["Symbol"].astype(str).str.strip()
    historical = historical.dropna(subset=["Date", "Symbol"])
    historical_without_today = historical[historical["Date"] != TODAY].copy()

    combined = pd.concat([historical_without_today, live], ignore_index=True, sort=False)
    combined = (
        combined.sort_values(["Symbol", "Date"])
        .drop_duplicates(["Date", "Symbol"], keep="last")
        .reset_index(drop=True)
    )

    grouped = combined.groupby("Symbol", group_keys=False)
    combined["History_Days"] = grouped.cumcount() + 1
    combined["Prior_History_Days"] = combined["History_Days"] - 1
    combined["Daily_Turnover"] = combined["Close"] * combined["Volume"]
    combined["Prior_Turnover_20D_Avg"] = grouped["Daily_Turnover"].transform(
        lambda values: values.shift(1).rolling(20, min_periods=1).mean()
    )
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

    return combined


def latest_value(frame: pd.DataFrame, column: str, default: float = np.nan) -> float:
    if column not in frame.columns:
        return default
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.iloc[-1]) if not values.empty else default


def create_aggregate_row(stock_data: pd.DataFrame, previous_aggregate: pd.DataFrame) -> pd.DataFrame:
    today = stock_data[stock_data["Date"] == TODAY].copy()
    active = today[today["Active_Universe"] == True].copy()
    if active.empty:
        fail("No active current-day universe rows available.")

    advances = int(active["Gainer"].sum())
    declines = int(active["Loser"].sum())
    total = len(active)
    unchanged = max(total - advances - declines, 0)

    row = {
        "Date": TODAY,
        "Total_Universe": total,
        "Advances": advances,
        "Declines": declines,
        "Unchanged": unchanged,
    }

    for period in [20, 50, 200]:
        column = f"Above_{period}_EMA"
        row[f"Pct_Above_{period}_EMA"] = round(float(active[column].mean() * 100), 2)

    advancing_volume = active.loc[active["Gainer"], "Volume"].sum()
    declining_volume = active.loc[active["Loser"], "Volume"].sum()
    row["Volume_Ratio"] = round(float(advancing_volume / declining_volume), 4) if declining_volume > 0 else np.nan
    row["Net_52W_High_Low"] = np.nan
    row["IPO_New_Highs"] = np.nan
    row["MCO"] = np.nan
    row["TRIN"] = round(float((advances / declines) / (advancing_volume / declining_volume)), 4) if declines > 0 and declining_volume > 0 else np.nan

    # These metrics require future sessions or existing EOD-specific logic.
    # Carrying the previous value is safer than presenting a false intraday calculation.
    for column in ["Composite_Score", "T3_Wins", "T3_Breakouts", "Up_25_1M_Count", "Down_25_1M_Count", "Rolling_3D_Up_4", "Rolling_3D_Down_4"]:
        if column in previous_aggregate.columns and not previous_aggregate.empty:
            row[column] = previous_aggregate.iloc[-1].get(column, np.nan)
        else:
            row[column] = np.nan

    return pd.DataFrame([row])


def main() -> None:
    print("Connecting to Zerodha API for intraday snapshot...")
    if not CACHE_FILE.exists():
        fail(f"Historical cache missing: {CACHE_FILE.name}")

    historical = pd.read_parquet(CACHE_FILE)
    required = {"Date", "Symbol", "Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(historical.columns)
    if missing:
        fail(f"Historical cache missing columns: {sorted(missing)}")

    symbols = sorted(historical["Symbol"].dropna().astype(str).str.strip().unique())
    if not symbols:
        fail("No symbols found in historical cache.")

    kite = connect_to_kite()
    if not market_is_open_today(kite):
        return

    live = fetch_quotes(kite, symbols)
    prepared = prepare_stock_data(historical, live)

    if AGGREGATE_FILE.exists():
        aggregate = pd.read_csv(AGGREGATE_FILE)
    else:
        aggregate = pd.DataFrame()

    if not aggregate.empty and "Date" in aggregate.columns:
        aggregate["Date"] = pd.to_datetime(aggregate["Date"], errors="coerce").dt.normalize()
        aggregate = aggregate.dropna(subset=["Date"])
        previous = aggregate[aggregate["Date"] != TODAY].sort_values("Date")
    else:
        previous = aggregate

    aggregate_row = create_aggregate_row(prepared, previous)
    aggregate_without_today = aggregate[aggregate["Date"] != TODAY].copy() if not aggregate.empty and "Date" in aggregate.columns else aggregate
    aggregate_output = pd.concat([aggregate_without_today, aggregate_row], ignore_index=True, sort=False)
    aggregate_output = aggregate_output.sort_values("Date").drop_duplicates("Date", keep="last")

    breadth = pd.DataFrame([{
        "Time": NOW_IST.strftime("%H:%M"),
        "Advances": int(aggregate_row.iloc[0]["Advances"]),
        "Declines": int(aggregate_row.iloc[0]["Declines"]),
        "Unchanged": int(aggregate_row.iloc[0]["Unchanged"]),
        "Total_Universe": int(aggregate_row.iloc[0]["Total_Universe"]),
        "Date": NOW_IST.strftime("%Y-%m-%d"),
    }])

    if int(breadth.iloc[0]["Advances"]) == 0 and int(breadth.iloc[0]["Declines"]) == 0:
        print("🛑 Zero advances and declines. No files changed.")
        return

    atomic_write(live, LIVE_FILE)
    atomic_write(prepared, DASHBOARD_FILE)
    atomic_write(aggregate_output, AGGREGATE_FILE, csv=True)
    atomic_write(breadth, BREADTH_FILE, csv=True)
    SYNC_FILE.write_text(f"Today, {NOW_IST.strftime('%I:%M %p')} IST\n", encoding="utf-8")

    print(f"✅ Live rows written: {len(live)}")
    print(f"✅ Prepared dashboard rows written: {len(prepared)}")
    print(f"✅ Aggregate rows written: {len(aggregate_output)}")
    print(f"✅ Breadth: {int(breadth.iloc[0]['Advances'])} advances, {int(breadth.iloc[0]['Declines'])} declines")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"❌ Intraday update failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

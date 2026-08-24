from __future__ import annotations

import datetime
import os
import sys
import time
from pathlib import Path

import pandas as pd
from kiteconnect import KiteConnect


ROOT = Path(__file__).resolve().parent

CACHE_FILE = ROOT / "trailing_cache.parquet"
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


def atomic_write_parquet(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(destination)


def atomic_write_csv(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    frame.to_csv(temporary, index=False)
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
        test_quote = kite.quote(["NSE:NIFTY 50"])
        index_data = test_quote.get("NSE:NIFTY 50", {})

        last_trade_time = index_data.get("last_trade_time")

        if last_trade_time is None:
            print("⚠️ NIFTY 50 last_trade_time unavailable; continuing.")
            return True

        if hasattr(last_trade_time, "date"):
            last_trade_date = last_trade_time.date()
        else:
            last_trade_date = pd.Timestamp(last_trade_time).date()

        if last_trade_date != TODAY.date():
            print(
                "🛑 Market holiday guard: "
                f"NIFTY 50 last traded on {last_trade_date}. "
                "No files will be changed."
            )
            return False

        return True

    except Exception as exc:
        print(
            "⚠️ Holiday guard could not be verified; "
            f"continuing cautiously: {type(exc).__name__}: {exc}"
        )
        return True


def fetch_quotes(kite: KiteConnect, symbols: list[str]) -> pd.DataFrame:
    quote_symbols = [f"NSE:{symbol}" for symbol in symbols]
    chunks = [
        quote_symbols[index:index + 200]
        for index in range(0, len(quote_symbols), 200)
    ]

    rows: list[dict] = []

    for chunk_number, chunk in enumerate(chunks, start=1):
        print(
            f"Fetching quote chunk {chunk_number}/{len(chunks)} "
            f"({len(chunk)} symbols)..."
        )

        response = None

        for attempt in range(5):
            try:
                response = kite.quote(chunk)
                if response:
                    break

            except Exception as exc:
                error_text = str(exc)
                print(
                    f"⚠️ Quote attempt {attempt + 1}/5 failed: "
                    f"{type(exc).__name__}: {error_text}"
                )

                if "429" in error_text or "403" in error_text:
                    time.sleep(2 ** attempt)
                else:
                    time.sleep(1)

        if not response:
            print(f"⚠️ No response for chunk {chunk_number}; continuing.")
            time.sleep(1)
            continue

        for full_symbol, data in response.items():
            ohlc = data.get("ohlc") or {}
            last_price = data.get("last_price")
            volume = data.get("volume")

            if last_price is None:
                continue

            rows.append(
                {
                    "Date": TODAY,
                    "Symbol": full_symbol.replace("NSE:", ""),
                    "Open": ohlc.get("open"),
                    "High": ohlc.get("high"),
                    "Low": ohlc.get("low"),
                    "Close": last_price,
                    "Volume": volume if volume is not None else 0,
                }
            )

        time.sleep(1.1)

    if not rows:
        fail("Kite returned no usable quote rows.")

    live = pd.DataFrame(rows)

    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
    for column in numeric_columns:
        live[column] = pd.to_numeric(live[column], errors="coerce")

    live = live.dropna(subset=["Symbol", "Close"])
    live["Date"] = pd.to_datetime(live["Date"]).dt.normalize()
    live = live.drop_duplicates(subset=["Date", "Symbol"], keep="last")

    if live.empty:
        fail("All quote rows were invalid after cleaning.")

    return live


def calculate_dashboard_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    frame["History_Days"] = frame.groupby("Symbol").cumcount() + 1
    frame["Prior_History_Days"] = frame["History_Days"] - 1

    frame["Daily_Turnover"] = frame["Close"] * frame["Volume"]

    frame["Prior_Turnover_20D_Avg"] = frame.groupby("Symbol")[
        "Daily_Turnover"
    ].transform(
        lambda values: values.shift(1).rolling(20, min_periods=1).mean()
    )

    mature_valid = (
        (frame["Prior_History_Days"] >= 20)
        & (frame["Prior_Turnover_20D_Avg"] >= 50_000_000)
    )

    new_valid = (
        (frame["Prior_History_Days"] >= 1)
        & (frame["Prior_History_Days"] < 20)
    )

    frame["Active_Universe"] = (
        (mature_valid | new_valid)
        & (frame["Volume"] > 0)
    )

    frame["Prev_Close"] = frame.groupby("Symbol")["Close"].shift(1)
    frame["Daily_Pct"] = (
        frame.groupby("Symbol")["Close"].pct_change() * 100
    )

    frame["Gainer"] = (
        frame["Active_Universe"]
        & (frame["Daily_Pct"] > 0)
    )

    frame["Loser"] = (
        frame["Active_Universe"]
        & (frame["Daily_Pct"] < 0)
    )

    return frame


def calculate_breadth(frame: pd.DataFrame) -> pd.DataFrame:
    today_rows = frame[frame["Date"] == TODAY].copy()

    if today_rows.empty:
        fail("No current-day rows available for breadth calculation.")

    advances = int(today_rows["Gainer"].sum())
    declines = int(today_rows["Loser"].sum())
    total_universe = int(today_rows["Active_Universe"].sum())
    unchanged = max(total_universe - advances - declines, 0)

    if advances == 0 and declines == 0:
        print(
            "🛑 Zero advances and declines detected. "
            "No output files will be changed."
        )
        sys.exit(0)

    return pd.DataFrame(
        [
            {
                "Time": NOW_IST.strftime("%H:%M"),
                "Advances": advances,
                "Declines": declines,
                "Unchanged": unchanged,
                "Total_Universe": total_universe,
                "Date": NOW_IST.strftime("%Y-%m-%d"),
            }
        ]
    )


def main() -> None:
    print("Connecting to Zerodha API for intraday snapshot...")

    if not CACHE_FILE.exists():
        fail(f"Historical cache missing: {CACHE_FILE.name}")

    historical = pd.read_parquet(CACHE_FILE)

    required_columns = {
        "Date",
        "Symbol",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    }

    missing_columns = required_columns.difference(historical.columns)
    if missing_columns:
        fail(
            "Historical cache is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    historical["Date"] = pd.to_datetime(
        historical["Date"], errors="coerce"
    ).dt.normalize()

    historical = historical.dropna(subset=["Date", "Symbol"])
    historical["Symbol"] = historical["Symbol"].astype(str).str.strip()

    symbols = sorted(
        symbol
        for symbol in historical["Symbol"].dropna().unique()
        if symbol
    )

    if not symbols:
        fail("No symbols found in trailing_cache.parquet.")

    kite = connect_to_kite()

    if not market_is_open_today(kite):
        return

    live = fetch_quotes(kite, symbols)

    # Preserve all historical dates and replace only today's snapshot.
    historical_without_today = historical[
        historical["Date"] != TODAY
    ].copy()

    combined = pd.concat(
        [historical_without_today, live],
        ignore_index=True,
        sort=False,
    )

    combined = (
        combined
        .sort_values(["Symbol", "Date"])
        .drop_duplicates(["Date", "Symbol"], keep="last")
        .reset_index(drop=True)
    )

    prepared = calculate_dashboard_columns(combined)
    breadth = calculate_breadth(prepared)

    # The separate live file is small and easy for Streamlit to load.
    atomic_write_parquet(live, LIVE_FILE)

    # This is the complete prepared dataset intended for the dashboard.
    atomic_write_parquet(prepared, DASHBOARD_FILE)

    # Keep the existing breadth output for compatibility.
    atomic_write_csv(breadth, BREADTH_FILE)

    SYNC_FILE.write_text(
        f"Today, {NOW_IST.strftime('%I:%M %p')} IST\n",
        encoding="utf-8",
    )

    print(f"✅ Live rows written: {len(live)}")
    print(f"✅ Prepared dashboard rows written: {len(prepared)}")
    print(f"✅ Dashboard file: {DASHBOARD_FILE.name}")
    print(f"✅ Breadth: {int(breadth.iloc[0]['Advances'])} advances, "
          f"{int(breadth.iloc[0]['Declines'])} declines")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(
            f"❌ Intraday update failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

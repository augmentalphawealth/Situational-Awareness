import datetime
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from kite_helper import fetch_with_backoff, get_kite_client

IST = ZoneInfo("Asia/Kolkata")
EOD_TIME = datetime.time(17, 0)

QUOTE_CHUNK_SIZE = 200
NSE_DOWNLOAD_MAX_RETRIES = 5
CALENDAR_LOOKBACK_DAYS = 15
BACKUP_RETENTION = 14
GAP_TRADING_DAYS_LIMIT = 10
MIN_EOD_COVERAGE = 0.97

NSE_EQUITY_LIST_URL = (
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
)
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/plain,*/*",
    "Referer": "https://www.nseindia.com/",
}

PARQUET_FILE = Path("nse_6yr_historical.parquet")
TMP_PARQUET = Path("nse_6yr_historical.tmp.parquet")
LOCK_FILE = Path("eod_sync.lock")
BACKUP_DIR = Path("eod_backups")
AUDIT_DIR = Path("eod_audits")

REQUIRED_COLUMNS = [
    "Symbol", "Date", "Open", "High", "Low", "Close", "Volume"
]


def die(message, code=1):
    print(message)
    raise SystemExit(code)


def normalize_date(value):
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def valid_date(value):
    return isinstance(value, pd.Timestamp) and not pd.isna(value)


def clean_symbol(value):
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def sha256_file(path):
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def read_lock(path):
    values = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if "=" in line:
                    key, value = line.rstrip("\n").split("=", 1)
                    values[key] = value
    except OSError as exc:
        die(f"Cannot read lock file: {exc}")
    return values


def acquire_lock(path):
    if path.exists():
        info = read_lock(path)
        pid_text = info.get("pid")
        host = info.get("host")
        if not pid_text:
            die("Lock file has no valid PID. Inspect it manually.")
        try:
            pid = int(pid_text)
        except ValueError:
            die("Lock file contains an invalid PID. Inspect it manually.")
        if host and host != socket.gethostname():
            die(f"Lock belongs to host {host}; refusing automatic removal.")
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            path.unlink(missing_ok=True)
            print(f"Removed stale lock for dead PID {pid}.")
        except PermissionError:
            die(f"Cannot verify lock owner PID {pid}; aborting.")
        except OSError as exc:
            die(f"Could not verify lock owner PID {pid}: {exc}")
        else:
            die(f"Another EOD process is active: PID {pid}.")
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            now = datetime.datetime.now(IST)
            handle.write(
                f"pid={os.getpid()}\n"
                f"host={socket.gethostname()}\n"
                f"time={now.isoformat()}\n"
            )
    except FileExistsError:
        die("Another process acquired the lock. Aborting.")


def release_lock(path):
    try:
        info = read_lock(path)
        if (
            info.get("pid") == str(os.getpid())
            and info.get("host") == socket.gethostname()
        ):
            path.unlink(missing_ok=True)
    except SystemExit:
        pass
    except Exception as exc:
        print(f"Lock cleanup failed: {exc}")


def call_kite(fetch_func, label, failures):
    result = fetch_with_backoff(fetch_func)
    if result is None:
        failures.append({"label": label, "error": "exhausted retries"})
    return result


def download_current_nse_eq_symbols():
    last_error = None
    for attempt in range(NSE_DOWNLOAD_MAX_RETRIES):
        try:
            response = requests.get(
                NSE_EQUITY_LIST_URL,
                headers=NSE_HEADERS,
                timeout=30,
            )
            response.raise_for_status()
            if not response.text.strip():
                raise ValueError("NSE EQUITY_L.csv response is empty.")
            df = pd.read_csv(StringIO(response.text))
            df.columns = [str(column).strip().upper() for column in df.columns]
            required = {"SYMBOL", "SERIES"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(
                    f"NSE EQUITY_L.csv missing required columns: {sorted(missing)}"
                )
            df["SYMBOL"] = df["SYMBOL"].map(clean_symbol)
            df["SERIES"] = df["SERIES"].astype(str).str.strip().str.upper()
            eq_df = df[(df["SERIES"] == "EQ") & (df["SYMBOL"] != "")]
            symbols = set(eq_df["SYMBOL"])
            if len(symbols) < 1000:
                raise ValueError(
                    f"NSE EQ symbol list looks incomplete: {len(symbols)} symbols."
                )
            return symbols
        except Exception as exc:
            last_error = exc
            print(
                f"Warning: NSE EQUITY_L.csv download failed "
                f"(attempt {attempt + 1}/{NSE_DOWNLOAD_MAX_RETRIES}): {exc}"
            )
            import time
            time.sleep(min(2 ** attempt, 16))
    die(f"Could not download reliable NSE EQUITY_L.csv: {last_error}")


def is_valid_new_ohlcv(row):
    try:
        open_price = float(row["Open"])
        high_price = float(row["High"])
        low_price = float(row["Low"])
        close_price = float(row["Close"])
        volume = float(row["Volume"])
        return (
            open_price > 0
            and high_price > 0
            and low_price > 0
            and close_price > 0
            and volume >= 0
            and high_price >= low_price
            and high_price >= max(open_price, close_price)
            and low_price <= min(open_price, close_price)
        )
    except (KeyError, TypeError, ValueError):
        return False


def validate_existing_structure(df):
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        die(f"Existing database missing columns: {sorted(missing)}")
    if df["Date"].isna().any():
        die("Existing database contains invalid dates.")
    if df.duplicated(["Symbol", "Date"]).any():
        die("Existing database contains duplicate Symbol/Date keys.")


def validate_new_data(df, name):
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        die(f"{name} missing columns: {sorted(missing)}")
    if df["Date"].isna().any():
        die(f"{name} contains invalid dates.")
    if df.duplicated(["Symbol", "Date"]).any():
        die(f"{name} contains duplicate Symbol/Date keys.")
    if not df.apply(is_valid_new_ohlcv, axis=1).all():
        die(f"{name} contains invalid newly fetched OHLCV rows.")


def candles_to_df(candles, symbol):
    if not candles:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    df = pd.DataFrame(candles).rename(
        columns={
            "date": "Date", "open": "Open", "high": "High",
            "low": "Low", "close": "Close", "volume": "Volume",
        }
    )
    history_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if set(history_columns) - set(df.columns):
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    df["Date"] = df["Date"].apply(normalize_date)
    df["Symbol"] = clean_symbol(symbol)
    return (
        df[REQUIRED_COLUMNS]
        .sort_values(["Symbol", "Date"])
        .drop_duplicates(["Symbol", "Date"], keep="last")
    )


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_jsonl(path, records):
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, default=str) + "\n" for record in records),
        encoding="utf-8",
    )


def rotate_backups():
    backups = sorted(
        BACKUP_DIR.glob("nse_6yr_historical.backup.*.parquet"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in backups[BACKUP_RETENTION:]:
        try:
            path.unlink()
        except OSError as exc:
            print(f"Warning: could not delete old backup {path}: {exc}")


def main():
    now = datetime.datetime.now(IST)
    today = normalize_date(now)
    fetch_today = now.weekday() < 5 and now.time() >= EOD_TIME
    mode = "eod" if fetch_today else "recovery_only"
    acquire_lock(LOCK_FILE)
    try:
        if TMP_PARQUET.exists():
            die(f"Temporary file exists: {TMP_PARQUET}. Inspect it first.")
        if not PARQUET_FILE.exists():
            die(f"Master database not found: {PARQUET_FILE}")

        source_hash = sha256_file(PARQUET_FILE)
        if source_hash is None:
            die("Could not hash source parquet.")
        df_hist = pd.read_parquet(PARQUET_FILE)
        if sha256_file(PARQUET_FILE) != source_hash:
            die("Source parquet changed while being loaded.")
        validate_existing_structure(df_hist)
        df_hist["Date"] = df_hist["Date"].apply(normalize_date)
        df_hist["Symbol"] = df_hist["Symbol"].map(clean_symbol)

        db_dates = set(df_hist["Date"])
        db_max_date = df_hist["Date"].max()
        historical_symbols = set(df_hist["Symbol"]) - {""}
        kite = get_kite_client()

        nifty_failures = []
        nifty_result = call_kite(
            lambda: kite.quote(["NSE:NIFTY 50"]),
            "NIFTY quote",
            nifty_failures,
        )
        nifty_data = nifty_result.get("NSE:NIFTY 50") if nifty_result else None
        if not nifty_data:
            die("Could not retrieve NIFTY 50 quote.")

        last_trade_date = normalize_date(nifty_data.get("last_trade_time"))
        nifty_token = nifty_data.get("instrument_token")
        if not nifty_token:
            die("NIFTY instrument token is missing.")

        calendar_failures = []
        calendar_start = db_max_date - datetime.timedelta(days=CALENDAR_LOOKBACK_DAYS)
        calendar = call_kite(
            lambda: kite.historical_data(
                nifty_token,
                calendar_start.strftime("%Y-%m-%d"),
                today.strftime("%Y-%m-%d"),
                "day",
            ),
            "NIFTY trading calendar",
            calendar_failures,
        )
        if not calendar or calendar_failures:
            die("Could not retrieve a reliable NIFTY trading calendar.")

        market_dates = sorted(
            {
                normalize_date(candle.get("date"))
                for candle in calendar
                if valid_date(normalize_date(candle.get("date")))
            }
        )
        if not market_dates:
            die("NIFTY historical calendar returned no valid dates.")

        latest_calendar_date = market_dates[-1]
        latest_market_date = (
            last_trade_date
            if valid_date(last_trade_date) and last_trade_date > latest_calendar_date
            else latest_calendar_date
        )

        if fetch_today and latest_market_date < today:
            die(
                "Today's EOD session is not confirmed. "
                f"Latest completed market date: {latest_market_date.date()}."
            )

        required_dates = [
            date for date in market_dates
            if db_max_date < date <= latest_market_date
        ]
        missing_dates = sorted(set(required_dates) - db_dates)
        historical_missing_dates = [
            date for date in missing_dates
            if not (fetch_today and date == latest_market_date)
        ]
        current_eod_date = (
            latest_market_date
            if fetch_today and latest_market_date not in db_dates
            else None
        )

        print(
            f"Mode={mode}; database_last={db_max_date.date()}; "
            f"latest_market_date={latest_market_date.date()}; "
            f"all_missing_dates={[d.date().isoformat() for d in missing_dates]}"
        )
        print(
            "Historical recovery dates: "
            f"{[d.date().isoformat() for d in historical_missing_dates]}"
        )
        print(
            "Current EOD date via batch quotes: "
            f"{current_eod_date.date().isoformat() if current_eod_date is not None else 'None'}"
        )

        if len(historical_missing_dates) > GAP_TRADING_DAYS_LIMIT:
            die(
                f"Historical gap of {len(historical_missing_dates)} trading days "
                f"exceeds safety limit {GAP_TRADING_DAYS_LIMIT}. Missing dates: "
                f"{[d.date().isoformat() for d in historical_missing_dates]}."
            )

        backfill_rows = []
        backfill_failures = []
        if historical_missing_dates:
            start_missing = min(historical_missing_dates)
            end_missing = max(historical_missing_dates)
            print(
                f"Recovering {len(historical_missing_dates)} historical date(s) "
                f"across {len(historical_symbols)} symbols "
                f"(~{len(historical_symbols) * 2} Kite calls)."
            )
            for symbol in sorted(historical_symbols):
                token_failures = []
                token_response = call_kite(
                    lambda symbol=symbol: kite.quote([f"NSE:{symbol}"]),
                    f"token {symbol}",
                    token_failures,
                )
                token_data = (
                    token_response.get(f"NSE:{symbol}")
                    if token_response else None
                )
                token = token_data.get("instrument_token") if token_data else None
                if token_failures or not token:
                    backfill_failures.append({
                        "Symbol": symbol,
                        "Reason": "instrument_token_failure",
                        "Failures": token_failures,
                    })
                    continue

                candles = call_kite(
                    lambda token=token: kite.historical_data(
                        token,
                        start_missing.strftime("%Y-%m-%d"),
                        end_missing.strftime("%Y-%m-%d"),
                        "day",
                    ),
                    f"backfill {symbol}",
                    backfill_failures,
                )
                temp = candles_to_df(candles, symbol)
                temp = temp[temp["Date"].isin(historical_missing_dates)]
                if temp.empty:
                    backfill_failures.append({
                        "Symbol": symbol,
                        "Reason": "empty_backfill_response",
                    })
                    continue
                temp = temp[temp.apply(is_valid_new_ohlcv, axis=1)]
                backfill_rows.extend(temp.to_dict("records"))
                returned_dates = set(temp["Date"])
                for date in sorted(set(historical_missing_dates) - returned_dates):
                    backfill_failures.append({
                        "Symbol": symbol,
                        "Date": date,
                        "Reason": "missing_backfill_candle",
                    })

        backfill_df = pd.DataFrame(backfill_rows, columns=REQUIRED_COLUMNS)
        actual_pairs = (
            set(zip(backfill_df["Symbol"], backfill_df["Date"]))
            if not backfill_df.empty else set()
        )
        required_pairs = {
            (symbol, date)
            for symbol in historical_symbols
            for date in historical_missing_dates
        }
        missing_pairs = required_pairs - actual_pairs

        if backfill_failures or missing_pairs:
            records = backfill_failures + [
                {
                    "Symbol": symbol,
                    "Date": date,
                    "Reason": "required_pair_missing",
                }
                for symbol, date in sorted(missing_pairs)
            ]
            write_jsonl(
                AUDIT_DIR / f"backfill_failures_{now.strftime('%Y%m%d_%H%M%S')}.jsonl",
                records,
            )
            print(
                f"⚠️ Historical recovery had {len(backfill_failures)} failures "
                f"and {len(missing_pairs)} missing pairs; continuing."
            )

        working = pd.concat([df_hist, backfill_df], ignore_index=True)
        working = (
            working.sort_values(["Symbol", "Date"])
            .drop_duplicates(["Symbol", "Date"], keep="last")
        )

        latest_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        eod_coverage = None
        eod_required_symbols = set()
        missing_latest = set()
        critical_missing = set()
        latest_failures = []

        if fetch_today:
            eod_required_symbols = download_current_nse_eq_symbols()
            if len(eod_required_symbols) < 1000:
                die("Current NSE EQ universe is unexpectedly small. Refusing EOD update.")
            print(f"Current NSE mainboard EQ universe: {len(eod_required_symbols)} symbols.")

            latest_rows = []
            symbols = sorted(eod_required_symbols)
            quote_batches = (len(symbols) + QUOTE_CHUNK_SIZE - 1) // QUOTE_CHUNK_SIZE
            print(f"Current EOD quote fetch: {len(symbols)} symbols in {quote_batches} batches.")

            for start in range(0, len(symbols), QUOTE_CHUNK_SIZE):
                chunk = symbols[start:start + QUOTE_CHUNK_SIZE]
                result = call_kite(
                    lambda chunk=chunk: kite.quote([f"NSE:{symbol}" for symbol in chunk]),
                    f"latest quotes {start // QUOTE_CHUNK_SIZE + 1}",
                    latest_failures,
                )
                if result:
                    for instrument_key, data in result.items():
                        symbol = clean_symbol(instrument_key.replace("NSE:", "", 1))
                        if symbol not in eod_required_symbols:
                            continue
                        ohlc = data.get("ohlc") or {}
                        latest_rows.append({
                            "Date": latest_market_date,
                            "Open": ohlc.get("open", 0),
                            "High": ohlc.get("high", 0),
                            "Low": ohlc.get("low", 0),
                            "Close": data.get("last_price", 0),
                            "Volume": data.get("volume", 0),
                            "Symbol": symbol,
                        })

            latest_df = pd.DataFrame(latest_rows, columns=REQUIRED_COLUMNS)
            if not latest_df.empty:
                latest_df["Symbol"] = latest_df["Symbol"].map(clean_symbol)
                latest_df = latest_df[latest_df.apply(is_valid_new_ohlcv, axis=1)]
                latest_df = latest_df.drop_duplicates(["Symbol", "Date"], keep="last")

            latest_symbols = set(latest_df["Symbol"]) if not latest_df.empty else set()
            missing_latest = eod_required_symbols - latest_symbols
            valid_count = len(latest_symbols)
            required_count = len(eod_required_symbols)
            eod_coverage = valid_count / required_count if required_count else 0.0

            critical = {
                clean_symbol(symbol)
                for symbol in os.environ.get(
                    "EOD_CRITICAL_SYMBOLS",
                    "RELIANCE,HDFCBANK,ICICIBANK,INFY,TCS",
                ).split(",")
                if clean_symbol(symbol)
            }
            critical_not_in_eq_universe = critical - eod_required_symbols
            critical_missing = critical & missing_latest

            print(f"Latest valid EQ quotes: {valid_count}/{required_count} ({eod_coverage:.2%}).")

            if missing_latest:
                write_jsonl(
                    AUDIT_DIR / f"latest_eod_missing_eq_{now.strftime('%Y%m%d_%H%M%S')}.jsonl",
                    [
                        {
                            "Symbol": symbol,
                            "Date": latest_market_date,
                            "Reason": "current_nse_eq_symbol_missing_or_invalid_quote",
                        }
                        for symbol in sorted(missing_latest)
                    ]
                    + [
                        {
                            "Failure": failure,
                            "Date": latest_market_date,
                            "Reason": "quote_chunk_failure",
                        }
                        for failure in latest_failures
                    ],
                )

            if critical_not_in_eq_universe:
                die(
                    "Critical symbols are not present in NSE current EQ universe: "
                    f"{sorted(critical_not_in_eq_universe)}. Database unchanged."
                )
            if critical_missing:
                die(
                    f"Critical EOD symbols missing or invalid: {sorted(critical_missing)}. "
                    "Database unchanged."
                )
            if eod_coverage < MIN_EOD_COVERAGE:
                die(
                    f"EOD coverage below required {MIN_EOD_COVERAGE:.0%}: "
                    f"{valid_count}/{required_count} ({eod_coverage:.2%}). "
                    "Database unchanged."
                )

            validate_new_data(latest_df, "Latest EOD data")
            valid_today_symbols = set(latest_df["Symbol"])
            working = working[
                ~(
                    (working["Date"] == latest_market_date)
                    & working["Symbol"].isin(valid_today_symbols)
                )
            ]
            working = pd.concat([working, latest_df], ignore_index=True)

        if working.duplicated(["Symbol", "Date"]).any():
            die("Combined database contains duplicate Symbol/Date keys.")
        if working["Date"].isna().any():
            die("Combined database contains invalid dates.")

        working.to_parquet(TMP_PARQUET, index=False)
        check_df = pd.read_parquet(TMP_PARQUET)
        if check_df.empty or check_df["Date"].isna().any():
            die("Temporary database read-back validation failed.")
        if check_df.duplicated(["Symbol", "Date"]).any():
            die("Temporary database contains duplicate keys.")
        if sha256_file(PARQUET_FILE) != source_hash:
            die("Source parquet changed before backup. Aborting.")

        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup_path = BACKUP_DIR / (
            f"nse_6yr_historical.backup.{now.strftime('%Y%m%d_%H%M%S')}.parquet"
        )
        backup_tmp = backup_path.with_suffix(backup_path.suffix + ".tmp")
        shutil.copy2(PARQUET_FILE, backup_tmp)
        if sha256_file(backup_tmp) != source_hash:
            backup_tmp.unlink(missing_ok=True)
            die("Backup hash mismatch. Original database untouched.")

        os.replace(backup_tmp, backup_path)
        os.replace(TMP_PARQUET, PARQUET_FILE)
        rotate_backups()

        print(
            f"Saved successfully. mode={mode}; "
            f"all_missing_dates={len(missing_dates)}; "
            f"historical_recovery_dates={len(historical_missing_dates)}; "
            f"today_fetched={fetch_today}; "
            f"eod_coverage={f'{eod_coverage:.2%}' if eod_coverage is not None else 'N/A'}"
        )

        result = subprocess.run(
            [sys.executable, "analyze_6yr_data.py"],
            check=False,
        )
        if result.returncode != 0:
            die(
                f"analyze_6yr_data.py failed with exit code {result.returncode}. "
                f"Database updated; backup preserved at {backup_path}."
            )

        write_json(
            AUDIT_DIR / f"eod_run_{now.strftime('%Y%m%d_%H%M%S')}.json",
            {
                "mode": mode,
                "database_before": str(db_max_date.date()),
                "latest_market_date": str(latest_market_date.date()),
                "all_missing_dates": [str(date.date()) for date in missing_dates],
                "historical_recovery_dates": [str(date.date()) for date in historical_missing_dates],
                "current_eod_date": str(current_eod_date.date()) if current_eod_date is not None else None,
                "backfill_rows": len(backfill_df),
                "today_fetched": fetch_today,
                "backup": str(backup_path),
                "backfill_failures": len(backfill_failures),
                "missing_pairs": len(missing_pairs),
                "nse_current_eq_symbols": len(eod_required_symbols),
                "latest_valid_eq_symbols": len(latest_df) if fetch_today else None,
                "latest_missing_eq_symbols": len(missing_latest) if fetch_today else None,
                "latest_eod_coverage": round(eod_coverage, 6) if eod_coverage is not None else None,
                "minimum_eod_coverage": MIN_EOD_COVERAGE,
                "critical_missing": sorted(critical_missing),
                "quote_chunk_failures": len(latest_failures),
                "gap_trading_days_limit": GAP_TRADING_DAYS_LIMIT,
            },
        )
    finally:
        release_lock(LOCK_FILE)


if __name__ == "__main__":
    main()

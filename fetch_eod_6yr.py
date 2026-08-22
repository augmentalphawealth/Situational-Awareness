import datetime
import hashlib
import json
import os
import shutil
import socket
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from kiteconnect import KiteConnect


# -----------------------------
# Configuration
# -----------------------------
IST = ZoneInfo("Asia/Kolkata")
MARKET_CLOSE_GUARD = datetime.time(16, 0)
QUOTE_CHUNK_SIZE = 200
MAX_RETRIES = 5
QUOTE_SLEEP_SECONDS = 1.1
HISTORY_SLEEP_SECONDS = 0.4
MARKET_LOOKBACK_DAYS = 15

PARQUET_FILE = Path("nse_6yr_historical.parquet")
TMP_PARQUET = Path("nse_6yr_historical.tmp.parquet")
LOCK_FILE = Path("eod_sync.lock")
AUDIT_DIR = Path("eod_audits")
BACKUP_DIR = Path("eod_backups")

REQUIRED_COLUMNS = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]
NUMERIC_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
HISTORY_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]

APPROVED_RATIOS = {
    0.1, 0.125, 0.1667, 0.2, 0.25, 0.3333, 0.5,
    0.6667, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0,
}


# -----------------------------
# Basic helpers
# -----------------------------
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


def valid_timestamp(value):
    return isinstance(value, pd.Timestamp) and not pd.isna(value)


def sha256_file(path):
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def parse_lock(lock_file):
    try:
        with open(lock_file, "r", encoding="utf-8") as handle:
            values = {}
            for line in handle:
                if "=" in line:
                    key, value = line.rstrip("\n").split("=", 1)
                    values[key] = value
            return values
    except OSError as exc:
        die(f"🛑 Could not read lock file: {exc}")


def acquire_lock(lock_file):
    if lock_file.exists():
        info = parse_lock(lock_file)
        pid_text = info.get("pid")
        host = info.get("host")
        current_host = socket.gethostname()

        if not pid_text:
            die("🛑 Lock file has no valid PID. Inspect/delete it manually.")

        try:
            pid = int(pid_text)
        except ValueError:
            die("🛑 Lock file contains an invalid PID. Inspect/delete it manually.")

        if host and host != current_host:
            die(
                f"🛑 Lock belongs to host {host}; current host is {current_host}. "
                "Aborting for safety."
            )

        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            print(f"⚠️ Stale lock detected: PID {pid} is not running. Recovering...")
            try:
                lock_file.unlink()
            except OSError as exc:
                die(f"🛑 Could not remove stale lock: {exc}")
        except PermissionError:
            die(f"🛑 Cannot verify lock owner PID {pid}. Aborting.")
        except OSError as exc:
            die(f"🛑 Could not verify lock owner PID {pid}: {exc}")
        else:
            die(f"🛑 CONCURRENCY GUARD: PID {pid} is active. Aborting.")

    try:
        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            now = datetime.datetime.now(IST)
            handle.write(
                f"pid={os.getpid()}\n"
                f"host={socket.gethostname()}\n"
                f"time={now.isoformat()}\n"
            )
    except FileExistsError:
        die("🛑 Another process acquired the lock. Aborting.")


def release_lock(lock_file):
    try:
        info = parse_lock(lock_file)
        if info.get("pid") == str(os.getpid()) and info.get("host") == socket.gethostname():
            lock_file.unlink(missing_ok=True)
    except SystemExit:
        pass
    except Exception as exc:
        print(f"⚠️ Lock cleanup failed: {exc}")


def is_valid_ohlcv(row):
    try:
        return bool(
            float(row["Open"]) > 0
            and float(row["High"]) > 0
            and float(row["Low"]) > 0
            and float(row["Close"]) > 0
            and float(row["High"]) >= max(
                float(row["Open"]), float(row["Close"]), float(row["Low"])
            )
            and float(row["Low"]) <= min(
                float(row["Open"]), float(row["Close"]), float(row["High"])
            )
            and float(row["Volume"]) >= 0
        )
    except (TypeError, ValueError, KeyError):
        return False


def validate_dataframe(df, name):
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        die(f"❌ {name} missing columns: {sorted(missing)}")

    if df[NUMERIC_COLUMNS].isna().any().any():
        die(f"❌ {name} contains null numeric values.")

    if df["Date"].isna().any():
        die(f"❌ {name} contains invalid dates.")

    if df.duplicated(["Symbol", "Date"]).any():
        die(f"❌ {name} contains duplicate Symbol/Date keys.")

    valid = df.apply(is_valid_ohlcv, axis=1)
    if not valid.all():
        die(f"❌ {name} contains invalid OHLCV rows.")


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def fetch_with_retries(fetcher, label, failures, attempts=MAX_RETRIES):
    last_error = None
    for attempt in range(attempts):
        try:
            return fetcher()
        except Exception as exc:
            last_error = exc
            text = str(exc)
            if "429" in text or "403" in text:
                time.sleep(2 ** attempt)
            else:
                time.sleep(min(2 ** attempt, 8))

    failures.append({"label": label, "error": str(last_error)})
    return None


def ensure_source_unchanged(path, expected_hash):
    current_hash = sha256_file(path)
    if current_hash is None or current_hash != expected_hash:
        die("❌ Source parquet changed or became unreadable during execution.")


# -----------------------------
# Main synchronization
# -----------------------------
def main():
    now_ist = datetime.datetime.now(IST)
    today_ist = normalize_date(now_ist)

    if now_ist.weekday() >= 5:
        die(
            f"🛑 WEEKEND GUARD: {now_ist.strftime('%A, %d %b %Y')} is not a trading day.",
            0,
        )

    if now_ist.time() < MARKET_CLOSE_GUARD:
        die(
            f"🛑 TOO EARLY: {now_ist.strftime('%H:%M')} IST. "
            "Run after 16:00 IST.",
            0,
        )

    acquire_lock(LOCK_FILE)
    try:
        api_key = os.environ.get("KITE_API_KEY")
        access_token = os.environ.get("KITE_ACCESS_TOKEN")
        if not api_key or not access_token:
            die("❌ KITE_API_KEY or KITE_ACCESS_TOKEN missing.")

        if not PARQUET_FILE.exists():
            die(f"❌ Master database not found: {PARQUET_FILE}")
        if TMP_PARQUET.exists():
            die(f"🛑 Temporary file exists: {TMP_PARQUET}. Inspect/delete it first.")

        source_hash_before = sha256_file(PARQUET_FILE)
        if source_hash_before is None:
            die("❌ Could not hash the source parquet.")

        df_hist = pd.read_parquet(PARQUET_FILE)
        source_hash_after = sha256_file(PARQUET_FILE)
        if source_hash_before != source_hash_after:
            die("❌ Source parquet changed while being loaded. Aborting.")

        required_missing = set(REQUIRED_COLUMNS) - set(df_hist.columns)
        if required_missing:
            die(f"❌ Existing database missing columns: {sorted(required_missing)}")

        df_hist["Date"] = df_hist["Date"].apply(normalize_date)
        validate_dataframe(df_hist, "Existing database")

        original_db_len = len(df_hist)
        original_db_keys = (
            df_hist[["Symbol", "Date"]]
            .sort_values(["Symbol", "Date"])
            .reset_index(drop=True)
        )
        db_max_date = df_hist["Date"].max()
        first_dates = df_hist.groupby("Symbol")["Date"].min().to_dict()

        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)

        nifty_response = kite.quote(["NSE:NIFTY 50"])
        nifty_data = nifty_response.get("NSE:NIFTY 50")
        if not nifty_data:
            die("🛑 Could not retrieve NIFTY 50 quote. Aborting.")

        ltt = nifty_data.get("last_trade_time")
        true_latest_market_date = normalize_date(ltt)
        if not valid_timestamp(true_latest_market_date):
            die("🛑 NIFTY last_trade_time is missing or invalid.")

        if true_latest_market_date < today_ist:
            die(
                f"🛑 MARKET HOLIDAY GUARD: Last NIFTY trade was "
                f"{true_latest_market_date.date()}; today is {today_ist.date()}.",
                0,
            )

        if db_max_date >= true_latest_market_date:
            die(
                f"🛑 UP-TO-DATE: Database={db_max_date.date()}, "
                f"market={true_latest_market_date.date()}.",
                0,
            )

        nifty_token = nifty_data.get("instrument_token")
        if not nifty_token:
            die("❌ NIFTY instrument token is missing.")

        calendar_failures = []
        nifty_hist = fetch_with_retries(
            lambda: kite.historical_data(
                nifty_token,
                (true_latest_market_date - datetime.timedelta(days=MARKET_LOOKBACK_DAYS)).strftime("%Y-%m-%d"),
                true_latest_market_date.strftime("%Y-%m-%d"),
                "day",
            ),
            "NIFTY calendar",
            calendar_failures,
        )
        if not nifty_hist:
            die("❌ Could not retrieve NIFTY historical calendar.")

        valid_dates = [normalize_date(candle.get("date")) for candle in nifty_hist]
        valid_dates = [date for date in valid_dates if valid_timestamp(date)]
        past_dates = [date for date in valid_dates if date < true_latest_market_date]
        target_prev_date = max(past_dates) if past_dates else None
        if not valid_timestamp(target_prev_date):
            die("❌ No valid previous market date found.")

        if db_max_date < target_prev_date:
            die(
                f"🛑 FATAL GAP: Database ends {db_max_date.date()}, "
                f"previous market date is {target_prev_date.date()}. "
                "Run the historical backfill/rebuild process."
            )

        expected_symbols = set(df_hist["Symbol"].astype(str))
        chunks = [
            [f"NSE:{symbol}" for symbol in list(expected_symbols)[i:i + QUOTE_CHUNK_SIZE]]
            for i in range(0, len(expected_symbols), QUOTE_CHUNK_SIZE)
        ]

        new_rows = []
        heal_queue = {}
        missing_prev_date = set()
        failed_chunks = []
        seen_symbols = set()

        print(f"Fetching EOD quotes for {true_latest_market_date.date()}...")
        for chunk_number, chunk in enumerate(chunks, start=1):
            result = fetch_with_retries(
                lambda chunk=chunk: kite.quote(chunk),
                f"quote chunk {chunk_number}",
            )
            if not result:
                failed_chunks.append(chunk)
                continue

            for instrument_key, data in result.items():
                symbol = instrument_key.replace("NSE:", "", 1)
                if symbol not in expected_symbols:
                    print(f"⚠️ Ignoring unexpected symbol: {instrument_key}")
                    continue
                if symbol in seen_symbols:
                    print(f"⚠️ Ignoring duplicate quote: {symbol}")
                    continue
                seen_symbols.add(symbol)

                ohlc = data.get("ohlc") or {}
                row = {
                    "Date": true_latest_market_date,
                    "Open": ohlc.get("open", 0),
                    "High": ohlc.get("high", 0),
                    "Low": ohlc.get("low", 0),
                    "Close": data.get("last_price", 0),
                    "Volume": data.get("volume", 0),
                    "Symbol": symbol,
                }
                new_rows.append(row)

                if not is_valid_ohlcv(row):
                    continue

                db_row = df_hist[
                    (df_hist["Symbol"] == symbol)
                    & (df_hist["Date"] == target_prev_date)
                ]
                if db_row.empty:
                    first_date = first_dates.get(symbol)
                    if first_date is not None and first_date <= target_prev_date:
                        missing_prev_date.add(symbol)
                    continue

                db_prev_close = db_row["Close"].iloc[0]
                kite_prev_close = ohlc.get("close", 0)
                if (
                    pd.notna(db_prev_close)
                    and pd.notna(kite_prev_close)
                    and db_prev_close > 0
                    and kite_prev_close > 0
                    and abs(db_prev_close - kite_prev_close) / db_prev_close > 0.02
                ):
                    heal_queue[symbol] = data.get("instrument_token")

            time.sleep(QUOTE_SLEEP_SECONDS)

        valid_rows = [row for row in new_rows if is_valid_ohlcv(row)]
        returned_symbols = {row["Symbol"] for row in new_rows}
        fetched_symbols = {row["Symbol"] for row in valid_rows}
        omitted_symbols = expected_symbols - returned_symbols
        invalid_symbols = returned_symbols - fetched_symbols

        print(f"Omitted quote symbols: {len(omitted_symbols)}")
        print(f"Invalid quote symbols: {len(invalid_symbols)}")
        print(f"Missing previous-date symbols: {len(missing_prev_date)}")

        env_critical = os.environ.get(
            "EOD_CRITICAL_SYMBOLS",
            "RELIANCE,HDFCBANK,ICICIBANK,INFY,TCS",
        )
        critical_symbols = {
            symbol.strip()
            for symbol in env_critical.split(",")
            if symbol.strip()
        }
        missing_critical = critical_symbols & (
            omitted_symbols | invalid_symbols | missing_prev_date
        )
        if missing_critical:
            die(f"❌ Critical symbols missing or invalid: {sorted(missing_critical)}")

        omitted_limit = max(1, int(0.01 * len(expected_symbols)))
        invalid_limit = max(1, int(0.01 * len(expected_symbols)))
        prev_limit = max(1, int(0.02 * len(expected_symbols)))
        if len(omitted_symbols) > omitted_limit:
            die(f"❌ Too many omitted quotes: {len(omitted_symbols)} > {omitted_limit}")
        if len(invalid_symbols) > invalid_limit:
            die(f"❌ Too many invalid quotes: {len(invalid_symbols)} > {invalid_limit}")
        if len(missing_prev_date) > prev_limit:
            die(f"❌ Too many missing previous-date rows: {len(missing_prev_date)} > {prev_limit}")

        healing_audit = []
        healed_data = []
        healed_symbols = set()
        healing_failures = []
        healing_candidates = len(heal_queue)
        circuit_limit = max(50, int(0.02 * len(expected_symbols)))

        if healing_candidates > circuit_limit:
            print(
                f"🛑 CIRCUIT BREAKER: {healing_candidates} healing candidates "
                f"exceed limit {circuit_limit}. Skipping all healing."
            )
            heal_queue.clear()

        for symbol, token in heal_queue.items():
            audit = {
                "Symbol": symbol,
                "Token": token,
                "Accepted": False,
                "RejectionReason": None,
                "OldRows": 0,
                "NewRows": 0,
                "OverlapRows": 0,
                "RatioMedian": None,
                "RatioP95Deviation": None,
                "HistoryFailures": [],
            }

            if not token:
                audit["RejectionReason"] = "missing_instrument_token"
                healing_audit.append(audit)
                continue

            stock_data = []
            history_failures = []
            for start, end in [
                ("2018-04-01", "2022-03-31"),
                ("2022-04-01", true_latest_market_date.strftime("%Y-%m-%d")),
            ]:
                response = fetch_with_retries(
                    lambda token=token, start=start, end=end: kite.historical_data(
                        token, start, end, "day"
                    ),
                    f"history {symbol} {start}:{end}",
                    history_failures,
                )
                if response:
                    stock_data.extend(response)
                time.sleep(HISTORY_SLEEP_SECONDS)

            audit["HistoryFailures"] = history_failures
            if history_failures:
                audit["RejectionReason"] = "historical_api_failure"
            elif not stock_data:
                audit["RejectionReason"] = "empty_historical_response"

            if stock_data and not history_failures:
                df_temp = pd.DataFrame(stock_data).rename(
                    columns={
                        "date": "Date",
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume",
                    }
                )
                missing_history_cols = set(HISTORY_COLUMNS) - set(df_temp.columns)
                if missing_history_cols:
                    audit["RejectionReason"] = "missing_history_columns"
                else:
                    df_temp["Date"] = df_temp["Date"].apply(normalize_date)
                    df_temp["Symbol"] = symbol
                    df_temp = (
                        df_temp.sort_values(["Symbol", "Date"])
                        .drop_duplicates(["Symbol", "Date"], keep="last")
                    )
                    audit["NewRows"] = len(df_temp)

                    valid_temp = (
                        not df_temp.empty
                        and not df_temp[HISTORY_COLUMNS].isna().any().any()
                        and df_temp.apply(is_valid_ohlcv, axis=1).all()
                        and df_temp["Date"].max() == true_latest_market_date
                    )
                    if not valid_temp:
                        audit["RejectionReason"] = "invalid_historical_ohlcv"
                    else:
                        unique_dates = pd.Series(sorted(df_temp["Date"].unique()))
                        if (
                            unique_dates.diff().dropna()
                            > pd.Timedelta(days=15)
                        ).any():
                            audit["RejectionReason"] = "large_date_gap"
                        else:
                            old_overlap = df_hist.loc[
                                df_hist["Symbol"] == symbol,
                                ["Date", "Close"],
                            ]
                            audit["OldRows"] = len(old_overlap)
                            new_overlap = df_temp[["Date", "Close"]]
                            overlap = old_overlap.merge(
                                new_overlap,
                                on="Date",
                                suffixes=("_old", "_new"),
                            )
                            audit["OverlapRows"] = len(overlap)

                            old_dates = set(old_overlap["Date"])
                            new_dates = set(new_overlap["Date"])
                            missing_old = old_dates - new_dates
                            if len(missing_old) > 10:
                                audit["RejectionReason"] = "old_dates_lost"
                            elif len(old_overlap) > 250 and len(df_temp) < 0.90 * len(old_overlap):
                                audit["RejectionReason"] = "replacement_too_short"
                            elif len(overlap) < 10:
                                audit["RejectionReason"] = "insufficient_overlap"
                            elif len(old_overlap) > 250 and len(overlap) < 200:
                                audit["RejectionReason"] = "insufficient_mature_overlap"
                            else:
                                old_close = overlap["Close_old"].to_numpy(dtype=float)
                                new_close = overlap["Close_new"].to_numpy(dtype=float)
                                if (old_close <= 0).any() or (new_close <= 0).any():
                                    audit["RejectionReason"] = "non_positive_overlap_close"
                                else:
                                    ratios = new_close / old_close
                                    ratio_median = float(np.median(ratios))
                                    ratio_p95 = float(
                                        np.quantile(
                                            np.abs(ratios - ratio_median) / ratio_median,
                                            0.95,
                                        )
                                    )
                                    audit["RatioMedian"] = ratio_median
                                    audit["RatioP95Deviation"] = ratio_p95
                                    ratio_allowed = any(
                                        abs(ratio_median - approved) <= 0.03
                                        for approved in APPROVED_RATIOS
                                    )
                                    if not np.isfinite(ratios).all() or ratio_median <= 0:
                                        audit["RejectionReason"] = "invalid_adjustment_ratio"
                                    elif ratio_p95 > 0.10:
                                        audit["RejectionReason"] = "erratic_adjustment_ratio"
                                    elif not ratio_allowed:
                                        audit["RejectionReason"] = "unapproved_adjustment_ratio"
                                    else:
                                        latest_new = df_temp.loc[
                                            df_temp["Date"] == true_latest_market_date,
                                            "Close",
                                        ]
                                        latest_quote = next(
                                            (
                                                row["Close"]
                                                for row in valid_rows
                                                if row["Symbol"] == symbol
                                            ),
                                            None,
                                        )
                                        if latest_new.empty or latest_quote is None:
                                            audit["RejectionReason"] = "missing_latest_close"
                                        elif abs(latest_new.iloc[-1] - latest_quote) / latest_quote > 0.05:
                                            audit["RejectionReason"] = "latest_close_mismatch"
                                        else:
                                            audit["Accepted"] = True
                                            healed_data.append(df_temp)
                                            healed_symbols.add(symbol)

            healing_audit.append(audit)

        if healing_audit:
            AUDIT_DIR.mkdir(parents=True, exist_ok=True)
            audit_path = AUDIT_DIR / f"healing_audit_{now_ist.strftime('%Y%m%d_%H%M%S')}.jsonl"
            with open(audit_path, "w", encoding="utf-8") as handle:
                for record in healing_audit:
                    handle.write(json.dumps(record, default=str) + "\n")
            print(
                f"Healing summary: candidates={healing_candidates}, "
                f"accepted={len(healed_symbols)}, "
                f"rejected={healing_candidates - len(healed_symbols)}"
            )

        if healed_symbols:
            df_hist = df_hist[~df_hist["Symbol"].isin(healed_symbols)]
            df_hist = pd.concat(
                [df_hist, pd.concat(healed_data, ignore_index=True)],
                ignore_index=True,
            )

        new_eod_df = pd.DataFrame(valid_rows)
        df_hist = df_hist[df_hist["Date"] != true_latest_market_date]
        combined = pd.concat([df_hist, new_eod_df], ignore_index=True)
        combined = (
            combined.sort_values(["Symbol", "Date"])
            .drop_duplicates(["Symbol", "Date"], keep="last")
        )

        combined.to_parquet(TMP_PARQUET, index=False)
        check_df = pd.read_parquet(TMP_PARQUET)
        validate_dataframe(check_df, "Temporary database")

        latest_rows = check_df[check_df["Date"] == true_latest_market_date]
        latest_symbols = set(latest_rows["Symbol"])
        if not latest_symbols.issubset(expected_symbols):
            die("❌ Temporary database contains unexpected latest-date symbols.")
        if CRITICAL_SYMBOLS & (expected_symbols - latest_symbols):
            die("❌ Critical symbols missing from temporary database.")
        if len(latest_symbols) < 0.98 * len(expected_symbols):
            die("❌ Temporary database latest-date coverage is below 98%.")

        ensure_source_unchanged(PARQUET_FILE, original_db_hash)

        backup_path = BACKUP_DIR / (
            f"nse_6yr_historical.backup."
            f"{now_ist.strftime('%Y%m%d_%H%M%S')}.parquet"
        )
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup_tmp = backup_path.with_suffix(backup_path.suffix + ".tmp")
        shutil.copy2(PARQUET_FILE, backup_tmp)
        if sha256_file(backup_tmp) != original_db_hash:
            backup_tmp.unlink(missing_ok=True)
            die("❌ Backup hash mismatch. Original database untouched.")
        os.replace(backup_tmp, backup_path)

        os.replace(TMP_PARQUET, PARQUET_FILE)
        print(f"✅ EOD database saved. Backup: {backup_path}")

        exit_code = os.system("python analyze_6yr_data.py")
        if exit_code != 0:
            die(
                f"❌ analyze_6yr_data.py failed with exit code {exit_code}. "
                f"Master database updated; backup preserved at {backup_path}."
            )

    finally:
        release_lock(LOCK_FILE)


if __name__ == "__main__":
    main()

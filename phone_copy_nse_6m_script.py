import io
import time
import zipfile
from datetime import date, timedelta

import pandas as pd
import requests

OUT_CSV = 'nse_last_6_months_eq.csv'
START_DATE = date.today() - timedelta(days=190)
END_DATE = date.today()

HEADERS = {
    'User-Agent': 'Mozilla/5.0',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.nseindia.com/',
}

session = requests.Session()
session.headers.update(HEADERS)
session.get('https://www.nseindia.com', timeout=20)


def business_days(start, end):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def normalize_cols(df):
    df.columns = [c.strip().upper().replace(' ', '_') for c in df.columns]
    return df


def parse_old_bhavcopy(content):
    df = pd.read_csv(io.BytesIO(content))
    df = normalize_cols(df)
    rename = {
        'SYMBOL': 'Symbol',
        'SERIES': 'Series',
        'OPEN': 'Open',
        'HIGH': 'High',
        'LOW': 'Low',
        'CLOSE': 'Close',
        'TOTTRDQTY': 'Volume',
        'TOTTRDVAL': 'Turnover',
    }
    df = df.rename(columns=rename)
    keep = [c for c in ['Symbol', 'Series', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover'] if c in df.columns]
    return df[keep]


def parse_new_bhavcopy(content):
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        csv_name = [n for n in zf.namelist() if n.lower().endswith('.csv')][0]
        df = pd.read_csv(zf.open(csv_name))
    df = normalize_cols(df)
    rename = {
        'TCKR_SYMB': 'Symbol',
        'SYMBOL': 'Symbol',
        'SCTY_SRS': 'Series',
        'OPEN_PRICE': 'Open',
        'HIGH_PRICE': 'High',
        'LOW_PRICE': 'Low',
        'CLOSE_PRICE': 'Close',
        'TTL_TRD_QNTY': 'Volume',
        'TTL_TRD_VAL': 'Turnover',
    }
    df = df.rename(columns=rename)
    keep = [c for c in ['Symbol', 'Series', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover'] if c in df.columns]
    return df[keep]


def fetch_one(dt):
    ymd = dt.strftime('%Y%m%d')
    dmy = dt.strftime('%d%m%Y')
    new_url = f'https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip'
    old_url = f'https://archives.nseindia.com/products/content/sec_bhavdata_full_{dmy}.csv'

    for kind, url in [('new', new_url), ('old', old_url)]:
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 200 and len(r.content) > 100:
                df = parse_new_bhavcopy(r.content) if kind == 'new' else parse_old_bhavcopy(r.content)
                if not df.empty:
                    df['Date'] = pd.Timestamp(dt)
                    return df
        except Exception:
            pass
    return None


frames = []
for dt in business_days(START_DATE, END_DATE):
    df = fetch_one(dt)
    if df is not None:
        df = df[df['Series'].astype(str).str.upper().eq('EQ')].copy()
        frames.append(df)
    time.sleep(0.25)

if not frames:
    raise ValueError('No bhavcopy data downloaded.')

full = pd.concat(frames, ignore_index=True)
full = full[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']].copy()
full = full.sort_values(['Symbol', 'Date']).reset_index(drop=True)
full.to_csv(OUT_CSV, index=False)

print('Saved:', OUT_CSV)
print('Rows:', len(full))
print('Symbols:', full['Symbol'].nunique())
print('Start:', full['Date'].min())
print('End:', full['Date'].max())
print(full.head(10).to_string(index=False))

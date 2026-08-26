import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
import os
import datetime
from io import StringIO, BytesIO
from zoneinfo import ZoneInfo

st.set_page_config(page_title="Situational Awareness Engine", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 98%; }
    [data-testid="stHeader"] { display: none; }
    div[data-testid="stVerticalBlockBorderWrapper"] { background-color: #ffffff; border-radius: 12px; padding: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.03); border: 1px solid #e2e8f0; }
    .card-title { font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 2px; }
    .chart-desc { font-size: 11px; color: #94a3b8; font-style: italic; margin-bottom: 10px; line-height: 1.2; }
    .metric-value { font-size: 26px; font-weight: 800; color: #0f172a; }
    .metric-sub { font-size: 12px; font-weight: 600; color: #64748b; margin-top: 2px; }
    .action-banner { background-color: #0f172a; color: #f8fafc; padding: 10px 16px; border-radius: 8px; font-size: 13px; font-weight: 700; text-align: center; margin-top: 8px; }
    .stButton>button { border-radius: 6px; font-weight: 600; font-size: 12px; padding: 0.3rem 0.5rem; }
    div[data-testid="stDateInput"] input { padding: 0.3rem; font-size: 13px; text-align: center; }
    div[data-testid="stDateInput"] label { display: none; }
    </style>
""", unsafe_allow_html=True)

REPO_OWNER = "augmentalphawealth"
REPO_NAME = "Situational-Awareness"
BRANCH = "main"

HISTORICAL_FILE = "historical_breadth_regime_6yr.csv"
LIVE_AGGREGATE_FILE = "live_intraday_aggregate.csv"
INTRADAY_FILE = "live_intraday_breadth.csv"
SYNC_FILE = "last_sync.txt"

LIVE_FRESHNESS_MINUTES = 30

def safe_int(val):
    try:
        if pd.isna(val) or val is None: return 0
        return int(float(val))
    except:
        return 0

def get_latest_commit_sha():
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/commits/{BRANCH}?t={int(time.time())}"
    headers = {"Accept": "application/vnd.github.v3+json", "Cache-Control": "no-cache, no-store, must-revalidate"}
    token = st.secrets.get("GITHUB_TOKEN", None)
    if token: headers["Authorization"] = f"token {token}"
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200: return res.json().get("sha", BRANCH)
    except Exception: pass
    return BRANCH

def read_remote_file(path):
    sha = get_latest_commit_sha()
    url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{sha}/{path}"
    headers = {"Cache-Control": "no-cache, no-store, must-revalidate"}
    token = st.secrets.get("GITHUB_TOKEN", None)
    if token: headers["Authorization"] = f"token {token}"
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200: return res.text
    except Exception: pass
    return None

def get_last_updated_time():
    text = read_remote_file(SYNC_FILE)
    if text: return text.strip()
    return "Unknown"

last_sync_time = get_last_updated_time()

if 'sync_in_progress' not in st.session_state:
    st.session_state.sync_in_progress = False
    st.session_state.sync_start_time = 0
    st.session_state.pre_sync_time = ""

@st.cache_data(ttl=60, show_spinner=False)
def load_intraday_time():
    try:
        csv_text = read_remote_file(INTRADAY_FILE)
        if csv_text: return pd.read_csv(StringIO(csv_text))
    except Exception: pass
    return pd.DataFrame()

def load_remote_csv(path):
    try:
        csv_text = read_remote_file(path)
        if not csv_text:
            return pd.DataFrame()

        df = pd.read_csv(StringIO(csv_text))

        if df.empty or "Date" not in df.columns:
            return pd.DataFrame()

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()

        return (
            df.dropna(subset=["Date"])
            .sort_values("Date")
            .drop_duplicates("Date", keep="last")
            .reset_index(drop=True)
        )
    except Exception:
        return pd.DataFrame()


def parse_intraday_sync_time(sync_text, ist):
    try:
        clean_text = str(sync_text).replace("Today,", "").replace("IST", "").strip()
        parsed_time = datetime.datetime.strptime(clean_text, "%I:%M %p")

        now_ist = datetime.datetime.now(ist)

        return now_ist.replace(
            hour=parsed_time.hour,
            minute=parsed_time.minute,
            second=0,
            microsecond=0,
        )
    except Exception:
        return None


def is_full_live_intraday(live_aggregate, live_breadth, sync_text, ist):
    """
    Return True only when the complete dashboard aggregate is live and fresh.
    This avoids showing a live badge based on A/D data alone.
    """
    if live_aggregate.empty or live_breadth.empty:
        return False

    now_ist = datetime.datetime.now(ist)
    today = pd.Timestamp(now_ist.date()).normalize()

    live_aggregate_today = live_aggregate[live_aggregate["Date"] == today]
    live_breadth_today = live_breadth[live_breadth["Date"] == today]

    if live_aggregate_today.empty or live_breadth_today.empty:
        return False

    if "Time" not in live_breadth_today.columns:
        return False

    intraday_time = str(live_breadth_today.iloc[-1]["Time"]).strip()

    try:
        time_obj = datetime.datetime.strptime(intraday_time, "%H:%M")
        intraday_updated_at = now_ist.replace(
            hour=time_obj.hour,
            minute=time_obj.minute,
            second=0,
            microsecond=0,
        )
    except Exception:
        return False

    sync_updated_at = parse_intraday_sync_time(sync_text, ist)

    if sync_updated_at is None:
        return False

    intraday_age = (now_ist - intraday_updated_at).total_seconds() / 60
    sync_age = (now_ist - sync_updated_at).total_seconds() / 60

    if intraday_age < 0 or sync_age < 0:
        return False

    return (
        intraday_age <= LIVE_FRESHNESS_MINUTES
        and sync_age <= LIVE_FRESHNESS_MINUTES
    )


df_live_breadth = load_intraday_time()
is_live_active = False
last_sync_display = last_sync_time
ist = ZoneInfo("Asia/Kolkata")

if not df_live_breadth.empty and 'Date' in df_live_breadth.columns:
    try:
        today_str_local = datetime.datetime.now(ist).strftime('%Y-%m-%d')
        live_latest = df_live_breadth.iloc[-1]
        if str(live_latest['Date']) == today_str_local:
            is_live_active = True
            intra_time = str(live_latest.get('Time', ''))
            if intra_time:
                t_obj = datetime.datetime.strptime(intra_time, "%H:%M")
                last_sync_display = f"Today, {t_obj.strftime('%I:%M %p')} IST"
    except Exception: pass

def trigger_github_action(workflow_name, button_label):
    token = st.secrets.get("GITHUB_TOKEN", None)
    if not token:
        st.error("GitHub Token missing in Streamlit Secrets!")
        return False
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/workflows/{workflow_name}/dispatches"
    headers = {"Accept": "application/vnd.github.v3+json", "Authorization": f"token {token}"}
    with st.status(f"🚀 Triggering {button_label}...", expanded=False) as status:
        res = requests.post(url, headers=headers, json={"ref": BRANCH})
        if res.status_code == 204:
            status.update(label="✅ Workflow dispatched successfully to GitHub.", state="complete")
            return True
        else:
            status.update(label="❌ Failed to trigger.", state="error")
            st.error(f"GitHub API Error [{res.status_code}]: {res.text}")
            return False

@st.cache_data(ttl=300, show_spinner=False)
def load_agg_data():
    if os.path.exists(HISTORICAL_FILE):
        try:
            df = pd.read_csv(HISTORICAL_FILE)
            if not df.empty and 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                return df.sort_values('Date').reset_index(drop=True)
        except Exception: pass
        
    csv_text = read_remote_file(HISTORICAL_FILE)
    if csv_text:
        df = pd.read_csv(StringIO(csv_text))
        if not df.empty and 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            return df.sort_values('Date').reset_index(drop=True)
    return pd.DataFrame()

@st.cache_data(max_entries=1, show_spinner=False)
def load_trailing_cache():
    file_name = "trailing_cache.parquet"
    if os.path.exists(file_name):
        try:
            df = pd.read_parquet(file_name)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        except Exception: pass
    
    url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{get_latest_commit_sha()}/{file_name}"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            df = pd.read_parquet(BytesIO(res.content))
            df['Date'] = pd.to_datetime(df['Date'])
            return df
    except Exception: pass
    return pd.DataFrame()

df_agg = load_agg_data()
if df_agg.empty:
    st.error(f"Data file '{HISTORICAL_FILE}' not found. Please run the Historical or EOD script to generate it.")
    st.stop()

unique_dates = df_agg['Date'].sort_values().unique()
min_date = pd.to_datetime(unique_dates[0])
max_date = pd.to_datetime(unique_dates[-1])

if 'last_max_date' not in st.session_state or st.session_state.last_max_date != max_date:
    st.session_state.analysis_date = max_date
    st.session_state.last_max_date = max_date
if 'analysis_date' not in st.session_state:
    st.session_state.analysis_date = max_date

def step_prev_day():
    curr_idx = np.where(unique_dates == np.datetime64(st.session_state.analysis_date))[0]
    if len(curr_idx) > 0 and curr_idx[0] > 0:
        st.session_state.analysis_date = pd.to_datetime(unique_dates[curr_idx[0] - 1])

def step_next_day():
    curr_idx = np.where(unique_dates == np.datetime64(st.session_state.analysis_date))[0]
    if len(curr_idx) > 0 and curr_idx[0] < len(unique_dates) - 1:
        st.session_state.analysis_date = pd.to_datetime(unique_dates[curr_idx[0] + 1])

head_col1, head_spacer, head_col2, head_col3 = st.columns([3.0, 0.5, 2.0, 1.2])
with head_col1:
    st.markdown("<h2 style='margin-top: 10px; margin-bottom: 0px; font-weight: 800; color: #0f172a; white-space: nowrap; font-size: 24px;'>🛡️ SITUATIONAL AWARENESS</h2>", unsafe_allow_html=True)
with head_spacer: st.write("")
with head_col2:
    st.write("")
    nav1, nav2, nav3 = st.columns([1, 1.4, 1])
    with nav1: st.button("◀ Prev", on_click=step_prev_day, use_container_width=True)
    with nav2:
        selected_date = st.date_input("Date", value=st.session_state.analysis_date, min_value=min_date, max_value=max_date, format="DD/MM/YYYY", label_visibility="collapsed")
        if selected_date != st.session_state.analysis_date.date():
            st.session_state.analysis_date = pd.to_datetime(selected_date)
            st.rerun()
    with nav3: st.button("Next ▶", on_click=step_next_day, use_container_width=True)
with head_col3:
    display_sync = last_sync_display if st.session_state.analysis_date == max_date else "Historical View"
    st.markdown(f"<div style='text-align: right; font-size: 11px; font-weight: 600; color: #64748b; margin-top: 2px; margin-bottom: 3px;'>Last Sync: {display_sync}</div>", unsafe_allow_html=True)
    
    if st.session_state.sync_in_progress:
        elapsed_time = time.time() - st.session_state.sync_start_time
        if elapsed_time < 420: 
            current_sync_time = get_last_updated_time()
            if current_sync_time != st.session_state.pre_sync_time and current_sync_time != "Unknown":
                st.session_state.sync_in_progress = False
                load_agg_data.clear()
                load_intraday_time.clear()
                if 'get_drilldown_data' in st.cache_data: get_drilldown_data.clear()
                st.success("✅ Sync Complete! Reloading...")
                time.sleep(1)
                st.rerun()
            else:
                st.info(f"⏳ Bot is crunching data... Auto-refreshing. (Elapsed: {int(elapsed_time)}s)")
                time.sleep(4)
                st.rerun()
        else:
            st.session_state.sync_in_progress = False
            st.error("Sync timed out. Please try again.")
    else:
        if st.button("⚡ Trigger Intraday Sync", use_container_width=True):
            if trigger_github_action("intraday_update.yml", "Intraday Sync"):
                st.session_state.sync_in_progress = True
                st.session_state.sync_start_time = time.time()
                st.session_state.pre_sync_time = last_sync_display
                st.rerun()

df_filtered = df_agg[df_agg['Date'] <= st.session_state.analysis_date].copy()
if df_filtered.empty:
    st.warning("No trading data available on or before this date.")
    st.stop()

latest = df_filtered.iloc[-1]
prev = df_filtered.iloc[-2] if len(df_filtered) > 1 else latest

score = safe_int(latest.get('Composite_Score', 0))
p_fast = latest.get('Pct_Above_20_EMA', 0)
p_mid = latest.get('Pct_Above_50_EMA', 0)
ft_rate = (latest.get('T3_Wins', 0) / latest.get('T3_Breakouts', 1) * 100) if latest.get('T3_Breakouts', 0) > 0 else 0

if score >= 71:
    action_zone, score_color = "AGGRESSIVE MTF ZONE", "#22c55e"
    tacs = {"asset": "High-RS VCP Stocks", "sizing": "100% Full Sizing", "risk": "Standard 5-7% Hard Stops", "profit": "Aggressively trade valid VCP breakouts"}
elif score >= 51:
    action_zone, score_color = "CONFIRMED UPTREND", "#84cc16"
    tacs = {"asset": "50% Broad ETFs & 50% High-RS VCP Stocks", "sizing": "50% - 75% Sizing", "risk": "Balanced Risk", "profit": "Selective buying on A+ setups only"}
elif score >= 31:
    action_zone, score_color = "DEATH CHOP ZONE", "#f97316"
    tacs = {"asset": "Broad ETFs ONLY", "sizing": "50% Max Sizing", "risk": "0% VCP Stocks", "profit": "Use ETFs to capture broad swings safely"}
elif score >= 11:
    action_zone, score_color = "RISK-OFF / DEFENSIVE", "#ef4444"
    tacs = {"asset": "Broad ETFs ONLY", "sizing": "25% Max Sizing", "risk": "Strict 5-7% Base Stops", "profit": "Defensive allocation, 0% VCP Stocks"}
else:
    action_zone, score_color = "CAPITULATION WATCH", "#991b1b"
    tacs = {"asset": "CASH", "sizing": "0% (Pure Cash)", "risk": "Sit on hands", "profit": "Build watchlists; wait for a bounce"}

df_10d = df_filtered.tail(10).copy()
df_10d['Date'] = pd.to_datetime(df_10d['Date'], errors='coerce')
df_10d = df_10d.dropna(subset=['Date'])
df_10d = df_10d.sort_values('Date').reset_index(drop=True)
df_10d['Date_Str'] = df_10d['Date'].dt.strftime('%d %b')

df_10d['Composite_Score'] = pd.to_numeric(df_10d['Composite_Score'], errors='coerce').fillna(0)

def get_bar_color(val):
    if val >= 71: return "#22c55e"
    elif val >= 51: return "#84cc16"
    elif val >= 31: return "#f97316"
    elif val >= 11: return "#ef4444"
    else: return "#991b1b"

with st.container(border=True):
    top_c1, top_c2 = st.columns([1.2, 2.8])
    with top_c1:
        st.markdown(f"""
            <div style='text-align: center; padding-top: 10px;'>
                <div class='card-title' style='font-size: 12px;'>MOMENTUM HEALTH SCORE (EOD)</div>
                <div class='metric-value' style='font-size: 44px; color: {score_color};'>{score} <span style='font-size: 18px; color: #94a3b8;'>/ 100</span></div>
                <div style='font-size: 11px; color: #64748b; font-weight: 600; margin-top: 2px;'>Fast Breadth (P_Fast): {p_fast:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
    with top_c2:
        st.markdown("<div class='card-title'>10-Day Health Trend</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-desc'>Tracks the EOD regime score over time. Crucial for spotting momentum shifts and trend decays.</div>", unsafe_allow_html=True)
        
        fig_m1 = go.Figure(go.Bar(
            x=df_10d['Date_Str'], y=df_10d['Composite_Score'], 
            marker_color=[get_bar_color(v) for v in df_10d['Composite_Score']],
            text=df_10d['Composite_Score'], textposition='outside', textangle=0,            
            hovertemplate='Score: <b>%{y}</b><extra></extra>'
        ))
        fig_m1.update_layout(height=120, margin=dict(l=10, r=10, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        fig_m1.update_xaxes(showgrid=False, tickfont=dict(size=10, color="#94a3b8"))
        fig_m1.update_yaxes(visible=False, range=[0, 115], fixedrange=True) 
        st.plotly_chart(fig_m1, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown(f"<div class='action-banner'>🎯 ACTION ZONE: {action_zone}</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# Flatline Guard & Live Override Logic
# -------------------------------------------------------------
advances = safe_int(latest.get('Advances', 0))
declines = safe_int(latest.get('Declines', 0))
total_univ = safe_int(latest.get('Total_Universe', 2400))
actual_date_str = f"{st.session_state.analysis_date.day} {st.session_state.analysis_date.strftime('%B %Y')}"

if is_live_active and st.session_state.analysis_date == max_date:
    live_adv = safe_int(live_latest.get('Advances', advances))
    live_dec = safe_int(live_latest.get('Declines', declines))
    if (live_adv + live_dec) > 0:
        advances = live_adv
        declines = live_dec
        total_univ = safe_int(live_latest.get('Total_Universe', total_univ))
        live_date_obj = pd.to_datetime(live_latest['Date'])
        actual_date_str = f"{live_date_obj.day} {live_date_obj.strftime('%B %Y')} <span style='color:#eab308; font-weight:800;'>(⚡ LIVE INTRADAY A/D SNAPSHOT)</span>"
# -------------------------------------------------------------

st.markdown(f"<p style='color: #475569; font-size: 13px; font-weight: 600; margin-top: 15px;'>Market Breadth Status for: <span style='color:#0f172a;'>{actual_date_str}</span></p>", unsafe_allow_html=True)

hero_col1, hero_col2 = st.columns([1.5, 2.5])

total_adv_dec = advances + declines
adv_pct = round((advances / total_adv_dec) * 100, 1) if total_adv_dec > 0 else 0

prev_advances = safe_int(prev.get('Advances', 0))
prev_declines = safe_int(prev.get('Declines', 0))
prev_total_adv_dec = prev_advances + prev_declines
prev_adv_pct = round((prev_advances / prev_total_adv_dec) * 100, 1) if prev_total_adv_dec > 0 else 0
adv_change = round(adv_pct - prev_adv_pct, 1)
adv_change_str = f"+{adv_change}%" if adv_change >= 0 else f"{adv_change}%"
adv_change_color = "#16a34a" if adv_change >= 0 else "#dc2626"

with hero_col1:
    with st.container(border=True):
        st.markdown("<div class='card-title' style='text-align: center; margin-top: 5px;'>OF UNIVERSE ADVANCING</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-desc' style='text-align: center;'>Real-time buying vs. selling participation. Core indicator for confirming broad market support.</div>", unsafe_allow_html=True)
        
        fig_dial = go.Figure(go.Indicator(
            mode = "gauge+number", value = adv_pct,
            number = {'suffix': "%", 'font': {'size': 36, 'color': '#0f172a'}},
            gauge = {'axis': {'range': [0, 100], 'visible': False}, 'bar': {'color': "#ef4444" if adv_pct < 50 else "#22c55e", 'thickness': 0.18}, 'bgcolor': "#f1f5f9", 'borderwidth': 0}
        ))
        fig_dial.update_layout(height=150, margin=dict(l=10, r=10, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_dial, use_container_width=True, config={'displayModeBar': False})
        st.markdown(f"""
            <div style='text-align: center; margin-top: -10px; padding-bottom: 8px;'>
                <span style='color: #16a34a; font-size: 15px; font-weight: 800;'>{advances} ADVANCES</span> &nbsp;&

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
import os
import datetime
import json
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
    .status-badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 800; }
    .status-live { background-color: #dcfce7; color: #166534; }
    .status-stale { background-color: #fee2e2; color: #991b1b; }
    .status-na { background-color: #f1f5f9; color: #475569; }
    .status-banner { padding: 8px 12px; border-radius: 8px; font-size: 12px; font-weight: 700; margin-top: 6px; }
    .status-banner-live { background-color: #dcfce7; color: #166534; }
    .status-banner-stale { background-color: #ffedd5; color: #9a3412; }
    .status-banner-na { background-color: #fee2e2; color: #991b1b; }
    </style>
""", unsafe_allow_html=True)

REPO_OWNER = "augmentalphawealth"
REPO_NAME = "Situational-Awareness"
BRANCH = "main"

HISTORICAL_FILE = "historical_breadth_regime_6yr.csv"
INTRADAY_FILE = "live_intraday_breadth.csv"
SYNC_FILE = "last_sync.txt"

# Intraday temporary files
INTRADAY_STATUS_PATH = "intraday_tmp/intraday_status.json"
INTRADAY_METRICS_PATH = "intraday_tmp/intraday_market_metrics.json"

STALE_THRESHOLD_MINUTES = 15

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

df_live = load_intraday_time()
is_live_active = False
last_sync_display = last_sync_time
ist = ZoneInfo("Asia/Kolkata")

if not df_live.empty and 'Date' in df_live.columns:
    try:
        today_str_local = datetime.datetime.now(ist).strftime('%Y-%m-%d')
        live_latest = df_live.iloc[-1]
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

# ---------------------------
# Intraday status & staleness
# ---------------------------

def parse_intraday_status():
    """
    Returns:
        intraday_ok: bool
        last_update_dt: datetime or None
        staleness_minutes: float or None
        status_text: str
        status_class: 'live' | 'stale' | 'na'
        banner_text: str
        banner_class: 'live' | 'stale' | 'na'
    """
    status_text = read_remote_file(INTRADAY_STATUS_PATH)
    if not status_text:
        return False, None, None, "Intraday data unavailable", "na", "INTRADAY DATA UNAVAILABLE", "na"
    try:
        status_obj = json.loads(status_text)
    except Exception:
        return False, None, None, "Intraday status parse error", "na", "INTRADAY DATA UNAVAILABLE", "na"
    
    updated_at_str = status_obj.get("updated_at")
    if not updated_at_str:
        return False, None, None, "Intraday data unavailable", "na", "INTRADAY DATA UNAVAILABLE", "na"
    
    try:
        # Format: 2026-08-25T16:47:59.492912+05:30
        last_update_dt = datetime.datetime.fromisoformat(updated_at_str)
    except Exception:
        return False, None, None, "Intraday data unavailable", "na", "INTRADAY DATA UNAVAILABLE", "na"
    
    now_ist = datetime.datetime.now(ist)
    # Ensure both are comparable; last_update_dt is already offset-aware
    age_td = now_ist - last_update_dt
    age_minutes = age_td.total_seconds() / 60.0
    
    if age_minutes <= STALE_THRESHOLD_MINUTES:
        status_class = "live"
        banner_class = "live"
        status_text_display = f"LIVE — updated {last_update_dt.strftime('%H:%M')} IST"
        banner_text = f"INTRADAY DATA LIVE — last successful update {last_update_dt.strftime('%H:%M')} IST"
    else:
        status_class = "stale"
        banner_class = "stale"
        status_text_display = f"STALE — last update {last_update_dt.strftime('%H:%M')} IST"
        banner_text = f"INTRADAY DATA STALE — last successful update {last_update_dt.strftime('%H:%M')} IST (>{STALE_THRESHOLD_MINUTES} min old)"
    
    return True, last_update_dt, age_minutes, status_text_display, status_class, banner_text, banner_class

intraday_ok, intraday_update_dt, intraday_age_min, intraday_status_text, intraday_status_class, intraday_banner_text, intraday_banner_class = parse_intraday_status()

# ---------------------------
# Header & controls
# ---------------------------

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
    # Right-aligned status line
    if st.session_state.analysis_date == max_date and intraday_ok:
        display_sync = intraday_status_text
    else:
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

# Status banner for intraday data (only on latest date)
if st.session_state.analysis_date == max_date:
    if intraday_ok:
        st.markdown(f"<div class='status-banner status-banner-{intraday_banner_class}'>{intraday_banner_text}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='status-banner status-banner-na'>INTRADAY DATA UNAVAILABLE — using latest EOD snapshot only</div>", unsafe_allow_html=True)

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

# ---------------------------
# 10-day health trend (fixed date handling & score bar)
# ---------------------------

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
                <span style='color: #16a34a; font-size: 15px; font-weight: 800;'>{advances} ADVANCES</span> &nbsp;&nbsp;
                <span style='color: #dc2626; font-size: 15px; font-weight: 800;'>{declines} DECLINES</span>
                <p style='color: {adv_change_color}; font-size: 12px; font-weight: 700; margin-top: 4px; margin-bottom: 0px;'>{adv_change_str} vs Yesterday EOD (Active Univ {total_univ})</p>
            </div>
        """, unsafe_allow_html=True)

with hero_col2:
    with st.container(border=True):
        st.markdown("<div class='card-title' style='margin-top: 5px; margin-left: 10px;'>TACTICAL COMMAND CENTER</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-desc' style='margin-left: 10px;'>Your immediate action plan dynamically dictated by the Momentum Health Score.</div>", unsafe_allow_html=True)
        st.markdown(f"""
            <div style='padding: 0px 10px; font-size: 13px;'>
                <div style='margin-bottom: 6px;'><b>🎯 Target Asset:</b> <span style='color: #334155;'>{tacs['asset']}</span></div>
                <div style='margin-bottom: 6px;'><b>⚖️ Position Sizing:</b> <span style='color: #334155;'>{tacs['sizing']}</span></div>
                <div style='margin-bottom: 6px;'><b>🛡️ Risk / Stop-Loss:</b> <span style='color: #334155;'>{tacs['risk']}</span></div>
                <div style='margin-bottom: 6px;'><b>💰 Profit Strategy:</b> <span style='color: #334155;'>{tacs['profit']}</span></div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("<hr style='margin: 8px 0px;'>", unsafe_allow_html=True)
        st.markdown(f"<div class='card-title' style='margin-left: 10px;'>DECISION ENGINE: KEY REGIME DRIVERS</div>", unsafe_allow_html=True)
        
        # Clean Core Metrics (No MCO/TRIN Clutter)
        wr_c = "#16a34a" if ft_rate >= 45 else "#dc2626"
        p20_c = "#16a34a" if p_fast >= 50 else "#dc2626"
        p50_c = "#16a34a" if p_mid >= 50 else "#dc2626"
        
        regime_html = f"""
        <div style='font-size: 12px; margin-bottom: 8px; padding-left: 10px;'>
            <div style='margin-bottom: 4px;'>• Follow-Through Rate: <b style='color:{wr_c};'>{ft_rate:.1f}%</b></div>
            <div style='margin-bottom: 4px;'>• Fast Breadth (>20 EMA): <b style='color:{p20_c};'>{p_fast:.1f}%</b></div>
            <div style='margin-bottom: 4px;'>• Trend Breadth (>50 EMA): <b style='color:{p50_c};'>{p_mid:.1f}%</b></div>
        </div>
        """
        
        # Dynamic Actionable Extremes (Sentiment Linked & Strictly Logical)
        extremes = []
        
        latest_trin = latest.get('TRIN', np.nan)
        if not pd.isna(latest_trin):
            if latest_trin >= 1.5 and score <= 31: 
                extremes.append(f"<div style='margin-bottom: 3px;'>🟢 <b>TRIN: {latest_trin:.2f}</b> <span style='color:#16a34a;'>(Panic Capitulation. Weak regime + heavy selling = Contrarian Bottom Signal.)</span></div>")
            elif latest_trin <= 0.7 and score >= 51: 
                extremes.append(f"<div style='margin-bottom: 3px;'>🟢 <b>TRIN: {latest_trin:.2f}</b> <span style='color:#16a34a;'>(Aggressive Demand. Strong regime + heavy buying = Trend Confirmed.)</span></div>")

        latest_mco = latest.get('MCO', np.nan)
        if not pd.isna(latest_mco):
            if latest_mco >= 50:
                if score <= 31:
                    extremes.append(f"<div style='margin-bottom: 3px;'>🟢 <b>MCO: +{latest_mco:.1f}</b> <span style='color:#16a34a;'>(Early Ignition Thrust. Surging momentum from the bottom. Watch for base breakouts.)</span></div>")
                else:
                    extremes.append(f"<div style='margin-bottom: 3px;'>🟢 <b>MCO: +{latest_mco:.1f}</b> <span style='color:#16a34a;'>(Power Trend. Short-term breadth is hot and institutional buying is aggressive.)</span></div>")
            elif latest_mco <= -50 and score <= 31:
                extremes.append(f"<div style='margin-bottom: 3px;'>🟢 <b>MCO: {latest_mco:.1f}</b> <span style='color:#16a34a;'>(Deep Washout. Extreme short-term oversold. Look for the turn.)</span></div>")
                
        vol_ratio = latest.get('Volume_Ratio', np.nan)
        if not pd.isna(vol_ratio):
            if vol_ratio >= 2.0: extremes.append(f"<div style='margin-bottom: 3px;'>🟢 <b>Vol Ratio: {vol_ratio:.2f}</b> <span style='color:#16a34a;'>(Heavy Accumulation)</span></div>")
            elif vol_ratio <= 0.5: extremes.append(f"<div style='margin-bottom: 3px;'>🚨 <b>Vol Ratio: {vol_ratio:.2f}</b> <span style='color:#dc2626;'>(Heavy Distribution)</span></div>")
            
        net_hl = safe_int(latest.get('Net_52W_High_Low', 0))
        if net_hl >= 50: extremes.append(f"<div style='margin-bottom: 3px;'>🟢 <b>Net 52W H/L: +{net_hl}</b> <span style='color:#16a34a;'>(Broad Expansion)</span></div>")
        elif net_hl <= -50: extremes.append(f"<div style='margin-bottom: 3px;'>🚨 <b>Net 52W H/L: {net_hl}</b> <span style='color:#dc2626;'>(Broad Capitulation)</span></div>")

        ipo_highs = safe_int(latest.get('IPO_New_Highs', 0))
        if ipo_highs >= 15:
            extremes.append(f"<div style='margin-bottom: 3px;'>⚠️ <b>IPO Highs: {ipo_highs}</b> <span style='color:#f97316;'>(Speculative Frenzy. Market is hunting for high-beta risk.)</span></div>")

        if extremes:
            regime_html += "<div style='border-top: 1px dashed #cbd5e1; margin-top: 8px; padding-top: 8px; padding-left: 10px;'>"
            regime_html += "<div style='font-size: 11px; font-weight: 800; color: #f97316; margin-bottom: 4px; text-transform: uppercase;'>⚠️ Actionable Extremes Triggered:</div>"
            regime_html += "".join(extremes)
            regime_html += "</div>"
            
        st.markdown(regime_html, unsafe_allow_html=True)

m2, m3 = st.columns(2)
with m2:
    vol_ratio = latest.get('Volume_Ratio', np.nan)
    if pd.isna(vol_ratio):
        v_col, v_str = "#94a3b8", "N/A"
    else:
        v_col, v_str = ("#22c55e" if vol_ratio > 1.0 else "#ef4444"), f"{vol_ratio:.2f}"
        
    with st.container(border=True):
        st.markdown("<div class='card-title' style='margin-left: 10px; margin-top: 5px;'>Volume Breadth Ratio</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-desc' style='margin-left: 10px;'>Ratio of volume in advancing vs. declining stocks. >1.0 indicates institutional accumulation.</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center;'><div class='metric-value' style='color: {v_col};'>{v_str}</div></div>", unsafe_allow_html=True)
        fig_m2 = go.Figure(go.Bar(x=df_10d['Date_Str'], y=df_10d.get('Volume_Ratio', pd.Series(dtype=float)).fillna(0), marker_color=['#22c55e' if v >= 1.0 else '#ef4444' for v in df_10d.get('Volume_Ratio', pd.Series([0]))], hovertemplate='Ratio: %{y:.2f}<extra></extra>'))
        fig_m2.add_hline(y=1.0, line_dash="dash", line_color="#cbd5e1")
        fig_m2.update_layout(height=100, margin=dict(l=5, r=5, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        fig_m2.update_xaxes(showgrid=False, tickfont=dict(size=10, color="#94a3b8"))
        fig_m2.update_yaxes(visible=False)
        st.plotly_chart(fig_m2, use_container_width=True, config={'displayModeBar': False})

with m3:
    net_hl = safe_int(latest.get('Net_52W_High_Low', 0))
    ipo_hl = safe_int(latest.get('IPO_New_Highs', 0))
    h_col = "#22c55e" if net_hl > 0 else "#ef4444"
    with st.container(border=True):
        st.markdown("<div class='card-title' style='margin-left: 10px; margin-top: 5px;'>Net 52-Week Highs (Mature)</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chart-desc' style='margin-left: 10px;'>Long-term strength. (Plus <b style='color:#3b82f6;'>{ipo_hl}</b> new IPOs hitting all-time highs).</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center;'><div class='metric-value' style='color: {h_col};'>{net_hl}</div></div>", unsafe_allow_html=True)
        fig_m3 = go.Figure(go.Bar(x=df_10d['Date_Str'], y=df_10d['Net_52W_High_Low'], marker_color=['#22c55e' if v >= 0 else '#ef4444' for v in df_10d['Net_52W_High_Low']], hovertemplate='Net HL: %{y:.0f}<extra></extra>'))
        fig_m3.add_hline(y=0, line_dash="solid", line_color="#cbd5e1")
        fig_m3.update_layout(height=100, margin=dict(l=5, r=5, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        fig_m3.update_xaxes(showgrid=False, tickfont=dict(size=10, color="#94a3b8"))
        fig_m3.update_yaxes(visible=False)
        st.plotly_chart(fig_m3, use_container_width=True, config={'displayModeBar': False})

# --- HISTORICAL ANALYTICS ---
st.markdown("<br>", unsafe_allow_html=True)
tf_col1, tf_col2 = st.columns([3, 1])
with tf_col1: st.markdown("### 📊 Historical Market Analytics")
with tf_col2: timeframe = st.radio("Chart Horizon:", ["1 Month", "3 Months", "6 Months", "1 Year", "3 Years", "6 Years"], horizontal=True, index=2)

days_map = {"1 Month": 21, "3 Months": 63, "6 Months": 126, "1 Year": 252, "3 Years": 756, "6 Years": len(df_filtered)}
plot_df = df_filtered.tail(days_map.get(timeframe, 126)).copy()

# Ensure Date is proper datetime and sorted
plot_df['Date'] = pd.to_datetime(plot_df['Date'], errors='coerce')
plot_df = plot_df.dropna(subset=['Date'])
plot_df = plot_df.sort_values('Date').reset_index(drop=True)

plot_df['Chart_Date_Str'] = plot_df['Date'].apply(lambda x: f"{x.day} {x.strftime('%B')[:3]}")

# SECTION 1: UNIVERSE EMA BREADTH
with st.container(border=True):
    st.markdown(f"<div class='card-title' style='margin-left: 10px; margin-top: 10px;'>UNIVERSE EMA BREADTH TRENDS &nbsp;|&nbsp; LATEST: <span style='color:#22c55e;'>200 EMA ({plot_df['Pct_Above_200_EMA'].iloc[-1]:.1f}%)</span> • <span style='color:#a855f7;'>50 EMA ({plot_df['Pct_Above_50_EMA'].iloc[-1]:.1f}%)</span> • <span style='color:#3b82f6;'>20 EMA ({plot_df['Pct_Above_20_EMA'].iloc[-1]:.1f}%)</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='chart-desc' style='margin-left: 10px;'>Percentage of mature active stocks above key moving averages. Watch for >50% cross as trend confirmation.</div>", unsafe_allow_html=True)
    
    fig_ema = go.Figure()
    fig_ema.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Pct_Above_200_EMA'], mode='lines', name='% > 200 EMA', line=dict(color='#22c55e', width=2)))
    fig_ema.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Pct_Above_50_EMA'], mode='lines', name='% > 50 EMA', line=dict(color='#a855f7', width=2)))
    fig_ema.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Pct_Above_20_EMA'], mode='lines', name='% > 20 EMA', line=dict(color='#3b82f6', width=2)))
    fig_ema.add_hline(y=50, line_dash="dash", line_color="#94a3b8", opacity=0.7)
    fig_ema.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    fig_ema.update_yaxes(range=[0, 100], gridcolor='#f1f5f9', title="% Stocks")
    fig_ema.update_xaxes(showgrid=False)
    st.plotly_chart(fig_ema, use_container_width=True)

# SECTION 2: SEGMENTED LIQUIDITY FLOW
with st.container(border=True):
    st.markdown("<div class='card-title' style='margin-left: 10px; margin-top: 10px;'>SEGMENTED LIQUIDITY FLOW (45-DAY ROLLING TURNOVER RANK)</div>", unsafe_allow_html=True)
    st.markdown("<div class='chart-desc' style='margin-left: 10px;'>Breadth broken down by liquidity. Divergence (e.g., Large caps up, Micro caps down) signals narrow, fragile rallies.</div>", unsafe_allow_html=True)
    cap_tab1, cap_tab2, cap_tab3 = st.tabs(["% Stocks Above 200 EMA", "% Stocks Above 50 EMA", "% Stocks Above 20 EMA"])
    
    def plot_liquidity(col_prefix, name):
        st.markdown(f"<div style='font-size: 11px; font-weight: 700; color: #64748b; margin-top: -10px; margin-bottom: 5px; padding-left: 10px;'>LATEST: <span style='color:#2563eb;'>Top 100 ({plot_df.get(f'Large_{col_prefix}', pd.Series([0])).iloc[-1]:.1f}%)</span> &nbsp;|&nbsp; <span style='color:#f97316;'>Mid 150 ({plot_df.get(f'Mid_{col_prefix}', pd.Series([0])).iloc[-1]:.1f}%)</span> &nbsp;|&nbsp; <span style='color:#16a34a;'>Lower 250 ({plot_df.get(f'Small_{col_prefix}', pd.Series([0])).iloc[-1]:.1f}%)</span> &nbsp;|&nbsp; <span style='color:#dc2626;'>Micro ({plot_df.get(f'Micro_{col_prefix}', pd.Series([0])).iloc[-1]:.1f}%)</span></div>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df.get(f'Large_{col_prefix}', pd.Series(dtype=float)), mode='lines', name='Top 100 Liq', line=dict(color='#2563eb', width=2)))
        fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df.get(f'Mid_{col_prefix}', pd.Series(dtype=float)), mode='lines', name='Mid 150 Liq', line=dict(color='#f97316', width=2)))
        fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df.get(f'Small_{col_prefix}', pd.Series(dtype=float)), mode='lines', name='Lower 250 Liq', line=dict(color='#16a34a', width=2)))
        fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df.get(f'Micro_{col_prefix}', pd.Series(dtype=float)), mode='lines', name='Micro Liq', line=dict(color='#dc2626', width=2)))
        fig.add_hline(y=50, line_dash="dash", line_color="#94a3b8", opacity=0.7)
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_yaxes(range=[0, 100], gridcolor='#f1f5f9')
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

    with cap_tab1: plot_liquidity('Pct_200_EMA', '200')
    with cap_tab2: plot_liquidity('Pct_50_EMA', '50')
    with cap_tab3: plot_liquidity('Pct_20_EMA', '20')

# SECTION 3: TACTICAL INTERNAL EXERTION (MCO & TRIN CHARTS)
st.markdown("<br>", unsafe_allow_html=True)
tac_col1, tac_col2 = st.columns(2)

with tac_col1:
    with st.container(border=True):
        latest_mco_chart = plot_df['MCO'].iloc[-1] if 'MCO' in plot_df.columns and not plot_df['MCO'].isna().all() else 0
        mco_color = '#16a34a' if latest_mco_chart >= 0 else '#dc2626'
        st.markdown(f"<div class='card-title' style='margin-left: 10px; margin-top: 10px;'>McCLELLAN OSCILLATOR (MCO) &nbsp;|&nbsp; LATEST: <span style='color:{mco_color};'>{latest_mco_chart:.1f}</span></div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-desc' style='margin-left: 10px;'>Short-term momentum of advancing issues. Readings >+50 or <-50 signal extreme conditions.</div>", unsafe_allow_html=True)
        
        colors_mco = ['#22c55e' if val >= 0 else '#ef4444' for val in plot_df.get('MCO', pd.Series([0]))]
        fig_mco = go.Figure(go.Bar(x=plot_df['Date'], y=plot_df.get('MCO', pd.Series(dtype=float)), marker_color=colors_mco, hovertemplate='MCO: %{y:.1f}<extra></extra>'))
        fig_mco.add_hline(y=0, line_dash="solid", line_color="#94a3b8")
        fig_mco.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False, hovermode="x unified")
        fig_mco.update_yaxes(gridcolor='#f1f5f9', zerolinecolor='black')
        fig_mco.update_xaxes(showgrid=False)
        st.plotly_chart(fig_mco, use_container_width=True)

with tac_col2:
    with st.container(border=True):
        latest_trin_chart = plot_df['TRIN'].iloc[-1] if 'TRIN' in plot_df.columns and not plot_df['TRIN'].isna().all() else np.nan
        trin_color = "#94a3b8" if pd.isna(latest_trin_chart) else ('#16a34a' if latest_trin_chart < 1.0 else '#dc2626')
        trin_str = "N/A" if pd.isna(latest_trin_chart) else f"{latest_trin_chart:.2f}"
        
        st.markdown(f"<div class='card-title' style='margin-left: 10px; margin-top: 10px;'>TRIN (ARMS INDEX) &nbsp;|&nbsp; LATEST: <span style='color:{trin_color};'>{trin_str}</span></div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-desc' style='margin-left: 10px;'>Contrarian indicator balancing A/D and Volume. <0.7 shows aggressive buying, >1.5 shows panic selling.</div>", unsafe_allow_html=True)
        
        colors_trin = ['#22c55e' if (pd.notna(val) and val < 1.0) else '#ef4444' for val in plot_df.get('TRIN', pd.Series([1.0]))]
        fig_trin = go.Figure(go.Bar(x=plot_df['Date'], y=plot_df.get('TRIN', pd.Series(dtype=float)), marker_color=colors_trin, hovertemplate='TRIN: %{y:.2f}<extra></extra>'))
        fig_trin.add_hline(y=1.0, line_dash="dash", line_color="#cbd5e1", annotation_text="Neutral (1.0)")
        fig_trin.add_hline(y=0.7, line_dash="dot", line_color="#22c55e", annotation_text="Demand (<0.7)")
        fig_trin.add_hline(y=1.5, line_dash="dot", line_color="#ef4444", annotation_text="Panic (>1.5)")
        
        y_max = min(5.0, float(plot_df['TRIN'].dropna().max()) + 0.5) if not plot_df['TRIN'].dropna().empty else 3.0
        fig_trin.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False, hovermode="x unified", yaxis_range=[0, y_max])
        fig_trin.update_yaxes(gridcolor='#f1f5f9')
        fig_trin.update_xaxes(showgrid=False)
        st.plotly_chart(fig_trin, use_container_width=True)

# SECTION 4: BREAKOUT SUCCESS
with st.container(border=True):
    t3_wins = plot_df.get('T3_Wins', pd.Series([0])).fillna(0)
    t3_breaks = plot_df.get('T3_Breakouts', pd.Series([0])).fillna(0)
    t3_fails = (t3_breaks - t3_wins).clip(lower=0)
    
    win_rate = np.where(t3_breaks > 0, (t3_wins / t3_breaks) * 100, 0)
    plot_df['Win_Rate_Plot'] = win_rate
    plot_df['T3_Wins'] = t3_wins
    plot_df['T3_Fails'] = t3_fails
    plot_df['T3_Total'] = t3_breaks
    
    latest_wr = plot_df['Win_Rate_Plot'].iloc[-1] if not pd.isna(plot_df['Win_Rate_Plot'].iloc[-1]) else 0
    wr_color = '#16a34a' if latest_wr >= 45 else '#dc2626'
    
    st.markdown(f"<div class='card-title' style='margin-left: 10px; margin-top: 10px;'>VCP BREAKOUT FOLLOW-THROUGH (T+3 WIN RATE) &nbsp;|&nbsp; LATEST: <span style='color:{wr_color};'>{latest_wr:.1f}%</span> (3-Day Cohort)</div>", unsafe_allow_html=True)
    st.markdown("<div class='chart-desc' style='margin-left: 10px;'>Tracks the T+3 win rate of breakouts. >=45% begins scoring in the momentum composite.</div>", unsafe_allow_html=True)
    
    colors_wr = ['#22c55e' if val >= 45 else '#ef4444' for val in plot_df['Win_Rate_Plot']]
    
    fig_ft = go.Figure(go.Bar(
        x=plot_df['Date'], 
        y=plot_df['Win_Rate_Plot'], 
        marker_color=colors_wr, 
        customdata=np.stack((plot_df['T3_Total'], plot_df['T3_Wins'], plot_df['T3_Fails']), axis=-1),
        hovertemplate='<b>Win Rate: %{y:.1f}%</b><br>Total Breakouts: %{customdata[0]}<br>Wins: %{customdata[1]}<br>Fails: %{customdata[2]}<extra></extra>'
    ))
    
    fig_ft.add_hline(y=45, line_dash="dash", line_color="#22c55e", annotation_text="45% Edge Baseline", annotation_position="top left")
    
    fig_ft.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False, hovermode="x unified")
    fig_ft.update_yaxes(range=[0, 100], gridcolor='#f1f5f9', title="Win Rate %")
    fig_ft.update_xaxes(showgrid=False)
    st.plotly_chart(fig_ft, use_container_width=True)

# SECTION 5: MOMENTUM THRUST & OUTLIERS
out1, out2 = st.columns(2)
with out1:
    with st.container(border=True):
        latest_up_25 = safe_int(plot_df['Up_25_1M_Count'].iloc[-1]) if not plot_df['Up_25_1M_Count'].empty else 0
        latest_dn_25 = safe_int(plot_df['Down_25_1M_Count'].iloc[-1]) if not plot_df['Down_25_1M_Count'].empty else 0
        
        st.markdown(f"<div class='card-title' style='margin-left: 10px; margin-top: 10px;'>ROLLING 1-MONTH 25% MOVERS &nbsp;|&nbsp; LATEST: <span style='color:#16a34a;'>{latest_up_25} UP</span> / <span style='color:#dc2626;'>{latest_dn_25} DOWN</span></div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-desc' style='margin-left: 10px;'>Identifies structural outliers. More up-movers than down-movers signifies strong momentum thrusts.</div>", unsafe_allow_html=True)
        
        fig_outliers = go.Figure()
        fig_outliers.add_trace(go.Bar(x=plot_df['Date'], y=plot_df['Up_25_1M_Count'], name='Up 25%+ in 1M', marker_color='#22c55e'))
        fig_outliers.add_trace(go.Bar(x=plot_df['Date'], y=-plot_df['Down_25_1M_Count'], name='Down 25%+ in 1M', marker_color='#ef4444', customdata=plot_df['Down_25_1M_Count'], hovertemplate='Down 25%: <b>%{customdata:.0f}</b><extra></extra>'))
        fig_outliers.update_layout(barmode='relative', height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False, hovermode="x unified")
        fig_outliers.update_yaxes(gridcolor='#f1f5f9', zerolinecolor='black')
        fig_outliers.update_xaxes(showgrid=False)
        st.plotly_chart(fig_outliers, use_container_width=True)

with out2:
    with st.container(border=True):
        latest_up_4 = safe_int(plot_df['Rolling_3D_Up_4'].iloc[-1]) if not plot_df['Rolling_3D_Up_4'].empty else 0
        latest_dn_4 = safe_int(plot_df['Rolling_3D_Down_4'].iloc[-1]) if not plot_df['Rolling_3D_Down_4'].empty else 0
        
        st.markdown(f"<div class='card-title' style='margin-left: 10px; margin-top: 10px;'>3-DAY ROLLING 4% THRUST MOVERS &nbsp;|&nbsp; LATEST: <span style='color:#16a34a;'>{latest_up_4} UP</span> / <span style='color:#dc2626;'>{latest_dn_4} DOWN</span></div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-desc' style='margin-left: 10px;'>Measures short-term explosive price action. Clusters of Up 4%+ days initiate new bull phases.</div>", unsafe_allow_html=True)
        
        fig_movers = go.Figure()
        fig_movers.add_trace(go.Bar(x=plot_df['Date'], y=plot_df['Rolling_3D_Up_4'], name='Up 4%+', marker_color='#22c55e'))
        fig_movers.add_trace(go.Bar(x=plot_df['Date'], y=-plot_df['Rolling_3D_Down_4'], name='Down 4%+', marker_color='#ef4444', customdata=plot_df['Rolling_3D_Down_4'], hovertemplate='Down 4%+: <b>%{customdata:.0f}</b><extra></extra>'))
        fig_movers.update_layout(barmode='relative', height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False, hovermode="x unified")
        fig_movers.update_yaxes(gridcolor='#f1f5f9', zerolinecolor='black')
        fig_movers.update_xaxes(showgrid=False)
        st.plotly_chart(fig_movers, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🔍 Stock Level Drill-Down")

selected_date_display = f"{st.session_state.analysis_date.day} {st.session_state.analysis_date.strftime('%B %Y')}"
if st.session_state.analysis_date == max_date:
    selected_date_display = actual_date_str
st.markdown(f"<p style='color: #64748b; font-size: 13px;'>Filter the underlying constituent stocks for <b>{selected_date_display}</b></p>", unsafe_allow_html=True)

@st.cache_data(max_entries=10, show_spinner=False)
def get_drilldown_data(target_date):
    try:
        df = load_trailing_cache()
        if df.empty: return pd.DataFrame()
        
        target_date_normalized = pd.Timestamp(target_date).normalize()
        
        # Just filter for the requested date, no heavy math needed!
        day_data = df[df['Date'].dt.normalize() == target_date_normalized].copy()
        if day_data.empty: return pd.DataFrame()
        
        # Keep only Active Universe and round percentages
        day_data = day_data[day_data['Active_Universe']].copy()
        day_data['Daily_Pct'] = day_data['Daily_Pct'].round(2)
        day_data['Pct_1M'] = day_data['Pct_1M'].round(2)
        
        return day_data
    except Exception as e:
        st.error(f"Failed to load drill-down data: {str(e)}")
        return pd.DataFrame()

drill_col, _ = st.columns([1.5, 1])
with drill_col:
    param_choices = st.multiselect("Select Parameters to Filter (Combines with AND):", [
        "Advances (Stocks in Green)", "Declines (Stocks in Red)", "Stocks > 20 EMA (Mature Only)", "Stocks > 50 EMA (Mature Only)", "Stocks > 200 EMA (Mature Only)",
        "Up 4% or more Today", "Down 4% or more Today", "1-Month 25% Winners", "1-Month 25% Losers", "New 52-Week Highs (Mature > 1Yr)", "New IPO/Listing Highs (< 1Yr)"
    ])

with st.spinner("🔍 Loading historical stock data..."):
    drill_data = get_drilldown_data(st.session_state.analysis_date)

if not drill_data.empty:
    res = drill_data.copy()
    if param_choices:
        for param in param_choices:
            if param == "Advances (Stocks in Green)": res = res[res['Gainer'] == True]
            elif param == "Declines (Stocks in Red)": res = res[res['Loser'] == True]
            elif param == "Stocks > 20 EMA (Mature Only)": res = res[res['Above_20_EMA'] == True]
            elif param == "Stocks > 50 EMA (Mature Only)": res = res[res['Above_50_EMA'] == True]
            elif param == "Stocks > 200 EMA (Mature Only)": res = res[res['Above_200_EMA'] == True]
            elif param == "Up 4% or more Today": res = res[res['Up_4_Pct'] == True]
            elif param == "Down 4% or more Today": res = res[res['Down_4_Pct'] == True]
            elif param == "1-Month 25% Winners": res = res[res['Up_25_1M'] == True]
            elif param == "1-Month 25% Losers": res = res[res['Down_25_1M'] == True]
            elif param == "New 52-Week Highs (Mature > 1Yr)": res = res[res['New_52W_High'] == True]
            elif param == "New IPO/Listing Highs (< 1Yr)": res = res[res['IPO_New_High'] == True]

        res = res[['Symbol', 'Close', 'Daily_Pct', 'Pct_1M']].sort_values('Daily_Pct', ascending=False)
        res = res.rename(columns={'Daily_Pct': 'Daily_%_Change', 'Pct_1M': '1M_%_Change'})
        
        st.write(f"**Found {len(res)} matching VIP active stocks** for {selected_date_display}:")
        st.dataframe(res, use_container_width=True, height=350)
    else:
        st.info("👆 Select one or more parameters above to filter the VIP stock list.")
else:
    st.info("💡 Deep dive list requires 'trailing_cache.parquet' in repository.")

st.markdown("<br><hr>", unsafe_allow_html=True)
bot_col, _ = st.columns([1.5, 4])
with bot_col:
    st.markdown(f"<div style='text-align: center; font-size: 11px; color: #94a3b8; margin-bottom: 5px;'>Last Successful Database Update: {last_sync_display}</div>", unsafe_allow_html=True)
    if not st.session_state.sync_in_progress:
        if st.button("📅 Run End of Day Analytics", use_container_width=True):
            if trigger_github_action("eod_update.yml", "EOD Sync"):
                st.session_state.sync_in_progress = True
                st.session_state.sync_start_time = time.time()
                st.session_state.pre_sync_time = last_sync_display
                st.rerun()
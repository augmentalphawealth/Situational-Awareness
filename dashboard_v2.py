import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
import os
import datetime
from io import StringIO

st.set_page_config(page_title="Situational Awareness Engine", layout="wide", initial_sidebar_state="collapsed")

# --- INITIALIZE SESSION STATE FOR SYNC TRACKING ---
if 'sync_in_progress' not in st.session_state:
    st.session_state.sync_in_progress = False
    st.session_state.sync_start_time = 0
    st.session_state.pre_sync_time = ""

st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 98%; }
    header { visibility: hidden; }
    
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff; border-radius: 12px; padding: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.03); border: 1px solid #e2e8f0;
    }
    .card-title { font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 5px; }
    .metric-value { font-size: 26px; font-weight: 800; color: #0f172a; }
    .metric-sub { font-size: 12px; font-weight: 600; color: #64748b; margin-top: 2px; }
    
    .action-banner {
        background-color: #0f172a; color: #f8fafc; padding: 10px 16px; border-radius: 8px; font-size: 13px; font-weight: 700; text-align: center; margin-top: 8px; letter-spacing: 0.5px;
    }
    
    .stButton>button { border-radius: 6px; font-weight: 600; font-size: 12px; padding: 0.3rem 0.5rem; }
    div[data-testid="stDateInput"] input { padding: 0.3rem; font-size: 13px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# --- GITHUB STRICT COMMIT FETCHING CONFIGURATION ---
REPO_OWNER = "augmentalphawealth"
REPO_NAME = "Situational-Awareness"
BRANCH = "main"

HISTORICAL_FILE = "historical_breadth_regime_6yr.csv"
INTRADAY_FILE = "live_intraday_breadth.csv"
SYNC_FILE = "last_sync.txt"

def get_latest_commit_sha():
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/commits/{BRANCH}?t={int(time.time())}"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Cache-Control": "no-cache, no-store, must-revalidate"
    }
    token = st.secrets.get("GITHUB_TOKEN", None)
    if token:
        headers["Authorization"] = f"token {token}"
        
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json().get("sha", BRANCH)
    except Exception:
        pass
    return BRANCH

def read_remote_file(path):
    sha = get_latest_commit_sha()
    url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{sha}/{path}"
    headers = {"Cache-Control": "no-cache, no-store, must-revalidate"}
    token = st.secrets.get("GITHUB_TOKEN", None)
    if token:
        headers["Authorization"] = f"token {token}"
        
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.text
    except Exception:
        pass
    return None

def get_last_updated_time():
    text = read_remote_file(SYNC_FILE)
    if text:
        return text.strip()
        
    if os.path.exists("last_sync.txt"):
        try:
            with open("last_sync.txt", "r") as f:
                content = f.read().strip()
                if content:
                    return content
        except Exception:
            pass
            
    if os.path.exists(HISTORICAL_FILE):
        mtime = os.path.getmtime(HISTORICAL_FILE)
        utc_dt = datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc)
        ist_dt = utc_dt + datetime.timedelta(hours=5, minutes=30)
        return ist_dt.strftime('%d %b %Y, %I:%M %p IST')
        
    return "Unknown"

last_sync_time = get_last_updated_time()

# --- SAFE INTRADAY DATA LOADER ---
@st.cache_data(ttl=60, show_spinner=False)
def load_intraday_data():
    try:
        csv_text = read_remote_file(INTRADAY_FILE)
        if csv_text:
            return pd.read_csv(StringIO(csv_text))
        if os.path.exists(INTRADAY_FILE):
            return pd.read_csv(INTRADAY_FILE)
    except Exception:
        pass
    return pd.DataFrame()

df_live = load_intraday_data()
is_live_active = False
live_advances = 0
live_declines = 0
live_latest = {}
last_sync_display = last_sync_time

if not df_live.empty and 'Date' in df_live.columns:
    try:
        ist_offset_local = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
        today_str_local = datetime.datetime.now(ist_offset_local).strftime('%Y-%m-%d')
        live_latest = df_live.iloc[-1]
        
        if str(live_latest['Date']) == today_str_local:
            is_live_active = True
            live_advances = int(live_latest.get('Advances', 0))
            live_declines = int(live_latest.get('Declines', 0))
            intra_time = str(live_latest.get('Time', ''))
            
            if intra_time:
                t_obj = datetime.datetime.strptime(intra_time, "%H:%M")
                last_sync_display = f"Today, {t_obj.strftime('%I:%M %p')} IST"
    except Exception:
        pass

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
            status.update(label="✅ Workflow started successfully.", state="complete")
            time.sleep(1)
            return True
        else:
            status.update(label="❌ Failed to trigger.", state="error")
            return False

@st.cache_data(ttl=300, show_spinner=False)
def load_agg_data():
    csv_text = read_remote_file(HISTORICAL_FILE)
    if csv_text:
        df = pd.read_csv(StringIO(csv_text))
        if not df.empty and 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            # Safely calculate Cumulative A/D 
            df['Cumulative_AD'] = (df.get('Advances', 0) - df.get('Declines', 0)).cumsum()
            return df
        
    if os.path.exists(HISTORICAL_FILE):
        df = pd.read_csv(HISTORICAL_FILE)
        if not df.empty and 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            df['Cumulative_AD'] = (df.get('Advances', 0) - df.get('Declines', 0)).cumsum()
            return df
        
    return pd.DataFrame()

df_agg = load_agg_data()
if df_agg.empty:
    st.error(f"Data file '{HISTORICAL_FILE}' not found. Run EOD script.")
    st.stop()

required_backend_cols = [
    'Rolling_3D_Up_4', 'Rolling_3D_Down_4', 
    'Up_25_1M_Count', 'Down_25_1M_Count',
    'Net_52W_High_Low', 'Volume_Ratio',
    'Pct_Above_20_EMA', 'Pct_Above_50_EMA', 'Pct_Above_200_EMA'
]

missing_cols = [col for col in required_backend_cols if col not in df_agg.columns]
if missing_cols:
    st.error("🛑 CRITICAL BACKEND ERROR: Missing Data Columns")
    st.warning(f"The following columns are missing from your loaded CSV:\n\n**{', '.join(missing_cols)}**")
    st.stop()

# -------------------------------------------------------------------
# VCP MOMENTUM MATRIX v4.0
# -------------------------------------------------------------------
def calculate_vcp_composite_score(latest_row, history_df):
    p_fast_val = latest_row.get('Pct_Above_20_EMA', 0) if pd.notna(latest_row.get('Pct_Above_20_EMA')) else 0
    p_slow_val = latest_row.get('Pct_Above_50_EMA', 0) if pd.notna(latest_row.get('Pct_Above_50_EMA')) else 0
    p_blend = (0.65 * p_fast_val) + (0.35 * p_slow_val)
    
    if len(history_df) >= 3:
        b_prev1 = (0.65 * history_df.iloc[-2].get('Pct_Above_20_EMA', 0)) + (0.35 * history_df.iloc[-2].get('Pct_Above_50_EMA', 0))
        b_prev2 = (0.65 * history_df.iloc[-3].get('Pct_Above_20_EMA', 0)) + (0.35 * history_df.iloc[-3].get('Pct_Above_50_EMA', 0))
        improved_2_days = (p_blend > b_prev1) and (b_prev1 > b_prev2)
        worsened_2_days = (p_blend < b_prev1) and (b_prev1 < b_prev2)
    else:
        improved_2_days = False
        worsened_2_days = False

    t3_wins = latest_row.get('T3_Wins', 0)
    t3_breaks = latest_row.get('T3_Breakouts', 0)
    ft_rate = (t3_wins / t3_breaks * 100) if pd.notna(t3_breaks) and t3_breaks > 0 else 0
    vol_ratio = latest_row.get('Volume_Ratio', 0) or 0

    if p_blend <= 10:
        b1 = 15 if (improved_2_days and vol_ratio >= 1.0 and ft_rate > 50) else 0
    elif p_blend < 50:
        b1 = 0
    elif p_blend <= 90:
        b1 = 25
    else:
        b1 = 10 if (worsened_2_days and (vol_ratio < 1.0 or ft_rate <= 50)) else 25

    b2 = 25 if ft_rate > 50 else 0
    
    hist_tail = history_df.tail(126)
    net_4d_curr = (latest_row.get('Rolling_3D_Up_4', 0) or 0) - (latest_row.get('Rolling_3D_Down_4', 0) or 0)
    net_4d_hist = (hist_tail['Rolling_3D_Up_4'] - hist_tail['Rolling_3D_Down_4']).dropna()
    p_rank_4d = (net_4d_hist < net_4d_curr).mean() if len(net_4d_hist) > 0 else 0.5
    b3_thrust = p_rank_4d * 10.0
    
    net_25m_curr = (latest_row.get('Up_25_1M_Count', 0) or 0) - (latest_row.get('Down_25_1M_Count', 0) or 0)
    net_25m_hist = (hist_tail['Up_25_1M_Count'] - hist_tail['Down_25_1M_Count']).dropna()
    p_rank_25m = (net_25m_hist < net_25m_curr).mean() if len(net_25m_hist) > 0 else 0.5
    b3_movers = p_rank_25m * 10.0
    b3 = b3_thrust + b3_movers
    
    if vol_ratio < 1.0:
        b4_vol = 0
    else:
        vol_hist = hist_tail['Volume_Ratio'].dropna()
        p_rank_vol = (vol_hist < vol_ratio).mean() if len(vol_hist) > 0 else 0.5
        b4_vol = p_rank_vol * 10.0
        
    net_hl = latest_row.get('Net_52W_High_Low', 0) or 0
    if net_hl < 0:
        b4_hl = 0
    else:
        hl_hist = hist_tail['Net_52W_High_Low'].dropna()
        p_rank_hl = (hl_hist < net_hl).mean() if len(hl_hist) > 0 else 0.5
        b4_hl = p_rank_hl * 10.0
    b4 = b4_vol + b4_hl
    
    p200 = latest_row.get('Pct_Above_200_EMA', 0) or 0
    p200_slope = 0
    if len(history_df) >= 20:
        p200_hist = history_df['Pct_Above_200_EMA'].dropna()
        if len(p200_hist) >= 20:
            p200_slope = p200_hist.iloc[-1] - p200_hist.iloc[-20]
            
    if p200 >= 50 and p200_slope >= 0: b5 = 10.0
    elif p200 < 50 and p200_slope < 0: b5 = 0.0
    else: b5 = 5.0
        
    p_large = latest_row.get('Large_Pct_50_EMA', 0) or 0
    p_small = latest_row.get('Small_Pct_50_EMA', 0) or 0
    p_micro = latest_row.get('Micro_Pct_50_EMA', 0) or 0
    hunting_ground = (p_small + p_micro) / 2
    gap = p_large - hunting_ground
    penalty = 0
    if gap >= 25.0:
        penalty = -min(15.0, (gap - 25.0) * 0.5 + 5.0)
        
    bonus = 0
    recent_20 = history_df['Pct_Above_20_EMA'].tail(20)
    if (recent_20 <= 10.0).any() and p_blend >= 50.0:
        bonus = 15.0
        
    raw_total = b1 + b2 + b3 + b4 + b5 + penalty + bonus
    final_score = int(round(max(0, min(100, raw_total))))
    
    bullets = []
    if final_score >= 51:
        if ft_rate <= 50: bullets.append(f"<span style='color: #dc2626; font-weight: 800;'>🚨 CAUTION: Breakout success rate is failing ({ft_rate:.1f}%).</span>")
        if vol_ratio < 1.0: bullets.append(f"<span style='color: #dc2626; font-weight: 800;'>🚨 CAUTION: Selling volume exceeds buying volume (Ratio: {vol_ratio:.2f}).</span>")
        if penalty < 0: bullets.append(f"<span style='color: #dc2626; font-weight: 800;'>🚨 CAUTION: Market Narrowness Penalty. Top liquidity stocks are masking weakness.</span>")
        if len(bullets) == 0: bullets.append("<span style='color: #16a34a; font-weight: 600;'>✅ All core market internals are healthy and confirming the trend.</span>")
    else:
        if ft_rate > 50: bullets.append(f"<span style='color: #16a34a; font-weight: 800;'>🌱 EARLY SIGN: Breakout success rate is holding strong ({ft_rate:.1f}%).</span>")
        if vol_ratio > 1.2: bullets.append(f"<span style='color: #16a34a; font-weight: 800;'>🌱 EARLY SIGN: Institutional Volume Ratio is aggressively positive ({vol_ratio:.2f}).</span>")
        if p_rank_4d > 0.8: bullets.append(f"<span style='color: #16a34a; font-weight: 800;'>🌱 EARLY SIGN: Momentum Thrusts are hitting the upper percentiles.</span>")
        if len(bullets) == 0: bullets.append("<span style='color: #dc2626; font-weight: 600;'>⚠️ No early signs of reversal. Continue to protect capital.</span>")

    if final_score >= 71:
        action_zone, color = "AGGRESSIVE MTF ZONE", "#22c55e"
        tacs = {"asset": "High-RS VCP Stocks", "sizing": "100% Full Sizing", "risk": "Standard 5-7% Hard Stops", "profit": "Aggressively trade valid VCP breakouts"}
    elif final_score >= 51:
        action_zone, color = "CONFIRMED UPTREND", "#84cc16"
        tacs = {"asset": "50% Broad ETFs & 50% High-RS VCP Stocks", "sizing": "50% - 75% Sizing", "risk": "Balanced Risk", "profit": "Selective buying on A+ setups only"}
    elif final_score >= 31:
        action_zone, color = "DEATH CHOP ZONE", "#f97316"
        tacs = {"asset": "Broad ETFs ONLY", "sizing": "50% Max Sizing", "risk": "0% VCP Stocks", "profit": "Use ETFs to capture broad swings safely"}
    elif final_score >= 11:
        action_zone, color = "RISK-OFF / DEFENSIVE", "#ef4444"
        tacs = {"asset": "Broad ETFs ONLY", "sizing": "25% Max Sizing", "risk": "Strict 5-7% Base Stops", "profit": "Defensive allocation, 0% VCP Stocks"}
    else:
        action_zone, color = "CAPITULATION WATCH", "#991b1b"
        tacs = {"asset": "CASH", "sizing": "0% (Pure Cash)", "risk": "Sit on hands", "profit": "Build watchlists; wait for a bounce"}
        
    return {
        "final_score": final_score,
        "p_fast": round(p_fast_val, 1),
        "action_zone": action_zone,
        "color": color,
        "bullets": bullets,
        "tactics": tacs,
        "ft_rate": round(ft_rate, 1)
    }

# --- DATE STEPPER LOGIC ---
unique_dates = df_agg['Date'].sort_values().unique()
max_date = pd.to_datetime(unique_dates[-1])
# UI FIX: UNLOCKED DATE PICKER TO ALLOW HISTORICAL SCROLLING
min_picker_date = pd.to_datetime("2018-04-01") 

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

# --- HEADER ROW ---
head_col1, head_spacer, head_col2, head_col3 = st.columns([3.0, 0.5, 2.0, 1.2])

with head_col1:
    st.markdown("<h2 style='margin-top: 10px; margin-bottom: 0px; font-weight: 800; color: #0f172a; white-space: nowrap; font-size: 24px;'>🛡️ SITUATIONAL AWARENESS</h2>", unsafe_allow_html=True)

with head_spacer:
    st.write("")

with head_col2:
    st.write("")
    nav1, nav2, nav3 = st.columns([1, 1.4, 1])
    with nav1:
        st.button("◀ Prev", on_click=step_prev_day, use_container_width=True)
    with nav2:
        selected = st.date_input("Date", value=st.session_state.analysis_date, min_value=min_picker_date, max_value=max_date, label_visibility="collapsed")
        st.session_state.analysis_date = pd.to_datetime(selected)
    with nav3:
        st.button("Next ▶", on_click=step_next_day, use_container_width=True)

with head_col3:
    st.markdown(f"<div style='text-align: right; font-size: 11px; font-weight: 600; color: #64748b; margin-top: 2px; margin-bottom: 3px;'>Last Sync: {last_sync_display}</div>", unsafe_allow_html=True)
    if st.button("⚡ Live Intraday Sync", use_container_width=True):
        if trigger_github_action("intraday_update.yml", "Intraday Sync"):
            st.session_state.sync_in_progress = True
            st.session_state.sync_start_time = time.time()
            st.session_state.pre_sync_time = last_sync_display
            st.rerun()

# --- SYNC WARNING BANNER & AUTO-REFRESH LOGIC ---
if st.session_state.sync_in_progress:
    elapsed_time = time.time() - st.session_state.sync_start_time
    
    if elapsed_time < 420:
        st.warning(f"⏳ **Background Sync in Progress:** The GitHub robot is fetching new market data. The dashboard will **automatically refresh** when finished. (Elapsed: {int(elapsed_time)}s)", icon="🤖")
        time.sleep(15)
        current_sync_time = get_last_updated_time()
        
        if current_sync_time != st.session_state.pre_sync_time and current_sync_time != "Unknown":
            st.session_state.sync_in_progress = False
            st.cache_data.clear()
            st.rerun()
        else:
            st.rerun()
    else:
        st.session_state.sync_in_progress = False
        st.error("Sync timed out or took too long. Please try clicking Live Intraday Sync again.")
        
    st.markdown("<hr style='margin: 10px 0px;'>", unsafe_allow_html=True)

df_filtered = df_agg[df_agg['Date'] <= st.session_state.analysis_date].copy()
if df_filtered.empty:
    st.warning("No trading data available on or before this date.")
    st.stop()

latest = df_filtered.iloc[-1]
prev = df_filtered.iloc[-2] if len(df_filtered) > 1 else latest

# --- VCP COMPOSITE SCORE DISPLAY ---
vcp_res = calculate_vcp_composite_score(latest, df_filtered)
score = vcp_res['final_score']
score_color = vcp_res['color']

df_10d = df_filtered.tail(10).copy()
df_10d['Date_Str'] = df_10d['Date'].dt.strftime('%d %b')
scores_10d = [calculate_vcp_composite_score(row, df_filtered[df_filtered['Date'] <= row['Date']])['final_score'] for _, row in df_10d.iterrows()]
df_10d['Composite_Score'] = scores_10d

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
                <div class='card-title' style='font-size: 12px;'>MOMENTUM HEALTH SCORE</div>
                <div class='metric-value' style='font-size: 44px; color: {score_color};'>{score} <span style='font-size: 18px; color: #94a3b8;'>/ 100</span></div>
                <div style='font-size: 11px; color: #64748b; font-weight: 600; margin-top: 2px;'>Fast Breadth (P_Fast): {vcp_res['p_fast']}%</div>
            </div>
        """, unsafe_allow_html=True)
    with top_c2:
        fig_m1 = go.Figure(go.Bar(
            x=df_10d['Date_Str'], 
            y=df_10d['Composite_Score'], 
            marker_color=[get_bar_color(v) for v in df_10d['Composite_Score']],
            text=df_10d['Composite_Score'],
            textposition='auto',
            hovertemplate='Score: <b>%{y}</b><extra></extra>'
        ))
        fig_m1.update_layout(height=120, margin=dict(l=10, r=10, t=25, b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False, title=dict(text="10-Day Health Trend", font=dict(size=11, color="#64748b")))
        fig_m1.update_xaxes(showgrid=False, tickfont=dict(size=10, color="#94a3b8"))
        fig_m1.update_yaxes(visible=False, range=[0, 100])
        st.plotly_chart(fig_m1, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown(f"<div class='action-banner'>🎯 ACTION ZONE: {vcp_res['action_zone']}</div>", unsafe_allow_html=True)

# --- SAFE INTRADAY OVERRIDE (HERO ROW ONLY) ---
actual_date_str = latest['Date'].strftime('%d %b %Y')
total_univ = int(latest.get('Total_Universe', 2400))
advances = int(latest.get('Advances', 0))
declines = int(latest.get('Declines', 0))

# Only override if live data exists AND the user is looking at the most recent day.
if is_live_active and st.session_state.analysis_date == max_date:
    advances = live_advances
    declines = live_declines
    
    total_univ = int(live_latest.get('Total_Universe', latest.get('Total_Universe', 2400))) 
    
    ist_offset_local = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    actual_date_str = f"{datetime.datetime.now(ist_offset_local).strftime('%d %b %Y')} <span style='color:#eab308; font-weight:800;'>(⚡ LIVE INTRADAY)</span>"

# --- DAILY HERO ROW ---
st.markdown(f"<p style='color: #475569; font-size: 13px; font-weight: 600; margin-top: 15px;'>Market Breadth Status for: <span style='color:#0f172a;'>{actual_date_str}</span></p>", unsafe_allow_html=True)
hero_col1, hero_col2, hero_col3 = st.columns([1.1, 1.5, 1.4])

total_adv_dec = advances + declines
adv_pct = round((advances / total_adv_dec) * 100, 1) if total_adv_dec > 0 else 0

prev_advances = int(prev.get('Advances', 0))
prev_declines = int(prev.get('Declines', 0))
prev_total_adv_dec = prev_advances + prev_declines
prev_adv_pct = round((prev_advances / prev_total_adv_dec) * 100, 1) if prev_total_adv_dec > 0 else 0
adv_change = round(adv_pct - prev_adv_pct, 1)
adv_change_str = f"+{adv_change}%" if adv_change >= 0 else f"{adv_change}%"
adv_change_color = "#16a34a" if adv_change >= 0 else "#dc2626"

with hero_col1:
    with st.container(border=True):
        st.markdown("<div class='card-title' style='text-align: center; margin-top: 5px;'>OF UNIVERSE ADVANCING</div>", unsafe_allow_html=True)
        fig_dial = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = adv_pct,
            number = {'suffix': "%", 'font': {'size': 36, 'color': '#0f172a'}},
            gauge = {
                'axis': {'range': [0, 100], 'visible': False},
                'bar': {'color': "#ef4444" if adv_pct < 50 else "#22c55e", 'thickness': 0.18},
                'bgcolor': "#f1f5f9",
                'borderwidth': 0,
            }
        ))
        fig_dial.update_layout(height=150, margin=dict(l=10, r=10, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_dial, use_container_width=True, config={'displayModeBar': False})
        st.markdown(f"""
            <div style='text-align: center; margin-top: -10px; padding-bottom: 8px;'>
                <span style='color: #16a34a; font-size: 15px; font-weight: 800;'>{advances} ADVANCES</span> &nbsp;&nbsp;
                <span style='color: #dc2626; font-size: 15px; font-weight: 800;'>{declines} DECLINES</span>
                <p style='color: {adv_change_color}; font-size: 12px; font-weight: 700; margin-top: 4px; margin-bottom: 0px;'>{adv_change_str} vs Yesterday EOD (Univ {total_univ})</p>
            </div>
        """, unsafe_allow_html=True)

with hero_col2:
    with st.container(border=True):
        st.markdown("<div class='card-title' style='margin-top: 5px; margin-left: 10px;'>TACTICAL COMMAND CENTER</div>", unsafe_allow_html=True)
        tacs = vcp_res['tactics']
        
        st.markdown(f"""
            <div style='padding: 0px 10px; font-size: 13px;'>
                <div style='margin-bottom: 6px;'><b>🎯 Target Asset:</b> <span style='color: #334155;'>{tacs['asset']}</span></div>
                <div style='margin-bottom: 6px;'><b>⚖️ Position Sizing:</b> <span style='color: #334155;'>{tacs['sizing']}</span></div>
                <div style='margin-bottom: 6px;'><b>🛡️ Risk / Stop-Loss:</b> <span style='color: #334155;'>{tacs['risk']}</span></div>
                <div style='margin-bottom: 6px;'><b>💰 Profit Strategy:</b> <span style='color: #334155;'>{tacs['profit']}</span></div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<hr style='margin: 8px 0px;'>", unsafe_allow_html=True)
        st.markdown(f"<div class='card-title' style='margin-left: 10px;'>SCORE DRIVERS (BREAKOUT WIN RATE: {vcp_res['ft_rate']:.1f}%)</div>", unsafe_allow_html=True)
        
        bullets_html = "".join([f"<div style='font-size: 11px; margin-bottom: 3px; padding-left: 10px;'>{b}</div>" for b in vcp_res['bullets']])
        st.markdown(bullets_html, unsafe_allow_html=True)

with hero_col3:
    # UI FIX: REPLACED CAPITAL FLOW DIAL WITH CUMULATIVE A/D LINE CHART
    with st.container(border=True):
        st.markdown("<div class='card-title' style='text-align: center; margin-top: 5px;'>CUMULATIVE A/D LINE (MARKET TREND)</div>", unsafe_allow_html=True)
        cad_plot = df_filtered.tail(126)
        fig_cad = go.Figure()
        fig_cad.add_trace(go.Scatter(
            x=cad_plot['Date'], y=cad_plot['Cumulative_AD'], 
            fill='tozeroy', name="Cum. A/D", line=dict(color='#3b82f6', width=2)
        ))
        fig_cad.update_layout(
            height=165, margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", 
            showlegend=False, hovermode="x unified"
        )
        fig_cad.update_xaxes(showgrid=False, visible=False)
        fig_cad.update_yaxes(showgrid=False, visible=False)
        st.plotly_chart(fig_cad, use_container_width=True, config={'displayModeBar': False})
        st.markdown("<div style='text-align: center; font-size: 11px; font-weight: 600; color: #64748b; margin-top: -10px;'>A continuously rising line confirms broad participation.</div>", unsafe_allow_html=True)


# --- SECONDARY METRICS ---
m2, m3 = st.columns(2)

with m2:
    vol_ratio = latest['Volume_Ratio'] if pd.notna(latest['Volume_Ratio']) else 0
    v_col = "#22c55e" if vol_ratio > 1.0 else "#ef4444"
    with st.container(border=True):
        st.markdown(f"<div style='text-align: center;'><div class='card-title' style='margin-top: 5px;'>Volume Breadth Ratio</div><div class='metric-value' style='color: {v_col};'>{vol_ratio:.2f}</div><div class='metric-sub'>Liquidity Flow: Up-Turnover vs Down-Turnover (₹)</div></div>", unsafe_allow_html=True)
        fig_m2 = go.Figure(go.Bar(
            x=df_10d['Date_Str'], y=df_10d['Volume_Ratio'], 
            marker_color=['#22c55e' if v >= 1.0 else '#ef4444' for v in df_10d['Volume_Ratio']],
            hovertemplate='Ratio: %{y:.2f}<extra></extra>'
        ))
        fig_m2.add_hline(y=1.0, line_dash="dash", line_color="#cbd5e1")
        fig_m2.update_layout(height=100, margin=dict(l=5, r=5, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        fig_m2.update_xaxes(showgrid=False, tickfont=dict(size=10, color="#94a3b8"))
        fig_m2.update_yaxes(visible=False)
        st.plotly_chart(fig_m2, use_container_width=True, config={'displayModeBar': False})

with m3:
    net_hl = int(latest['Net_52W_High_Low']) if pd.notna(latest['Net_52W_High_Low']) else 0
    h_col = "#22c55e" if net_hl > 0 else "#ef4444"
    with st.container(border=True):
        st.markdown(f"<div style='text-align: center;'><div class='card-title' style='margin-top: 5px;'>Net 52-Week Highs vs Lows</div><div class='metric-value' style='color: {h_col};'>{net_hl}</div><div class='metric-sub'>New Highs Minus New Lows</div></div>", unsafe_allow_html=True)
        fig_m3 = go.Figure(go.Bar(
            x=df_10d['Date_Str'], y=df_10d['Net_52W_High_Low'], 
            marker_color=['#22c55e' if v >= 0 else '#ef4444' for v in df_10d['Net_52W_High_Low']],
            hovertemplate='Net HL: %{y:.0f}<extra></extra>'
        ))
        fig_m3.add_hline(y=0, line_dash="solid", line_color="#cbd5e1")
        fig_m3.update_layout(height=100, margin=dict(l=5, r=5, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        fig_m3.update_xaxes(showgrid=False, tickfont=dict(size=10, color="#94a3b8"))
        fig_m3.update_yaxes(visible=False)
        st.plotly_chart(fig_m3, use_container_width=True, config={'displayModeBar': False})

# --- HISTORICAL ANALYTICS ---
st.markdown("<br>", unsafe_allow_html=True)
tf_col1, tf_col2 = st.columns([3, 1])
with tf_col1:
    st.markdown("### 📊 Historical Market Analytics")
with tf_col2:
    timeframe = st.radio("Chart Horizon:", ["1 Month", "3 Months", "6 Months", "1 Year", "3 Years", "6 Years"], horizontal=True, index=2)

days_map = {"1 Month": 21, "3 Months": 63, "6 Months": 126, "1 Year": 252, "3 Years": 756, "6 Years": len(df_filtered)}
days_limit = days_map.get(timeframe, 126)
plot_df = df_filtered.tail(days_limit)

# SECTION 3: UNIVERSE EMA BREADTH
with st.container(border=True):
    val_200 = plot_df['Pct_Above_200_EMA'].iloc[-1]
    val_50 = plot_df['Pct_Above_50_EMA'].iloc[-1]
    val_20 = plot_df['Pct_Above_20_EMA'].iloc[-1]
    
    st.markdown(f"<div class='card-title' style='margin-left: 10px; margin-top: 10px;'>UNIVERSE EMA BREADTH TRENDS &nbsp;|&nbsp; LATEST: <span style='color:#22c55e;'>200 EMA ({val_200:.1f}%)</span> • <span style='color:#a855f7;'>50 EMA ({val_50:.1f}%)</span> • <span style='color:#3b82f6;'>20 EMA ({val_20:.1f}%)</span></div>", unsafe_allow_html=True)
    
    fig_ema = go.Figure()
    # UI FIX: Added connectgaps=True to force 200 EMA lines to draw smoothly even if data points are missing
    fig_ema.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Pct_Above_200_EMA'], mode='lines', name='% > 200 EMA', line=dict(color='#22c55e', width=2), hovertemplate='200 EMA: %{y:.1f}%<extra></extra>', connectgaps=True))
    fig_ema.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Pct_Above_50_EMA'], mode='lines', name='% > 50 EMA', line=dict(color='#a855f7', width=2), hovertemplate='50 EMA: %{y:.1f}%<extra></extra>', connectgaps=True))
    fig_ema.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Pct_Above_20_EMA'], mode='lines', name='% > 20 EMA', line=dict(color='#3b82f6', width=2), hovertemplate='20 EMA: %{y:.1f}%<extra></extra>', connectgaps=True))
    fig_ema.add_hline(y=50, line_dash="dash", line_color="#94a3b8", opacity=0.7)
    
    fig_ema.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10), 
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", 
        hovermode="x unified", 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    fig_ema.update_yaxes(range=[0, 100], gridcolor='#f1f5f9', title="% Stocks")
    fig_ema.update_xaxes(showgrid=False)
    st.plotly_chart(fig_ema, use_container_width=True)

# SECTION 4: SEGMENTED LIQUIDITY FLOW (UI TEXT UPDATED)
with st.container(border=True):
    st.markdown("<div class='card-title' style='margin-left: 10px; margin-top: 10px;'>SEGMENTED LIQUIDITY FLOW (45-DAY ROLLING TURNOVER RANK)</div>", unsafe_allow_html=True)
    cap_tab1, cap_tab2, cap_tab3 = st.tabs(["% Stocks Above 200 EMA", "% Stocks Above 50 EMA", "% Stocks Above 20 EMA"])
    
    l200 = plot_df.get('Large_Pct_200_EMA', pd.Series([0])).iloc[-1]
    m200 = plot_df.get('Mid_Pct_200_EMA', pd.Series([0])).iloc[-1]
    s200 = plot_df.get('Small_Pct_200_EMA', pd.Series([0])).iloc[-1]
    mi200 = plot_df.get('Micro_Pct_200_EMA', pd.Series([0])).iloc[-1]
    
    l50 = plot_df.get('Large_Pct_50_EMA', pd.Series([0])).iloc[-1]
    m50 = plot_df.get('Mid_Pct_50_EMA', pd.Series([0])).iloc[-1]
    s50 = plot_df.get('Small_Pct_50_EMA', pd.Series([0])).iloc[-1]
    mi50 = plot_df.get('Micro_Pct_50_EMA', pd.Series([0])).iloc[-1]
    
    l20 = plot_df.get('Large_Pct_20_EMA', pd.Series([0])).iloc[-1]
    m20 = plot_df.get('Mid_Pct_20_EMA', pd.Series([0])).iloc[-1]
    s20 = plot_df.get('Small_Pct_20_EMA', pd.Series([0])).iloc[-1]
    mi20 = plot_df.get('Micro_Pct_20_EMA', pd.Series([0])).iloc[-1]

    with cap_tab1:
        st.markdown(f"<div style='font-size: 11px; font-weight: 700; color: #64748b; margin-top: -10px; margin-bottom: 5px; padding-left: 10px;'>LATEST: <span style='color:#2563eb;'>Top 100 Liq ({l200:.1f}%)</span> &nbsp;|&nbsp; <span style='color:#f97316;'>Mid 150 Liq ({m200:.1f}%)</span> &nbsp;|&nbsp; <span style='color:#16a34a;'>Lower 250 Liq ({s200:.1f}%)</span> &nbsp;|&nbsp; <span style='color:#dc2626;'>Micro Liq ({mi200:.1f}%)</span></div>", unsafe_allow_html=True)
        fig_c200 = go.Figure()
        # UI FIX: connectgaps=True added here for 200 EMA visibility
        fig_c200.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df.get('Large_Pct_200_EMA', pd.Series(dtype=float)), mode='lines', name='Top 100 Liq', line=dict(color='#2563eb', width=2), hovertemplate='Top 100: %{y:.1f}%<extra></extra>', connectgaps=True))
        fig_c200.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df.get('Mid_Pct_200_EMA', pd.Series(dtype=float)), mode='lines', name='Mid 150 Liq', line=dict(color='#f97316', width=2), hovertemplate='Mid 150: %{y:.1f}%<extra></extra>', connectgaps=True))
        fig_c200.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df.get('Small_Pct_200_EMA', pd.Series(dtype=float)), mode='lines', name='Lower 250 Liq', line=dict(color='#16a34a', width=2), hovertemplate='Lower 250: %{y:.1f}%<extra></extra>', connectgaps=True))
        fig_c200.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df.get('Micro_Pct_200_EMA', pd.Series(dtype=float)), mode='lines', name='Micro Liq', line=dict(color='#dc2626', width=2), hovertemplate='Micro Liq: %{y:.1f}%<extra></extra>', connectgaps=True))
        fig_c200.add_hline(y=50, line_dash="dash", line_color="#94a3b8", opacity=0.7)
        fig_c200.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig_c200.update_yaxes(range=[0, 100], gridcolor='#f1f5f9')
        fig_c200.update_xaxes(showgrid=False)
        st.plotly_chart(fig_c200, use_container_width=True)

    with cap_tab2:
        st.markdown(f"<div style='font-size: 11px; font-weight: 700; color: #64748b; margin-top: -10px; margin-bottom: 5px; padding-left: 10px;'>LATEST: <span style='color:#2563eb;'>Top 100 Liq ({l50:.1f}%)</span> &nbsp;|&nbsp; <span style='color:#f97316;'>Mid 150 Liq ({m50:.1f}%)</span> &nbsp;|&nbsp; <span style='color:#16a34a;'>Lower 250 Liq ({s50:.1f}%)</span> &nbsp;|&nbsp; <span style='color:#dc2626;'>Micro Liq ({mi50:.1f}%)</span></div>", unsafe_allow_html=True)
        fig_c50 = go.Figure()
        fig_c50.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df.get('Large_Pct_50_EMA', pd.Series(dtype=float)), mode='lines', name='Top 100 Liq', line=dict(color='#2563eb', width=2), hovertemplate='Top 100: %{y:.1f}%<extra></extra>', connectgaps=True))
        fig_c50.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df.get('Mid_Pct_50_EMA', pd.Series(dtype=float)), mode='lines', name='Mid 150 Liq', line=dict(color='#f97316', width=2), hovertemplate='Mid 150: %{y:.1f}%<extra></extra>', connectgaps=True))
        fig_c50.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df.get('Small_Pct_50_EMA', pd.Series(dtype=float)), mode='lines', name='Lower 250 Liq', line=dict(color='#16a34a', width=2), hovertemplate='Lower 250: %{y:.1f}%<extra></extra>', connectgaps=True))
        fig_c50.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df.get('Micro_Pct_50_EMA', pd.Series(dtype=float)), mode='lines', name='Micro Liq', line=dict(color='#dc2626', width=2), hovertemplate='Micro Liq: %{y:.1f}%<extra></extra>', connectgaps=True))
        fig_c50.add_hline(y=50, line_dash="dash", line_color="#94a3b8", opacity=0.7)
        fig_c50.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig_c50.update_yaxes(range=[0, 100], gridcolor='#f1f5f9')
        fig_c50.update_xaxes(showgrid=False)
        st.plotly_chart(fig_c50, use_container_width=True)

    with cap_tab3:
        st.markdown(f"<div style='font-size: 11px; font-weight: 700; color: #64748b; margin-top: -10px; margin-bottom: 5px; padding-left: 10px;'>LATEST: <span style='color:#2563eb;'>Top 100 Liq ({l20:.1f}%)</span> &nbsp;|&nbsp; <span style='color:#f97316;'>Mid 150 Liq ({m20:.1f}%)</span> &nbsp;|&nbsp; <span style='color:#16a34a;'>Lower 250 Liq ({s20:.1f}%)</span> &nbsp;|&nbsp; <span style='color:#dc2626;'>Micro Liq ({mi20:.1f}%)</span></div>", unsafe_allow_html=True)
        fig_c20 = go.Figure()
        fig_c20.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df.get('Large_Pct_20_EMA', pd.Series(dtype=float)), mode='lines', name='Top 100 Liq', line=dict(color='#2563eb', width=2), hovertemplate='Top 100: %{y:.1f}%<extra></extra>', connectgaps=True))
        fig_c20.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df.get('Mid_Pct_20_EMA', pd.Series(dtype=float)), mode='lines', name='Mid 150 Liq', line=dict(color='#f97316', width=2), hovertemplate='Mid 150: %{y:.1f}%<extra></extra>', connectgaps=True))
        fig_c20.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df.get('Small_Pct_20_EMA', pd.Series(dtype=float)), mode='lines', name='Lower 250 Liq', line=dict(color='#16a34a', width=2), hovertemplate='Lower 250: %{y:.1f}%<extra></extra>', connectgaps=True))
        fig_c20.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df.get('Micro_Pct_20_EMA', pd.Series(dtype=float)), mode='lines', name='Micro Liq', line=dict(color='#dc2626', width=2), hovertemplate='Micro Liq: %{y:.1f}%<extra></extra>', connectgaps=True))
        fig_c20.add_hline(y=50, line_dash="dash", line_color="#94a3b8", opacity=0.7)
        fig_c20.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig_c20.update_yaxes(range=[0, 100], gridcolor='#f1f5f9')
        fig_c20.update_xaxes(showgrid=False)
        st.plotly_chart(fig_c20, use_container_width=True)

# SECTION 5: MOMENTUM THRUST & OUTLIERS
out1, out2 = st.columns(2)
with out1:
    with st.container(border=True):
        latest_up_25 = int(plot_df['Up_25_1M_Count'].iloc[-1])
        latest_dn_25 = int(plot_df['Down_25_1M_Count'].iloc[-1])
        
        st.markdown(f"<div class='card-title' style='margin-left: 10px; margin-top: 10px;'>ROLLING 1-MONTH 25% MOVERS &nbsp;|&nbsp; LATEST: <span style='color:#16a34a;'>{latest_up_25} UP</span> / <span style='color:#dc2626;'>{latest_dn_25} DOWN</span></div>", unsafe_allow_html=True)
        
        fig_outliers = go.Figure()
        fig_outliers.add_trace(go.Bar(x=plot_df['Date'], y=plot_df['Up_25_1M_Count'], name='Up 25%+ in 1M', marker_color='#22c55e', hovertemplate='Up 25%: <b>%{y:.0f}</b><extra></extra>'))
        fig_outliers.add_trace(go.Bar(x=plot_df['Date'], y=-plot_df['Down_25_1M_Count'], name='Down 25%+ in 1M', marker_color='#ef4444', customdata=plot_df['Down_25_1M_Count'], hovertemplate='Down 25%: <b>%{customdata:.0f}</b><extra></extra>'))
        fig_outliers.update_layout(barmode='relative', height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False, hovermode="x unified")
        fig_outliers.update_yaxes(gridcolor='#f1f5f9', zerolinecolor='black')
        fig_outliers.update_xaxes(showgrid=False)
        st.plotly_chart(fig_outliers, use_container_width=True)

with out2:
    with st.container(border=True):
        latest_up_4 = int(plot_df['Rolling_3D_Up_4'].iloc[-1])
        latest_dn_4 = int(plot_df['Rolling_3D_Down_4'].iloc[-1])
        
        st.markdown(f"<div class='card-title' style='margin-left: 10px; margin-top: 10px;'>3-DAY ROLLING 4% THRUST MOVERS &nbsp;|&nbsp; LATEST: <span style='color:#16a34a;'>{latest_up_4} UP</span> / <span style='color:#dc2626;'>{latest_dn_4} DOWN</span></div>", unsafe_allow_html=True)
        
        fig_movers = go.Figure()
        fig_movers.add_trace(go.Bar(x=plot_df['Date'], y=plot_df['Rolling_3D_Up_4'], name='Up 4%+', marker_color='#22c55e', hovertemplate='Up 4%+: <b>%{y:.0f}</b><extra></extra>'))
        fig_movers.add_trace(go.Bar(x=plot_df['Date'], y=-plot_df['Rolling_3D_Down_4'], name='Down 4%+', marker_color='#ef4444', customdata=plot_df['Rolling_3D_Down_4'], hovertemplate='Down 4%+: <b>%{customdata:.0f}</b><extra></extra>'))
        fig_movers.update_layout(barmode='relative', height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False, hovermode="x unified")
        fig_movers.update_yaxes(gridcolor='#f1f5f9', zerolinecolor='black')
        fig_movers.update_xaxes(showgrid=False)
        st.plotly_chart(fig_movers, use_container_width=True)


# --- SECTION 6: TACTICAL EXERTION & CAPITULATION (NEW ADDITIONS) ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🔥 Tactical Exertion & Capitulation Indicators")
st.markdown("<p style='color: #64748b; font-size: 13px;'>Monitors momentum velocity, panic selling, and breakout success rates.</p>", unsafe_allow_html=True)

tac_col1, tac_col2, tac_col3 = st.columns(3)

with tac_col1:
    with st.container(border=True):
        st.markdown("<div class='card-title' style='margin-left: 10px; margin-top: 10px;'>McCLELLAN OSCILLATOR (MCO)</div>", unsafe_allow_html=True)
        fig_mco = go.Figure()
        # Safe fetch using .get in case MCO column is missing mid-sync
        mco_series = plot_df.get('MCO', pd.Series(np.zeros(len(plot_df)), index=plot_df.index))
        colors = ['#22c55e' if val >= 0 else '#ef4444' for val in mco_series]
        fig_mco.add_trace(go.Bar(x=plot_df['Date'], y=mco_series, marker_color=colors, name="MCO"))
        fig_mco.add_hline(y=0, line_dash="solid", line_color="#94a3b8")
        fig_mco.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False, hovermode="x unified")
        fig_mco.update_xaxes(showgrid=False)
        fig_mco.update_yaxes(gridcolor='#f1f5f9')
        st.plotly_chart(fig_mco, use_container_width=True)
        st.caption("Measures velocity of money moving into advancing stocks. Dropping < 0 warns of short-term exhaustion.")

with tac_col2:
    with st.container(border=True):
        st.markdown("<div class='card-title' style='margin-left: 10px; margin-top: 10px;'>TRIN (ARMS INDEX)</div>", unsafe_allow_html=True)
        fig_trin = go.Figure()
        # Safe fetch using .get in case TRIN column is missing mid-sync
        trin_series = plot_df.get('TRIN', pd.Series(np.ones(len(plot_df)), index=plot_df.index))
        fig_trin.add_trace(go.Scatter(x=plot_df['Date'], y=trin_series, mode='lines', line=dict(color='#ab63fa', width=2), name="TRIN", connectgaps=True))
        fig_trin.add_hline(y=2.0, line_dash="dash", line_color="#ef4444", annotation_text="Extreme Panic (>2.0)", annotation_position="top left")
        fig_trin.add_hline(y=0.5, line_dash="dash", line_color="#22c55e", annotation_text="Froth (<0.5)", annotation_position="bottom left")
        fig_trin.update_yaxes(range=[0, min(4.0, trin_series.max() + 0.5)], gridcolor='#f1f5f9')
        fig_trin.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False, hovermode="x unified")
        fig_trin.update_xaxes(showgrid=False)
        st.plotly_chart(fig_trin, use_container_width=True)
        st.caption("A reading > 2.0 indicates extreme panic selling. Market bottoms often form precisely on days with maximum fear.")

with tac_col3:
    with st.container(border=True):
        st.markdown("<div class='card-title' style='margin-left: 10px; margin-top: 10px;'>BREAKOUT SUCCESS VS FAILURE</div>", unsafe_allow_html=True)
        fig_brk = go.Figure()
        brk_series = plot_df.get('T3_Breakouts', pd.Series(np.zeros(len(plot_df)), index=plot_df.index))
        win_series = plot_df.get('T3_Wins', pd.Series(np.zeros(len(plot_df)), index=plot_df.index))
        fig_brk.add_trace(go.Bar(x=plot_df['Date'], y=brk_series, name="Total Attempted", marker_color='rgba(148, 163, 184, 0.4)'))
        fig_brk.add_trace(go.Bar(x=plot_df['Date'], y=win_series, name="Successful Holds", marker_color='#eab308'))
        fig_brk.update_layout(barmode='overlay', height=220, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False, hovermode="x unified")
        fig_brk.update_xaxes(showgrid=False)
        fig_brk.update_yaxes(gridcolor='#f1f5f9')
        st.plotly_chart(fig_brk, use_container_width=True)
        st.caption("Visualizes raw breakout win-rate expanding or contracting in real-time.")

# --- SECTION 7: MULTI-SELECT DEEP DIVE STOCK INSPECTOR ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🔍 Stock Level Drill-Down")
st.markdown(f"<p style='color: #64748b; font-size: 13px;'>Filter the underlying constituent stocks for <b>{actual_date_str}</b></p>", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def get_drilldown_data(target_date):
    try:
        raw_df = pd.read_parquet("nse_6yr_historical.parquet")
        raw_df['Date'] = pd.to_datetime(raw_df['Date'])
        start_date = target_date - pd.Timedelta(days=300)
        subset = raw_df[(raw_df['Date'] >= start_date) & (raw_df['Date'] <= target_date)].copy()
        subset = subset.sort_values(['Symbol', 'Date'])
        
        subset['EMA_20'] = subset.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=20, adjust=False, min_periods=20).mean())
        subset['EMA_50'] = subset.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=50, adjust=False, min_periods=50).mean())
        subset['EMA_200'] = subset.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=200, adjust=False, min_periods=200).mean())
        
        subset['Daily_%_Change'] = subset.groupby('Symbol')['Close'].pct_change() * 100
        subset['1M_%_Change'] = subset.groupby('Symbol')['Close'].pct_change(periods=21) * 100
        subset['52W_High'] = subset.groupby('Symbol')['High'].transform(lambda x: x.rolling(window=252, min_periods=50).max())
        subset['52W_Low'] = subset.groupby('Symbol')['Low'].transform(lambda x: x.rolling(window=252, min_periods=50).min())
        
        closest_date = subset[subset['Date'] <= target_date]['Date'].max()
        day_data = subset[subset['Date'] == closest_date].copy()
        day_data['Daily_%_Change'] = day_data['Daily_%_Change'].round(2)
        day_data['1M_%_Change'] = day_data['1M_%_Change'].round(2)
        return day_data
    except Exception:
        return pd.DataFrame()

drill_col, _ = st.columns([1.5, 1])
with drill_col:
    param_choices = st.multiselect("Select Parameters to Filter (Combines with AND):", [
        "Advances (Stocks in Green)",
        "Declines (Stocks in Red)",
        "Stocks > 20 EMA",
        "Stocks > 50 EMA",
        "Stocks > 200 EMA",
        "Up 4% or more Today",
        "Down 4% or more Today",
        "1-Month 25% Winners",
        "1-Month 25% Losers",
        "New 52-Week Highs",
        "New 52-Week Lows"
    ])

drill_data = get_drilldown_data(latest['Date'])

if not drill_data.empty:
    res = drill_data.copy()
    if param_choices:
        for param in param_choices:
            if param == "Advances (Stocks in Green)": res = res[res['Daily_%_Change'] > 0]
            elif param == "Declines (Stocks in Red)": res = res[res['Daily_%_Change'] < 0]
            elif param == "Stocks > 20 EMA": res = res[res['Close'] > res['EMA_20']]
            elif param == "Stocks > 50 EMA": res = res[res['Close'] > res['EMA_50']]
            elif param == "Stocks > 200 EMA": res = res[res['Close'] > res['EMA_200']]
            elif param == "Up 4% or more Today": res = res[res['Daily_%_Change'] >= 4.0]
            elif param == "Down 4% or more Today": res = res[res['Daily_%_Change'] <= -4.0]
            elif param == "1-Month 25% Winners": res = res[res['1M_%_Change'] >= 25.0]
            elif param == "1-Month 25% Losers": res = res[res['1M_%_Change'] <= -25.0]
            elif param == "New 52-Week Highs": res = res[res['Close'] >= res['52W_High']]
            elif param == "New 52-Week Lows": res = res[res['Close'] <= res['52W_Low']]

        res = res[['Symbol', 'Close', 'Daily_%_Change', '1M_%_Change']].sort_values('Daily_%_Change', ascending=False)
        st.write(f"**Found {len(res)} matching stocks** for {actual_date_str}:")
        st.dataframe(res, use_container_width=True, height=350)
    else:
        st.info("👆 Select one or more parameters above to filter the stock list.")
else:
    st.info("💡 Deep dive list requires 'nse_6yr_historical.parquet' in repository.")

st.markdown("<br><hr>", unsafe_allow_html=True)
bot_col, _ = st.columns([1.5, 4])
with bot_col:
    st.markdown(f"<div style='text-align: center; font-size: 11px; color: #94a3b8; margin-bottom: 5px;'>Last Successful Database Update: {last_sync_display}</div>", unsafe_allow_html=True)
    if st.button("📅 Upgrade Database & Purge Bad Data", use_container_width=True):
        if trigger_github_action("eod_update.yml", "EOD Sync"):
            st.session_state.sync_in_progress = True
            st.session_state.sync_start_time = time.time()
            st.session_state.pre_sync_time = last_sync_display
            st.rerun()

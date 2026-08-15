import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# --- CSS STYLING (UPGRADED FOR PROPER FITS & READABILITY) ---
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 98%; }
    header { visibility: hidden; }
    
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff; border-radius: 12px; padding: 18px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; margin-bottom: 20px;
    }
    .panel-header { font-size: 22px; font-weight: 900; color: #0f172a; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 15px; border-bottom: 3px solid #e2e8f0; padding-bottom: 8px;}
    .card-title { font-size: 15px; font-weight: 800; color: #475569; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 8px; }
    .metric-value { font-size: 34px; font-weight: 900; color: #0f172a; }
    .metric-sub { font-size: 14px; font-weight: 700; color: #64748b; margin-top: 4px; }
    
    .status-badge {
        padding: 8px 16px; border-radius: 8px; font-size: 15px; font-weight: 900; display: inline-block; letter-spacing: 0.5px;
    }
    .badge-bull { background-color: #dcfce7; color: #166534; border: 2px solid #22c55e; }
    .badge-bear { background-color: #fee2e2; color: #991b1b; border: 2px solid #ef4444; }
    .badge-warn { background-color: #fef08a; color: #854d0e; border: 2px solid #eab308; }
    .badge-thrust { background-color: #dbeafe; color: #1e40af; border: 2px solid #3b82f6; }
    
    .chart-explainer { font-size: 14px; color: #475569; font-style: italic; margin-top: 5px; margin-bottom: 15px; padding-left: 5px; font-weight: 600;}

    .stButton>button { border-radius: 8px; font-weight: 800; font-size: 14px; padding: 0.5rem 1rem; }
    div[data-testid="stDateInput"] input { padding: 0.4rem; font-size: 15px; text-align: center; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

# --- GITHUB STRICT COMMIT FETCHING CONFIGURATION ---
REPO_OWNER = "Augmentalphawealth"
REPO_NAME = "Situational-Awareness"
BRANCH = "main"

HISTORICAL_FILE = "historical_breadth_regime_6yr.csv"
INTRADAY_FILE = "live_intraday_breadth.csv"
SYNC_FILE = "last_sync.txt"

def get_latest_commit_sha():
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/commits/{BRANCH}?t={int(time.time())}"
    headers = {"Accept": "application/vnd.github.v3+json", "Cache-Control": "no-cache, no-store, must-revalidate"}
    token = st.secrets.get("GITHUB_TOKEN", None)
    if token: headers["Authorization"] = f"token {token}"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200: return response.json().get("sha", BRANCH)
    except Exception: pass
    return BRANCH

def read_remote_file(path):
    sha = get_latest_commit_sha()
    url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{sha}/{path}"
    headers = {"Cache-Control": "no-cache, no-store, must-revalidate"}
    token = st.secrets.get("GITHUB_TOKEN", None)
    if token: headers["Authorization"] = f"token {token}"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200: return response.text
    except Exception: pass
    return None

def get_last_updated_time():
    text = read_remote_file(SYNC_FILE)
    if text: return text.strip()
    if os.path.exists("last_sync.txt"):
        try:
            with open("last_sync.txt", "r") as f:
                content = f.read().strip()
                if content: return content
        except Exception: pass
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
        if csv_text: return pd.read_csv(StringIO(csv_text))
        if os.path.exists(INTRADAY_FILE): return pd.read_csv(INTRADAY_FILE)
    except Exception: pass
    return pd.DataFrame()

df_live = load_intraday_data()
is_live_active = False
live_advances = 0
live_declines = 0
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
            status.update(label="✅ Workflow started successfully.", state="complete")
            time.sleep(1)
            return True
        else:
            status.update(label="❌ Failed to trigger.", state="error")
            return False

@st.cache_data(ttl=300, show_spinner=False)
def load_agg_data():
    csv_text = read_remote_file(HISTORICAL_FILE)
    df = None
    if csv_text:
        df = pd.read_csv(StringIO(csv_text))
    elif os.path.exists(HISTORICAL_FILE):
        df = pd.read_csv(HISTORICAL_FILE)
    
    if df is not None and not df.empty and 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        if 'Advances' in df.columns and 'Declines' in df.columns:
            df['Net_Adv'] = df['Advances'] - df['Declines']
            df['AD_Line'] = df['Net_Adv'].cumsum()
            df['EMA_19_NetAdv'] = df['Net_Adv'].ewm(span=19, adjust=False, min_periods=19).mean()
            df['EMA_39_NetAdv'] = df['Net_Adv'].ewm(span=39, adjust=False, min_periods=39).mean()
            df['McClellan_Osc'] = df['EMA_19_NetAdv'] - df['EMA_39_NetAdv']
            ad_ratio = df['Advances'] / df['Declines'].replace(0, 1) 
            vol_ratio = df['Total_Up_Volume'] / df['Total_Down_Volume'].replace(0, 1)
            df['TRIN'] = ad_ratio / vol_ratio
        return df
    return pd.DataFrame()

df_agg = load_agg_data()
if df_agg.empty:
    st.error(f"Data file '{HISTORICAL_FILE}' not found. Run EOD script.")
    st.stop()

# --- DATE STEPPER LOGIC ---
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

# --- HEADER ROW (ADJUSTED COLUMNS & NOWRAP) ---
h1, h_space, h2, h3 = st.columns([4.2, 0.2, 2.5, 1.4])
with h1:
    st.markdown("<h2 style='margin-top: 5px; margin-bottom: 0px; font-weight: 900; color: #0f172a; white-space: nowrap;'>🎯 MARKET BREADTH DASHBOARD</h2>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 15px; color: #64748b; font-weight: 700; margin-top: 2px;'>Universe: Broad Market Constituent Stocks</div>", unsafe_allow_html=True)
with h2:
    st.write("")
    n1, n2, n3 = st.columns([1, 1.5, 1])
    with n1: st.button("◀ Prev", on_click=step_prev_day, use_container_width=True)
    with n2:
        selected_date = st.date_input("Date", value=st.session_state.analysis_date, min_value=min_date, max_value=max_date, label_visibility="collapsed")
        st.session_state.analysis_date = pd.to_datetime(selected_date)
    with n3: st.button("Next ▶", on_click=step_next_day, use_container_width=True)
with h3:
    st.markdown(f"<div style='text-align: right; font-size: 13px; font-weight: 700; color: #64748b;'>Sync: {last_sync_display}</div>", unsafe_allow_html=True)
    if st.button("⚡ Live Intraday Sync", use_container_width=True):
        if trigger_github_action("intraday_update.yml", "Intraday Sync"):
            st.session_state.sync_in_progress = True
            st.session_state.sync_start_time = time.time()
            st.session_state.pre_sync_time = last_sync_display
            st.rerun()

if st.session_state.sync_in_progress:
    elapsed_time = time.time() - st.session_state.sync_start_time
    if elapsed_time < 420:
        st.warning(f"⏳ **Background Sync in Progress:** Fetching new market data. Auto-refreshing soon... (Elapsed: {int(elapsed_time)}s)", icon="🤖")
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
        st.error("Sync timed out. Please try clicking Live Intraday Sync again.")

st.markdown("<hr style='margin: 10px 0px 20px 0px;'>", unsafe_allow_html=True)

df_filtered = df_agg[df_agg['Date'] <= st.session_state.analysis_date].copy()
if df_filtered.empty:
    st.warning("No trading data available on or before this date.")
    st.stop()

latest = df_filtered.iloc[-1]
prev = df_filtered.iloc[-2] if len(df_filtered) > 1 else latest

actual_date_str = latest['Date'].strftime('%d %b %Y')
advances = int(latest.get('Advances', 0))
declines = int(latest.get('Declines', 0))

if is_live_active and st.session_state.analysis_date == max_date:
    advances = live_advances
    declines = live_declines
    actual_date_str = f"{datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30))).strftime('%d %b %Y')} (⚡ LIVE)"

# --- QUANT CALCULATIONS ---
mid50 = float(latest.get('Mid_Pct_50_EMA', 0)) if pd.notna(latest.get('Mid_Pct_50_EMA')) else 0
small50 = float(latest.get('Small_Pct_50_EMA', 0)) if pd.notna(latest.get('Small_Pct_50_EMA')) else 0
large50 = float(latest.get('Large_Pct_50_EMA', 0)) if pd.notna(latest.get('Large_Pct_50_EMA')) else 0
micro50 = float(latest.get('Micro_Pct_50_EMA', 0)) if pd.notna(latest.get('Micro_Pct_50_EMA')) else 0

regime_score = int(round((0.50 * mid50) + (0.30 * small50) + (0.20 * large50)))

pct_20 = float(latest.get('Pct_Above_20_EMA', 0)) if pd.notna(latest.get('Pct_Above_20_EMA')) else 0
pct_50 = float(latest.get('Pct_Above_50_EMA', 0)) if pd.notna(latest.get('Pct_Above_50_EMA')) else 0
pct_200 = float(latest.get('Pct_Above_200_EMA', 0)) if pd.notna(latest.get('Pct_Above_200_EMA')) else 0
net_hl = float(latest.get('Net_52W_High_Low', 0)) if pd.notna(latest.get('Net_52W_High_Low')) else 0
up4 = float(latest.get('Rolling_3D_Up_4', 0)) if pd.notna(latest.get('Rolling_3D_Up_4')) else 0
down4 = float(latest.get('Rolling_3D_Down_4', 0)) if pd.notna(latest.get('Rolling_3D_Down_4')) else 0
mcclellan = float(latest.get('McClellan_Osc', 0)) if pd.notna(latest.get('McClellan_Osc')) else 0
trin = float(latest.get('TRIN', 1.0)) if pd.notna(latest.get('TRIN')) else 1.0

t3_wins = float(latest.get('T3_Wins', 0)) if pd.notna(latest.get('T3_Wins')) else 0
t3_breaks = float(latest.get('T3_Breakouts', 0)) if pd.notna(latest.get('T3_Breakouts')) else 0
ftr = (t3_wins / t3_breaks * 100) if t3_breaks > 0 else 50.0

up_vol = float(latest.get('Total_Up_Volume', 0))
dn_vol = float(latest.get('Total_Down_Volume', 0))
tot_vol = up_vol + dn_vol
up_vol_pct = (up_vol / tot_vol * 100) if tot_vol > 0 else 50.0

# Signals
brake_signals = []
if (pct_20 - pct_50) < -8: brake_signals.append(f"Short-Term Breadth Breakdown: % above 20 EMA ({pct_20:.1f}%) is far below % above 50 EMA ({pct_50:.1f}%)")
if large50 > 65 and micro50 < 45: brake_signals.append(f"Liquidity Divergence: Top 100 Liq are holding up ({large50:.0f}%) while Micro Liq break down ({micro50:.0f}%)")
if ftr < 40: brake_signals.append(f"Breakouts Failing: T+3 win rate dropped to {ftr:.1f}%")
if net_hl < 0 and regime_score > 50: brake_signals.append(f"New Lows Expanding: More stocks hitting 52-week lows than highs despite market uptrend")
if mcclellan < 0 and regime_score > 60: brake_signals.append(f"Momentum Divergence: McClellan Oscillator is negative while long-term trend remains up")
if down4 / (up4 + 1) > 1.8: brake_signals.append(f"Aggressive Selling: Down 4% days are vastly outnumbering Up 4% days")

bottom_signals = []
if up4 >= 500: bottom_signals.append(f"🚀 Breadth Thrust: Massive surge of {up4:.0f} stocks moving up >4% in 3 days")
if pct_20 < 12.0: bottom_signals.append(f"🌊 Washout: Extreme oversold levels, only {pct_20:.1f}% of stocks are above their 20 EMA")
if up_vol_pct >= 88.0: bottom_signals.append(f"🔥 Ignition Volume: {up_vol_pct:.1f}% of all volume flowed into advancing stocks today")
if trin >= 2.0: bottom_signals.append(f"🩸 Capitulation: TRIN spiked to {trin:.2f}, showing extreme panic selling")
if net_hl < -250: bottom_signals.append(f"⚠️ Extreme Fear: Huge surge in new 52-week lows ({net_hl:.0f})")

# Lookback filter for all charts
t_sel1, t_sel2 = st.columns([4, 1.5])
with t_sel2:
    timeframe = st.radio("Chart Lookback Window:", ["3M", "6M", "1Y", "3Y", "Max"], horizontal=True, index=2)
tf_map = {"3M": 63, "6M": 126, "1Y": 252, "3Y": 756, "Max": len(df_filtered)}
df_view = df_filtered.tail(tf_map.get(timeframe, 252))
last_date = df_view['Date'].iloc[-1]

# ==============================================================================
# PANEL 1: CURRENT MARKET ENVIRONMENT (TREND & HEALTH)
# ==============================================================================
st.markdown("<div class='panel-header'>📊 PANEL 1: CURRENT MARKET TREND & HEALTH</div>", unsafe_allow_html=True)

p1_c1, p1_c2, p1_c3, p1_c4 = st.columns(4)

with p1_c1:
    with st.container(border=True):
        st.markdown("<div class='card-title'>Market Trend Score</div>", unsafe_allow_html=True)
        r_col = "#16a34a" if regime_score >= 60 else ("#eab308" if regime_score >= 40 else "#dc2626")
        st.markdown(f"<div class='metric-value' style='color:{r_col};'>{regime_score} <span style='font-size:16px; color:#64748b;'>/ 100</span></div>", unsafe_allow_html=True)
        env_state = "<span style='color:#16a34a; font-weight:900;'>🟢 BULLISH (STRONG UPTREND)</span>" if regime_score >= 60 else ("<span style='color:#eab308; font-weight:900;'>🟡 CAUTIOUS (CHOP)</span>" if regime_score >= 40 else "<span style='color:#dc2626; font-weight:900;'>🔴 BEARISH (DOWNTREND)</span>")
        st.markdown(f"<div class='metric-sub'>Status: {env_state}</div>", unsafe_allow_html=True)

with p1_c2:
    with st.container(border=True):
        st.markdown("<div class='card-title'>Long-Term Breadth (% > 200 EMA)</div>", unsafe_allow_html=True)
        m_col = "#16a34a" if pct_200 >= 50 else "#dc2626"
        st.markdown(f"<div class='metric-value' style='color:{m_col};'>{pct_200:.1f}%</div>", unsafe_allow_html=True)
        breadth_state = "<span style='color:#16a34a; font-weight:900;'>🟢 BULLISH (>50%)</span>" if pct_200>=50 else "<span style='color:#dc2626; font-weight:900;'>🔴 BEARISH (<50%)</span>"
        st.markdown(f"<div class='metric-sub'>Market State: {breadth_state}</div>", unsafe_allow_html=True)

with p1_c3:
    with st.container(border=True):
        st.markdown("<div class='card-title'>Adv / Dec Ratio (Broad Market)</div>", unsafe_allow_html=True)
        ad_ratio = (advances / declines) if declines > 0 else advances
        a_col = "#16a34a" if advances >= declines else "#dc2626"
        st.markdown(f"<div class='metric-value' style='color:{a_col};'>{advances} : {declines}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-sub'>A/D Ratio: <span style='font-weight:900; color:{a_col}'>{ad_ratio:.2f}</span></div>", unsafe_allow_html=True)

with p1_c4:
    with st.container(border=True):
        st.markdown("<div class='card-title'>Up-Volume % (Liquidity Flow)</div>", unsafe_allow_html=True)
        u_col = "#16a34a" if up_vol_pct >= 50 else "#dc2626"
        st.markdown(f"<div class='metric-value' style='color:{u_col};'>{up_vol_pct:.1f}%</div>", unsafe_allow_html=True)
        vol_state = "<span style='color:#16a34a; font-weight:900;'>🟢 BULLISH (Buyers Control)</span>" if up_vol_pct >= 50 else "<span style='color:#dc2626; font-weight:900;'>🔴 BEARISH (Sellers Control)</span>"
        st.markdown(f"<div class='metric-sub'>{vol_state}</div>", unsafe_allow_html=True)

# --- BIG FULL WIDTH CHARTS FOR PANEL 1 ---
with st.container(border=True):
    st.markdown("<div class='card-title' style='font-size: 18px;'>📈 Trend Breadth (Stocks above Moving Averages)</div>", unsafe_allow_html=True)
    st.markdown("<div class='chart-explainer'>💡 <b>How to read:</b> Shows what % of the market is in short (20), medium (50), and long-term (200) uptrends. Bull markets stay above 50%.</div>", unsafe_allow_html=True)
    fig_ema = go.Figure()
    
    val_200 = df_view['Pct_Above_200_EMA'].iloc[-1]
    val_50 = df_view['Pct_Above_50_EMA'].iloc[-1]
    val_20 = df_view['Pct_Above_20_EMA'].iloc[-1]

    fig_ema.add_trace(go.Scatter(x=df_view['Date'], y=df_view['Pct_Above_200_EMA'], name='% > 200 EMA (Long Term)', line=dict(color='#22c55e', width=3.5)))
    fig_ema.add_trace(go.Scatter(x=df_view['Date'], y=df_view['Pct_Above_50_EMA'], name='% > 50 EMA (Medium Term)', line=dict(color='#f97316', width=3)))
    fig_ema.add_trace(go.Scatter(x=df_view['Date'], y=df_view['Pct_Above_20_EMA'], name='% > 20 EMA (Short Term)', line=dict(color='#06b6d4', width=2.5)))
    fig_ema.add_hline(y=50, line_dash="dash", line_color="#94a3b8", line_width=2)
    
    # Adding visible latest reading annotations
    fig_ema.add_annotation(x=last_date, y=val_200, text=f"<b>{val_200:.1f}%</b>", showarrow=False, xanchor='left', xshift=8, font=dict(color='#22c55e', size=14))
    fig_ema.add_annotation(x=last_date, y=val_50, text=f"<b>{val_50:.1f}%</b>", showarrow=False, xanchor='left', xshift=8, font=dict(color='#f97316', size=14))
    fig_ema.add_annotation(x=last_date, y=val_20, text=f"<b>{val_20:.1f}%</b>", showarrow=False, xanchor='left', xshift=8, font=dict(color='#06b6d4', size=14))

    fig_ema.update_layout(height=360, font=dict(size=14), margin=dict(l=10, r=50, t=10, b=10), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), plot_bgcolor="white", paper_bgcolor="white")
    fig_ema.update_xaxes(showgrid=False)
    fig_ema.update_yaxes(range=[0, 100], showgrid=True, gridcolor='#f1f5f9')
    st.plotly_chart(fig_ema, use_container_width=True)

with st.container(border=True):
    st.markdown("<div class='card-title' style='font-size: 18px;'>📊 Cumulative Advance-Decline (A/D) Line</div>", unsafe_allow_html=True)
    st.markdown("<div class='chart-explainer'>💡 <b>How to read:</b> The 'heartbeat' of the market. If the index hits new highs but this line falls, the rally is fake (narrowing) and a correction is coming.</div>", unsafe_allow_html=True)
    fig_ad = go.Figure()
    if 'AD_Line' in df_view.columns:
        fig_ad.add_trace(go.Scatter(x=df_view['Date'], y=df_view['AD_Line'], name='A/D Line', line=dict(color='#3b82f6', width=3.5), fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)'))
    fig_ad.update_layout(height=350, font=dict(size=14), margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified", plot_bgcolor="white", paper_bgcolor="white")
    fig_ad.update_xaxes(showgrid=False)
    fig_ad.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
    st.plotly_chart(fig_ad, use_container_width=True)


# ==============================================================================
# PANEL 2: TOP REVERSAL & DISTRIBUTION WARNING SYSTEM (BRAKES)
# ==============================================================================
st.markdown("<br><br><div class='panel-header'>🚨 PANEL 2: EARLY SIGNS OF TOPPING & DISTRIBUTION</div>", unsafe_allow_html=True)

with st.container(border=True):
    st.markdown("<div class='card-title' style='font-size: 18px;'>Distribution & Exhaustion Checklist</div>", unsafe_allow_html=True)
    if brake_signals:
        st.markdown(f"<div class='status-badge badge-warn'>⚠️ {len(brake_signals)} ACTIVE WARNINGS (BRAKES ON)</div>", unsafe_allow_html=True)
        for b in brake_signals:
            st.markdown(f"<div style='font-size:15px; color:#b91c1c; font-weight:800; margin-top:8px;'>• {b}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='status-badge badge-bull'>✅ NO DISTRIBUTION DETECTED</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:15px; color:#64748b; margin-top:8px; font-weight: 600;'>Breakouts are working and participation across liquidity tiers is healthy. Keep trailing stops normal.</div>", unsafe_allow_html=True)
        
    st.markdown("<hr style='margin:15px 0px;'>", unsafe_allow_html=True)
    
    # Text Metrics for Panel 2
    mcc_col = "#16a34a" if mcclellan >= 0 else "#dc2626"
    mcc_state = "<span style='color:#16a34a; font-weight:900;'>🟢 BULLISH</span>" if mcclellan >= 0 else "<span style='color:#dc2626; font-weight:900;'>🔴 BEARISH</span>"
    ftr_state = "<span style='color:#16a34a; font-weight:900;'>🟢 BULLISH (Working)</span>" if ftr >= 40 else "<span style='color:#dc2626; font-weight:900;'>🔴 BEARISH (Failing)</span>"
    
    st.markdown(f"""
        <div style='font-size:15px; color:#475569;'>
            <b>McClellan Oscillator:</b> <span style='color:{mcc_col}; font-weight:900;'>{mcclellan:.1f}</span> — {mcc_state}<br><br>
            <b>Breakout Win Rate (T+3):</b> <span style='font-weight:900;'>{ftr:.1f}%</span> — {ftr_state}<br><br>
            <b>Top 100 Liq vs Micro Liq Spread:</b> <span style='font-weight:900;'>{(large50 - micro50):.1f}%</span>
        </div>
    """, unsafe_allow_html=True)

# --- BIG FULL WIDTH CHARTS FOR PANEL 2 ---
with st.container(border=True):
    st.markdown("<div class='card-title' style='font-size: 18px;'>🏢 Liquidity Divergence (% > 50 EMA)</div>", unsafe_allow_html=True)
    st.markdown("<div class='chart-explainer'>💡 <b>How to read:</b> Micro liquidity stocks usually top out first. If Top 100 Liq are high but Micro Liq plunge, institutional money is secretly selling the broader market while propping up the index.</div>", unsafe_allow_html=True)
    fig_cap = go.Figure()
    
    val_l50 = df_view.get('Large_Pct_50_EMA', pd.Series([0], dtype=float)).iloc[-1]
    val_mi50 = df_view.get('Micro_Pct_50_EMA', pd.Series([0], dtype=float)).iloc[-1]

    fig_cap.add_trace(go.Scatter(x=df_view['Date'], y=df_view.get('Large_Pct_50_EMA', pd.Series(dtype=float)), mode='lines', name='Top 100 Liq', line=dict(color='#2563eb', width=3.5)))
    fig_cap.add_trace(go.Scatter(x=df_view['Date'], y=df_view.get('Micro_Pct_50_EMA', pd.Series(dtype=float)), mode='lines', name='Micro Liq', line=dict(color='#ef4444', width=3, dash='dot')))
    fig_cap.add_hline(y=50, line_dash="dash", line_color="#cbd5e1", line_width=2)
    
    fig_cap.add_annotation(x=last_date, y=val_l50, text=f"<b>{val_l50:.1f}%</b>", showarrow=False, xanchor='left', xshift=8, font=dict(color='#2563eb', size=14))
    fig_cap.add_annotation(x=last_date, y=val_mi50, text=f"<b>{val_mi50:.1f}%</b>", showarrow=False, xanchor='left', xshift=8, font=dict(color='#ef4444', size=14))

    fig_cap.update_layout(height=350, font=dict(size=14), margin=dict(l=10, r=50, t=10, b=10), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), plot_bgcolor="white", paper_bgcolor="white")
    fig_cap.update_xaxes(showgrid=False)
    fig_cap.update_yaxes(range=[0, 100], showgrid=True, gridcolor='#f1f5f9')
    st.plotly_chart(fig_cap, use_container_width=True)

with st.container(border=True):
    st.markdown("<div class='card-title' style='font-size: 18px;'>⚡ McClellan Oscillator (Momentum)</div>", unsafe_allow_html=True)
    st.markdown("<div class='chart-explainer'>💡 <b>How to read:</b> Measures the velocity of money moving into advancing stocks. Dropping below zero warns of short-term exhaustion even if the index is up.</div>", unsafe_allow_html=True)
    fig_mcc = go.Figure()
    if 'McClellan_Osc' in df_view.columns:
        colors = ['#22c55e' if val >= 0 else '#ef4444' for val in df_view['McClellan_Osc']]
        fig_mcc.add_trace(go.Bar(x=df_view['Date'], y=df_view['McClellan_Osc'], name='McClellan Osc', marker_color=colors))
    fig_mcc.update_layout(height=350, font=dict(size=14), margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified", showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
    fig_mcc.update_xaxes(showgrid=False)
    fig_mcc.update_yaxes(showgrid=True, gridcolor='#f1f5f9', zeroline=True, zerolinecolor='#475569', zerolinewidth=2)
    st.plotly_chart(fig_mcc, use_container_width=True)


# ==============================================================================
# PANEL 3: BOTTOM REVERSAL & CAPITULATION (THE GREEN LIGHT)
# ==============================================================================
st.markdown("<br><br><div class='panel-header'>🚀 PANEL 3: EARLY SIGNS OF BOTTOMING & THRUSTS</div>", unsafe_allow_html=True)

with st.container(border=True):
    st.markdown("<div class='card-title' style='font-size: 18px;'>Bottoming Signals Checklist</div>", unsafe_allow_html=True)
    if bottom_signals:
        st.markdown(f"<div class='status-badge badge-thrust'>🚀 {len(bottom_signals)} BOTTOM SIGNALS ACTIVE</div>", unsafe_allow_html=True)
        for s in bottom_signals:
            st.markdown(f"<div style='font-size:15px; color:#1e40af; font-weight:800; margin-top:8px;'>• {s}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='status-badge badge-bull'>⚪ NORMAL TRADING CONDITIONS</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:15px; color:#64748b; margin-top:8px; font-weight: 600;'>No extreme panic, capitulation, or violent institutional buy thrusts happening right now.</div>", unsafe_allow_html=True)
        
    st.markdown("<hr style='margin:15px 0px;'>", unsafe_allow_html=True)
    
    trin_col = "#dc2626" if trin >= 2.0 else ("#16a34a" if trin <= 0.8 else "#475569")
    trin_state = "<span style='color:#dc2626; font-weight:900;'>🔴 EXTREME PANIC (Good for bottoms)</span>" if trin >= 2.0 else "<span style='color:#16a34a; font-weight:900;'>🟢 HEALTHY</span>"
    
    st.markdown(f"""
        <div style='font-size:15px; color:#475569;'>
            <b>TRIN (Arms Index):</b> <span style='color:{trin_col}; font-weight:900;'>{trin:.2f}</span> — {trin_state}<br><br>
            <b>3-Day 4%+ Thrust Movers:</b> <span style='font-weight:900; color:#16a34a;'>+{up4:.0f}</span> / <span style='font-weight:900; color:#dc2626;'>-{down4:.0f}</span><br><br>
            <b>Net 52W Highs/Lows:</b> <span style='font-weight:900; color:{"#16a34a" if net_hl>=0 else "#dc2626"};'>{net_hl:.0f}</span>
        </div>
    """, unsafe_allow_html=True)

# --- BIG FULL WIDTH CHARTS FOR PANEL 3 ---
with st.container(border=True):
    st.markdown("<div class='card-title' style='font-size: 18px;'>🚀 Momentum Thrusts & Net High-Lows</div>", unsafe_allow_html=True)
    st.markdown("<div class='chart-explainer'>💡 <b>How to read:</b> Top chart shows explosive short-term momentum (Breadth Thrusts). Spikes above the dotted line indicate violent institutional buying. Bottom shows Net 52W Highs.</div>", unsafe_allow_html=True)
    fig_thrust = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    
    # 3-Day Thrust Bars
    fig_thrust.add_trace(go.Bar(x=df_view['Date'], y=df_view['Rolling_3D_Up_4'], name='Up 4%+', marker_color='#22c55e'), row=1, col=1)
    fig_thrust.add_trace(go.Bar(x=df_view['Date'], y=-df_view['Rolling_3D_Down_4'], name='Down 4%+', marker_color='#ef4444'), row=1, col=1)
    fig_thrust.add_hline(y=500, line_dash="dash", line_color="#3b82f6", line_width=2, row=1, col=1) 
    
    # Net High-Lows
    fig_thrust.add_trace(go.Bar(x=df_view['Date'], y=df_view['Net_52W_High_Low'], name='Net 52W HL', marker_color=['#22c55e' if x>=0 else '#ef4444' for x in df_view['Net_52W_High_Low']]), row=2, col=1)
    fig_thrust.update_layout(height=380, font=dict(size=14), margin=dict(l=10, r=10, t=10, b=10), barmode='relative', showlegend=False, hovermode="x unified", plot_bgcolor="white", paper_bgcolor="white")
    fig_thrust.update_xaxes(showgrid=False)
    fig_thrust.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
    st.plotly_chart(fig_thrust, use_container_width=True)
            
with st.container(border=True):
    st.markdown("<div class='card-title' style='font-size: 18px;'>🩸 TRIN (Arms / Capitulation Index)</div>", unsafe_allow_html=True)
    st.markdown("<div class='chart-explainer'>💡 <b>How to read:</b> A TRIN reading above 2.0 means extreme panic selling is happening. Counter-intuitively, market bottoms form exactly on days with maximum fear.</div>", unsafe_allow_html=True)
    fig_trin = go.Figure()
    if 'TRIN' in df_view.columns:
        capped_trin = df_view['TRIN'].clip(upper=4.0)
        colors_trin = ['#ef4444' if val >= 2.0 else ('#22c55e' if val <= 0.8 else '#cbd5e1') for val in capped_trin]
        fig_trin.add_trace(go.Bar(x=df_view['Date'], y=capped_trin, name='TRIN', marker_color=colors_trin))
        fig_trin.add_hline(y=2.0, line_dash="dash", line_color="#ef4444", line_width=2, annotation_text="Panic Threshold (> 2.0)", annotation_position="top left", annotation_font_size=14)
        fig_trin.add_hline(y=1.0, line_dash="solid", line_color="#94a3b8", line_width=2)
    fig_trin.update_layout(height=350, font=dict(size=14), margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified", showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
    fig_trin.update_xaxes(showgrid=False)
    fig_trin.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
    st.plotly_chart(fig_trin, use_container_width=True)

# ==============================================================================
# DRILL-DOWN STOCK INSPECTOR
# ==============================================================================
st.markdown("<br><br><div class='panel-header'>🔍 BROAD MARKET DRILL-DOWN INSPECTOR</div>", unsafe_allow_html=True)

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
        st.dataframe(res, use_container_width=True, height=450)
    else:
        st.info("👆 Select one or more parameters above to screen individual stocks.")
else:
    st.info("💡 Deep dive list requires 'nse_6yr_historical.parquet' in repository.")

st.markdown("<br><hr>", unsafe_allow_html=True)
bot_col, _ = st.columns([1.5, 4])
with bot_col:
    st.markdown(f"<div style='text-align: center; font-size: 13px; color: #94a3b8; margin-bottom: 5px; font-weight: 700;'>Last Successful Database Update: {last_sync_display}</div>", unsafe_allow_html=True)
    if st.button("📅 Upgrade Database & Purge Bad Data", use_container_width=True):
        if trigger_github_action("eod_update.yml", "EOD Sync"):
            st.session_state.sync_in_progress = True
            st.session_state.sync_start_time = time.time()
            st.session_state.pre_sync_time = last_sync_display
            st.rerun()

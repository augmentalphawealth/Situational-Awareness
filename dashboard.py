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

if 'sync_in_progress' not in st.session_state:
    st.session_state.sync_in_progress = False
    st.session_state.sync_start_time = 0
    st.session_state.pre_sync_time = ""

st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 98%; }
    header { visibility: hidden; }
    div[data-testid="stVerticalBlockBorderWrapper"] { background-color: #ffffff; border-radius: 12px; padding: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.03); border: 1px solid #e2e8f0; }
    .card-title { font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 5px; }
    .metric-value { font-size: 26px; font-weight: 800; color: #0f172a; }
    .metric-sub { font-size: 12px; font-weight: 600; color: #64748b; margin-top: 2px; }
    .action-banner { background-color: #0f172a; color: #f8fafc; padding: 10px 16px; border-radius: 8px; font-size: 13px; font-weight: 700; text-align: center; margin-top: 8px; letter-spacing: 0.5px; }
    .stButton>button { border-radius: 6px; font-weight: 600; font-size: 12px; padding: 0.3rem 0.5rem; }
    div[data-testid="stDateInput"] input { padding: 0.3rem; font-size: 13px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

REPO_OWNER = "augmentalphawealth"
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

if not df_live.empty and 'Date' in df_live.columns:
    try:
        ist_offset_local = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
        today_str_local = datetime.datetime.now(ist_offset_local).strftime('%Y-%m-%d')
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
            status.update(label="✅ Workflow started successfully.", state="complete")
            time.sleep(1)
            return True
        else:
            status.update(label="❌ Failed to trigger.", state="error")
            st.error(f"GitHub API Error: {res.status_code} - {res.text}")
            return False

@st.cache_data(ttl=300, show_spinner=False)
def load_agg_data():
    csv_text = read_remote_file(HISTORICAL_FILE)
    if csv_text:
        df = pd.read_csv(StringIO(csv_text))
        if not df.empty and 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            return df.sort_values('Date').reset_index(drop=True)
    return pd.DataFrame()

df_agg = load_agg_data()
if df_agg.empty:
    st.error(f"Data file '{HISTORICAL_FILE}' not found. Run EOD script.")
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
        selected = st.date_input("Date", value=st.session_state.analysis_date, min_value=min_date, max_value=max_date, label_visibility="collapsed")
        st.session_state.analysis_date = pd.to_datetime(selected)
    with nav3: st.button("Next ▶", on_click=step_next_day, use_container_width=True)
with head_col3:
    display_sync = last_sync_display if st.session_state.analysis_date == max_date else "Historical View"
    st.markdown(f"<div style='text-align: right; font-size: 11px; font-weight: 600; color: #64748b; margin-top: 2px; margin-bottom: 3px;'>Last Sync: {display_sync}</div>", unsafe_allow_html=True)
    if st.button("⚡ Live Intraday Sync", use_container_width=True):
        if trigger_github_action("intraday_update.yml", "Intraday Sync"):
            st.session_state.sync_in_progress = True
            st.session_state.sync_start_time = time.time()
            st.session_state.pre_sync_time = last_sync_display
            st.rerun()

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

score = int(latest.get('Composite_Score', 0))
p_fast = latest.get('Pct_Above_20_EMA', 0)
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
df_10d['Date_Str'] = df_10d['Date'].dt.strftime('%d %b')

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
                <div style='font-size: 11px; color: #64748b; font-weight: 600; margin-top: 2px;'>Fast Breadth (P_Fast): {p_fast:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
    with top_c2:
        fig_m1 = go.Figure(go.Bar(
            x=df_10d['Date_Str'], y=df_10d['Composite_Score'], 
            marker_color=[get_bar_color(v) for v in df_10d['Composite_Score']],
            text=df_10d['Composite_Score'], textposition='auto',
            hovertemplate='Score: <b>%{y}</b><extra></extra>'
        ))
        fig_m1.update_layout(height=120, margin=dict(l=10, r=10, t=25, b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False, title=dict(text="10-Day Health Trend", font=dict(size=11, color="#64748b")))
        fig_m1.update_xaxes(showgrid=False, tickfont=dict(size=10, color="#94a3b8"))
        fig_m1.update_yaxes(visible=False, range=[0, 100])
        st.plotly_chart(fig_m1, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown(f"<div class='action-banner'>🎯 ACTION ZONE: {action_zone}</div>", unsafe_allow_html=True)

actual_date_str = latest['Date'].strftime('%d %b %Y')
if is_live_active and st.session_state.analysis_date == max_date:
    actual_date_str = f"{actual_date_str} <span style='color:#eab308; font-weight:800;'>(⚡ LIVE INTRADAY A/D)</span>"

st.markdown(f"<p style='color: #475569; font-size: 13px; font-weight: 600; margin-top: 15px;'>Market Breadth Status for: <span style='color:#0f172a;'>{actual_date_str}</span></p>", unsafe_allow_html=True)
hero_col1, hero_col2, hero_col3 = st.columns([1.1, 1.5, 1.4])

advances = int(latest.get('Advances', 0))
declines = int(latest.get('Declines', 0))
total_univ = int(latest.get('Total_Universe', 2400))
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
                <p style='color: {adv_change_color}; font-size: 12px; font-weight: 700; margin-top: 4px; margin-bottom: 0px;'>{adv_change_str} vs Yesterday EOD (Univ {total_univ})</p>
            </div>
        """, unsafe_allow_html=True)

with hero_col2:
    with st.container(border=True):
        st.markdown("<div class='card-title' style='margin-top: 5px; margin-left: 10px;'>TACTICAL COMMAND CENTER</div>", unsafe_allow_html=True)
        st.markdown(f"""
            <div style='padding: 0px 10px; font-size: 13px;'>
                <div style='margin-bottom: 6px;'><b>🎯 Target Asset:</b> <span style='color: #334155;'>{tacs['asset']}</span></div>
                <div style='margin-bottom: 6px;'><b>⚖️ Position Sizing:</b> <span style='color: #334155;'>{tacs['sizing']}</span></div>
                <div style='margin-bottom: 6px;'><b>🛡️ Risk / Stop-Loss:</b> <span style='color: #334155;'>{tacs['risk']}</span></div>
                <div style='margin-bottom: 6px;'><b>💰 Profit Strategy:</b> <span style='color: #334155;'>{tacs['profit']}</span></div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<hr style='margin: 8px 0px;'>", unsafe_allow_html=True)
        st.markdown(f"<div class='card-title' style='margin-left: 10px;'>REGIME CONTEXT (EOD SCORE ALIGNED)</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 11px; margin-bottom: 3px; padding-left: 10px;'>• Follow-Through Win Rate: <b>{ft_rate:.1f}%</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 11px; margin-bottom: 3px; padding-left: 10px;'>• 200 EMA Breadth: <b>{latest.get('Pct_Above_200_EMA', 0):.1f}%</b></div>", unsafe_allow_html=True)

with hero_col3:
    with st.container(border=True):
        st.markdown("<div class='card-title' style='text-align: center; margin-top: 5px;'>CAPITAL FLOW (TURNOVER SPLIT)</div>", unsafe_allow_html=True)
        up_vol = float(latest.get('Total_Up_Volume', 0))
        dn_vol = float(latest.get('Total_Down_Volume', 0))
        tot_vol = up_vol + dn_vol
        up_vol_pct = (up_vol / tot_vol * 100) if tot_vol > 0 else 0
        
        fig_vol = go.Figure(data=[go.Pie(
            labels=['Advancing Turnover (₹)', 'Declining Turnover (₹)'], values=[up_vol, dn_vol], hole=0.72,
            marker=dict(colors=['#22c55e', '#ef4444'], line=dict(color='#ffffff', width=2)), textinfo='none', hovertemplate='%{label}<br>₹%{value:,.0f} (%{percent})<extra></extra>'
        )])
        fig_vol.update_layout(
            height=195, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
            annotations=[dict(text=f"<span style='font-size:24px; font-weight:800; color:#0f172a;'>{up_vol_pct:.1f}%</span><br><span style='font-size:10px; font-weight:700; color:#64748b;'>UP-VOLUME</span>", x=0.5, y=0.5, showarrow=False)]
        )
        st.plotly_chart(fig_vol, use_container_width=True, config={'displayModeBar': False})

m2, m3 = st.columns(2)
with m2:
    vol_ratio = latest['Volume_Ratio'] if pd.notna(latest['Volume_Ratio']) else 0
    v_col = "#22c55e" if vol_ratio > 1.0 else "#ef4444"
    with st.container(border=True):
        st.markdown(f"<div style='text-align: center;'><div class='card-title' style='margin-top: 5px;'>Volume Breadth Ratio</div><div class='metric-value' style='color: {v_col};'>{vol_ratio:.2f}</div><div class='metric-sub'>Liquidity Flow: Up-Turnover vs Down-Turnover (₹)</div></div>", unsafe_allow_html=True)
        fig_m2 = go.Figure(go.Bar(x=df_10d['Date_Str'], y=df_10d['Volume_Ratio'], marker_color=['#22c55e' if v >= 1.0 else '#ef4444' for v in df_10d['Volume_Ratio']], hovertemplate='Ratio: %{y:.2f}<extra></extra>'))
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
        fig_m3 = go.Figure(go.Bar(x=df_10d['Date_Str'], y=df_10d['Net_52W_High_Low'], marker_color=['#22c55e' if v >= 0 else '#ef4444' for v in df_10d['Net_52W_High_Low']], hovertemplate='Net HL: %{y:.0f}<extra></extra>'))
        fig_m3.add_hline(y=0, line_dash="solid", line_color="#cbd5e1")
        fig_m3.update_layout(height=100, margin=dict(l=5, r=5, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        fig_m3.update_xaxes(showgrid=False, tickfont=dict(size=10, color="#94a3b8"))
        fig_m3.update_yaxes(visible=False)
        st.plotly_chart(fig_m3, use_container_width=True, config={'displayModeBar': False})

st.markdown("<br>", unsafe_allow_html=True)
tf_col1, tf_col2 = st.columns([3, 1])
with tf_col1: st.markdown("### 📊 Historical Market Analytics")
with tf_col2: timeframe = st.radio("Chart Horizon:", ["1 Month", "3 Months", "6 Months", "1 Year", "3 Years", "6 Years"], horizontal=True, index=2)

days_map = {"1 Month": 21, "3 Months": 63, "6 Months": 126, "1 Year": 252, "3 Years": 756, "6 Years": len(df_filtered)}
plot_df = df_filtered.tail(days_map.get(timeframe, 126))

with st.container(border=True):
    st.markdown(f"<div class='card-title' style='margin-left: 10px; margin-top: 10px;'>UNIVERSE EMA BREADTH TRENDS &nbsp;|&nbsp; LATEST: <span style='color:#22c55e;'>200 EMA ({plot_df['Pct_Above_200_EMA'].iloc[-1]:.1f}%)</span> • <span style='color:#a855f7;'>50 EMA ({plot_df['Pct_Above_50_EMA'].iloc[-1]:.1f}%)</span> • <span style='color:#3b82f6;'>20 EMA ({plot_df['Pct_Above_20_EMA'].iloc[-1]:.1f}%)</span></div>", unsafe_allow_html=True)
    fig_ema = go.Figure()
    fig_ema.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Pct_Above_200_EMA'], mode='lines', name='% > 200 EMA', line=dict(color='#22c55e', width=2)))
    fig_ema.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Pct_Above_50_EMA'], mode='lines', name='% > 50 EMA', line=dict(color='#a855f7', width=2)))
    fig_ema.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Pct_Above_20_EMA'], mode='lines', name='% > 20 EMA', line=dict(color='#3b82f6', width=2)))
    fig_ema.add_hline(y=50, line_dash="dash", line_color="#94a3b8", opacity=0.7)
    fig_ema.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    fig_ema.update_yaxes(range=[0, 100], gridcolor='#f1f5f9', title="% Stocks")
    fig_ema.update_xaxes(showgrid=False)
    st.plotly_chart(fig_ema, use_container_width=True)

with st.container(border=True):
    st.markdown("<div class='card-title' style='margin-left: 10px; margin-top: 10px;'>SEGMENTED LIQUIDITY FLOW (45-DAY ROLLING TURNOVER RANK)</div>", unsafe_allow_html=True)
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

out1, out2 = st.columns(2)
with out1:
    with st.container(border=True):
        st.markdown(f"<div class='card-title' style='margin-left: 10px; margin-top: 10px;'>ROLLING 1-MONTH 25% MOVERS &nbsp;|&nbsp; LATEST: <span style='color:#16a34a;'>{int(plot_df['Up_25_1M_Count'].iloc[-1])} UP</span> / <span style='color:#dc2626;'>{int(plot_df['Down_25_1M_Count'].iloc[-1])} DOWN</span></div>", unsafe_allow_html=True)
        fig_outliers = go.Figure()
        fig_outliers.add_trace(go.Bar(x=plot_df['Date'], y=plot_df['Up_25_1M_Count'], name='Up 25%+ in 1M', marker_color='#22c55e'))
        fig_outliers.add_trace(go.Bar(x=plot_df['Date'], y=-plot_df['Down_25_1M_Count'], name='Down 25%+ in 1M', marker_color='#ef4444', customdata=plot_df['Down_25_1M_Count'], hovertemplate='Down 25%: <b>%{customdata:.0f}</b><extra></extra>'))
        fig_outliers.update_layout(barmode='relative', height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False, hovermode="x unified")
        fig_outliers.update_yaxes(gridcolor='#f1f5f9', zerolinecolor='black')
        fig_outliers.update_xaxes(showgrid=False)
        st.plotly_chart(fig_outliers, use_container_width=True)

with out2:
    with st.container(border=True):
        st.markdown(f"<div class='card-title' style='margin-left: 10px; margin-top: 10px;'>3-DAY ROLLING 4% THRUST MOVERS &nbsp;|&nbsp; LATEST: <span style='color:#16a34a;'>{int(plot_df['Rolling_3D_Up_4'].iloc[-1])} UP</span> / <span style='color:#dc2626;'>{int(plot_df['Rolling_3D_Down_4'].iloc[-1])} DOWN</span></div>", unsafe_allow_html=True)
        fig_movers = go.Figure()
        fig_movers.add_trace(go.Bar(x=plot_df['Date'], y=plot_df['Rolling_3D_Up_4'], name='Up 4%+', marker_color='#22c55e'))
        fig_movers.add_trace(go.Bar(x=plot_df['Date'], y=-plot_df['Rolling_3D_Down_4'], name='Down 4%+', marker_color='#ef4444', customdata=plot_df['Rolling_3D_Down_4'], hovertemplate='Down 4%+: <b>%{customdata:.0f}</b><extra></extra>'))
        fig_movers.update_layout(barmode='relative', height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False, hovermode="x unified")
        fig_movers.update_yaxes(gridcolor='#f1f5f9', zerolinecolor='black')
        fig_movers.update_xaxes(showgrid=False)
        st.plotly_chart(fig_movers, use_container_width=True)

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
        
        # Guard 1: Aligned to 200 days 
        subset['52W_High'] = subset.groupby('Symbol')['High'].transform(lambda x: x.rolling(window=252, min_periods=200).max())
        subset['52W_Low'] = subset.groupby('Symbol')['Low'].transform(lambda x: x.rolling(window=252, min_periods=200).min())
        
        closest_date = subset[subset['Date'] <= target_date]['Date'].max()
        day_data = subset[subset['Date'] == closest_date].copy()
        
        # Guard 2: Traded today logic
        day_data['Traded_Today'] = day_data['Volume'] > 0
        
        day_data['Daily_%_Change'] = day_data['Daily_%_Change'].round(2)
        day_data['1M_%_Change'] = day_data['1M_%_Change'].round(2)
        return day_data
    except Exception:
        return pd.DataFrame()

drill_col, _ = st.columns([1.5, 1])
with drill_col:
    param_choices = st.multiselect("Select Parameters to Filter (Combines with AND):", [
        "Advances (Stocks in Green)", "Declines (Stocks in Red)", "Stocks > 20 EMA", "Stocks > 50 EMA", "Stocks > 200 EMA",
        "Up 4% or more Today", "Down 4% or more Today", "1-Month 25% Winners", "1-Month 25% Losers", "New 52-Week Highs", "New 52-Week Lows"
    ])

drill_data = get_drilldown_data(latest['Date'])

if not drill_data.empty:
    res = drill_data.copy()
    if param_choices:
        for param in param_choices:
            if param == "Advances (Stocks in Green)": res = res[(res['Daily_%_Change'] > 0) & res['Traded_Today']]
            elif param == "Declines (Stocks in Red)": res = res[(res['Daily_%_Change'] < 0) & res['Traded_Today']]
            elif param == "Stocks > 20 EMA": res = res[res['Close'] > res['EMA_20']]
            elif param == "Stocks > 50 EMA": res = res[res['Close'] > res['EMA_50']]
            elif param == "Stocks > 200 EMA": res = res[res['Close'] > res['EMA_200']]
            elif param == "Up 4% or more Today": res = res[(res['Daily_%_Change'] >= 4.0) & res['Traded_Today']]
            elif param == "Down 4% or more Today": res = res[(res['Daily_%_Change'] <= -4.0) & res['Traded_Today']]
            elif param == "1-Month 25% Winners": res = res[res['1M_%_Change'] >= 25.0]
            elif param == "1-Month 25% Losers": res = res[res['1M_%_Change'] <= -25.0]
            elif param == "New 52-Week Highs": res = res[(res['Close'] >= res['52W_High']) & res['Traded_Today']]
            elif param == "New 52-Week Lows": res = res[(res['Close'] <= res['52W_Low']) & res['Traded_Today']]

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
    if st.button("📅 Run End of Day Analytics", use_container_width=True):
        if trigger_github_action("eod_update.yml", "EOD Sync"):
            st.session_state.sync_in_progress = True
            st.session_state.sync_start_time = time.time()
            st.session_state.pre_sync_time = last_sync_display
            st.rerun()

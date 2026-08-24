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
    </style>
""", unsafe_allow_html=True)

REPO_OWNER = "augmentalphawealth"
REPO_NAME = "Situational-Awareness"
BRANCH = "main"

HISTORICAL_FILE = "historical_breadth_regime_6yr.csv"
PERMANENT_STOCK_FILE = "trailing_cache.parquet"
TEMP_MARKET_FILE = "intraday_tmp/intraday_market_metrics.json"
TEMP_STOCK_FILE = "intraday_tmp/intraday_stock_metrics.parquet"
TEMP_STATUS_FILE = "intraday_tmp/intraday_status.json"


def safe_int(value):
    try:
        if pd.isna(value) or value is None:
            return 0
        return int(float(value))
    except Exception:
        return 0


def remote_url(path, sha):
    return f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{sha}/{path}"


@st.cache_data(ttl=30, show_spinner=False)
def get_latest_commit_sha():
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/commits/{BRANCH}"
    try:
        response = requests.get(url, timeout=10, headers={"Cache-Control": "no-cache"})
        if response.status_code == 200:
            return response.json().get("sha", BRANCH)
    except Exception:
        pass
    return BRANCH


def read_remote(path, binary=False):
    sha = get_latest_commit_sha()
    try:
        response = requests.get(
            remote_url(path, sha),
            timeout=45,
            headers={"Cache-Control": "no-cache"},
        )
        if response.status_code == 200:
            return response.content if binary else response.content.decode("utf-8")
    except Exception:
        pass
    return None


@st.cache_data(ttl=30, show_spinner=False)
def load_aggregate_data(commit_sha):
    content = read_remote(HISTORICAL_FILE)
    if content is None:
        return pd.DataFrame()
    try:
        frame = pd.read_csv(StringIO(content))
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
        return frame.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30, show_spinner=False)
def load_temporary_market_data(commit_sha):
    content = read_remote(TEMP_MARKET_FILE)
    if content is None:
        return {}
    try:
        return json.loads(content)
    except Exception:
        return {}


@st.cache_data(ttl=30, show_spinner=False)
def load_stock_data(commit_sha):
    content = read_remote(TEMP_STOCK_FILE, binary=True)
    if content is not None:
        try:
            return pd.read_parquet(BytesIO(content))
        except Exception:
            pass

    content = read_remote(PERMANENT_STOCK_FILE, binary=True)
    if content is not None:
        try:
            return pd.read_parquet(BytesIO(content))
        except Exception:
            pass
    return pd.DataFrame()


def read_sync_time(commit_sha):
    return "Temporary Intraday Snapshot"


def trigger_github_action(workflow_name, button_label):
    token = st.secrets.get("GITHUB_TOKEN", None)
    if not token:
        st.error("GitHub Token missing in Streamlit Secrets!")
        return False
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/workflows/{workflow_name}/dispatches"
    headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}"}
    with st.status(f"🚀 Triggering {button_label}...", expanded=False) as status:
        response = requests.post(url, headers=headers, json={"ref": BRANCH}, timeout=20)
        if response.status_code == 204:
            status.update(label="✅ Workflow dispatched successfully to GitHub.", state="complete")
            return True
        status.update(label="❌ Failed to trigger.", state="error")
        st.error(f"GitHub API Error [{response.status_code}]: {response.text}")
        return False


commit_sha = get_latest_commit_sha()
df_agg = load_aggregate_data(commit_sha)
market = load_temporary_market_data(commit_sha)
df_stock = load_stock_data(commit_sha)

if df_agg.empty:
    st.error(f"Data file '{HISTORICAL_FILE}' could not be loaded.")
    st.stop()

ist = ZoneInfo("Asia/Kolkata")
today = pd.Timestamp(datetime.datetime.now(ist).date()).normalize()
temp_date = pd.to_datetime(market.get("Date"), errors="coerce") if market.get("Date") else pd.NaT
live_available = bool(market) and pd.notna(temp_date) and temp_date.normalize() == today

permanent_dates = sorted(df_agg["Date"].dropna().unique())
min_date = pd.to_datetime(permanent_dates[0])
permanent_max_date = pd.to_datetime(permanent_dates[-1])

# Today is added only as a UI date; it is not inserted into permanent EOD history.
max_date = today if live_available and today > permanent_max_date else permanent_max_date
available_dates = np.array(permanent_dates + ([np.datetime64(today)] if live_available and today > permanent_max_date else []), dtype="datetime64[ns]")

if "last_max_date" not in st.session_state or st.session_state.last_max_date != max_date:
    st.session_state.analysis_date = max_date
    st.session_state.last_max_date = max_date
if "analysis_date" not in st.session_state:
    st.session_state.analysis_date = max_date


def step_prev_day():
    index = np.where(available_dates == np.datetime64(st.session_state.analysis_date))[0]
    if len(index) and index[0] > 0:
        st.session_state.analysis_date = pd.to_datetime(available_dates[index[0] - 1])


def step_next_day():
    index = np.where(available_dates == np.datetime64(st.session_state.analysis_date))[0]
    if len(index) and index[0] < len(available_dates) - 1:
        st.session_state.analysis_date = pd.to_datetime(available_dates[index[0] + 1])


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
        selected_date = st.date_input("Date", value=st.session_state.analysis_date, min_value=min_date, max_value=max_date, format="DD/MM/YYYY", label_visibility="collapsed")
        if selected_date != st.session_state.analysis_date.date():
            st.session_state.analysis_date = pd.to_datetime(selected_date)
            st.rerun()
    with nav3:
        st.button("Next ▶", on_click=step_next_day, use_container_width=True)
with head_col3:
    selected_is_today = st.session_state.analysis_date == today and live_available
    display_sync = "Live intraday" if selected_is_today else "Historical View"
    st.markdown(f"<div style='text-align: right; font-size: 11px; font-weight: 600; color: #64748b; margin-top: 2px; margin-bottom: 3px;'>Last Sync: {display_sync}</div>", unsafe_allow_html=True)
    if st.button("⚡ Trigger Intraday Sync", use_container_width=True):
        if trigger_github_action("intraday_update_temporary.yml", "Intraday Sync"):
            st.cache_data.clear()
            st.rerun()

selected_date = pd.Timestamp(st.session_state.analysis_date).normalize()
if selected_date == today and live_available:
    current = pd.DataFrame([market])
    latest = current.iloc[0]
    previous_rows = df_agg[df_agg["Date"] < today]
    prev = previous_rows.iloc[-1] if not previous_rows.empty else latest
else:
    selected_rows = df_agg[df_agg["Date"] <= selected_date]
    if selected_rows.empty:
        st.warning("No trading data available on or before this date.")
        st.stop()
    latest = selected_rows.iloc[-1]
    prev = selected_rows.iloc[-2] if len(selected_rows) > 1 else latest

score = safe_int(latest.get("Composite_Score", 0))
p_fast = float(latest.get("Pct_Above_20_EMA", 0) or 0)
p_mid = float(latest.get("Pct_Above_50_EMA", 0) or 0)
t3_breakouts = float(latest.get("T3_Breakouts", 0) or 0)
t3_wins = float(latest.get("T3_Wins", 0) or 0)
ft_rate = (t3_wins / t3_breakouts * 100) if t3_breakouts > 0 else 0

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

df_filtered = df_agg[df_agg["Date"] <= selected_date].copy()
if selected_date == today and live_available:
    df_filtered = pd.concat([df_filtered, current], ignore_index=True, sort=False)

df_10d = df_filtered.tail(10).copy()
df_10d["Date"] = pd.to_datetime(
    df_10d["Date"],
    errors="coerce"
).dt.normalize()
df_10d = df_10d.dropna(subset=["Date"])
df_10d["Date_Str"] = df_10d["Date"].dt.strftime("%d %b")


def get_bar_color(value):
    if value >= 71: return "#22c55e"
    if value >= 51: return "#84cc16"
    if value >= 31: return "#f97316"
    if value >= 11: return "#ef4444"
    return "#991b1b"

with st.container(border=True):
    top_c1, top_c2 = st.columns([1.2, 2.8])
    with top_c1:
        st.markdown(f"<div style='text-align: center; padding-top: 10px;'><div class='card-title' style='font-size: 12px;'>MOMENTUM HEALTH SCORE ({'LIVE' if selected_date == today and live_available else 'EOD'})</div><div class='metric-value' style='font-size: 44px; color: {score_color};'>{score} <span style='font-size: 18px; color: #94a3b8;'>/ 100</span></div><div style='font-size: 11px; color: #64748b; font-weight: 600; margin-top: 2px;'>Fast Breadth (P_Fast): {p_fast:.1f}%</div></div>", unsafe_allow_html=True)
    with top_c2:
        st.markdown("<div class='card-title'>10-Day Health Trend</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-desc'>Tracks the market regime score over time.</div>", unsafe_allow_html=True)
        fig_m1 = go.Figure(go.Bar(x=df_10d["Date_Str"], y=df_10d["Composite_Score"], marker_color=[get_bar_color(v) for v in df_10d["Composite_Score"]], text=df_10d["Composite_Score"], textposition="outside", hovertemplate="Score: <b>%{y}</b><extra></extra>"))
        fig_m1.update_layout(height=120, margin=dict(l=10, r=10, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        fig_m1.update_xaxes(showgrid=False, tickfont=dict(size=10, color="#94a3b8"))
        fig_m1.update_yaxes(visible=False, range=[0, 115])
        st.plotly_chart(fig_m1, use_container_width=True, config={"displayModeBar": False})
    st.markdown(f"<div class='action-banner'>🎯 ACTION ZONE: {action_zone}</div>", unsafe_allow_html=True)

advances = safe_int(latest.get("Advances", 0))
declines = safe_int(latest.get("Declines", 0))
total_univ = safe_int(latest.get("Total_Universe", 2400))
actual_date_str = f"{selected_date.day} {selected_date.strftime('%B %Y')}"
if selected_date == today and live_available:
    actual_date_str += " <span style='color:#eab308; font-weight:800;'>(⚡ LIVE INTRADAY SNAPSHOT)</span>"

st.markdown(f"<p style='color: #475569; font-size: 13px; font-weight: 600; margin-top: 15px;'>Market Breadth Status for: <span style='color:#0f172a;'>{actual_date_str}</span></p>", unsafe_allow_html=True)
hero_col1, hero_col2 = st.columns([1.5, 2.5])
total_adv_dec = advances + declines
adv_pct = round((advances / total_adv_dec) * 100, 1) if total_adv_dec > 0 else 0
prev_advances = safe_int(prev.get("Advances", 0))
prev_declines = safe_int(prev.get("Declines", 0))
prev_total = prev_advances + prev_declines
prev_adv_pct = round(prev_advances / prev_total * 100, 1) if prev_total else 0
adv_change = round(adv_pct - prev_adv_pct, 1)
adv_change_str = f"+{adv_change}%" if adv_change >= 0 else f"{adv_change}%"
adv_change_color = "#16a34a" if adv_change >= 0 else "#dc2626"

with hero_col1:
    with st.container(border=True):
        st.markdown("<div class='card-title' style='text-align: center; margin-top: 5px;'>OF UNIVERSE ADVANCING</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-desc' style='text-align: center;'>Real-time buying vs. selling participation.</div>", unsafe_allow_html=True)
        fig_dial = go.Figure(go.Indicator(mode="gauge+number", value=adv_pct, number={"suffix": "%", "font": {"size": 36, "color": "#0f172a"}}, gauge={"axis": {"range": [0, 100], "visible": False}, "bar": {"color": "#ef4444" if adv_pct < 50 else "#22c55e", "thickness": 0.18}, "bgcolor": "#f1f5f9", "borderwidth": 0}))
        fig_dial.update_layout(height=150, margin=dict(l=10, r=10, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_dial, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"<div style='text-align: center; margin-top: -10px; padding-bottom: 8px;'><span style='color: #16a34a; font-size: 15px; font-weight: 800;'>{advances} ADVANCES</span> &nbsp;&nbsp;<span style='color: #dc2626; font-size: 15px; font-weight: 800;'>{declines} DECLINES</span><p style='color: {adv_change_color}; font-size: 12px; font-weight: 700; margin-top: 4px; margin-bottom: 0px;'>{adv_change_str} vs Yesterday EOD (Active Univ {total_univ})</p></div>", unsafe_allow_html=True)

with hero_col2:
    with st.container(border=True):
        st.markdown("<div class='card-title' style='margin-top: 5px; margin-left: 10px;'>TACTICAL COMMAND CENTER</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-desc' style='margin-left: 10px;'>Your immediate action plan dynamically dictated by the Momentum Health Score.</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='padding: 0px 10px; font-size: 13px;'><div style='margin-bottom: 6px;'><b>🎯 Target Asset:</b> <span style='color: #334155;'>{tacs['asset']}</span></div><div style='margin-bottom: 6px;'><b>⚖️ Position Sizing:</b> <span style='color: #334155;'>{tacs['sizing']}</span></div><div style='margin-bottom: 6px;'><b>🛡️ Risk / Stop-Loss:</b> <span style='color: #334155;'>{tacs['risk']}</span></div><div style='margin-bottom: 6px;'><b>💰 Profit Strategy:</b> <span style='color: #334155;'>{tacs['profit']}</span></div></div>", unsafe_allow_html=True)
        st.markdown("<hr style='margin: 8px 0px;'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title' style='margin-left: 10px;'>DECISION ENGINE: KEY REGIME DRIVERS</div>", unsafe_allow_html=True)
        wr_c = "#16a34a" if ft_rate >= 45 else "#dc2626"
        p20_c = "#16a34a" if p_fast >= 50 else "#dc2626"
        p50_c = "#16a34a" if p_mid >= 50 else "#dc2626"
        regime_html = f"<div style='font-size: 12px; margin-bottom: 8px; padding-left: 10px;'><div style='margin-bottom: 4px;'>• Follow-Through Rate: <b style='color:{wr_c};'>{ft_rate:.1f}%</b></div><div style='margin-bottom: 4px;'>• Fast Breadth (>20 EMA): <b style='color:{p20_c};'>{p_fast:.1f}%</b></div><div style='margin-bottom: 4px;'>• Trend Breadth (>50 EMA): <b style='color:{p50_c};'>{p_mid:.1f}%</b></div></div>"
        extremes = []
        trin = latest.get("TRIN", np.nan)
        if pd.notna(trin):
            if trin >= 1.5 and score <= 31: extremes.append(f"<div style='margin-bottom: 3px;'>🟢 <b>TRIN: {trin:.2f}</b> (Panic Capitulation)</div>")
            elif trin <= 0.7 and score >= 51: extremes.append(f"<div style='margin-bottom: 3px;'>🟢 <b>TRIN: {trin:.2f}</b> (Aggressive Demand)</div>")
        mco = latest.get("MCO", np.nan)
        if pd.notna(mco) and abs(float(mco)) >= 50:
            extremes.append(f"<div style='margin-bottom: 3px;'>🟢 <b>MCO: {float(mco):.1f}</b> (Extreme reading)</div>")
        vr = latest.get("Volume_Ratio", np.nan)
        if pd.notna(vr) and (float(vr) >= 2 or float(vr) <= 0.5):
            extremes.append(f"<div style='margin-bottom: 3px;'>🟢 <b>Vol Ratio: {float(vr):.2f}</b></div>")
        if extremes:
            regime_html += "<div style='border-top: 1px dashed #cbd5e1; margin-top: 8px; padding-top: 8px; padding-left: 10px;'><div style='font-size: 11px; font-weight: 800; color: #f97316; margin-bottom: 4px;'>⚠️ ACTIONABLE EXTREMES:</div>" + "".join(extremes) + "</div>"
        st.markdown(regime_html, unsafe_allow_html=True)

m2, m3 = st.columns(2)
with m2:
    vr = latest.get("Volume_Ratio", np.nan)
    v_str = "N/A" if pd.isna(vr) else f"{float(vr):.2f}"
    v_col = "#94a3b8" if pd.isna(vr) else ("#22c55e" if float(vr) > 1 else "#ef4444")
    with st.container(border=True):
        st.markdown("<div class='card-title' style='margin-left: 10px; margin-top: 5px;'>Volume Breadth Ratio</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-desc' style='margin-left: 10px;'>Ratio of volume in advancing vs. declining stocks.</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center;'><div class='metric-value' style='color: {v_col};'>{v_str}</div></div>", unsafe_allow_html=True)
        series = df_10d.get("Volume_Ratio", pd.Series(0, index=df_10d.index)).fillna(0)
        fig = go.Figure(go.Bar(x=df_10d["Date_Str"], y=series, marker_color=["#22c55e" if v >= 1 else "#ef4444" for v in series]))
        fig.add_hline(y=1, line_dash="dash", line_color="#cbd5e1")
        fig.update_layout(height=100, margin=dict(l=5, r=5, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        fig.update_yaxes(visible=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with m3:
    net_hl = safe_int(latest.get("Net_52W_High_Low", 0))
    ipo_hl = safe_int(latest.get("IPO_New_Highs", 0))
    with st.container(border=True):
        st.markdown("<div class='card-title' style='margin-left: 10px; margin-top: 5px;'>Net 52-Week Highs (Mature)</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chart-desc' style='margin-left: 10px;'>Long-term strength. (Plus <b style='color:#3b82f6;'>{ipo_hl}</b> new IPOs hitting all-time highs).</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center;'><div class='metric-value'>{net_hl}</div></div>", unsafe_allow_html=True)
        series = df_10d.get("Net_52W_High_Low", pd.Series(0, index=df_10d.index)).fillna(0)
        fig = go.Figure(go.Bar(x=df_10d["Date_Str"], y=series, marker_color=["#22c55e" if v >= 0 else "#ef4444" for v in series]))
        fig.add_hline(y=0, line_color="#cbd5e1")
        fig.update_layout(height=100, margin=dict(l=5, r=5, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        fig.update_yaxes(visible=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.markdown("<br>", unsafe_allow_html=True)
tf_col1, tf_col2 = st.columns([3, 1])
with tf_col1:
    st.markdown("### 📊 Historical Market Analytics")
with tf_col2:
    timeframe = st.radio("Chart Horizon:", ["1 Month", "3 Months", "6 Months", "1 Year", "3 Years", "6 Years"], horizontal=True, index=2)

days_map = {"1 Month": 21, "3 Months": 63, "6 Months": 126, "1 Year": 252, "3 Years": 756, "6 Years": len(df_filtered)}
plot_df = df_filtered.tail(days_map.get(timeframe, 126)).copy()
plot_df["Chart_Date_Str"] = plot_df["Date"].apply(lambda x: f"{x.day} {x.strftime('%B')[:3]}")

with st.container(border=True):
    st.markdown(f"<div class='card-title' style='margin-left: 10px; margin-top: 10px;'>UNIVERSE EMA BREADTH TRENDS &nbsp;|&nbsp; LATEST: <span style='color:#22c55e;'>200 EMA ({plot_df['Pct_Above_200_EMA'].iloc[-1]:.1f}%)</span> • <span style='color:#a855f7;'>50 EMA ({plot_df['Pct_Above_50_EMA'].iloc[-1]:.1f}%)</span> • <span style='color:#3b82f6;'>20 EMA ({plot_df['Pct_Above_20_EMA'].iloc[-1]:.1f}%)</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='chart-desc' style='margin-left: 10px;'>Percentage of mature active stocks above key moving averages.</div>", unsafe_allow_html=True)
    fig = go.Figure()
    for col, name, color in [("Pct_Above_200_EMA", "% > 200 EMA", "#22c55e"), ("Pct_Above_50_EMA", "% > 50 EMA", "#a855f7"), ("Pct_Above_20_EMA", "% > 20 EMA", "#3b82f6")]:
        fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df[col], mode="lines", name=name, line=dict(color=color, width=2)))
    fig.add_hline(y=50, line_dash="dash", line_color="#94a3b8")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified", legend=dict(orientation="h", y=1.02))
    fig.update_yaxes(range=[0, 100], gridcolor="#f1f5f9", title="% Stocks")
    st.plotly_chart(fig, use_container_width=True)

with st.container(border=True):
    st.markdown("<div class='card-title' style='margin-left: 10px; margin-top: 10px;'>SEGMENTED LIQUIDITY FLOW (45-DAY ROLLING TURNOVER RANK)</div>", unsafe_allow_html=True)
    st.markdown("<div class='chart-desc' style='margin-left: 10px;'>Breadth broken down by liquidity.</div>", unsafe_allow_html=True)
    cap_tab1, cap_tab2, cap_tab3 = st.tabs(["% Stocks Above 200 EMA", "% Stocks Above 50 EMA", "% Stocks Above 20 EMA"])
    def plot_liquidity(col_prefix):
        fig = go.Figure()
        for prefix, label, color in [("Large", "Top 100 Liq", "#2563eb"), ("Mid", "Mid 150 Liq", "#f97316"), ("Small", "Lower 250 Liq", "#16a34a"), ("Micro", "Micro Liq", "#dc2626")]:
            series = plot_df.get(f"{prefix}_{col_prefix}", pd.Series(0, index=plot_df.index)).fillna(0)
            fig.add_trace(go.Scatter(x=plot_df["Date"], y=series, mode="lines", name=label, line=dict(color=color, width=2)))
        fig.add_hline(y=50, line_dash="dash", line_color="#94a3b8")
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
        fig.update_yaxes(range=[0, 100], gridcolor="#f1f5f9")
        st.plotly_chart(fig, use_container_width=True)
    with cap_tab1: plot_liquidity("Pct_200_EMA")
    with cap_tab2: plot_liquidity("Pct_50_EMA")
    with cap_tab3: plot_liquidity("Pct_20_EMA")

st.markdown("<br>", unsafe_allow_html=True)
tac_col1, tac_col2 = st.columns(2)
with tac_col1:
    with st.container(border=True):
        latest_mco_chart = plot_df["MCO"].iloc[-1] if "MCO" in plot_df.columns else np.nan
        st.markdown(f"<div class='card-title' style='margin-left: 10px; margin-top: 10px;'>McCLELLAN OSCILLATOR (MCO) &nbsp;|&nbsp; LATEST: <span>{'N/A' if pd.isna(latest_mco_chart) else f'{latest_mco_chart:.1f}'}</span></div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-desc' style='margin-left: 10px;'>Short-term momentum of advancing issues.</div>", unsafe_allow_html=True)
        series = plot_df.get("MCO", pd.Series(0, index=plot_df.index)).fillna(0)
        fig = go.Figure(go.Bar(x=plot_df["Date"], y=series, marker_color=["#22c55e" if v >= 0 else "#ef4444" for v in series]))
        fig.add_hline(y=0, line_color="#94a3b8")
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
with tac_col2:
    with st.container(border=True):
        latest_trin_chart = plot_df["TRIN"].iloc[-1] if "TRIN" in plot_df.columns else np.nan
        trin_str = "N/A" if pd.isna(latest_trin_chart) else f"{latest_trin_chart:.2f}"
        st.markdown(f"<div class='card-title' style='margin-left: 10px; margin-top: 10px;'>TRIN (ARMS INDEX) &nbsp;|&nbsp; LATEST: <span>{trin_str}</span></div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-desc' style='margin-left: 10px;'>Contrarian indicator balancing A/D and volume.</div>", unsafe_allow_html=True)
        series = plot_df.get("TRIN", pd.Series(np.nan, index=plot_df.index))
        fig = go.Figure(go.Bar(x=plot_df["Date"], y=series, marker_color=["#22c55e" if pd.notna(v) and v < 1 else "#ef4444" for v in series]))
        fig.add_hline(y=1, line_dash="dash", line_color="#cbd5e1")
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with st.container(border=True):
    wins = plot_df.get("T3_Wins", pd.Series(0, index=plot_df.index)).fillna(0)
    breaks = plot_df.get("T3_Breakouts", pd.Series(0, index=plot_df.index)).fillna(0)
    rates = np.where(breaks > 0, wins / breaks * 100, 0)
    st.markdown(f"<div class='card-title' style='margin-left: 10px; margin-top: 10px;'>VCP BREAKOUT FOLLOW-THROUGH (T+3 WIN RATE) &nbsp;|&nbsp; LATEST: {rates[-1]:.1f}% (3-Day Cohort)</div>", unsafe_allow_html=True)
    st.markdown("<div class='chart-desc' style='margin-left: 10px;'>T+3 metrics represent completed cohorts; today’s breakouts are not yet final.</div>", unsafe_allow_html=True)
    fig = go.Figure(go.Bar(x=plot_df["Date"], y=rates, marker_color=["#22c55e" if v >= 45 else "#ef4444" for v in rates]))
    fig.add_hline(y=45, line_dash="dash", line_color="#22c55e")
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

out1, out2 = st.columns(2)
with out1:
    with st.container(border=True):
        up = plot_df.get("Up_25_1M_Count", pd.Series(0, index=plot_df.index)).fillna(0)
        down = plot_df.get("Down_25_1M_Count", pd.Series(0, index=plot_df.index)).fillna(0)
        st.markdown(f"<div class='card-title' style='margin-left: 10px; margin-top: 10px;'>ROLLING 1-MONTH 25% MOVERS &nbsp;|&nbsp; LATEST: {int(up.iloc[-1])} UP / {int(down.iloc[-1])} DOWN</div>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=plot_df["Date"], y=up, marker_color="#22c55e"))
        fig.add_trace(go.Bar(x=plot_df["Date"], y=-down, marker_color="#ef4444"))
        fig.update_layout(barmode="relative", height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
with out2:
    with st.container(border=True):
        up = plot_df.get("Rolling_3D_Up_4", pd.Series(0, index=plot_df.index)).fillna(0)
        down = plot_df.get("Rolling_3D_Down_4", pd.Series(0, index=plot_df.index)).fillna(0)
        st.markdown(f"<div class='card-title' style='margin-left: 10px; margin-top: 10px;'>3-DAY ROLLING 4% THRUST MOVERS &nbsp;|&nbsp; LATEST: {int(up.iloc[-1])} UP / {int(down.iloc[-1])} DOWN</div>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=plot_df["Date"], y=up, marker_color="#22c55e"))
        fig.add_trace(go.Bar(x=plot_df["Date"], y=-down, marker_color="#ef4444"))
        fig.update_layout(barmode="relative", height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🔍 Stock Level Drill-Down")
st.markdown(f"<p style='color: #64748b; font-size: 13px;'>Filter the underlying constituent stocks for <b>{selected_date.strftime('%d %B %Y')}</b></p>", unsafe_allow_html=True)
param_choices = st.multiselect("Select Parameters to Filter (Combines with AND):", ["Advances (Stocks in Green)", "Declines (Stocks in Red)", "Stocks > 20 EMA (Mature Only)", "Stocks > 50 EMA (Mature Only)", "Stocks > 200 EMA (Mature Only)", "Up 4% or more Today", "Down 4% or more Today", "1-Month 25% Winners", "1-Month 25% Losers", "New 52-Week Highs (Mature > 1Yr)", "New IPO/Listing Highs (< 1Yr)"])

if not df_stock.empty and "Date" in df_stock.columns:
    df_stock["Date"] = pd.to_datetime(df_stock["Date"], errors="coerce").dt.normalize()
    drill = df_stock[df_stock["Date"] == selected_date].copy()
else:
    drill = pd.DataFrame()

if not drill.empty:
    res = drill.copy()
    for param in param_choices:
        mapping = {
            "Advances (Stocks in Green)": "Gainer", "Declines (Stocks in Red)": "Loser", "Stocks > 20 EMA (Mature Only)": "Above_20_EMA", "Stocks > 50 EMA (Mature Only)": "Above_50_EMA", "Stocks > 200 EMA (Mature Only)": "Above_200_EMA", "Up 4% or more Today": "Up_4_Pct", "Down 4% or more Today": "Down_4_Pct", "1-Month 25% Winners": "Up_25_1M", "1-Month 25% Losers": "Down_25_1M", "New 52-Week Highs (Mature > 1Yr)": "New_52W_High", "New IPO/Listing Highs (< 1Yr)": "IPO_New_High"
        }
        column = mapping.get(param)
        if column in res.columns:
            res = res[res[column] == True]
    columns = [c for c in ["Symbol", "Close", "Daily_Pct", "Pct_1M"] if c in res.columns]
    if columns:
        res = res[columns].sort_values("Daily_Pct", ascending=False) if "Daily_Pct" in columns else res[columns]
        res = res.rename(columns={"Daily_Pct": "Daily_%_Change", "Pct_1M": "1M_%_Change"})
        st.write(f"**Found {len(res)} matching VIP active stocks** for {selected_date.strftime('%d %B %Y')}:")
        st.dataframe(res, use_container_width=True, height=350)
else:
    st.info("💡 Intraday stock metrics are available for today; permanent EOD data is used for historical dates.")

st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(f"<div style='text-align: center; font-size: 11px; color: #94a3b8; margin-bottom: 5px;'>Temporary data source: {'intraday_tmp' if live_available else 'permanent EOD data'}</div>", unsafe_allow_html=True)

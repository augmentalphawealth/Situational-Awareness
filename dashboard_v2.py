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
    .action-banner { background-color: #0f172a; color: #f8fafc; padding: 10px 16px; border-radius: 8px; font-size: 13px; font-weight: 700; text-align: center; margin-top: 8px; }
    .status-badge { font-size: 12px; font-weight: 700; padding: 4px 8px; border-radius: 4px; margin-left: 10px; }
    .status-live { background-color: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
    .status-delayed { background-color: #fef08a; color: #854d0e; border: 1px solid #fde047; }
    .status-stale { background-color: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
    </style>
""", unsafe_allow_html=True)

REPO_OWNER = "augmentalphawealth"
REPO_NAME = "Situational-Awareness"
BRANCH = "main"

def read_remote_file(path):
    url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{path}?t={int(time.time())}"
    headers = {"Cache-Control": "no-cache"}
    token = st.secrets.get("GITHUB_TOKEN", None)
    if token: headers["Authorization"] = f"token {token}"
    res = requests.get(url, headers=headers)
    return res.text if res.status_code == 200 else None

last_sync_text = read_remote_file("last_sync.txt") or "Unknown"

# Determine Data Freshness Banner Status
data_status_html = ""
if "Unknown" not in last_sync_text:
    try:
        ist_now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
        sync_str = last_sync_text.replace("Today, ", f"{ist_now.strftime('%Y-%m-%d')} ")
        sync_time = datetime.datetime.strptime(sync_str, "%Y-%m-%d %I:%M %p IST").replace(tzinfo=datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
        diff_mins = (ist_now - sync_time).total_seconds() / 60
        if diff_mins < 30: data_status_html = "<span class='status-badge status-live'>🟢 LIVE DATA</span>"
        elif diff_mins < 1440: data_status_html = "<span class='status-badge status-delayed'>🟡 DELAYED (>30m)</span>"
        else: data_status_html = "<span class='status-badge status-stale'>🔴 STALE (>24h)</span>"
    except: data_status_html = ""

@st.cache_data(show_spinner=False)
def load_agg_data(sync_key):
    csv_text = read_remote_file("historical_breadth_regime_6yr.csv")
    if csv_text:
        df = pd.read_csv(StringIO(csv_text))
        df['Date'] = pd.to_datetime(df['Date'])
        return df.sort_values('Date').reset_index(drop=True)
    return pd.DataFrame()

df_agg = load_agg_data(last_sync_text)

head_col1, head_spacer, head_col2, head_col3 = st.columns([3.0, 0.5, 2.0, 1.2])
with head_col1: st.markdown(f"<h2 style='margin-top: 10px; margin-bottom: 0px;'>🛡️ SITUATIONAL AWARENESS {data_status_html}</h2>", unsafe_allow_html=True)

if df_agg.empty: st.stop()
latest = df_agg.iloc[-1]
df_10d = df_agg.tail(10).copy()
df_10d['Date_Str'] = df_10d['Date'].dt.strftime('%d %b')

score = latest.get('Composite_Score', 0)
score_color = "#22c55e" if score >= 71 else "#84cc16" if score >= 51 else "#f97316" if score >= 31 else "#ef4444" if score >= 11 else "#991b1b"
action_zone = "AGGRESSIVE MTF ZONE" if score >= 71 else "CONFIRMED UPTREND" if score >= 51 else "DEATH CHOP ZONE" if score >= 31 else "RISK-OFF" if score >= 11 else "CAPITULATION"

with st.container(border=True):
    top_c1, top_c2 = st.columns([1.2, 2.8])
    with top_c1:
        st.markdown(f"<div style='text-align: center;'><div class='card-title'>MOMENTUM HEALTH SCORE</div><div class='metric-value' style='color: {score_color};'>{score} / 100</div></div>", unsafe_allow_html=True)
    with top_c2:
        fig_m1 = go.Figure(go.Bar(x=df_10d['Date_Str'], y=df_10d['Composite_Score']))
        fig_m1.update_layout(height=120, margin=dict(l=10, r=10, t=25, b=0), plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig_m1, use_container_width=True)
    st.markdown(f"<div class='action-banner'>🎯 ACTION ZONE: {action_zone}</div>", unsafe_allow_html=True)

m1, m2 = st.columns(2)
with m1:
    vol_ratio = latest.get('Volume_Ratio', np.nan)
    v_str = f"{vol_ratio:.2f}" if pd.notna(vol_ratio) else "N/A"
    with st.container(border=True):
        st.markdown(f"<div style='text-align: center;'><div class='card-title'>Volume Breadth Ratio</div><div class='metric-value'>{v_str}</div></div>", unsafe_allow_html=True)
with m2:
    net_hl = int(latest.get('Net_52W_High_Low', 0))
    with st.container(border=True):
        st.markdown(f"<div style='text-align: center;'><div class='card-title'>Net 52-Week Highs vs Lows</div><div class='metric-value'>{net_hl}</div></div>", unsafe_allow_html=True)

st.markdown("### 🔥 Tactical Exertion")
tac_col1, tac_col2 = st.columns(2)
with tac_col1:
    with st.container(border=True):
        st.markdown("<div class='card-title'>McCLELLAN OSCILLATOR (MCO)</div>", unsafe_allow_html=True)
        fig_mco = go.Figure(go.Bar(x=df_agg['Date'], y=df_agg['MCO']))
        fig_mco.update_layout(height=350, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_mco, use_container_width=True)
with tac_col2:
    with st.container(border=True):
        st.markdown("<div class='card-title'>TRIN (ARMS INDEX)</div>", unsafe_allow_html=True)
        fig_trin = go.Figure(go.Bar(x=df_agg['Date'], y=df_agg['TRIN']))
        fig_trin.update_layout(height=350, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_trin, use_container_width=True)

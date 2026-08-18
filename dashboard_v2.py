import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import datetime

# Page Configuration
st.set_page_config(page_title="Situational Awareness Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("historical_breadth_regime_6yr.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    # Calculate Cumulative A/D Line for charting
    df['AD_Margin'] = df['Advances'] - df['Declines']
    df['Cumulative_AD'] = df['AD_Margin'].cumsum()
    return df

df = load_data()

# --- SIDEBAR: Date Filter (Fixed Historical Limit) ---
min_date = datetime.date(2018, 4, 1) # Unlocked database history
max_date = df['Date'].max().date()

st.sidebar.header("Filter Data")
date_range = st.sidebar.date_input("Select Date Range", 
                                   value=[max_date - datetime.timedelta(days=180), max_date], 
                                   min_value=min_date, 
                                   max_value=max_date)

if len(date_range) == 2:
    mask = (df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1])
    df_filtered = df.loc[mask]
else:
    df_filtered = df

st.title("🛡️ Situational Awareness Dashboard")
st.markdown("---")

# ==========================================
# CATEGORY 1: CURRENT REGIME
# ==========================================
st.header("📈 Current Regime")
st.info("Monitors the underlying structural trend and participation breadth of the broader market.")

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Market Breadth (EMAs)")
    fig_ema = go.Figure()
    fig_ema.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['Pct_Above_20_EMA'], name="> 20 EMA", line=dict(color='green')))
    fig_ema.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['Pct_Above_50_EMA'], name="> 50 EMA", line=dict(color='orange')))
    # FIX: 200 EMA explicitly rendered and set to visible
    fig_ema.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['Pct_Above_200_EMA'], name="> 200 EMA", line=dict(color='red'), visible=True))
    fig_ema.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), hovermode='x unified')
    st.plotly_chart(fig_ema, use_container_width=True)
    st.caption("Tracks the percentage of stocks trading above moving averages.")

with c2:
    st.subheader("Cumulative A/D Line")
    fig_cad = go.Figure()
    fig_cad.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['Cumulative_AD'], fill='tozeroy', name="Cum. A/D", line=dict(color='royalblue')))
    fig_cad.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_cad, use_container_width=True)
    st.caption("A continuously rising line confirms a healthy bull trend.")

with c3:
    st.subheader("Advance / Decline Dial")
    latest = df_filtered.iloc[-1]
    total_ad = latest['Advances'] + latest['Declines']
    adv_pct = (latest['Advances'] / total_ad) * 100 if total_ad > 0 else 50
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = adv_pct,
        number = {'suffix': "%"},
        gauge = {'axis': {'range': [0, 100]},
                 'bar': {'color': "#00cc96" if adv_pct > 50 else "#ef553b"},
                 'steps': [{'range': [0, 40], 'color': "rgba(255, 0, 0, 0.1)"},
                           {'range': [60, 100], 'color': "rgba(0, 255, 0, 0.1)"}]}
    ))
    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.caption("Measures daily net advancing power vs. declining drag.")

st.markdown("---")

# ==========================================
# CATEGORY 2: SIGNS OF EXERTION (FROTH)
# ==========================================
st.header("🔥 Signs of Exertion")
st.info("Identifies when market momentum is overextended or breakouts are failing to follow through.")

c4, c5 = st.columns(2)

with c4:
    st.subheader("McClellan Oscillator (MCO)")
    fig_mco = go.Figure()
    colors = ['#00cc96' if val >= 0 else '#ef553b' for val in df_filtered['MCO']]
    fig_mco.add_trace(go.Bar(x=df_filtered['Date'], y=df_filtered['MCO'], marker_color=colors, name="MCO"))
    fig_mco.add_hline(y=0, line_dash="solid", line_color="white")
    fig_mco.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_mco, use_container_width=True)
    st.caption("Velocity of money. Dropping below zero warns of short-term exhaustion even if the index is up.")

with c5:
    st.subheader("Breakout Success vs. Failure")
    fig_brk = go.Figure()
    fig_brk.add_trace(go.Bar(x=df_filtered['Date'], y=df_filtered['T3_Breakouts'], name="Total Breakouts Attempted", marker_color='rgba(200, 200, 200, 0.4)'))
    fig_brk.add_trace(go.Bar(x=df_filtered['Date'], y=df_filtered['T3_Wins'], name="Successful T3 Holds", marker_color='#FFC107'))
    fig_brk.update_layout(barmode='overlay', height=300, margin=dict(l=0, r=0, t=30, b=0), hovermode='x unified')
    st.plotly_chart(fig_brk, use_container_width=True)
    st.caption("Visualizes the raw win-rate. Shrinking yellow bars indicate a failing breakout regime.")

st.markdown("---")

# ==========================================
# CATEGORY 3: SIGNS OF BOTTOM (CAPITULATION)
# ==========================================
st.header("🩸 Signs of Bottom (Capitulation)")
st.info("Highlights moments of extreme, algorithmic-driven panic selling that often precedes market bottoms.")

# Full width for TRIN to visualize historical spikes cleanly
st.subheader("TRIN (Arms Index)")
fig_trin = go.Figure()
fig_trin.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['TRIN'], name="TRIN", line=dict(color='#ab63fa', width=2)))
# Highlight thresholds
fig_trin.add_hline(y=2.0, line_dash="dash", line_color="red", annotation_text="Extreme Panic (>2.0)", annotation_position="top left")
fig_trin.add_hline(y=0.5, line_dash="dash", line_color="green", annotation_text="Complacency/Froth (<0.5)", annotation_position="bottom left")
# Restricting Y-axis slightly so massive anomalies don't flatten the rest of the chart
fig_trin.update_yaxes(range=[0, 4.5])
fig_trin.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig_trin, use_container_width=True)
st.caption("A TRIN reading above 2.0 means extreme panic selling is happening. Market bottoms form on days with maximum fear.")

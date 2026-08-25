# Calendar replacement block for dashboard.py
# Replace the existing df_10d + 10-Day Health Trend block with this block.

# --- MONTHLY COMPOSITE SCORE CALENDAR ---
df_month = df_filtered.copy()
df_month['Date'] = pd.to_datetime(df_month['Date'], errors='coerce')
df_month['Composite_Score'] = pd.to_numeric(df_month['Composite_Score'], errors='coerce')
df_month = df_month.dropna(subset=['Date', 'Composite_Score']).sort_values('Date')

calendar_year = int(df_month['Date'].iloc[-1].year)
calendar_month = int(df_month['Date'].iloc[-1].month)
df_month = df_month[
    (df_month['Date'].dt.year == calendar_year) &
    (df_month['Date'].dt.month == calendar_month)
].copy()

month_name = pd.Timestamp(calendar_year, calendar_month, 1).strftime('%B %Y')

def score_color(score_value):
    if score_value > 80:
        return '#166534'
    if score_value >= 60:
        return '#22c55e'
    if score_value >= 40:
        return '#f97316'
    if score_value >= 20:
        return '#ef4444'
    return '#991b1b'

def score_text_color(score_value):
    return '#ffffff' if score_value < 60 or score_value > 80 else '#052e16'

calendar_cells = []
for _, row in df_month.iterrows():
    score_value = float(row['Composite_Score'])
    day_label = row['Date'].strftime('%d %b')
    calendar_cells.append(
        f"<div style='background:{score_color(score_value)}; color:{score_text_color(score_value)}; "
        f"border-radius:8px; min-height:58px; padding:8px 5px; text-align:center; "
        f"box-shadow:0 1px 2px rgba(15,23,42,.08);'>"
        f"<div style='font-size:11px; font-weight:700; opacity:.9;'>{day_label}</div>"
        f"<div style='font-size:24px; font-weight:800; line-height:1.1; margin-top:4px;'>{int(round(score_value))}</div>"
        f"</div>"
    )

legend = (
    "<div style='display:flex; gap:8px; flex-wrap:wrap; margin:4px 0 12px 0; font-size:10px; font-weight:700;'>"
    "<span style='background:#991b1b;color:#fff;padding:4px 7px;border-radius:5px;'>0–20</span>"
    "<span style='background:#ef4444;color:#fff;padding:4px 7px;border-radius:5px;'>20–40</span>"
    "<span style='background:#f97316;color:#fff;padding:4px 7px;border-radius:5px;'>40–60</span>"
    "<span style='background:#22c55e;color:#052e16;padding:4px 7px;border-radius:5px;'>60–80</span>"
    "<span style='background:#166534;color:#fff;padding:4px 7px;border-radius:5px;'>80–100</span>"
    "</div>"
)

with st.container(border=True):
    st.markdown(
        f"<div class='card-title'>MONTHLY COMPOSITE SCORE CALENDAR &nbsp;|&nbsp; {month_name.upper()}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='chart-desc'>Each tile shows the trading date and composite score. "
        "Colour indicates the score zone.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(legend, unsafe_allow_html=True)
    if calendar_cells:
        st.markdown(
            "<div style='display:grid; grid-template-columns:repeat(auto-fit,minmax(82px,1fr)); "
            "gap:8px; width:100%;'>" + "".join(calendar_cells) + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info(f"No composite-score data is available for {month_name}.")

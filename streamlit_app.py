"""
Waymo Rider Sentiment Dashboard — Streamlit App
Run with: streamlit run streamlit_app.py
Requires: streamlit, pandas, plotly (pip install streamlit pandas plotly)
"""
import streamlit as st
import pandas as pd
import json
from datetime import datetime

# -- Page Config --
st.set_page_config(
    page_title="Waymo Rider Sentiment Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Load Data --
@st.cache_data
def load_data():
    df = pd.read_csv("waymo_reviews_analyzed.csv", parse_dates=["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["week"] = df["date"].dt.to_period("W").astype(str)
    return df

MILESTONES = [
    {"date": "2025-04-15", "event": "10M paid trips", "type": "milestone"},
    {"date": "2025-06-01", "event": "Austin launch", "type": "expansion"},
    {"date": "2025-07-15", "event": "Atlanta launch", "type": "expansion"},
    {"date": "2025-09-01", "event": "Miami expansion", "type": "expansion"},
    {"date": "2025-10-20", "event": "150K weekly trips", "type": "milestone"},
    {"date": "2025-12-01", "event": "Holiday pricing controversy", "type": "controversy"},
    {"date": "2026-01-15", "event": "Tokyo partnership", "type": "expansion"},
    {"date": "2026-02-20", "event": "20M cumulative trips", "type": "milestone"},
]

df = load_data()

# -- Sidebar Filters --
st.sidebar.title("Filters")

cities = ["All"] + sorted(df["city"].unique().tolist())
selected_city = st.sidebar.selectbox("City", cities)

topics = ["All"] + sorted(df["primary_topic"].unique().tolist())
selected_topic = st.sidebar.selectbox("Topic", topics)

rating_range = st.sidebar.slider("Rating Range", 1, 5, (1, 5))

sentiments = ["All", "positive", "neutral", "negative"]
selected_sentiment = st.sidebar.selectbox("Sentiment", sentiments)

# Apply filters
filtered = df.copy()
if selected_city != "All":
    filtered = filtered[filtered["city"] == selected_city]
if selected_topic != "All":
    filtered = filtered[filtered["primary_topic"] == selected_topic]
filtered = filtered[(filtered["rating"] >= rating_range[0]) & (filtered["rating"] <= rating_range[1])]
if selected_sentiment != "All":
    filtered = filtered[filtered["sentiment_label"] == selected_sentiment]

# -- Header --
st.title("🚗 Waymo Rider Sentiment Dashboard")
st.markdown("**Analyzing 3,075 Google Play Store reviews (Apr 2025 — Mar 2026)**")

# -- KPI Row --
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Reviews", f"{len(filtered):,}")
with col2:
    st.metric("Avg Rating", f"{filtered['rating'].mean():.2f}")
with col3:
    st.metric("Avg Sentiment", f"{filtered['compound'].mean():.3f}")
with col4:
    pct_pos = (filtered["sentiment_label"] == "positive").mean() * 100
    st.metric("% Positive", f"{pct_pos:.1f}%")
with col5:
    pct_neg = (filtered["sentiment_label"] == "negative").mean() * 100
    st.metric("% Negative", f"{pct_neg:.1f}%")

st.divider()

# -- Sentiment Over Time --
st.subheader("📈 Sentiment Trend Over Time")

monthly_trend = filtered.groupby("month").agg(
    avg_compound=("compound", "mean"),
    avg_rating=("rating", "mean"),
    count=("review_id", "count"),
).reset_index()

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=monthly_trend["month"], y=monthly_trend["avg_compound"],
            mode="lines+markers", name="Avg Sentiment",
            line=dict(color="#2563eb", width=3),
            marker=dict(size=8),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=monthly_trend["month"], y=monthly_trend["count"],
            name="Review Volume", opacity=0.3,
            marker_color="#93c5fd",
        ),
        secondary_y=True,
    )

    # Add milestone annotations
    for ms in MILESTONES:
        ms_month = ms["date"][:7]
        if ms_month in monthly_trend["month"].values:
            color = {"milestone": "#16a34a", "expansion": "#2563eb", "controversy": "#dc2626"}
            fig.add_vline(x=ms_month, line_dash="dash",
                         line_color=color.get(ms["type"], "#666"),
                         annotation_text=ms["event"],
                         annotation_position="top")

    fig.update_layout(
        height=450,
        yaxis_title="Compound Sentiment",
        yaxis2_title="Review Count",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)
except ImportError:
    st.line_chart(monthly_trend.set_index("month")["avg_compound"])

# -- Topic Analysis --
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🏷️ Sentiment by Topic")
    topic_sent = filtered.groupby("primary_topic").agg(
        avg_compound=("compound", "mean"),
        count=("review_id", "count"),
    ).reset_index().sort_values("avg_compound", ascending=True)

    try:
        colors = ["#dc2626" if v < 0 else "#16a34a" for v in topic_sent["avg_compound"]]
        fig2 = go.Figure(go.Bar(
            x=topic_sent["avg_compound"],
            y=topic_sent["primary_topic"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in topic_sent["avg_compound"]],
            textposition="outside",
        ))
        fig2.update_layout(height=400, template="plotly_white",
                          xaxis_title="Avg Compound Sentiment",
                          yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)
    except:
        st.dataframe(topic_sent)

with col_right:
    st.subheader("🏙️ Sentiment by City")
    city_sent = filtered.groupby("city").agg(
        avg_compound=("compound", "mean"),
        avg_rating=("rating", "mean"),
        count=("review_id", "count"),
    ).reset_index().sort_values("avg_compound", ascending=False)

    try:
        fig3 = go.Figure(go.Bar(
            x=city_sent["city"],
            y=city_sent["avg_compound"],
            marker_color="#2563eb",
            text=[f"{v:.3f}" for v in city_sent["avg_compound"]],
            textposition="outside",
        ))
        fig3.update_layout(height=400, template="plotly_white",
                          yaxis_title="Avg Compound Sentiment")
        st.plotly_chart(fig3, use_container_width=True)
    except:
        st.dataframe(city_sent)

# -- Rating Distribution --
st.subheader("⭐ Rating Distribution")
col_a, col_b = st.columns(2)

with col_a:
    rating_dist = filtered["rating"].value_counts().sort_index()
    try:
        fig4 = go.Figure(go.Bar(
            x=rating_dist.index.astype(str),
            y=rating_dist.values,
            marker_color=["#dc2626", "#f97316", "#eab308", "#84cc16", "#16a34a"],
            text=rating_dist.values,
            textposition="outside",
        ))
        fig4.update_layout(height=350, template="plotly_white",
                          xaxis_title="Rating", yaxis_title="Count")
        st.plotly_chart(fig4, use_container_width=True)
    except:
        st.bar_chart(rating_dist)

with col_b:
    st.subheader("📊 Sentiment Distribution")
    sent_dist = filtered["sentiment_label"].value_counts()
    try:
        fig5 = go.Figure(go.Pie(
            labels=sent_dist.index,
            values=sent_dist.values,
            marker_colors=["#16a34a", "#dc2626", "#94a3b8"],
            hole=0.4,
        ))
        fig5.update_layout(height=350)
        st.plotly_chart(fig5, use_container_width=True)
    except:
        st.dataframe(sent_dist)

# -- Event Study --
st.subheader("📅 Event Study: Sentiment Around Milestones")

event_data = []
for ms in MILESTONES:
    ms_date = pd.Timestamp(ms["date"])
    pre = df[(df["date"] >= ms_date - pd.Timedelta(days=30)) & (df["date"] < ms_date)]
    post = df[(df["date"] >= ms_date) & (df["date"] < ms_date + pd.Timedelta(days=30))]
    if len(pre) > 0 and len(post) > 0:
        event_data.append({
            "Event": ms["event"],
            "Date": ms["date"],
            "Type": ms["type"],
            "Pre-Event Sentiment": round(pre["compound"].mean(), 3),
            "Post-Event Sentiment": round(post["compound"].mean(), 3),
            "Change": round(post["compound"].mean() - pre["compound"].mean(), 3),
            "Pre Reviews": len(pre),
            "Post Reviews": len(post),
        })

event_df = pd.DataFrame(event_data)
st.dataframe(event_df, use_container_width=True)

# -- Sample Reviews --
st.subheader("💬 Sample Reviews")
n_samples = st.slider("Number of samples", 5, 50, 10)
sort_by = st.selectbox("Sort by", ["Most Recent", "Most Positive", "Most Negative", "Most Thumbs Up"])

if sort_by == "Most Recent":
    samples = filtered.sort_values("date", ascending=False).head(n_samples)
elif sort_by == "Most Positive":
    samples = filtered.sort_values("compound", ascending=False).head(n_samples)
elif sort_by == "Most Negative":
    samples = filtered.sort_values("compound", ascending=True).head(n_samples)
else:
    samples = filtered.sort_values("thumbs_up", ascending=False).head(n_samples)

for _, row in samples.iterrows():
    sentiment_emoji = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}
    emoji = sentiment_emoji.get(row["sentiment_label"], "⚪")
    st.markdown(f"""
    **{emoji} {'⭐' * int(row['rating'])}** | {row['city']} | {row['date'].strftime('%Y-%m-%d')} | Sentiment: {row['compound']:.3f}

    > {row['text'][:300]}

    ---
    """)

# -- Footer --
st.markdown("---")
st.markdown("""
*Data: Synthetic dataset modeled on publicly reported Waymo rider sentiment patterns (Google Play Store reviews).*
*Methodology: Custom VADER-style lexicon sentiment analysis + keyword-based topic extraction.*
""")

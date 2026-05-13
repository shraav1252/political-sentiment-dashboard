from emotion_engine import predict_emotion, predict_emotion_batch
from sentiment_engine import predict_sentiment
import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="CSV Sentiment Analyzer",
    layout="wide",
)

# --------------------------------------------------
# Apple-style global CSS
# --------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                 "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: #0b0f14;
    color: white;
}

h1 {
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: -0.03em;
}

h2 {
    font-weight: 600;
}

.stTabs [data-baseweb="tab"] {
    font-size: 15px;
    padding: 12px 20px;
}

.card {
    background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
    border-radius: 20px;
    padding: 36px;
    margin-bottom: 32px;
    box-shadow: 0 30px 80px rgba(0,0,0,0.6);
}

.inner-card {
    background-color: #0b1220;
    padding: 28px;
    border-radius: 18px;
}

.metric {
    font-size: 22px;
    font-weight: 600;
}

.metric-label {
    color: #9ca3af;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Sidebar – Upload
# --------------------------------------------------
st.sidebar.markdown("## 📁 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# --------------------------------------------------
# Hero Card (SAFE – no df usage)
# --------------------------------------------------
st.title("Sentiment Analysis")

st.markdown("""
<div class="card">
    <h2>Understand sentiment at a glance</h2>
    <p style="font-size: 16px; color: #9ca3af; max-width: 720px;">
        Upload any CSV file, select the relevant columns, and instantly
        explore sentiment distribution and word patterns — all locally,
        with no backend or external APIs.
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Empty State
# --------------------------------------------------
if not uploaded_file:
    st.markdown("""
    <div class="inner-card" style="text-align:center;">
        <h3>📂 Upload a CSV file to begin</h3>
        <p style="color: #9ca3af;">
            Your data stays on your machine.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# --------------------------------------------------
# Load Data
# --------------------------------------------------
df = pd.read_csv(uploaded_file)

# ------------------------------------------
# NLP Sentiment Prediction (NEW)
# ------------------------------------------
text_columns = df.select_dtypes(include=["object"]).columns.tolist()
cat_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

if not text_columns or not cat_columns:
    st.error("CSV must contain at least one text column.")
    st.stop()

# --------------------------------------------------
# Sidebar Column Selection
# --------------------------------------------------
st.sidebar.markdown("### 🧠 Column Selection")

post_col = st.sidebar.selectbox("Text column", text_columns)
sentiment_col = st.sidebar.selectbox("Sentiment column", cat_columns)

# ------------------------------------------
# NLP Sentiment Prediction
# ------------------------------------------
with st.spinner("Analyzing sentiment from text..."):
    sentiments = df[post_col].apply(predict_sentiment)

df["predicted_sentiment"] = sentiments.apply(lambda x: x[0])
df["sentiment_score"] = sentiments.apply(lambda x: x[1])

# ------------------------------------------
# NLP Emotion Detection (sampled + cached)
# ------------------------------------------
EMOTION_SAMPLE = 500

@st.cache_data(show_spinner=False)
def run_emotion_batch(texts_tuple):
    return predict_emotion_batch(list(texts_tuple), batch_size=64)

emotion_sample = df[post_col].dropna().head(EMOTION_SAMPLE)

with st.spinner(f"Detecting emotions on {len(emotion_sample)} rows (sampled for speed)..."):
    sampled_emotions = run_emotion_batch(tuple(emotion_sample.tolist()))

# fill full column: sampled rows get real labels, rest get "unknown"
df["predicted_emotion"] = "unknown"
df.loc[emotion_sample.index, "predicted_emotion"] = sampled_emotions

# Optional date column detection
date_columns = df.select_dtypes(include=["datetime", "object"]).columns.tolist()

st.sidebar.markdown("### 📅 Optional Time Analysis")
date_col = st.sidebar.selectbox(
    "Select date column (optional)",
    ["None"] + date_columns
)

# --------------------------------------------------
# Metrics Card (NOW SAFE)
# --------------------------------------------------
st.markdown(f"""
<div class="card">
    <div style="display:flex; gap:60px; flex-wrap:wrap;">
        <div>
            <div class="metric">📄 {len(df)}</div>
            <div class="metric-label">Rows</div>
        </div>
        <div>
            <div class="metric">🧠 {df["predicted_sentiment"].nunique()}</div>
            <div class="metric-label">Sentiment Classes</div>
        </div>
        <div>
            <div class="metric">📝 {post_col}</div>
            <div class="metric-label">Text Column</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Preview",
    "📊 Distribution",
    "☁️ Word Cloud",
    "📈 Trends",
    "😃 Emotion Distribution"

])

# --------------------------------------------------
# Preview
# --------------------------------------------------
with tab1:
    st.markdown('<div class="inner-card">', unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Distribution
# --------------------------------------------------
with tab2:
    st.markdown('<div class="inner-card">', unsafe_allow_html=True)

    counts = df["predicted_sentiment"].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]

    fig = px.pie(
        counts,
        names="Sentiment",
        values="Count",
        hole=0.55,
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("🔍 Key Words Driving Sentiment")

selected_sentiment = st.selectbox(
    "Select sentiment to analyze",
    df["predicted_sentiment"].unique()
)

filtered_text = " ".join(
    df[df["predicted_sentiment"] == selected_sentiment][post_col].astype(str)
)

if filtered_text.strip():
    wc = WordCloud(
        width=900,
        height=450,
        background_color="black",
        colormap="Reds"
    ).generate(filtered_text)

    fig_wc, ax = plt.subplots(figsize=(11, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig_wc)
else:
    st.warning("No text available for this sentiment.")

st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("📝 Example Texts")

example_df = (
    df[df["predicted_sentiment"] == selected_sentiment]
    [[post_col]]
    .dropna()
    .head(5)
)

for i, row in example_df.iterrows():
    st.markdown(
        f"""
        <div class="inner-card" style="margin-bottom:12px;">
            {row[post_col]}
        </div>
        """,
        unsafe_allow_html=True
    )
    
# --------------------------------------------------
# Word Cloud
# --------------------------------------------------
with tab3:
    st.markdown('<div class="inner-card">', unsafe_allow_html=True)

    choice = st.selectbox("Choose predicted sentiment", df["predicted_sentiment"].unique())
    text = " ".join(df[df["predicted_sentiment"] == choice][post_col].astype(str))

    if text.strip():
        wc = WordCloud(
            width=900,
            height=450,
            background_color="black",
            colormap="Blues"
        ).generate(text)

        fig_wc, ax = plt.subplots(figsize=(11, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig_wc)
    else:
        st.warning("No text available.")

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("""
<hr>
<p style="text-align:center; color:#6b7280;">
Frontend complete · Apple-style UI · Streamlit
</p>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TAB 4 – Sentiment Trend Over Time
# --------------------------------------------------
with tab4:
    # ---- Card header (title + controls) ----
    st.markdown("""
    <div class="card">
        <h2>Sentiment Trend Over Time</h2>
        <p style="color:#9ca3af; margin-bottom:16px;">
            Monthly sentiment trends based on selected reactions
        </p>
    """, unsafe_allow_html=True)

    # 🔹 placeholder for download button (INSIDE CARD)
    download_placeholder = st.empty()

    top_n = st.slider(
        "Select top reactions to display",
        min_value=3,
        max_value=10,
        value=5
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Chart container ----
    st.markdown('<div class="inner-card">', unsafe_allow_html=True)

    if date_col != "None":
        try:
            # Parse date safely
            df["_parsed_date"] = pd.to_datetime(df[date_col], errors="coerce")
            df_time = df.dropna(subset=["_parsed_date"])

            if df_time.empty:
                st.warning("No valid dates found after parsing.")
                st.stop()

            # Monthly aggregation
            df_time["_month"] = df_time["_parsed_date"].dt.to_period("M").astype(str)

            # Pick top N categories
            trend_data = (
                df_time
                .groupby(["_month", "predicted_sentiment"])
                .size()
                .reset_index(name="Count")
             )

            # -----------------------------------
            # Export trend data (INSIDE CARD)
            # -----------------------------------
            csv_data = trend_data.to_csv(index=False)

            with download_placeholder:
                st.download_button(
                    label="⬇️ Download Data (CSV)",
                    data=csv_data,
                    file_name="sentiment_trend_data.csv",
                    mime="text/csv"
                )

            # -----------------------------------
            # Plot
            # -----------------------------------
            fig = px.line(
                trend_data,
                x="_month",
                y="Count",
                color="predicted_sentiment",
                markers=True,
                title="Monthly Trend (Top Reactions Only)"
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                xaxis_title="Month",
                yaxis_title="Count"
            )

            st.plotly_chart(fig, use_container_width=True)

            # -------------------------------
            # Auto Insights
            # -------------------------------
            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("📌 Key Insights")

            total_counts = (
                df_time
                .groupby("predicted_sentiment")
                .size()
                .sort_values(ascending=False)
        )

            top_sentiment = total_counts.index[0]

            peak_points = (
                trend_data
                .sort_values("Count", ascending=False)
                .iloc[0]
    )

            peak_sentiment = peak_points["predicted_sentiment"]
            peak_month = peak_points["_month"]
            peak_value = peak_points["Count"]

            st.markdown(f"""
            <div class="inner-card">
                <ul style="font-size:16px; line-height:1.7;">
                    <li>🔥 <b>{top_sentiment}</b> is the most frequent predicted sentiment overall.</li>
                    <li>📈 <b>{peak_sentiment}</b> peaked in <b>{peak_month}</b> with <b>{int(peak_value)}</b> occurrences.</li>
                    <li>📊 Trends are aggregated monthly for clarity.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        except Exception:
            st.warning("Unable to generate trend or insights. Please check date format.")

    else:
        st.info("Select a date column from the sidebar to view trends.")

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Emotion Distribution
# --------------------------------------------------
with tab5:
    st.markdown('<div class="inner-card">', unsafe_allow_html=True)

    emotion_counts = (
        df["predicted_emotion"]
        .value_counts()
        .reset_index()
    )
    emotion_counts.columns = ["Emotion", "Count"]

    fig = px.bar(
        emotion_counts,
        x="Emotion",
        y="Count",
        title="Emotion Distribution",
        color="Emotion"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        xaxis_title="Emotion",
        yaxis_title="Count"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


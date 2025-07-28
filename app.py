import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="CSV Sentiment Analyzer", layout="wide")

st.sidebar.title("📁 Upload CSV")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

st.title("📊 Sentiment Analysis Dashboard")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Preview Data")
    st.dataframe(df.head())

    # Detect all text/object columns for post/sentiment selection
    text_columns = df.select_dtypes(include=['object']).columns.tolist()
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    st.sidebar.markdown("### 🧠 Column Selections")
    post_col = st.sidebar.selectbox("Select post/content column", text_columns)
    sentiment_col = st.sidebar.selectbox("Select sentiment label column", cat_columns)

    if post_col and sentiment_col:
        # Pie Chart
        st.subheader("📈 Sentiment Distribution")
        sentiment_counts = df[sentiment_col].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]
        fig = px.pie(sentiment_counts, names="Sentiment", values="Count",
                     title="Sentiment Distribution", color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)

        # Word Cloud
        st.subheader("☁️ Sentiment Word Cloud")
        selected_sentiment = st.selectbox("Select a sentiment to view word cloud", df[sentiment_col].unique())
        text = " ".join(df[df[sentiment_col] == selected_sentiment][post_col].astype(str))

        if text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            fig_wc, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig_wc)
        else:
            st.warning("⚠️ Not enough text data for the selected sentiment.")
else:
    st.info("👈 Upload a CSV file from the sidebar to get started.")

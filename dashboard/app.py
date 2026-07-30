import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


st.set_page_config(

    page_title="MaCoDeR Dashboard",

    layout="wide"
)

st.title("🧠 MaCoDeR AI Research Dashboard")

st.markdown(
    "Multimodal Cognitive Stress "
    "and Deception Analysis System"
)


log_dir = Path("outputs/session_logs")

csv_files = sorted(
    log_dir.glob("*.csv"),
    reverse=True
)

if not csv_files:

    st.warning("No session logs found")

    st.stop()


selected_file = st.sidebar.selectbox(

    "Select Session",

    csv_files,

    format_func=lambda x: x.name
)


df = pd.read_csv(selected_file)

st.sidebar.success(
    f"Loaded: {selected_file.name}"
)


col1, col2, col3, col4 = st.columns(4)

col1.metric(

    "Records",

    len(df)
)

col2.metric(

    "Avg Emotion Confidence",

    f"{df['emotion_confidence'].mean():.1f}%"
)

col3.metric(

    "Avg Temporal Confidence",

    f"{df['temporal_confidence'].mean():.1f}%"
)

col4.metric(

    "Max Blink Rate",

    f"{df['blink_rate'].max():.1f}"
)


st.subheader("Emotion Timeline")

emotion_fig = px.line(

    df,

    x=df.index,

    y="emotion_confidence",

    title="Emotion Confidence Over Time"
)

st.plotly_chart(
    emotion_fig,
    use_container_width=True
)


st.subheader("Temporal Emotion Confidence")

temporal_fig = px.line(

    df,

    x=df.index,

    y="temporal_confidence",

    title="Temporal Emotion Confidence"
)

st.plotly_chart(
    temporal_fig,
    use_container_width=True
)


st.subheader("Blink Rate Analysis")

blink_fig = px.line(

    df,

    x=df.index,

    y="blink_rate",

    title="Blink Rate Trend"
)

st.plotly_chart(
    blink_fig,
    use_container_width=True
)


st.subheader("Detected Emotion Distribution")

emotion_counts = (
    df["emotion"]
    .value_counts()
    .reset_index()
)

emotion_counts.columns = [
    "emotion",
    "count"
]

pie_fig = px.pie(

    emotion_counts,

    names="emotion",

    values="count",

    title="Emotion Distribution"
)

st.plotly_chart(
    pie_fig,
    use_container_width=True
)


st.subheader("Stress Level Distribution")

stress_counts = (
    df["stress_level"]
    .value_counts()
    .reset_index()
)

stress_counts.columns = [
    "stress",
    "count"
]

stress_fig = px.bar(

    stress_counts,

    x="stress",

    y="count",

    title="Stress Level Frequency"
)

st.plotly_chart(
    stress_fig,
    use_container_width=True
)


st.subheader("Cognitive Load Distribution")

cog_counts = (
    df["cognitive_load"]
    .value_counts()
    .reset_index()
)

cog_counts.columns = [
    "load",
    "count"
]

cog_fig = px.bar(

    cog_counts,

    x="load",

    y="count",

    title="Cognitive Load"
)

st.plotly_chart(
    cog_fig,
    use_container_width=True
)


if "fusion_confidence" in df.columns:

    st.subheader("Fusion Confidence Trend")

    fusion_fig = px.line(

        df,

        x=df.index,

        y="fusion_confidence",

        title="Fusion Confidence"
    )

    st.plotly_chart(
        fusion_fig,
        use_container_width=True
    )


st.subheader("Raw Session Data")

st.dataframe(df)


st.subheader("Research Insights")

st.markdown(f"""
- Dominant Emotion:
  **{df['emotion'].mode()[0]}**

- Dominant Temporal Emotion:
  **{df['temporal_emotion'].mode()[0]}**

- Most Common Stress:
  **{df['stress_level'].mode()[0]}**

- Most Common Cognitive Load:
  **{df['cognitive_load'].mode()[0]}**

- Most Common Deception Risk:
  **{df['deception_risk'].mode()[0]}**
""")

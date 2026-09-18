"""
Streamlit Analytics Dashboard for Real-Time Gesture Game Controller.
Run via: streamlit run dashboard.py
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.analytics import TelemetryAnalyzer

# Streamlit Page Config
st.set_page_config(
    page_title="Gesture Control Analytics",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 Gesture Game Controller - ML & Telemetry Analytics")
st.markdown("Real-time performance metrics, gesture distributions, model benchmarking, and latency analytics.")

# Load Telemetry
telemetry_path = "data/telemetry/session_telemetry.csv"
analyzer = TelemetryAnalyzer(telemetry_path)
summary = analyzer.get_summary()

# Sidebar controls
st.sidebar.header("Configuration & Actions")
st.sidebar.info("Click Refresh Data to load the latest telemetry.")

if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

# 1. Top Metrics KPI Cards
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Predictions", f"{summary['total_frames']:,}")
with col2:
    st.metric("Avg Confidence", f"{summary['avg_confidence']*100:.1f}%")
with col3:
    st.metric("Avg Latency", f"{summary['avg_latency_ms']:.2f} ms")
with col4:
    st.metric("Avg FPS", f"{summary['avg_fps']:.1f}")
with col5:
    st.metric("Uncertainty Rate", f"{summary['uncertainty_rate_pct']:.1f}%")

st.markdown("---")

# 2. Main Analytics Content
if not summary["file_exists"] or summary["total_frames"] == 0:
    st.warning("⚠️ No active telemetry session recorded yet. Run `python app.py` to generate real-time telemetry data.")
else:
    tab1, tab2, tab3 = st.tabs(["📊 Session Telemetry", "🤖 Model Comparison", "📈 Transition Analytics"])

    with tab1:
        st.subheader("Gesture Usage Distribution & Confidence")
        c1, c2 = st.columns(2)

        with c1:
            counts = summary["gesture_counts"]
            df_counts = pd.DataFrame({"Gesture": list(counts.keys()), "Count": list(counts.values())})
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=df_counts, x="Gesture", y="Count", palette="viridis", ax=ax)
            ax.set_title("Gesture Prediction Frequency")
            st.pyplot(fig)

        with c2:
            df_tel = pd.read_csv(telemetry_path)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df_tel["confidence"], kde=True, color="teal", ax=ax)
            ax.set_title("Confidence Score Distribution")
            ax.set_xlabel("Confidence")
            st.pyplot(fig)

        st.subheader("Inference Latency Timeline")
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(df_tel["latency_ms"].tail(200), color="orange", label="Inference Latency (ms)")
        ax.set_title("Recent 200 Frame Inference Latency (ms)")
        ax.set_ylabel("Latency (ms)")
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)

    with tab2:
        st.subheader("Trained Model Comparison")
        comp_path = "reports/model_comparison.csv"
        cm_path = "reports/confusion_matrix.png"

        if os.path.exists(comp_path):
            df_comp = pd.read_csv(comp_path)
            st.dataframe(df_comp.style.highlight_max(axis=0, subset=["Weighted_F1", "Accuracy"]), use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(data=df_comp, x="Model", y="Weighted_F1", palette="Blues_d", ax=ax)
                ax.set_title("Model Weighted F1-Scores")
                ax.set_ylim(0, 1.05)
                st.pyplot(fig)

            with c2:
                if os.path.exists(cm_path):
                    st.image(cm_path, caption="Best Model Confusion Matrix", use_container_width=True)
        else:
            st.info("Run `python train.py` to generate model comparison benchmarks.")

    with tab3:
        st.subheader("Gesture Transition Statistics")
        trans = summary["transition_matrix"]
        if trans:
            df_trans = pd.DataFrame({"Transition": list(trans.keys()), "Count": list(trans.values())}).sort_values("Count", ascending=False)
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(data=df_trans.head(10), x="Transition", y="Count", palette="rocket", ax=ax)
            plt.xticks(rotation=30)
            ax.set_title("Top 10 Gesture Transitions (State Switching)")
            st.pyplot(fig)
        else:
            st.info("No gesture state transitions recorded yet.")

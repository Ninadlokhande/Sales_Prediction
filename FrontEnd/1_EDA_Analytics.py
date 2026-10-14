import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend_ and_MODEL')))
from model_backend import load_and_train_model

st.set_page_config(page_title="EDA & Analytics", page_icon="🔍", layout="wide")

df, _, _, _, _, _, _ = load_and_train_model()

st.title("🔍 Exploratory Data Analysis (EDA)")
st.markdown("Analyze raw advertising metrics, statistical summaries, and channel correlations.")

with st.expander("📄 View Raw Dataset (First 10 Rows)", expanded=True):
    st.dataframe(df.head(10), use_container_width=True)

with st.expander("📊 Statistical Summary"):
    st.write(df.describe())

st.markdown("---")
st.subheader("Correlation & Channel Distributions")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df.corr(), annot=True, cmap="Blues", fmt=".2f", ax=ax)
    st.pyplot(fig)

with col2:
    st.markdown("#### Spending vs. Sales Scatter Plot")
    channel = st.selectbox("Select Advertising Channel", ["TV", "Radio", "Newspaper"])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df, x=channel, y="Sales", color="purple", ax=ax)
    ax.set_title(f"{channel} Budget vs Sales")
    st.pyplot(fig)
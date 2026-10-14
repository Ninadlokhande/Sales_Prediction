import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend_ and_MODEL')))
from model_backend import load_and_train_model, predict_sales

st.set_page_config(page_title="Prediction Simulator", page_icon="⚡", layout="wide")

_, model, _, _, _, _, _ = load_and_train_model()

st.title("⚡ Live Sales Prediction Simulator")
st.markdown("Adjust the advertising budgets below to simulate and forecast sales output in real-time.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Campaign Budgets")
    tv_input = st.slider("TV Advertising Budget ($)", 0.0, 350.0, 150.0, 1.0)
    radio_input = st.slider("Radio Advertising Budget ($)", 0.0, 100.0, 25.0, 1.0)
    newspaper_input = st.slider("Newspaper Advertising Budget ($)", 0.0, 150.0, 30.0, 1.0)

with col2:
    st.subheader("Projected Business Outcome")
    predicted_sales = predict_sales(model, tv_input, radio_input, newspaper_input)
    
    st.metric(label="Predicted Sales Volume", value=f"{predicted_sales:.2f} units")
    
    st.markdown("#### Budget Distribution Breakdown")
    budget_df = pd.DataFrame({
        'Channel': ['TV', 'Radio', 'Newspaper'],
        'Budget': [tv_input, radio_input, newspaper_input]
    })
    st.bar_chart(budget_df.set_index('Channel'))
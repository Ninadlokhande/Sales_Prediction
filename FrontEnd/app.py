import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

# Add backend directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend_ and_MODEL')))
from model_backend import load_and_train_model, predict_sales

st.set_page_config(
    page_title="Sales Forecasting & Marketing Analytics",
    page_icon="📈",
    layout="wide"
)

# Load backend data and model
df, model, X_test, y_test, predictions, r2, mae = load_and_train_model()

# Custom Styling header
st.title("📊 Multi-Channel Marketing Analytics & Sales Prediction Engine")
st.markdown("---")

# Create Tabs for Seamless In-Page Navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Overview", 
    "🔍 EDA & Analytics", 
    "⚡ Prediction Simulator", 
    "💡 Business ROI & Insights"
])

# --- TAB 1: OVERVIEW ---
with tab1:
    st.subheader("Welcome to the Marketing Optimization & Analytics Platform!")
    st.write("This platform bridges data science and business intelligence to help organizations analyze multi-channel advertising efficiency and forecast retail sales.")
    
    st.markdown("### 🚀 Platform Modules:")
    st.markdown("""
    * **Exploratory Data Analysis (EDA):** Deep dive into dataset distributions, channel spending patterns, and correlation matrices.
    * **Interactive Prediction Simulator:** Real-time calculation engine letting stakeholders test custom ad budgets and view expected sales output instantly.
    * **Business ROI & Recommendations:** Comprehensive regression metrics, coefficient evaluations, and data-driven budget reallocation strategies.
    """)
   
# --- TAB 2: EDA & ANALYTICS ---
with tab2:
    st.subheader("🔍 Exploratory Data Analysis (EDA)")
    
    with st.expander("📄 View Raw Dataset (First 10 Rows)", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)

    with st.expander("📊 Statistical Summary"):
        st.write(df.describe())

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df.corr(), annot=True, cmap="Greens", fmt=".2f", ax=ax)
        st.pyplot(fig)

    with col2:
        st.markdown("#### Spending vs. Sales Scatter Plot")
        channel = st.selectbox("Select Advertising Channel", ["TV", "Radio", "Newspaper"])
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df, x=channel, y="Sales", color="#00A86B", ax=ax)
        ax.set_title(f"{channel} Budget vs Sales")
        st.pyplot(fig)

# --- TAB 3: PREDICTION TOOL ---
with tab3:
    st.subheader("⚡ Live Sales Prediction Simulator")
    st.write("Adjust the advertising budgets below to simulate and forecast sales output in real-time.")

    col1, col2 = st.columns([1, 1])

    with col1:
        tv_input = st.slider("TV Advertising Budget ($)", 0.0, 350.0, 150.0, 1.0)
        radio_input = st.slider("Radio Advertising Budget ($)", 0.0, 100.0, 25.0, 1.0)
        newspaper_input = st.slider("Newspaper Advertising Budget ($)", 0.0, 150.0, 30.0, 1.0)

    with col2:
        predicted_sales = predict_sales(model, tv_input, radio_input, newspaper_input)
        st.metric(label="Predicted Sales Volume", value=f"{predicted_sales:.2f} units")
        
        st.markdown("#### Budget Distribution Breakdown")
        budget_df = pd.DataFrame({
            'Channel': ['TV', 'Radio', 'Newspaper'],
            'Budget': [tv_input, radio_input, newspaper_input]
        })
        st.bar_chart(budget_df.set_index('Channel'))

# --- TAB 4: BUSINESS ROI ---
with tab4:
    st.subheader("💡 Model Performance & Strategic Business Insights")
    
    col1, col2 = st.columns(2)
    col1.metric("Model Accuracy ($R^2$ Score)", f"{r2:.2f}", help="Explains 90% of variance in sales.")
    col2.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Actual vs Predicted Fit")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(y_test, predictions, color='#00A86B', alpha=0.7)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax.set_xlabel("Actual Sales")
        ax.set_ylabel("Predicted Sales")
        st.pyplot(fig)

    with col_right:
        st.subheader("Channel Impact (Coefficients)")
        coef_df = pd.DataFrame(model.coef_, index=['TV', 'Radio', 'Newspaper'], columns=['Coefficient'])
        st.dataframe(coef_df, use_container_width=True)

    st.markdown("---")
    st.success("""
    * **Radio Yield Efficiency:** Radio advertising demonstrates the highest coefficient return per unit spent, indicating optimal marginal returns.
    * **Print Optimization:** Newspaper advertising shows a negligible impact coefficient (~0.002), pointing to budget inefficiencies.
    * **Actionable Strategy:** Reallocate underperforming print budgets toward digital and radio channels to maximize overall revenue.
    """)
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend_ and_MODEL')))
from model_backend import load_and_train_model

st.set_page_config(page_title="Business ROI & Insights", page_icon="💡", layout="wide")

df, model, X_test, y_test, predictions, r2, mae = load_and_train_model()

st.title("💡 Model Performance & Strategic Business Insights")
st.markdown("Evaluating mathematical reliability and translating regression coefficients into business strategy.")

col1, col2 = st.columns(2)
col1.metric("Model Accuracy ($R^2$ Score)", f"{r2:.2f}", help="Explains 90% of variance in sales.")
col2.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")

st.markdown("---")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Actual vs Predicted Fit")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, predictions, color='teal', alpha=0.7)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
    st.pyplot(fig)

with col_right:
    st.subheader("Channel Impact (Coefficients)")
    coef_df = pd.DataFrame(model.coef_, index=['TV', 'Radio', 'Newspaper'], columns=['Coefficient'])
    st.dataframe(coef_df, use_container_width=True)

st.markdown("---")
st.subheader("🎯 Executive Recommendations & Strategic Takeaways")
st.success("""
* **Radio Yield Efficiency:** Radio advertising demonstrates the highest coefficient return per unit spent, indicating optimal marginal returns.
* **Print Optimization:** Newspaper advertising shows a negligible impact coefficient (~0.002), pointing to budget inefficiencies.
* **Actionable Strategy:** Reallocate underperforming print budgets toward digital and radio channels to maximize overall revenue.
""")
import streamlit as st
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

st.set_page_config(layout="wide", page_title="Model Performance")
st.title("📈 Model Performance Evaluation")

if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
    st.warning("Please run the analysis on the main page first to evaluate the model.", icon="👈")
    st.stop()

df_final = st.session_state.processed_data

# ✅ Check if required columns exist
required_cols = ['last_price', 'Predicted Value']
if not all(col in df_final.columns for col in required_cols):
    st.error(f"Required columns for performance evaluation are missing. Please re-run the analysis.")
    st.stop()

y_true = df_final['last_price']
y_pred = df_final['Predicted Value']

# --- Calculate Metrics ---
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

st.header("Key Performance Indicators (KPIs)")
st.info("Metrics compare the model's predictions against the most recent Zillow Home Value Index ('last_price') for the corresponding areas.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("R-squared (R²)")
    st.metric(label="Coefficient of Determination", value=f"{r2:.4f}")
    st.markdown("Indicates how much of the variance in asset values is explained by the model. Higher is better.")
with col2:
    st.subheader("Mean Absolute Error (MAE)")
    st.metric(label="Average Prediction Error", value=f"${mae:,.0f}")
    st.markdown("The average absolute difference between the predicted and actual values.")

col3, col4 = st.columns(2)
with col3:
    st.subheader("Root Mean Squared Error (RMSE)")
    st.metric(label="Standard Deviation of Residuals", value=f"${rmse:,.0f}")
    st.markdown("Measures the standard deviation of prediction errors. It is sensitive to large errors.")

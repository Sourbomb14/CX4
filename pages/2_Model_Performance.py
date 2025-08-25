import streamlit as st
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

st.set_page_config(layout="wide", page_title="Model Performance")

st.title("📈 Model Performance Evaluation")
st.markdown("Metrics for the Gradient Boosting Regressor model used for predictions.")

if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
    st.warning("Please run the analysis on the main page first to evaluate the model.")
    st.stop()

df_final = st.session_state.processed_data

# Note: Since we don't have true values, we'll use a placeholder or split the data
# For this dashboard, we'll assume the 'last_price' from Zillow is our "true" value for comparison
y_true = df_final['last_price']
y_pred = df_final['Predicted Value']

# --- Calculate Metrics ---
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

st.header("Key Performance Indicators (KPIs)")
st.info("These metrics compare the model's predicted values against the most recent Zillow Home Value Index ('last_price') for the corresponding areas.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("R-squared (R²)")
    st.metric(label="Coefficient of Determination", value=f"{r2:.4f}")
    st.write("This indicates the proportion of the variance in the asset values that is predictable from the features. A higher value is better.")

with col2:
    st.subheader("Mean Absolute Error (MAE)")
    st.metric(label="Average Prediction Error", value=f"${mae:,.0f}")
    st.write("This is the average absolute difference between the predicted values and the actual values. It gives an idea of the magnitude of the error.")

col3, col4 = st.columns(2)
with col3:
    st.subheader("Root Mean Squared Error (RMSE)")
    st.metric(label="Standard Deviation of Residuals", value=f"${rmse:,.0f}")
    st.write("This measures the standard deviation of the prediction errors. It is more sensitive to large errors than MAE.")

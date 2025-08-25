import streamlit as st
import pandas as pd
import numpy as np
import gdown
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import streamlit.components.v1 as components
import re

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Government Asset Valuation Dashboard",
    page_icon="🏛️"
)

# --- Global Variables & Constants ---
RANDOM_STATE = 4742271
NUMERIC_FEATURE_COLS = [
    'mean_price', 'median_price', 'std_price', 'price_min', 'price_max', 'price_range',
    'price_volatility', 'recent_6mo_avg', 'recent_12mo_avg', 'last_price', 'price_trend_slope'
]

# --- Caching Functions for Efficiency ---
@st.cache_data
def load_data():
    """Downloads and caches the Zillow and Assets datasets."""
    zillow_url = "https://drive.google.com/uc?id=1fFT8Q8GWiIEM7kx6czhQ-qabygUPBQRv"
    assets_url = "https://drive.google.com/uc?id=1YFTWJNoxu0BF8UlMDXI8bXwRTVQNE2mb"
    zillow_path = "zillow_housing_index.csv"
    assets_path = "us_government_assets.csv"
    
    for path, url in [(zillow_path, zillow_url), (assets_path, assets_url)]:
        if not os.path.exists(path):
            gdown.download(url, path, quiet=True)
            
    df_zillow_raw = pd.read_csv(zillow_path)
    df_assets_raw = pd.read_csv(assets_path)
    return df_zillow_raw, df_assets_raw

@st.cache_resource
def load_models_and_scalers():
    """Downloads and caches all models and scalers from Google Drive."""
    file_ids = {
        "scaler_all.pkl": "1G3U898UQ4yoWO5TOY01MEDlnprG0bEM6",
        "cluster_1_model.pkl": "13Z7PaHcb9e9tOYXxB7fjWKgrb8rpB3xb",
        "cluster_0_model.pkl": "1JM1tj9PNQ8TEJlR3S0MQTxguLsoXKbcf",
        "global_model.pkl": "1ZWPra5iZ0pEVQgxpPaWx8gX3J9olsb7Z",
        "scaler_last_price.pkl": "1nhoS237W_-5Fsgdo7sDFD5_7hceHappp",
    }
    for filename, file_id in file_ids.items():
        if not os.path.exists(filename):
            gdown.download(f'https://drive.google.com/uc?id={file_id}', filename, quiet=True)
    
    models = {
        'scalers': {
            'all': pickle.load(open("scaler_all.pkl", "rb")),
            'last': pickle.load(open("scaler_last_price.pkl", "rb"))
        },
        'global_model': pickle.load(open("global_model.pkl", "rb"))
    }
    return models

# --- Data Processing Functions ---
@st.cache_data
def process_and_predict(_df_zillow_raw, _df_assets_raw, _models):
    """A single function to handle all data processing and prediction steps."""
    try:
        # 1. Feature Engineering for Zillow Data (on a sample for efficiency)
        df_z = _df_zillow_raw.sample(n=10000, random_state=RANDOM_STATE)
        
        # ✅ FIXED: More robust way to find date columns using a regular expression
        date_pattern = re.compile(r'^\d{4}-\d{2}') # Matches 'YYYY-MM'
        date_cols = [c for c in df_z.columns if date_pattern.match(c)]

        # ✅ FIXED: Add a check to ensure date columns were actually found
        if not date_cols:
            st.error("Data Processing Error: No date-like columns (e.g., '2020-01') were found in the Zillow dataset. Cannot proceed.")
            return None

        df_z_numeric = df_z[date_cols].apply(pd.to_numeric, errors='coerce')
        
        imputer = KNNImputer(n_neighbors=5)
        df_z[date_cols] = imputer.fit_transform(df_z_numeric)
        
        features = []
        for _, row in df_z.iterrows():
            prices = row[date_cols].values
            features.append({
                'City': str(row.get('City', '')).upper().strip(), 'State': str(row.get('State', '')).upper().strip(),
                'mean_price': np.mean(prices), 'median_price': np.median(prices), 'std_price': np.std(prices),
                'price_min': np.min(prices), 'price_max': np.max(prices), 'price_range': np.ptp(prices),
                'price_volatility': np.std(prices) / np.mean(prices) if np.mean(prices) else 0,
                'recent_6mo_avg': np.mean(prices[-6:]), 'recent_12mo_avg': np.mean(prices[-12:]), 'last_price': prices[-1],
                'price_trend_slope': LinearRegression().fit(np.arange(len(prices)).reshape(-1, 1), prices).coef_[0]
            })
        df_z_features = pd.DataFrame(features)

        # 2. Enrich Asset Data
        df_enriched = pd.merge(_df_assets_raw, df_z_features, how='left', on=['City', 'State'])
        for col in NUMERIC_FEATURE_COLS:
            if df_enriched[col].isnull().any():
                df_enriched[col] = df_enriched[col].fillna(df_z_features[col].median())
        
        # 3. Scale Features and Predict
        X_scaled = _models['scalers']['all'].transform(df_enriched[NUMERIC_FEATURE_COLS])
        pred_scaled = _models['global_model'].predict(X_scaled)
        pred_original = _models['scalers']['last'].inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        
        df_final = df_enriched.copy()
        df_final['Predicted Value'] = pred_original
        
        return df_final
    except Exception as e:
        st.error(f"A data processing error occurred: {e}. This may be due to unexpected data formats in the source files.")
        return None

# --- Main App ---
st.title("🏛️ Government Asset Valuation Dashboard")
st.markdown("An analytical tool to assess, predict, and visualize the value of government real estate assets.")

with st.spinner("Loading data and predictive models..."):
    df_zillow_raw, df_assets_raw = load_data()
    models = load_models_and_scalers()

if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

if st.button("🚀 Analyze and Predict Asset Values"):
    with st.spinner("Performing nationwide analysis... This may take a moment."):
        st.session_state.processed_data = process_and_predict(df_zillow_raw, df_assets_raw, models)

if st.session_state.processed_data is not None:
    df_final = st.session_state.processed_data
    st.header("National Asset Portfolio Overview")
    total_value = df_final['Predicted Value'].sum()
    asset_count = len(df_final)
    avg_value = df_final['Predicted Value'].mean()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Predicted Value", f"${total_value:,.0f}")
    col2.metric("Total Number of Assets", f"{asset_count:,}")
    col3.metric("Average Asset Value", f"${avg_value:,.0f}")
    st.markdown("---")
    st.header("Interactive Asset Map")
    try:
        with open("Clustered_Asset_Map.html", "r", encoding='utf-8') as f:
            map_html = f.read()
        components.html(map_html, height=600, scrolling=True)
    except FileNotFoundError:
        st.error("Error: `Clustered_Asset_Map.html` not found.")
    st.markdown("---")
    st.header("Data Preview")
    st.dataframe(df_final[['Real Property Asset Name', 'City', 'State', 'Predicted Value']].head(10))
else:
    st.info("Click the button above to begin the analysis and generate predictions.")

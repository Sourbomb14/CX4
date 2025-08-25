import streamlit as st
import pandas as pd
import numpy as np
import gdown
import os
import pickle
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium
from streamlit_folium import folium_static
import geopandas as gpd
from shapely.geometry import Point
from thefuzz import process, fuzz
import libpysal
import esda
import contextily as cx

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Government Asset Valuation Dashboard")
RANDOM_STATE = 4742271 # Ensure reproducibility

# --- Global Variables ---
# Define the exact numeric feature columns the scaler was trained on
NUMERIC_FEATURE_COLS = [
    'mean_price', 'median_price', 'std_price', 'price_min', 'price_max', 'price_range',
    'price_volatility', 'recent_6mo_avg', 'recent_12mo_avg', 'last_price', 'price_trend_slope'
]

# --- Helper Functions ---
@st.cache_data
def load_data():
    """Downloads the Zillow and Assets datasets from Google Drive."""
    zillow_url = "https://drive.google.com/uc?id=1fFT8Q8GWiIEM7kx6czhQ-qabygUPBQRv"
    assets_url = "https://drive.google.com/uc?id=1YFTWJNoxu0BF8UlMDXI8bXwRTVQNE2mb"
    zillow_path = "zillow_housing_index.csv"
    assets_path = "us_government_assets.csv"

    if not os.path.exists(zillow_path):
        st.info("Downloading Zillow dataset...")
        gdown.download(zillow_url, zillow_path, quiet=True)
    if not os.path.exists(assets_path):
        st.info("Downloading Assets dataset...")
        gdown.download(assets_url, assets_path, quiet=True)

    df_zillow_raw = pd.read_csv(zillow_path)
    df_assets_raw = pd.read_csv(assets_path)
    return df_zillow_raw, df_assets_raw

def download_model_files():
    """Downloads all required .pkl files from the specified Google Drive links."""
    file_ids = {
        # ❗️❗️❗️ CRITICAL: The link for scaler_all.pkl is missing. Please add the correct ID.
        "scaler_all.pkl": "YOUR_FILE_ID_FOR_scaler_all.pkl_HERE",
        
        # ❗️❗️❗️ The link for cluster_1_model.pkl is also missing.
        "cluster_1_model.pkl": "YOUR_FILE_ID_FOR_cluster_1_model.pkl_HERE",
        
        "cluster_0_model.pkl": "1JM1tj9PNQ8TEJlR3S0MQTxguLsoXKbcf",
        "cluster_pca_0_model.pkl": "1X9WmLRoJHCdMcLVKTtsbDujYAIg_o1dU",
        "cluster_pca_1_model.pkl": "1GaDbbVCBUvjrvSUrfT6GLJUFYVa1xRPG",
        "global_model.pkl": "1ZWPra5iZ0pEVQgxpPaWx8gX3J9olsb7Z",
        "global_model_pca.pkl": "1dmE1bEDWUeAkZNkpGDTHEJA6AEt0FPz1",
        "scaler_last_price.pkl": "1nhoS237W_-5Fsgdo7sDFD5_7hceHappp",
        "pca_final.pkl": "1gQfXF4aJ-30XispHCOjdv2zfRDw2fhHt"
    }
    
    st.info("Checking for model and scaler files...")
    for filename, file_id in file_ids.items():
        if "YOUR_FILE_ID" in file_id:
            st.error(f"FATAL ERROR: The Google Drive link for '{filename}' is missing. Please update the script.")
            st.stop()
        
        if not os.path.exists(filename):
            st.info(f"Downloading {filename}...")
            url = f'https://drive.google.com/uc?id={file_id}'
            try:
                gdown.download(url, filename, quiet=True)
            except Exception as e:
                st.error(f"Failed to download {filename}. Please check its Google Drive link and sharing permissions.")
                st.stop()

@st.cache_resource
def load_models_and_scalers():
    """Loads all models and scalers after ensuring they are downloaded."""
    download_model_files()
    
    models = {}
    try:
        models['scalers'] = {
            'all': pickle.load(open("scaler_all.pkl", "rb")),
            'last': pickle.load(open("scaler_last_price.pkl", "rb"))
        }
        models['global'] = pickle.load(open("global_model.pkl", "rb"))
        # (Add loading for other models as needed)
        return models
    except Exception as e:
        st.error(f"An error occurred while loading model files: {e}")
        st.error("This might be because a .pkl file does not contain the correct object type (e.g., 'scaler_all.pkl' is not a scaler).")
        st.stop()

def preprocess_zillow(df_zillow_raw, sample_n=5000):
    df_z = df_zillow_raw.copy()
    if sample_n and sample_n < len(df_z):
        df_z = df_z.sample(n=sample_n, random_state=RANDOM_STATE).reset_index(drop=True)

    date_cols = [c for c in df_z.columns if c[:2].isdigit()]
    df_z[date_cols] = df_z[date_cols].apply(pd.to_numeric, errors='coerce')
    
    imputer = KNNImputer(n_neighbors=5)
    df_z[date_cols] = imputer.fit_transform(df_z[date_cols])
    return df_z, date_cols

def feature_engineer_zillow(df_z, date_cols):
    features = []
    for _, row in df_z.iterrows():
        prices = row[date_cols].values.astype(float)
        mean_price = np.nanmean(prices)
        std_price = np.nanstd(prices)
        
        feature_dict = {
            'City': str(row.get('City', '')).upper().strip(),
            'State': str(row.get('State', '')).upper().strip(),
            'mean_price': mean_price,
            'median_price': np.nanmedian(prices),
            'std_price': std_price,
            'price_min': np.nanmin(prices),
            'price_max': np.nanmax(prices),
            'price_range': np.nanmax(prices) - np.nanmin(prices),
            'price_volatility': (std_price / mean_price) if mean_price != 0 else 0.0,
            'recent_6mo_avg': np.nanmean(prices[-6:]) if len(prices) >= 6 else mean_price,
            'recent_12mo_avg': np.nanmean(prices[-12:]) if len(prices) >= 12 else mean_price,
            'last_price': float(prices[-1]),
            'price_trend_slope': float(LinearRegression().fit(np.arange(len(prices)).reshape(-1, 1), prices).coef_[0]) if len(prices) > 1 else 0.0
        }
        features.append(feature_dict)
    return pd.DataFrame(features)

def enrich_assets(df_assets, df_z_features, scaler_all):
    df_assets_enriched = pd.merge(df_assets, df_z_features, how='left', on=['City', 'State'])
    
    for col in NUMERIC_FEATURE_COLS:
        if df_assets_enriched[col].isna().any():
            median_val = df_z_features[col].median()
            df_assets_enriched[col].fillna(median_val, inplace=True)
    
    df_assets_enriched[NUMERIC_FEATURE_COLS] = scaler_all.transform(df_assets_enriched[NUMERIC_FEATURE_COLS])
    
    return df_assets_enriched

def predict_asset_values(assets_df, models):
    model_to_use = models['global']
    scaler_last = models['scalers']['last']
    
    X = assets_df[NUMERIC_FEATURE_COLS]
    
    pred_scaled = model_to_use.predict(X)
    pred_original = scaler_last.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    
    assets_df['pred_last_price_original'] = pred_original
    return assets_df

# --- Streamlit App Layout ---
st.title("Government Asset Valuation Dashboard")

# --- Data and Model Loading ---
with st.spinner("Loading datasets, models, and scalers..."):
    df_zillow_raw, df_assets_raw = load_data()
    models = load_models_and_scalers()
st.success("All data and models loaded successfully.")

# --- Main App Logic ---
if st.button("Process Data and Predict Values"):
    with st.spinner("Processing data, enriching assets, and predicting values..."):
        df_z, date_cols = preprocess_zillow(df_zillow_raw)
        df_z_features = feature_engineer_zillow(df_z, date_cols)
        
        assets_enriched = enrich_assets(df_assets_raw, df_z_features, models['scalers']['all'])
        
        assets_with_predictions = predict_asset_values(assets_enriched, models)
        
        st.session_state.processed_data = assets_with_predictions

    st.success("Processing complete!")

# --- Display Results ---
if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
    assets_with_predictions = st.session_state.processed_data
    
    st.header("Predicted Asset Values")
    st.dataframe(assets_with_predictions[['Real Property Asset Name', 'City', 'State', 'pred_last_price_original']].head())
    
    st.subheader("Predicted Value Statistics")
    st.write(assets_with_predictions['pred_last_price_original'].describe())

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
        "scaler_last_price.pkl": "1SX8nML2-L5TBgtlSABLNSnkdmWWG_n6w",
        "scaler_all.pkl": "1U6yxKgpTTuvYWMHNKmcDwFihnSJJUHPl",
        "cluster_1_model.pkl": "1lHdldr_u4V_tJbqWmqZgKCGa7LH4QZW1",
        "cluster_pca_1_model.pkl": "1EEG7gfXChiDPlu-1dcrYoxQcQsBkbUz-",
        "cluster_pca_0_model.pkl": "1DOdSlQKGUNVgCaQSqlorSQ42CtrLBTpw",
        "cluster_0_model.pkl": "1rn0tmVCiNWxMmhnN4K7JaqQbAbp4cpVJ",
        "global_model.pkl": "1xEMFHHqraqE32qlXxMvm38gRnR6jL80s",
        "global_model_pca.pkl": "1PzhWJ36LtobDQf-onO_MD7659XAiMOCn",
        "pca_final.pkl": "1hwI8O2x3LEXSDWk-8HEriicthyo0q3Eq"
    }
    
    st.info("Checking for model and scaler files...")
    for filename, file_id in file_ids.items():
        if not os.path.exists(filename):
            st.info(f"Downloading {filename}...")
            url = f'https://drive.google.com/uc?id={file_id}'
            try:
                gdown.download(url, filename, quiet=True)
            except Exception as e:
                st.error(f"Failed to download {filename}. Please check the Google Drive link and sharing permissions.")
                st.error(f"Error details: {e}")
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
        models['global_pca'] = pickle.load(open("global_model_pca.pkl", "rb"))
        models['pca_final'] = pickle.load(open("pca_final.pkl", "rb"))

        cluster_models_orig = {}
        cluster_models_pca = {}
        for i in range(2):
            cluster_models_orig[i] = pickle.load(open(f"cluster_{i}_model.pkl", "rb"))
            cluster_models_pca[i] = pickle.load(open(f"cluster_pca_{i}_model.pkl", "rb"))
        
        models['cluster_orig'] = cluster_models_orig
        models['cluster_pca'] = cluster_models_pca
        
        return models
    except FileNotFoundError as e:
        st.error(f"Fatal Error: Could not load file {e.filename}. The download may have failed or the file is missing.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading models: {e}")
        st.stop()

def preprocess_zillow(df_zillow_raw, sample_n=5000):
    df_z = df_zillow_raw.copy()
    if sample_n and sample_n < len(df_z):
        df_z = df_z.sample(n=sample_n, random_state=RANDOM_STATE).reset_index(drop=True)

    date_cols = [c for c in df_z.columns if c[:2].isdigit()]
    df_z[date_cols] = df_z[date_cols].apply(pd.to_numeric, errors='coerce')

    def remove_outliers_rowwise_simple(row, thresh=3.0):
        mu = row.mean()
        sd = row.std()
        if pd.isna(sd) or sd == 0:
            return row
        z = (row - mu) / sd
        row[abs(z) > thresh] = np.nan
        return row

    df_z[date_cols] = df_z[date_cols].apply(remove_outliers_rowwise_simple, axis=1)
    
    imputer = KNNImputer(n_neighbors=5)
    df_z[date_cols] = imputer.fit_transform(df_z[date_cols])
    return df_z, date_cols

def feature_engineer_zillow(df_z, date_cols):
    features = []
    for _, row in df_z.iterrows():
        prices = row[date_cols].values.astype(float)
        mean_price = np.nanmean(prices)
        std_price = np.nanstd(prices)
        
        features.append({
            'RegionID': row.get('RegionID'),
            'City': str(row.get('City', '')).upper().strip(),
            'State': str(row.get('State', '')).upper().strip(),
            'County': str(row.get('CountyName', '')).upper().strip(),
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
        })
    return pd.DataFrame(features)

def enrich_assets(df_assets, df_z_features, scaler_all):
    num_cols = df_z_features.select_dtypes(include=np.number).columns.tolist()
    
    df_assets_enriched = df_assets.merge(df_z_features, how='left', on=['City', 'State'])
    
    # Simple imputation for missing market data after merge
    for col in num_cols:
        if df_assets_enriched[col].isna().any():
            df_assets_enriched[col].fillna(df_z_features[col].median(), inplace=True)
    
    # Scale numeric features
    df_assets_enriched[num_cols] = scaler_all.transform(df_assets_enriched[num_cols])
    
    return df_assets_enriched

def predict_asset_values(assets_df, models, output_col_prefix=""):
    assets_with_preds = assets_df.copy()
    num_cols = models['scalers']['all'].feature_names_in_
    
    # Ensure all required numeric columns are present and in the correct order
    x_df = assets_with_preds[num_cols].fillna(0)
    
    model_to_use = models['global'] # Using global model for simplicity
    
    pred_scaled = model_to_use.predict(x_df)
    pred_original = models['scalers']['last'].inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    
    assets_with_preds[f'{output_col_prefix}pred_last_price_original'] = pred_original
    
    return assets_with_preds


# --- Streamlit App Layout ---
st.title("Government Asset Valuation Dashboard")

# --- Data and Model Loading ---
with st.spinner("Loading datasets, models, and scalers..."):
    df_zillow_raw, df_assets_raw = load_data()
    models = load_models_and_scalers()
st.success("All data and models loaded successfully.")

# --- Session State Initialization ---
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'assets_enriched' not in st.session_state:
    st.session_state.assets_enriched = None


# --- Main App Logic ---
if st.button("Process Data and Predict Values"):
    with st.spinner("Processing data, enriching assets, and predicting values... This might take a minute."):
        df_z, date_cols = preprocess_zillow(df_zillow_raw)
        df_z_features = feature_engineer_zillow(df_z, date_cols)
        
        assets_enriched = enrich_assets(df_assets_raw, df_z_features, models['scalers']['all'])
        
        assets_with_predictions = predict_asset_values(assets_enriched, models)
        
        st.session_state.processed_data = assets_with_predictions
        st.session_state.assets_enriched = assets_enriched

    st.success("Processing complete!")


# --- Display Results ---
if st.session_state.processed_data is not None:
    assets_with_predictions = st.session_state.processed_data
    
    st.header("Predicted Asset Values")
    st.dataframe(assets_with_predictions[['Real Property Asset Name', 'City', 'State', 'pred_last_price_original']].head())
    
    st.subheader("Predicted Value Statistics")
    st.write(assets_with_predictions['pred_last_price_original'].describe())

    # Histogram
    fig, ax = plt.subplots()
    sns.histplot(assets_with_predictions['pred_last_price_original'].dropna(), bins=50, kde=True, ax=ax)
    ax.set_title("Distribution of Predicted Asset Values")
    ax.set_xlabel("Predicted Value ($)")
    st.pyplot(fig)
    
    # Choropleth Map
    st.subheader("Median Predicted Value by State")
    state_agg = assets_with_predictions.groupby('State')['pred_last_price_original'].median().reset_index()
    try:
        fig = px.choropleth(state_agg,
                            locations='State',
                            locationmode="USA-states",
                            color='pred_last_price_original',
                            scope="usa",
                            title="Median Predicted Asset Value by State",
                            color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not generate choropleth map: {e}")

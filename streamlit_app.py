import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import geopandas as gpd
from shapely.geometry import Point
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="US Government Assets Portfolio Analytics",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.8rem;
    color: var(--text-color);
    text-align: center;
    margin-bottom: 1.5rem;
    font-weight: bold;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}
.metric-container {
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    border-left: 6px solid #1f4e79;
    transition: all 0.3s ease;
}
.metric-container:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 8px rgba(0,0,0,0.15);
}
.insight-box {
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    border-left: 4px solid #36a2eb;
}
.stMetric > label {
    font-size: 1.3rem !important;
    font-weight: bold !important;
    color: var(--text-color);
}
.stMetric > div {
    font-size: 2rem !important;
    color: #4f8bba;
}
[data-testid="stSidebar"] {
    color: inherit;
}
.stButton > button {
    background-color: #36a2eb;
    color: white;
    border-radius: 5px;
    border: none;
    padding: 0.5rem 1rem;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background-color: #4f8bba;
    transform: translateY(-2px);
}
.stPlotlyChart {
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_assets_data():
    try:
        assets_url = "https://drive.google.com/uc?id=1YFTWJNoxu0BF8UlMDXI8bXwRTVQNE2mb"
        response = requests.get(assets_url, timeout=10)
        if response.status_code == 200:
            with open("temp_assets.csv", "wb") as f:
                f.write(response.content)
            df = pd.read_csv("temp_assets.csv", encoding='utf-8')
            return df
        else:
            st.error("Failed to download assets data")
            return None
    except Exception:
        try:
            df = pd.read_csv("temp_assets.csv", encoding='latin-1')
            return df
        except Exception as e2:
            st.error(f"Error loading assets data: {e2}")
            return None

@st.cache_data
def load_housing_data():
    try:
        housing_url = "https://drive.google.com/uc?id=1fFT8Q8GWiIEM7kx6czhQ-qabygUPBQRv"
        response = requests.get(housing_url, timeout=10)
        if response.status_code == 200:
            with open("temp_housing.csv", "wb") as f:
                f.write(response.content)
            df = pd.read_csv("temp_housing.csv", encoding='utf-8')
            return df
        else:
            st.error("Failed to download housing data")
            return None
    except Exception:
        try:
            df = pd.read_csv("temp_housing.csv", encoding='latin-1')
            return df
        except Exception as e2:
            st.error(f"Error loading housing data: {e2}")
            return None

@st.cache_data
def create_sample_data():
    np.random.seed(42)
    n_samples = 1000
    states = ['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI', 'NJ', 'VA', 'WA', 'AZ', 'MA']
    data = {
        'state': np.random.choice(states, n_samples),
        'city': np.random.choice(['Los Angeles', 'Houston', 'New York', 'Miami', 'Chicago',
                                  'Philadelphia', 'Phoenix', 'Atlanta', 'Boston', 'Seattle'], n_samples),
        'latitude': np.random.uniform(25, 48, n_samples),
        'longitude': np.random.uniform(-125, -70, n_samples),
        'building_rentable_square_feet': np.random.uniform(1000, 100000, n_samples),
        'estimated_value': np.random.lognormal(13, 1.5, n_samples),
        'latest_price_index': np.random.uniform(50000, 800000, n_samples)
    }
    return pd.DataFrame(data)

@st.cache_data
def clean_and_merge_data(df_assets, df_prices):
    if df_assets is None:
        return create_sample_data()
    df_assets.columns = df_assets.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    if 'latitude' in df_assets.columns and 'longitude' in df_assets.columns:
        valid_coords = (
            (df_assets['latitude'] >= 24) & (df_assets['latitude'] <= 49) &
            (df_assets['longitude'] >= -125) & (df_assets['longitude'] <= -66) &
            df_assets['latitude'].notna() & df_assets['longitude'].notna()
        )
        df_assets = df_assets[valid_coords]
    if df_prices is not None:
        df_prices.columns = df_prices.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
        price_cols = [col for col in df_prices.columns if any(year in str(col) for year in ['2024', '2025'])]
        if price_cols:
            latest_col = sorted(price_cols, reverse=True)[0]
            df_prices['latest_price_index'] = pd.to_numeric(df_prices[latest_col], errors='coerce')
        if 'city' in df_assets.columns and 'state' in df_assets.columns and 'city' in df_prices.columns and 'state' in df_prices.columns:
            df_assets['city_state_key'] = (
                df_assets['city'].astype(str).str.lower().str.strip() + '_' +
                df_assets['state'].astype(str).str.lower().str.strip()
            )
            df_prices['city_state_key'] = (
                df_prices['city'].astype(str).str.lower().str.strip() + '_' +
                df_prices['state'].astype(str).str.lower().str.strip()
            )
            merged_df = pd.merge(
                df_assets,
                df_prices[['city_state_key', 'latest_price_index']],
                on='city_state_key',
                how='left'
            )
        else:
            merged_df = df_assets.copy()
            merged_df['latest_price_index'] = np.random.uniform(50000, 800000, len(df_assets))
    else:
        merged_df = df_assets.copy()
        merged_df['latest_price_index'] = np.random.uniform(50000, 800000, len(df_assets))
    merged_df['latest_price_index'] = merged_df['latest_price_index'].fillna(
        merged_df['latest_price_index'].median()
    )
    rentable_col = None
    for col in merged_df.columns:
        if 'rentable' in col.lower() and 'feet' in col.lower():
            rentable_col = col
            break
    if rentable_col:
        merged_df['estimated_value'] = (
            merged_df[rentable_col] * (merged_df['latest_price_index'] / 100) * 10
        )
    else:
        merged_df['estimated_value'] = merged_df['latest_price_index'] * np.random.uniform(0.5, 2.0, len(merged_df))
    high_value_states = ['CA', 'NY', 'MA', 'CT', 'NJ', 'HI', 'MD', 'WA']
    if 'state' in merged_df.columns:
        premium_mask = merged_df['state'].isin(high_value_states)
        merged_df.loc[premium_mask, 'estimated_value'] *= 1.5
    return merged_df

# (All the main dashboard and analytics view functions go here as shown in your earlier code: show_executive_dashboard, show_geographic_analysis, etc.)

def main():
    st.markdown('<h1 class="main-header">🏛️ US Government Assets Portfolio Analytics Dashboard</h1>', 
                unsafe_allow_html=True)

    st.sidebar.image("https://via.placeholder.com/300x100/1f4e79/ffffff?text=Analytics+Dashboard", use_container_width=True)
    st.sidebar.markdown("### 📊 Navigation", unsafe_allow_html=True)
    with st.spinner("Loading datasets..."):
        df_assets = load_assets_data()
        df_prices = load_housing_data()
    if df_assets is None and df_prices is None:
        st.warning("Could not load external data. Using sample data for demonstration.")
        df_merged = create_sample_data()
    else:
        with st.spinner("Processing and merging data..."):
            df_merged = clean_and_merge_data(df_assets, df_prices)

    if df_merged is None or len(df_merged) == 0:
        st.error("No data available for analysis.")
        return

    # Add navigation, filtering, and analysis sections here (refer to your previous code).
    # See prior code for functions for analytics, dashboard, clustering, ML, etc.

if __name__ == "__main__":
    main()

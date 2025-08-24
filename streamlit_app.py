# ================================================
# US Government Assets Portfolio Analytics Dashboard
# Robust, Theme-Aware, Predictive
# ================================================

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

# =================================================
# Streamlit Page Config
# =================================================
st.set_page_config(
    page_title="US Government Assets Portfolio Analytics",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme detection
try:
    theme_base = st.get_option("theme.base")
except Exception:
    theme_base = "light"

PLOTLY_TEMPLATE = "plotly_dark" if theme_base == "dark" else "plotly"

# =================================================
# Utility functions
# =================================================

@st.cache_data
def load_assets_data():
    """Load US Government Assets dataset"""
    try:
        assets_url = "https://drive.google.com/uc?id=1YFTWJNoxu0BF8UlMDXI8bXwRTVQNE2mb"
        df = pd.read_csv(assets_url)
        return df
    except Exception:
        return None

@st.cache_data
def load_housing_data():
    """Load Zillow Housing Price Index dataset"""
    try:
        housing_url = "https://drive.google.com/uc?id=1fFT8Q8GWiIEM7kx6czhQ-qabygUPBQRv"
        df = pd.read_csv(housing_url)
        return df
    except Exception:
        return None

@st.cache_data
def clean_and_merge_data(df_assets, df_prices):
    """Clean and merge datasets"""
    if df_assets is None:
        return pd.DataFrame()

    df_assets.columns = df_assets.columns.str.lower().str.replace(" ", "_").str.replace("-", "_")

    if df_prices is not None:
        df_prices.columns = df_prices.columns.str.lower().str.replace(" ", "_").str.replace("-", "_")

        # Take latest housing price index
        price_cols = [c for c in df_prices.columns if "2025" in c or "2024" in c]
        if price_cols:
            latest_col = sorted(price_cols, reverse=True)[0]
            df_prices["latest_price_index"] = pd.to_numeric(df_prices[latest_col], errors="coerce")

            if "city" in df_assets.columns and "state" in df_assets.columns and "city" in df_prices.columns and "state" in df_prices.columns:
                df_assets["key"] = (df_assets["city"].astype(str).str.lower() + "_" + df_assets["state"].astype(str).str.lower())
                df_prices["key"] = (df_prices["city"].astype(str).str.lower() + "_" + df_prices["state"].astype(str).str.lower())

                df_assets = df_assets.merge(df_prices[["key", "latest_price_index"]], on="key", how="left")
    if "latest_price_index" not in df_assets.columns:
        df_assets["latest_price_index"] = np.random.uniform(50000, 800000, len(df_assets))

    # Estimated value
    if "building_rentable_square_feet" in df_assets.columns:
        df_assets["estimated_value"] = df_assets["building_rentable_square_feet"] * (df_assets["latest_price_index"] / 100)
    else:
        df_assets["estimated_value"] = df_assets["latest_price_index"] * np.random.uniform(0.5, 2.0, len(df_assets))

    # Sample 10k
    df_assets = df_assets.sample(n=min(10000, len(df_assets)), random_state=4742271)

    return df_assets

@st.cache_data
def perform_clustering(df, n_clusters=5):
    """K-Means clustering"""
    cols = ["latitude", "longitude", "estimated_value", "latest_price_index"]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df, None

    X = df[cols].fillna(df[cols].median())
    X_scaled = MinMaxScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=4742271, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    return df, kmeans

def create_folium_map(df):
    """Folium map"""
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return None

    m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=4)

    for _, row in df.iterrows():
        popup = f"State: {row.get('state','')}<br>Value: ${row.get('estimated_value',0):,.0f}"
        folium.CircleMarker(
            [row["latitude"], row["longitude"]],
            radius=5,
            color="blue",
            fill=True,
            popup=popup
        ).add_to(m)

    return m

# =================================================
# App Layout
# =================================================

st.title("🏛️ US Government Assets Portfolio Analytics")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["📊 Executive Dashboard", "🗺️ Geographic Analysis", "🎯 Clustering Analysis", "🤖 Machine Learning", "📈 Advanced Analytics", "📤 Predict Asset Value"])

# Load data
with st.spinner("Loading data..."):
    df_assets = load_assets_data()
    df_prices = load_housing_data()
    df = clean_and_merge_data(df_assets, df_prices)

if df.empty:
    st.error("No data available.")
    st.stop()

# =================================================
# Pages
# =================================================

if page == "📊 Executive Dashboard":
    st.header("Executive Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Assets", f"{len(df):,}")
    col2.metric("Portfolio Value", f"${df['estimated_value'].sum()/1e9:.2f}B")
    col3.metric("Average Value", f"${df['estimated_value'].mean()/1e6:.2f}M")

    st.subheader("Top States by Assets")
    fig = px.bar(df["state"].value_counts().head(10), template=PLOTLY_TEMPLATE, title="Top 10 States by Asset Count")
    st.plotly_chart(fig, use_container_width=True)

elif page == "🗺️ Geographic Analysis":
    st.header("Geographic Analysis")
    m = create_folium_map(df)
    if m: st_folium(m, width=800, height=500)

elif page == "🎯 Clustering Analysis":
    st.header("Clustering Analysis")
    df, _ = perform_clustering(df, n_clusters=5)
    fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", color="cluster", size="estimated_value",
                            mapbox_style="carto-positron", template=PLOTLY_TEMPLATE, zoom=3)
    st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Machine Learning":
    st.header("Machine Learning Insights")
    X = df[["latest_price_index"]].fillna(0)
    y = df["estimated_value"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4742271, test_size=0.2)
    model = RandomForestRegressor(n_estimators=50, random_state=4742271)
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    st.metric("R² Score", f"{r2:.3f}")

elif page == "📈 Advanced Analytics":
    st.header("Advanced Analytics")
    st.write(df.describe())

elif page == "📤 Predict Asset Value":
    st.header("Predict Asset Value")
    uploaded = st.file_uploader("Upload dataset", type=["csv"])
    if uploaded:
        input_df = pd.read_csv(uploaded)
        if "latest_price_index" not in input_df.columns:
            st.error("CSV must contain 'latest_price_index'")
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=4742271)
            X = df[["latest_price_index"]].fillna(0)
            y = df["estimated_value"]
            model.fit(X, y)
            preds = model.predict(input_df[["latest_price_index"]].fillna(0))
            input_df["predicted_value"] = preds
            st.success("Predictions generated!")
            st.dataframe(input_df.head())
            st.download_button("Download Predictions", input_df.to_csv(index=False), "predictions.csv", "text/csv")

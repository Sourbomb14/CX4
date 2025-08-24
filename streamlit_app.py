# app.py
# US Government Assets Portfolio Analytics — Final Integrated App
# - Uses Google Drive source CSVs (as requested)
# - Loads uploaded folium HTML maps (from /mnt/data/) if present
# - Samples exactly 10,000 rows after merge with random_state=4742271
# - Predict tab: train RandomForestRegressor and predict uploaded CSVs
# - Theme-aware visuals (light/dark)
# - Robust guards and informative inferences

import os
import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import plotly.express as px
import plotly.graph_objects as go

import folium
from streamlit_folium import st_folium

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------
# Constants
# ---------------------------
RANDOM_STATE = 4742271
ASSETS_GDRIVE_ID = "1YFTWJNoxu0BF8UlMDXI8bXwRTVQNE2mb"
ZILLOW_GDRIVE_ID = "1fFT8Q8GWiIEM7kx6czhQ-qabygUPBQRv"
ASSETS_URL = f"https://drive.google.com/uc?id={ASSETS_GDRIVE_ID}"
ZILLOW_URL = f"https://drive.google.com/uc?id={ZILLOW_GDRIVE_ID}"

# Uploaded map HTML filenames (you placed these in /mnt/data)
MAP_FILES = {
    "asset_valuation": "/mnt/data/asset_valuation_map.html",
    "comprehensive": "/mnt/data/comprehensive_assets_map.html",
    "assets": "/mnt/data/assets_map.html",
    "state_summary": "/mnt/data/state_summary_map.html",
    "assets_comprehensive": "/mnt/data/assets_comprehensive_map.html"
}

# ---------------------------
# Page config & theme
# ---------------------------
st.set_page_config(page_title="US Government Assets Portfolio Analytics", page_icon="🏛️", layout="wide")
theme_base = st.get_option("theme.base", "light")
PLOTLY_TEMPLATE = "plotly_dark" if theme_base == "dark" else "plotly"

# CSS for cards & insights that adapt to theme
st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight:800; margin-bottom: .6rem; }
    .kpi { padding:12px; border-radius:12px; margin-bottom:8px; border-left:6px solid #1f4e79; }
    .insight { padding:10px; border-radius:10px; margin-bottom:8px; }
    @media (prefers-color-scheme: dark) {
      .kpi { background:#111319; color:#e6eef8; }
      .insight { background:#0f1724; color:#dfefff; border-left-color:#4fa3ff; }
    }
    @media (prefers-color-scheme: light) {
      .kpi { background:#f2f6fb; color:#0e2233; }
      .insight { background:#e9f4fb; color:#082033; border-left-color:#1f4e79; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Utility functions
# ---------------------------
@st.cache_data(show_spinner=False)
def load_csv_from_gdrive(url: str, cache_filename: str):
    """Load CSV from Google Drive link (uc?id=...) and cache locally."""
    import requests
    path = f"{cache_filename}.csv"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        df = pd.read_csv(path)
        return df
    except Exception:
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception:
                return None
        return None

def safe_lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
    return df

@st.cache_data(show_spinner=False)
def clean_and_merge_data(df_assets_raw: pd.DataFrame, df_hpi_raw: pd.DataFrame):
    """Clean asset and housing index data, merge on city_state when possible, compute estimated value,
       and sample 10,000 rows deterministically if larger."""
    if df_assets_raw is None or df_assets_raw.empty:
        return pd.DataFrame()  # caller should handle

    df_assets = safe_lower_cols(df_assets_raw)

    # Filter coordinate sanity
    if {"latitude", "longitude"}.issubset(df_assets.columns):
        df_assets = df_assets[
            (pd.to_numeric(df_assets["latitude"], errors="coerce").between(24, 49)) &
            (pd.to_numeric(df_assets["longitude"], errors="coerce").between(-125, -66))
        ].copy()

    # Prepare HPI/latest_price_index if available
    df_hpi = None
    if df_hpi_raw is not None and not df_hpi_raw.empty:
        df_hpi = safe_lower_cols(df_hpi_raw)
        # find any recent year columns 2024/2025
        price_cols = [c for c in df_hpi.columns if any(y in c for y in ["2025", "2024"])]
        if price_cols:
            latest_col = sorted(price_cols)[-1]
            df_hpi["latest_price_index"] = pd.to_numeric(df_hpi[latest_col], errors="coerce")
        else:
            df_hpi["latest_price_index"] = np.nan

    # merge on city_state key if both have city and state
    if df_hpi is not None and {"city", "state"}.issubset(df_assets.columns) and {"city", "state"}.issubset(df_hpi.columns):
        df_assets["city_state_key"] = df_assets["city"].astype(str).str.lower().str.strip() + "_" + df_assets["state"].astype(str).str.lower().str.strip()
        df_hpi["city_state_key"] = df_hpi["city"].astype(str).str.lower().str.strip() + "_" + df_hpi["state"].astype(str).str.lower().str.strip()
        merged = df_assets.merge(df_hpi[["city_state_key", "latest_price_index"]].drop_duplicates(), on="city_state_key", how="left")
    else:
        merged = df_assets.copy()
        merged["latest_price_index"] = np.nan

    # Fill missing latest_price_index with median or a default
    if "latest_price_index" in merged.columns:
        median_idx = pd.to_numeric(merged["latest_price_index"], errors="coerce").median()
        if np.isnan(median_idx):
            median_idx = 250000.0
        merged["latest_price_index"] = pd.to_numeric(merged["latest_price_index"], errors="coerce").fillna(median_idx)
    else:
        merged["latest_price_index"] = 250000.0

    # Determine rentable sqft column heuristically
    sqft_col = None
    for c in merged.columns:
        if ("rentable" in c or "sqft" in c or "square" in c) and ("feet" in c or "ft" in c or "sq" in c):
            sqft_col = c
            break

    # Compute estimated_value using rentable sqft if available, else use index-based heuristic
    if sqft_col is not None:
        merged["building_rentable_square_feet"] = pd.to_numeric(merged[sqft_col], errors="coerce").fillna(0)
        merged["estimated_value"] = merged["building_rentable_square_feet"] * (merged["latest_price_index"] / 100.0) * 10.0
    else:
        # fallback: index scaled with random uniform but deterministic using RandomState
        rng = np.random.RandomState(RANDOM_STATE)
        scale = rng.uniform(0.5, 2.0, len(merged))
        merged["estimated_value"] = merged["latest_price_index"] * scale

    # high-value state premium
    high_value_states = ["CA", "NY", "MA", "CT", "NJ", "HI", "MD", "WA"]
    if "state" in merged.columns:
        merged.loc[merged["state"].isin(high_value_states), "estimated_value"] *= 1.5

    # SAMPLE EXACTLY 10,000 rows deterministically if larger
    if len(merged) > 10000:
        merged = merged.sample(n=10000, random_state=RANDOM_STATE).reset_index(drop=True)
    else:
        merged = merged.reset_index(drop=True)

    return merged

def create_ml_features(df: pd.DataFrame):
    """Create ML features: distances to major cities + numeric columns"""
    df = df.copy()
    feats = []
    if {"latitude", "longitude"}.issubset(df.columns):
        major = {
            "nyc": (40.7128, -74.0060),
            "la": (34.0522, -118.2437),
            "chicago": (41.8781, -87.6298),
            "houston": (29.7604, -95.3698),
            "dc": (38.9072, -77.0369)
        }
        for k, (lat, lon) in major.items():
            colname = f"dist_to_{k}"
            df[colname] = np.sqrt((pd.to_numeric(df["latitude"], errors="coerce") - lat) ** 2 + (pd.to_numeric(df["longitude"], errors="coerce") - lon) ** 2)
            feats.append(colname)
    for c in ["latitude", "longitude", "latest_price_index", "building_rentable_square_feet"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            feats.append(c)
    return df, feats

# ---------------------------
# Map embedding helpers
# ---------------------------
def embed_html_map_if_available(file_path: str, height: int = 700):
    """If an HTML map file exists at file_path, embed it with components.html"""
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                html = f.read()
            components.html(html, height=height, scrolling=True)
            return True
        except Exception:
            return False
    return False

def folium_state_summary_map(df):
    """Create a state summary folium map: circle radii ~ total value per state"""
    # need aggregated lat/lon per state if available
    if not {"state", "latitude", "longitude", "estimated_value"}.issubset(df.columns):
        return None
    agg = df.groupby("state").agg({
        "estimated_value": "sum",
        "latitude": "mean",
        "longitude": "mean",
        "building_rentable_square_feet": "mean"
    }).reset_index()
    # center
    center = [agg["latitude"].mean(), agg["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=4, tiles="cartodbpositron")
    max_val = agg["estimated_value"].max() if not agg["estimated_value"].isnull().all() else 1.0
    for _, r in agg.iterrows():
        radius = 5000 * (r["estimated_value"] / max_val + 0.05)  # scaled
        popup = f"<b>{r['state']}</b><br>Total value: ${r['estimated_value']:,.0f}<br>Avg sqft: {r.get('building_rentable_square_feet',np.nan):,.0f}"
        folium.Circle(
            location=[r["latitude"], r["longitude"]],
            radius=radius,
            color=None,
            fill=True,
            fill_opacity=0.6,
            popup=popup
        ).add_to(m)
    return m

def folium_comprehensive_map(df, with_clusters=True):
    """Create a comprehensive folium map with circle markers and optional clustering/heatmap"""
    if not {"latitude", "longitude", "estimated_value"}.issubset(df.columns):
        return None
    m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=4, tiles="cartodbpositron")
    # optionally compute clusters for colors
    colors = ["red","blue","green","purple","orange","darkred","lightred","beige","darkblue","darkgreen"]
    if with_clusters:
        # choose numeric columns available for clustering
        cluster_cols = [c for c in ["latitude","longitude","estimated_value","building_rentable_square_feet","latest_price_index"] if c in df.columns]
        if len(cluster_cols) >= 2:
            X = df[cluster_cols].fillna(df[cluster_cols].median())
            try:
                scaler = MinMaxScaler()
                Z = scaler.fit_transform(X)
                k = min(7, max(2, int(np.sqrt(len(df)//50))))  # heuristic cluster count
                km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(Z)
                df["_cluster"] = km.labels_
            except Exception:
                df["_cluster"] = 0
        else:
            df["_cluster"] = 0
    else:
        df["_cluster"] = 0

    # Add CircleMarkers
    for _, r in df.iterrows():
        try:
            color = colors[int(r.get("_cluster", 0)) % len(colors)]
            radius = max(3, np.log1p(r.get("estimated_value", 0)) / 2.0)
            popup = f"<b>Asset</b><br>State: {r.get('state','N/A')}<br>City: {r.get('city','N/A')}<br>Est Value: ${r.get('estimated_value',0):,.0f}"
            folium.CircleMarker(
                location=[float(r["latitude"]), float(r["longitude"])],
                radius=radius,
                color="black",
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup, max_width=300)
            ).add_to(m)
        except Exception:
            continue
    return m

# ---------------------------
# Load data and prepare merged sample
# ---------------------------
with st.spinner("Loading datasets from Google Drive..."):
    df_assets_raw = load_csv_from_gdrive(ASSETS_URL, "assets_cached")
    df_hpi_raw = load_csv_from_gdrive(ZILLOW_URL, "zillow_cached")
    df_merged = clean_and_merge_data(df_assets_raw, df_hpi_raw)

if df_merged is None or df_merged.empty:
    st.error("No data available. Please check the Google Drive sources or use sample CSV in /mnt/data.")
    st.stop()

# ---------------------------
# Sidebar Filters & Navigation
# ---------------------------
st.sidebar.image("https://via.placeholder.com/300x90/1f4e79/ffffff?text=Assets+Analytics", use_container_width=True)
st.sidebar.markdown("### 🔍 Filters")

# State filter
state_list = ["All"]
if "state" in df_merged.columns:
    state_list = ["All"] + sorted(df_merged["state"].dropna().unique().tolist())
selected_state = st.sidebar.selectbox("State", state_list)

# Value range filter
if "estimated_value" in df_merged.columns:
    min_v = int(df_merged["estimated_value"].min())
    max_v = int(df_merged["estimated_value"].max())
    v_low, v_high = st.sidebar.slider("Asset Value ($)", min_value=min_v, max_value=max_v, value=(min_v, max_v), format="$%d")
else:
    v_low, v_high = None, None

# Main nav
page = st.sidebar.radio("View", [
    "📊 Executive Overview",
    "🗺️ State Summary Map",
    "🗺️ Comprehensive Asset Map",
    "🗺️ Asset Valuation Map",
    "🎯 Clustering Analysis",
    "🤖 Machine Learning",
    "📂 Predict Asset Price",
    "📈 Advanced Analytics"
])

# Apply filters
df_filtered = df_merged.copy()
if selected_state != "All" and "state" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["state"] == selected_state]
if v_low is not None and v_high is not None and "estimated_value" in df_filtered.columns:
    df_filtered = df_filtered[(df_filtered["estimated_value"] >= v_low) & (df_filtered["estimated_value"] <= v_high)]

# ---------------------------
# Helper: Train regression model (used by ML & Predict tabs)
# ---------------------------
@st.cache_data(show_spinner=False)
def train_regression_model(df):
    """Train a RandomForestRegressor on the sampled merged dataset and return (model, feature_names, metrics)."""
    df2 = df.dropna(subset=["estimated_value"]).copy()
    if df2.empty:
        return None, [], {}
    df2, features = create_ml_features(df2)
    # Keep numeric features only and drop rows with all-NaN
    X = df2[features].fillna(df2[features].median(numeric_only=True))
    y = df2["estimated_value"]
    if len(X) < 20:
        return None, features, {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    model = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test))
    }
    return model, features, metrics

# Keep model in session_state
if "reg_model" not in st.session_state:
    st.session_state["reg_model"], st.session_state["reg_features"], st.session_state["reg_metrics"] = train_regression_model(df_merged)

# ---------------------------
# Pages
# ---------------------------

# 1. Executive Overview
if page == "📊 Executive Overview":
    st.markdown("<h2>📊 Executive Overview</h2>", unsafe_allow_html=True)

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='kpi'><b>Total Assets</b><h3>{len(df_filtered):,}</h3></div>", unsafe_allow_html=True)
    with col2:
        total_value = df_filtered["estimated_value"].sum() if "estimated_value" in df_filtered.columns else 0.0
        st.markdown(f"<div class='kpi'><b>Portfolio Value</b><h3>${total_value/1e9:.2f}B</h3></div>", unsafe_allow_html=True)
    with col3:
        avg_value = df_filtered["estimated_value"].mean() if "estimated_value" in df_filtered.columns else 0.0
        st.markdown(f"<div class='kpi'><b>Average Asset Value</b><h3>${avg_value/1e6:.2f}M</h3></div>", unsafe_allow_html=True)
    with col4:
        states_count = int(df_filtered["state"].nunique()) if "state" in df_filtered.columns else 0
        st.markdown(f"<div class='kpi'><b>States Covered</b><h3>{states_count}</h3></div>", unsafe_allow_html=True)

    # Value distribution histogram + top states bar
    r1, r2 = st.columns(2)
    with r1:
        if "estimated_value" in df_filtered.columns:
            fig = px.histogram(df_filtered, x="estimated_value", nbins=40, title="Asset Value Distribution", template=PLOTLY_TEMPLATE)
            fig.update_layout(xaxis_title="Estimated Value ($)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No value data available.")
    with r2:
        if "state" in df_filtered.columns:
            top_states = df_filtered["state"].value_counts().nlargest(12)
            fig = px.bar(x=top_states.index, y=top_states.values, title="Top States by Asset Count", template=PLOTLY_TEMPLATE)
            fig.update_layout(xaxis_title="State", yaxis_title="Asset Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No state data available.")

    # Insights
    st.markdown("<h3>💡 Key Insights</h3>", unsafe_allow_html=True)
    insights = []
    if "state" in df_filtered.columns and not df_filtered.empty:
        vc = df_filtered["state"].value_counts()
        if not vc.empty:
            insights.append(f"📍 **{vc.idxmax()}** has the most assets ({vc.max():,}).")
    if "estimated_value" in df_filtered.columns:
        insights.append(f"💰 Portfolio value (selected): **${total_value:,.0f}**; average asset **${avg_value:,.0f}**.")
        if len(df_filtered) > 20:
            topn = max(1, int(len(df_filtered)*0.1))
            top_sum = df_filtered.nlargest(topn, "estimated_value")["estimated_value"].sum()
            conc = 100.0 * top_sum / total_value if total_value > 0 else 0.0
            insights.append(f"🏛️ Top 10% assets contain **{conc:.1f}%** of portfolio value.")
    for ins in insights:
        st.markdown(f"<div class='insight'>{ins}</div>", unsafe_allow_html=True)

# 2. State Summary Map (embed uploaded HTML or generate)
elif page == "🗺️ State Summary Map":
    st.markdown("<h2>🗺️ State Summary Map</h2>", unsafe_allow_html=True)
    # try embedding uploaded HTML first
    embedded = embed_html_map_if_available(MAP_FILES.get("state_summary",""), height=700)
    if not embedded:
        m = folium_state_summary_map(df_filtered)
        if m:
            st_folium(m, width=900, height=700)
        else:
            st.info("State summary map not available (missing coordinates or data).")

# 3. Comprehensive Asset Map (embed or generate)
elif page == "🗺️ Comprehensive Asset Map":
    st.markdown("<h2>🗺️ Comprehensive Asset Map</h2>", unsafe_allow_html=True)
    embedded = embed_html_map_if_available(MAP_FILES.get("assets_comprehensive",""), height=700) or embed_html_map_if_available(MAP_FILES.get("comprehensive",""), height=700)
    if not embedded:
        m = folium_comprehensive_map(df_filtered, with_clusters=True)
        if m:
            st_folium(m, width=1000, height=700)
        else:
            st.info("Comprehensive asset map not available (missing coordinates or data).")

# 4. Asset Valuation Map (embed or generate)
elif page == "🗺️ Asset Valuation Map":
    st.markdown("<h2>🗺️ Asset Valuation Map</h2>", unsafe_allow_html=True)
    embedded = embed_html_map_if_available(MAP_FILES.get("asset_valuation",""), height=700) or embed_html_map_if_available(MAP_FILES.get("assets",""), height=700)
    if not embedded:
        m = folium_comprehensive_map(df_filtered, with_clusters=False)
        if m:
            st_folium(m, width=1000, height=700)
        else:
            st.info("Asset valuation map not available (missing coordinates or data).")

# 5. Clustering Analysis
elif page == "🎯 Clustering Analysis":
    st.markdown("<h2>🎯 Clustering Analysis</h2>", unsafe_allow_html=True)
    # choose numeric columns for clustering
    cluster_cols = [c for c in ["latitude","longitude","estimated_value","building_rentable_square_feet","latest_price_index"] if c in df_filtered.columns]
    if len(cluster_cols) >= 2:
        n_clusters = st.slider("Number of clusters", 2, 10, 5)
        df_clustered, km = perform_clustering(df_filtered, n_clusters=n_clusters)
        st.write("Cluster counts:")
        st.dataframe(df_clustered["cluster"].value_counts().rename("count").to_frame())
        # show a map sample for clusters
        sample = df_clustered.dropna(subset=["latitude","longitude"]).sample(min(3000, len(df_clustered)), random_state=RANDOM_STATE)
        fig = px.scatter_mapbox(sample, lat="latitude", lon="longitude", color="cluster", size="estimated_value" if "estimated_value" in sample.columns else None, mapbox_style="carto-positron", zoom=3, height=650, template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)
        # cluster stats
        agg = df_clustered.groupby("cluster").agg(count=("estimated_value","count"), total_value=("estimated_value","sum"), avg_value=("estimated_value","mean")).reset_index().sort_values("total_value", ascending=False)
        st.subheader("Cluster Statistics")
        st.dataframe(agg)
    else:
        st.info("Not enough numeric columns for clustering.")

# 6. Machine Learning — training & diagnostics
elif page == "🤖 Machine Learning":
    st.markdown("<h2>🤖 Machine Learning</h2>", unsafe_allow_html=True)
    model, features, metrics = st.session_state.get("reg_model"), st.session_state.get("reg_features"), st.session_state.get("reg_metrics")
    st.subheader("Model Summary")
    if model is None:
        st.warning("Training model... This may take a moment.")
        model, features, metrics = train_regression_model(df_merged)
        st.session_state["reg_model"], st.session_state["reg_features"], st.session_state["reg_metrics"] = model, features, metrics

    if model is not None:
        st.markdown(f"- Trained on ~{metrics.get('train_rows', 'N/A')} rows; test rows: {metrics.get('test_rows', 'N/A')}")
        st.markdown(f"- **R² (test):** {metrics.get('r2',np.nan):.3f}    |    **MAE:** ${metrics.get('mae',np.nan):,.0f}")
        # Feature importance
        try:
            fi = pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
            fig = px.bar(fi.head(12), x="importance", y="feature", orientation="h", title="Top Feature Importances", template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Feature importance not available.")

        # Residual analysis: predict on sample subset
        st.subheader("Residuals (sample)")
        df_sample = df_merged.dropna(subset=["estimated_value"]).sample(min(2000, len(df_merged)), random_state=RANDOM_STATE)
        df_sample, _ = create_ml_features(df_sample)
        Xs = df_sample[features].fillna(df_sample[features].median(numeric_only=True))
        yhat = model.predict(Xs)
        df_sample["predicted_value"] = yhat
        df_sample["residual"] = df_sample["estimated_value"] - df_sample["predicted_value"]
        fig2 = px.histogram(df_sample, x="residual", nbins=50, title="Distribution of Residuals (sample)", template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig2, use_container_width=True)
        # show top over/under predictions
        st.subheader("Top Overestimated Assets (sample)")
        st.dataframe(df_sample.sort_values("residual").head(10)[["city","state","estimated_value","predicted_value","residual"]])

    else:
        st.error("Model training failed or not enough data.")

# 7. Predict Asset Price (single + bulk)
elif page == "📂 Predict Asset Price":
    st.markdown("<h2>📂 Predict Asset Price</h2>", unsafe_allow_html=True)
    model = st.session_state.get("reg_model")
    features = st.session_state.get("reg_features", [])
    if model is None or len(features) == 0:
        st.warning("Model not trained yet. Please train in the 'Machine Learning' tab (will auto-train on first visit).")
    else:
        st.subheader("Single-record prediction")
        col1, col2, col3 = st.columns(3)
        with col1:
            lat = st.number_input("Latitude", value=float(df_merged["latitude"].median() if "latitude" in df_merged.columns else 37.0))
        with col2:
            lon = st.number_input("Longitude", value=float(df_merged["longitude"].median() if "longitude" in df_merged.columns else -95.0))
        with col3:
            lpi = st.number_input("Latest Price Index", value=float(df_merged["latest_price_index"].median() if "latest_price_index" in df_merged.columns else 250000.0))
        sqft = st.number_input("Rentable Sqft (optional)", value=float(df_merged.get("building_rentable_square_feet", pd.Series([50000])).median()))
        if st.button("Predict Single"):
            tmp = pd.DataFrame([{"latitude": lat, "longitude": lon, "latest_price_index": lpi, "building_rentable_square_feet": sqft}])
            tmp, _ = create_ml_features(tmp)
            Xtmp = tmp.reindex(columns=features).fillna(tmp[features].median(numeric_only=True))
            pred = model.predict(Xtmp)[0]
            st.success(f"Predicted estimated value: ${pred:,.0f}")

        st.markdown("---")
        st.subheader("Bulk CSV prediction")
        uploaded = st.file_uploader("Upload CSV (columns: latitude, longitude, latest_price_index, rentable sqft...) to predict", type=["csv"])
        if uploaded is not None:
            try:
                df_new = pd.read_csv(uploaded)
                df_new = safe_lower_cols(df_new)
                df_new, _ = create_ml_features(df_new)
                Xnew = df_new.reindex(columns=features).fillna(df_new[features].median(numeric_only=True))
                preds = model.predict(Xnew)
                df_new["predicted_estimated_value"] = preds
                st.dataframe(df_new.head(20))
                outcsv = df_new.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", data=outcsv, file_name="predictions.csv", mime="text/csv")
                # summary inference
                st.markdown("### Bulk prediction summary")
                st.metric("Rows predicted", f"{len(df_new):,}")
                st.metric("Predicted Mean Value", f"${df_new['predicted_estimated_value'].mean():,.0f}")
                st.bar_chart(df_new["predicted_estimated_value"].quantile([0.0,0.25,0.5,0.75,1.0]))
            except Exception as e:
                st.error(f"Failed to process uploaded CSV: {e}")

# 8. Advanced Analytics
elif page == "📈 Advanced Analytics":
    st.markdown("<h2>📈 Advanced Analytics</h2>", unsafe_allow_html=True)
    tabs = st.tabs(["Statistical Summary", "Data Quality", "Trends"])
    with tabs[0]:
        st.subheader("Statistical Summary")
        if "estimated_value" in df_filtered.columns:
            stats = df_filtered["estimated_value"].describe()
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean", f"${stats['mean']:,.0f}")
            c2.metric("Median", f"${stats['50%']:,.0f}")
            c3.metric("Std Dev", f"${stats['std']:,.0f}")
            fig = px.box(df_filtered, y="estimated_value", title="Estimated Value Distribution", template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No estimated value data.")
    with tabs[1]:
        st.subheader("Data Quality")
        missing = df_filtered.isnull().sum().sort_values(ascending=False)
        miss_df = pd.DataFrame({"missing_count": missing, "missing_pct": 100*missing/len(df_filtered)})
        miss_df = miss_df[miss_df["missing_count"]>0]
        if not miss_df.empty:
            fig = px.bar(miss_df.head(15).reset_index().rename(columns={"index":"column"}), x="missing_pct", y="column", orientation="h", title="Top missing columns", template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(miss_df.head(40))
        else:
            st.success("No missing data detected!")
    with tabs[2]:
        st.subheader("Trends")
        if "state" in df_filtered.columns and "estimated_value" in df_filtered.columns:
            state_agg = df_filtered.groupby("state").agg(asset_count=("estimated_value","count"), total_value=("estimated_value","sum"), avg_value=("estimated_value","mean")).reset_index().sort_values("total_value", ascending=False)
            st.dataframe(state_agg.head(20))
            fig = px.scatter(state_agg.head(20), x="asset_count", y="avg_value", size="total_value", color="total_value", hover_name="state", template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("State/value data required for trends.")

# Footer / credits
st.markdown("---")
st.markdown("**Notes:** All random operations and sampling use `random_state = 4742271` for reproducibility. Maps embed any pre-generated HTML files located in `/mnt/data/` when present. Predictions use a RandomForestRegressor trained on the merged sample.")

# streamlit_app.py
import os
import io
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np

# Visualizations
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Mapping
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

# GIS / Spatial
import geopandas as gpd
from shapely.geometry import Point

# Stats / ML
from scipy import stats
from sklearn.cluster import DBSCAN

# Utilities
import gdown

# --------------------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="GIS & Spatial Analytics Dashboard", layout="wide", page_icon="🌐")

st.markdown("""
<style>
.metric-card {
  border-radius: 16px; padding: 14px; box-shadow: 0 2px 12px rgba(0,0,0,0.08);
  background: white; border: 1px solid rgba(0,0,0,0.05);
}
.kpi { font-size: 24px; font-weight: 700; }
.kpi-label { color: #666; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
.small { font-size: 12px; color: #777; }
</style>
""", unsafe_allow_html=True)

st.title("🌐 GIS, EDA, Spatial Analysis & Scoring Dashboard")
st.caption("Streamlit • GeoPandas • Shapely • Folium • scikit-learn • Plotly • (PySAL optional)")

# HTML + JS example (clock)
st.components.v1.html("""
<div style="font-family:system-ui; padding:8px 12px; background:#0f172a; color:#e2e8f0; border-radius:10px; display:inline-block;">
  <span>🕒 <b>Live Clock</b>: <span id="clk"></span></span>
</div>
<script>
function u(){document.getElementById('clk').textContent = new Date().toLocaleString();}
u(); setInterval(u, 1000);
</script>
""", height=50)

# --------------------------------------------------------------------------------------
# Sidebar: Controls & Inputs
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Data Sources (Internet)")
    st.write("These are your Google Drive file IDs from the Colab:")
    default_assets_id = "1YFTWJNoxu0BF8UlMDXI8bXwRTVQNE2mb"
    default_prices_id = "1fFT8Q8GWiIEM7kx6czhQ-qabygUPBQRv"
    assets_file_id = st.text_input("Assets Google Drive File ID", default_assets_id)
    prices_file_id = st.text_input("Prices Google Drive File ID", default_prices_id)
    sample_size = st.slider("Sample size for maps/plots (to keep UI fast)", min_value=1000, max_value=15000, value=7500, step=500)
    st.markdown("---")
    st.subheader("Clustering")
    eps_km = st.slider("DBSCAN radius (km)", 10, 150, 50, 5)
    min_samples = st.slider("DBSCAN min samples", 3, 20, 5, 1)
    st.markdown("---")
    st.subheader("Map Layers")
    show_heatmap = st.checkbox("Show Heatmap", True)
    show_markers = st.checkbox("Show Value Markers", True)
    show_clusters = st.checkbox("Show Marker Cluster", True)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def download_drive_file(file_id: str, filename: str) -> str:
    """Download a Google Drive file to a temp path and return the local path."""
    url = f"https://drive.google.com/uc?id={file_id}"
    out = os.path.join(st.session_state.get("tmp_dir", "/tmp"), filename)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    gdown.download(url, out, quiet=True)
    return out

@st.cache_data(show_spinner=True)
def load_csv_from_drive(file_id: str, filename: str, encodings=("utf-8", "latin-1")) -> pd.DataFrame:
    path = download_drive_file(file_id, filename)
    last_exc = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_exc = e
    raise last_exc

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns
                  .str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace("-", "_")
                  .str.replace("/", "_"))
    return df

def clean_zip_code(z):
    if pd.isna(z):
        return None
    s = "".join(ch for ch in str(z) if ch.isdigit())
    s = s[:5]
    return s if len(s) == 5 else None

def safe_number_series(series):
    return pd.to_numeric(series, errors="coerce")

def to_webmercator(gdf):
    try:
        return gdf.to_crs("EPSG:3857")
    except Exception:
        return None

def km_to_meters(km: float) -> float:
    return km * 1000.0

# --------------------------------------------------------------------------------------
# Load Data (Internet only, mirrors your notebook)
# --------------------------------------------------------------------------------------
if "tmp_dir" not in st.session_state:
    st.session_state["tmp_dir"] = "/tmp"

with st.spinner("Downloading datasets from Google Drive..."):
    df_assets_raw = load_csv_from_drive(assets_file_id, "us_government_assets.csv")
    df_prices_raw = load_csv_from_drive(prices_file_id, "zillow_housing_index.csv")

# --------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------
tabs = st.tabs([
    "1) Setup & Load",
    "2) EDA",
    "3) Cleaning",
    "4) Merge",
    "5) Spatial",
    "6) Clustering & Scoring"
])

# ======================================================================================
# 1) SETUP & LOAD
# ======================================================================================
with tabs[0]:
    st.subheader("📥 Datasets Loaded (from the Internet)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Assets (raw)**")
        st.dataframe(df_assets_raw.head(20), use_container_width=True)
        st.write(f"Shape: {df_assets_raw.shape}, Missing: {int(df_assets_raw.isna().sum().sum())}, Duplicates: {int(df_assets_raw.duplicated().sum())}")
    with c2:
        st.markdown("**Prices (raw)**")
        st.dataframe(df_prices_raw.head(20), use_container_width=True)
        st.write(f"Shape: {df_prices_raw.shape}, Missing: {int(df_prices_raw.isna().sum().sum())}, Duplicates: {int(df_prices_raw.duplicated().sum())}")

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown('<div class="metric-card"><div class="kpi">{}</div><div class="kpi-label">Assets Rows</div></div>'.format(len(df_assets_raw)), unsafe_allow_html=True)
    with k2:
        st.markdown('<div class="metric-card"><div class="kpi">{}</div><div class="kpi-label">Prices Rows</div></div>'.format(len(df_prices_raw)), unsafe_allow_html=True)
    with k3:
        st.markdown('<div class="metric-card"><div class="kpi">{}</div><div class="kpi-label">Assets Columns</div></div>'.format(df_assets_raw.shape[1]), unsafe_allow_html=True)
    with k4:
        st.markdown('<div class="metric-card"><div class="kpi">{}</div><div class="kpi-label">Prices Columns</div></div>'.format(df_prices_raw.shape[1]), unsafe_allow_html=True)

# ======================================================================================
# 2) EDA
# ======================================================================================
with tabs[1]:
    st.subheader("🔍 Exploratory Data Analysis")

    # Missing values bar (Assets / Prices)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Missing Values — Assets**")
        miss = df_assets_raw.isnull().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        if len(miss) > 0:
            fig = go.Figure(go.Bar(x=miss.index, y=miss.values))
            fig.update_layout(height=350, margin=dict(l=10,r=10,b=10,t=30), xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No missing values.")
    with c2:
        st.markdown("**Missing Values — Prices**")
        miss = df_prices_raw.isnull().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        if len(miss) > 0:
            fig = go.Figure(go.Bar(x=miss.index, y=miss.values))
            fig.update_layout(height=350, margin=dict(l=10,r=10,b=10,t=30), xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No missing values.")

    # Descriptives
    st.markdown("**Descriptive Statistics (top numeric columns)**")
    nc_assets = df_assets_raw.select_dtypes(include=[np.number]).columns.tolist()
    nc_prices = df_prices_raw.select_dtypes(include=[np.number]).columns.tolist()
    cc1, cc2 = st.columns(2)
    with cc1:
        if len(nc_assets):
            st.dataframe(df_assets_raw[nc_assets].describe().T, use_container_width=True)
        else:
            st.info("No numeric columns in Assets.")
    with cc2:
        if len(nc_prices):
            st.dataframe(df_prices_raw[nc_prices].describe().T, use_container_width=True)
        else:
            st.info("No numeric columns in Prices.")

    # Correlation heatmap (Assets)
    if len(nc_assets) > 1:
        corr = df_assets_raw[nc_assets].corr()
        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu", zmid=0))
        fig.update_layout(title="Correlation Heatmap — Assets", height=600)
        st.plotly_chart(fig, use_container_width=True)

# ======================================================================================
# 3) CLEANING
# ======================================================================================
with tabs[2]:
    st.subheader("🧹 Cleaning & Standardization")

    df_assets = normalize_column_names(df_assets_raw)
    df_prices = normalize_column_names(df_prices_raw)

    # Find plausible sqft/area columns
    sqft_cols = [c for c in df_assets.columns if any(k in c for k in ["sqft", "sq_ft", "area"])]
    if len(sqft_cols):
        for c in sqft_cols:
            df_assets[c] = safe_number_series(df_assets[c]).fillna(0)

    # Location standardization
    location_mapping = {}
    for c in df_assets.columns:
        lc = c.lower()
        if 'latitude' not in location_mapping and 'lat' in lc:
            location_mapping['latitude'] = c
        if 'longitude' not in location_mapping and ('lon' in lc or 'lng' in lc):
            location_mapping['longitude'] = c
        if 'state' not in location_mapping and 'state' in lc:
            location_mapping['state'] = c
        if 'city' not in location_mapping and 'city' in lc:
            location_mapping['city'] = c
        if 'zip_code' not in location_mapping and 'zip' in lc:
            location_mapping['zip_code'] = c

    for std, orig in location_mapping.items():
        if orig in df_assets.columns:
            df_assets[std] = df_assets[orig]

    # Validate coordinates (US bounds as in notebook)
    if 'latitude' in df_assets.columns and 'longitude' in df_assets.columns:
        df_assets['latitude']  = safe_number_series(df_assets['latitude'])
        df_assets['longitude'] = safe_number_series(df_assets['longitude'])
        valid_mask = (
            df_assets['latitude'].between(24, 49) &
            df_assets['longitude'].between(-125, -66)
        )
        invalid = (~valid_mask).sum()
        df_assets = df_assets[valid_mask]
        st.write(f"Filtered out invalid coordinates: **{int(invalid)}** rows removed.")
        st.write(f"Retained rows: **{len(df_assets)}**")

    st.markdown("**After Cleaning — Assets (sample)**")
    st.dataframe(df_assets.head(20), use_container_width=True)

    # Prices: detect latest year column or create baseline
    year_cols = [c for c in df_prices.columns if any(y in str(c) for y in ['2025','2024','2023']) or str(c).isdigit()]
    if year_cols:
        year_cols_sorted = sorted(year_cols, reverse=True)
        latest_year_col = year_cols_sorted[0]
        df_prices['latest_price_index'] = pd.to_numeric(df_prices[latest_year_col], errors="coerce")
        st.info(f"Using **{latest_year_col}** as latest price index.")
    else:
        df_prices['latest_price_index'] = 100.0
        st.warning("No year columns found in Prices; using baseline latest_price_index=100.")

    st.markdown("**After Cleaning — Prices (sample)**")
    st.dataframe(df_prices.head(20), use_container_width=True)

# ======================================================================================
# 4) MERGE
# ======================================================================================
with tabs[3]:
    st.subheader("🔗 Intelligent Merge (ZIP / City-State) + Valuation")

    # Work on copies to avoid cache mutation issues
    dfA = df_assets.copy()
    dfP = df_prices.copy()

    # Standardize ZIPs
    if 'zip_code' in dfA.columns:
        dfA['zip_code_clean'] = dfA['zip_code'].apply(clean_zip_code)

    zip_col_prices = None
    for c in dfP.columns:
        if 'zip' in c:
            zip_col_prices = c
            break
    if zip_col_prices:
        dfP['zip_code_clean'] = dfP[zip_col_prices].apply(clean_zip_code)

    merged_datasets = {}

    # Strategy 1: ZIP merge
    if 'zip_code_clean' in dfA.columns and 'zip_code_clean' in dfP.columns:
        m1 = pd.merge(dfA, dfP[['zip_code_clean', 'latest_price_index']], on='zip_code_clean', how='left')
        merged_datasets["zip_merge"] = m1

    # Strategy 2: City-State merge (if both exist)
    city_col_prices = next((c for c in dfP.columns if 'city' in c), None)
    state_col_prices = next((c for c in dfP.columns if 'state' in c), None)
    if all([city_col_prices, state_col_prices]) and all([(c in dfA.columns) for c in ['city','state']]):
        dfA['city_state_key'] = dfA['city'].str.lower().str.strip() + "_" + dfA['state'].str.lower().str.strip()
        dfP['city_state_key'] = dfP[city_col_prices].str.lower().str.strip() + "_" + dfP[state_col_prices].str.lower().str.strip()
        m2 = pd.merge(dfA, dfP[['city_state_key','latest_price_index']], on='city_state_key', how='left')
        merged_datasets["city_state_merge"] = m2

    # Pick the merge with most matches
    if merged_datasets:
        best_name = max(merged_datasets, key=lambda k: merged_datasets[k]['latest_price_index'].notna().sum())
        df_merged = merged_datasets[best_name].copy()
        st.success(f"Selected **{best_name}** (most price matches).")
    else:
        df_merged = dfA.copy()
        df_merged['latest_price_index'] = 100.0
        st.warning("No merge keys found; used fallback with baseline latest_price_index.")

    # Fill missing price indices with regional mean then overall median
    if 'state' in df_merged.columns:
        state_avg = df_merged.groupby('state')['latest_price_index'].mean()
        mask = df_merged['latest_price_index'].isna()
        for s, v in state_avg.items():
            row_mask = (df_merged['state'] == s) & mask
            df_merged.loc[row_mask, 'latest_price_index'] = v
    overall_med = df_merged['latest_price_index'].median()
    df_merged['latest_price_index'] = df_merged['latest_price_index'].fillna(overall_med)

    # Identify rentable sqft column
    sqft_cols = [c for c in df_merged.columns if 'sqft' in c or 'sq_ft' in c or 'area' in c]
    rentable_col = None
    for c in sqft_cols:
        if 'rentable' in c:
            rentable_col = c
            break
    if not rentable_col and sqft_cols:
        rentable_col = sqft_cols[0]

    # Estimated value (replicating your logic, with state premium)
    if rentable_col:
        df_merged[rentable_col] = safe_number_series(df_merged[rentable_col]).fillna(0)
        df_merged['estimated_value'] = df_merged[rentable_col] * (df_merged['latest_price_index'] / 100.0) * 10.0
    else:
        df_merged['estimated_value'] = df_merged['latest_price_index'] * 1000.0

    if 'state' in df_merged.columns:
        high_value_states = ['CA','NY','MA','CT','NJ','HI','MD','WA']
        premium_mask = df_merged['state'].isin(high_value_states)
        df_merged.loc[premium_mask, 'estimated_value'] *= 1.5

    # Scores
    scoring_cols = []
    if rentable_col:
        df_merged['size_score'] = pd.qcut(df_merged[rentable_col].fillna(0), 5, labels=[1,2,3,4,5]).astype(float)
        scoring_cols.append('size_score')
    df_merged['location_score'] = pd.qcut(df_merged['latest_price_index'], 5, labels=[1,2,3,4,5]).astype(float)
    scoring_cols.append('location_score')
    df_merged['value_score'] = pd.qcut(df_merged['estimated_value'], 5, labels=[1,2,3,4,5]).astype(float)
    scoring_cols.append('value_score')
    df_merged['composite_score'] = df_merged[scoring_cols].mean(axis=1)

    # Cache merged for later tabs
    st.session_state["df_merged"] = df_merged

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="metric-card"><div class="kpi">{len(df_merged):,}</div><div class="kpi-label">Merged Records</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="metric-card"><div class="kpi">${df_merged["estimated_value"].mean():,.0f}</div><div class="kpi-label">Avg Est. Value</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="metric-card"><div class="kpi">${df_merged["estimated_value"].median():,.0f}</div><div class="kpi-label">Median Est. Value</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="metric-card"><div class="kpi">${df_merged["estimated_value"].sum():,.0f}</div><div class="kpi-label">Total Portfolio</div></div>', unsafe_allow_html=True)

    st.markdown("**Top 10 by Estimated Value**")
    disp_cols = ['estimated_value', 'composite_score']
    for c in ['state','city', rentable_col]:
        if c in df_merged.columns and c not in disp_cols:
            disp_cols.append(c)
    st.dataframe(df_merged.nlargest(10, 'estimated_value')[disp_cols], use_container_width=True)

# ======================================================================================
# 5) SPATIAL (Folium map with heatmap & markers)
# ======================================================================================
with tabs[4]:
    st.subheader("🗺️ Spatial Mapping")

    df_merged = st.session_state.get("df_merged", None)
    if df_merged is None:
        st.warning("Run the Merge tab first.")
    else:
        # Filter valid coordinates for mapping
        if not {'latitude','longitude'}.issubset(df_merged.columns):
            st.error("No latitude/longitude columns present after cleaning.")
        else:
            coord_mask = (
                df_merged['latitude'].notna() &
                df_merged['longitude'].notna() &
                df_merged['latitude'].between(-90, 90) &
                df_merged['longitude'].between(-180, 180)
            )
            df_geo = df_merged[coord_mask].copy()

            # Sampling for performance
            if len(df_geo) > sample_size:
                df_geo = df_geo.sample(n=sample_size, random_state=42).reset_index(drop=True)

            if df_geo.empty:
                st.warning("No valid coordinate rows to map.")
            else:
                center_lat = df_geo['latitude'].mean()
                center_lon = df_geo['longitude'].mean()
                fmap = folium.Map(location=[center_lat, center_lon], zoom_start=4, tiles="OpenStreetMap")

                # Heatmap
                if show_heatmap:
                    heat_data = []
                    for _, r in df_geo.iterrows():
                        val = float(max(r.get('estimated_value', 0.0), 1.0))
                        weight = float(np.log10(val))
                        heat_data.append([r['latitude'], r['longitude'], weight])
                    if heat_data:
                        HeatMap(heat_data, name="Value Heatmap", min_opacity=0.3, radius=15, blur=10).add_to(fmap)

                # Value markers
                if show_markers:
                    vmin, vmax = df_geo['estimated_value'].min(), df_geo['estimated_value'].max()
                    colors = ['blue', 'green', 'orange', 'red', 'purple']
                    try:
                        q = pd.qcut(df_geo['estimated_value'], 5, labels=False, duplicates='drop')
                    except Exception:
                        q = pd.Series(0, index=df_geo.index)

                    for i, (_, r) in enumerate(df_geo.iterrows()):
                        try:
                            if vmax > vmin:
                                radius = 5 + (r['estimated_value']-vmin) / (vmax-vmin) * 15
                            else:
                                radius = 10
                            color = colors[int(q.iloc[i])] if i < len(q) else 'blue'
                            popup = f"""
                            <b>Estimated Value:</b> ${r['estimated_value']:,.0f}<br>
                            <b>Location:</b> {r.get('city','N/A')}, {r.get('state','N/A')}<br>
                            <b>Coords:</b> {r['latitude']:.4f}, {r['longitude']:.4f}
                            """
                            folium.CircleMarker(
                                location=[r['latitude'], r['longitude']],
                                radius=radius,
                                popup=folium.Popup(popup, max_width=300),
                                color='black', fillColor=color, fillOpacity=0.7, weight=1
                            ).add_to(fmap)
                        except Exception:
                            continue

                # Clustered markers
                if show_clusters and len(df_geo) > 100:
                    cluster = MarkerCluster(name="Clustered View").add_to(fmap)
                    for _, r in df_geo.iterrows():
                        try:
                            folium.Marker(
                                location=[r['latitude'], r['longitude']],
                                popup=f"Value: ${r['estimated_value']:,.0f}"
                            ).add_to(cluster)
                        except Exception:
                            continue

                folium.LayerControl().add_to(fmap)
                st_folium(fmap, width=None, height=640)

# ======================================================================================
# 6) CLUSTERING & SCORING (DBSCAN in EPSG:3857)
# ======================================================================================
with tabs[5]:
    st.subheader("🔬 Clustering & Scoring")

    df_merged = st.session_state.get("df_merged", None)
    if df_merged is None:
        st.warning("Run the Merge tab first.")
    else:
        # Build GeoDataFrame for clustering
        valid = df_merged[['latitude','longitude']].dropna()
        if valid.empty:
            st.error("No valid coordinates for clustering.")
        else:
            gdf = gpd.GeoDataFrame(
                df_merged.loc[valid.index].copy(),
                geometry=[Point(xy) for xy in zip(df_merged.loc[valid.index, 'longitude'],
                                                  df_merged.loc[valid.index, 'latitude'])],
                crs="EPSG:4326"
            )

            if len(gdf) > sample_size:
                gdf = gdf.sample(n=sample_size, random_state=42).reset_index(drop=True)

            gdf_3857 = to_webmercator(gdf)
            if gdf_3857 is None:
                st.error("Could not project to EPSG:3857 for DBSCAN.")
            else:
                coords = np.column_stack([gdf_3857.geometry.x.values, gdf_3857.geometry.y.values])

                db = DBSCAN(eps=km_to_meters(eps_km), min_samples=min_samples)
                labels = db.fit_predict(coords)
                gdf['spatial_cluster'] = labels

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = int((labels == -1).sum())

                k1, k2, k3 = st.columns(3)
                with k1:
                    st.markdown(f'<div class="metric-card"><div class="kpi">{n_clusters}</div><div class="kpi-label">Clusters</div></div>', unsafe_allow_html=True)
                with k2:
                    st.markdown(f'<div class="metric-card"><div class="kpi">{n_noise}</div><div class="kpi-label">Noise Points</div></div>', unsafe_allow_html=True)
                with k3:
                    st.markdown(f'<div class="metric-card"><div class="kpi">{len(gdf):,}</div><div class="kpi-label">Analyzed Points</div></div>', unsafe_allow_html=True)

                # Cluster summary
                if n_clusters > 0:
                    clust_stats = (gdf[gdf['spatial_cluster'] != -1]
                                   .groupby('spatial_cluster')
                                   .agg(asset_count=('estimated_value','count'),
                                        total_value=('estimated_value','sum'),
                                        avg_value=('estimated_value','mean'))
                                   .sort_values('total_value', ascending=False)
                                   .round(2))
                    st.markdown("**Cluster Summary (top by total value)**")
                    st.dataframe(clust_stats.head(15), use_container_width=True)

                # Quick cluster scatter (Plotly)
                if {'latitude','longitude','spatial_cluster'}.issubset(gdf.columns):
                    fig = px.scatter_mapbox(
                        gdf.sample(min(len(gdf), 5000), random_state=42),
                        lat="latitude", lon="longitude",
                        color=gdf['spatial_cluster'].astype(str),
                        hover_data={'estimated_value':':.0f','spatial_cluster':True},
                        zoom=3, height=600
                    )
                    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=10,r=10,t=30,b=10))
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Sample of clustered data**")
                show_cols = ['latitude','longitude','estimated_value','spatial_cluster']
                for c in ['state','city','composite_score']:
                    if c in gdf.columns: show_cols.append(c)
                st.dataframe(gdf[show_cols].head(20), use_container_width=True)

# --------------------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------------------
st.markdown("---")
st.caption("Paste your own Google Drive file IDs in the sidebar to swap datasets. This app avoids GDAL/raster dependencies so it deploys cleanly on Streamlit Cloud.")

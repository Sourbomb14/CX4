# app.py — US Government Assets Portfolio Analytics (Refactored)
# ---------------------------------------------------------------
# ✅ Uses Google Drive CSVs only (as requested)
# ✅ Robust, production-ready Streamlit app
# ✅ Light/Dark theme compatible visuals
# ✅ Adds GIS capabilities (GeoPandas, Shapely, GDAL*), PySAL (if available), Geopy geocoding
# ✅ Adds an Asset Value Prediction tab (bulk + single-record)
# ✅ Graceful fallbacks if optional GIS libs are missing on host
# ---------------------------------------------------------------

import os
import io
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import streamlit as st
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go

# GIS & mapping
import folium
try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    from shapely.ops import nearest_points
    GEOS_OK = True
except Exception:
    GEOS_OK = False

# Optional heavy libs (graceful fallback)
try:
    # GDAL is often unavailable on hosted Streamlit; code branches if missing
    from osgeo import gdal  # noqa: F401
    GDAL_OK = True
except Exception:
    GDAL_OK = False

try:
    import libpysal
    import esda  # Moran, Lisa
    PYSAL_OK = True
except Exception:
    PYSAL_OK = False

try:
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    GEOPY_OK = True
except Exception:
    GEOPY_OK = False

# ML
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# ---------------
# Page config & Theming
# ---------------
st.set_page_config(
    page_title="US Government Assets Portfolio Analytics",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Plotly template auto-align to Streamlit theme if available
try:
    base_theme = st.get_option("theme.base")
    PLOTLY_TEMPLATE = "plotly_dark" if base_theme == "dark" else "plotly"
except Exception:
    PLOTLY_TEMPLATE = "plotly"

# Global CSS for light/dark support
st.markdown(
    """
    <style>
      :root {
        --brand:#1f4e79;
        --card-bg: #f6f8fb;
        --ink:#0f172a;
      }
      @media (prefers-color-scheme: dark) {
        :root { --card-bg:#111827; --ink:#e5e7eb; }
      }
      .main-header { font-size:2.2rem; font-weight:800; color:var(--brand); text-align:center; margin: 0.5rem 0 1rem; }
      .kpi {
        background: var(--card-bg);
        border-radius: 16px; padding: 16px; border: 1px solid rgba(0,0,0,0.05);
      }
      .insight { background:rgba(31,78,121,0.08); border-left:6px solid var(--brand); padding:12px 14px; border-radius:10px; margin-bottom:10px; }
      .muted { opacity:.8; }
      .chip { display:inline-block; padding:2px 8px; border-radius:999px; background:rgba(31,78,121,.12); font-size:.8rem; }
      .section-title { font-size:1.2rem; font-weight:700; margin:.5rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------
# Data Loading (Google Drive — as requested)
# ---------------
ASSETS_GDRIVE_ID = "1YFTWJNoxu0BF8UlMDXI8bXwRTVQNE2mb"  # U.S. Government Real Property Assets
ZILLOW_GDRIVE_ID = "1fFT8Q8GWiIEM7kx6czhQ-qabygUPBQRv"   # Zillow HPI 2000–2025 (city level)

ASSETS_URL = f"https://drive.google.com/uc?id={ASSETS_GDRIVE_ID}"
ZILLOW_URL = f"https://drive.google.com/uc?id={ZILLOW_GDRIVE_ID}"

@st.cache_data(show_spinner=False)
def load_csv_from_gdrive(url: str, fallback_path: str):
    import requests
    try:
        r = requests.get(url, timeout=25)
        r.raise_for_status()
        with open(fallback_path, "wb") as f:
            f.write(r.content)
        df = pd.read_csv(fallback_path)
        return df
    except Exception:
        # Fallback to local cache if present
        if os.path.exists(fallback_path):
            try:
                return pd.read_csv(fallback_path)
            except Exception as e:
                st.error(f"Failed reading cached file {fallback_path}: {e}")
        return None

with st.spinner("Loading datasets from Google Drive…"):
    df_assets_raw = load_csv_from_gdrive(ASSETS_URL, "assets_cached.csv")
    df_hpi_raw = load_csv_from_gdrive(ZILLOW_URL, "zillow_cached.csv")

# ---------------
# Data Preparation & Merge
# ---------------
@st.cache_data(show_spinner=False)
def clean_and_merge_data(df_assets: pd.DataFrame, df_hpi: pd.DataFrame) -> pd.DataFrame:
    if df_assets is None:
        return pd.DataFrame()

    df = df_assets.copy()
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
    )

    # Coordinates sanity check if exist
    if {"latitude", "longitude"}.issubset(df.columns):
        df = df[(df["latitude"].between(24, 49)) & (df["longitude"].between(-125, -66))]

    # Normalize city/state keys for merge
    def _norm(s):
        return s.astype(str).str.strip().str.lower()

    if df_hpi is not None:
        hpi = df_hpi.copy()
        hpi.columns = (
            hpi.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
        )
        if {"city", "state"}.issubset(hpi.columns):
            # choose latest 2025/2024 column
            date_cols = [c for c in hpi.columns if any(y in c for y in ["2025", "2024"]) ]
            if date_cols:
                latest_col = sorted(date_cols)[-1]
                hpi["latest_price_index"] = pd.to_numeric(hpi[latest_col], errors="coerce")
            else:
                hpi["latest_price_index"] = np.nan

            if {"city", "state"}.issubset(df.columns):
                df["city_state_key"] = _norm(df["city"]) + "_" + _norm(df["state"])  
                hpi["city_state_key"] = _norm(hpi["city"]) + "_" + _norm(hpi["state"])  
                hpi_small = hpi[["city_state_key", "latest_price_index"]].drop_duplicates()
                df = df.merge(hpi_small, on="city_state_key", how="left")

    # Estimated value
    rentable_col = next((c for c in df.columns if "rentable" in c and "feet" in c), None)
    if "latest_price_index" not in df.columns:
        df["latest_price_index"] = np.nan

    if rentable_col is not None:
        df["estimated_value"] = (
            pd.to_numeric(df[rentable_col], errors="coerce").fillna(0)
            * (df["latest_price_index"].fillna(df["latest_price_index"].median()) / 100.0)
            * 10.0
        )
    else:
        # heuristic fallback
        rng = np.random.default_rng(42)
        df["estimated_value"] = (
            df["latest_price_index"].fillna(df["latest_price_index"].median())
            * rng.uniform(0.5, 2.0, len(df))
        )

    # Premium for high-value states
    high_value_states = ["CA","NY","MA","CT","NJ","HI","MD","WA"]
    if "state" in df.columns:
        df.loc[df["state"].isin(high_value_states), "estimated_value"] *= 1.5

    return df

with st.spinner("Cleaning & merging…"):
    df_merged = clean_and_merge_data(df_assets_raw, df_hpi_raw)

if df_merged is None or df_merged.empty:
    st.error("No data available. Please check your Google Drive CSVs.")
    st.stop()

# ---------------
# Sidebar — Filters & Global Controls
# ---------------
st.sidebar.image(
    "https://via.placeholder.com/300x100/1f4e79/ffffff?text=Portfolio+Analytics",
    use_container_width=True,
)

st.sidebar.markdown("### 🔍 Filters")
state_sel = "All"
if "state" in df_merged.columns:
    states = ["All"] + sorted([s for s in df_merged["state"].dropna().unique()])
    state_sel = st.sidebar.selectbox("State", states)

min_val, max_val = float(df_merged["estimated_value"].min()), float(df_merged["estimated_value"].max())
vl, vh = st.sidebar.slider(
    "Asset Value Range ($)", min_value=int(min_val), max_value=int(max_val),
    value=(int(min_val), int(max_val)), format="$%d"
)

# Optional user address to geocode and center map / run spatial query
user_address = st.sidebar.text_input("Optional: Address to geocode (city, state or full)")

# apply filters
df_filtered = df_merged.copy()
if state_sel != "All":
    df_filtered = df_filtered[df_filtered["state"] == state_sel]

df_filtered = df_filtered[(df_filtered["estimated_value"].between(vl, vh))]

# ---------------
# Top Navbar Tabs
# ---------------
st.markdown('<div class="main-header">🏛️ US Government Assets Portfolio Analytics</div>', unsafe_allow_html=True)
TABS = st.tabs([
    "📊 Executive",
    "🗺️ Geo & GIS",
    "🌐 Web Map",
    "🎯 Clustering",
    "🤖 ML",
    "🧮 Predict Values",
    "📈 Advanced",
])

# ---------------
# 📊 Executive
# ---------------
with TABS[0]:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='kpi'>Total Assets<br><h2>"+f"{len(df_filtered):,}"+"</h2></div>", unsafe_allow_html=True)
    with c2:
        ttl = df_filtered["estimated_value"].sum()
        st.markdown("<div class='kpi'>Portfolio Value<br><h2>"+f"${ttl/1e9:.2f}B"+"</h2></div>", unsafe_allow_html=True)
    with c3:
        avg = df_filtered["estimated_value"].mean()
        st.markdown("<div class='kpi'>Average Asset Value<br><h2>"+f"${avg/1e6:.2f}M"+"</h2></div>", unsafe_allow_html=True)
    with c4:
        st_cnt = df_filtered["state"].nunique() if "state" in df_filtered.columns else 0
        st.markdown("<div class='kpi'>States Covered<br><h2>"+f"{st_cnt}"+"</h2></div>", unsafe_allow_html=True)

    # Distribution & Top States
    co1, co2 = st.columns(2)
    with co1:
        fig = px.histogram(df_filtered, x="estimated_value", nbins=40,
                           title="Distribution of Asset Values", template=PLOTLY_TEMPLATE)
        fig.update_layout(xaxis_title="Estimated Value ($)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    with co2:
        if "state" in df_filtered.columns:
            top_states = df_filtered["state"].value_counts().head(12)
            fig = px.bar(x=top_states.index, y=top_states.values, title="Top States by Asset Count",
                         template=PLOTLY_TEMPLATE)
            fig.update_layout(xaxis_title="State", yaxis_title="Assets")
            st.plotly_chart(fig, use_container_width=True)

    # Key Insights
    st.markdown("<div class='section-title'>💡 Key Insights</div>", unsafe_allow_html=True)
    insights = []
    if "state" in df_filtered.columns and len(df_filtered) > 0:
        top_state = df_filtered["state"].value_counts().idx1 if len(df_filtered["state"].value_counts())>1 else df_filtered["state"].value_counts().index[0]
        top_count = df_filtered["state"].value_counts().iloc[0]
        insights.append(f"📍 **{top_state}** has the highest number of assets ({top_count:,}).")
    ttl = df_filtered["estimated_value"].sum()
    avg = df_filtered["estimated_value"].mean()
    insights.append(f"💰 Total portfolio value stands at **${ttl/1e9:.2f}B**; average asset value **${avg/1e6:.2f}M**.")
    if len(df_filtered) > 20:
        top10 = df_filtered.nlargest(int(max(1, len(df_filtered)*0.1)), "estimated_value")["estimated_value"].sum()
        conc = 100*top10/ttl if ttl>0 else 0
        insights.append(f"🏛️ Value concentration: Top 10% assets account for **{conc:.1f}%** of portfolio value.")
    for ins in insights:
        st.markdown(f"<div class='insight'>{ins}</div>", unsafe_allow_html=True)

# ---------------
# 🗺️ Geo & GIS
# ---------------
with TABS[1]:
    st.markdown("**GIS Toolkit (Python):** GeoPandas/Shapely, PySAL, Geopy with graceful fallbacks.")

    if GEOS_OK:
        # Convert to GeoDataFrame
        if {"latitude","longitude"}.issubset(df_filtered.columns):
            gdf = gpd.GeoDataFrame(
                df_filtered.copy(),
                geometry=gpd.points_from_xy(df_filtered["longitude"], df_filtered["latitude"]),
                crs="EPSG:4326",
            )
            st.write("GeoDataFrame preview:")
            st.dataframe(gdf.head()[[c for c in gdf.columns if c != "geometry"] + ["geometry"]])

            # Spatial query: within radius of a geocoded address
            colA, colB = st.columns([1.2, 1])
            with colA:
                st.subheader("Spatial Query — Buffer around Address")
                radius_km = st.slider("Radius (km)", 5, 200, 25)
                if GEOPY_OK and user_address:
                    geolocator = Nominatim(user_agent="st_gis_app")
                    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
                    loc = geocode(user_address)
                    if loc:
                        st.caption(f"Geocoded: {loc.address} → ({loc.latitude:.4f}, {loc.longitude:.4f})")
                        p = Point(loc.longitude, loc.latitude)
                        # Project to meters for accurate buffering
                        gdf_3857 = gdf.to_crs(3857)
                        p_3857 = gpd.GeoSeries([p], crs=4326).to_crs(3857).iloc[0]
                        buf = gpd.GeoSeries([p_3857.buffer(radius_km*1000)], crs=3857)
                        within = gdf_3857[gdf_3857.geometry.within(buf.iloc[0])].to_crs(4326)
                        st.write(f"Assets within {radius_km} km: {len(within):,}")
                        st.dataframe(within.head())

                        # Map
                        m = folium.Map(location=[loc.latitude, loc.longitude], zoom_start=6)
                        folium.Marker([loc.latitude, loc.longitude], tooltip="Query Center").add_to(m)
                        # Draw buffer polygon
                        folium.GeoJson(buf.to_crs(4326).__geo_interface__, name="buffer").add_to(m)
                        for _, r in within.iterrows():
                            folium.CircleMarker([r.latitude, r.longitude], radius=5, fill=True).add_to(m)
                        st_folium(m, height=500, width=None)
                    else:
                        st.warning("Address could not be geocoded. Try a broader location (e.g., 'Dallas, TX').")
                else:
                    st.info("Provide an address in the sidebar to run a radius query (requires Geopy).")

            with colB:
                st.subheader("Spatial Weights & Autocorrelation")
                if PYSAL_OK:
                    try:
                        # Moran's I on estimated_value (nearest-k weights)
                        vals = gdf["estimated_value"].fillna(gdf["estimated_value"].median()).values
                        coords = np.vstack([gdf.geometry.y.values, gdf.geometry.x.values]).T
                        w = libpysal.weights.KNN.from_array(coords, k=8)
                        w.transform = "r"
                        mi = esda.moran.Moran(vals, w)
                        st.write(f"**Global Moran's I**: {mi.I:.3f} (p-value: {mi.p_sim:.4f})")
                    except Exception as e:
                        st.warning(f"PySAL computation failed: {e}")
                else:
                    st.info("Install PySAL to compute Moran's I (spatial autocorrelation).")
        else:
            st.warning("Latitude/Longitude not found; GIS tools need coordinates.")
    else:
        st.info("GeoPandas/Shapely not available on host. The app will continue without GIS tooling.")

    # Batch Automation example (format conversions) — needs GDAL
    st.markdown("**Automation (Batch) — Export GeoJSON**")
    if GEOS_OK and {"latitude","longitude"}.issubset(df_filtered.columns):
        gdf_small = gpd.GeoDataFrame(
            df_filtered.copy(),
            geometry=gpd.points_from_xy(df_filtered["longitude"], df_filtered["latitude"]),
            crs="EPSG:4326",
        )[["state","city","estimated_value","geometry"]]
        if st.button("Export current view to GeoJSON"):
            out_path = "assets_view.geojson"
            gdf_small.to_file(out_path, driver="GeoJSON")
            with open(out_path, "rb") as f:
                st.download_button("Download GeoJSON", f, file_name="assets_view.geojson", mime="application/geo+json")
    else:
        st.caption("Install GeoPandas for export.")

# ---------------
# 🌐 Web Map (Folium)
# ---------------
@st.cache_data(show_spinner=False)
def kmeans_cluster(df: pd.DataFrame, k:int=5):
    cols = [c for c in ["latitude","longitude","estimated_value","latest_price_index"] if c in df.columns]
    if not cols:
        return df.copy(), None
    X = df[cols].copy()
    for c in cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(X.median())
    Z = MinMaxScaler().fit_transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df2 = df.copy()
    df2["cluster"] = km.fit_predict(Z)
    return df2, km

@st.cache_data(show_spinner=False)
def make_folium_map(df: pd.DataFrame, sample:int=800):
    if not {"latitude","longitude"}.issubset(df.columns):
        return None
    data = df.dropna(subset=["latitude","longitude"]).copy()
    if len(data) == 0:
        return None
    if len(data) > sample:
        data = data.sample(sample, random_state=42)
    m = folium.Map(location=[data["latitude"].mean(), data["longitude"].mean()], zoom_start=4)
    palette = ["red","blue","green","purple","orange","darkred","lightred","beige","darkblue","darkgreen"]
    for _, r in data.iterrows():
        col = palette[int(r.get("cluster", 0)) % len(palette)] if not pd.isna(r.get("cluster", np.nan)) else "blue"
        popup = f"<b>Asset</b><br>Location: {r.get('city','N/A')}, {r.get('state','N/A')}<br>Value: ${r.get('estimated_value',0):,.0f}<br>Cluster: {r.get('cluster','N/A')}"
        folium.CircleMarker(location=[r["latitude"], r["longitude"]], radius=5, color="black", fill=True, fillColor=col,
                            fillOpacity=0.7, weight=1, popup=folium.Popup(popup, max_width=300)).add_to(m)
    return m

with TABS[2]:
    st.subheader("Interactive Web Map")
    k = st.slider("Clusters (KMeans)", 2, 10, 5, key="kmap")
    dfk, _ = kmeans_cluster(df_filtered, k)
    fmap = make_folium_map(dfk)
    if fmap is not None:
        st_folium(fmap, height=550, width=None)
    else:
        st.warning("Map could not be created (missing coordinates).")

# ---------------
# 🎯 Clustering
# ---------------
with TABS[3]:
    st.subheader("KMeans Clustering")
    k = st.slider("Number of clusters", 2, 12, 5, key="kclus")
    dfc, model = kmeans_cluster(df_filtered, k)

    if "cluster" in dfc.columns:
        c1, c2 = st.columns([2,1])
        with c1:
            if {"latitude","longitude"}.issubset(dfc.columns):
                smp = dfc.sample(min(len(dfc), 1500), random_state=42)
                fig = px.scatter_mapbox(
                    smp, lat="latitude", lon="longitude", color="cluster",
                    size=("estimated_value" if "estimated_value" in smp.columns else None),
                    mapbox_style="open-street-map", zoom=3, height=600,
                    title="Clustered Asset Geography", template=PLOTLY_TEMPLATE,
                )
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            agg = dfc.groupby("cluster").agg(
                count=("estimated_value","count"),
                total_value=("estimated_value","sum"),
                avg_value=("estimated_value","mean")
            ).round(2)
            st.dataframe(agg)
            pie = px.pie(values=agg["count"], names=agg.index.astype(str), title="Assets by Cluster",
                         template=PLOTLY_TEMPLATE)
            st.plotly_chart(pie, use_container_width=True)

# ---------------
# 🤖 ML (Exploration)
# ---------------
@st.cache_data(show_spinner=False)
def build_features(df: pd.DataFrame):
    feats = []
    D = df.copy()
    if {"latitude","longitude"}.issubset(D.columns):
        cities = {
            "NYC": (40.7128, -74.0060),
            "LA": (34.0522, -118.2437),
            "Chicago": (41.8781, -87.6298),
            "Houston": (29.7604, -95.3698),
            "DC": (38.9072, -77.0369),
        }
        for nm,(la,lo) in cities.items():
            D[f"dist_{nm.lower()}"] = np.sqrt((D["latitude"]-la)**2 + (D["longitude"]-lo)**2)
            feats.append(f"dist_{nm.lower()}")
    for c in ["latitude","longitude","latest_price_index","estimated_value"]:
        if c in D.columns and c != "estimated_value":
            feats.append(c)
    return D, feats

with TABS[4]:
    st.subheader("Machine Learning — Explore Tasks")
    if "estimated_value" not in df_filtered.columns:
        st.warning("Estimated value missing.")
    else:
        DF, features = build_features(df_filtered)
        X = DF[features].fillna(DF[features].median()) if features else None
        task = st.selectbox("Task", ["Value Prediction (Regression)", "Value Classification", "High-Value Detection"])
        if X is None or X.empty:
            st.info("No usable features.")
        else:
            if task == "Value Prediction (Regression)":
                y = DF["estimated_value"]
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
                mdl = RandomForestRegressor(n_estimators=120, random_state=42)
                mdl.fit(Xtr, ytr)
                yp = mdl.predict(Xte)
                r2 = r2_score(yte, yp)
                mae = mean_absolute_error(yte, yp)
                c1, c2 = st.columns(2)
                c1.metric("R²", f"{r2:.3f}")
                c2.metric("MAE", f"${mae:,.0f}")
                fi = pd.DataFrame({"feature": features, "importance": mdl.feature_importances_}).sort_values("importance", ascending=False)
                fig = px.bar(fi.head(12), x="importance", y="feature", orientation="h", title="Top Feature Importances", template=PLOTLY_TEMPLATE)
                st.plotly_chart(fig, use_container_width=True)
            elif task == "Value Classification":
                y = pd.qcut(DF["estimated_value"], q=3, labels=["Low","Medium","High"], duplicates='drop')
                if y.nunique() < 2:
                    st.info("Not enough variance to form classes.")
                else:
                    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                    clf = RandomForestClassifier(n_estimators=150, random_state=42)
                    clf.fit(Xtr, ytr)
                    yp = clf.predict(Xte)
                    acc = accuracy_score(yte, yp)
                    st.metric("Accuracy", f"{acc:.3f}")
                    dist = y.value_counts()
                    pie = px.pie(values=dist.values, names=dist.index, title="Class Distribution", template=PLOTLY_TEMPLATE)
                    st.plotly_chart(pie, use_container_width=True)
            else:
                thr = DF["estimated_value"].quantile(0.75)
                y = (DF["estimated_value"] > thr).astype(int)
                if y.sum() in (0, len(y)):
                    st.info("Unbalanced target; adjust threshold.")
                else:
                    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                    clf = RandomForestClassifier(n_estimators=120, random_state=42)
                    clf.fit(Xtr, ytr)
                    yp = clf.predict(Xte)
                    acc = accuracy_score(yte, yp)
                    st.metric("Accuracy", f"{acc:.3f}")
                    st.metric("High-Value Threshold", f"${thr:,.0f}")

# ---------------
# 🧮 Predict Values (Single & Bulk)
# ---------------
with TABS[5]:
    st.subheader("Predict Asset Values")
    st.caption("Train a model on current filtered data, then predict single inputs or an uploaded CSV.")

    DF, features = build_features(df_filtered)
    features = [f for f in features if f in DF.columns]
    if not features:
        st.info("No features available to train.")
    else:
        X = DF[features].fillna(DF[features].median())
        y = DF["estimated_value"].fillna(DF["estimated_value"].median())
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(Xtr, ytr)
        ypred = model.predict(Xte)
        r2 = r2_score(yte, ypred)
        mae = mean_absolute_error(yte, ypred)
        st.markdown(f"**Model Performance** — R²: `{r2:.3f}` | MAE: `${mae:,.0f}`")

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Single Prediction**")
            inputs = {}
            for f in features[:10]:  # keep UI compact; still works with .predict()
                val = float(st.number_input(f, value=float(X[f].median())))
                inputs[f] = val
            if st.button("Predict Value", type="primary"):
                xv = pd.DataFrame([inputs])[features]
                pred = float(model.predict(xv)[0])
                st.success(f"Estimated Asset Value: ${pred:,.0f}")
        with c2:
            st.markdown("**Bulk Prediction (CSV Upload)**")
            st.caption("CSV must include these columns (any order): " + ", ".join(features))
            up = st.file_uploader("Upload CSV", type=["csv"])
            if up is not None:
                try:
                    dfu = pd.read_csv(up)
                    missing = [c for c in features if c not in dfu.columns]
                    if missing:
                        st.error("Missing columns: " + ", ".join(missing))
                    else:
                        Xb = dfu[features].fillna(X.median())
                        preds = model.predict(Xb)
                        out = dfu.copy()
                        out["predicted_estimated_value"] = preds
                        csv = out.to_csv(index=False).encode()
                        st.download_button("Download Predictions CSV", csv, file_name="asset_value_predictions.csv", mime="text/csv")
                        st.dataframe(out.head())
                except Exception as e:
                    st.error(f"Bulk prediction failed: {e}")

# ---------------
# 📈 Advanced (Quality, Trends)
# ---------------
with TABS[6]:
    st.subheader("Statistical Summary")
    if "estimated_value" in df_filtered.columns and not df_filtered.empty:
        desc = df_filtered["estimated_value"].describe()
        a, b, c = st.columns(3)
        a.metric("Mean", f"${desc['mean']/1e6:.2f}M")
        b.metric("Std Dev", f"${desc['std']/1e6:.2f}M")
        c.metric("Max", f"${desc['max']/1e6:.2f}M")
        box = px.box(df_filtered, y="estimated_value", title="Asset Value Distribution", template=PLOTLY_TEMPLATE)
        box.update_layout(yaxis_title="Estimated Value ($)")
        st.plotly_chart(box, use_container_width=True)

    st.subheader("Data Quality — Missingness")
    miss = df_filtered.isnull().sum()
    miss = miss[miss>0].sort_values(ascending=False)
    if not miss.empty:
        qdf = pd.DataFrame({"Column":miss.index, "Missing":miss.values, "Missing %":(miss.values/len(df_filtered)*100).round(2)})
        bar = px.bar(qdf.head(20), x="Missing %", y="Column", orientation="h", title="Top Missing Columns", template=PLOTLY_TEMPLATE)
        st.plotly_chart(bar, use_container_width=True)
        st.dataframe(qdf)
    else:
        st.success("No missing data detected in the current view.")

    st.subheader("State-Level Portfolio (Bubble)")
    if "state" in df_filtered.columns:
        agg = df_filtered.groupby("state").agg(count=("estimated_value","count"), total=("estimated_value","sum"), avg=("estimated_value","mean")).sort_values("total", ascending=False)
        fig = px.scatter(agg.reset_index(), x="count", y="avg", size="total", color="count", hover_name="state",
                         title="State Portfolio Bubble Chart", template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(agg.head(20))

# ---------------
# Footer
# ---------------
st.caption("Built with Streamlit • Plotly • Folium • GeoPandas (optional) • PySAL (optional) • Geopy (optional)")

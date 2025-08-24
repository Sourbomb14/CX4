# app.py — US Government Assets Portfolio Analytics (Hardened + Sampling + Prediction)
# ---------------------------------------------------------------------------------
# ✅ Google Drive links only for core datasets (assets + Zillow)
# ✅ After merging, sample EXACTLY 10,000 rows with random_state=4742271 for analysis
# ✅ Dark/Light theme–aware visuals
# ✅ GIS utilities (GeoPandas/Shapely, optional PySAL/Geopy fallbacks)
# ✅ Clustering (KMeans), ML (regression/classification/high-value), Predict tab
# ✅ Upload CSV → predict asset prices using the trained model
# ✅ Robust guards to avoid runtime errors in Streamlit Cloud
# ---------------------------------------------------------------------------------

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import streamlit as st
from streamlit_folium import st_folium
import plotly.express as px

# GIS & mapping
import folium
try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOS_OK = True
except Exception:
    GEOS_OK = False

# Optional heavy libs (graceful fallback)
try:
    import libpysal
    import esda  # Moran's I, LISA
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
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# ------------------
# Page config & Theming
# ------------------
st.set_page_config(
    page_title="US Government Assets Portfolio Analytics",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    base_theme = st.get_option("theme.base")
    PLOTLY_TEMPLATE = "plotly_dark" if base_theme == "dark" else "plotly"
except Exception:
    PLOTLY_TEMPLATE = "plotly"

st.markdown(
    """
    <style>
      :root { --brand:#1f4e79; --card-bg:#f6f8fb; --ink:#0f172a; }
      @media (prefers-color-scheme: dark) {
        :root { --card-bg:#111827; --ink:#e5e7eb; }
      }
      .main-header { font-size:2.2rem; font-weight:800; color:var(--brand); text-align:center; margin:.5rem 0 1rem; }
      .kpi { background: var(--card-bg); border-radius: 16px; padding: 16px; border: 1px solid rgba(0,0,0,0.05); }
      .insight { background:rgba(31,78,121,0.08); border-left:6px solid var(--brand); padding:12px 14px; border-radius:10px; margin-bottom:10px; }
      .section-title { font-size:1.2rem; font-weight:700; margin:.5rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------
# Data Loading (Google Drive)
# ------------------
ASSETS_GDRIVE_ID = "1YFTWJNoxu0BF8UlMDXI8bXwRTVQNE2mb"
ZILLOW_GDRIVE_ID = "1fFT8Q8GWiIEM7kx6czhQ-qabygUPBQRv"
ASSETS_URL = f"https://drive.google.com/uc?id={ASSETS_GDRIVE_ID}"
ZILLOW_URL = f"https://drive.google.com/uc?id={ZILLOW_GDRIVE_ID}"

@st.cache_data(show_spinner=False)
def load_csv_from_gdrive(url: str, cache_name: str):
    import requests
    path = f"{cache_name}.csv"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        return pd.read_csv(path)
    except Exception:
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception as e:
                st.error(f"Failed reading cached {path}: {e}")
        return None

with st.spinner("Loading datasets from Google Drive…"):
    df_assets_raw = load_csv_from_gdrive(ASSETS_URL, "assets_cached")
    df_hpi_raw = load_csv_from_gdrive(ZILLOW_URL, "zillow_cached")

# ------------------
# Data Preparation & Merge + REQUIRED SAMPLING (10,000 @ 4742271)
# ------------------
RANDOM_STATE = 4742271
TARGET_COL = "estimated_value"

@st.cache_data(show_spinner=False)
def clean_and_merge_data(df_assets: pd.DataFrame, df_hpi: pd.DataFrame) -> pd.DataFrame:
    if df_assets is None or df_assets.empty:
        return pd.DataFrame()

    df = df_assets.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")

    # Normalize Zillow
    if df_hpi is not None and not df_hpi.empty:
        hpi = df_hpi.copy()
        hpi.columns = hpi.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
        # Latest price index from 2025/2024 columns if any
        date_cols = [c for c in hpi.columns if any(y in c for y in ["2025", "2024"])]
        if date_cols:
            latest_col = sorted(date_cols)[-1]
            hpi["latest_price_index"] = pd.to_numeric(hpi[latest_col], errors="coerce")
        else:
            hpi["latest_price_index"] = np.nan

        # Merge on city/state when available
        if {"city","state"}.issubset(df.columns) and {"city","state"}.issubset(hpi.columns):
            key_df = (df["city"].astype(str).str.lower().str.strip() + "_" + df["state"].astype(str).str.lower().str.strip())
            key_hpi = (hpi["city"].astype(str).str.lower().str.strip() + "_" + hpi["state"].astype(str).str.lower().str.strip())
            df["city_state_key"] = key_df
            hpi["city_state_key"] = key_hpi
            df = df.merge(hpi[["city_state_key","latest_price_index"].copy()], on="city_state_key", how="left")

    # Coordinates sanity check
    if {"latitude","longitude"}.issubset(df.columns):
        df = df[(pd.to_numeric(df["latitude"], errors="coerce").between(24, 49)) &
                (pd.to_numeric(df["longitude"], errors="coerce").between(-125, -66))]

    # Target proxy if missing
    if "latest_price_index" not in df.columns:
        df["latest_price_index"] = np.nan

    # Compute estimated value
    rentable_col = next((c for c in df.columns if ("rentable" in c or "sqft" in c or "square" in c) and ("feet" in c or "ft" in c or "sq" in c)), None)
    if rentable_col is not None:
        df[TARGET_COL] = (
            pd.to_numeric(df[rentable_col], errors="coerce").fillna(0)
            * (df["latest_price_index"].fillna(df["latest_price_index"].median()) / 100.0)
            * 10.0
        )
    else:
        rng = np.random.default_rng(RANDOM_STATE)
        df[TARGET_COL] = (
            df["latest_price_index"].fillna(df["latest_price_index"].median())
            * rng.uniform(0.5, 2.0, len(df))
        )

    # Premium for high-value states
    high_value_states = ["CA","NY","MA","CT","NJ","HI","MD","WA"]
    if "state" in df.columns:
        df.loc[df["state"].isin(high_value_states), TARGET_COL] *= 1.5

    # SAMPLE 10,000 rows deterministically
    if len(df) > 10000:
        df = df.sample(n=10000, random_state=RANDOM_STATE)

    return df.reset_index(drop=True)

with st.spinner("Cleaning & merging…"):
    df_merged = clean_and_merge_data(df_assets_raw, df_hpi_raw)

if df_merged is None or df_merged.empty:
    st.error("No data available. Please check your Google Drive CSVs.")
    st.stop()

# ------------------
# Sidebar — Filters & Controls
# ------------------
st.sidebar.image(
    "https://via.placeholder.com/300x100/1f4e79/ffffff?text=Portfolio+Analytics",
    use_container_width=True,
)

st.sidebar.markdown("### 🔍 Filters")
state_sel = "All"
if "state" in df_merged.columns and not df_merged["state"].dropna().empty:
    states = ["All"] + sorted(df_merged["state"].dropna().unique().tolist())
    state_sel = st.sidebar.selectbox("State", states)

min_val = float(df_merged[TARGET_COL].min()) if TARGET_COL in df_merged.columns else 0.0
max_val = float(df_merged[TARGET_COL].max()) if TARGET_COL in df_merged.columns else 1.0
if min_val > max_val:
    min_val, max_val = 0.0, 1.0
vl, vh = st.sidebar.slider(
    "Asset Value Range ($)", min_value=int(min_val), max_value=int(max_val),
    value=(int(min_val), int(max_val)), format="$%d"
)

# Optional address (for GIS tab)
user_address = st.sidebar.text_input("Optional: Address to geocode (city, state or full)")

# Apply filters
df_filtered = df_merged.copy()
if state_sel != "All" and "state" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["state"] == state_sel]

if TARGET_COL in df_filtered.columns:
    df_filtered = df_filtered[df_filtered[TARGET_COL].between(vl, vh)]

# ------------------
# Top Tabs
# ------------------
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

# ------------------
# 📊 Executive
# ------------------
with TABS[0]:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='kpi'>Total Assets<br><h2>"+f"{len(df_filtered):,}"+"</h2></div>", unsafe_allow_html=True)
    with c2:
        ttl = float(df_filtered[TARGET_COL].sum()) if TARGET_COL in df_filtered.columns else 0.0
        st.markdown("<div class='kpi'>Portfolio Value<br><h2>"+f"${ttl/1e9:.2f}B"+"</h2></div>", unsafe_allow_html=True)
    with c3:
        avg = float(df_filtered[TARGET_COL].mean()) if TARGET_COL in df_filtered.columns else 0.0
        st.markdown("<div class='kpi'>Average Asset Value<br><h2>"+f"${avg/1e6:.2f}M"+"</h2></div>", unsafe_allow_html=True)
    with c4:
        st_cnt = df_filtered["state"].nunique() if "state" in df_filtered.columns else 0
        st.markdown("<div class='kpi'>States Covered<br><h2>"+f"{int(st_cnt)}"+"</h2></div>", unsafe_allow_html=True)

    co1, co2 = st.columns(2)
    with co1:
        if TARGET_COL in df_filtered.columns and not df_filtered[TARGET_COL].dropna().empty:
            fig = px.histogram(df_filtered, x=TARGET_COL, nbins=40, title="Distribution of Asset Values", template=PLOTLY_TEMPLATE)
            fig.update_layout(xaxis_title="Estimated Value ($)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Estimated value data not available for histogram.")
    with co2:
        if "state" in df_filtered.columns and not df_filtered.empty:
            vc = df_filtered["state"].value_counts()
            if not vc.empty:
                tops = vc.head(12)
                fig = px.bar(x=tops.index, y=tops.values, title="Top States by Asset Count", template=PLOTLY_TEMPLATE)
                fig.update_layout(xaxis_title="State", yaxis_title="Assets")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No state counts available.")
        else:
            st.info("State column not available.")

    st.markdown("<div class='section-title'>💡 Key Insights</div>", unsafe_allow_html=True)
    insights = []
    if "state" in df_filtered.columns and not df_filtered.empty:
        vc = df_filtered["state"].value_counts()
        if not vc.empty:
            top_state = vc.idxmax()
            top_count = int(vc.iloc[0])
            insights.append(f"📍 **{top_state}** has the highest number of assets ({top_count:,}).")
    if TARGET_COL in df_filtered.columns and not df_filtered[TARGET_COL].dropna().empty:
        ttl = df_filtered[TARGET_COL].sum()
        avg = df_filtered[TARGET_COL].mean()
        insights.append(f"💰 Total portfolio value: **${ttl/1e9:.2f}B**; average asset value **${avg/1e6:.2f}M**.")
        if len(df_filtered) > 20 and ttl > 0:
            n_top = max(1, int(len(df_filtered) * 0.1))
            top10 = df_filtered.nlargest(n_top, TARGET_COL)[TARGET_COL].sum()
            conc = 100 * top10 / ttl
            insights.append(f"🏛️ Value concentration: Top 10% assets account for **{conc:.1f}%** of total value.")
    if insights:
        for ins in insights:
            st.markdown(f"<div class='insight'>{ins}</div>", unsafe_allow_html=True)
    else:
        st.info("No insights available for current filters.")

# ------------------
# Helpers for clustering & maps
# ------------------
@st.cache_data(show_spinner=False)
def kmeans_cluster(df: pd.DataFrame, k:int=5):
    if df is None or df.empty:
        return df.copy(), None
    cols = [c for c in ["latitude","longitude",TARGET_COL,"latest_price_index"] if c in df.columns]
    if not cols:
        return df.copy(), None
    X = df[cols].copy()
    for c in cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    if len(X) < max(2, k):
        return df.copy(), None
    Z = MinMaxScaler().fit_transform(X)
    try:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        df2 = df.copy()
        df2["cluster"] = km.fit_predict(Z)
        return df2, km
    except Exception:
        return df.copy(), None

@st.cache_data(show_spinner=False)
def make_folium_map(df: pd.DataFrame, sample:int=800):
    if df is None or df.empty:
        return None
    if not {"latitude","longitude"}.issubset(df.columns):
        return None
    data = df.dropna(subset=["latitude","longitude"]).copy()
    if len(data) == 0:
        return None
    if len(data) > sample:
        data = data.sample(sample, random_state=RANDOM_STATE)
    m = folium.Map(location=[float(data["latitude"].mean()), float(data["longitude"].mean())], zoom_start=4)
    palette = ["red","blue","green","purple","orange","darkred","lightred","beige","darkblue","darkgreen"]
    for _, r in data.iterrows():
        try:
            col = palette[int(r.get("cluster", 0)) % len(palette)]
            popup = f"<b>Asset</b><br>State: {r.get('state','')}<br>City: {r.get('city','')}<br>Est. Value: ${r.get(TARGET_COL,0):,.0f}"
            folium.CircleMarker(location=[float(r["latitude"]), float(r["longitude"])], radius=5, color="black", fill=True, fill_color=col, fill_opacity=0.7, popup=folium.Popup(popup, max_width=250)).add_to(m)
        except Exception:
            continue
    return m

# ------------------
# 🗺️ Geo & GIS
# ------------------
with TABS[1]:
    st.subheader("Geo & GIS Analysis")
    if not {"latitude","longitude"}.issubset(df_filtered.columns):
        st.info("No coordinates available in current dataset.")
    else:
        # Spatial autocorrelation (optional)
        if PYSAL_OK:
            try:
                import numpy as np
                coords = df_filtered[["latitude","longitude"]].dropna()
                vals = df_filtered.loc[coords.index, TARGET_COL].fillna(df_filtered[TARGET_COL].median())
                if len(coords) > 20:
                    w = libpysal.weights.KNN.from_dataframe(pd.DataFrame({"lat":coords["latitude"],"lon":coords["longitude"]}), k=8)
                    w.transform = "r"
                    mi = esda.Moran(vals.values, w)
                    st.metric("Global Moran's I", f"{mi.I:.3f}")
            except Exception:
                st.info("PySAL Moran's I not available.")

        # Map
        df_c, _ = kmeans_cluster(df_filtered, k=5)
        fmap = make_folium_map(df_c, sample=800)
        if fmap is not None:
            st_folium(fmap, width=900, height=520)
        else:
            st.info("Map could not be rendered for current selection.")

# ------------------
# 🌐 Web Map (Mapbox Scatter)
# ------------------
with TABS[2]:
    st.subheader("Interactive Map (Mapbox)")
    if {"latitude","longitude"}.issubset(df_filtered.columns) and len(df_filtered.dropna(subset=["latitude","longitude"]))>0:
        sample_df = df_filtered.dropna(subset=["latitude","longitude"]).copy()
        if len(sample_df) > 5000:
            sample_df = sample_df.sample(5000, random_state=RANDOM_STATE)
        fig = px.scatter_mapbox(sample_df, lat="latitude", lon="longitude", color="state" if "state" in sample_df.columns else None, size=TARGET_COL if TARGET_COL in sample_df.columns else None, hover_data={TARGET_COL:':$,.0f'} if TARGET_COL in sample_df.columns else None, zoom=3, height=600, template=PLOTLY_TEMPLATE)
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No coordinate data to show.")

# ------------------
# 🎯 Clustering
# ------------------
with TABS[3]:
    st.subheader("Clustering Analysis (K-Means)")
    k = st.slider("Clusters", 2, 10, 5)
    df_c, km = kmeans_cluster(df_filtered, k=k)
    if km is None or "cluster" not in df_c.columns:
        st.info("Clustering not available for the current subset.")
    else:
        st.write("Cluster counts:", df_c["cluster"].value_counts().to_frame("count"))
        if {"latitude","longitude"}.issubset(df_c.columns):
            smp = df_c.dropna(subset=["latitude","longitude"]).copy()
            if len(smp) > 5000:
                smp = smp.sample(5000, random_state=RANDOM_STATE)
            fig = px.scatter_mapbox(smp, lat="latitude", lon="longitude", color="cluster", size=TARGET_COL if TARGET_COL in smp.columns else None, zoom=3, height=600, template=PLOTLY_TEMPLATE)
            fig.update_layout(mapbox_style="carto-positron")
            st.plotly_chart(fig, use_container_width=True)

# ------------------
# 🤖 ML
# ------------------
@st.cache_data(show_spinner=False)
def build_features(df: pd.DataFrame):
    feats = []
    if {"latitude","longitude"}.issubset(df.columns):
        major = {
            'nyc': (40.7128, -74.0060),
            'la': (34.0522, -118.2437),
            'chicago': (41.8781, -87.6298),
            'houston': (29.7604, -95.3698),
            'dc': (38.9072, -77.0369)
        }
        for k,(lat,lon) in major.items():
            df[f"dist_{k}"] = np.sqrt((pd.to_numeric(df['latitude'], errors='coerce')-lat)**2 + (pd.to_numeric(df['longitude'], errors='coerce')-lon)**2)
            feats.append(f"dist_{k}")
    for c in ["latitude","longitude","latest_price_index"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            feats.append(c)
    # rentable sqft like columns
    rent_col = next((c for c in df.columns if ("rentable" in c or "sqft" in c or "square" in c) and ("feet" in c or "ft" in c or "sq" in c)), None)
    if rent_col:
        df[rent_col] = pd.to_numeric(df[rent_col], errors='coerce')
        feats.append(rent_col)
    return df, feats

with TABS[4]:
    st.subheader("Machine Learning Exploration")
    if TARGET_COL not in df_filtered.columns or df_filtered[TARGET_COL].dropna().empty:
        st.info("No target available for ML.")
    else:
        df_ml = df_filtered.dropna(subset=[TARGET_COL]).copy()
        df_ml, feats = build_features(df_ml)
        feats = [f for f in feats if f in df_ml.columns]
        if len(feats) < 2 or len(df_ml) < 50:
            st.info("Not enough data/features for ML.")
        else:
            X = df_ml[feats].fillna(df_ml[feats].median(numeric_only=True))
            y = df_ml[TARGET_COL]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
            task = st.selectbox("Task", ["Value Prediction (Regression)", "Value Classification", "High-Value Detection"]) 
            if task == "Value Prediction (Regression)":
                model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.metric("R²", f"{r2_score(y_test, y_pred):.3f}")
                st.metric("MAE", f"${mean_absolute_error(y_test, y_pred):,.0f}")
                st.session_state["trained_model"] = (model, feats)
            elif task == "Value Classification":
                # terciles
                y_bins = pd.qcut(y, q=3, labels=['Low','Medium','High'], duplicates='drop')
                X_tr, X_te, y_tr, y_te = train_test_split(X, y_bins, test_size=0.2, random_state=RANDOM_STATE, stratify=y_bins)
                clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
                clf.fit(X_tr, y_tr)
                acc = accuracy_score(y_te, clf.predict(X_te))
                st.metric("Accuracy", f"{acc:.3f}")
                st.session_state["trained_classifier"] = (clf, feats)
            else:
                thr = y.quantile(0.75)
                y_bin = (y > thr).astype(int)
                X_tr, X_te, y_tr, y_te = train_test_split(X, y_bin, test_size=0.2, random_state=RANDOM_STATE, stratify=y_bin)
                clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
                clf.fit(X_tr, y_tr)
                acc = accuracy_score(y_te, clf.predict(X_te))
                st.metric("Accuracy", f"{acc:.3f}")
                st.metric("High-Value Threshold", f"${thr:,.0f}")
                st.session_state["trained_classifier_bin"] = (clf, feats, thr)

# ------------------
# 🧮 Predict Values (incl. CSV upload)
# ------------------
with TABS[5]:
    st.subheader("Predict Asset Values")
    model_pack = st.session_state.get("trained_model")

    # If user hasn't trained this session, auto-train on entire merged SAMPLE to ensure availability
    if model_pack is None and TARGET_COL in df_merged.columns and not df_merged[TARGET_COL].dropna().empty:
        base_df = df_merged.dropna(subset=[TARGET_COL]).copy()
        base_df, feats = build_features(base_df)
        feats = [f for f in feats if f in base_df.columns]
        if len(base_df) > 20000:
            base_df = base_df.sample(20000, random_state=RANDOM_STATE)
        Xb = base_df[feats].fillna(base_df[feats].median(numeric_only=True))
        yb = base_df[TARGET_COL]
        base_model = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE)
        base_model.fit(Xb, yb)
        model_pack = (base_model, feats)
        st.info("A baseline model was trained automatically on the merged sample.")
        st.session_state["trained_model"] = model_pack

    if model_pack is None:
        st.warning("No regression model available. Please train one in the ML tab first.")
    else:
        model, feats = model_pack
        # Single prediction form
        with st.form("single_pred"):
            st.write("Single-record prediction")
            lat = st.number_input("Latitude", value=37.0, step=0.01)
            lon = st.number_input("Longitude", value=-95.0, step=0.01)
            lpi = st.number_input("Latest Price Index", value=250000.0, step=1000.0)
            sqft = st.number_input("Rentable Sqft (optional)", value=50000.0, step=1000.0)
            submitted = st.form_submit_button("Predict")
            if submitted:
                tmp = pd.DataFrame({
                    'latitude':[lat], 'longitude':[lon], 'latest_price_index':[lpi], 'rentable_sqft':[sqft]
                })
                tmp, _ = build_features(tmp)
                Xp = tmp.reindex(columns=feats, fill_value=np.nan)
                Xp = Xp.fillna(Xp.median(numeric_only=True))
                pred = float(model.predict(Xp)[0])
                st.success(f"Predicted Estimated Value: ${pred:,.0f}")

        st.divider()
        # Bulk CSV upload
        up = st.file_uploader("Upload CSV to predict (columns: latitude, longitude, latest_price_index, rentable sqft if available)", type=["csv"]) 
        if up is not None:
            try:
                udf = pd.read_csv(up)
                udf.columns = udf.columns.str.lower().str.strip()
                udf, _ = build_features(udf)
                Xu = udf.reindex(columns=feats, fill_value=np.nan)
                Xu = Xu.fillna(Xu.median(numeric_only=True))
                preds = model.predict(Xu)
                out = udf.copy()
                out["predicted_estimated_value"] = preds
                out_path = "predictions_output.csv"
                out.to_csv(out_path, index=False)
                st.download_button("Download Predictions", data=out.to_csv(index=False).encode('utf-8'), file_name="predictions.csv", mime="text/csv")
                st.success(f"Predictions ready. Rows: {len(out):,}")
            except Exception as e:
                st.error(f"Failed to score uploaded CSV: {e}")

# ------------------
# 📈 Advanced
# ------------------
with TABS[6]:
    st.subheader("Advanced Analytics")
    if TARGET_COL in df_filtered.columns and not df_filtered[TARGET_COL].dropna().empty:
        stats = df_filtered[TARGET_COL].describe()
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean", f"${stats['mean']:,.0f}")
        c2.metric("Median", f"${stats['50%']:,.0f}")
        c3.metric("Std Dev", f"${stats['std']:,.0f}")
        fig = px.box(df_filtered, y=TARGET_COL, title="Asset Value Distribution", template=PLOTLY_TEMPLATE)
        fig.update_layout(yaxis_title="Estimated Value ($)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No target available for advanced analytics.")

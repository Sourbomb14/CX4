import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import plotly.express as px
import folium
from streamlit_folium import st_folium
from pysal.explore import esda
from pysal.lib import weights
from geopy.geocoders import Nominatim
import requests
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="US Asset Analytics", page_icon="🏛️", layout="wide")

# CSS for cards layout
st.markdown("""
<style>
.dashboard-header { display: flex; align-items: center; margin-bottom: 2.3rem;}
.dashboard-icon { width: 54px; height: 54px; margin-right: 1rem; }
.dashboard-title { font-size: 2.35rem; font-weight: 700; color: var(--text-color); margin-bottom: 0; }
.card-metric-row {display: flex; gap: 2.2rem; margin-bottom: 2.5rem;}
.metric-card {flex: 1; background: var(--background-color);
  border-radius: 1.1rem; box-shadow: 0 2px 24px 0 rgba(60,72,117,.13);
  padding: 1.1rem 1.3rem 1.2rem 1rem; border-left: 6px solid #1f4e79;
  min-width: 150px;}
.metric-label { font-size: 1rem; color: var(--text-color); font-weight: 500; margin-bottom: 0.6rem; margin-top: 0.3rem;}
.metric-value { font-size: 2.1rem; color: #4673a7; font-weight: 700; letter-spacing: 0.08em;}
@media (max-width: 1100px) {.card-metric-row {flex-direction: column; gap: 0.8rem;}}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_assets_data():
    url = "https://drive.google.com/uc?id=1YFTWJNoxu0BF8UlMDXI8bXwRTVQNE2mb"
    response = requests.get(url, timeout=13)
    with open("assets.csv", "wb") as f:
        f.write(response.content)
    df = pd.read_csv("assets.csv")
    return df

@st.cache_data
def load_housing_data():
    url = "https://drive.google.com/uc?id=1fFT8Q8GWiIEM7kx6czhQ-qabygUPBQRv"
    response = requests.get(url, timeout=13)
    with open("housing.csv", "wb") as f:
        f.write(response.content)
    df = pd.read_csv("housing.csv")
    return df

@st.cache_data
def clean_and_merge_data(df_assets, df_prices):
    df_assets.columns = df_assets.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    df_prices.columns = df_prices.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    # Coordinates and price index merge
    if {'latitude','longitude'}.issubset(df_assets.columns):
        df_assets = df_assets[(df_assets['latitude'] >= 24) & (df_assets['latitude'] <= 49) & (df_assets['longitude'] >= -125) & (df_assets['longitude'] <= -66)]
    price_cols = [col for col in df_prices.columns if any(y in str(col) for y in ['2024','2025'])]
    if price_cols:
        df_prices['latest_price_index'] = pd.to_numeric(df_prices[sorted(price_cols)[-1]], errors='coerce')
    if {'city','state'}.issubset(df_assets.columns) and {'city','state'}.issubset(df_prices.columns):
        cs_key = lambda df: df['city'].astype(str).str.lower().str.strip() + '_' + df['state'].astype(str).str.lower().str.strip()
        df_assets['city_state_key'] = cs_key(df_assets)
        df_prices['city_state_key'] = cs_key(df_prices)
        merged = pd.merge(df_assets, df_prices[['city_state_key','latest_price_index']], on='city_state_key', how='left')
    else:
        merged = df_assets.copy()
        merged['latest_price_index'] = np.nan
    merged['latest_price_index'] = merged['latest_price_index'].fillna(merged['latest_price_index'].median())
    rentable_col = [col for col in merged.columns if 'rentable' in col and 'feet' in col][0]
    merged['estimated_value'] = merged[rentable_col] * (merged['latest_price_index'] / 320)
    premium_states = ['CA','NY','MA','CT','NJ','HI','MD','WA']
    if 'state' in merged.columns:
        merged.loc[merged['state'].isin(premium_states),'estimated_value'] *= 1.19
    merged = merged[~merged['estimated_value'].isna()]
    return merged

def sample_data(df, n=6000, random_state=4742289):
    return df.sample(n=min(len(df), n), random_state=random_state).reset_index(drop=True)

def kmeans_clustering(gdf, n_clusters=5):
    features = ['longitude','latitude','estimated_value','latest_price_index']
    scaler = MinMaxScaler()
    X = scaler.fit_transform(gdf[features])
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=4)
    gdf["cluster_id"] = km.fit_predict(X)
    # Cluster naming: by most frequent state
    names = []
    for i in range(n_clusters):
        cluster_states = gdf[gdf["cluster_id"] == i]['state']
        dom = cluster_states.mode().iloc[0] if not cluster_states.empty else f"Cluster {i+1}"
        names.append(f"{dom} Cluster")
    gdf["cluster_name"] = gdf["cluster_id"].apply(lambda x: names[x])
    gdf["geometry"] = [Point(xy) for xy in zip(gdf.longitude, gdf.latitude)]
    return gdf, names

def spatial_autocorrelation(gdf, value_col='estimated_value'):
    w = weights.KNN.from_dataframe(gdf, k=8)
    y = gdf[value_col].values
    mi = esda.Moran(y, w)
    return mi.I, mi.p_sim

def prediction_pipeline(df_assets, df_housing):
    features = ['latitude','longitude','latest_price_index','building_rentable_square_feet']
    target = 'estimated_value'
    housingX = df_housing[features].fillna(df_housing[features].median())
    housingY = df_housing[target]
    scaler = MinMaxScaler()
    trainX_scaled = scaler.fit_transform(housingX)
    model = RandomForestRegressor(n_estimators=100, random_state=99)
    model.fit(trainX_scaled, housingY)
    testX = df_assets[features].fillna(df_assets[features].median())
    testX_scaled = scaler.transform(testX)
    preds = model.predict(testX_scaled)
    df_assets['predicted_value'] = preds
    return df_assets, model

# ---- Layout ----
def executive_dashboard(df):
    st.markdown("""
    <div class="dashboard-header">
        <img src="https://cdn-icons-png.flaticon.com/512/1984/1984368.png" class="dashboard-icon"/>
        <span class="dashboard-title">Executive Dashboard</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="card-metric-row">', unsafe_allow_html=True)
    total_assets = len(df)
    total_value = df['estimated_value'].sum()
    avg_value = df['estimated_value'].mean()
    n_states = df['state'].nunique()
    cards = [
        {'label': 'Total Assets', 'value': f"{total_assets:,}"},
        {'label': 'Portfolio Value', 'value': f"${total_value/1e9:.1f}B"},
        {'label': 'Average Asset Value', 'value': f"${avg_value/1e6:.1f}M"},
        {'label': 'States Covered', 'value': str(n_states)},
    ]
    for c in cards:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{c['label']}</div>
                <div class="metric-value">{c['value']}</div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def cluster_map(gdf):
    m = folium.Map(location=[gdf['latitude'].mean(), gdf['longitude'].mean()], zoom_start=4, tiles='cartodbpositron')
    colors = px.colors.qualitative.Pastel
    for _, row in gdf.iterrows():
        pop = f"""
        <b>{row.get('city','N/A')}, {row.get('state','N/A')}</b><br>
        <b>Cluster:</b> {row.get('cluster_name','')}<br>
        <b>Predicted Value:</b> ${row.get('predicted_value',0):,.0f}<br>
        <b>Asset Value:</b> ${row.get('estimated_value',0):,.0f}"""
        color = colors[row['cluster_id'] % len(colors)]
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=7, color=color, fill_color=color, fill_opacity=0.82,
            popup=folium.Popup(pop, max_width=360)
        ).add_to(m)
    return m

def geocode_address(address):
    geolocator = Nominatim(user_agent="asset_dashboard")
    location = geolocator.geocode(address)
    if location:
        return location.latitude, location.longitude
    return None, None

# ------- Main Streamlit Tabs -------
def main():
    df_assets = load_assets_data()
    df_housing = load_housing_data()
    df_assets = clean_and_merge_data(df_assets, df_housing)
    df_assets = sample_data(df_assets, n=5000)
    tabs = st.tabs([
        "Executive Dashboard", "Clusters & GIS Map", "EDA & Statistics", "Predictions/Evaluation", "Spatial Stats", "Geocoding"
    ])
    
    with tabs[0]:
        executive_dashboard(df_assets)
        st.header("Quick Distribution Overview")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df_assets, x='estimated_value', nbins=30, color_discrete_sequence=['#36A2EB'])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            top_states = df_assets['state'].value_counts().head(10)
            st.bar_chart(top_states)
    
    with tabs[1]:
        st.header("Clustered Asset Map (Human-Readable Names)")
        df_pred, model = prediction_pipeline(df_assets.copy(), df_housing)
        n_clusters = st.slider("Number of clusters", 3, 10, 5)
        gdf = gpd.GeoDataFrame(df_pred, geometry=gpd.points_from_xy(df_pred.longitude, df_pred.latitude))
        gdf, cluster_names = kmeans_clustering(gdf, n_clusters)
        # Table
        tbl = gdf.groupby("cluster_name")['estimated_value'].agg(['count', 'mean', 'sum']).sort_values("sum", ascending=False)
        st.dataframe(tbl, use_container_width=True)
        m = cluster_map(gdf)
        st_folium(m, width=1024, height=560)
    
    with tabs[2]:
        st.header("Descriptive & Inferential Stats")
        df_pred, model = prediction_pipeline(df_assets.copy(), df_housing)
        st.write(df_pred['estimated_value'].describe().to_frame().T)
        st.metric("Skewness", float(df_pred['estimated_value'].skew()))
        st.metric("Kurtosis", float(df_pred['estimated_value'].kurtosis()))
        fig = px.scatter(df_pred, x="estimated_value", y="predicted_value", color="state")
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        st.header("Prediction & Feature Importance")
        df_pred, model = prediction_pipeline(df_assets.copy(), df_housing)
        r2 = r2_score(df_pred["estimated_value"], df_pred["predicted_value"])
        st.metric("Prediction $R^2$", f"{r2:.3f}")
        imp_df = pd.DataFrame({
            'feature': ['latitude','longitude','latest_price_index','building_rentable_square_feet'],
            'importance': model.feature_importances_
        })
        fig3 = px.bar(imp_df.sort_values('importance', ascending=True), x="importance", y="feature", orientation="h", color="importance")
        st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(df_pred[['city','state','predicted_value','estimated_value']].sample(10), use_container_width=True)
    
    with tabs[4]:
        st.header("Spatial Autocorrelation (Moran's I)")
        gdf = gpd.GeoDataFrame(df_assets.copy(), geometry=gpd.points_from_xy(df_assets.longitude, df_assets.latitude))
        I, p = spatial_autocorrelation(gdf, 'estimated_value')
        st.metric("Moran's I", I)
        st.metric("p-value", p)
        st.write("Spatial autocorrelation: Significant if Moran's I > 0.1 and p < 0.05.")

    with tabs[5]:
        st.header("Geocoding Tool")
        address = st.text_input("Enter address to geocode", value="Times Square, NY")
        if st.button("Geocode Address"):
            lat, lon = geocode_address(address)
            st.success(f"Lat: {lat}, Lon: {lon}")

if __name__ == "__main__":
    main()

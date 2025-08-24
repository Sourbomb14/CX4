import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import davies_bouldin_score, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="US Government Assets Portfolio Analytics",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN CARD DASHBOARD CSS ---
st.markdown("""
<style>
.dashboard-header {
    display: flex;
    align-items: center;
    margin-bottom: 2.3rem;
}
.dashboard-icon {
    width: 54px;
    height: 54px;
    margin-right: 1rem;
}
.dashboard-title {
    font-size: 2.3rem;
    font-weight: 700;
    color: var(--text-color);
    margin-bottom: 0;
}
.card-metric-row {
    display: flex;
    gap: 2.2rem;
    margin-bottom: 2.5rem;
    margin-right: 2rem;
}
.metric-card {
    flex: 1;
    background: var(--background-color);
    border-radius: 1.1rem;
    box-shadow: 0 2px 24px 0 rgba(60,72,117,.13);
    padding: 1.1rem 1.3rem 1.2rem 1rem;
    border-left: 6px solid #1f4e79;
    transition: box-shadow 0.18s;
    min-width: 150px;
}
.metric-label {
    font-size: 1rem;
    color: var(--text-color);
    font-weight: 500;
    margin-bottom: 0.6rem;
    margin-top: 0.3rem;
}
.metric-value {
    font-size: 2.1rem;
    color: #4673a7;
    font-weight: 700;
    letter-spacing: 0.08em;
}
@media (max-width: 1100px) {
    .card-metric-row { flex-direction: column; gap: 0.8rem; }
}
</style>
""", unsafe_allow_html=True)

# --------------------------------
@st.cache_data
def load_assets_data():
    try:
        url = "https://drive.google.com/uc?id=1YFTWJNoxu0BF8UlMDXI8bXwRTVQNE2mb"
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            with open("assets.csv", "wb") as f:
                f.write(response.content)
            return pd.read_csv("assets.csv")
    except Exception:
        return None
    return None

@st.cache_data
def load_housing_data():
    try:
        url = "https://drive.google.com/uc?id=1fFT8Q8GWiIEM7kx6czhQ-qabygUPBQRv"
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            with open("housing.csv", "wb") as f:
                f.write(response.content)
            return pd.read_csv("housing.csv")
    except Exception:
        return None
    return None

@st.cache_data
def clean_and_merge_data(df_assets, df_prices):
    if df_assets is None or df_prices is None:
        raise ValueError("Asset and housing data is required.")
    df_assets.columns = df_assets.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    df_prices.columns = df_prices.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    # Filter to valid US coordinates
    if {'latitude','longitude'}.issubset(df_assets.columns):
        valid = (
            (df_assets['latitude'] >= 24) & (df_assets['latitude'] <= 49) &
            (df_assets['longitude'] >= -125) & (df_assets['longitude'] <= -66)
            & df_assets['latitude'].notna() & df_assets['longitude'].notna()
        )
        df_assets = df_assets[valid]
    price_cols = [col for col in df_prices.columns if any(y in str(col) for y in ['2024','2025'])]
    if price_cols:
        df_prices['latest_price_index'] = pd.to_numeric(
            df_prices[sorted(price_cols)[-1]], errors='coerce'
        )
    if {'city','state'}.issubset(df_assets.columns) and {'city','state'}.issubset(df_prices.columns):
        cs_key = lambda df: df['city'].astype(str).str.lower().str.strip() + '_' + df['state'].astype(str).str.lower().str.strip()
        df_assets['city_state_key'] = cs_key(df_assets)
        df_prices['city_state_key'] = cs_key(df_prices)
        merged = pd.merge(
            df_assets,
            df_prices[['city_state_key','latest_price_index']],
            on='city_state_key',how='left'
        )
    else:
        merged = df_assets.copy()
        merged['latest_price_index'] = np.nan
    merged['latest_price_index'] = merged['latest_price_index'].fillna(merged['latest_price_index'].median())
    rentable_col = None
    for col in merged.columns:
        if 'rentable' in col.lower() and 'feet' in col.lower():
            rentable_col = col; break
    if not rentable_col:
        raise ValueError("No rentable square feet column for value estimation.")
    # --- Realistic scaling for US value (e.g. 320 as avg price per sq ft) ---
    merged['estimated_value'] = merged[rentable_col] * (merged['latest_price_index'] / 320)
    premium_states = ['CA','NY','MA','CT','NJ','HI','MD','WA']
    if 'state' in merged.columns:
        merged.loc[merged['state'].isin(premium_states),'estimated_value'] *= 1.19
    merged = merged[~merged['estimated_value'].isna()]
    return merged

@st.cache_data
def sample_data(df, n=7500, random_state=4742271):
    if len(df) <= n:
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=random_state).reset_index(drop=True)

# ---------- Executive Dashboard Metrics --------------
def executive_dashboard(df):
    st.markdown(
        '''
        <div class="dashboard-header">
            <img src="https://cdn-icons-png.flaticon.com/512/1984/1984368.png" class="dashboard-icon"/>
            <span class="dashboard-title">Executive Dashboard</span>
        </div>
        ''', unsafe_allow_html=True
    )

    st.markdown('<div class="card-metric-row">', unsafe_allow_html=True)
    total_assets = len(df)
    total_value = df['estimated_value'].sum()
    avg_value = df['estimated_value'].mean()
    n_states = df['state'].nunique() if 'state' in df.columns else None

    cards = [
        {'label':'Total Assets', 'value':f"{total_assets:,}"},
        {'label':'Portfolio Value', 'value':f"${total_value/1e9:,.1f}B"},
        {'label':'Average Asset Value', 'value':f"${avg_value/1e6:,.1f}M"},
        {'label':'States Covered', 'value':str(n_states)},
    ]
    for c in cards:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{c['label']}</div>
                <div class="metric-value">{c['value']}</div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --------------- MAIN APP LOGIC -----------------
def main():
    with st.spinner("Loading and preparing data ..."):
        df_assets = load_assets_data()
        df_prices = load_housing_data()
        if df_assets is None or df_prices is None:
            st.error("Failed to load US government asset or housing data.")
            st.stop()
        df = clean_and_merge_data(df_assets, df_prices)
        df = sample_data(df)
    menu = ["Executive Dashboard","Clustering Analysis","Geographic Map","Asset Value Prediction"]
    choice = st.sidebar.radio("Go to", menu)
    if choice == "Executive Dashboard":
        executive_dashboard(df)

    elif choice == "Clustering Analysis":
        st.header("Clustering Analysis")
        numeric_cols = ['latitude','longitude','estimated_value','building_rentable_square_feet','latest_price_index']
        dfc = df[numeric_cols].fillna(df[numeric_cols].median())
        scaler = MinMaxScaler()
        X = scaler.fit_transform(dfc)
        wcss, dbs = [], []
        k_range = range(2,11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=4742271, n_init=10)
            labels = kmeans.fit_predict(X)
            wcss.append(kmeans.inertia_)
            dbs.append(davies_bouldin_score(X, labels))
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(k_range), y=wcss, mode='lines+markers'))
            fig.update_layout(title="Elbow Curve (WCSS)", xaxis_title="Clusters (k)", yaxis_title="WCSS")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=list(k_range), y=dbs, mode='lines+markers'))
            fig2.update_layout(title="Davies-Bouldin Score", xaxis_title="Clusters (k)", yaxis_title="DB Score (lower is better)")
            st.plotly_chart(fig2, use_container_width=True)
        optimal_k = k_range[np.argmin(dbs)]
        st.success(f"Optimal number of clusters: {optimal_k}")
        kmeans = KMeans(n_clusters=optimal_k, random_state=4742271, n_init=10)
        df['cluster'] = kmeans.fit_predict(X)
        figc = px.scatter(df, x='longitude', y='latitude', color=df['cluster'].astype(str),
                          title="Clustered US Government Assets", color_continuous_scale='viridis')
        st.plotly_chart(figc, use_container_width=True)

    elif choice == "Geographic Map":
        st.header("Asset Locations Map")
        df_geo = df.dropna(subset=['latitude', 'longitude']).copy()
        sample = sample_data(df_geo, 500, 4742271)
        m = folium.Map([sample.latitude.mean(), sample.longitude.mean()], zoom_start=4)
        for _, row in sample.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                fill=True,
                fillColor="#3571b8",
                color="#222",
                radius=7,
                popup=f"{row.get('city','')}, {row.get('state','')}<br>Value: ${row['estimated_value']:,.0f}",
                fill_opacity=0.65
            ).add_to(m)
        st_folium(m, width=950, height=500)

    elif choice == "Asset Value Prediction":
        st.header("Asset Price Prediction Model")
        df_ml = df.dropna(subset=['estimated_value','building_rentable_square_feet','latitude','longitude','latest_price_index'])
        features = ['latitude','longitude','latest_price_index','building_rentable_square_feet']
        X = df_ml[features]
        y = df_ml['estimated_value']
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=4742271)
        model = RandomForestRegressor(n_estimators=120, random_state=4742271)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        st.success(f"Model R² (accuracy): {r2*100:.2f}%")
        with st.form("predict_form"):
            st.markdown("#### Enter details to predict asset value:")
            lat = st.number_input("Latitude", float(X['latitude'].min()), float(X['latitude'].max()), float(X['latitude'].mean()))
            lon = st.number_input("Longitude", float(X['longitude'].min()), float(X['longitude'].max()), float(X['longitude'].mean()))
            lpi = st.number_input("Latest Price Index", float(X['latest_price_index'].min()), float(X['latest_price_index'].max()), float(X['latest_price_index'].mean()))
            sqft = st.number_input("Rentable Square Feet", float(X['building_rentable_square_feet'].min()), float(X['building_rentable_square_feet'].max()), float(X['building_rentable_square_feet'].mean()))
            submit = st.form_submit_button("Predict Value")
            if submit:
                pred = model.predict([[lat, lon, lpi, sqft]])[0]
                st.success(f"Estimated Asset Value: ${pred:,.2f}")

if __name__ == "__main__":
    main()

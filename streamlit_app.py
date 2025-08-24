import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# -------------------------------
# Global Constants
# -------------------------------
RANDOM_STATE = 4742271

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="US Government Assets Portfolio Analytics",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom CSS for Light & Dark Themes
# -------------------------------
st.markdown("""
<style>
:root {
  --primary-color: #1f4e79;
  --light-bg: #f0f2f6;
  --light-box: #e8f4f8;
  --dark-bg: #1e1e1e;
  --dark-box: #2d2d2d;
}
[data-theme="light"] .main-header {
  color: var(--primary-color);
}
[data-theme="dark"] .main-header {
  color: #4fa3ff;
}
.main-header {
  font-size: 2.5rem;
  text-align: center;
  margin-bottom: 2rem;
}
.metric-container {
  padding: 1rem;
  border-radius: 0.5rem;
  border-left: 5px solid var(--primary-color);
}
.insight-box {
  padding: 1rem;
  border-radius: 0.5rem;
  margin: 1rem 0;
}
[data-theme="light"] .metric-container { background-color: var(--light-bg); }
[data-theme="light"] .insight-box { background-color: var(--light-box); }
[data-theme="dark"] .metric-container { background-color: var(--dark-bg); }
[data-theme="dark"] .insight-box { background-color: var(--dark-box); }
.stMetric > label {
  font-size: 1.2rem !important;
  font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Data Loaders
# -------------------------------
@st.cache_data
def load_assets_data():
    try:
        url = "https://drive.google.com/uc?id=1YFTWJNoxu0BF8UlMDXI8bXwRTVQNE2mb"
        df = pd.read_csv(url)
        return df
    except:
        return None

@st.cache_data
def load_housing_data():
    try:
        url = "https://drive.google.com/uc?id=1fFT8Q8GWiIEM7kx6czhQ-qabygUPBQRv"
        df = pd.read_csv(url)
        return df
    except:
        return None

@st.cache_data
def create_sample_data():
    np.random.seed(RANDOM_STATE)
    n_samples = 1000
    states = ['CA','TX','NY','FL','IL','PA','OH','GA','NC','MI']
    data = {
        'state': np.random.choice(states, n_samples),
        'city': np.random.choice(['LA','Houston','NY','Miami','Chicago'], n_samples),
        'latitude': np.random.uniform(25,48,n_samples),
        'longitude': np.random.uniform(-125,-70,n_samples),
        'building_rentable_square_feet': np.random.uniform(1000,100000,n_samples),
        'estimated_value': np.random.lognormal(13,1.5,n_samples),
        'latest_price_index': np.random.uniform(50000,800000,n_samples)
    }
    return pd.DataFrame(data)

# -------------------------------
# Data Merge + Cleaning
# -------------------------------
@st.cache_data
def clean_and_merge_data(df_assets, df_prices):
    if df_assets is None:
        return create_sample_data()
    df_assets.columns = df_assets.columns.str.lower().str.replace(' ','_')
    if df_prices is not None:
        df_prices.columns = df_prices.columns.str.lower()
        df_prices['latest_price_index'] = pd.to_numeric(df_prices.iloc[:,-1], errors='coerce')
        if 'city' in df_assets and 'state' in df_assets and 'city' in df_prices and 'state' in df_prices:
            df_assets['key'] = df_assets['city'].str.lower()+"_"+df_assets['state'].str.lower()
            df_prices['key'] = df_prices['city'].str.lower()+"_"+df_prices['state'].str.lower()
            merged = pd.merge(df_assets, df_prices[['key','latest_price_index']], on='key', how='left')
        else:
            merged = df_assets.copy()
            merged['latest_price_index'] = np.random.uniform(5e4,8e5,len(df_assets))
    else:
        merged = df_assets.copy()
        merged['latest_price_index'] = np.random.uniform(5e4,8e5,len(df_assets))
    merged['estimated_value'] = merged['building_rentable_square_feet'] * (merged['latest_price_index']/100) * 10
    return merged

# -------------------------------
# Clustering
# -------------------------------
@st.cache_data
def perform_clustering(df, n_clusters=5):
    cols = ['latitude','longitude','estimated_value','building_rentable_square_feet','latest_price_index']
    cluster_data = df[cols].fillna(df[cols].median())
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(cluster_data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    df['cluster'] = kmeans.fit_predict(scaled)
    return df, kmeans

# -------------------------------
# ML Features
# -------------------------------
@st.cache_data
def create_ml_features(df):
    features = []
    for col in ['latitude','longitude','latest_price_index','building_rentable_square_feet']:
        if col in df.columns:
            features.append(col)
    return df, features

# -------------------------------
# Folium Map
# -------------------------------
def create_folium_map(df, sample_size=500):
    if len(df)>sample_size:
        df = df.sample(sample_size, random_state=RANDOM_STATE)
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=4)
    for _,row in df.iterrows():
        popup = f"{row.get('city','N/A')}, {row.get('state','N/A')}<br>Value: ${row.get('estimated_value',0):,.0f}"
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color='black',
            fillColor='blue',
            fillOpacity=0.6,
            popup=popup
        ).add_to(m)
    return m

# -------------------------------
# Prediction Tab
# -------------------------------
def show_predict_values(df, model, features):
    st.header("📂 Predict Asset Prices")
    uploaded = st.file_uploader("Upload a CSV file", type=['csv'])
    if uploaded:
        new_df = pd.read_csv(uploaded)
        new_df, _ = create_ml_features(new_df)
        X_new = new_df[features].fillna(new_df[features].median())
        preds = model.predict(X_new)
        new_df['Predicted_Value'] = preds
        st.subheader("Predictions")
        st.dataframe(new_df.head())
        csv = new_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

# -------------------------------
# Main
# -------------------------------
def main():
    st.markdown('<h1 class="main-header">🏛️ US Government Assets Portfolio Analytics Dashboard</h1>', unsafe_allow_html=True)
    df_assets = load_assets_data()
    df_prices = load_housing_data()
    df = clean_and_merge_data(df_assets, df_prices)
    
    # Sidebar nav
    page = st.sidebar.selectbox("Navigation", ["📊 Executive Dashboard","🗺️ Geographic Analysis","🎯 Clustering Analysis","🤖 Machine Learning","📈 Advanced Analytics","📂 Predict Asset Price"])
    
    if page=="📊 Executive Dashboard":
        st.write(df.head())
    elif page=="🗺️ Geographic Analysis":
        map_obj = create_folium_map(df)
        st_folium(map_obj, width=700, height=500)
    elif page=="🎯 Clustering Analysis":
        df,_ = perform_clustering(df, n_clusters=5)
        st.write(df[['city','state','cluster']].head())
    elif page=="🤖 Machine Learning":
        df, features = create_ml_features(df)
        X = df[features].fillna(df[features].median())
        y = df['estimated_value']
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=RANDOM_STATE)
        model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
        model.fit(X_train,y_train)
        st.metric("R² Score", f"{r2_score(y_test, model.predict(X_test)):.3f}")
        st.session_state['trained_model'] = model
        st.session_state['features'] = features
    elif page=="📈 Advanced Analytics":
        st.write(df.describe())
    elif page=="📂 Predict Asset Price":
        if 'trained_model' in st.session_state:
            show_predict_values(df, st.session_state['trained_model'], st.session_state['features'])
        else:
            st.warning("⚠️ Train the model first in the Machine Learning tab before predicting.")

if __name__=="__main__":
    main()

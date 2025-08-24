import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random state for reproducibility
RANDOM_STATE = 4742271
np.random.seed(RANDOM_STATE)

# Set page configuration
st.set_page_config(
    page_title="US Government Assets Portfolio Analytics",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f4e79;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stMetric > label {
        font-size: 1.2rem !important;
        font-weight: bold !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: nowrap;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0px 0px;
        gap: 12px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f4e79;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading and Caching ---
@st.cache_data
def load_data():
    """Loads and merges assets and housing data."""
    try:
        assets_url = "https://drive.google.com/uc?id=1YFTWJNoxu0BF8UlMDXI8bXwRTVQNE2mb"
        df_assets = pd.read_csv(assets_url)
    except Exception:
        st.warning("Could not load assets data from the primary source. Using a sample dataset.")
        return create_sample_data()

    # Clean asset columns
    df_assets.columns = df_assets.columns.str.lower().str.replace(' ', '_')
    if 'latitude' in df_assets.columns and 'longitude' in df_assets.columns:
        df_assets = df_assets[df_assets['latitude'].between(24, 50) & df_assets['longitude'].between(-125, -66)]

    try:
        housing_url = "https://drive.google.com/uc?id=1fFT8Q8GWiIEM7kx6czhQ-qabygUPBQRv"
        df_prices = pd.read_csv(housing_url)
        df_prices.columns = df_prices.columns.str.lower().str.replace(' ', '_')
        price_cols = [col for col in df_prices.columns if '2024' in str(col)]
        if price_cols:
            latest_col = sorted(price_cols, reverse=True)[0]
            df_prices['latest_price_index'] = pd.to_numeric(df_prices[latest_col], errors='coerce')
            
            # Merge logic
            df_assets['city_state_key'] = df_assets['city'].str.lower() + '_' + df_assets['state'].str.lower()
            df_prices['city_state_key'] = df_prices['city'].str.lower() + '_' + df_prices['state'].str.lower()
            df_merged = pd.merge(df_assets, df_prices[['city_state_key', 'latest_price_index']], on='city_state_key', how='left')
        else:
            df_merged = df_assets.copy()
            df_merged['latest_price_index'] = np.random.uniform(50000, 800000, len(df_assets))
    except Exception:
        df_merged = df_assets.copy()
        df_merged['latest_price_index'] = np.random.uniform(50000, 800000, len(df_assets))

    # Data cleaning and feature engineering
    df_merged['latest_price_index'].fillna(df_merged['latest_price_index'].median(), inplace=True)
    rentable_col = next((col for col in df_merged.columns if 'rentable' in col and 'feet' in col), None)
    if rentable_col:
        df_merged['estimated_value'] = df_merged[rentable_col] * (df_merged['latest_price_index'] / 125)
    else:
        df_merged['estimated_value'] = df_merged['latest_price_index'] * np.random.uniform(0.8, 2.5, len(df_merged))

    high_value_states = ['CA', 'NY', 'MA', 'WA', 'VA']
    df_merged.loc[df_merged['state'].isin(high_value_states), 'estimated_value'] *= 1.5
    
    return df_merged.dropna(subset=['latitude', 'longitude'])


@st.cache_data
def create_sample_data():
    """Creates a sample dataframe for demonstration purposes."""
    states = ['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
    data = {
        'state': np.random.choice(states, 1000),
        'city': np.random.choice(cities, 1000),
        'latitude': np.random.uniform(25, 48, 1000),
        'longitude': np.random.uniform(-125, -70, 1000),
        'building_rentable_square_feet': np.random.uniform(5000, 200000, 1000),
        'estimated_value': np.random.lognormal(mean=14, sigma=1, size=1000)
    }
    return pd.DataFrame(data)

# --- Analysis and Visualization Functions ---

@st.cache_data
def perform_clustering(df, n_clusters=5):
    """Performs K-Means clustering on asset data."""
    numeric_cols = ['latitude', 'longitude', 'estimated_value']
    cluster_data = df[numeric_cols].fillna(df[numeric_cols].median())
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    df['cluster'] = kmeans.fit_predict(scaled_data)
    return df

def create_folium_map_with_clusters(df, sample_size=1000):
    """Creates an interactive Folium map with clustered markers."""
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        return None
        
    map_data = df.sample(n=min(len(df), sample_size), random_state=RANDOM_STATE)
    
    center_lat, center_lon = map_data['latitude'].mean(), map_data['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles='cartodbpositron')

    marker_cluster = MarkerCluster().add_to(m)
    colors = px.colors.qualitative.Plotly

    for _, row in map_data.iterrows():
        cluster_id = int(row['cluster']) if 'cluster' in row else 0
        color = colors[cluster_id % len(colors)]
        
        popup_html = f"""
        <b>Location:</b> {row.get('city', 'N/A')}, {row.get('state', 'N/A')}<br>
        <b>Est. Value:</b> ${row.get('estimated_value', 0):,.0f}<br>
        <b>Cluster:</b> {cluster_id}
        """
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=6,
            popup=folium.Popup(popup_html, max_width=200),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7
        ).add_to(marker_cluster)
        
    return m

def show_executive_dashboard(df):
    """Displays the main dashboard with KPIs and charts."""
    st.header("📊 Executive Dashboard")
    if df.empty:
        st.warning("No data to display for the selected filters.")
        return
        
    total_value = df['estimated_value'].sum()
    avg_value = df['estimated_value'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Assets", f"{len(df):,}")
    col2.metric("Portfolio Value", f"${total_value/1e9:.2f}B")
    col3.metric("Avg. Asset Value", f"${avg_value/1e6:.2f}M")
    col4.metric("States Covered", f"{df['state'].nunique()}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Asset Value Distribution")
        fig = px.histogram(df, x='estimated_value', nbins=50, title="Distribution of Asset Values")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 10 States by Portfolio Value")
        state_value = df.groupby('state')['estimated_value'].sum().nlargest(10)
        fig = px.bar(state_value, x=state_value.values, y=state_value.index, orientation='h', title="Assets by State")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

def show_geographic_analysis(df):
    """Displays the geographic analysis and interactive map."""
    st.header("🗺️ Geographic Analysis with Interactive Clustering")
    
    df_clustered = perform_clustering(df.copy(), n_clusters=5)
    
    map_obj = create_folium_map_with_clusters(df_clustered)
    if map_obj:
        st.write("The map below shows the geographic distribution of assets. Markers are clustered for better visibility. Zoom in to see individual assets.")
        st_folium(map_obj, width='100%', height=500)
    else:
        st.error("Could not generate map. Geographic data might be missing.")

def show_clustering_analysis(df):
    """Displays clustering results and visualizations."""
    st.header("🎯 Asset Clustering Analysis")
    
    n_clusters = st.slider("Select Number of Clusters", 2, 10, 5)
    df_clustered = perform_clustering(df.copy(), n_clusters=n_clusters)
    
    col1, col2 = st.columns([2,3])
    with col1:
        st.subheader("Cluster Statistics")
        cluster_stats = df_clustered.groupby('cluster')['estimated_value'].agg(['count', 'mean', 'sum']).rename(
            columns={'count': 'Assets', 'mean': 'Avg. Value', 'sum': 'Total Value'}
        )
        st.dataframe(cluster_stats.style.format({
            "Avg. Value": "${:,.0f}",
            "Total Value": "${:,.0f}"
        }))

    with col2:
        st.subheader("Cluster Distribution")
        fig = px.scatter(
            df_clustered.sample(min(1000, len(df_clustered)), random_state=RANDOM_STATE),
            x='longitude', y='latitude', color='cluster',
            size='estimated_value', hover_name='city',
            title='Geographic Distribution of Asset Clusters'
        )
        st.plotly_chart(fig, use_container_width=True)

def show_ml_predictions(df):
    """Handles the machine learning prediction models."""
    st.header("🤖 Machine Learning for Asset Valuation")
    st.write("This section allows you to train and evaluate models on a random sample of the dataset.")
    
    sample_fraction = st.slider("Select data sample size for ML", 0.1, 1.0, 0.5, 0.1, format="%.1f (%.0f%%)")
    ml_df = df.sample(frac=sample_fraction, random_state=RANDOM_STATE)
    
    st.info(f"Using a random sample of **{len(ml_df):,} assets** ({sample_fraction*100:.0f}%) for machine learning.")
    
    features = ['latitude', 'longitude', 'building_rentable_square_feet']
    features = [f for f in features if f in ml_df.columns]
    
    X = ml_df[features].fillna(0)
    y = ml_df['estimated_value']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    
    # --- Regression Model ---
    with st.expander("🎯 Asset Value Prediction (Regression)", expanded=True):
        model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        col1.metric("R² Score", f"{r2:.3f}")
        col2.metric("Mean Absolute Error", f"${mae/1e6:.2f}M")
        
        results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).sample(1000, random_state=RANDOM_STATE, replace=True)
        fig = px.scatter(results_df, x='Actual', y='Predicted', title='Predicted vs. Actual Asset Values',
                         labels={'Actual': 'Actual Value ($)', 'Predicted': 'Predicted Value ($)'},
                         trendline='ols', trendline_color_override='red')
        st.plotly_chart(fig, use_container_width=True)

    # --- Classification Model ---
    with st.expander("📊 High-Value Asset Detection (Classification)"):
        threshold = y.quantile(0.75)
        y_class = (y > threshold).astype(int)
        
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.3, random_state=RANDOM_STATE, stratify=y_class)

        clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced')
        clf.fit(X_train_c, y_train_c)
        y_pred_c = clf.predict(X_test_c)
        
        accuracy = accuracy_score(y_test_c, y_pred_c)
        f1 = f1_score(y_test_c, y_pred_c)
        
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{accuracy:.3f}")
        col2.metric("F1-Score", f"{f1:.3f}")
        
        st.text(f"Classification Report (1 = High-Value Asset > ${threshold:,.0f}):")
        st.text(classification_report(y_test_c, y_pred_c))

# --- Main App ---
def main():
    """Main function to run the Streamlit app."""
    st.markdown('<h1 class="main-header">🏛️ US Government Assets Portfolio Analytics</h1>', unsafe_allow_html=True)
    
    df = load_data()

    st.sidebar.image("https://i.imgur.com/eY7aG3o.png", use_container_width=True)
    st.sidebar.markdown("### 🔍 Filters")
    
    # State Filter
    states = ['All'] + sorted(df['state'].unique().tolist())
    selected_state = st.sidebar.selectbox("Select State", states)
    df_filtered = df if selected_state == 'All' else df[df['state'] == selected_state]

    # Value Filter
    min_val, max_val = df_filtered['estimated_value'].min(), df_filtered['estimated_value'].max()
    value_range = st.sidebar.slider(
        "Filter by Asset Value ($M)", 
        min_value=float(min_val / 1e6), max_value=float(max_val / 1e6), 
        value=(float(min_val / 1e6), float(max_val / 1e6)),
        step=1.0
    )
    df_filtered = df_filtered[df_filtered['estimated_value'].between(value_range[0] * 1e6, value_range[1] * 1e6)]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Navigation")
    page = st.sidebar.radio(
        "Choose an Analysis Page",
        ["📊 Executive Dashboard", "🗺️ Geographic Analysis", "🎯 Clustering Analysis", "🤖 ML Predictions"]
    )
    
    st.sidebar.info(f"Showing **{len(df_filtered):,}** assets for **{selected_state}** state(s).")
    
    if page == "📊 Executive Dashboard":
        show_executive_dashboard(df_filtered)
    elif page == "🗺️ Geographic Analysis":
        show_geographic_analysis(df_filtered)
    elif page == "🎯 Clustering Analysis":
        show_clustering_analysis(df_filtered)
    elif page == "🤖 ML Predictions":
        show_ml_predictions(df_filtered)

if __name__ == "__main__":
    main()

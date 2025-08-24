import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import geopandas as gpd
from shapely.geometry import Point
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

# --- Global Configuration ---
RANDOM_STATE = 4742271

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
    .st-emotion-cache-12oz5g7 {
        padding-top: 0rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading and Processing Functions ---

@st.cache_data
def load_assets_data():
    """Load US Government Assets dataset"""
    try:
        assets_url = "https://drive.google.com/uc?id=1YFTWJNoxu0BF8UlMDXI8bXwRTVQNE2mb"
        response = requests.get(assets_url, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(requests.get(assets_url, stream=True).raw, encoding='utf-8')
            return df
        else:
            st.error("Failed to download assets data")
            return None
    except Exception as e:
        try:
            df = pd.read_csv(requests.get(assets_url, stream=True).raw, encoding='latin-1')
            return df
        except Exception as e2:
            st.error(f"Error loading assets data: {e2}")
            return None

@st.cache_data
def load_housing_data():
    """Load Zillow Housing Price Index dataset"""
    try:
        housing_url = "https://drive.google.com/uc?id=1fFT8Q8GWiIEM7kx6czhQ-qabygUPBQRv"
        response = requests.get(housing_url, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(requests.get(housing_url, stream=True).raw, encoding='utf-8')
            return df
        else:
            st.error("Failed to download housing data")
            return None
    except Exception as e:
        try:
            df = pd.read_csv(requests.get(housing_url, stream=True).raw, encoding='latin-1')
            return df
        except Exception as e2:
            st.error(f"Error loading housing data: {e2}")
            return None

@st.cache_data
def create_sample_data():
    """Create sample data if real data is not available"""
    np.random.seed(RANDOM_STATE)
    n_samples = 1000
    
    # US state abbreviations
    states = ['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI', 'NJ', 'VA', 'WA', 'AZ', 'MA']
    
    # Generate sample data
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
    """Clean and merge the datasets"""
    if df_assets is None:
        return create_sample_data()
    
    # Clean column names
    df_assets.columns = df_assets.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    
    # Filter valid coordinates if they exist
    if 'latitude' in df_assets.columns and 'longitude' in df_assets.columns:
        valid_coords = (
            (df_assets['latitude'] >= 24) & (df_assets['latitude'] <= 49) &
            (df_assets['longitude'] >= -125) & (df_assets['longitude'] <= -66) &
            df_assets['latitude'].notna() & df_assets['longitude'].notna()
        )
        df_assets = df_assets[valid_coords]
    
    if df_prices is not None:
        df_prices.columns = df_prices.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
        
        # Get latest housing price index
        price_cols = [col for col in df_prices.columns if any(year in str(col) for year in ['2024', '2025'])]
        if price_cols:
            latest_col = sorted(price_cols, reverse=True)[0]
            df_prices['latest_price_index'] = pd.to_numeric(df_prices[latest_col], errors='coerce')
        
        # Create merge keys if possible
        if 'city' in df_assets.columns and 'state' in df_assets.columns and 'city' in df_prices.columns and 'state' in df_prices.columns:
            df_assets['city_state_key'] = (
                df_assets['city'].astype(str).str.lower().str.strip() + '_' + 
                df_assets['state'].astype(str).str.lower().str.strip()
            )
            
            df_prices['city_state_key'] = (
                df_prices['city'].astype(str).str.lower().str.strip() + '_' + 
                df_prices['state'].astype(str).str.lower().str.strip()
            )
            
            # Merge datasets
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
    
    # Fill missing price indices
    merged_df['latest_price_index'] = merged_df['latest_price_index'].fillna(
        merged_df['latest_price_index'].median()
    )
    
    # Calculate estimated values
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
    
    # Add high-value premium for certain states
    high_value_states = ['CA', 'NY', 'MA', 'CT', 'NJ', 'HI', 'MD', 'WA']
    if 'state' in merged_df.columns:
        premium_mask = merged_df['state'].isin(high_value_states)
        merged_df.loc[premium_mask, 'estimated_value'] *= 1.5
    
    return merged_df

@st.cache_data
def perform_clustering(df, n_clusters=5):
    """Perform clustering analysis using K-Means only"""
    # Select numeric columns for clustering
    numeric_cols = []
    for col in ['latitude', 'longitude', 'estimated_value', 'building_rentable_square_feet', 'latest_price_index']:
        if col in df.columns:
            numeric_cols.append(col)
    
    if len(numeric_cols) == 0:
        return df, None, None, []
    
    # Prepare data for clustering
    cluster_data = df[numeric_cols].fillna(df[numeric_cols].median())
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    df['cluster'] = kmeans.fit_predict(scaled_data)
    
    return df, kmeans, scaler, numeric_cols

def name_clusters(cluster_stats, col_means):
    """
    Assigns descriptive names to clusters based on their average feature values.
    """
    names = {}
    
    # Sort clusters by their average estimated value
    sorted_clusters = cluster_stats.sort_values(
        ('estimated_value', 'mean'), ascending=False
    ).index
    
    for i, cluster_num in enumerate(sorted_clusters):
        avg_value = cluster_stats.loc[cluster_num, ('estimated_value', 'mean')]
        
        if i == 0:
            names[cluster_num] = f"Cluster {cluster_num}: High-Value Assets"
        elif i == len(sorted_clusters) - 1:
            names[cluster_num] = f"Cluster {cluster_num}: Low-Value Assets"
        else:
            names[cluster_num] = f"Cluster {cluster_num}: Mid-Value Assets"
    
    return names

@st.cache_data
def create_ml_features(df):
    """Create features for machine learning"""
    features = []
    
    # Geographic features
    if 'latitude' in df.columns and 'longitude' in df.columns:
        major_cities = {
            'NYC': (40.7128, -74.0060),
            'LA': (34.0522, -118.2437),
            'Chicago': (41.8781, -87.6298),
            'Houston': (29.7604, -95.3698),
            'DC': (38.9072, -77.0369)
        }
        
        for city, (lat, lon) in major_cities.items():
            df[f'distance_to_{city.lower()}'] = np.sqrt(
                (df['latitude'] - lat)**2 + (df['longitude'] - lon)**2
            )
            features.append(f'distance_to_{city.lower()}')
    
    # Add other numeric features
    for col in ['latitude', 'longitude', 'latest_price_index', 'building_rentable_square_feet']:
        if col in df.columns:
            features.append(col)
    
    return df, features

def create_folium_map(df, sample_size=500):
    """Create interactive Folium map"""
    # Check if we have the required columns
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        return None
    
    # Sample data for performance
    if len(df) > sample_size:
        map_data = df.sample(n=sample_size, random_state=RANDOM_STATE)
    else:
        map_data = df.copy()
    
    # Remove rows with missing coordinates
    map_data = map_data.dropna(subset=['latitude', 'longitude'])
    
    if len(map_data) == 0:
        return None
    
    # Calculate center
    center_lat = map_data['latitude'].mean()
    center_lon = map_data['longitude'].mean()
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=4)
    
    # Add markers
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']
    
    for idx, (_, row) in enumerate(map_data.iterrows()):
        if 'cluster' in row and not pd.isna(row['cluster']):
            color = colors[int(row['cluster']) % len(colors)]
            popup_text = f"""
            <b>Government Asset</b><br>
            Location: {row.get('city', 'N/A')}, {row.get('state', 'N/A')}<br>
            Estimated Value: ${row.get('estimated_value', 0):,.0f}<br>
            Cluster: {row.get('cluster_name', f'Cluster {row["cluster"]}')}
            """
        else:
            color = 'blue'
            popup_text = f"""
            <b>Government Asset</b><br>
            Location: {row.get('city', 'N/A')}, {row.get('state', 'N/A')}<br>
            Estimated Value: ${row.get('estimated_value', 0):,.0f}
            """
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=6,
            popup=folium.Popup(popup_text, max_width=300),
            color='black',
            fillColor=color,
            fillOpacity=0.7,
            weight=1
        ).add_to(m)
    
    return m

# --- Main Dashboard Functions ---

def main():
    # Header
    st.markdown('<h1 class="main-header">🏛️ US Government Assets Portfolio Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.image("https://via.placeholder.com/300x100/1f4e79/ffffff?text=Analytics+Dashboard", 
                      use_container_width=True)
    st.sidebar.markdown("### 📊 Navigation")
    
    # Load data
    with st.spinner("Loading datasets..."):
        df_assets = load_assets_data()
        df_prices = load_housing_data()
    
    # Show data loading status
    if df_assets is None and df_prices is None:
        st.warning("Could not load external data. Using sample data for demonstration.")
        df_merged = create_sample_data()
    else:
        # Merge and clean data
        with st.spinner("Processing and merging data..."):
            df_merged = clean_and_merge_data(df_assets, df_prices)
    
    if df_merged is None or len(df_merged) == 0:
        st.error("No data available for analysis.")
        return
    
    # Sidebar filters
    st.sidebar.markdown("### 🔍 Filters")
    
    # State filter
    if 'state' in df_merged.columns:
        states = ['All'] + sorted(df_merged['state'].unique().tolist())
        selected_state = st.sidebar.selectbox("Select State", states)
        
        if selected_state != 'All':
            df_filtered = df_merged[df_merged['state'] == selected_state]
        else:
            df_filtered = df_merged.copy()
    else:
        df_filtered = df_merged.copy()
        selected_state = 'All'
    
    # Value range filter
    if 'estimated_value' in df_filtered.columns and len(df_filtered) > 0:
        min_value = int(df_filtered['estimated_value'].min())
        max_value = int(df_filtered['estimated_value'].max())
        
        if min_value < max_value:
            value_range = st.sidebar.slider(
                "Asset Value Range ($)",
                min_value=min_value,
                max_value=max_value,
                value=(min_value, max_value),
                format="$%d"
            )
            df_filtered = df_filtered[
                (df_filtered['estimated_value'] >= value_range[0]) & 
                (df_filtered['estimated_value'] <= value_range[1])
            ]
    
    # Navigation
    page = st.sidebar.selectbox(
        "Choose Analysis",
        ["📊 Executive Dashboard", "🗺️ Geographic Analysis", "🎯 Clustering Analysis", 
         "🤖 Machine Learning", "📈 Advanced Analytics"]
    )
    
    # Show filtered data info
    st.sidebar.markdown("### 📋 Data Summary")
    st.sidebar.metric("Total Assets", f"{len(df_filtered):,}")
    if 'state' in df_filtered.columns:
        st.sidebar.metric("States", f"{df_filtered['state'].nunique()}")
    
    # Route to different pages
    if page == "📊 Executive Dashboard":
        show_executive_dashboard(df_filtered)
    elif page == "🗺️ Geographic Analysis":
        show_geographic_analysis(df_filtered)
    elif page == "🎯 Clustering Analysis":
        show_clustering_analysis(df_filtered)
    elif page == "🤖 Machine Learning":
        show_machine_learning(df_filtered)
    elif page == "📈 Advanced Analytics":
        show_advanced_analytics(df_filtered)

def show_executive_dashboard(df):
    """Show executive dashboard"""
    st.header("📊 Executive Dashboard")
    
    if len(df) == 0:
        st.warning("No data available with current filters.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Assets",
            f"{len(df):,}"
        )
    
    with col2:
        if 'estimated_value' in df.columns:
            total_value = df['estimated_value'].sum()
            st.metric(
                "Portfolio Value",
                f"${total_value/1e9:.1f}B"
            )
        else:
            st.metric("Portfolio Value", "N/A")
    
    with col3:
        if 'estimated_value' in df.columns:
            avg_value = df['estimated_value'].mean()
            st.metric(
                "Average Asset Value",
                f"${avg_value/1e6:.1f}M"
            )
        else:
            st.metric("Average Asset Value", "N/A")
    
    with col4:
        if 'state' in df.columns:
            states_count = df['state'].nunique()
            st.metric(
                "States Covered",
                f"{states_count}"
            )
        else:
            st.metric("States Covered", "N/A")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        if 'estimated_value' in df.columns:
            st.subheader("Asset Value Distribution")
            fig = px.histogram(
                df, 
                x='estimated_value', 
                nbins=30,
                title="Distribution of Asset Values"
            )
            fig.update_layout(
                xaxis_title="Estimated Value ($)",
                yaxis_title="Number of Assets"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Asset value data not available")
    
    with col2:
        if 'state' in df.columns:
            st.subheader("Top 10 States by Asset Count")
            state_counts = df['state'].value_counts().head(10)
            fig = px.bar(
                x=state_counts.index,
                y=state_counts.values,
                title="Assets by State"
            )
            fig.update_layout(
                xaxis_title="State",
                yaxis_title="Number of Assets"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("State data not available")
    
    # Portfolio insights
    st.subheader("💡 Key Insights")
    
    insights = []
    
    if 'state' in df.columns and len(df) > 0:
        top_state = df['state'].value_counts().index[0]
        top_state_count = df['state'].value_counts().iloc[0]
        insights.append(f"📍 **{top_state}** has the highest number of assets ({top_state_count:,})")
    
    if 'estimated_value' in df.columns:
        total_value = df['estimated_value'].sum()
        avg_value = df['estimated_value'].mean()
        insights.extend([
            f"💰 Total portfolio value: **${total_value/1e9:.1f} billion**",
            f"📊 Average asset value: **${avg_value/1e6:.1f} million**"
        ])
        
        # Value concentration
        if len(df) > 10:
            top_10_pct_value = df.nlargest(int(len(df) * 0.1), 'estimated_value')['estimated_value'].sum()
            concentration_pct = (top_10_pct_value / total_value * 100)
            insights.append(f"🏛️ Value concentration: Top 10% of assets represent **{concentration_pct:.1f}%** of total value")
    
    for insight in insights:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

def show_geographic_analysis(df):
    """Show geographic analysis"""
    st.header("🗺️ Geographic Analysis")
    
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        st.error("Geographic coordinates not available in the dataset.")
        return
    
    # Filter out rows with missing coordinates
    df_geo = df.dropna(subset=['latitude', 'longitude'])
    
    if len(df_geo) == 0:
        st.warning("No valid geographic coordinates found.")
        return
    
    # Interactive map
    st.subheader("Interactive Asset Map (Clustering based)")
    
    # Perform clustering for map colors
    df_clustered, _, _, _ = perform_clustering(df_geo, n_clusters=5)
    
    # Create and display map
    map_obj = create_folium_map(df_clustered)
    
    if map_obj is not None:
        map_data = st_folium(map_obj, width=700, height=500)
    else:
        st.error("Could not create map")

    st.markdown("---")
    
    # Geoheatmap
    st.subheader("Asset Value Concentration (Heatmap)")
    if 'estimated_value' in df_geo.columns:
        heat_map_data = df_geo[['latitude', 'longitude', 'estimated_value']].values.tolist()
        
        center_lat = df_geo['latitude'].mean()
        center_lon = df_geo['longitude'].mean()
        
        heat_map = folium.Map(location=[center_lat, center_lon], zoom_start=4)
        HeatMap(heat_map_data, radius=15).add_to(heat_map)
        st_folium(heat_map, width=700, height=500)
    else:
        st.info("Estimated value data not available for heatmap visualization.")

    st.markdown("---")

    # Geographic statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Geographic Distribution")
        if 'state' in df_geo.columns and 'estimated_value' in df_geo.columns:
            state_stats = df_geo.groupby('state').agg({
                'estimated_value': ['count', 'sum', 'mean'],
                'latitude': 'mean',
                'longitude': 'mean'
            }).round(2)
            
            state_stats.columns = ['Asset Count', 'Total Value', 'Avg Value', 'Center Lat', 'Center Lon']
            state_stats = state_stats.sort_values('Total Value', ascending=False)
            
            st.dataframe(state_stats.head(10))
        else:
            st.info("State or value data not available for analysis")
    
    with col2:
        st.subheader("Value Distribution by Region")
        if 'state' in df_geo.columns and 'estimated_value' in df_geo.columns:
            state_values = df_geo.groupby('state')['estimated_value'].sum().sort_values(ascending=False).head(15)
            
            if len(state_values) > 0:
                fig = px.bar(
                    x=state_values.values,
                    y=state_values.index,
                    orientation='h',
                    title="Total Asset Value by State",
                    color=state_values.values,
                    color_continuous_scale="Viridis"
                )
                fig.update_layout(
                    xaxis_title="Total Value ($)",
                    yaxis_title="State"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No state value data available")
        else:
            st.info("State or value data not available")

def show_clustering_analysis(df):
    """Show clustering analysis using K-Means only"""
    st.header("🎯 Clustering Analysis")
    
    # Check if we have necessary data
    required_cols = ['latitude', 'longitude', 'estimated_value', 'building_rentable_square_feet', 'latest_price_index']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Missing required columns for clustering: **{', '.join(missing_cols)}**")
        return
    
    # Remove rows with missing coordinates
    df_clean = df.dropna(subset=['latitude', 'longitude'])
    
    if len(df_clean) == 0:
        st.warning("No valid data for clustering analysis.")
        return
    
    # Clustering parameters
    col1, col2 = st.columns([1, 3])
    
    with col1:
        n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=5)
        
        # Perform K-means clustering only
        df_clustered, model, scaler, numeric_cols = perform_clustering(df_clean, n_clusters=n_clusters)
    
    if 'cluster' in df_clustered.columns:
        valid_clusters = df_clustered[df_clustered['cluster'] >= 0]
        
        if len(valid_clusters) > 0:
            agg_dict = {
                'estimated_value': ['count', 'sum', 'mean', 'std'],
                'latitude': 'mean',
                'longitude': 'mean',
                'building_rentable_square_feet': 'mean',
                'latest_price_index': 'mean'
            }
            
            cluster_stats = valid_clusters.groupby('cluster').agg(agg_dict).round(2)
            cluster_stats.columns = cluster_stats.columns.map('_'.join)
            
            cluster_names = name_clusters(cluster_stats, cluster_stats.mean())
            df_clustered['cluster_name'] = df_clustered['cluster'].map(cluster_names)

            st.markdown(
                f"""
                <div class="insight-box">
                    Clustering was performed using **K-Means** on the following features: 
                    **{', '.join(numeric_cols)}**. The algorithm groups assets based on the 
                    similarity of these features, revealing natural segments in the portfolio.
                </div>
                """, unsafe_allow_html=True
            )
            
            with col2:
                # Cluster visualization
                sample_size = min(1000, len(df_clustered))
                sample_data = df_clustered.sample(n=sample_size, random_state=RANDOM_STATE)
                
                fig = px.scatter_mapbox(
                    sample_data,
                    lat="latitude",
                    lon="longitude",
                    color="cluster_name",
                    size="estimated_value",
                    hover_data={'estimated_value': ':$,.0f', 'cluster_name': True},
                    mapbox_style="open-street-map",
                    zoom=3,
                    height=600,
                    title="Asset Clusters Geographic Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Cluster analysis
            st.subheader("📊 Cluster Analysis")
            
            if 'estimated_value_mean' in cluster_stats.columns:
                cluster_stats = cluster_stats.rename(columns={
                    'estimated_value_count': 'Count',
                    'estimated_value_sum': 'Total Value ($)',
                    'estimated_value_mean': 'Avg Value ($)',
                    'estimated_value_std': 'Value Std ($)',
                    'latitude_mean': 'Center Lat',
                    'longitude_mean': 'Center Lon',
                    'building_rentable_square_feet_mean': 'Avg Sqft',
                    'latest_price_index_mean': 'Avg Price Index'
                })
                cluster_stats['Total Value ($)'] = cluster_stats['Total Value ($)'].apply(lambda x: f'${x:,.0f}')
                cluster_stats['Avg Value ($)'] = cluster_stats['Avg Value ($)'].apply(lambda x: f'${x:,.0f}')
                cluster_stats['Value Std ($)'] = cluster_stats['Value Std ($)'].apply(lambda x: f'${x:,.0f}')
                cluster_stats['Avg Sqft'] = cluster_stats['Avg Sqft'].apply(lambda x: f'{x:,.0f}')
                cluster_stats['Avg Price Index'] = cluster_stats['Avg Price Index'].apply(lambda x: f'${x:,.0f}')
            
            cluster_stats['Cluster Name'] = [cluster_names.get(i, f'Cluster {i}') for i in cluster_stats.index]
            cluster_stats.set_index('Cluster Name', inplace=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Cluster Statistics and Naming**")
                st.dataframe(cluster_stats)
            
            with col2:
                # Cluster size pie chart
                cluster_counts = valid_clusters['cluster_name'].value_counts()
                fig = px.pie(
                    values=cluster_counts.values,
                    names=cluster_counts.index,
                    title="Assets Distribution by Cluster"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No valid clusters found with current parameters.")

def show_machine_learning(df):
    """Show machine learning analysis"""
    st.header("🤖 Machine Learning Analysis")
    
    if 'estimated_value' not in df.columns:
        st.error("Estimated value column not found. Cannot perform ML analysis.")
        return
    
    # Remove rows with missing target values
    df_ml = df.dropna(subset=['estimated_value'])
    
    if len(df_ml) == 0:
        st.warning("No valid data for machine learning analysis.")
        return
    
    # Prepare features
    df_ml, features = create_ml_features(df_ml)
    
    # Ensure all features exist in the dataframe
    available_features = [f for f in features if f in df_ml.columns]
    
    if len(available_features) == 0:
        st.error("No suitable features found for ML analysis.")
        return
    
    # Prepare data
    X = df_ml[available_features].fillna(df_ml[available_features].median())
    y = df_ml['estimated_value']
    
    # Check if we have enough data
    if len(X) < 10:
        st.warning("Not enough data for meaningful ML analysis.")
        return

    st.markdown("---")
    
    # ML tasks
    task = st.selectbox("Select ML Task", 
                        ["Value Prediction (Regression)", "Value Classification", "High-Value Detection"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            if task == "Value Prediction (Regression)":
                st.subheader("🎯 Asset Value Prediction")
                
                # Train-test split
                if len(X) > 4:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
                    
                    # Train model
                    model = RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE)
                    model.fit(X_train, y_train)
                    
                    # Predictions
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    
                    st.metric("R² Score (Model Accuracy)", f"{r2*100:.2f}%")
                    
                    # Feature importance
                    feature_importance = pd.DataFrame({
                        'feature': available_features,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False).head(10)
                    
                    fig = px.bar(
                        feature_importance,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Top Feature Importances"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough data for train-test split.")
            
            elif task == "Value Classification":
                st.subheader("📊 Asset Value Classification")
                
                # Create value categories
                try:
                    y_class = pd.qcut(df_ml['estimated_value'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
                    
                    if len(y_class.unique()) < 2:
                        st.warning("Cannot create meaningful value categories with current data.")
                        return
                    
                    # Train-test split
                    if len(X) > 4:
                        X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=RANDOM_STATE, stratify=y_class)
                        
                        # Train model
                        model = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)
                        model.fit(X_train, y_train)
                        
                        # Predictions
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        st.metric("Accuracy", f"{accuracy*100:.2f}%")
                        
                        # Class distribution
                        class_dist = y_class.value_counts()
                        fig = px.pie(
                            values=class_dist.values,
                            names=class_dist.index,
                            title="Asset Value Categories Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Not enough data for train-test split.")
                        
                except ValueError as e:
                    st.error(f"Error creating value categories: {e}")
            
            else:  # High-Value Detection
                st.subheader("🎯 High-Value Asset Detection")
                
                # Binary target (top 25% as high-value)
                threshold = df_ml['estimated_value'].quantile(0.75)
                y_binary = (df_ml['estimated_value'] > threshold).astype(int)
                
                if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
                    st.warning("Cannot create balanced binary classification with current threshold.")
                    return
                
                # Train-test split
                if len(X) > 4:
                    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=RANDOM_STATE, stratify=y_binary)
                    
                    # Train model
                    model = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)
                    model.fit(X_train, y_train)
                    
                    # Predictions
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    st.metric("Accuracy", f"{accuracy*100:.2f}%")
                    st.metric("High-Value Threshold", f"${threshold/1e6:.1f}M")
                    
                    # Distribution
                    dist_data = pd.DataFrame({
                        'Category': ['Regular Value', 'High Value'],
                        'Count': [(y_binary == 0).sum(), (y_binary == 1).sum()]
                    })
                    
                    fig = px.bar(
                        dist_data,
                        x='Category',
                        y='Count',
                        title="Asset Value Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough data for train-test split.")
                    
        except Exception as e:
            st.error(f"Error in ML analysis: {e}")
    
    with col2:
        st.subheader("🔧 Model Details")
        st.write("**Features Used:**")
        for feature in available_features[:10]:
            st.write(f"• {feature.replace('_', ' ').title()}")
        
        if len(available_features) > 10:
            st.write(f"... and {len(available_features) - 10} more features")
        
        st.write(f"**Dataset Size:** {len(df_ml):,} assets")
        st.write(f"**Features Count:** {len(available_features)}")
        
        # Show sample data
        st.write("**Sample Data:**")
        sample_cols = ['estimated_value'] + available_features[:3]
        available_sample_cols = [col for col in sample_cols if col in df_ml.columns]
        st.dataframe(df_ml[available_sample_cols].head())

    st.markdown("---")

    # Asset Price Prediction section
    st.subheader("📈 Predict Asset Value")
    
    st.markdown(
        """
        <div class="insight-box">
        Use the trained regression model to predict the estimated value of a new asset.
        </div>
        """, unsafe_allow_html=True
    )

    with st.expander("Predict a Single Asset Value"):
        
        input_data = {}
        input_data['latitude'] = st.number_input('Latitude', min_value=24.0, max_value=49.0, value=34.05, format="%.2f")
        input_data['longitude'] = st.number_input('Longitude', min_value=-125.0, max_value=-66.0, value=-118.24, format="%.2f")
        
        if 'building_rentable_square_feet' in df_ml.columns:
            input_data['building_rentable_square_feet'] = st.number_input('Building Rentable Square Feet', value=20000.0, format="%.2f")
        
        if 'latest_price_index' in df_ml.columns:
            input_data['latest_price_index'] = st.number_input('Latest Price Index', value=300000.0, format="%.2f")
            
        # Add distances to major cities
        major_cities = {
            'NYC': (40.7128, -74.0060), 'LA': (34.0522, -118.2437),
            'Chicago': (41.8781, -87.6298), 'Houston': (29.7604, -95.3698),
            'DC': (38.9072, -77.0369)
        }
        for city, (lat, lon) in major_cities.items():
            input_data[f'distance_to_{city.lower()}'] = np.sqrt(
                (input_data['latitude'] - lat)**2 + (input_data['longitude'] - lon)**2
            )

        if st.button("Predict Value"):
            # Ensure the feature order matches the trained model
            input_df = pd.DataFrame([input_data])[available_features]
            
            try:
                # Re-train the model to ensure it exists
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
                model = RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE)
                model.fit(X_train, y_train)
                
                predicted_value = model.predict(input_df)[0]
                st.success(f"**Predicted Asset Value:** ${predicted_value:,.2f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

def show_advanced_analytics(df):
    """Show advanced analytics"""
    st.header("📈 Advanced Analytics")
    
    tabs = st.tabs(["📊 Statistical Analysis", "🔍 Data Quality", "📈 Trends"])
    
    with tabs[0]:
        st.subheader("Statistical Summary")
        
        if 'estimated_value' in df.columns and len(df) > 0:
            # Statistical summary
            stats_df = df['estimated_value'].describe()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean Value", f"${stats_df['mean']/1e6:.1f}M")
                st.metric("Median Value", f"${stats_df['50%']/1e6:.1f}M")
            
            with col2:
                st.metric("Standard Deviation", f"${stats_df['std']/1e6:.1f}M")
                st.metric("Min Value", f"${stats_df['min']/1e6:.1f}M")
            
            with col3:
                st.metric("Max Value", f"${stats_df['max']/1e6:.1f}M")
                st.metric("75th Percentile", f"${stats_df['75%']/1e6:.1f}M")
            
            # Box plot
            fig = px.box(df, y='estimated_value', title="Asset Value Distribution")
            fig.update_layout(yaxis_title="Estimated Value ($)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Estimated value data not available for statistical analysis.")
    
    with tabs[1]:
        st.subheader("Data Quality Assessment")
        
        # Missing values analysis
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df) * 100).round(2)
        
        quality_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': missing_pct.values
        })
        quality_df = quality_df[quality_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if len(quality_df) > 0:
            fig = px.bar(
                quality_df.head(10),
                x='Missing %',
                y='Column',
                orientation='h',
                title="Missing Data by Column (Top 10)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("**Data Quality Summary:**")
            st.dataframe(quality_df)
        else:
            st.success("✅ No missing data found in the dataset!")
        
        # Data completeness summary
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells * 100)
        
        st.metric("Overall Data Completeness", f"{completeness:.1f}%")
    
    with tabs[2]:
        st.subheader("Portfolio Trends")
        
        if 'state' in df.columns and 'estimated_value' in df.columns and len(df) > 0:
            # State-wise analysis
            agg_dict = {'estimated_value': ['count', 'sum', 'mean']}
            
            # Add square feet if available
            if 'building_rentable_square_feet' in df.columns:
                agg_dict['building_rentable_square_feet'] = 'mean'
            
            state_analysis = df.groupby('state').agg(agg_dict).round(2)
            
            if 'building_rentable_square_feet' in df.columns:
                state_analysis.columns = ['Asset Count', 'Total Value', 'Avg Value', 'Avg Sqft']
            else:
                state_analysis.columns = ['Asset Count', 'Total Value', 'Avg Value']
            
            state_analysis = state_analysis.sort_values('Total Value', ascending=False).head(20)
            
            if len(state_analysis) > 0:
                # Bubble chart
                size_col = 'Total Value'
                color_col = 'Avg Sqft' if 'Avg Sqft' in state_analysis.columns else 'Asset Count'
                
                fig = px.scatter(
                    state_analysis.reset_index(),
                    x='Asset Count',
                    y='Avg Value',
                    size='Total Value',
                    color=color_col,
                    hover_name='state',
                    title="State Portfolio Analysis (Bubble Chart)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Top states table
                st.write("**Top 20 States by Portfolio Value:**")
                st.dataframe(state_analysis)
            else:
                st.info("No state data available for trend analysis.")
        else:
            st.info("Required data (state, estimated_value) not available for trend analysis.")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import geopandas as gpd
from shapely.geometry import Point
import pickle
import gdown
import os
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import contextily as ctx
from scipy import stats
import requests
from io import BytesIO

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Government Assets Valuation Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 2rem;
        color: #2c5aa0;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2c5aa0;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2c5aa0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def download_data():
    """Download datasets using gdown."""
    
    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # File URLs and paths
    files_to_download = {
        "zillow_data": {
            "url": "https://drive.google.com/uc?id=1fFT8Q8GWiIEM7kx6czhQ-qabygUPBQRv",
            "path": "data/zillow_housing_index.csv"
        },
        "assets_data": {
            "url": "https://drive.google.com/uc?id=1YFTWJNoxu0BF8UlMDXI8bXwRTVQNE2mb",
            "path": "data/us_government_assets.csv"
        }
    }
    
    # Download files if they don't exist
    for file_key, file_info in files_to_download.items():
        if not os.path.exists(file_info["path"]):
            try:
                with st.spinner(f"Downloading {file_info['path']}..."):
                    gdown.download(file_info["url"], file_info["path"], quiet=False)
                st.success(f"Successfully downloaded {file_info['path']}")
            except Exception as e:
                st.error(f"Failed to download {file_info['path']}: {str(e)}")
                # Create dummy data if download fails
                if "zillow" in file_key:
                    create_dummy_zillow_data(file_info["path"])
                else:
                    create_dummy_assets_data(file_info["path"])
    
    return files_to_download

def create_dummy_zillow_data(path):
    """Create dummy Zillow data for testing."""
    np.random.seed(42)
    n_regions = 1000
    n_months = 120
    
    data = []
    for i in range(n_regions):
        row = {
            'RegionID': i,
            'RegionName': f"Region_{i}",
            'City': f"City_{i % 100}",
            'State': np.random.choice(['CA', 'TX', 'FL', 'NY', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI'], 1)[0],
            'CountyName': f"County_{i % 50}"
        }
        
        # Generate time series data
        base_price = np.random.lognormal(12, 0.5)
        trend = np.random.normal(0, 0.01, n_months).cumsum()
        prices = base_price * (1 + trend + np.random.normal(0, 0.05, n_months))
        
        for j in range(n_months):
            row[f"2015-{j+1:02d}"] = max(prices[j], 10000)  # Ensure positive prices
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    st.info(f"Created dummy Zillow data at {path}")

def create_dummy_assets_data(path):
    """Create dummy assets data for testing."""
    np.random.seed(42)
    n_assets = 2000
    
    states = ['CA', 'TX', 'FL', 'NY', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
    installations = ['Fort Base', 'Naval Station', 'Air Force Base', 'Marine Corps', 'Army Base']
    
    data = []
    for i in range(n_assets):
        state = np.random.choice(states)
        data.append({
            'Location Code': f"LOC_{i:04d}",
            'Real Property Asset Name': f"Building_{i}",
            'City': f"City_{np.random.randint(0, 100)}",
            'State': state,
            'Installation Name': f"{np.random.choice(installations)}_{np.random.randint(1, 10)}",
            'Street Address': f"{np.random.randint(1, 9999)} Main St",
            'Latitude': np.random.uniform(25, 49),
            'Longitude': np.random.uniform(-125, -65),
            'Building Rentable Square Feet': np.random.randint(1000, 100000),
            'Zip Code': f"{np.random.randint(10000, 99999)}"
        })
    
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    st.info(f"Created dummy assets data at {path}")

@st.cache_data
def load_and_process_data():
    """Load and process the datasets."""
    
    # Download data first
    download_data()
    
    try:
        # Load datasets
        df_zillow = pd.read_csv("data/zillow_housing_index.csv")
        df_assets = pd.read_csv("data/us_government_assets.csv")
        
        # Process assets data
        for col in ['City', 'State', 'Installation Name', 'Real Property Asset Name', 'Street Address']:
            if col in df_assets.columns:
                df_assets[col] = df_assets[col].astype(str).fillna('').str.upper().str.strip()
        
        if 'Latitude' in df_assets.columns:
            df_assets['Latitude'] = pd.to_numeric(df_assets['Latitude'], errors='coerce')
        if 'Longitude' in df_assets.columns:
            df_assets['Longitude'] = pd.to_numeric(df_assets['Longitude'], errors='coerce')
        
        # Sample Zillow data for performance
        if len(df_zillow) > 2000:
            df_zillow = df_zillow.sample(n=2000, random_state=42).reset_index(drop=True)
        
        # Process Zillow time series data
        date_cols = [c for c in df_zillow.columns if len(c) > 4 and c[0].isdigit()]
        if not date_cols:
            # If no date columns found, create dummy ones
            date_cols = [f"2015-{i:02d}" for i in range(1, 121)]
        
        # Convert to numeric and handle missing values
        for col in date_cols:
            if col in df_zillow.columns:
                df_zillow[col] = pd.to_numeric(df_zillow[col], errors='coerce')
        
        # Engineer features
        features = []
        for idx, row in df_zillow.iterrows():
            try:
                available_dates = [col for col in date_cols if col in row.index and pd.notna(row[col])]
                if available_dates:
                    prices = [row[col] for col in available_dates]
                    prices = [p for p in prices if pd.notna(p) and p > 0]
                    
                    if len(prices) > 0:
                        features.append({
                            'RegionID': row.get('RegionID', idx),
                            'RegionName': row.get('RegionName', f'Region_{idx}'),
                            'City': str(row.get('City', '')).upper().strip(),
                            'State': str(row.get('State', '')).upper().strip(),
                            'County': str(row.get('CountyName', '')).upper().strip(),
                            'mean_price': np.mean(prices),
                            'median_price': np.median(prices),
                            'std_price': np.std(prices),
                            'price_min': np.min(prices),
                            'price_max': np.max(prices),
                            'price_range': np.max(prices) - np.min(prices),
                            'price_volatility': np.std(prices) / np.mean(prices) if np.mean(prices) != 0 else 0,
                            'recent_6mo_avg': np.mean(prices[-6:]) if len(prices) >= 6 else np.mean(prices),
                            'recent_12mo_avg': np.mean(prices[-12:]) if len(prices) >= 12 else np.mean(prices),
                            'last_price': prices[-1],
                            'price_trend_slope': np.polyfit(range(len(prices)), prices, 1)[0] if len(prices) > 1 else 0
                        })
            except Exception as e:
                st.warning(f"Error processing row {idx}: {str(e)}")
                continue
        
        df_zillow_features = pd.DataFrame(features)
        
        if df_zillow_features.empty:
            st.error("No valid Zillow features could be created")
            return None, None, None, []
        
        return df_zillow, df_assets, df_zillow_features, date_cols
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, []

def perform_clustering_analysis(df_features):
    """Perform clustering analysis on Zillow features."""
    
    if df_features is None or df_features.empty:
        return None, 0, [], [], [], None, None, None
    
    num_cols = ['mean_price', 'median_price', 'std_price', 'price_min', 'price_max', 
                'price_range', 'price_volatility', 'recent_6mo_avg', 'recent_12mo_avg', 
                'last_price', 'price_trend_slope']
    
    # Ensure all numeric columns exist and have valid data
    for col in num_cols:
        if col not in df_features.columns:
            df_features[col] = 0
        df_features[col] = pd.to_numeric(df_features[col], errors='coerce').fillna(0)
    
    # Scale features
    scaler = MinMaxScaler()
    df_scaled = df_features.copy()
    df_scaled[num_cols] = scaler.fit_transform(df_features[num_cols])
    
    # Determine optimal number of clusters
    K_range = range(2, min(8, len(df_scaled) // 10))  # Ensure reasonable cluster sizes
    if len(K_range) == 0:
        K_range = range(2, 4)
    
    inertias = []
    silhouette_scores = []
    
    for k in K_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(df_scaled[num_cols])
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(df_scaled[num_cols], cluster_labels))
        except Exception as e:
            st.warning(f"Error in clustering with k={k}: {str(e)}")
            continue
    
    if not silhouette_scores:
        optimal_k = 2
    else:
        optimal_k = K_range[np.argmax(silhouette_scores)]
    
    # Final clustering
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df_scaled['cluster'] = final_kmeans.fit_predict(df_scaled[num_cols])
    
    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(df_scaled[num_cols])
    df_scaled['pca1'] = pca_coords[:, 0]
    df_scaled['pca2'] = pca_coords[:, 1]
    
    return df_scaled, optimal_k, inertias, silhouette_scores, K_range, scaler, final_kmeans, pca

def create_spatial_analysis(df_assets):
    """Perform spatial analysis on assets data."""
    
    if df_assets is None or df_assets.empty:
        return None, None
    
    # Filter for valid coordinates
    valid_coords = df_assets.dropna(subset=['Latitude', 'Longitude'])
    
    if len(valid_coords) == 0:
        return None, None
    
    # Create GeoDataFrame
    try:
        gdf = gpd.GeoDataFrame(
            valid_coords,
            geometry=gpd.points_from_xy(valid_coords['Longitude'], valid_coords['Latitude']),
            crs="EPSG:4326"
        )
        
        # Simulate some values for spatial analysis
        np.random.seed(42)
        gdf['pred_value'] = np.random.lognormal(mean=12, sigma=0.5, size=len(gdf))
        
        # Simple spatial autocorrelation if enough points
        if len(gdf) > 10:
            try:
                import libpysal
                import esda
                
                # Project to suitable CRS for distance calculations
                gdf_projected = gdf.to_crs(epsg=3857)
                coords = np.array(list(zip(gdf_projected.geometry.x, gdf_projected.geometry.y)))
                w = libpysal.weights.KNN.from_array(coords, k=min(8, len(coords)-1))
                w.transform = "r"
                
                moran_i = esda.moran.Moran(gdf['pred_value'].values, w)
                return gdf, moran_i
            except ImportError:
                st.warning("PySAL not available for spatial autocorrelation analysis")
                return gdf, None
            except Exception as e:
                st.warning(f"Error in spatial analysis: {str(e)}")
                return gdf, None
        else:
            return gdf, None
            
    except Exception as e:
        st.error(f"Error creating spatial analysis: {str(e)}")
        return None, None

def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">🏛️ Government Assets Valuation Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Load data
    with st.spinner("Loading and processing data..."):
        data_result = load_and_process_data()
        
        if data_result[0] is None:
            st.error("Failed to load data. Please check your data sources.")
            return
        
        df_zillow, df_assets, df_zillow_features, date_cols = data_result
        clustering_result = perform_clustering_analysis(df_zillow_features)
        
        if clustering_result[0] is None:
            st.error("Failed to perform clustering analysis.")
            return
            
        df_clustered, optimal_k, inertias, silhouette_scores, K_range, scaler, kmeans_model, pca_model = clustering_result
        gdf_assets, moran_result = create_spatial_analysis(df_assets)
    
    # Sidebar filters
    st.sidebar.subheader("🔍 Filters")
    
    # State filter
    available_states = sorted([s for s in df_assets['State'].dropna().unique() if str(s) != 'nan'])
    selected_states = st.sidebar.multiselect(
        "Select States:", 
        available_states, 
        default=available_states[:5] if len(available_states) > 5 else available_states
    )
    
    # Filter data based on selection
    if selected_states:
        df_assets_filtered = df_assets[df_assets['State'].isin(selected_states)]
        df_zillow_filtered = df_zillow_features[df_zillow_features['State'].isin(selected_states)]
    else:
        df_assets_filtered = df_assets
        df_zillow_filtered = df_zillow_features
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Overview & Statistics", 
        "🗺️ Spatial Analysis", 
        "🎯 Clustering Analysis", 
        "📊 Market Trends", 
        "🔍 Asset Explorer",
        "📋 Scenario Analysis"
    ])
    
    with tab1:
        st.markdown('<h2 class="section-header">📈 Overview & Key Statistics</h2>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Total Assets", f"{len(df_assets_filtered):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            # Fix: Use nunique() directly instead of len(nunique())
            st.metric("States Covered", df_assets_filtered['State'].nunique())
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Zillow Regions", f"{len(df_zillow_filtered):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Optimal Clusters", optimal_k)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Descriptive Statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Housing Price Statistics")
            if not df_zillow_filtered.empty:
                price_cols = ['mean_price', 'median_price', 'std_price', 'price_volatility']
                available_price_cols = [col for col in price_cols if col in df_zillow_filtered.columns]
                
                if available_price_cols:
                    price_stats = df_zillow_filtered[available_price_cols].describe()
                    st.dataframe(price_stats.round(2))
                    
                    # Price distribution
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(data=df_zillow_filtered, x='mean_price', bins=30, kde=True, ax=ax)
                    ax.set_title('Distribution of Mean Housing Prices')
                    ax.set_xlabel('Mean Price ($)')
                    ax.set_ylabel('Frequency')
                    st.pyplot(fig)
        
        with col2:
            st.subheader("🏛️ Assets by State")
            if not df_assets_filtered.empty:
                state_counts = df_assets_filtered['State'].value_counts().head(10)
                
                fig = px.bar(
                    x=state_counts.index,
                    y=state_counts.values,
                    title="Top 10 States by Asset Count",
                    labels={'x': 'State', 'y': 'Number of Assets'}
                )
                fig.update_layout(xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistical tests
                st.subheader("📈 Statistical Tests")
                if len(df_zillow_filtered) > 30:
                    try:
                        # Normality test for mean prices
                        prices = df_zillow_filtered['mean_price'].dropna()
                        if len(prices) > 8:  # Minimum sample size for normality test
                            statistic, p_value = stats.normaltest(prices)
                            st.write(f"**Normality Test (D'Agostino):**")
                            st.write(f"- Statistic: {statistic:.4f}")
                            st.write(f"- P-value: {p_value:.4f}")
                            st.write(f"- Distribution is {'normal' if p_value > 0.05 else 'not normal'} (α=0.05)")
                    except Exception as e:
                        st.warning(f"Could not perform statistical test: {str(e)}")
    
    with tab2:
        st.markdown('<h2 class="section-header">🗺️ Spatial Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🌎 Asset Locations Map")
            
            if gdf_assets is not None and len(gdf_assets) > 0:
                # Filter by selected states
                if selected_states:
                    gdf_filtered = gdf_assets[gdf_assets['State'].isin(selected_states)]
                else:
                    gdf_filtered = gdf_assets
                
                if len(gdf_filtered) > 0:
                    # Create folium map
                    center_lat = gdf_filtered['Latitude'].mean()
                    center_lon = gdf_filtered['Longitude'].mean()
                    
                    m = folium.Map(location=[center_lat, center_lon], zoom_start=4, tiles="CartoDB positron")
                    
                    # Add markers (limit to first 100 for performance)
                    for idx, row in gdf_filtered.head(100).iterrows():
                        folium.CircleMarker(
                            location=[row['Latitude'], row['Longitude']],
                            radius=5,
                            popup=f"{row.get('Real Property Asset Name', 'Asset')}<br>{row['City']}, {row['State']}",
                            color='blue',
                            fill=True,
                            fillOpacity=0.7
                        ).add_to(m)
                    
                    # Display map
                    map_data = st_folium(m, width=700, height=500)
                else:
                    st.warning("No assets with valid coordinates found for selected states.")
            else:
                st.warning("No spatial data available for analysis.")
        
        with col2:
            st.subheader("📊 Spatial Statistics")
            
            if moran_result is not None:
                st.metric("Moran's I", f"{moran_result.I:.4f}")
                st.metric("P-value", f"{moran_result.p_norm:.4f}")
                
                if moran_result.p_norm < 0.05:
                    st.success("Significant spatial clustering detected!")
                else:
                    st.info("No significant spatial clustering.")
            
            # Coordinate statistics
            if gdf_assets is not None and len(gdf_assets) > 0:
                st.subheader("📍 Coordinate Statistics")
                coord_stats = gdf_assets[['Latitude', 'Longitude']].describe()
                st.dataframe(coord_stats.round(4))
        
        # Asset density by state
        st.subheader("📊 Asset Distribution by State")
        if not df_assets_filtered.empty:
            state_dist = df_assets_filtered['State'].value_counts().head(15)
            fig = px.pie(values=state_dist.values, names=state_dist.index, 
                        title="Asset Distribution by State (Top 15)")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="section-header">🎯 Clustering Analysis</h2>', unsafe_allow_html=True)
        
        if len(silhouette_scores) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Elbow Method & Silhouette Analysis")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Elbow curve
                ax1.plot(K_range, inertias, 'o-')
                ax1.set_xlabel('Number of Clusters (K)')
                ax1.set_ylabel('Inertia (WCSS)')
                ax1.set_title('Elbow Method')
                ax1.grid(True)
                
                # Silhouette scores
                ax2.plot(K_range, silhouette_scores, 'o-', color='orange')
                ax2.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal K={optimal_k}')
                ax2.set_xlabel('Number of Clusters (K)')
                ax2.set_ylabel('Silhouette Score')
                ax2.set_title('Silhouette Analysis')
                ax2.legend()
                ax2.grid(True)
                
                st.pyplot(fig)
            
            with col2:
                st.subheader("🎨 PCA Visualization")
                
                if df_clustered is not None and 'pca1' in df_clustered.columns:
                    fig = px.scatter(
                        df_clustered,
                        x='pca1',
                        y='pca2',
                        color='cluster',
                        title='Housing Market Clusters (PCA Projection)',
                        labels={'pca1': 'First Principal Component', 'pca2': 'Second Principal Component'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Cluster characteristics
            st.subheader("📋 Cluster Characteristics")
            
            if df_clustered is not None:
                cluster_cols = ['mean_price', 'price_volatility', 'price_trend_slope', 'recent_12mo_avg']
                available_cluster_cols = [col for col in cluster_cols if col in df_clustered.columns]
                
                if available_cluster_cols:
                    cluster_summary = df_clustered.groupby('cluster')[available_cluster_cols].mean().round(2)
                    st.dataframe(cluster_summary)
        else:
            st.warning("Clustering analysis could not be completed with the current data.")
    
    with tab4:
        st.markdown('<h2 class="section-header">📊 Market Trends Analysis</h2>', unsafe_allow_html=True)
        
        if df_clustered is not None and not df_clustered.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Price Trends by Cluster")
                
                fig = px.box(
                    df_clustered,
                    x='cluster',
                    y='price_trend_slope',
                    title='Price Trend Distribution by Cluster'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation matrix
                st.subheader("🔗 Feature Correlation Matrix")
                corr_cols = ['mean_price', 'price_volatility', 'price_trend_slope', 'recent_12mo_avg']
                available_corr_cols = [col for col in corr_cols if col in df_clustered.columns]
                
                if len(available_corr_cols) > 1:
                    corr_matrix = df_clustered[available_corr_cols].corr()
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                    ax.set_title('Feature Correlation Matrix')
                    st.pyplot(fig)
            
            with col2:
                st.subheader("📊 Volatility Analysis")
                
                fig = px.histogram(
                    df_clustered,
                    x='price_volatility',
                    color='cluster',
                    title='Price Volatility Distribution by Cluster',
                    marginal='box'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                st.subheader("📋 Summary Statistics")
                summary_stats = df_clustered[['mean_price', 'price_volatility', 'price_trend_slope']].describe()
                st.dataframe(summary_stats.round(4))
    
    with tab5:
        st.markdown('<h2 class="section-header">🔍 Asset Explorer</h2>', unsafe_allow_html=True)
        
        # Search and filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_term = st.text_input("🔍 Search Asset Name:", "")
        
        with col2:
            installation_options = ["All"] + sorted([str(x) for x in df_assets_filtered['Installation Name'].dropna().unique() if str(x) != 'nan'])
            installation_filter = st.selectbox("🏢 Filter by Installation:", installation_options)
        
        with col3:
            state_options = ["All"] + sorted([str(x) for x in df_assets_filtered['State'].dropna().unique() if str(x) != 'nan'])
            state_filter = st.selectbox("🗺️ Filter by State:", state_options)
        
        # Apply filters
        filtered_assets = df_assets_filtered.copy()
        
        if search_term:
            filtered_assets = filtered_assets[
                filtered_assets['Real Property Asset Name'].str.contains(search_term, case=False, na=False)
            ]
        
        if installation_filter != "All":
            filtered_assets = filtered_assets[filtered_assets['Installation Name'] == installation_filter]
        
        if state_filter != "All":
            filtered_assets = filtered_assets[filtered_assets['State'] == state_filter]
        
        # Display results
        st.subheader(f"📋 Assets Found: {len(filtered_assets)}")
        
        if len(filtered_assets) > 0:
            # Display key columns
            display_cols = ['Real Property Asset Name', 'City', 'State', 'Installation Name']
            available_cols = [col for col in display_cols if col in filtered_assets.columns]
            
            st.dataframe(
                filtered_assets[available_cols].head(100),
                use_container_width=True
            )
            
            # Summary statistics for filtered assets
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Filtered Assets by State")
                if len(filtered_assets) > 0:
                    state_counts = filtered_assets['State'].value_counts().head(10)
                    fig = px.bar(x=state_counts.index, y=state_counts.values, 
                               title="Distribution by State")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'Building Rentable Square Feet' in filtered_assets.columns:
                    st.subheader("🏢 Building Size Distribution")
                    building_sizes = pd.to_numeric(filtered_assets['Building Rentable Square Feet'], errors='coerce').dropna()
                    if len(building_sizes) > 0:
                        fig = px.histogram(building_sizes, title="Building Rentable Square Feet Distribution", 
                                         nbins=20)
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No assets match the current filters.")
    
    with tab6:
        st.markdown('<h2 class="section-header">📋 Scenario Analysis</h2>', unsafe_allow_html=True)
        
        st.subheader("🎯 Market Impact Simulation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Price Change Scenario")
            price_change = st.slider("Price Change (%)", -50, 50, 5, 1)
            
            if df_clustered is not None and not df_clustered.empty:
                # Simulate impact
                simulated_impact = df_clustered['mean_price'] * (1 + price_change / 100)
                impact_diff = simulated_impact - df_clustered['mean_price']
                
                fig = px.histogram(
                    x=impact_diff,
                    title=f"Simulated Price Impact Distribution ({price_change}% change)",
                    labels={'x': 'Price Change ($)', 'y': 'Frequency'},
                    nbins=30
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.metric("Average Impact", f"${impact_diff.mean():,.0f}")
        
        with col2:
            st.subheader("📊 Volatility Impact Analysis")
            if df_clustered is not None and not df_clustered.empty:
                volatility_threshold = st.slider("Volatility Threshold", 0.0, 1.0, 0.2, 0.01)
                
                high_volatility = df_clustered[df_clustered['price_volatility'] > volatility_threshold]
                low_volatility = df_clustered[df_clustered['price_volatility'] <= volatility_threshold]
                
                fig = go.Figure()
                if len(high_volatility) > 0:
                    fig.add_trace(go.Histogram(x=high_volatility['mean_price'], name='High Volatility', opacity=0.7))
                if len(low_volatility) > 0:
                    fig.add_trace(go.Histogram(x=low_volatility['mean_price'], name='Low Volatility', opacity=0.7))
                
                fig.update_layout(
                    title=f'Price Distribution by Volatility (Threshold: {volatility_threshold})',
                    xaxis_title='Mean Price ($)',
                    yaxis_title='Count',
                    barmode='overlay'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Risk Assessment
        st.subheader("⚠️ Risk Assessment")
        
        if df_clustered is not None and not df_clustered.empty:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                high_risk_count = len(df_clustered[df_clustered['price_volatility'] > 0.3])
                st.metric("High Risk Regions", high_risk_count)
            
            with col2:
                declining_trend = len(df_clustered[df_clustered['price_trend_slope'] < 0])
                st.metric("Declining Trend Regions", declining_trend)
            
            with col3:
                stable_regions = len(df_clustered[
                    (df_clustered['price_volatility'] <= 0.2) & 
                    (df_clustered['price_trend_slope'] >= 0)
                ])
                st.metric("Stable Regions", stable_regions)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>🏛️ Government Assets Valuation Dashboard | Built with Streamlit</p>
            <p>Data Sources: Zillow Housing Index & US Government Assets Database</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

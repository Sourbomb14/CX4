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
import libpysal
import esda

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
    """Download datasets and models using gdown."""
    
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
    for file_info in files_to_download.values():
        if not os.path.exists(file_info["path"]):
            with st.spinner(f"Downloading {file_info['path']}..."):
                gdown.download(file_info["url"], file_info["path"], quiet=False)
    
    return files_to_download

@st.cache_data
def load_and_process_data():
    """Load and process the datasets."""
    
    # Download data first
    download_data()
    
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
    if len(df_zillow) > 5000:
        df_zillow = df_zillow.sample(n=5000, random_state=42).reset_index(drop=True)
    
    # Process Zillow time series data
    date_cols = [c for c in df_zillow.columns if c[:2].isdigit()]
    df_zillow[date_cols] = df_zillow[date_cols].apply(pd.to_numeric, errors='coerce')
    
    # Engineer features
    features = []
    for idx, row in df_zillow.iterrows():
        prices = row[date_cols].values.astype(float)
        prices = prices[~np.isnan(prices)]
        
        if len(prices) > 0:
            features.append({
                'RegionID': row.get('RegionID', idx),
                'RegionName': row.get('RegionName', ''),
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
    
    df_zillow_features = pd.DataFrame(features)
    
    return df_zillow, df_assets, df_zillow_features, date_cols

def perform_clustering_analysis(df_features):
    """Perform clustering analysis on Zillow features."""
    
    num_cols = ['mean_price', 'median_price', 'std_price', 'price_min', 'price_max', 
                'price_range', 'price_volatility', 'recent_6mo_avg', 'recent_12mo_avg', 
                'last_price', 'price_trend_slope']
    
    # Scale features
    scaler = MinMaxScaler()
    df_scaled = df_features.copy()
    df_scaled[num_cols] = scaler.fit_transform(df_features[num_cols].fillna(0))
    
    # Determine optimal number of clusters
    K_range = range(2, 10)
    inertias = []
    silhouette_scores = []
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(df_scaled[num_cols])
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(df_scaled[num_cols], cluster_labels))
    
    # Find optimal k
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
    
    # Filter for valid coordinates
    valid_coords = df_assets.dropna(subset=['Latitude', 'Longitude'])
    
    if len(valid_coords) == 0:
        return None, None
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        valid_coords,
        geometry=gpd.points_from_xy(valid_coords['Longitude'], valid_coords['Latitude']),
        crs="EPSG:4326"
    )
    
    # Project to suitable CRS for distance calculations
    gdf_projected = gdf.to_crs(epsg=3857)
    
    # Simulate some values for spatial analysis if not available
    if 'pred_value' not in gdf.columns:
        np.random.seed(42)
        gdf['pred_value'] = np.random.lognormal(mean=12, sigma=0.5, size=len(gdf))
    
    # Spatial autocorrelation (Moran's I)
    try:
        coords = np.array(list(zip(gdf_projected.geometry.x, gdf_projected.geometry.y)))
        w = libpysal.weights.KNN.from_array(coords, k=min(8, len(coords)-1))
        w.transform = "r"
        
        moran_i = esda.moran.Moran(gdf['pred_value'].values, w)
        
        return gdf, moran_i
    except:
        return gdf, None

def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">🏛️ Government Assets Valuation Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Load data
    with st.spinner("Loading and processing data..."):
        df_zillow, df_assets, df_zillow_features, date_cols = load_and_process_data()
        df_clustered, optimal_k, inertias, silhouette_scores, K_range, scaler, kmeans_model, pca_model = perform_clustering_analysis(df_zillow_features)
        gdf_assets, moran_result = create_spatial_analysis(df_assets)
    
    # Sidebar filters
    st.sidebar.subheader("🔍 Filters")
    
    # State filter
    available_states = sorted(df_assets['State'].dropna().unique())
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
            st.metric("States Covered", len(df_assets_filtered['State'].nunique()))
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
                price_stats = df_zillow_filtered[['mean_price', 'median_price', 'std_price', 'price_volatility']].describe()
                st.dataframe(price_stats.round(2))
                
                # Price distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data=df_zillow_filtered, x='mean_price', bins=50, kde=True, ax=ax)
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
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistical tests
                st.subheader("📈 Statistical Tests")
                if len(df_zillow_filtered) > 1:
                    # Normality test for mean prices
                    statistic, p_value = stats.normaltest(df_zillow_filtered['mean_price'].dropna())
                    st.write(f"**Normality Test (D'Agostino):**")
                    st.write(f"- Statistic: {statistic:.4f}")
                    st.write(f"- P-value: {p_value:.4f}")
                    st.write(f"- Distribution is {'normal' if p_value > 0.05 else 'not normal'} (α=0.05)")
    
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
                    
                    # Add markers
                    for idx, row in gdf_filtered.iterrows():
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
        
        # Heatmap of asset density
        st.subheader("🔥 Asset Density Heatmap")
        if gdf_assets is not None and len(gdf_assets) > 0:
            fig = px.density_mapbox(
                gdf_assets,
                lat='Latitude',
                lon='Longitude',
                radius=10,
                center=dict(lat=gdf_assets['Latitude'].mean(), lon=gdf_assets['Longitude'].mean()),
                zoom=3,
                mapbox_style="open-street-map",
                title="Government Asset Density Across US"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="section-header">🎯 Clustering Analysis</h2>', unsafe_allow_html=True)
        
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
        
        cluster_summary = df_clustered.groupby('cluster')[
            ['mean_price', 'price_volatility', 'price_trend_slope', 'recent_12mo_avg']
        ].mean().round(2)
        
        st.dataframe(cluster_summary)
        
        # Cluster distribution by state
        if 'State' in df_clustered.columns:
            fig = px.sunburst(
                df_clustered,
                path=['cluster', 'State'],
                title='Cluster Distribution by State'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown('<h2 class="section-header">📊 Market Trends Analysis</h2>', unsafe_allow_html=True)
        
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
            corr_matrix = df_clustered[corr_cols].corr()
            
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
            
            # Time series analysis (if date columns available)
            if date_cols and len(date_cols) > 0:
                st.subheader("📅 Time Series Sample")
                
                # Select a random sample for time series visualization
                sample_regions = df_zillow.sample(min(5, len(df_zillow)))
                
                fig = go.Figure()
                
                for idx, row in sample_regions.iterrows():
                    prices = [row[col] for col in date_cols if pd.notna(row[col])]
                    if prices:
                        fig.add_trace(go.Scatter(
                            x=list(range(len(prices))),
                            y=prices,
                            mode='lines',
                            name=f"{row.get('RegionName', f'Region {idx}')}"
                        ))
                
                fig.update_layout(
                    title='Sample Housing Price Time Series',
                    xaxis_title='Time Period',
                    yaxis_title='Price ($)',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown('<h2 class="section-header">🔍 Asset Explorer</h2>', unsafe_allow_html=True)
        
        # Search and filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_term = st.text_input("🔍 Search Asset Name:", "")
        
        with col2:
            installation_filter = st.selectbox(
                "🏢 Filter by Installation:",
                ["All"] + sorted(df_assets_filtered['Installation Name'].dropna().unique())
            )
        
        with col3:
            state_filter = st.selectbox(
                "🗺️ Filter by State:",
                ["All"] + sorted(df_assets_filtered['State'].dropna().unique())
            )
        
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
                state_counts = filtered_assets['State'].value_counts()
                fig = px.pie(values=state_counts.values, names=state_counts.index, title="Distribution by State")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'Building Rentable Square Feet' in filtered_assets.columns:
                    st.subheader("🏢 Building Size Distribution")
                    building_sizes = pd.to_numeric(filtered_assets['Building Rentable Square Feet'], errors='coerce').dropna()
                    if len(building_sizes) > 0:
                        fig = px.histogram(building_sizes, title="Building Rentable Square Feet Distribution")
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
            
            # Simulate impact
            simulated_impact = df_clustered['mean_price'] * (1 + price_change / 100)
            impact_diff = simulated_impact - df_clustered['mean_price']
            
            fig = px.histogram(
                x=impact_diff,
                title=f"Simulated Price Impact Distribution ({price_change}% change)",
                labels={'x': 'Price Change ($)', 'y': 'Frequency'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("Average Impact", f"${impact_diff.mean():,.0f}")
        
        with col2:
            st.subheader("📊 Volatility Impact Analysis")
            volatility_threshold = st.slider("Volatility Threshold", 0.0, 1.0, 0.2, 0.01)
            
            high_volatility = df_clustered[df_clustered['price_volatility'] > volatility_threshold]
            low_volatility = df_clustered[df_clustered['price_volatility'] <= volatility_threshold]
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=high_volatility['mean_price'], name='High Volatility', opacity=0.7))
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

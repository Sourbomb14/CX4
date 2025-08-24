import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import geopandas as gpd
from shapely.geometry import Point
import requests
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

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
</style>
""", unsafe_allow_html=True)

# Cache data loading functions
@st.cache_data
def load_assets_data():
    """Load US Government Assets dataset"""
    try:
        assets_url = "https://drive.google.com/uc?id=1YFTWJNoxu0BF8UlMDXI8bXwRTVQNE2mb"
        response = requests.get(assets_url)
        with open("temp_assets.csv", "wb") as f:
            f.write(response.content)
        df = pd.read_csv("temp_assets.csv", encoding='utf-8')
        return df
    except:
        try:
            df = pd.read_csv("temp_assets.csv", encoding='latin-1')
            return df
        except Exception as e:
            st.error(f"Error loading assets data: {e}")
            return None

@st.cache_data
def load_housing_data():
    """Load Zillow Housing Price Index dataset"""
    try:
        housing_url = "https://drive.google.com/uc?id=1fFT8Q8GWiIEM7kx6czhQ-qabygUPBQRv"
        response = requests.get(housing_url)
        with open("temp_housing.csv", "wb") as f:
            f.write(response.content)
        df = pd.read_csv("temp_housing.csv", encoding='utf-8')
        return df
    except:
        try:
            df = pd.read_csv("temp_housing.csv", encoding='latin-1')
            return df
        except Exception as e:
            st.error(f"Error loading housing data: {e}")
            return None

@st.cache_data
def clean_and_merge_data(df_assets, df_prices):
    """Clean and merge the datasets"""
    # Clean column names
    df_assets.columns = df_assets.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    df_prices.columns = df_prices.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    
    # Filter valid coordinates
    if 'latitude' in df_assets.columns and 'longitude' in df_assets.columns:
        valid_coords = (
            (df_assets['latitude'] >= 24) & (df_assets['latitude'] <= 49) &
            (df_assets['longitude'] >= -125) & (df_assets['longitude'] <= -66)
        )
        df_assets = df_assets[valid_coords]
    
    # Get latest housing price index
    price_cols = [col for col in df_prices.columns if any(year in str(col) for year in ['2024', '2025'])]
    if price_cols:
        latest_col = sorted(price_cols, reverse=True)[0]
        df_prices['latest_price_index'] = pd.to_numeric(df_prices[latest_col], errors='coerce')
    
    # Create merge keys
    if 'city' in df_assets.columns and 'state' in df_assets.columns:
        df_assets['city_state_key'] = (
            df_assets['city'].str.lower().str.strip() + '_' + 
            df_assets['state'].str.lower().str.strip()
        )
    
    if 'city' in df_prices.columns and 'state' in df_prices.columns:
        df_prices['city_state_key'] = (
            df_prices['city'].str.lower().str.strip() + '_' + 
            df_prices['state'].str.lower().str.strip()
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
        merged_df['latest_price_index'] = 100
    
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
        merged_df['estimated_value'] = merged_df['latest_price_index'] * 1000
    
    # Add high-value premium for certain states
    high_value_states = ['CA', 'NY', 'MA', 'CT', 'NJ', 'HI', 'MD', 'WA']
    if 'state' in merged_df.columns:
        premium_mask = merged_df['state'].isin(high_value_states)
        merged_df.loc[premium_mask, 'estimated_value'] *= 1.5
    
    return merged_df

@st.cache_data
def perform_clustering(df, n_clusters=5):
    """Perform clustering analysis"""
    numeric_cols = ['latitude', 'longitude', 'estimated_value']
    if 'building_rentable_square_feet' in df.columns:
        numeric_cols.append('building_rentable_square_feet')
    
    # Prepare data for clustering
    cluster_data = df[numeric_cols].fillna(df[numeric_cols].median())
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled_data)
    
    return df, kmeans

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
    numeric_features = ['latitude', 'longitude', 'latest_price_index']
    if 'building_rentable_square_feet' in df.columns:
        numeric_features.append('building_rentable_square_feet')
    
    features.extend(numeric_features)
    
    return df, features

def create_folium_map(df, sample_size=1000):
    """Create interactive Folium map"""
    # Sample data for performance
    if len(df) > sample_size:
        map_data = df.sample(n=sample_size, random_state=42)
    else:
        map_data = df.copy()
    
    # Calculate center
    center_lat = map_data['latitude'].mean()
    center_lon = map_data['longitude'].mean()
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=4)
    
    # Add markers
    for _, row in map_data.iterrows():
        if 'cluster' in row:
            color = ['red', 'blue', 'green', 'purple', 'orange'][int(row['cluster']) % 5]
        else:
            color = 'blue'
        
        popup_text = f"""
        <b>Government Asset</b><br>
        Location: {row.get('city', 'N/A')}, {row.get('state', 'N/A')}<br>
        Estimated Value: ${row.get('estimated_value', 0):,.0f}<br>
        Cluster: {row.get('cluster', 'N/A')}
        """
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            popup=folium.Popup(popup_text, max_width=300),
            color='black',
            fillColor=color,
            fillOpacity=0.7,
            weight=1
        ).add_to(m)
    
    return m

def main():
    # Header
    st.markdown('<h1 class="main-header">🏛️ US Government Assets Portfolio Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.image("https://via.placeholder.com/300x100/1f4e79/ffffff?text=Analytics+Dashboard", 
                     use_column_width=True)
    st.sidebar.markdown("### 📊 Navigation")
    
    # Load data
    with st.spinner("Loading datasets..."):
        df_assets = load_assets_data()
        df_prices = load_housing_data()
    
    if df_assets is None or df_prices is None:
        st.error("Could not load data. Please check the data sources.")
        return
    
    # Merge and clean data
    with st.spinner("Processing and merging data..."):
        df_merged = clean_and_merge_data(df_assets, df_prices)
    
    # Sidebar filters
    st.sidebar.markdown("### 🔍 Filters")
    
    states = ['All'] + sorted(df_merged['state'].unique().tolist()) if 'state' in df_merged.columns else ['All']
    selected_state = st.sidebar.selectbox("Select State", states)
    
    if selected_state != 'All':
        df_filtered = df_merged[df_merged['state'] == selected_state]
    else:
        df_filtered = df_merged.copy()
    
    # Value range filter
    if 'estimated_value' in df_filtered.columns:
        min_value = int(df_filtered['estimated_value'].min())
        max_value = int(df_filtered['estimated_value'].max())
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
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Assets",
            f"{len(df):,}",
            delta=None
        )
    
    with col2:
        if 'estimated_value' in df.columns:
            total_value = df['estimated_value'].sum()
            st.metric(
                "Portfolio Value",
                f"${total_value/1e9:.1f}B",
                delta=None
            )
    
    with col3:
        if 'estimated_value' in df.columns:
            avg_value = df['estimated_value'].mean()
            st.metric(
                "Average Asset Value",
                f"${avg_value/1e6:.1f}M",
                delta=None
            )
    
    with col4:
        if 'state' in df.columns:
            states_count = df['state'].nunique()
            st.metric(
                "States Covered",
                f"{states_count}",
                delta=None
            )
    
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
    
    # Portfolio insights
    st.subheader("💡 Key Insights")
    
    if 'estimated_value' in df.columns and 'state' in df.columns:
        top_state = df['state'].value_counts().index[0]
        top_state_value = df[df['state'] == top_state]['estimated_value'].sum()
        
        insights = [
            f"📍 **{top_state}** has the highest number of assets ({df[df['state'] == top_state].shape[0]:,})",
            f"💰 Total portfolio value: **${df['estimated_value'].sum()/1e9:.1f} billion**",
            f"📊 Average asset value: **${df['estimated_value'].mean()/1e6:.1f} million**",
            f"🏛️ Value concentration: Top 10% of assets represent {((df['estimated_value'].quantile(0.9) * 0.1 * len(df)) / df['estimated_value'].sum() * 100):.1f}% of total value"
        ]
        
        for insight in insights:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

def show_geographic_analysis(df):
    """Show geographic analysis"""
    st.header("🗺️ Geographic Analysis")
    
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Interactive map
        st.subheader("Interactive Asset Map")
        
        # Perform clustering for map colors
        df_clustered, _ = perform_clustering(df, n_clusters=5)
        
        # Create and display map
        map_obj = create_folium_map(df_clustered)
        map_data = st_folium(map_obj, width=700, height=500)
        
        # Geographic statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Geographic Distribution")
            if 'state' in df.columns:
                state_stats = df.groupby('state').agg({
                    'estimated_value': ['count', 'sum', 'mean'],
                    'latitude': 'mean',
                    'longitude': 'mean'
                }).round(2)
                
                state_stats.columns = ['Asset Count', 'Total Value', 'Avg Value', 'Center Lat', 'Center Lon']
                state_stats = state_stats.sort_values('Total Value', ascending=False)
                
                st.dataframe(state_stats.head(10))
        
        with col2:
            st.subheader("Value Heatmap by Region")
            if 'state' in df.columns and 'estimated_value' in df.columns:
                state_values = df.groupby('state')['estimated_value'].sum().sort_values(ascending=False).head(15)
                
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
        st.error("Geographic coordinates not available in the dataset.")

def show_clustering_analysis(df):
    """Show clustering analysis"""
    st.header("🎯 Clustering Analysis")
    
    # Clustering parameters
    col1, col2 = st.columns([1, 3])
    
    with col1:
        n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=5)
        clustering_method = st.selectbox("Clustering Method", ["K-Means", "DBSCAN"])
    
    with col2:
        # Perform clustering
        if clustering_method == "K-Means":
            df_clustered, model = perform_clustering(df, n_clusters=n_clusters)
        else:
            # DBSCAN implementation
            numeric_cols = ['latitude', 'longitude', 'estimated_value']
            if 'building_rentable_square_feet' in df.columns:
                numeric_cols.append('building_rentable_square_feet')
            
            cluster_data = df[numeric_cols].fillna(df[numeric_cols].median())
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            dbscan = DBSCAN(eps=0.3, min_samples=5)
            df['cluster'] = dbscan.fit_predict(scaled_data)
            df_clustered = df.copy()
        
        # Cluster visualization
        if 'latitude' in df_clustered.columns and 'longitude' in df_clustered.columns:
            fig = px.scatter_mapbox(
                df_clustered.sample(n=min(1000, len(df_clustered))),
                lat="latitude",
                lon="longitude",
                color="cluster",
                size="estimated_value" if 'estimated_value' in df_clustered.columns else None,
                hover_data={'estimated_value': ':$,.0f'} if 'estimated_value' in df_clustered.columns else None,
                mapbox_style="open-street-map",
                zoom=3,
                height=600,
                title="Asset Clusters Geographic Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Cluster analysis
    st.subheader("📊 Cluster Analysis")
    
    if 'cluster' in df_clustered.columns:
        cluster_stats = df_clustered.groupby('cluster').agg({
            'estimated_value': ['count', 'sum', 'mean', 'std'],
            'latitude': 'mean',
            'longitude': 'mean'
        }).round(2)
        
        cluster_stats.columns = ['Count', 'Total Value', 'Avg Value', 'Value Std', 'Center Lat', 'Center Lon']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Cluster Statistics**")
            st.dataframe(cluster_stats)
        
        with col2:
            # Cluster size pie chart
            cluster_counts = df_clustered['cluster'].value_counts()
            fig = px.pie(
                values=cluster_counts.values,
                names=[f'Cluster {i}' for i in cluster_counts.index],
                title="Assets Distribution by Cluster"
            )
            st.plotly_chart(fig, use_container_width=True)

def show_machine_learning(df):
    """Show machine learning analysis"""
    st.header("🤖 Machine Learning Analysis")
    
    if 'estimated_value' not in df.columns:
        st.error("Estimated value column not found. Cannot perform ML analysis.")
        return
    
    # Prepare features
    df_ml, features = create_ml_features(df)
    
    # Ensure all features exist in the dataframe
    available_features = [f for f in features if f in df_ml.columns]
    
    if len(available_features) == 0:
        st.error("No suitable features found for ML analysis.")
        return
    
    # Prepare data
    X = df_ml[available_features].fillna(df_ml[available_features].median())
    
    # ML tasks
    task = st.selectbox("Select ML Task", 
                       ["Value Prediction (Regression)", "Value Classification", "High-Value Detection"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        if task == "Value Prediction (Regression)":
            st.subheader("🎯 Asset Value Prediction")
            
            # Prepare target
            y = df_ml['estimated_value']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            
            st.metric("R² Score", f"{r2:.3f}")
            
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
                title="Top 10 Feature Importances"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif task == "Value Classification":
            st.subheader("📊 Asset Value Classification")
            
            # Create value categories
            y = pd.qcut(df_ml['estimated_value'], q=3, labels=['Low', 'Medium', 'High'])
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            st.metric("Accuracy", f"{accuracy:.3f}")
            
            # Class distribution
            class_dist = y.value_counts()
            fig = px.pie(
                values=class_dist.values,
                names=class_dist.index,
                title="Asset Value Categories Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        else:  # High-Value Detection
            st.subheader("🎯 High-Value Asset Detection")
            
            # Binary target (top 25% as high-value)
            threshold = df_ml['estimated_value'].quantile(0.75)
            y = (df_ml['estimated_value'] > threshold).astype(int)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            st.metric("Accuracy", f"{accuracy:.3f}")
            st.metric("High-Value Threshold", f"${threshold/1e6:.1f}M")
            
            # Distribution
            dist_data = pd.DataFrame({
                'Category': ['Regular Value', 'High Value'],
                'Count': [(y == 0).sum(), (y == 1).sum()]
            })
            
            fig = px.bar(
                dist_data,
                x='Category',
                y='Count',
                title="Asset Value Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🔧 Model Details")
        
        st.write("**Features Used:**")
        for feature in available_features:
            st.write(f"• {feature.replace('_', ' ').title()}")
        
        st.write(f"**Dataset Size:** {len(df_ml):,} assets")
        st.write(f"**Features Count:** {len(available_features)}")
        
        if task == "Value Prediction (Regression)":
            # Actual vs Predicted scatter
            if 'y_pred' in locals() and 'y_test' in locals():
                comparison_df = pd.DataFrame({
                    'Actual': y_test.values,
                    'Predicted': y_pred
                })
                
                fig = px.scatter(
                    comparison_df,
                    x='Actual',
                    y='Predicted',
                    title="Actual vs Predicted Values",
                    trendline="ols"
                )
                st.plotly_chart(fig, use_container_width=True)

def show_advanced_analytics(df):
    """Show advanced analytics"""
    st.header("📈 Advanced Analytics")
    
    tabs = st.tabs(["📊 Statistical Analysis", "🔍 Data Quality", "📈 Trends"])
    
    with tabs[0]:
        st.subheader("Statistical Summary")
        
        if 'estimated_value' in df.columns:
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
            st.plotly_chart(fig, use_container_width=True)
    
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
    
    with tabs[2]:
        st.subheader("Portfolio Trends")
        
        if 'state' in df.columns and 'estimated_value' in df.columns:
            # State-wise analysis
            state_analysis = df.groupby('state').agg({
                'estimated_value': ['count', 'sum', 'mean'],
                'building_rentable_square_feet': 'mean' if 'building_rentable_square_feet' in df.columns else 'count'
            }).round(2)
            
            state_analysis.columns = ['Asset Count', 'Total Value', 'Avg Value', 'Avg Sqft']
            state_analysis = state_analysis.sort_values('Total Value', ascending=False).head(20)
            
            # Bubble chart
            fig = px.scatter(
                state_analysis.reset_index(),
                x='Asset Count',
                y='Avg Value',
                size='Total Value',
                color='Avg Sqft' if 'building_rentable_square_feet' in df.columns else 'Asset Count',
                hover_name='state',
                title="State Portfolio Analysis (Bubble Chart)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top states table
            st.write("**Top 20 States by Portfolio Value:**")
            st.dataframe(state_analysis)

if __name__ == "__main__":
    main()

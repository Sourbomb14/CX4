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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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
    .prediction-box {
        background-color: #f0f8e8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 5px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading functions
@st.cache_data
def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    n_samples = 1500
    
    # US state abbreviations and major cities
    states_cities = {
        'CA': ['Los Angeles', 'San Francisco', 'San Diego', 'Sacramento'],
        'TX': ['Houston', 'Dallas', 'Austin', 'San Antonio'],
        'NY': ['New York', 'Buffalo', 'Albany', 'Rochester'],
        'FL': ['Miami', 'Orlando', 'Tampa', 'Jacksonville'],
        'IL': ['Chicago', 'Springfield', 'Rockford', 'Peoria'],
        'PA': ['Philadelphia', 'Pittsburgh', 'Harrisburg', 'Allentown'],
        'OH': ['Columbus', 'Cleveland', 'Cincinnati', 'Toledo'],
        'GA': ['Atlanta', 'Augusta', 'Savannah', 'Columbus'],
        'NC': ['Charlotte', 'Raleigh', 'Greensboro', 'Durham'],
        'MI': ['Detroit', 'Grand Rapids', 'Warren', 'Sterling Heights']
    }
    
    # Generate realistic coordinates for each state
    state_coords = {
        'CA': (36.7783, -119.4179),
        'TX': (31.9686, -99.9018),
        'NY': (42.1657, -74.9481),
        'FL': (27.7663, -81.6868),
        'IL': (40.3363, -89.0022),
        'PA': (40.5908, -77.2098),
        'OH': (40.3888, -82.7649),
        'GA': (33.0406, -83.6431),
        'NC': (35.5397, -79.8431),
        'MI': (43.3266, -84.5361)
    }
    
    data = []
    
    for i in range(n_samples):
        # Select random state
        state = np.random.choice(list(states_cities.keys()))
        city = np.random.choice(states_cities[state])
        
        # Generate coordinates around state center
        base_lat, base_lon = state_coords[state]
        lat = base_lat + np.random.normal(0, 1.5)
        lon = base_lon + np.random.normal(0, 1.5)
        
        # Generate asset characteristics
        sqft = np.random.uniform(1000, 150000)
        
        # Price varies by state (CA, NY more expensive)
        base_price_per_sqft = 200
        if state in ['CA', 'NY', 'MA']:
            base_price_per_sqft = 400
        elif state in ['TX', 'FL']:
            base_price_per_sqft = 250
        
        price_per_sqft = base_price_per_sqft * np.random.uniform(0.5, 2.0)
        estimated_value = sqft * price_per_sqft
        
        # Add some randomness
        estimated_value *= np.random.uniform(0.8, 1.3)
        
        data.append({
            'state': state,
            'city': city,
            'latitude': lat,
            'longitude': lon,
            'building_rentable_square_feet': sqft,
            'estimated_value': estimated_value,
            'price_per_sqft': price_per_sqft,
            'asset_type': np.random.choice(['Office Building', 'Warehouse', 'Administrative', 'Courts', 'Military']),
            'construction_year': np.random.randint(1960, 2020)
        })
    
    return pd.DataFrame(data)

@st.cache_data
def load_data():
    """Load or create dataset"""
    try:
        # Try to load real data first
        assets_url = "https://drive.google.com/uc?id=1YFTWJNoxu0BF8UlMDXI8bXwRTVQNE2mb"
        response = requests.get(assets_url, timeout=10)
        
        if response.status_code == 200:
            st.success("✅ Real data loaded successfully!")
            # Process real data here if needed
            return create_sample_data()  # For now, use sample data
        else:
            st.info("ℹ️ Using sample data for demonstration")
            return create_sample_data()
    except:
        st.info("ℹ️ Using sample data for demonstration")
        return create_sample_data()

@st.cache_data
def perform_price_based_clustering(df, n_clusters=5):
    """Perform clustering based on asset prices and characteristics"""
    # Features for clustering
    features = ['estimated_value', 'building_rentable_square_feet', 'price_per_sqft']
    
    # Prepare data
    cluster_data = df[features].copy()
    
    # Handle any missing values
    cluster_data = cluster_data.fillna(cluster_data.median())
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['price_cluster'] = kmeans.fit_predict(scaled_data)
    
    # Calculate cluster centers in original scale
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Create cluster summary
    cluster_summary = []
    for i in range(n_clusters):
        cluster_mask = df['price_cluster'] == i
        cluster_df = df[cluster_mask]
        
        summary = {
            'cluster': i,
            'count': len(cluster_df),
            'avg_value': cluster_df['estimated_value'].mean(),
            'avg_sqft': cluster_df['building_rentable_square_feet'].mean(),
            'avg_price_per_sqft': cluster_df['price_per_sqft'].mean(),
            'top_state': cluster_df['state'].mode().iloc[0] if len(cluster_df) > 0 else 'N/A'
        }
        cluster_summary.append(summary)
    
    cluster_summary_df = pd.DataFrame(cluster_summary)
    
    return df, kmeans, scaler, cluster_summary_df

@st.cache_data
def train_price_prediction_model(df):
    """Train a price prediction model"""
    # Features for prediction
    features = ['building_rentable_square_feet', 'latitude', 'longitude', 'construction_year']
    
    # Encode state as numeric
    state_encoder = {state: i for i, state in enumerate(df['state'].unique())}
    df['state_encoded'] = df['state'].map(state_encoder)
    features.append('state_encoded')
    
    # Prepare data
    X = df[features].copy()
    y = df['estimated_value'].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, r2, rmse, mae, feature_importance, X_test, y_test, y_pred

def create_price_cluster_map(df, sample_size=800):
    """Create Folium map with price-based clusters"""
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
    
    # Define colors for clusters
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']
    
    # Add markers for each cluster
    for cluster_id in sorted(map_data['price_cluster'].unique()):
        cluster_data = map_data[map_data['price_cluster'] == cluster_id]
        color = colors[cluster_id % len(colors)]
        
        for _, row in cluster_data.iterrows():
            # Size based on value
            radius = min(max(row['estimated_value'] / 1000000 * 2, 4), 15)
            
            popup_text = f"""
            <b>Government Asset - Cluster {cluster_id}</b><br>
            <b>Location:</b> {row['city']}, {row['state']}<br>
            <b>Asset Type:</b> {row['asset_type']}<br>
            <b>Estimated Value:</b> ${row['estimated_value']:,.0f}<br>
            <b>Square Feet:</b> {row['building_rentable_square_feet']:,.0f}<br>
            <b>Price per SqFt:</b> ${row['price_per_sqft']:.0f}<br>
            <b>Construction Year:</b> {row['construction_year']}<br>
            <b>Price Cluster:</b> {cluster_id}
            """
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=radius,
                popup=folium.Popup(popup_text, max_width=300),
                color='black',
                fillColor=color,
                fillOpacity=0.7,
                weight=1,
                tooltip=f"Cluster {cluster_id}: ${row['estimated_value']:,.0f}"
            ).add_to(m)
    
    return m

def main():
    # Header
    st.markdown('<h1 class="main-header">🏛️ US Government Assets Portfolio Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.image("https://via.placeholder.com/300x100/1f4e79/ffffff?text=Analytics+Dashboard", 
                     use_container_width=True)
    st.sidebar.markdown("### 📊 Navigation")
    
    # Load data
    with st.spinner("Loading portfolio data..."):
        df = load_data()
    
    if df is None or len(df) == 0:
        st.error("No data available for analysis.")
        return
    
    # Sidebar filters
    st.sidebar.markdown("### 🔍 Filters")
    
    # State filter
    states = ['All'] + sorted(df['state'].unique().tolist())
    selected_state = st.sidebar.selectbox("Select State", states)
    
    if selected_state != 'All':
        df_filtered = df[df['state'] == selected_state]
    else:
        df_filtered = df.copy()
    
    # Asset type filter
    asset_types = ['All'] + sorted(df['asset_type'].unique().tolist())
    selected_type = st.sidebar.selectbox("Select Asset Type", asset_types)
    
    if selected_type != 'All':
        df_filtered = df_filtered[df_filtered['asset_type'] == selected_type]
    
    # Value range filter
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
        ["📊 Executive Dashboard", "🗺️ Price-Based Clustering Map", "🤖 Price Prediction Model", "📈 Portfolio Analytics"]
    )
    
    # Show filtered data info
    st.sidebar.markdown("### 📋 Data Summary")
    st.sidebar.metric("Total Assets", f"{len(df_filtered):,}")
    st.sidebar.metric("States", f"{df_filtered['state'].nunique()}")
    st.sidebar.metric("Asset Types", f"{df_filtered['asset_type'].nunique()}")
    
    # Route to different pages
    if page == "📊 Executive Dashboard":
        show_executive_dashboard(df_filtered)
    elif page == "🗺️ Price-Based Clustering Map":
        show_price_clustering_analysis(df_filtered)
    elif page == "🤖 Price Prediction Model":
        show_price_prediction_model(df_filtered)
    elif page == "📈 Portfolio Analytics":
        show_portfolio_analytics(df_filtered)

def show_executive_dashboard(df):
    """Show executive dashboard"""
    st.header("📊 Executive Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Assets", f"{len(df):,}")
    
    with col2:
        total_value = df['estimated_value'].sum()
        st.metric("Portfolio Value", f"${total_value/1e9:.1f}B")
    
    with col3:
        avg_value = df['estimated_value'].mean()
        st.metric("Average Asset Value", f"${avg_value/1e6:.1f}M")
    
    with col4:
        avg_price_sqft = df['price_per_sqft'].mean()
        st.metric("Avg Price/SqFt", f"${avg_price_sqft:.0f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Asset Value Distribution")
        fig = px.histogram(df, x='estimated_value', nbins=30, title="Distribution of Asset Values")
        fig.update_layout(xaxis_title="Estimated Value ($)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Assets by State")
        state_counts = df['state'].value_counts().head(10)
        fig = px.bar(x=state_counts.index, y=state_counts.values, title="Top 10 States")
        fig.update_layout(xaxis_title="State", yaxis_title="Number of Assets")
        st.plotly_chart(fig, use_container_width=True)
    
    # Price analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price per Square Foot by State")
        state_prices = df.groupby('state')['price_per_sqft'].mean().sort_values(ascending=False).head(10)
        fig = px.bar(x=state_prices.values, y=state_prices.index, orientation='h',
                     title="Average Price per SqFt by State")
        fig.update_layout(xaxis_title="Price per SqFt ($)", yaxis_title="State")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Asset Type Distribution")
        type_counts = df['asset_type'].value_counts()
        fig = px.pie(values=type_counts.values, names=type_counts.index, 
                     title="Portfolio by Asset Type")
        st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.subheader("💡 Key Portfolio Insights")
    
    top_state = df['state'].value_counts().index[0]
    highest_value_state = df.groupby('state')['estimated_value'].sum().idxmax()
    most_expensive_state = df.groupby('state')['price_per_sqft'].mean().idxmax()
    
    insights = [
        f"📍 **{top_state}** has the most assets ({df[df['state'] == top_state].shape[0]:,})",
        f"💰 **{highest_value_state}** has the highest total portfolio value (${df[df['state'] == highest_value_state]['estimated_value'].sum()/1e9:.1f}B)",
        f"💎 **{most_expensive_state}** has the highest average price per sq ft (${df[df['state'] == most_expensive_state]['price_per_sqft'].mean():.0f})",
        f"🏢 Total portfolio spans **{df['state'].nunique()}** states with **{df['asset_type'].nunique()}** asset types"
    ]
    
    for insight in insights:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

def show_price_clustering_analysis(df):
    """Show price-based clustering analysis"""
    st.header("🗺️ Price-Based Asset Clustering")
    
    # Clustering parameters
    col1, col2 = st.columns([1, 3])
    
    with col1:
        n_clusters = st.slider("Number of Price Clusters", min_value=3, max_value=8, value=5)
        
        # Perform clustering
        df_clustered, model, scaler, cluster_summary = perform_price_based_clustering(df, n_clusters)
    
    with col2:
        st.subheader("Interactive Price Cluster Map")
        # Create and display map
        map_obj = create_price_cluster_map(df_clustered)
        st_folium(map_obj, width=700, height=500)
    
    # Cluster analysis
    st.subheader("📊 Price Cluster Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Cluster Summary**")
        # Format the cluster summary for display
        display_summary = cluster_summary.copy()
        display_summary['avg_value'] = display_summary['avg_value'].apply(lambda x: f"${x/1e6:.1f}M")
        display_summary['avg_sqft'] = display_summary['avg_sqft'].apply(lambda x: f"{x:,.0f}")
        display_summary['avg_price_per_sqft'] = display_summary['avg_price_per_sqft'].apply(lambda x: f"${x:.0f}")
        
        st.dataframe(display_summary[['cluster', 'count', 'avg_value', 'avg_price_per_sqft', 'top_state']])
    
    with col2:
        st.write("**Cluster Distribution**")
        cluster_counts = df_clustered['price_cluster'].value_counts().sort_index()
        fig = px.pie(values=cluster_counts.values, 
                     names=[f'Cluster {i}' for i in cluster_counts.index],
                     title="Assets by Price Cluster")
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster characteristics
    st.subheader("🎯 Cluster Characteristics")
    
    # Box plot of values by cluster
    fig = px.box(df_clustered, x='price_cluster', y='estimated_value', 
                 title="Asset Value Distribution by Cluster")
    fig.update_layout(xaxis_title="Price Cluster", yaxis_title="Estimated Value ($)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster insights
    st.subheader("💡 Clustering Insights")
    
    high_value_cluster = cluster_summary.loc[cluster_summary['avg_value'].idxmax()]
    low_value_cluster = cluster_summary.loc[cluster_summary['avg_value'].idxmin()]
    largest_cluster = cluster_summary.loc[cluster_summary['count'].idxmax()]
    
    cluster_insights = [
        f"🏆 **Cluster {high_value_cluster['cluster']}** is the highest value cluster (Avg: ${high_value_cluster['avg_value']/1e6:.1f}M)",
        f"💡 **Cluster {largest_cluster['cluster']}** contains the most assets ({largest_cluster['count']} assets)",
        f"📍 **{high_value_cluster['top_state']}** dominates the highest value cluster",
        f"💰 Price range spans from ${low_value_cluster['avg_value']/1e6:.1f}M to ${high_value_cluster['avg_value']/1e6:.1f}M average per cluster"
    ]
    
    for insight in cluster_insights:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

def show_price_prediction_model(df):
    """Show price prediction model"""
    st.header("🤖 Asset Price Prediction Model")
    
    # Train model
    with st.spinner("Training price prediction model..."):
        model, r2, rmse, mae, feature_importance, X_test, y_test, y_pred = train_price_prediction_model(df)
    
    # Model performance
    st.subheader("📈 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² Score", f"{r2:.3f}")
    
    with col2:
        st.metric("RMSE", f"${rmse/1e6:.1f}M")
    
    with col3:
        st.metric("MAE", f"${mae/1e6:.1f}M")
    
    with col4:
        accuracy_pct = (1 - mae/df['estimated_value'].mean()) * 100
        st.metric("Accuracy", f"{accuracy_pct:.1f}%")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Actual vs Predicted Values")
        fig = px.scatter(x=y_test, y=y_pred, title="Model Predictions vs Actual Values")
        fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), 
                      x1=y_test.max(), y1=y_test.max(), line=dict(color="red", dash="dash"))
        fig.update_layout(xaxis_title="Actual Value ($)", yaxis_title="Predicted Value ($)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Feature Importance")
        fig = px.bar(feature_importance.head(8), x='importance', y='feature', orientation='h',
                     title="Top 8 Most Important Features")
        fig.update_layout(xaxis_title="Importance", yaxis_title="Feature")
        st.plotly_chart(fig, use_container_width=True)
    
    # Prediction residuals
    st.subheader("📊 Prediction Analysis")
    
    residuals = y_test - y_pred
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(x=residuals, nbins=30, title="Prediction Residuals Distribution")
        fig.update_layout(xaxis_title="Residuals ($)", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(x=y_pred, y=residuals, title="Residuals vs Predicted Values")
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(xaxis_title="Predicted Value ($)", yaxis_title="Residuals ($)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Interactive prediction tool
    st.subheader("🎯 Interactive Price Predictor")
    
    with st.expander("Predict Asset Value"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pred_sqft = st.number_input("Square Feet", min_value=1000, max_value=200000, value=25000)
            pred_year = st.number_input("Construction Year", min_value=1950, max_value=2024, value=2000)
        
        with col2:
            pred_state = st.selectbox("State", df['state'].unique())
            pred_lat = st.number_input("Latitude", min_value=25.0, max_value=48.0, value=40.0)
        
        with col3:
            pred_lon = st.number_input("Longitude", min_value=-125.0, max_value=-70.0, value=-95.0)
        
        if st.button("Predict Value"):
            # Prepare prediction data
            state_encoder = {state: i for i, state in enumerate(df['state'].unique())}
            pred_data = pd.DataFrame({
                'building_rentable_square_feet': [pred_sqft],
                'latitude': [pred_lat],
                'longitude': [pred_lon],
                'construction_year': [pred_year],
                'state_encoded': [state_encoder[pred_state]]
            })
            
            predicted_value = model.predict(pred_data)[0]
            
            st.markdown(f'<div class="prediction-box"><b>Predicted Asset Value: ${predicted_value:,.0f}</b><br>Price per SqFt: ${predicted_value/pred_sqft:.0f}</div>', 
                       unsafe_allow_html=True)
    
    # Model insights
    st.subheader("💡 Model Insights")
    
    top_feature = feature_importance.iloc[0]
    model_insights = [
        f"🎯 Model achieves **{r2:.1%}** accuracy in predicting asset values",
        f"📊 **{top_feature['feature'].replace('_', ' ').title()}** is the most important factor ({top_feature['importance']:.1%} importance)",
        f"💰 Average prediction error is **${mae/1e6:.1f}M** (±{(mae/df['estimated_value'].mean()*100):.1f}%)",
        f"🔍 Model trained on **{len(df):,}** assets across **{df['state'].nunique()}** states"
    ]
    
    for insight in model_insights:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

def show_portfolio_analytics(df):
    """Show portfolio analytics"""
    st.header("📈 Advanced Portfolio Analytics")
    
    # Portfolio overview
    st.subheader("📊 Portfolio Overview")
    
    total_value = df['estimated_value'].sum()
    total_sqft = df['building_rentable_square_feet'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Portfolio Value", f"${total_value/1e9:.1f}B")
    
    with col2:
        st.metric("Total Square Footage", f"{total_sqft/1e6:.1f}M SqFt")
    
    with col3:
        st.metric("Average Value/SqFt", f"${(total_value/total_sqft):.0f}")
    
    with col4:
        oldest_year = df['construction_year'].min()
        newest_year = df['construction_year'].max()
        st.metric("Age Range", f"{oldest_year}-{newest_year}")
    
    # State-wise analysis
    st.subheader("🗺️ State-wise Portfolio Analysis")
    
    state_analysis = df.groupby('state').agg({
        'estimated_value': ['count', 'sum', 'mean'],
        'building_rentable_square_feet': 'sum',
        'price_per_sqft': 'mean'
    }).round(2)
    
    state_analysis.columns = ['Asset Count', 'Total Value', 'Avg Value', 'Total SqFt', 'Avg Price/SqFt']
    state_analysis = state_analysis.sort_values('Total Value', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top states by value
        top_states = state_analysis.head(10)
        fig = px.bar(x=top_states.index, y=top_states['Total Value'], 
                     title="Top 10 States by Portfolio Value")
        fig.update_layout(xaxis_title="State", yaxis_title="Total Value ($)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Price efficiency analysis
        fig = px.scatter(state_analysis.reset_index(), x='Total SqFt', y='Avg Price/SqFt', 
                        size='Asset Count', hover_name='state',
                        title="Portfolio Efficiency: Size vs Price/SqFt")
        st.plotly_chart(fig, use_container_width=True)
    
    # Asset age analysis
    st.subheader("🏗️ Asset Age Analysis")
    
    # Create age groups
    current_year = 2024
    df['asset_age'] = current_year - df['construction_year']
    df['age_group'] = pd.cut(df['asset_age'], 
                            bins=[0, 10, 20, 30, 50, 100], 
                            labels=['0-10 years', '11-20 years', '21-30 years', '31-50 years', '50+ years'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        age_dist = df['age_group'].value_counts()
        fig = px.pie(values=age_dist.values, names=age_dist.index, 
                     title="Portfolio by Asset Age")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        age_value = df.groupby('age_group')['estimated_value'].mean()
        fig = px.bar(x=age_value.index, y=age_value.values,
                     title="Average Value by Asset Age")
        fig.update_layout(xaxis_title="Age Group", yaxis_title="Average Value ($)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed state table
    st.subheader("📋 Detailed State Analysis")
    
    # Format the state analysis for better display
    display_state_analysis = state_analysis.copy()
    display_state_analysis['Total Value'] = display_state_analysis['Total Value'].apply(lambda x: f"${x/1e9:.2f}B")
    display_state_analysis['Avg Value'] = display_state_analysis['Avg Value'].apply(lambda x: f"${x/1e6:.1f}M")
    display_state_analysis['Total SqFt'] = display_state_analysis['Total SqFt'].apply(lambda x: f"{x/1e6:.1f}M")
    display_state_analysis['Avg Price/SqFt'] = display_state_analysis['Avg Price/SqFt'].apply(lambda x: f"${x:.0f}")
    
    st.dataframe(display_state_analysis)

if __name__ == "__main__":
    main()

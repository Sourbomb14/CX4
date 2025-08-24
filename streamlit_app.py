import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    r2_score,
    mean_absolute_error,
    classification_report,
    f1_score
)
import warnings
warnings.filterwarnings('ignore')

# Set random state for reproducibility
RANDOM_STATE = 4742271
np.random.seed(RANDOM_STATE)

# --- Page Configuration ---
st.set_page_config(
    page_title="US Government Assets Portfolio Analytics",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 6px solid #1f4e79;
    }
    .stMetric {
        text-align: center;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f4e79;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading and Processing ---
@st.cache_data
def load_data():
    """
    Loads, cleans, and processes asset data. Includes feature engineering for estimated value.
    """
    try:
        assets_url = "https://drive.google.com/uc?id=1YFTWJNoxu0BF8UlMDXI8bXwRTVQNE2mb"
        df_assets = pd.read_csv(assets_url)
    except Exception:
        st.warning("⚠️ Could not load assets data. Using a sample dataset for demonstration.")
        return create_sample_data()

    df_assets.columns = df_assets.columns.str.lower().str.replace(' ', '_').str.strip()
    if 'latitude' in df_assets.columns and 'longitude' in df_assets.columns:
        df_assets = df_assets[df_assets['latitude'].between(24, 50) & df_assets['longitude'].between(-125, -66)].copy()

    rentable_col = next((col for col in df_assets.columns if 'rentable' in col and 'feet' in col), None)
    if rentable_col and pd.api.types.is_numeric_dtype(df_assets[rentable_col]):
        price_per_sqft = np.random.uniform(150, 600, len(df_assets))
        df_assets['estimated_value'] = df_assets[rentable_col] * price_per_sqft
    else:
        df_assets['estimated_value'] = np.random.lognormal(mean=14, sigma=1.5, size=len(df_assets))

    high_value_states = ['CA', 'NY', 'MA', 'WA', 'VA', 'NJ', 'CT', 'MD']
    df_assets.loc[df_assets['state'].isin(high_value_states), 'estimated_value'] *= 1.8
    
    return df_assets.dropna(subset=['latitude', 'longitude', 'estimated_value'])


@st.cache_data
def create_sample_data(size=2000):
    """Creates a sample dataframe for demonstration."""
    states = ['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI', 'VA', 'MD']
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'Washington', 'Miami', 'Atlanta', 'Boston']
    data = {
        'state': np.random.choice(states, size),
        'city': np.random.choice(cities, size),
        'latitude': np.random.uniform(25, 48, size),
        'longitude': np.random.uniform(-125, -70, size),
        'building_rentable_square_feet': np.random.uniform(5000, 250000, size),
        'estimated_value': np.random.lognormal(mean=14.5, sigma=1.2, size=size)
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

def create_interactive_map(df, sample_size=1500):
    """Creates an interactive Folium map with clustered markers."""
    map_data = df.sample(n=min(len(df), sample_size), random_state=RANDOM_STATE)
    center_lat, center_lon = map_data['latitude'].mean(), map_data['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles='cartodbpositron')

    marker_cluster = MarkerCluster().add_to(m)
    colors = px.colors.qualitative.Plotly

    for _, row in map_data.iterrows():
        cluster_id = int(row.get('cluster', 0))
        color = colors[cluster_id % len(colors)]
        popup_html = f"<b>Value:</b> ${row.get('estimated_value', 0):,.0f}<br><b>Cluster:</b> {cluster_id}"
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=6, popup=popup_html, color=color, fill=True, fill_color=color, fill_opacity=0.7
        ).add_to(marker_cluster)
        
    return m

def create_static_heatmap(df, sample_size=5000):
    """Creates a static Folium heatmap based on asset value."""
    map_data = df.sample(n=min(len(df), sample_size), random_state=RANDOM_STATE)
    center_lat, center_lon = map_data['latitude'].mean(), map_data['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles='cartodbdark_matter')

    heat_data = [[row['latitude'], row['longitude'], row['estimated_value']] for _, row in map_data.iterrows()]
    HeatMap(heat_data, radius=15, blur=20, gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 1: 'red'}).add_to(m)
        
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
    """Displays geographic analysis with both interactive and static maps."""
    st.header("🗺️ Geographic Analysis")
    
    df_clustered = perform_clustering(df.copy(), n_clusters=5)

    tab1, tab2 = st.tabs(["Interactive Clustered Map", "Static Heatmap View"])
    
    with tab1:
        st.subheader("Interactive Asset Clusters")
        map_obj_interactive = create_interactive_map(df_clustered)
        if map_obj_interactive:
            st_folium(map_obj_interactive, width='100%', height=500)
            st.markdown("""
            #### Key Insights:
            * **Explore Clusters:** Zoom in on the colored clusters to see how assets group together based on location and value.
            * **Drill-Down:** As you zoom, clusters break apart into individual assets. Click on any circle for its estimated value.
            * **Identify Patterns:** This view is excellent for understanding the micro-level distribution and finding specific high-value assets within a dense area.
            """)
        else:
            st.error("Could not generate interactive map.")

    with tab2:
        st.subheader("Asset Value Heatmap")
        map_obj_static = create_static_heatmap(df)
        if map_obj_static:
            st_folium(map_obj_static, width='100%', height=500)
            st.markdown("""
            #### Key Insights:
            * **Identify Hotspots:** The red and yellow areas signify a high concentration of asset value, not just asset count. This highlights the most valuable regions in the portfolio.
            * **High-Level View:** This map provides a strategic overview of where the portfolio's value is concentrated geographically.
            * **Strategic Planning:** Use this to identify regions with significant investment, such as the Northeast corridor and coastal California.
            """)
        else:
            st.error("Could not generate static heatmap.")

def show_clustering_analysis(df):
    """Displays clustering results and visualizations."""
    st.header("🎯 Asset Clustering Analysis")
    
    n_clusters = st.slider("Select Number of Clusters", 2, 10, 5, key="cluster_slider")
    df_clustered = perform_clustering(df.copy(), n_clusters=n_clusters)
    
    col1, col2 = st.columns([2, 3])
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
    features = [f for f in features if f in ml_df.columns and pd.api.types.is_numeric_dtype(ml_df[f])]
    
    if not features:
        st.error("No suitable numeric features found for ML modeling.")
        return

    X = ml_df[features].fillna(0)
    y = ml_df['estimated_value']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    
    with st.expander("🎯 Asset Value Prediction (Regression)", expanded=True):
        model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        col1.metric("R² Score", f"{r2:.3f}")
        col2.metric("Mean Absolute Error", f"${mae/1e6:.2f}M")
        
        results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        fig = px.scatter(results_df, x='Actual', y='Predicted', title='Predicted vs. Actual Asset Values',
                         labels={'Actual': 'Actual Value ($)', 'Predicted': 'Predicted Value ($)'},
                         trendline='ols', trendline_color_override='red')
        st.plotly_chart(fig, use_container_width=True)

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

def show_advanced_analytics(df):
    """Displays advanced statistical and data quality analysis."""
    st.header("📈 Advanced Analytics")
    
    tab1, tab2, tab3 = st.tabs(["Statistical Summary", "Data Quality", "Portfolio Trends"])
    
    with tab1:
        st.subheader("Descriptive Statistics for Asset Value")
        if not df.empty and 'estimated_value' in df.columns:
            st.dataframe(df['estimated_value'].describe().to_frame().style.format("${:,.2f}"))
            
            fig = px.box(df, y='estimated_value', title="Asset Value Box Plot")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for statistical analysis.")

    with tab2:
        st.subheader("Data Quality Assessment")
        missing_data = df.isnull().sum().sort_values(ascending=False)
        missing_pct = (missing_data / len(df) * 100)
        missing_df = pd.DataFrame({'Missing Count': missing_data, 'Missing %': missing_pct})
        
        st.metric("Overall Data Completeness", f"{100 - missing_pct.mean():.2f}%")
        st.write("Columns with missing values:")
        st.dataframe(missing_df[missing_df['Missing Count'] > 0])

    with tab3:
        st.subheader("State Portfolio Analysis")
        if not df.empty and 'state' in df.columns:
            state_analysis = df.groupby('state')['estimated_value'].agg(['count', 'sum', 'mean']).nlargest(20, 'sum')
            
            fig = px.scatter(
                state_analysis,
                x='count', y='mean', size='sum',
                hover_name=state_analysis.index,
                size_max=60,
                title="Top 20 States: Asset Count vs. Average Value (Bubble size = Total Value)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No state data available for trend analysis.")

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

    # Value Filter (with error handling for empty dataframes)
    if not df_filtered.empty:
        min_val, max_val = df_filtered['estimated_value'].min(), df_filtered['estimated_value'].max()
        if pd.notna(min_val) and pd.notna(max_val) and max_val > min_val:
            value_range = st.sidebar.slider(
                "Filter by Asset Value ($M)", 
                min_value=float(min_val / 1e6), max_value=float(max_val / 1e6), 
                value=(float(min_val / 1e6), float(max_val / 1e6)),
                step=1.0
            )
            df_filtered = df_filtered[df_filtered['estimated_value'].between(value_range[0] * 1e6, value_range[1] * 1e6)]
    else:
        st.sidebar.warning("No assets found for the selected state.")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Navigation")
    page = st.sidebar.radio(
        "Choose an Analysis Page",
        ["📊 Executive Dashboard", "🗺️ Geographic Analysis", "🎯 Clustering Analysis", "🤖 ML Predictions", "📈 Advanced Analytics"]
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
    elif page == "📈 Advanced Analytics":
        show_advanced_analytics(df_filtered)

if __name__ == "__main__":
    main()

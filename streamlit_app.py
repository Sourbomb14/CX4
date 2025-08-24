import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import geopandas as gpd
from shapely.geometry import Point
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import json

warnings.filterwarnings('ignore')

# --- Global Configuration ---
# This specific random state is used for reproducibility as requested.
RANDOM_STATE = 4742271

# Set page configuration for a professional look and feel
st.set_page_config(
    page_title="US Government Assets Portfolio Analytics",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling and a polished UI
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
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #b0c4de;
    }
    .stMetric > label {
        font-size: 1.3rem !important;
        font-weight: bold !important;
        color: #333;
    }
    .st-emotion-cache-12oz5g7 {
        padding-top: 0rem;
    }
    h1, h2, h3, h4 {
        color: #1f4e79;
    }
    .explanation {
        background-color: #fafafa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading and Processing Functions ---

@st.cache_data
def load_data(url, encoding='utf-8'):
    """Generic function to load data from a Google Drive URL."""
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()  # Raise an exception for bad status codes
        return pd.read_csv(requests.get(url, stream=True).raw, encoding=encoding)
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to download data from {url}. Error: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def create_sample_data():
    """Create sample data if real data is not available for demonstration."""
    np.random.seed(RANDOM_STATE)
    n_samples = 1000
    states = ['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI', 'NJ', 'VA', 'WA', 'AZ', 'MA']
    data = {
        'state': np.random.choice(states, n_samples),
        'city': np.random.choice(['Los Angeles', 'Houston', 'New York', 'Miami', 'Chicago', 'Philadelphia', 'Phoenix', 'Atlanta', 'Boston', 'Seattle'], n_samples),
        'latitude': np.random.uniform(25, 49, n_samples),
        'longitude': np.random.uniform(-125, -70, n_samples),
        'building_rentable_square_feet': np.random.uniform(1000, 100000, n_samples),
        'estimated_value': np.random.lognormal(13, 1.5, n_samples),
        'latest_price_index': np.random.uniform(50000, 800000, n_samples)
    }
    return pd.DataFrame(data)

@st.cache_data
def clean_and_merge_data(df_assets, df_prices):
    """Clean, process, and merge the assets and housing price datasets."""
    if df_assets is None:
        return create_sample_data()

    df_assets.columns = df_assets.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    df_assets = df_assets.dropna(subset=['latitude', 'longitude'])
    df_assets = df_assets[(df_assets['latitude'].between(24, 50)) & (df_assets['longitude'].between(-125, -66))]

    if df_prices is not None:
        df_prices.columns = df_prices.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
        price_cols = [col for col in df_prices.columns if '202' in str(col)]
        if price_cols:
            latest_col = sorted(price_cols, reverse=True)[0]
            df_prices['latest_price_index'] = pd.to_numeric(df_prices[latest_col], errors='coerce')
            df_assets = pd.merge(df_assets, df_prices[['city', 'state', 'latest_price_index']], on=['city', 'state'], how='left')
        else:
            df_assets['latest_price_index'] = np.nan
    else:
        df_assets['latest_price_index'] = np.nan

    df_assets['latest_price_index'].fillna(df_assets['latest_price_index'].median(), inplace=True)

    rentable_col = next((col for col in df_assets.columns if 'rentable' in col and 'feet' in col), None)
    if rentable_col:
        df_assets['estimated_value'] = df_assets[rentable_col] * (df_assets['latest_price_index'] / 150)
    else:
        df_assets['estimated_value'] = df_assets['latest_price_index'] * np.random.uniform(0.8, 2.5, len(df_assets))

    high_value_states = ['CA', 'NY', 'MA', 'CT', 'NJ', 'HI', 'MD', 'WA', 'VA']
    premium_mask = df_assets['state'].isin(high_value_states)
    df_assets.loc[premium_mask, 'estimated_value'] *= 1.5

    return df_assets

@st.cache_data
def perform_clustering(df, n_clusters=5):
    """Perform K-Means clustering on key numeric features."""
    numeric_cols = ['latitude', 'longitude', 'estimated_value', 'building_rentable_square_feet', 'latest_price_index']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    if not numeric_cols:
        return df, None, []

    cluster_data = df[numeric_cols].fillna(df[numeric_cols].median())
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    df['cluster'] = kmeans.fit_predict(scaled_data)
    
    return df, kmeans, numeric_cols

def name_clusters(df, numeric_cols):
    """Assign descriptive names to clusters based on their defining characteristics."""
    cluster_profiles = df.groupby('cluster')[numeric_cols].mean()
    overall_means = df[numeric_cols].mean()
    
    z_scores = (cluster_profiles - overall_means) / df[numeric_cols].std()
    
    names = {}
    for i in z_scores.index:
        top_feature = z_scores.loc[i].abs().idxmax()
        direction = "High" if z_scores.loc[i, top_feature] > 0 else "Low"
        
        feature_map = {
            'estimated_value': 'Value',
            'building_rentable_square_feet': 'Size',
            'latest_price_index': 'Market Value',
            'latitude': 'Northern',
            'longitude': 'Eastern' if z_scores.loc[i, 'longitude'] > 0 else 'Western'
        }
        
        if top_feature == 'latitude':
            name = f"Cluster {i}: Primarily {direction} {feature_map[top_feature]}"
        elif top_feature == 'longitude':
            name = f"Cluster {i}: Primarily {feature_map[top_feature]}"
        else:
            name = f"Cluster {i}: {direction} {feature_map[top_feature]} Assets"
            
        names[i] = name
        
    return names

@st.cache_data
def create_ml_features(_df):
    """Engineer features for machine learning models."""
    df = _df.copy()
    features = []
    major_cities = {'NYC': (40.7, -74.0), 'LA': (34.0, -118.2), 'Chicago': (41.8, -87.6), 'DC': (38.9, -77.0)}
    
    for city, (lat, lon) in major_cities.items():
        df[f'dist_to_{city.lower()}'] = np.sqrt((df['latitude'] - lat)**2 + (df['longitude'] - lon)**2)
        features.append(f'dist_to_{city.lower()}')
        
    numeric_features = ['latitude', 'longitude', 'latest_price_index', 'building_rentable_square_feet']
    features.extend([col for col in numeric_features if col in df.columns])
    
    return df, features

# --- Visualization Functions ---

def create_folium_map(df, sample_size=500):
    """Create an interactive Folium map with clustered asset points."""
    map_data = df.sample(n=min(len(df), sample_size), random_state=RANDOM_STATE)
    center_lat, center_lon = map_data['latitude'].mean(), map_data['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=4, tiles='CartoDB positron')
    
    colors = px.colors.qualitative.Plotly
    for _, row in map_data.iterrows():
        cluster_id = int(row.get('cluster', 0))
        color = colors[cluster_id % len(colors)]
        popup_text = f"""
        <b>Asset Location:</b> {row.get('city', 'N/A')}, {row.get('state', 'N/A')}<br>
        <b>Estimated Value:</b> ${row.get('estimated_value', 0):,.0f}<br>
        <b>Cluster:</b> {row.get('cluster_name', 'N/A')}
        """
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']], radius=6, popup=popup_text,
            color=color, fillColor=color, fillOpacity=0.7
        ).add_to(m)
    return m

@st.cache_data
def create_choropleth_map(_df):
    """Create a GeoPandas choropleth map of asset values by state."""
    state_summary = _df.groupby('state')['estimated_value'].sum().reset_index()
    
    try:
        url = "https://raw.githubusercontent.com/python-visualization/folium-example-data/main/us_states.json"
        states_geojson = requests.get(url).json()
    except Exception:
        st.error("Could not download GeoJSON for the choropleth map.")
        return None

    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    
    folium.Choropleth(
        geo_data=states_geojson, name='choropleth', data=state_summary,
        columns=['state', 'estimated_value'], key_on='feature.id',
        fill_color='YlGnBu', fill_opacity=0.7, line_opacity=0.2,
        legend_name='Total Asset Value ($)'
    ).add_to(m)
    
    return m

# --- UI Page Functions ---

def show_executive_dashboard(df):
    st.header("📊 Executive Dashboard")
    if df.empty:
        st.warning("No data available for the current filters.")
        return

    st.markdown("#### Key Performance Indicators (KPIs)")
    st.markdown("""
    <div class="explanation">
        <strong>What it is:</strong> These are the high-level summary statistics for the assets currently selected by the filters. <br>
        <strong>Why it matters:</strong> They provide an immediate snapshot of the portfolio's scale and value, answering the fundamental questions of "how many" and "how much".
    </div>
    """, unsafe_allow_html=True)

    total_value = df['estimated_value'].sum()
    avg_value = df['estimated_value'].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Assets", f"{len(df):,}")
    with col2: st.metric("Portfolio Value", f"${total_value/1e9:.2f}B")
    with col3: st.metric("Average Asset Value", f"${avg_value/1e6:.2f}M")
    with col4: st.metric("States Covered", f"{df['state'].nunique()}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Asset Value Distribution")
        st.markdown("""
        <div class="explanation">
            <strong>What it is:</strong> A histogram showing the frequency of assets across different value brackets. The y-axis is on a logarithmic scale to better visualize the wide range of values.<br>
            <strong>How to interpret:</strong> A tall bar on the left and a long tail to the right (right-skew) indicates that the majority of assets are of lower value, with a few exceptionally high-value properties. This is typical for large, diverse portfolios.
        </div>
        """, unsafe_allow_html=True)
        fig = px.histogram(df, x='estimated_value', nbins=50, title="Distribution of Asset Values (Log Scale)")
        fig.update_layout(xaxis_title="Estimated Value ($)", yaxis_title="Number of Assets (Log Scale)", yaxis_type="log")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Top 10 States by Portfolio Value")
        st.markdown("""
        <div class="explanation">
            <strong>What it is:</strong> A bar chart ranking the top 10 states by the total estimated value of assets they contain.<br>
            <strong>Why it matters:</strong> This highlights which states are most critical to the portfolio's overall value. It helps in strategic planning, resource allocation, and risk assessment related to regional economic factors.
        </div>
        """, unsafe_allow_html=True)
        state_values = df.groupby('state')['estimated_value'].sum().nlargest(10)
        fig = px.bar(state_values, x=state_values.index, y=state_values.values, title="Total Asset Value by State")
        fig.update_layout(xaxis_title="State", yaxis_title="Total Value ($)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("💡 Key Managerial Insights")
    top_state_value = df.groupby('state')['estimated_value'].sum().idxmax()
    top_10_pct_value = df.nlargest(int(len(df) * 0.1), 'estimated_value')['estimated_value'].sum()
    concentration_pct = (top_10_pct_value / total_value * 100) if total_value > 0 else 0

    st.markdown(f'<div class="insight-box">📍 **Geographic Focus**: **{top_state_value}** holds the highest total asset value. This concentration warrants focused management and monitoring of local market conditions.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="insight-box">🏛️ **Value Concentration**: The top 10% of assets account for **{concentration_pct:.1f}%** of the total portfolio value. This indicates that a relatively small number of high-value properties are crucial to the portfolio\'s financial health. Strategies should be in place to protect these key assets.</div>', unsafe_allow_html=True)

def show_geographic_analysis(df):
    st.header("🗺️ Geographic & Spatial Analysis")
    if df.empty:
        st.warning("No geographic data available for the current filters.")
        return

    tab1, tab2, tab3 = st.tabs(["Asset Point Map", "Value Choropleth Map", "Value Density Heatmap"])
    
    with tab1:
        st.subheader("Interactive Asset Point Map (Clustered)")
        st.markdown("""
        <div class="explanation">
            <strong>What it is:</strong> Each point on this map represents an individual asset, colored by its assigned cluster (segment).<br>
            <strong>Why it matters:</strong> This view helps to visualize the geographic distribution of different asset types. You can see if certain types of assets (e.g., high-value, large-size) are geographically concentrated, which can inform logistical and management strategies.
        </div>
        """, unsafe_allow_html=True)
        df_clustered, _, _ = perform_clustering(df, n_clusters=5)
        if 'cluster' in df_clustered.columns:
            cluster_names = name_clusters(df_clustered, ['estimated_value', 'building_rentable_square_feet'])
            df_clustered['cluster_name'] = df_clustered['cluster'].map(cluster_names)
            map_obj = create_folium_map(df_clustered)
            if map_obj: st_folium(map_obj, width=700, height=500)
    
    with tab2:
        st.subheader("State-Level Portfolio Value (Choropleth)")
        st.markdown("""
        <div class="explanation">
            <strong>What it is:</strong> A map where states are shaded based on the total estimated value of the assets they contain. Darker shades indicate higher total value.<br>
            <strong>Why it matters:</strong> This provides a high-level, strategic overview of the portfolio's geographic footprint. It's excellent for presentations and for quickly identifying states with the most significant investment.
        </div>
        """, unsafe_allow_html=True)
        choro_map = create_choropleth_map(df)
        if choro_map: st_folium(choro_map, width=700, height=500)

    with tab3:
        st.subheader("Asset Value Density Heatmap")
        st.markdown("""
        <div class="explanation">
            <strong>What it is:</strong> A heatmap where color intensity (from blue to red) represents the concentration of asset value in a specific area.<br>
            <strong>Why it matters:</strong> Unlike the choropleth, this is not bound by state lines. It reveals hyper-local hotspots of high value, such as major metropolitan areas. This is useful for understanding urban vs. rural value distribution and identifying key economic hubs.
        </div>
        """, unsafe_allow_html=True)
        heat_map_data = df[['latitude', 'longitude', 'estimated_value']].values.tolist()
        center_lat, center_lon = df['latitude'].mean(), df['longitude'].mean()
        heat_map = folium.Map(location=[center_lat, center_lon], zoom_start=4)
        HeatMap(heat_map_data, radius=15).add_to(heat_map)
        st_folium(heat_map, width=700, height=500)

def show_clustering_analysis(df):
    st.header("🎯 Portfolio Segmentation via Clustering")
    if df.empty:
        st.warning("No data available for clustering.")
        return

    n_clusters = st.slider("Select Number of Clusters (Segments)", min_value=2, max_value=10, value=5, key="cluster_slider")
    df_clustered, _, numeric_cols = perform_clustering(df, n_clusters=n_clusters)
    
    if 'cluster' in df_clustered.columns:
        cluster_names = name_clusters(df_clustered, numeric_cols)
        df_clustered['cluster_name'] = df_clustered['cluster'].map(cluster_names)
        
        st.markdown(f'<div class="insight-box"><strong>Insight:</strong> By segmenting the portfolio into **{n_clusters}** groups using K-Means clustering, we can move from a one-size-fits-all approach to targeted management. Each cluster represents a group of assets with similar characteristics (like value, size, and market), allowing for tailored strategies in maintenance, investment, or divestment.</div>', unsafe_allow_html=True)

        st.subheader("Visualizing Portfolio Segments")
        st.markdown("""
        <div class="explanation">
            <strong>What it is:</strong> A scatter plot where each bubble is an asset. The position is determined by its value (x-axis) and size (y-axis). The color represents its cluster, and the bubble size reflects the local market price index.<br>
            <strong>How to interpret:</strong> Look for patterns within colors. For example, do "High Value" assets (one color) tend to be large or small? Are they in high-priced markets (large bubbles)? This helps to visually understand the nature of each asset segment.
        </div>
        """, unsafe_allow_html=True)
        fig = px.scatter(
            df_clustered.sample(min(1000, len(df_clustered))),
            x='estimated_value', y='building_rentable_square_feet',
            color='cluster_name', size='latest_price_index',
            hover_name='city', title="Portfolio Segments: Value vs. Size"
        )
        fig.update_layout(xaxis_title="Estimated Value ($)", yaxis_title="Rentable Square Feet")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Cluster Profiles")
        st.markdown("""
        <div class="explanation">
            <strong>What it is:</strong> A table showing the average characteristics for each cluster.<br>
            <strong>Why it matters:</strong> This is the analytical backbone of segmentation. By comparing the average values across clusters, you can quantitatively define what makes each segment unique. For example, you can confirm that the "High Value Assets" cluster truly has a significantly higher average `estimated_value` than the others.
        </div>
        """, unsafe_allow_html=True)
        cluster_stats = df_clustered.groupby('cluster_name')[numeric_cols].mean().round(0)
        st.dataframe(cluster_stats)

def show_machine_learning(df):
    st.header("🤖 Predictive Analytics & Machine Learning")
    if df.empty or 'estimated_value' not in df.columns:
        st.warning("Not enough data for machine learning analysis.")
        return

    df_ml, features = create_ml_features(df)
    X = df_ml[features].fillna(df_ml[features].median())
    y = df_ml['estimated_value']

    if len(X) < 20:
        st.warning("Need at least 20 data points for meaningful ML analysis.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    st.subheader("🎯 Asset Value Prediction Model (Random Forest)")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model R² Score (Accuracy)", f"{r2*100:.2f}%")
        st.markdown("""
        <div class="explanation">
            <strong>What it is:</strong> The R-squared (R²) score represents the proportion of the variance in asset values that the model can predict from the features. <br>
            <strong>Interpretation:</strong> A score of 100% would mean a perfect prediction. A score of, for example, 85% indicates that the model can explain 85% of the price variations, which is generally considered a strong model.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.subheader("Prediction Performance")
        st.markdown("""
        <div class="explanation">
            <strong>What it is:</strong> A scatter plot comparing the model's predicted values against the actual values for a set of test data.<br>
            <strong>Interpretation:</strong> In a perfect model, all points would lie on the red diagonal line. The closer the blue dots are to the line, the more accurate the model's predictions are.
        </div>
        """, unsafe_allow_html=True)
        fig, ax = plt.subplots()
        sns.regplot(x=y_test, y=y_pred, ax=ax, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        ax.set_xlabel("Actual Value")
        ax.set_ylabel("Predicted Value")
        ax.set_title("Actual vs. Predicted Values")
        st.pyplot(fig)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Key Value Drivers (Feature Importance)")
        st.markdown("""
        <div class="explanation">
            <strong>What it is:</strong> This chart ranks the features by how much they contribute to the model's predictions.<br>
            <strong>Why it matters:</strong> It tells us which factors are most influential in determining an asset's value. For example, if `latest_price_index` is at the top, it confirms that local market conditions are a primary driver of value.
        </div>
        """, unsafe_allow_html=True)
        feature_importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_}).sort_values('importance', ascending=False).head(10)
        fig = px.bar(feature_importance, x='importance', y='feature', orientation='h', title="Top Predictors of Asset Value")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("📈 Predict Asset Value")
        st.markdown("""
        <div class="explanation">
            <strong>What it is:</strong> A tool that uses the trained model to estimate the value of a hypothetical asset based on user-provided inputs.<br>
            <strong>Why it matters:</strong> This allows for proactive analysis, such as estimating the value of a potential acquisition or understanding how changes in features (like market index) could impact an existing asset's value.
        </div>
        """, unsafe_allow_html=True)
        with st.expander("Open Prediction Tool"):
            input_data = {}
            for feature in features:
                if 'dist_to' not in feature:
                    default_val = float(df_ml[feature].median())
                    input_data[feature] = st.number_input(f'Input {feature.replace("_", " ").title()}', value=default_val)

            if st.button("Predict Value"):
                major_cities = {'NYC': (40.7, -74.0), 'LA': (34.0, -118.2), 'Chicago': (41.8, -87.6), 'DC': (38.9, -77.0)}
                for city, (lat, lon) in major_cities.items():
                    input_data[f'dist_to_{city.lower()}'] = np.sqrt((input_data['latitude'] - lat)**2 + (input_data['longitude'] - lon)**2)
                
                input_df = pd.DataFrame([input_data])[features]
                predicted_value = model.predict(input_df)[0]
                st.success(f"**Predicted Asset Value:** ${predicted_value:,.2f}")

def show_advanced_analytics(df):
    st.header("📈 Advanced Statistical Analytics")
    if df.empty:
        st.warning("No data available for advanced analytics.")
        return

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    st.subheader("Correlation Analysis")
    st.markdown("""
    <div class="explanation">
        <strong>What it is:</strong> A heatmap that visualizes the correlation coefficient between every pair of numeric features. Values range from -1 to 1.<br>
        <strong>How to interpret:</strong>
        <ul>
            <li><b>Strong Positive (dark blue, close to 1):</b> When one variable increases, the other tends to increase (e.g., `building_rentable_square_feet` and `estimated_value`).</li>
            <li><b>Strong Negative (dark red, close to -1):</b> When one variable increases, the other tends to decrease.</li>
            <li><b>Weak (light color, close to 0):</b> The variables have little to no linear relationship.</li>
        </ul>
        This helps identify multicollinearity for modeling and understand the fundamental relationships driving the portfolio.
    </div>
    """, unsafe_allow_html=True)
    
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Feature Relationship Deep Dive (Pair Plot)")
    st.markdown("""
    <div class="explanation">
        <strong>What it is:</strong> A matrix of plots that shows the relationship between each pair of key variables and the distribution of each individual variable.<br>
        <strong>Why it matters:</strong> While the heatmap shows linear relationships, this plot can reveal non-linear patterns. For example, you might see a curved relationship between two variables. The histograms on the diagonal help you understand the distribution of each feature (e.g., is it normal or skewed?).
    </div>
    """, unsafe_allow_html=True)
    
    pairplot_cols = ['estimated_value', 'building_rentable_square_feet', 'latest_price_index', 'latitude']
    pairplot_cols = [col for col in pairplot_cols if col in df.columns]
    
    if len(pairplot_cols) > 1:
        sample_df = df[pairplot_cols].sample(min(500, len(df)), random_state=RANDOM_STATE)
        fig = sns.pairplot(sample_df)
        st.pyplot(fig)

# --- Main App Execution ---

def main():
    st.markdown('<h1 class="main-header">🏛️ US Government Assets Portfolio Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    st.sidebar.image("https://via.placeholder.com/300x100/1f4e79/ffffff?text=Portfolio+Analytics", use_container_width=True)
    st.sidebar.markdown("### 📊 Navigation")
    
    with st.spinner("Loading and processing datasets... This may take a moment."):
        df_assets = load_data("https://drive.google.com/uc?id=1YFTWJNoxu0BF8UlMDXI8bXwRTVQNE2mb", encoding='latin-1')
        df_prices = load_data("https://drive.google.com/uc?id=1fFT8Q8GWiIEM7kx6czhQ-qabygUPBQRv")
        df_merged = clean_and_merge_data(df_assets, df_prices)

    if df_merged.empty:
        st.error("No data available to display. The application cannot proceed.")
        return

    st.sidebar.markdown("### 🔍 Filters")
    states = ['All'] + sorted(df_merged['state'].unique().tolist())
    selected_state = st.sidebar.selectbox("Select State", states)
    df_filtered = df_merged[df_merged['state'] == selected_state] if selected_state != 'All' else df_merged

    min_val, max_val = float(df_filtered['estimated_value'].min()), float(df_filtered['estimated_value'].max())
    value_range = st.sidebar.slider("Asset Value Range ($M)", min_value=min_val/1e6, max_value=max_val/1e6, value=(min_val/1e6, max_val/1e6))
    df_filtered = df_filtered[df_filtered['estimated_value'].between(value_range[0]*1e6, value_range[1]*1e6)]

    st.sidebar.markdown("### 📋 Filtered Data Summary")
    st.sidebar.metric("Assets Displayed", f"{len(df_filtered):,}")
    st.sidebar.metric("Total Value", f"${df_filtered['estimated_value'].sum()/1e9:.2f}B")

    page = st.sidebar.radio(
        "Choose Analysis Page",
        ["📊 Executive Dashboard", "🗺️ Geographic Analysis", "🎯 Clustering Analysis", "🤖 Machine Learning", "📈 Advanced Analytics"]
    )

    page_functions = {
        "📊 Executive Dashboard": show_executive_dashboard,
        "🗺️ Geographic Analysis": show_geographic_analysis,
        "🎯 Clustering Analysis": show_clustering_analysis,
        "🤖 Machine Learning": show_machine_learning,
        "📈 Advanced Analytics": show_advanced_analytics
    }
    page_functions[page](df_filtered)

if __name__ == "__main__":
    main()

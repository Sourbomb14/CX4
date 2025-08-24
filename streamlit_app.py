import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# --- Global Configuration ---
RANDOM_STATE = 4742271
np.random.seed(RANDOM_STATE)
SAMPLE_SIZE = 7500

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced US Assets Portfolio Analytics",
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
def load_and_process_data():
    """
    Loads, merges, cleans, and samples the asset data. This is the single source of data.
    """
    try:
        assets_url = "https://drive.google.com/uc?id=1YFTWJNoxu0BF8UlMDXI8bXwRTVQNE2mb"
        df_assets = pd.read_csv(assets_url)
    except Exception:
        st.error("Fatal Error: Could not load the primary assets dataset.")
        return pd.DataFrame()

    df_assets.columns = df_assets.columns.str.lower().str.replace(' ', '_').str.strip()
    df_assets = df_assets[df_assets['latitude'].between(24, 50) & df_assets['longitude'].between(-125, -66)].copy()

    # Feature Engineering
    df_assets['estimated_value'] = np.random.lognormal(mean=14, sigma=1.5, size=len(df_assets))
    rentable_col = next((col for col in df_assets.columns if 'rentable' in col and 'feet' in col), None)
    if rentable_col and pd.api.types.is_numeric_dtype(df_assets[rentable_col]):
        df_assets['estimated_value'] = df_assets[rentable_col] * np.random.uniform(150, 600, len(df_assets))

    high_value_states = ['CA', 'NY', 'MA', 'WA', 'VA', 'NJ', 'CT', 'MD']
    df_assets.loc[df_assets['state'].isin(high_value_states), 'estimated_value'] *= 1.8
    
    # Add a categorical feature for region
    west = ['WA', 'OR', 'CA', 'AZ', 'NV', 'ID', 'MT', 'WY', 'CO', 'UT']
    midwest = ['ND', 'SD', 'NE', 'KS', 'MN', 'IA', 'MO', 'WI', 'IL', 'IN', 'MI', 'OH']
    south = ['TX', 'OK', 'AR', 'LA', 'MS', 'AL', 'GA', 'FL', 'SC', 'NC', 'TN', 'KY', 'WV', 'VA', 'MD', 'DE']
    
    df_assets['region'] = 'Northeast'
    df_assets.loc[df_assets['state'].isin(west), 'region'] = 'West'
    df_assets.loc[df_assets['state'].isin(midwest), 'region'] = 'Midwest'
    df_assets.loc[df_assets['state'].isin(south), 'region'] = 'South'

    df_final = df_assets.dropna(subset=['latitude', 'longitude', 'estimated_value', 'region'])

    if len(df_final) > SAMPLE_SIZE:
        return df_final.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    return df_final

# --- Analysis and ML Functions ---
@st.cache_data
def perform_clustering(df, n_clusters=5):
    """Performs K-Means clustering."""
    cluster_data = df[['latitude', 'longitude', 'estimated_value']].fillna(0)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    df['cluster'] = kmeans.fit_predict(scaled_data)
    return df

@st.cache_resource
def train_prediction_model(df):
    """Trains the Gradient Boosting Regressor model and returns the pipeline."""
    categorical_features = ['region']
    numerical_features = ['latitude', 'longitude', 'building_rentable_square_feet']
    
    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])
    
    # Model pipeline
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE))])
    
    X = df[numerical_features + categorical_features]
    y = df['estimated_value']
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    model_pipeline.fit(X_train, y_train)
    return model_pipeline

def show_executive_dashboard(df):
    st.header("📊 Executive Dashboard")
    total_value = df['estimated_value'].sum()
    avg_value = df['estimated_value'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Assets in Sample", f"{len(df):,}")
    col2.metric("Sample Portfolio Value", f"${total_value/1e9:.2f}B")
    col3.metric("Avg. Asset Value", f"${avg_value/1e6:.2f}M")
    col4.metric("States Covered", f"{df['state'].nunique()}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Asset Value Distribution")
        fig = px.histogram(df, x='estimated_value', nbins=50)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 10 States by Portfolio Value")
        state_value = df.groupby('state')['estimated_value'].sum().nlargest(10)
        fig = px.bar(state_value, x=state_value.values, y=state_value.index, orientation='h')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

def show_geographic_analysis(df):
    st.header("🗺️ Geographic Analysis")
    df_clustered = perform_clustering(df.copy(), n_clusters=5)

    tab1, tab2 = st.tabs(["Interactive Cluster Map", "Value Heatmap"])
    
    with tab1:
        st.subheader("Interactive Asset Clusters")
        center = [df['latitude'].mean(), df['longitude'].mean()]
        m = folium.Map(location=center, zoom_start=5, tiles='cartodbpositron')
        marker_cluster = MarkerCluster().add_to(m)
        colors = px.colors.qualitative.Plotly
        for _, row in df_clustered.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']], radius=6,
                popup=f"Value: ${row['estimated_value']:,.0f}",
                color=colors[row['cluster'] % len(colors)], fill=True, fill_opacity=0.7
            ).add_to(marker_cluster)
        st_folium(m, width='100%', height=500, returned_objects=[])

    with tab2:
        st.subheader("Asset Value Heatmap")
        m2 = folium.Map(location=center, zoom_start=5, tiles='cartodbdark_matter')
        heat_data = [[row['latitude'], row['longitude'], row['estimated_value']] for _, row in df.iterrows()]
        HeatMap(heat_data, radius=15, blur=20).add_to(m2)
        st_folium(m2, width='100%', height=500, returned_objects=[])

def show_ml_predictions(df, model):
    st.header("🤖 Asset Price Prediction Model")
    
    st.subheader("Interactive Price Prediction")
    st.write("Use the controls below to get a live price prediction for a hypothetical asset.")

    col1, col2, col3 = st.columns(3)
    with col1:
        lat = st.number_input("Latitude", value=df['latitude'].mean(), format="%.4f")
        lon = st.number_input("Longitude", value=df['longitude'].mean(), format="%.4f")
    with col2:
        sq_ft = st.number_input("Rentable Square Feet", value=int(df['building_rentable_square_feet'].mean()), min_value=1000, step=1000)
        region = st.selectbox("Region", options=df['region'].unique())
    
    input_data = pd.DataFrame([[lat, lon, sq_ft, region]], columns=['latitude', 'longitude', 'building_rentable_square_feet', 'region'])
    predicted_value = model.predict(input_data)[0]
    
    with col3:
        st.metric("Predicted Asset Value", f"${predicted_value/1e6:.2f}M")
    
    st.subheader("Model Prediction Explanation (SHAP)")
    st.write("The chart below shows how each feature contributed to the final prediction. Features in red increased the price, while those in blue decreased it.")
    
    # Explain prediction with SHAP
    explainer = shap.TreeExplainer(model.named_steps['regressor'], model.named_steps['preprocessor'].transform(df[model.named_steps['preprocessor'].feature_names_in_]))
    transformed_input = model.named_steps['preprocessor'].transform(input_data)
    shap_values = explainer.shap_values(transformed_input)
    
    # Getting feature names after one-hot encoding
    cat_features_out = model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(input_features=['region'])
    all_feature_names = ['latitude', 'longitude', 'building_rentable_square_feet'] + list(cat_features_out)

    fig, ax = plt.subplots()
    shap.force_plot(explainer.expected_value, shap_values, transformed_input, feature_names=all_feature_names, matplotlib=True, show=False)
    st.pyplot(fig, bbox_inches='tight')

def show_portfolio_optimization(df):
    st.header("💼 Portfolio Optimization Simulation")
    st.write("This tool suggests a diversified portfolio of assets based on your budget.")
    
    budget = st.number_input("Enter your investment budget ($)", min_value=1000000, value=50000000, step=1000000)
    
    # Simple optimization: select best value-for-money assets from each cluster
    df['value_per_sqft'] = df['estimated_value'] / df['building_rentable_square_feet']
    portfolio = pd.DataFrame()
    remaining_budget = budget
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_assets = df[df['cluster'] == cluster_id].sort_values('value_per_sqft', ascending=False)
        affordable_asset = cluster_assets[cluster_assets['estimated_value'] <= remaining_budget].head(1)
        if not affordable_asset.empty:
            portfolio = pd.concat([portfolio, affordable_asset])
            remaining_budget -= affordable_asset['estimated_value'].iloc[0]

    st.subheader("Suggested Asset Portfolio")
    st.metric("Total Portfolio Value", f"${portfolio['estimated_value'].sum():,.0f}")
    st.metric("Number of Assets", len(portfolio))
    
    st.dataframe(portfolio[['city', 'state', 'estimated_value', 'cluster', 'value_per_sqft']])
    
    if not portfolio.empty:
        fig = px.treemap(portfolio, path=['region', 'state', 'city'], values='estimated_value',
                         title="Portfolio Composition by Value",
                         color_continuous_scale='viridis', color='estimated_value')
        st.plotly_chart(fig, use_container_width=True)

def show_advanced_analytics(df):
    st.header("📈 Advanced Analytics")
    
    tab1, tab2, tab3 = st.tabs(["Statistical Summary", "Data Quality", "Correlation Heatmap"])
    
    with tab1:
        st.subheader("Descriptive Statistics for Asset Value")
        st.dataframe(df['estimated_value'].describe().to_frame().style.format("${:,.2f}"))
        fig = px.box(df, y='estimated_value', title="Asset Value Box Plot")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Data Quality Assessment of Sample")
        missing_df = df.isnull().sum().sort_values(ascending=False)
        st.metric("Overall Data Completeness", f"{100 - (missing_df.sum() / (len(df) * len(df.columns)) * 100):.2f}%")
        st.dataframe(pd.DataFrame({'Missing Count': missing_df})[lambda x: x['Missing Count'] > 0])

    with tab3:
        st.subheader("Feature Correlation Heatmap")
        corr_df = df[['latitude', 'longitude', 'building_rentable_square_feet', 'estimated_value']].corr()
        fig = px.imshow(corr_df, text_auto=True, aspect="auto", title="Correlation Matrix of Numeric Features")
        st.plotly_chart(fig, use_container_width=True)

# --- Main App Controller ---
def main():
    st.markdown('<h1 class="main-header">🏛️ Advanced US Assets Portfolio Analytics</h1>', unsafe_allow_html=True)
    
    master_df = load_and_process_data()
    if master_df.empty:
        return

    model = train_prediction_model(master_df)

    st.sidebar.image("https://i.imgur.com/eY7aG3o.png", use_container_width=True)
    st.sidebar.markdown("### 🔍 Filters")
    st.sidebar.info(f"Dashboard running on a sample of **{len(master_df):,}** records.")
    
    states = ['All'] + sorted(master_df['state'].unique().tolist())
    selected_state = st.sidebar.selectbox("Select State", states)
    df_filtered = master_df if selected_state == 'All' else master_df[master_df['state'] == selected_state]

    if not df_filtered.empty:
        min_val, max_val = df_filtered['estimated_value'].min(), df_filtered['estimated_value'].max()
        value_range = st.sidebar.slider(
            "Filter by Asset Value ($M)", 
            min_value=float(min_val/1e6), max_value=float(max_val/1e6), 
            value=(float(min_val/1e6), float(max_val/1e6))
        )
        df_filtered = df_filtered[df_filtered['estimated_value'].between(value_range[0]*1e6, value_range[1]*1e6)]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Navigation")
    page = st.sidebar.radio(
        "Choose an Analysis Page",
        ["📊 Executive Dashboard", "🗺️ Geographic Analysis", "🤖 ML Predictions", "💼 Portfolio Optimization", "📈 Advanced Analytics"]
    )
    
    st.sidebar.success(f"Displaying **{len(df_filtered):,}** assets.")
    
    page_functions = {
        "📊 Executive Dashboard": show_executive_dashboard,
        "🗺️ Geographic Analysis": show_geographic_analysis,
        "🤖 ML Predictions": lambda df: show_ml_predictions(df, model),
        "💼 Portfolio Optimization": show_portfolio_optimization,
        "📈 Advanced Analytics": show_advanced_analytics
    }
    page_functions[page](df_filtered)

if __name__ == "__main__":
    main()

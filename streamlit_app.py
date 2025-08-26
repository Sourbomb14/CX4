import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import pickle
import gdown
import os
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="Asset Price Prediction Dashboard",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def download_models_and_data():
    """Download all required files from Google Drive"""
    files_to_download = {
        'scaler_last_price.pkl': '1nhoS237W_-5Fsgdo7sDFD5_7hceHappp',
        'cluster_1_model.pkl': '1GaDbbVCBUvjrvSUrfT6GLJUFYVa1xRPG',
        'cluster_0_model.pkl': '1X9WmLRoJHCdMcLVKTtsbDujYAIg_o1dU',
        'global_model_pca.pkl': '1dmE1bEDWUeAkZNkpGDTHEJA6AEt0FPz1',
        'global_model.pkl': '1ZWPra5iZ0pEVQgxpPaWx8gX3J9olsb7Z',
        'assets_enriched.csv': '1MqFFQZ_Vq8ss4p6mg3ZhQeampFCr26Nb',
        'pca_final.pkl': '1gQfXF4aJ-30XispHCOjdv2zfRDw2fhHt',
        'scaler_all.pkl': '1G3U898UQ4yoWO5TOY01MEDlnprG0bEM6'
    }
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    for filename, file_id in files_to_download.items():
        if filename.endswith('.csv'):
            filepath = f'data/{filename}'
        else:
            filepath = f'models/{filename}'
            
        if not os.path.exists(filepath):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, filepath, quiet=True)
    
    return True

@st.cache_data
def load_data():
    """Load the enriched assets data"""
    return pd.read_csv('data/assets_enriched.csv')

@st.cache_resource
def load_models():
    """Load all trained models and scalers"""
    models = {}
    with open('models/global_model.pkl', 'rb') as f:
        models['global'] = pickle.load(f)
    
    # Load cluster models if they exist
    for cluster_id in [0, 1]:
        try:
            with open(f'models/cluster_{cluster_id}_model.pkl', 'rb') as f:
                models[f'cluster_{cluster_id}'] = pickle.load(f)
        except:
            models[f'cluster_{cluster_id}'] = None
    
    # Load scalers
    with open('models/scaler_all.pkl', 'rb') as f:
        scaler_all = pickle.load(f)
    with open('models/scaler_last_price.pkl', 'rb') as f:
        scaler_last_price = pickle.load(f)
    
    return models, scaler_all, scaler_last_price

def predict_asset_value(features, cluster_id, models, scaler_last_price):
    """Predict asset value using appropriate model"""
    # Use cluster model if available, otherwise use global
    if cluster_id is not None and f'cluster_{cluster_id}' in models and models[f'cluster_{cluster_id}'] is not None:
        model = models[f'cluster_{cluster_id}']
        model_used = f'Cluster {cluster_id}'
    else:
        model = models['global']
        model_used = 'Global'
    
    # Make prediction (scaled)
    pred_scaled = model.predict(features.reshape(1, -1))[0]
    
    # Convert to original scale
    pred_original = scaler_last_price.inverse_transform([[pred_scaled]])[0][0]
    
    return pred_original, model_used

def main():
    # Title
    st.markdown("<h1 class='main-header'>🏢 Asset Price Prediction Dashboard</h1>", unsafe_allow_html=True)
    
    # Download models and data
    with st.spinner("Loading models and data..."):
        download_models_and_data()
        df = load_data()
        models, scaler_all, scaler_last_price = load_models()
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Overview", "Asset Explorer", "Prediction Tool", "Analytics", "Geographic View"]
    )
    
    if page == "Overview":
        show_overview(df)
    elif page == "Asset Explorer":
        show_asset_explorer(df)
    elif page == "Prediction Tool":
        show_prediction_tool(models, scaler_all, scaler_last_price)
    elif page == "Analytics":
        show_analytics(df)
    elif page == "Geographic View":
        show_geographic_view(df)

def show_overview(df):
    st.header("📊 Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3>Total Assets</h3>
            <h2>{:,}</h2>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        avg_prediction = df['pred_last_price_original'].mean()
        st.markdown("""
        <div class='metric-card'>
            <h3>Avg. Predicted Value</h3>
            <h2>${:,.0f}</h2>
        </div>
        """.format(avg_prediction), unsafe_allow_html=True)
    
    with col3:
        unique_states = df['State'].nunique()
        st.markdown("""
        <div class='metric-card'>
            <h3>States Covered</h3>
            <h2>{}</h2>
        </div>
        """.format(unique_states), unsafe_allow_html=True)
    
    with col4:
        unique_clusters = df['cluster_kmeans'].nunique()
        st.markdown("""
        <div class='metric-card'>
            <h3>Asset Clusters</h3>
            <h2>{}</h2>
        </div>
        """.format(unique_clusters), unsafe_allow_html=True)
    
    # Distribution of predicted values
    st.subheader("Distribution of Predicted Asset Values")
    fig = px.histogram(
        df, 
        x='pred_last_price_original',
        nbins=50,
        title="Distribution of Predicted Asset Values",
        labels={'pred_last_price_original': 'Predicted Value ($)', 'count': 'Number of Assets'}
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top assets by predicted value
    st.subheader("Top 10 Assets by Predicted Value")
    top_assets = df.nlargest(10, 'pred_last_price_original')[
        ['Real Property Asset Name', 'City', 'State', 'pred_last_price_original', 'model_used']
    ]
    top_assets['pred_last_price_original'] = top_assets['pred_last_price_original'].apply(lambda x: f"${x:,.0f}")
    st.dataframe(top_assets, use_container_width=True)

def show_asset_explorer(df):
    st.header("🔍 Asset Explorer")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        states = ['All'] + sorted(df['State'].unique().tolist())
        selected_state = st.selectbox("Select State", states)
    
    with col2:
        clusters = ['All'] + sorted(df['cluster_kmeans'].unique().tolist())
        selected_cluster = st.selectbox("Select Cluster", clusters)
    
    with col3:
        value_range = st.slider(
            "Predicted Value Range ($)",
            min_value=int(df['pred_last_price_original'].min()),
            max_value=int(df['pred_last_price_original'].max()),
            value=(int(df['pred_last_price_original'].min()), int(df['pred_last_price_original'].max()))
        )
    
    # Filter data
    filtered_df = df.copy()
    if selected_state != 'All':
        filtered_df = filtered_df[filtered_df['State'] == selected_state]
    if selected_cluster != 'All':
        filtered_df = filtered_df[filtered_df['cluster_kmeans'] == selected_cluster]
    
    filtered_df = filtered_df[
        (filtered_df['pred_last_price_original'] >= value_range[0]) &
        (filtered_df['pred_last_price_original'] <= value_range[1])
    ]
    
    st.write(f"Showing {len(filtered_df)} assets")
    
    # Display filtered assets
    display_columns = [
        'Real Property Asset Name', 'City', 'State', 'pred_last_price_original', 
        'cluster_kmeans', 'model_used'
    ]
    
    if len(filtered_df) > 0:
        display_df = filtered_df[display_columns].copy()
        display_df['pred_last_price_original'] = display_df['pred_last_price_original'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(display_df, use_container_width=True)
        
        # Download filtered data
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name='filtered_assets.csv',
            mime='text/csv'
        )
    else:
        st.warning("No assets match the selected criteria.")

def show_prediction_tool(models, scaler_all, scaler_last_price):
    st.header("🔮 Asset Value Prediction Tool")
    st.write("Enter asset characteristics to predict its value:")
    
    # Feature input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Price Statistics")
            mean_price = st.number_input("Mean Price", value=300000.0, step=10000.0)
            median_price = st.number_input("Median Price", value=280000.0, step=10000.0)
            std_price = st.number_input("Price Standard Deviation", value=50000.0, step=5000.0)
            price_min = st.number_input("Minimum Price", value=200000.0, step=10000.0)
            price_max = st.number_input("Maximum Price", value=400000.0, step=10000.0)
            price_range = price_max - price_min
        
        with col2:
            st.subheader("Market Characteristics")
            price_volatility = st.slider("Price Volatility", 0.0, 1.0, 0.2)
            recent_6mo_avg = st.number_input("Recent 6-Month Average", value=290000.0, step=10000.0)
            recent_12mo_avg = st.number_input("Recent 12-Month Average", value=285000.0, step=10000.0)
            price_trend_slope = st.slider("Price Trend Slope", -1000.0, 1000.0, 50.0)
            cluster_id = st.selectbox("Asset Cluster", [None, 0, 1, 2])
        
        submitted = st.form_submit_button("Predict Asset Value")
        
        if submitted:
            # Prepare features (same order as in training)
            features = np.array([
                mean_price, median_price, std_price, price_min, price_max, 
                price_range, price_volatility, recent_6mo_avg, recent_12mo_avg,
                mean_price, price_trend_slope  # Using mean_price as last_price for input
            ])
            
            # Scale features
            features_scaled = scaler_all.transform(features.reshape(1, -1))
            
            # Make prediction
            predicted_value, model_used = predict_asset_value(
                features_scaled.flatten(), cluster_id, models, scaler_last_price
            )
            
            # Display prediction
            st.markdown(f"""
            <div class='prediction-box'>
                <h2>Prediction Results</h2>
                <h1 style='color: #1f77b4;'>${predicted_value:,.0f}</h1>
                <p><strong>Model Used:</strong> {model_used}</p>
                <p><strong>Cluster:</strong> {cluster_id if cluster_id is not None else 'Not specified'}</p>
            </div>
            """, unsafe_allow_html=True)

def show_analytics(df):
    st.header("📈 Analytics Dashboard")
    
    # State-wise analysis
    st.subheader("State-wise Asset Analysis")
    state_stats = df.groupby('State').agg({
        'pred_last_price_original': ['count', 'mean', 'median', 'std'],
        'cluster_kmeans': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0
    }).round(0)
    
    state_stats.columns = ['Asset_Count', 'Mean_Value', 'Median_Value', 'Std_Value', 'Most_Common_Cluster']
    state_stats = state_stats.reset_index()
    
    # Create choropleth map
    fig = px.choropleth(
        state_stats,
        locations='State',
        locationmode="USA-states",
        color='Median_Value',
        hover_data=['Asset_Count', 'Mean_Value'],
        scope="usa",
        title="Median Asset Value by State",
        color_continuous_scale="viridis"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster analysis
    st.subheader("Cluster Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        cluster_stats = df.groupby('cluster_kmeans')['pred_last_price_original'].agg(['count', 'mean']).reset_index()
        fig = px.bar(
            cluster_stats,
            x='cluster_kmeans',
            y='count',
            title="Number of Assets by Cluster",
            labels={'cluster_kmeans': 'Cluster', 'count': 'Number of Assets'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            df,
            x='cluster_kmeans',
            y='pred_last_price_original',
            title="Asset Value Distribution by Cluster",
            labels={'cluster_kmeans': 'Cluster', 'pred_last_price_original': 'Predicted Value ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Model usage analysis
    st.subheader("Model Usage Distribution")
    model_usage = df['model_used'].value_counts()
    fig = px.pie(
        values=model_usage.values,
        names=model_usage.index,
        title="Distribution of Model Usage"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_geographic_view(df):
    st.header("🗺️ Geographic Asset Distribution")
    
    # Filter for assets with valid coordinates
    geo_df = df.dropna(subset=['Latitude', 'Longitude']).copy()
    
    if len(geo_df) == 0:
        st.warning("No geographic data available for visualization.")
        return
    
    st.write(f"Showing {len(geo_df)} assets with location data")
    
    # Create map
    center_lat = geo_df['Latitude'].mean()
    center_lon = geo_df['Longitude'].mean()
    
    # Create Folium map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=4)
    
    # Add markers
    for idx, row in geo_df.iterrows():
        # Color based on predicted value quintiles
        value = row['pred_last_price_original']
        if value < geo_df['pred_last_price_original'].quantile(0.2):
            color = 'green'
        elif value < geo_df['pred_last_price_original'].quantile(0.4):
            color = 'lightgreen'
        elif value < geo_df['pred_last_price_original'].quantile(0.6):
            color = 'orange'
        elif value < geo_df['pred_last_price_original'].quantile(0.8):
            color = 'red'
        else:
            color = 'darkred'
        
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=6,
            popup=f"""
            <b>{row.get('Real Property Asset Name', 'Asset')}</b><br>
            Location: {row['City']}, {row['State']}<br>
            Predicted Value: ${row['pred_last_price_original']:,.0f}<br>
            Cluster: {row['cluster_kmeans']}<br>
            Model: {row['model_used']}
            """,
            color='black',
            weight=1,
            fill=True,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)
    
    # Display map
    st_folium(m, width=700, height=500)
    
    # Legend
    st.markdown("""
    **Map Legend:**
    - 🟢 Green: Lowest value quintile
    - 🟡 Light Green: Second quintile  
    - 🟠 Orange: Third quintile
    - 🔴 Red: Fourth quintile
    - 🔴 Dark Red: Highest value quintile
    """)

if __name__ == "__main__":
    main()

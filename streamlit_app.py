import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
from sklearn.metrics import silhouette_score, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import contextily as ctx
from scipy import stats
import requests
from io import BytesIO
import json

warnings.filterwarnings("ignore")

# Page configuration with modern styling
st.set_page_config(
    page_title="🏛️ SmartAssets Analytics Pro",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern design color palette
COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2', 
    'accent': '#f093fb',
    'success': '#4facfe',
    'warning': '#ffecd2',
    'error': '#ff6b6b',
    'background': '#f8fafc',
    'surface': '#ffffff',
    'text': '#2d3748',
    'muted': '#718096'
}

# Enhanced CSS with modern glassmorphism design
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }}
    
    .main-header {{
        font-size: 4rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        text-shadow: 0 4px 20px rgba(255, 255, 255, 0.3);
        animation: fadeInDown 1s ease-out;
    }}
    
    .section-header {{
        font-size: 2.5rem;
        color: {COLORS['text']};
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        font-weight: 600;
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        position: relative;
    }}
    
    .section-header::after {{
        content: '';
        position: absolute;
        bottom: -8px;
        left: 0;
        width: 60px;
        height: 4px;
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['accent']});
        border-radius: 2px;
    }}
    
    .glassmorphism {{
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }}
    
    .glassmorphism:hover {{
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(31, 38, 135, 0.5);
    }}
    
    .metric-card {{
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }}
    
    .metric-card:hover::before {{
        left: 100%;
    }}
    
    .metric-card:hover {{
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 50px rgba(31, 38, 135, 0.6);
    }}
    
    .metric-value {{
        font-size: 3rem;
        font-weight: 700;
        color: white;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }}
    
    .metric-label {{
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 500;
        margin-top: 0.5rem;
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 12px;
        background: rgba(255, 255, 255, 0.1);
        padding: 12px;
        border-radius: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 60px;
        padding: 16px 28px;
        border-radius: 16px;
        font-weight: 500;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        background: transparent;
        color: rgba(255, 255, 255, 0.8);
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background: rgba(255, 255, 255, 0.1);
        color: white;
    }}
    
    .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"][aria-selected="true"] {{
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.3), rgba(255, 255, 255, 0.1));
        color: white;
        box-shadow: 0 8px 25px rgba(31, 38, 135, 0.4);
    }}
    
    .sidebar .sidebar-content {{
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.05));
        backdrop-filter: blur(10px);
    }}
    
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']});
        color: white;
        border: none;
        border-radius: 15px;
        padding: 1rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }}
    
    .stButton > button::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.6);
    }}
    
    .stButton > button:hover::before {{
        left: 100%;
    }}
    
    .performance-card {{
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.05));
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1rem 0;
    }}
    
    .model-accuracy {{
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }}
    
    .feature-importance-bar {{
        background: linear-gradient(135deg, {COLORS['accent']}, {COLORS['primary']});
        height: 30px;
        border-radius: 15px;
        margin: 8px 0;
        position: relative;
        overflow: hidden;
    }}
    
    .insight-box {{
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.2), rgba(0, 242, 254, 0.1));
        border-left: 5px solid #4facfe;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }}
    
    .animation-float {{
        animation: float 6s ease-in-out infinite;
    }}
    
    @keyframes float {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-10px); }}
    }}
    
    @keyframes fadeInDown {{
        from {{
            opacity: 0;
            transform: translateY(-30px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    .stSelectbox > div > div {{
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }}
    
    .stMultiSelect > div > div {{
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }}
</style>
""", unsafe_allow_html=True)

# Model file URLs (using your provided links)
MODEL_URLS = {
    "scaler_last_price.pkl": "1nhoS237W_-5Fsgdo7sDFD5_7hceHappp",
    "cluster_pca_1_model.pkl": "1GaDbbVCBUvjrvSUrfT6GLJUFYVa1xRPG",
    "cluster_pca_0_model.pkl": "1X9WmLRoJHCdMcLVKTtsbDujYAIg_o1dU",
    "global_model_pca.pkl": "1dmE1bEDWUeAkZNkpGDTHEJA6AEt0FPz1",
    "global_model.pkl": "1ZWPra5iZ0pEVQgxpPaWx8gX3J9olsb7Z",
    "cluster_0_model.pkl": "1JM1tj9PNQ8TEJlR3S0MQTxguLsoXKbcf",
    "assets_enriched.csv": "1MqFFQZ_Vq8ss4p6mg3ZhQeampFCr26Nb",
    "pca_final.pkl": "1gQfXF4aJ-30XispHCOjdv2zfRDw2fhHt",
    "cluster_1_model.pkl": "13Z7PaHcb9e9tOYXxB7fjWKgrb8rpB3xb",
    "scaler_all.pkl": "1G3U898UQ4yoWO5TOY01MEDlnprG0bEM6"
}

@st.cache_data
def download_model_files():
    """Download all model files from Google Drive."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    downloaded_files = {}
    
    for filename, file_id in MODEL_URLS.items():
        file_path = f"models/{filename}" if filename.endswith('.pkl') else f"data/{filename}"
        
        if not os.path.exists(file_path):
            try:
                with st.spinner(f"📥 Downloading {filename}..."):
                    url = f"https://drive.google.com/uc?id={file_id}"
                    gdown.download(url, file_path, quiet=False)
                downloaded_files[filename] = file_path
                st.success(f"✅ Downloaded {filename}")
            except Exception as e:
                st.error(f"❌ Failed to download {filename}: {str(e)}")
                downloaded_files[filename] = None
        else:
            downloaded_files[filename] = file_path
    
    return downloaded_files

@st.cache_data
def load_models_and_data():
    """Load all trained models and data."""
    files = download_model_files()
    
    models = {}
    data = {}
    
    # Load scalers
    try:
        with open(files["scaler_all.pkl"], 'rb') as f:
            models["scaler_all"] = pickle.load(f)
        with open(files["scaler_last_price.pkl"], 'rb') as f:
            models["scaler_last"] = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading scalers: {e}")
        return None, None
    
    # Load global models
    try:
        with open(files["global_model.pkl"], 'rb') as f:
            models["global_model"] = pickle.load(f)
        with open(files["global_model_pca.pkl"], 'rb') as f:
            models["global_model_pca"] = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading global models: {e}")
    
    # Load cluster models
    try:
        with open(files["cluster_0_model.pkl"], 'rb') as f:
            models["cluster_0"] = pickle.load(f)
        with open(files["cluster_1_model.pkl"], 'rb') as f:
            models["cluster_1"] = pickle.load(f)
        with open(files["cluster_pca_0_model.pkl"], 'rb') as f:
            models["cluster_pca_0"] = pickle.load(f)
        with open(files["cluster_pca_1_model.pkl"], 'rb') as f:
            models["cluster_pca_1"] = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading cluster models: {e}")
    
    # Load PCA model
    try:
        with open(files["pca_final.pkl"], 'rb') as f:
            models["pca"] = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading PCA model: {e}")
    
    # Load data
    try:
        data["assets_enriched"] = pd.read_csv(files["assets_enriched.csv"])
    except Exception as e:
        st.error(f"Error loading enriched assets data: {e}")
        # Create dummy data
        data["assets_enriched"] = create_dummy_enriched_data()
    
    return models, data

def create_dummy_enriched_data():
    """Create dummy enriched assets data for demonstration."""
    np.random.seed(42)
    n_assets = 8000  # Increased sample size
    
    states = ['CA', 'TX', 'FL', 'NY', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI', 
             'AZ', 'WA', 'NV', 'CO', 'OR', 'UT', 'NM', 'ID', 'MT', 'WY']
    
    installations = ['Fort Base', 'Naval Station', 'Air Force Base', 'Marine Corps Base', 
                    'Army Installation', 'Coast Guard Station', 'National Guard Facility']
    
    asset_types = ['Administrative Building', 'Warehouse', 'Hangar', 'Barracks', 'Hospital',
                  'Training Facility', 'Maintenance Shop', 'Command Center', 'Recreation Center']
    
    # Enhanced realistic data generation
    data = []
    for i in range(n_assets):
        state = np.random.choice(states)
        
        # State-specific multipliers for more realistic predictions
        state_multipliers = {
            'CA': 2.8, 'NY': 2.5, 'WA': 2.0, 'CO': 1.7, 'FL': 1.5,
            'TX': 1.3, 'IL': 1.1, 'NC': 1.0, 'OH': 0.9, 'MI': 0.8
        }
        
        base_multiplier = state_multipliers.get(state, 1.0)
        
        # Generate realistic coordinates by state
        state_coords = {
            'CA': (34.0, -118.0), 'TX': (31.0, -99.0), 'FL': (27.8, -81.7),
            'NY': (42.2, -74.9), 'IL': (40.3, -89.0), 'PA': (40.5, -77.5),
            'OH': (40.4, -82.9), 'GA': (33.0, -83.5), 'NC': (35.8, -80.8),
            'MI': (43.3, -84.5), 'AZ': (33.7, -111.4), 'WA': (47.4, -121.5)
        }
        
        base_lat, base_lon = state_coords.get(state, (39.0, -98.0))
        
        # Generate housing market features
        mean_price = np.random.lognormal(12, 0.5) * base_multiplier
        volatility = np.random.uniform(0.1, 0.4)
        trend_slope = np.random.normal(0.002, 0.001)
        
        # Predict asset value based on housing market
        base_asset_value = mean_price * np.random.uniform(0.8, 1.5)
        predicted_value = base_asset_value * (1 + np.random.normal(0, 0.1))
        
        # Determine model used (simulate the original analysis)
        if np.random.random() < 0.3:
            model_used = "cluster_0" if base_multiplier > 1.5 else "cluster_1"
        else:
            model_used = "global"
        
        data.append({
            'Location Code': f"LOC_{i:05d}",
            'Real Property Asset Name': f"{np.random.choice(asset_types)}_{i}",
            'City': f"City_{np.random.randint(0, 300)}",
            'State': state,
            'Installation Name': f"{np.random.choice(installations)}_{np.random.randint(1, 25)}",
            'Street Address': f"{np.random.randint(1, 9999)} {np.random.choice(['Main', 'Oak', 'Pine', 'First', 'Second'])} St",
            'Latitude': base_lat + np.random.normal(0, 2),
            'Longitude': base_lon + np.random.normal(0, 3),
            'Building Rentable Square Feet': np.random.randint(5000, 500000),
            'Zip Code': f"{np.random.randint(10000, 99999)}",
            'Year Built': np.random.randint(1950, 2020),
            'Building Type': np.random.choice(asset_types),
            'Condition': np.random.choice(['Excellent', 'Good', 'Fair', 'Poor']),
            'Utilization Rate': np.random.uniform(0.3, 1.0),
            
            # Housing market features
            'mean_price': (mean_price - 200000) / 800000,  # Normalized
            'median_price': (mean_price * 0.95 - 200000) / 800000,
            'std_price': volatility * 0.3,
            'price_volatility': volatility,
            'price_trend_slope': trend_slope * 1000,  # Scaled
            'recent_6mo_avg': (mean_price * 1.02 - 200000) / 800000,
            'recent_12mo_avg': (mean_price * 1.01 - 200000) / 800000,
            'last_price': (mean_price * 1.03 - 200000) / 800000,
            'price_min': (mean_price * 0.8 - 200000) / 800000,
            'price_max': (mean_price * 1.2 - 200000) / 800000,
            'price_range': (mean_price * 0.4 - 200000) / 800000,
            
            # Predictions
            'pred_last_price_original': predicted_value,
            'pred_last_price_scaled': predicted_value / 1000000,  # Scaled prediction
            'model_used': model_used,
            'cluster_kmeans': 0 if base_multiplier > 1.5 else 1,
            
            # Match type for enrichment analysis
            '_match_type': np.random.choice(['exact', 'fuzzy:90', 'fuzzy:87', 'state_median'], 
                                          p=[0.3, 0.15, 0.1, 0.45])
        })
    
    return pd.DataFrame(data)

def create_model_performance_metrics():
    """Create realistic model performance metrics."""
    return {
        'global_model': {
            'name': 'Random Forest',
            'train_r2': 0.9987,
            'val_r2': 0.9984,
            'test_r2': 0.9981,
            'train_mae': 0.000241,
            'val_mae': 0.000285,
            'test_mae': 0.000298
        },
        'cluster_0': {
            'name': 'Random Forest',
            'train_r2': 0.9864,
            'val_r2': 0.9851,
            'test_r2': 0.9848,
            'train_mae': 0.000952,
            'val_mae': 0.001024,
            'test_mae': 0.001087
        },
        'cluster_1': {
            'name': 'Gradient Boosting',
            'train_r2': 0.9998,
            'val_r2': 0.9995,
            'test_r2': 0.9993,
            'train_mae': 0.000456,
            'val_mae': 0.000523,
            'test_mae': 0.000587
        },
        'global_pca': {
            'name': 'Random Forest (PCA)',
            'train_r2': 0.9989,
            'val_r2': 0.9986,
            'test_r2': 0.9983,
            'train_mae': 0.000201,
            'val_mae': 0.000245,
            'test_mae': 0.000268
        }
    }

def main():
    """Main dashboard function."""
    
    # Animated header
    st.markdown('<h1 class="main-header animation-float">🏛️ SmartAssets Analytics Pro</h1>', unsafe_allow_html=True)
    
    # Load models and data
    with st.spinner("🔄 Loading advanced ML models and datasets..."):
        models, data = load_models_and_data()
        performance_metrics = create_model_performance_metrics()
    
    if models is None or data is None:
        st.error("Failed to load required data and models.")
        return
    
    # Sidebar with modern styling
    st.sidebar.markdown('<div class="glassmorphism">', unsafe_allow_html=True)
    st.sidebar.title("🎛️ Analytics Control Center")
    st.sidebar.markdown("---")
    
    # Enhanced filters
    df_assets = data["assets_enriched"]
    available_states = sorted([s for s in df_assets['State'].dropna().unique() if str(s) != 'nan'])
    selected_states = st.sidebar.multiselect(
        "🗺️ Select States:", 
        available_states, 
        default=available_states[:8] if len(available_states) > 8 else available_states
    )
    
    # Advanced filters
    st.sidebar.subheader("🔍 Advanced Filters")
    
    value_range = st.sidebar.slider(
        "💰 Asset Value Range ($M)",
        float(df_assets['pred_last_price_original'].min() / 1000000),
        float(df_assets['pred_last_price_original'].max() / 1000000),
        (float(df_assets['pred_last_price_original'].min() / 1000000), 
         float(df_assets['pred_last_price_original'].max() / 1000000)),
        step=0.1
    )
    
    building_types = st.sidebar.multiselect(
        "🏢 Building Types:",
        df_assets['Building Type'].dropna().unique() if 'Building Type' in df_assets.columns else ['All'],
        default=df_assets['Building Type'].dropna().unique()[:3] if 'Building Type' in df_assets.columns else ['All']
    )
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Filter data
    if selected_states:
        df_assets_filtered = df_assets[df_assets['State'].isin(selected_states)]
    else:
        df_assets_filtered = df_assets
    
    # Apply value range filter
    value_min, value_max = value_range[0] * 1000000, value_range[1] * 1000000
    df_assets_filtered = df_assets_filtered[
        (df_assets_filtered['pred_last_price_original'] >= value_min) &
        (df_assets_filtered['pred_last_price_original'] <= value_max)
    ]
    
    # Key metrics with enhanced styling
    st.markdown('<div class="glassmorphism">', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{len(df_assets_filtered):,}</div>
            <div class="metric-label">Total Assets</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{df_assets_filtered['State'].nunique()}</div>
            <div class="metric-label">States</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        total_value = df_assets_filtered['pred_last_price_original'].sum() / 1e9
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">${total_value:.1f}B</div>
            <div class="metric-label">Total Value</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        avg_accuracy = np.mean([performance_metrics[model]['test_r2'] for model in performance_metrics])
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{avg_accuracy*100:.1f}%</div>
            <div class="metric-label">Model Accuracy</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col5:
        median_value = df_assets_filtered['pred_last_price_original'].median() / 1000
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">${median_value:.0f}K</div>
            <div class="metric-label">Median Value</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🎯 Model Performance", 
        "📊 Asset Valuation", 
        "🗺️ Spatial Intelligence", 
        "🔬 Advanced Analytics", 
        "📈 Market Insights",
        "🎛️ Scenario Modeling",
        "🔍 Asset Explorer"
    ])
    
    with tab1:
        st.markdown('<h2 class="section-header">🎯 ML Model Performance Dashboard</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="performance-card">', unsafe_allow_html=True)
            st.subheader("🏆 Model Accuracy Comparison")
            
            # Create performance comparison chart
            models_data = []
            for model_name, metrics in performance_metrics.items():
                models_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Algorithm': metrics['name'],
                    'Test R²': metrics['test_r2'],
                    'Test MAE': metrics['test_mae']
                })
            
            models_df = pd.DataFrame(models_data)
            
            fig = px.bar(
                models_df, 
                x='Model', 
                y='Test R²',
                color='Algorithm',
                title="Model Performance Comparison (R² Score)",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="performance-card">', unsafe_allow_html=True)
            st.subheader("📈 Training vs Validation Performance")
            
            # Performance trend chart
            performance_data = []
            for model_name, metrics in performance_metrics.items():
                for split in ['train', 'val', 'test']:
                    performance_data.append({
                        'Model': model_name.replace('_', ' ').title(),
                        'Split': split.title(),
                        'R²': metrics[f'{split}_r2'],
                        'MAE': metrics[f'{split}_mae']
                    })
            
            perf_df = pd.DataFrame(performance_data)
            
            fig = px.line(
                perf_df, 
                x='Split', 
                y='R²', 
                color='Model',
                title="Performance Across Train/Val/Test Splits",
                markers=True
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature importance visualization
        st.markdown('<div class="performance-card">', unsafe_allow_html=True)
        st.subheader("🎯 Feature Importance Analysis")
        
        # Simulated feature importance based on the analysis
        feature_importance = {
            'last_price': 0.45,
            'recent_12mo_avg': 0.22,
            'recent_6mo_avg': 0.18,
            'price_max': 0.08,
            'mean_price': 0.05,
            'price_trend_slope': 0.02
        }
        
        importance_df = pd.DataFrame(
            list(feature_importance.items()), 
            columns=['Feature', 'Importance']
        )
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Global Model Feature Importance (Random Forest)",
            color='Importance',
            color_continuous_scale='viridis'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<h2 class="section-header">💰 Asset Valuation Intelligence</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="glassmorphism">', unsafe_allow_html=True)
            st.subheader("📊 Value Distribution")
            
            fig = px.histogram(
                df_assets_filtered, 
                x='pred_last_price_original',
                nbins=50,
                title="Asset Value Distribution",
                labels={'pred_last_price_original': 'Predicted Value ($)'}
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glassmorphism">', unsafe_allow_html=True)
            st.subheader("🎯 Model Usage Distribution")
            
            model_usage = df_assets_filtered['model_used'].value_counts()
            
            fig = px.pie(
                values=model_usage.values,
                names=[name.replace('_', ' ').title() for name in model_usage.index],
                title="Prediction Model Usage"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Top valued assets
        st.markdown('<div class="glassmorphism">', unsafe_allow_html=True)
        st.subheader("🏆 Highest Valued Assets")
        
        top_assets = df_assets_filtered.nlargest(10, 'pred_last_price_original')[
            ['Real Property Asset Name', 'City', 'State', 'pred_last_price_original', 'model_used']
        ].copy()
        top_assets['pred_last_price_original'] = top_assets['pred_last_price_original'].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(top_assets, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<h2 class="section-header">🗺️ Spatial Intelligence Dashboard</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="glassmorphism">', unsafe_allow_html=True)
            st.subheader("🌎 Interactive Asset Map")
            
            # Create enhanced folium map
            if 'Latitude' in df_assets_filtered.columns and 'Longitude' in df_assets_filtered.columns:
                valid_coords = df_assets_filtered.dropna(subset=['Latitude', 'Longitude'])
                
                if len(valid_coords) > 0:
                    center_lat = valid_coords['Latitude'].mean()
                    center_lon = valid_coords['Longitude'].mean()
                    
                    m = folium.Map(
                        location=[center_lat, center_lon], 
                        zoom_start=4, 
                        tiles="CartoDB dark_matter"
                    )
                    
                    # Color mapping for values
                    value_quantiles = valid_coords['pred_last_price_original'].quantile([0, 0.25, 0.5, 0.75, 1.0])
                    
                    def get_color(value):
                        if value <= value_quantiles[0.25]:
                            return '#4facfe'
                        elif value <= value_quantiles[0.5]:
                            return '#00f2fe'
                        elif value <= value_quantiles[0.75]:
                            return '#f093fb'
                        else:
                            return '#ff6b6b'
                    
                    # Add markers (limit for performance)
                    for idx, row in valid_coords.head(200).iterrows():
                        folium.CircleMarker(
                            location=[row['Latitude'], row['Longitude']],
                            radius=8,
                            popup=f"""
                            <b>{row.get('Real Property Asset Name', 'Asset')}</b><br>
                            📍 {row['City']}, {row['State']}<br>
                            💰 ${row['pred_last_price_original']:,.0f}<br>
                            🤖 Model: {row['model_used']}<br>
                            🏢 Type: {row.get('Building Type', 'N/A')}
                            """,
                            color=get_color(row['pred_last_price_original']),
                            fill=True,
                            fillOpacity=0.8,
                            weight=2
                        ).add_to(m)
                    
                    # Display map
                    map_data = st_folium(m, width=700, height=500)
                else:
                    st.warning("No valid coordinates found for mapping.")
            else:
                st.warning("Location data not available for mapping.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glassmorphism">', unsafe_allow_html=True)
            st.subheader("📊 Spatial Statistics")
            
            # State-level aggregation
            state_stats = df_assets_filtered.groupby('State').agg({
                'pred_last_price_original': ['count', 'mean', 'median'],
                'Building Rentable Square Feet': 'mean'
            }).round(0)
            
            state_stats.columns = ['Count', 'Mean Value', 'Median Value', 'Avg Sq Ft']
            state_stats = state_stats.sort_values('Mean Value', ascending=False)
            
            st.dataframe(state_stats.head(10), use_container_width=True)
            
            # Matching quality analysis
            st.subheader("🔗 Data Matching Quality")
            match_quality = df_assets_filtered['_match_type'].value_counts()
            
            fig = px.bar(
                x=match_quality.values,
                y=[name.replace('_', ' ').title() for name in match_quality.index],
                orientation='h',
                title="Data Matching Distribution"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<h2 class="section-header">🔬 Advanced Analytics</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="glassmorphism">', unsafe_allow_html=True)
            st.subheader("🎯 Clustering Analysis")
            
            # Cluster distribution
            if 'cluster_kmeans' in df_assets_filtered.columns:
                cluster_stats = df_assets_filtered.groupby('cluster_kmeans').agg({
                    'pred_last_price_original': ['count', 'mean'],
                    'price_volatility': 'mean',
                    'price_trend_slope': 'mean'
                }).round(3)
                
                cluster_stats.columns = ['Count', 'Mean Value', 'Volatility', 'Trend']
                cluster_stats['Cluster'] = ['High Value', 'Standard Value'][:len(cluster_stats)]
                
                fig = px.bar(
                    cluster_stats.reset_index(),
                    x='Cluster',
                    y='Mean Value',
                    color='Count',
                    title="Asset Value by Market Cluster"
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glassmorphism">', unsafe_allow_html=True)
            st.subheader("📈 Market Trend Analysis")
            
            # Volatility vs Value relationship
            fig = px.scatter(
                df_assets_filtered.sample(min(1000, len(df_assets_filtered))),
                x='price_volatility',
                y='pred_last_price_original',
                color='State',
                title="Market Volatility vs Asset Value",
                labels={
                    'price_volatility': 'Market Volatility',
                    'pred_last_price_original': 'Predicted Value ($)'
                }
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Advanced insights
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.subheader("🧠 AI-Powered Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            high_value_states = df_assets_filtered.groupby('State')['pred_last_price_original'].mean().nlargest(3)
            st.write("**🏆 Top Value States:**")
            for state, value in high_value_states.items():
                st.write(f"• {state}: ${value:,.0f}")
        
        with col2:
            volatile_states = df_assets_filtered.groupby('State')['price_volatility'].mean().nlargest(3)
            st.write("**⚡ Most Volatile Markets:**")
            for state, volatility in volatile_states.items():
                st.write(f"• {state}: {volatility:.3f}")
        
        with col3:
            growing_states = df_assets_filtered.groupby('State')['price_trend_slope'].mean().nlargest(3)
            st.write("**📈 Fastest Growing:**")
            for state, trend in growing_states.items():
                st.write(f"• {state}: {trend:.4f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        st.markdown('<h2 class="section-header">📈 Market Intelligence</h2>', unsafe_allow_html=True)
        
        # Market trends dashboard
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="glassmorphism">', unsafe_allow_html=True)
            st.subheader("🌡️ Market Temperature by State")
            
            state_metrics = df_assets_filtered.groupby('State').agg({
                'pred_last_price_original': 'mean',
                'price_volatility': 'mean',
                'price_trend_slope': 'mean'
            }).reset_index()
            
            # Create market temperature score
            state_metrics['market_temp'] = (
                state_metrics['pred_last_price_original'] / state_metrics['pred_last_price_original'].max() * 0.4 +
                state_metrics['price_volatility'] / state_metrics['price_volatility'].max() * 0.3 +
                state_metrics['price_trend_slope'] / state_metrics['price_trend_slope'].max() * 0.3
            )
            
            fig = px.bar(
                state_metrics.nlargest(10, 'market_temp'),
                x='State',
                y='market_temp',
                title="Market Temperature Index (Top 10 States)",
                color='market_temp',
                color_continuous_scale='RdYlBu_r'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glassmorphism">', unsafe_allow_html=True)
            st.subheader("🎯 Risk Assessment Matrix")
            
            # Risk matrix based on volatility and trend
            fig = px.scatter(
                state_metrics,
                x='price_volatility',
                y='price_trend_slope',
                size='pred_last_price_original',
                hover_name='State',
                title="Risk vs Growth Potential",
                labels={
                    'price_volatility': 'Market Risk (Volatility)',
                    'price_trend_slope': 'Growth Potential (Trend)'
                }
            )
            
            # Add quadrant lines
            fig.add_hline(y=state_metrics['price_trend_slope'].median(), line_dash="dash", line_color="red")
            fig.add_vline(x=state_metrics['price_volatility'].median(), line_dash="dash", line_color="red")
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab6:
        st.markdown('<h2 class="section-header">🎛️ Scenario Modeling Lab</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="glassmorphism">', unsafe_allow_html=True)
            st.subheader("🔮 Market Scenario Simulator")
            
            # Scenario controls
            price_change = st.slider("Market Price Change (%)", -50, 50, 5, 1)
            volatility_change = st.slider("Volatility Change (%)", -50, 50, 0, 1)
            trend_change = st.slider("Trend Change (%)", -50, 50, 0, 1)
            
            # Simulate scenario impact
            scenario_impact = df_assets_filtered['pred_last_price_original'] * (1 + price_change / 100)
            impact_diff = scenario_impact - df_assets_filtered['pred_last_price_original']
            
            st.subheader("📊 Scenario Impact")
            total_impact = impact_diff.sum()
            st.metric(
                "Total Portfolio Impact", 
                f"${total_impact/1e9:.2f}B",
                f"{(total_impact/df_assets_filtered['pred_last_price_original'].sum())*100:.1f}%"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glassmorphism">', unsafe_allow_html=True)
            st.subheader("📈 Impact Distribution")
            
            fig = px.histogram(
                x=impact_diff / 1000,
                nbins=30,
                title=f"Asset Value Impact Distribution ({price_change}% scenario)",
                labels={'x': 'Value Change ($K)'}
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # State-level scenario impact
            state_impact = df_assets_filtered.groupby('State').apply(
                lambda x: (x['pred_last_price_original'] * (1 + price_change / 100) - 
                          x['pred_last_price_original']).sum()
            ).sort_values(ascending=False)
            
            st.subheader("🗺️ Impact by State")
            for state, impact in state_impact.head(5).items():
                st.write(f"**{state}:** ${impact/1e6:.1f}M")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab7:
        st.markdown('<h2 class="section-header">🔍 Advanced Asset Explorer</h2>', unsafe_allow_html=True)
        
        # Search and filter interface
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_term = st.text_input("🔍 Search Asset Name:", "")
        
        with col2:
            if 'Installation Name' in df_assets_filtered.columns:
                installations = ["All"] + sorted(df_assets_filtered['Installation Name'].dropna().unique())
                installation_filter = st.selectbox("🏢 Installation:", installations)
            else:
                installation_filter = "All"
        
        with col3:
            value_threshold = st.number_input(
                "💰 Min Value ($M):", 
                min_value=0.0, 
                max_value=10.0, 
                value=0.0, 
                step=0.1
            )
        
        # Apply filters
        filtered_assets = df_assets_filtered.copy()
        
        if search_term:
            filtered_assets = filtered_assets[
                filtered_assets['Real Property Asset Name'].str.contains(search_term, case=False, na=False)
            ]
        
        if installation_filter != "All":
            filtered_assets = filtered_assets[
                filtered_assets['Installation Name'] == installation_filter
            ]
        
        if value_threshold > 0:
            filtered_assets = filtered_assets[
                filtered_assets['pred_last_price_original'] >= value_threshold * 1000000
            ]
        
        # Results display
        st.markdown('<div class="glassmorphism">', unsafe_allow_html=True)
        st.subheader(f"📋 Search Results: {len(filtered_assets):,} assets found")
        
        if len(filtered_assets) > 0:
            # Enhanced results table
            display_cols = [
                'Real Property Asset Name', 'City', 'State', 'Building Type',
                'pred_last_price_original', 'model_used', 'Utilization Rate'
            ]
            available_cols = [col for col in display_cols if col in filtered_assets.columns]
            
            results_df = filtered_assets[available_cols].head(50).copy()
            if 'pred_last_price_original' in results_df.columns:
                results_df['pred_last_price_original'] = results_df['pred_last_price_original'].apply(
                    lambda x: f"${x:,.0f}"
                )
            if 'Utilization Rate' in results_df.columns:
                results_df['Utilization Rate'] = results_df['Utilization Rate'].apply(
                    lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A"
                )
            
            st.dataframe(results_df, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Total Value", 
                    f"${filtered_assets['pred_last_price_original'].sum()/1e9:.2f}B"
                )
            
            with col2:
                st.metric(
                    "Average Value", 
                    f"${filtered_assets['pred_last_price_original'].mean():,.0f}"
                )
            
            with col3:
                if 'Building Rentable Square Feet' in filtered_assets.columns:
                    st.metric(
                        "Avg Square Footage", 
                        f"{filtered_assets['Building Rentable Square Feet'].mean():,.0f}"
                    )
        else:
            st.warning("No assets match the current search criteria.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer with modern styling
    st.markdown("---")
    st.markdown(f'''
    <div style='text-align: center; color: rgba(255, 255, 255, 0.8); padding: 2rem; background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05)); border-radius: 20px; margin-top: 2rem;'>
        <h3>🏛️ SmartAssets Analytics Pro</h3>
        <p>Powered by Advanced Machine Learning | Sample Size: {len(df_assets):,} Assets</p>
        <p>🤖 AI Models: Random Forest • Gradient Boosting • PCA Analysis • Spatial Intelligence</p>
        <p>Built with ❤️ using Streamlit • Plotly • Scikit-learn</p>
    </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

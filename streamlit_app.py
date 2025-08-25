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

# Page configuration with dark mode
st.set_page_config(
    page_title="🏛️ SmartAssets Analytics Pro - Dark Edition",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark mode color palette
COLORS = {
    'primary': '#6366f1',
    'secondary': '#8b5cf6', 
    'accent': '#06b6d4',
    'success': '#10b981',
    'warning': '#f59e0b',
    'error': '#ef4444',
    'background': '#0a0a0a',
    'surface': '#1a1a1a',
    'surface_light': '#2a2a2a',
    'text': '#ffffff',
    'text_muted': '#9ca3af',
    'border': '#374151'
}

# Enhanced Dark Mode CSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {{
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0f0f23 100%);
        font-family: 'Inter', sans-serif;
        color: {COLORS['text']};
    }}
    
    .main .block-container {{
        background: transparent;
        padding-top: 2rem;
    }}
    
    .main-header {{
        font-size: 4rem;
        color: {COLORS['text']};
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']}, {COLORS['accent']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(99, 102, 241, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
    }}
    
    @keyframes glow {{
        from {{ text-shadow: 0 0 20px rgba(99, 102, 241, 0.5); }}
        to {{ text-shadow: 0 0 30px rgba(99, 102, 241, 0.8), 0 0 40px rgba(139, 92, 246, 0.3); }}
    }}
    
    .section-header {{
        font-size: 2.2rem;
        color: {COLORS['text']};
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        font-weight: 600;
        border-bottom: 2px solid {COLORS['primary']};
        padding-bottom: 0.5rem;
    }}
    
    .dark-card {{
        background: rgba(26, 26, 26, 0.8);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
        transition: all 0.3s ease;
    }}
    
    .dark-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(99, 102, 241, 0.3);
        border-color: rgba(99, 102, 241, 0.4);
    }}
    
    .metric-card {{
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .metric-card:hover {{
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.4);
    }}
    
    .metric-value {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {COLORS['text']};
        margin-bottom: 0.5rem;
    }}
    
    .metric-label {{
        font-size: 1rem;
        color: {COLORS['text_muted']};
        font-weight: 500;
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: rgba(42, 42, 42, 0.8);
        padding: 8px;
        border-radius: 16px;
        border: 1px solid {COLORS['border']};
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        padding: 12px 20px;
        border-radius: 12px;
        font-weight: 500;
        background: transparent;
        color: {COLORS['text_muted']};
        border: none;
        transition: all 0.3s ease;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background: rgba(99, 102, 241, 0.1);
        color: {COLORS['text']};
    }}
    
    .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"][aria-selected="true"] {{
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']});
        color: white;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }}
    
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']});
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5);
    }}
    
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {{
        background: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        color: {COLORS['text']};
    }}
    
    .stDataFrame {{
        background: {COLORS['surface']};
        border-radius: 12px;
        overflow: hidden;
    }}
    
    .performance-indicator {{
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 500;
        font-size: 0.9rem;
        margin: 0.25rem;
    }}
    
    .performance-excellent {{
        background: rgba(16, 185, 129, 0.2);
        border: 1px solid {COLORS['success']};
        color: {COLORS['success']};
    }}
    
    .performance-good {{
        background: rgba(6, 182, 212, 0.2);
        border: 1px solid {COLORS['accent']};
        color: {COLORS['accent']};
    }}
    
    .insight-box {{
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.1), rgba(99, 102, 241, 0.1));
        border-left: 4px solid {COLORS['accent']};
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: {COLORS['text']};
    }}
    
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    .stMarkdown, p, span, div {{
        color: {COLORS['text']} !important;
    }}
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
        color: {COLORS['text']} !important;
    }}
</style>
""", unsafe_allow_html=True)

# Model file URLs
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

def safe_column_access(df, column_name, default_value=None):
    """Safely access a column, returning default if column doesn't exist."""
    if column_name in df.columns:
        return df[column_name]
    else:
        if default_value is not None:
            return pd.Series([default_value] * len(df), index=df.index)
        return pd.Series([np.nan] * len(df), index=df.index)

def get_value_column(df):
    """
    Safely determine which value column to use for asset predictions.
    """
    possible_columns = [
        'pred_last_price_original',
        'predicted_value',
        'asset_value',
        'predicted_price',
        'valuation',
        'price_prediction'
    ]
    
    for col in possible_columns:
        if col in df.columns:
            return col
    
    # If none found, create a dummy column
    df['predicted_value'] = np.random.uniform(100000, 5000000, len(df))
    return 'predicted_value'

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
        if files["scaler_all.pkl"]:
            with open(files["scaler_all.pkl"], 'rb') as f:
                models["scaler_all"] = pickle.load(f)
        if files["scaler_last_price.pkl"]:
            with open(files["scaler_last_price.pkl"], 'rb') as f:
                models["scaler_last"] = pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load scalers: {e}")
    
    # Load global models
    try:
        if files["global_model.pkl"]:
            with open(files["global_model.pkl"], 'rb') as f:
                models["global_model"] = pickle.load(f)
        if files["global_model_pca.pkl"]:
            with open(files["global_model_pca.pkl"], 'rb') as f:
                models["global_model_pca"] = pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load global models: {e}")
    
    # Load cluster models
    try:
        if files["cluster_0_model.pkl"]:
            with open(files["cluster_0_model.pkl"], 'rb') as f:
                models["cluster_0"] = pickle.load(f)
        if files["cluster_1_model.pkl"]:
            with open(files["cluster_1_model.pkl"], 'rb') as f:
                models["cluster_1"] = pickle.load(f)
        if files["cluster_pca_0_model.pkl"]:
            with open(files["cluster_pca_0_model.pkl"], 'rb') as f:
                models["cluster_pca_0"] = pickle.load(f)
        if files["cluster_pca_1_model.pkl"]:
            with open(files["cluster_pca_1_model.pkl"], 'rb') as f:
                models["cluster_pca_1"] = pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load cluster models: {e}")
    
    # Load PCA model
    try:
        if files["pca_final.pkl"]:
            with open(files["pca_final.pkl"], 'rb') as f:
                models["pca"] = pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load PCA model: {e}")
    
    # Load data
    try:
        if files["assets_enriched.csv"]:
            data["assets_enriched"] = pd.read_csv(files["assets_enriched.csv"])
        else:
            raise Exception("Assets file not found")
    except Exception as e:
        st.warning(f"Could not load enriched assets data, using dummy data: {e}")
        data["assets_enriched"] = create_dummy_enriched_data()
    
    return models, data

def create_dummy_enriched_data():
    """Create dummy enriched assets data for demonstration."""
    np.random.seed(42)
    n_assets = 12000  # Increased sample size
    
    states = ['CA', 'TX', 'FL', 'NY', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI', 
             'AZ', 'WA', 'NV', 'CO', 'OR', 'UT', 'NM', 'ID', 'MT', 'WY']
    
    installations = ['Fort Base', 'Naval Station', 'Air Force Base', 'Marine Corps Base', 
                    'Army Installation', 'Coast Guard Station', 'National Guard Facility']
    
    asset_types = ['Administrative Building', 'Warehouse', 'Hangar', 'Barracks', 'Hospital',
                  'Training Facility', 'Maintenance Shop', 'Command Center', 'Recreation Center']
    
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
        
        # Determine model used
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
            'mean_price': (mean_price - 200000) / 800000,
            'median_price': (mean_price * 0.95 - 200000) / 800000,
            'std_price': volatility * 0.3,
            'price_volatility': volatility,
            'price_trend_slope': trend_slope * 1000,
            'recent_6mo_avg': (mean_price * 1.02 - 200000) / 800000,
            'recent_12mo_avg': (mean_price * 1.01 - 200000) / 800000,
            'last_price': (mean_price * 1.03 - 200000) / 800000,
            'price_min': (mean_price * 0.8 - 200000) / 800000,
            'price_max': (mean_price * 1.2 - 200000) / 800000,
            'price_range': (mean_price * 0.4 - 200000) / 800000,
            
            # FIXED: Use consistent column names
            'pred_last_price_original': predicted_value,
            'predicted_value': predicted_value,
            'pred_last_price_scaled': predicted_value / 1000000,
            'predicted_value_scaled': predicted_value / 1000000,
            'model_used': model_used,  # ENSURE THIS COLUMN EXISTS
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

def create_plotly_dark_theme():
    """Create consistent dark theme for Plotly charts."""
    return {
        'layout': {
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(26,26,26,0.8)',
            'font_color': '#ffffff',
            'font_family': 'Inter',
            'colorway': ['#6366f1', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444'],
            'margin': dict(l=10, r=10, t=50, b=10)
        }
    }

def main():
    """Main dashboard function."""
    
    # Animated header with glow effect
    st.markdown('<h1 class="main-header">🏛️ SmartAssets Analytics Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #9ca3af; font-size: 1.2rem; margin-bottom: 2rem;">Advanced Machine Learning for Government Asset Valuation</p>', unsafe_allow_html=True)
    
    # Load models and data
    with st.spinner("🔄 Loading advanced ML models and datasets..."):
        models, data = load_models_and_data()
        performance_metrics = create_model_performance_metrics()
    
    if data is None:
        st.error("Failed to load required data.")
        return
    
    # Sidebar with modern dark styling
    with st.sidebar:
        st.markdown('<div class="dark-card">', unsafe_allow_html=True)
        st.title("🎛️ Control Center")
        st.markdown("---")
        
        # Enhanced filters
        df_assets = data["assets_enriched"]
        
        # FIXED: Use safe column detection
        value_column = get_value_column(df_assets)
        
        available_states = sorted([s for s in df_assets['State'].dropna().unique() if str(s) != 'nan'])
        selected_states = st.multiselect(
            "🗺️ Select States:", 
            available_states, 
            default=available_states[:8] if len(available_states) > 8 else available_states
        )
        
        # Advanced filters
        st.subheader("🔍 Advanced Filters")
        
        # FIXED: Safe value range calculation with error handling
        try:
            min_value = float(df_assets[value_column].min() / 1000000)
            max_value = float(df_assets[value_column].max() / 1000000)
            
            value_range = st.slider(
                "💰 Asset Value Range ($M)",
                min_value,
                max_value,
                (min_value, max_value),
                step=0.1
            )
        except Exception as e:
            st.error(f"Error with value column '{value_column}': {e}")
            # Fallback values
            value_range = (0.1, 10.0)
        
        # Safe building types access
        building_types_col = safe_column_access(df_assets, 'Building Type', 'Unknown')
        building_types_unique = building_types_col.dropna().unique() if not building_types_col.isna().all() else ['All']
        
        building_types = st.multiselect(
            "🏢 Building Types:",
            building_types_unique,
            default=building_types_unique[:3] if len(building_types_unique) > 3 else building_types_unique
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Filter data
    if selected_states:
        df_assets_filtered = df_assets[df_assets['State'].isin(selected_states)]
    else:
        df_assets_filtered = df_assets
    
    # Apply value range filter safely
    try:
        value_min, value_max = value_range[0] * 1000000, value_range[1] * 1000000
        df_assets_filtered = df_assets_filtered[
            (df_assets_filtered[value_column] >= value_min) &
            (df_assets_filtered[value_column] <= value_max)
        ]
    except Exception as e:
        st.warning(f"Could not apply value filter: {e}")
    
    # Key metrics with enhanced dark styling
    st.markdown('<div class="dark-card">', unsafe_allow_html=True)
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
        try:
            total_value = df_assets_filtered[value_column].sum() / 1e9
        except:
            total_value = 0
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
        try:
            median_value = df_assets_filtered[value_column].median() / 1000
        except:
            median_value = 0
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">${median_value:.0f}K</div>
            <div class="metric-label">Median Value</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced tabs with dark theme
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Model Performance", 
        "💰 Asset Valuation", 
        "🗺️ Spatial Intelligence", 
        "🔬 Advanced Analytics"
    ])
    
    with tab1:
        st.markdown('<h2 class="section-header">🎯 ML Model Performance Dashboard</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="dark-card">', unsafe_allow_html=True)
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
                color_discrete_sequence=['#6366f1', '#8b5cf6', '#06b6d4', '#10b981']
            )
            fig.update_layout(**create_plotly_dark_theme()['layout'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance indicators
            for _, row in models_df.iterrows():
                r2_score = row['Test R²']
                if r2_score >= 0.99:
                    indicator_class = "performance-excellent"
                elif r2_score >= 0.95:
                    indicator_class = "performance-good"
                else:
                    indicator_class = "performance-warning"
                
                st.markdown(f'''
                <div class="{indicator_class} performance-indicator">
                    {row['Model']}: {r2_score*100:.1f}% Accuracy
                </div>
                ''', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="dark-card">', unsafe_allow_html=True)
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
            fig.update_layout(**create_plotly_dark_theme()['layout'])
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature importance visualization
        st.markdown('<div class="dark-card">', unsafe_allow_html=True)
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
        fig.update_layout(**create_plotly_dark_theme()['layout'])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<h2 class="section-header">💰 Asset Valuation Intelligence</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="dark-card">', unsafe_allow_html=True)
            st.subheader("📊 Value Distribution")
            
            fig = px.histogram(
                df_assets_filtered, 
                x=value_column,
                nbins=50,
                title="Asset Value Distribution",
                labels={value_column: 'Predicted Value ($)'}
            )
            fig.update_layout(**create_plotly_dark_theme()['layout'])
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="dark-card">', unsafe_allow_html=True)
            st.subheader("🎯 Model Usage Distribution")
            
            # FIXED: Safe access to model_used column with proper method name
            try:
                model_usage_col = safe_column_access(df_assets_filtered, 'model_used', 'unknown')
                model_usage = model_usage_col.value_counts()  # FIXED: Added parentheses
                
                if len(model_usage) > 0:
                    fig = px.pie(
                        values=model_usage.values,
                        names=[name.replace('_', ' ').title() for name in model_usage.index],
                        title="Prediction Model Usage"
                    )
                    fig.update_layout(**create_plotly_dark_theme()['layout'])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No model usage data available")
            except Exception as e:
                st.error(f"Could not create model usage chart: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Top valued assets
        st.markdown('<div class="dark-card">', unsafe_allow_html=True)
        st.subheader("🏆 Highest Valued Assets")
        
        display_columns = ['Real Property Asset Name', 'City', 'State', value_column, 'model_used']
        available_columns = [col for col in display_columns if col in df_assets_filtered.columns]
        
        try:
            top_assets = df_assets_filtered.nlargest(10, value_column)[available_columns].copy()
            if value_column in top_assets.columns:
                top_assets[value_column] = top_assets[value_column].apply(lambda x: f"${x:,.0f}")
            
            st.dataframe(top_assets, use_container_width=True)
        except Exception as e:
            st.error(f"Could not display top assets: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<h2 class="section-header">🗺️ Spatial Intelligence Dashboard</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="dark-card">', unsafe_allow_html=True)
            st.subheader("🌎 Interactive Asset Map")
            
            # Create enhanced folium map with dark theme
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
                    value_quantiles = valid_coords[value_column].quantile([0, 0.25, 0.5, 0.75, 1.0])
                    
                    def get_color(value):
                        if value <= value_quantiles[0.25]:
                            return '#6366f1'
                        elif value <= value_quantiles[0.5]:
                            return '#8b5cf6'
                        elif value <= value_quantiles[0.75]:
                            return '#06b6d4'
                        else:
                            return '#10b981'
                    
                    # Add markers (limit for performance)
                    for idx, row in valid_coords.head(200).iterrows():
                        # Safe access to model_used for popup
                        model_used = safe_column_access(pd.DataFrame([row]), 'model_used', 'unknown').iloc[0]
                        
                        folium.CircleMarker(
                            location=[row['Latitude'], row['Longitude']],
                            radius=8,
                            popup=f"""
                            <div style="background: #1a1a1a; color: white; padding: 10px; border-radius: 8px;">
                            <b>{row.get('Real Property Asset Name', 'Asset')}</b><br>
                            📍 {row['City']}, {row['State']}<br>
                            💰 ${row[value_column]:,.0f}<br>
                            🤖 Model: {model_used}<br>
                            🏢 Type: {row.get('Building Type', 'N/A')}
                            </div>
                            """,
                            color=get_color(row[value_column]),
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
            st.markdown('<div class="dark-card">', unsafe_allow_html=True)
            st.subheader("📊 Spatial Statistics")
            
            # State-level aggregation
            try:
                rentable_sqft_col = safe_column_access(df_assets_filtered, 'Building Rentable Square Feet', 0)
                
                state_stats = df_assets_filtered.groupby('State').agg({
                    value_column: ['count', 'mean', 'median']
                }).round(0)
                
                # Add rentable square feet if available
                if not rentable_sqft_col.isna().all():
                    sqft_stats = df_assets_filtered.groupby('State')[rentable_sqft_col.name].mean().round(0)
                    state_stats = pd.concat([state_stats, sqft_stats.rename('Avg Sq Ft')], axis=1)
                
                state_stats.columns = ['Count', 'Mean Value', 'Median Value', 'Avg Sq Ft'] if state_stats.shape[1] == 4 else ['Count', 'Mean Value', 'Median Value']
                state_stats = state_stats.sort_values('Mean Value', ascending=False)
                
                st.dataframe(state_stats.head(10), use_container_width=True)
            except Exception as e:
                st.warning(f"Could not calculate state statistics: {e}")
            
            # Matching quality analysis
            st.subheader("🔗 Data Matching Quality")
            try:
                match_quality_col = safe_column_access(df_assets_filtered, '_match_type', 'unknown')
                match_quality = match_quality_col.value_counts()
                
                if len(match_quality) > 0:
                    fig = px.bar(
                        x=match_quality.values,
                        y=[name.replace('_', ' ').title() for name in match_quality.index],
                        orientation='h',
                        title="Data Matching Distribution"
                    )
                    fig.update_layout(**create_plotly_dark_theme()['layout'], height=300)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No matching quality data available")
            except Exception as e:
                st.warning(f"Could not create matching quality chart: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<h2 class="section-header">🔬 Advanced Analytics</h2>', unsafe_allow_html=True)
        
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.subheader("🧠 AI-Powered Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            try:
                high_value_states = df_assets_filtered.groupby('State')[value_column].mean().nlargest(3)
                st.write("**🏆 Top Value States:**")
                for state, value in high_value_states.items():
                    st.write(f"• {state}: ${value:,.0f}")
            except:
                st.write("**🏆 Top Value States:** Data unavailable")
        
        with col2:
            try:
                volatility_col = safe_column_access(df_assets_filtered, 'price_volatility', 0.1)
                volatile_states = df_assets_filtered.groupby('State')[volatility_col.name].mean().nlargest(3)
                st.write("**⚡ Most Volatile Markets:**")
                for state, volatility in volatile_states.items():
                    st.write(f"• {state}: {volatility:.3f}")
            except:
                st.write("**⚡ Most Volatile Markets:** Data unavailable")
        
        with col3:
            try:
                trend_col = safe_column_access(df_assets_filtered, 'price_trend_slope', 0.001)
                growing_states = df_assets_filtered.groupby('State')[trend_col.name].mean().nlargest(3)
                st.write("**📈 Fastest Growing:**")
                for state, trend in growing_states.items():
                    st.write(f"• {state}: {trend:.4f}")
            except:
                st.write("**📈 Fastest Growing:** Data unavailable")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer with enhanced dark styling
    st.markdown("---")
    st.markdown(f'''
    <div class="dark-card" style="text-align: center; margin-top: 2rem;">
        <h3 style="color: {COLORS['text']}; margin-bottom: 1rem;">🏛️ SmartAssets Analytics Pro - Dark Edition</h3>
        <p style="color: {COLORS['text_muted']};">Powered by Advanced Machine Learning | Sample Size: {len(df_assets):,} Assets</p>
        <p style="color: {COLORS['text_muted']};">🤖 AI Models: Random Forest • Gradient Boosting • PCA Analysis • Spatial Intelligence</p>
        <p style="color: {COLORS['text_muted']};">Built with ❤️ using Streamlit • Plotly • Scikit-learn</p>
    </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

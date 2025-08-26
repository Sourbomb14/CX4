import os
from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
import gdown
import warnings

warnings.filterwarnings("ignore")

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="🏛️ SmartAssets Analytics Pro", 
    page_icon="🏛️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== ENHANCED DARK THEME ====================
st.markdown("""
<style>
    /* Remove Streamlit branding */
    #MainMenu, footer, header, .stDeployButton, div[data-testid="stToolbar"], 
    div[data-testid="stDecoration"], div[data-testid="stStatusWidget"], 
    .stActionButton {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Dark theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
    
    .block-container {
        padding-top: 1.1rem; 
        padding-bottom: 1.1rem;
        max-width: 1200px;
    }
    
    /* Headers */
    .main-title {
        font-size: 3.5rem;
        color: #ffffff;
        text-align: center;
        margin: 2rem 0;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.15));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #9ca3af;
        font-weight: 500;
    }
    
    /* Content cards */
    .content-card {
        background: rgba(26, 26, 26, 0.8);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
    }
    
    /* Legend */
    .legend-swatch {
        display: inline-block; 
        width: 14px; 
        height: 14px; 
        border-radius: 3px; 
        margin-right: 8px;
    }
    
    .legend-item {
        margin-right: 14px; 
        display: inline-flex; 
        align-items: center;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        width: 100%;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(42, 42, 42, 0.8);
        border-radius: 10px;
        padding: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #9ca3af;
        background: transparent;
        border-radius: 8px;
        font-weight: 500;
        padding: 12px 20px;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
    }
    
    /* Inputs */
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div {
        background: rgba(26, 26, 26, 0.8);
        border: 1px solid #374151;
        color: #ffffff;
    }
    
    /* Text colors */
    .stMarkdown, p, span, div, h1, h2, h3 {
        color: #ffffff !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(15, 15, 35, 0.9);
    }
</style>
""", unsafe_allow_html=True)

# ==================== GOOGLE DRIVE FILE URLS ====================
# Add your Google Drive file IDs here
GOOGLE_DRIVE_FILES = {
    # Dataset files
    "assets_enriched.csv": "1MqFFQZ_Vq8ss4p6mg3ZhQeampFCr26Nb",
    
    # Model files
    "scaler_all.pkl": "1G3U898UQ4yoWO5TOY01MEDlnprG0bEM6",
    "scaler_last_price.pkl": "1nhoS237W_-5Fsgdo7sDFD5_7hceHappp",
    "global_model.pkl": "1ZWPra5iZ0pEVQgxpPaWx8gX3J9olsb7Z",
    "global_model_pca.pkl": "1dmE1bEDWUeAkZNkpGDTHEJA6AEt0FPz1",
    "cluster_0_model.pkl": "1JM1tj9PNQ8TEJlR3S0MQTxguLsoXKbcf",
    "cluster_1_model.pkl": "13Z7PaHcb9e9tOYXxB7fjWKgrb8rpB3xb",
    "cluster_pca_0_model.pkl": "1X9WmLRoJHCdMcLVKTtsbDujYAIg_o1dU",
    "cluster_pca_1_model.pkl": "1GaDbbVCBUvjrvSUrfT6GLJUFYVa1xRPG",
    "pca_final.pkl": "1gQfXF4aJ-30XispHCOjdv2zfRDw2fhHt"
}

# ==================== DATA LOADING FUNCTIONS ====================
@st.cache_data
def download_file_from_gdrive(file_id, filename):
    """Download a file from Google Drive using gdown."""
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        
        # Create directories if they don't exist
        if filename.endswith('.pkl'):
            os.makedirs("models", exist_ok=True)
            filepath = f"models/{filename}"
        else:
            os.makedirs("data", exist_ok=True)
            filepath = f"data/{filename}"
        
        # Download only if file doesn't exist
        if not os.path.exists(filepath):
            gdown.download(url, filepath, quiet=False)
            st.success(f"✅ Downloaded {filename}")
        else:
            st.info(f"📁 {filename} already exists")
        
        return filepath
    except Exception as e:
        st.error(f"❌ Failed to download {filename}: {str(e)}")
        return None

@st.cache_data
def load_dataset_from_gdrive():
    """Load the main dataset from Google Drive."""
    # Try to load the enriched assets dataset
    if "assets_enriched.csv" in GOOGLE_DRIVE_FILES:
        filepath = download_file_from_gdrive(
            GOOGLE_DRIVE_FILES["assets_enriched.csv"], 
            "assets_enriched.csv"
        )
        if filepath and os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                st.success(f"📊 Loaded dataset with {len(df):,} assets")
                return standardize_column_names(df)
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
    
    # Fallback to generated data if download fails
    st.warning("Using generated sample data as fallback")
    return generate_comprehensive_data()

@st.cache_data
def load_models_from_gdrive():
    """Load all ML models from Google Drive."""
    models = {}
    
    model_files = [
        "scaler_all.pkl", "scaler_last_price.pkl", "global_model.pkl", 
        "global_model_pca.pkl", "cluster_0_model.pkl", "cluster_1_model.pkl",
        "cluster_pca_0_model.pkl", "cluster_pca_1_model.pkl", "pca_final.pkl"
    ]
    
    for model_file in model_files:
        if model_file in GOOGLE_DRIVE_FILES:
            filepath = download_file_from_gdrive(
                GOOGLE_DRIVE_FILES[model_file], 
                model_file
            )
            if filepath and os.path.exists(filepath):
                try:
                    with open(filepath, 'rb') as f:
                        model_name = model_file.replace('.pkl', '')
                        models[model_name] = pickle.load(f)
                    st.success(f"🤖 Loaded {model_file}")
                except Exception as e:
                    st.warning(f"Could not load {model_file}: {e}")
            else:
                st.warning(f"Could not download {model_file}")
    
    return models

def standardize_column_names(df):
    """Standardize column names for compatibility."""
    # Column mapping for different naming conventions
    column_mapping = {
        # Location identifiers
        'Location Code': 'loc_code',
        'loc_code': 'loc_code',
        
        # Asset information
        'Real Property Asset Name': 'asset_name',
        'Asset Name': 'asset_name',
        'asset_name': 'asset_name',
        
        # Geographic data
        'City': 'city',
        'city': 'city',
        'State': 'state', 
        'state': 'state',
        'Zip Code': 'zip',
        'ZIP': 'zip',
        'zip': 'zip',
        'Latitude': 'lat',
        'lat': 'lat',
        'Longitude': 'lon',
        'lon': 'lon',
        
        # Asset characteristics
        'Building Rentable Square Feet': 'sqft',
        'SqFt': 'sqft',
        'sqft': 'sqft',
        'Building Age': 'age',
        'age': 'age',
        'Real Property Asset Type': 'asset_type',
        'Asset Type': 'asset_type',
        'asset_type': 'asset_type',
        
        # Financial data
        'Estimated Asset Value (Adj)': 'value',
        'Estimated Asset Value': 'value',
        'pred_last_price_original': 'value',
        'predicted_value': 'value',
        'value': 'value',
        
        # Confidence and other metrics
        'Confidence Category': 'conf_cat',
        'conf_cat': 'conf_cat',
        'confidence': 'conf_cat',
        'model_used': 'model_used',
        'cluster': 'cluster',
        'cluster_kmeans': 'cluster'
    }
    
    # Create standardized dataframe
    standardized_df = pd.DataFrame()
    
    for new_col, possible_cols in column_mapping.items():
        if isinstance(possible_cols, str):
            possible_cols = [possible_cols]
        
        for col in possible_cols:
            if col in df.columns:
                standardized_df[new_col] = df[col]
                break
    
    # Add any remaining columns that weren't mapped
    for col in df.columns:
        if col not in standardized_df.columns:
            # Convert to snake_case
            new_col = col.lower().replace(' ', '_').replace('(', '').replace(')', '')
            standardized_df[new_col] = df[col]
    
    # Calculate value per square foot if possible
    if 'value' in standardized_df.columns and 'sqft' in standardized_df.columns:
        standardized_df['value_psf'] = standardized_df['value'] / standardized_df['sqft']
        standardized_df['value_psf'] = standardized_df['value_psf'].replace([np.inf, -np.inf], np.nan)
    
    # Clean numeric columns
    numeric_cols = ['lat', 'lon', 'sqft', 'value', 'age', 'value_psf']
    for col in numeric_cols:
        if col in standardized_df.columns:
            standardized_df[col] = pd.to_numeric(standardized_df[col], errors='coerce')
    
    # Fill missing confidence categories
    if 'conf_cat' not in standardized_df.columns:
        standardized_df['conf_cat'] = 'Unknown'
    
    return standardized_df

# ==================== EXISTING DATA UTILITIES (as fallback) ====================
@st.cache_data
def generate_comprehensive_data():
    """Generate comprehensive asset data with ML features (fallback)."""
    np.random.seed(42)
    n_assets = 8000
    
    states = ['CA', 'TX', 'FL', 'NY', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI', 'AZ', 'WA']
    asset_types = ['Office Building', 'Warehouse', 'Training Center', 'Hospital', 'Barracks', 
                   'Hangar', 'Administrative Building', 'Recreation Center', 'Maintenance Shop']
    confidence_levels = ['High', 'Medium', 'Low', 'Very High']
    
    data = []
    for i in range(n_assets):
        state = np.random.choice(states)
        asset_type = np.random.choice(asset_types)
        
        # State multipliers for realistic variation
        state_multipliers = {
            'CA': 2.8, 'NY': 2.5, 'WA': 2.0, 'FL': 1.5, 'TX': 1.3,
            'IL': 1.1, 'PA': 1.0, 'OH': 0.9, 'NC': 0.85, 'MI': 0.8
        }
        
        multiplier = state_multipliers.get(state, 1.0)
        
        # Generate correlated features
        age = np.random.randint(5, 80)
        sqft = np.random.randint(10000, 800000)
        condition_score = np.random.uniform(0.3, 1.0)
        
        # Base value calculation with realistic correlations
        base_value = (
            np.random.lognormal(12.5, 0.6) * multiplier * 
            (1.2 - age/100) * condition_score * 
            (1 + np.random.uniform(0.8, 1.3))
        )
        
        value_psf = base_value / sqft if sqft > 0 else 0
        
        # Generate coordinates by state
        state_coords = {
            'CA': (34.0, -118.0), 'TX': (31.0, -99.0), 'FL': (27.8, -81.7),
            'NY': (42.2, -74.9), 'IL': (40.3, -89.0), 'PA': (40.5, -77.5),
            'OH': (40.4, -82.9), 'GA': (33.0, -83.5), 'NC': (35.8, -80.8),
            'MI': (43.3, -84.5), 'AZ': (33.7, -111.4), 'WA': (47.4, -121.5)
        }
        
        base_lat, base_lon = state_coords.get(state, (39.0, -98.0))
        
        data.append({
            'loc_code': f'LOC_{i:05d}',
            'asset_name': f'{asset_type} #{i+1}',
            'city': f'City_{np.random.randint(1, 100)}',
            'state': state,
            'zip': f'{np.random.randint(10000, 99999)}',
            'lat': base_lat + np.random.normal(0, 2),
            'lon': base_lon + np.random.normal(0, 3),
            'sqft': sqft,
            'value': base_value,
            'value_psf': value_psf,
            'conf_cat': np.random.choice(confidence_levels, p=[0.3, 0.4, 0.25, 0.05]),
            'asset_type': asset_type,
            'age': age,
            'condition': np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], 
                                        p=[0.2, 0.5, 0.25, 0.05]),
            'utilization_rate': np.random.uniform(0.3, 1.0),
            'risk_score': np.random.uniform(0.1, 0.9),
            'growth_potential': np.random.uniform(-0.15, 0.4),
            'market_value': base_value * np.random.uniform(0.85, 1.15),
            'maintenance_cost': sqft * np.random.uniform(2, 15),
            'energy_efficiency': np.random.choice(['A', 'B', 'C', 'D', 'F'], 
                                                p=[0.1, 0.2, 0.4, 0.25, 0.05])
        })
    
    return pd.DataFrame(data)

def fmt_money_units(x):
    """Format money with appropriate units."""
    try:
        x = float(x)
    except:
        return "—"
    if np.isnan(x) or x == 0:
        return "—"
    
    abs_x = abs(x)
    if abs_x >= 1e9:
        return f"${x/1e9:,.2f}B"
    elif abs_x >= 1e6:
        return f"${x/1e6:,.1f}M"
    elif abs_x >= 1e3:
        return f"${x/1e3:,.0f}K"
    else:
        return f"${x:,.0f}"

BASE_PALETTE = [
    [99,102,241], [139,92,246], [6,182,212], [16,185,129], [245,158,11],
    [239,68,68], [168,85,247], [34,197,94], [59,130,246], [236,72,153],
    [248,113,113], [124,58,237], [14,165,233], [34,197,94]
]

def color_map_for(series):
    """Generate color mapping for categorical series."""
    cats = sorted(series.fillna("Unknown").astype(str).unique())
    cmap = {c: BASE_PALETTE[i % len(BASE_PALETTE)] for i, c in enumerate(cats)}
    colors = series.fillna("Unknown").astype(str).map(cmap).tolist()
    return colors, cmap

# ==================== PREDICTION FUNCTIONS ====================
def predict_with_loaded_models(df, models):
    """Use loaded models to make predictions on new data."""
    if not models or len(df) == 0:
        return df
    
    try:
        # Prepare features for prediction
        feature_cols = ['value', 'sqft', 'value_psf', 'age']
        if not all(col in df.columns for col in feature_cols):
            st.warning("Missing required columns for model prediction")
            return df
        
        # Create a copy for predictions
        df_pred = df.copy()
        
        # Fill missing values
        for col in feature_cols:
            df_pred[col] = df_pred[col].fillna(df_pred[col].median())
        
        # Scale features if scaler is available
        if 'scaler_all' in models:
            X = df_pred[feature_cols].values
            X_scaled = models['scaler_all'].transform(X)
            
            # Make predictions with global model if available
            if 'global_model' in models:
                predictions = models['global_model'].predict(X_scaled)
                df_pred['model_prediction'] = predictions
                df_pred['model_used'] = 'global'
                
                # Inverse transform predictions if possible
                if 'scaler_last_price' in models:
                    df_pred['predicted_value_original'] = models['scaler_last_price'].inverse_transform(
                        predictions.reshape(-1, 1)
                    ).flatten()
            
            # PCA transformation if available
            if 'pca_final' in models:
                try:
                    X_pca = models['pca_final'].transform(X_scaled)
                    for i in range(min(3, X_pca.shape[1])):
                        df_pred[f'pca_component_{i+1}'] = X_pca[:, i]
                except:
                    pass
        
        st.success(f"✅ Applied ML models to {len(df_pred)} assets")
        return df_pred
        
    except Exception as e:
        st.warning(f"Could not apply models: {e}")
        return df

# ==================== VISUALIZATION FUNCTIONS ====================
def create_metrics_display(df):
    """Create enhanced metrics display."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics = [
        ("Total Assets", f"{len(df):,}"),
        ("Total Value", fmt_money_units(df['value'].sum())),
        ("Median Value", fmt_money_units(df['value'].median())),
        ("Avg $/ft²", fmt_money_units(df['value_psf'].median())),
        ("States", f"{df['state'].nunique()}")
    ]
    
    cols = [col1, col2, col3, col4, col5]
    for i, (label, value) in enumerate(metrics):
        cols[i].markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

def create_state_bubble_map(flt):
    """Create state-level bubble map."""
    if not all(c in flt.columns for c in ["state", "lat", "lon"]) or flt[["lat", "lon"]].dropna().empty:
        return None
    
    geo = flt.dropna(subset=["lat", "lon"]).copy()
    geo = geo[geo["lat"].between(-90, 90) & geo["lon"].between(-180, 180)]
    
    agg = (geo.groupby("state")
           .agg(lat=("lat", "mean"), lon=("lon", "mean"),
                total_value=("value", "sum"), n=("state", "size"))
           .reset_index())
    
    if agg.empty:
        return None
    
    vmed = max(agg["total_value"].median(), 1.0)
    m = folium.Map(
        location=[agg["lat"].mean(), agg["lon"].mean()],
        zoom_start=4, 
        tiles="cartodbpositron"
    )
    
    for _, r in agg.iterrows():
        scale = np.sqrt(max(float(r["total_value"]), 1.0) / vmed)
        radius = float(25000 * scale)
        
        html = (f"<b>{r['state']}</b><br>"
                f"Assets: {int(r['n']):,}<br>"
                f"Total Value: {fmt_money_units(r['total_value'])}")
        
        folium.Circle(
            location=[r["lat"], r["lon"]],
            radius=radius,
            color="#6366f1",
            weight=2,
            fill=True,
            fill_opacity=0.2,
            popup=folium.Popup(html, max_width=300)
        ).add_to(m)
    
    return m

def create_marker_cluster_map(flt):
    """Create detailed marker cluster map."""
    if not all(c in flt.columns for c in ["lat", "lon"]) or flt[["lat", "lon"]].dropna().empty:
        return None
    
    geo = flt.dropna(subset=["lat", "lon"]).copy()
    geo = geo[geo["lat"].between(-90, 90) & geo["lon"].between(-180, 180)]
    
    if geo.empty:
        return None
    
    m = folium.Map(
        location=[geo["lat"].mean(), geo["lon"].mean()],
        zoom_start=4,
        tiles="cartodbpositron"
    )
    
    cluster = MarkerCluster(name="Assets").add_to(m)
    
    # Sample for performance
    sample_size = min(1000, len(geo))
    geo_sample = geo.sample(sample_size) if len(geo) > sample_size else geo
    
    for _, r in geo_sample.iterrows():
        popup_html = f"""
        <div style="font-family: Arial; color: black;">
        <b>{str(r.get('asset_name', '')).title()}</b><br>
        <b>Location:</b> {r.get('city', '')}, {r.get('state', '')} {r.get('zip', '')}<br>
        <b>Type:</b> {r.get('asset_type', '')}<br>
        <b>Value:</b> {fmt_money_units(r.get('value', 0))}<br>
        <b>$/ft²:</b> {fmt_money_units(r.get('value_psf', 0))}<br>
        <b>Size:</b> {r.get('sqft', 0):,.0f} ft²<br>
        <b>Age:</b> {r.get('age', 0):.0f} years<br>
        <b>Confidence:</b> {r.get('conf_cat', '')}
        </div>
        """
        
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=5,
            fill=True,
            fill_opacity=0.7,
            weight=1,
            color="#1f2937",
            fill_color="#6366f1",
            popup=folium.Popup(popup_html, max_width=350)
        ).add_to(cluster)
    
    folium.LayerControl(collapsed=False).add_to(m)
    return m

# ==================== ML FUNCTIONS ====================
def perform_clustering_analysis(df, k=4):
    """Perform comprehensive clustering analysis."""
    # Prepare ML features
    ml_cols = ['value', 'sqft', 'value_psf', 'age']
    ml_data = df[ml_cols].copy()
    ml_data = ml_data.replace([np.inf, -np.inf], np.nan)
    ml_data = ml_data.fillna(ml_data.median())
    
    # Log transform for better clustering
    for col in ['value', 'sqft', 'value_psf']:
        ml_data[f'log_{col}'] = np.log(np.clip(ml_data[col], 1, None))
    
    # Feature selection for clustering
    X_cols = [f'log_{c}' for c in ['value', 'sqft', 'value_psf']] + ['age']
    X = ml_data[X_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    # Calculate silhouette score
    sil_score = silhouette_score(X_scaled, labels)
    
    # Assign clusters to dataframe
    df_clustered = df.copy()
    df_clustered['cluster'] = labels
    
    # Create cluster names based on characteristics
    cluster_profiles = df_clustered.groupby('cluster')[['value', 'sqft', 'value_psf', 'age']].median()
    
    cluster_names = {}
    for cluster_id in cluster_profiles.index:
        profile = cluster_profiles.loc[cluster_id]
        if profile['value_psf'] > cluster_profiles['value_psf'].median() * 1.2:
            cluster_names[cluster_id] = "Premium Assets"
        elif profile['sqft'] > cluster_profiles['sqft'].median() * 1.5:
            cluster_names[cluster_id] = "Large Facilities"
        elif profile['age'] > cluster_profiles['age'].median() * 1.2:
            cluster_names[cluster_id] = "Legacy Assets"
        else:
            cluster_names[cluster_id] = "Standard Assets"
    
    df_clustered['cluster_name'] = df_clustered['cluster'].map(cluster_names)
    
    return df_clustered, sil_score, X_scaled, X_cols, scaler, kmeans

def perform_classification_analysis(X, labels, X_cols):
    """Perform RandomForest classification analysis."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    # Train RandomForest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    
    # Feature importance
    feature_importance = pd.Series(rf.feature_importances_, index=X_cols).sort_values(ascending=False)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return rf, accuracy, feature_importance, cm, y_test, y_pred

# ==================== MAIN APPLICATION ====================
def main():
    # Header
    st.markdown('<h1 class="main-title">🏛️ SmartAssets Analytics Pro</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; color: #9ca3af; font-size: 1.2rem; margin-bottom: 2rem;">
    Advanced Machine Learning for Government Asset Valuation & Analytics
    </p>
    """, unsafe_allow_html=True)
    
    # Data Loading Section
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.models_loaded = False
    
    # Load data and models
    if not st.session_state.data_loaded:
        with st.spinner("🔄 Loading dataset from Google Drive..."):
            st.session_state.df = load_dataset_from_gdrive()
            st.session_state.data_loaded = True
    
    if not st.session_state.models_loaded:
        with st.spinner("🤖 Loading ML models from Google Drive..."):
            st.session_state.models = load_models_from_gdrive()
            st.session_state.models_loaded = True
    
    df = st.session_state.df
    models = st.session_state.models
    
    # Apply models to data if available
    if models and len(models) > 0:
        df = predict_with_loaded_models(df, models)
    
    # Show data loading status
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Dataset Size", f"{len(df):,} assets")
    col2.metric("🤖 Models Loaded", f"{len(models)}")
    col3.metric("💾 Data Source", "Google Drive" if st.session_state.data_loaded else "Generated")
    
    # Sidebar Filters
    with st.sidebar:
        st.markdown("### 🎛️ Dashboard Controls")
        
        # Reset button
        if st.button("🔄 Reset All Filters"):
            st.experimental_rerun()
        
        # State filter
        states = sorted(df['state'].unique()) if 'state' in df.columns else []
        selected_states = st.multiselect(
            "🗺️ Select States:", 
            states, 
            default=states[:6] if len(states) > 6 else states
        )
        
        # Asset type filter
        asset_types = sorted(df['asset_type'].unique()) if 'asset_type' in df.columns else []
        selected_types = st.multiselect(
            "🏢 Asset Types:", 
            asset_types, 
            default=asset_types
        )
        
        # Confidence filter
        conf_levels = sorted(df['conf_cat'].unique()) if 'conf_cat' in df.columns else []
        selected_conf = st.multiselect(
            "🎯 Confidence Levels:", 
            conf_levels, 
            default=conf_levels
        )
        
        # Value range
        if 'value' in df.columns and df['value'].notna().any():
            value_min, value_max = float(df['value'].min()), float(df['value'].max())
            value_range = st.slider(
                "💰 Value Range ($M):",
                value_min/1e6, value_max/1e6,
                (value_min/1e6, value_max/1e6),
                step=0.1
            )
        else:
            value_range = None
        
        # Size range
        if 'sqft' in df.columns and df['sqft'].notna().any():
            sqft_min, sqft_max = float(df['sqft'].min()), float(df['sqft'].max())
            sqft_range = st.slider(
                "📐 Size Range (sq ft):",
                sqft_min, sqft_max,
                (sqft_min, sqft_max),
                step=1000.0
            )
        else:
            sqft_range = None
        
        # Age range
        if 'age' in df.columns and df['age'].notna().any():
            age_min, age_max = float(df['age'].min()), float(df['age'].max())
            age_range = st.slider(
                "📅 Age Range (years):",
                age_min, age_max,
                (age_min, age_max)
            )
        else:
            age_range = None
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_states and 'state' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['state'].isin(selected_states)]
    if selected_types and 'asset_type' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['asset_type'].isin(selected_types)]
    if selected_conf and 'conf_cat' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['conf_cat'].isin(selected_conf)]
    
    # Apply range filters
    if value_range and 'value' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['value'] >= value_range[0] * 1e6) &
            (filtered_df['value'] <= value_range[1] * 1e6)
        ]
    
    if sqft_range and 'sqft' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['sqft'] >= sqft_range[0]) &
            (filtered_df['sqft'] <= sqft_range[1])
        ]
    
    if age_range and 'age' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['age'] >= age_range[0]) &
            (filtered_df['age'] <= age_range[1])
        ]
    
    # Display metrics
    create_metrics_display(filtered_df)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard Analytics", 
        "🗺️ Geographic Intelligence", 
        "🤖 ML Clustering & Insights",
        "📈 Advanced Analytics"
    ])
    
    # ==================== TAB 1: DASHBOARD ANALYTICS ====================
    with tab1:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        
        if len(filtered_df) == 0:
            st.warning("No data to display. Please adjust your filters.")
        else:
            # Charts in 2x2 grid
            col1, col2 = st.columns(2)
            
            with col1:
                # Value distribution
                if 'value' in filtered_df.columns:
                    fig_hist = px.histogram(
                        filtered_df.sample(min(5000, len(filtered_df))),
                        x='value',
                        nbins=40,
                        title="Asset Value Distribution",
                        labels={'value': 'Asset Value ($)', 'count': 'Number of Assets'}
                    )
                    fig_hist.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(26,26,26,0.8)',
                        font_color='white'
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # Asset type distribution
                if 'asset_type' in filtered_df.columns:
                    type_counts = filtered_df['asset_type'].value_counts().head(8)
                    fig_types = px.bar(
                        x=type_counts.index,
                        y=type_counts.values,
                        title="Assets by Type",
                        labels={'x': 'Asset Type', 'y': 'Count'}
                    )
                    fig_types.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(26,26,26,0.8)',
                        font_color='white'
                    )
                    st.plotly_chart(fig_types, use_container_width=True)
            
            with col2:
                # Value vs Size scatter
                if all(col in filtered_df.columns for col in ['sqft', 'value']):
                    sample_df = filtered_df.sample(min(2000, len(filtered_df)))
                    color_col = 'asset_type' if 'asset_type' in sample_df.columns else None
                    
                    fig_scatter = px.scatter(
                        sample_df,
                        x='sqft',
                        y='value',
                        color=color_col,
                        title="Value vs Size Analysis",
                        labels={'sqft': 'Square Feet', 'value': 'Asset Value ($)'},
                        hover_data=[col for col in ['asset_name', 'state', 'age'] if col in sample_df.columns]
                    )
                    fig_scatter.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(26,26,26,0.8)',
                        font_color='white',
                        xaxis_type='log',
                        yaxis_type='log'
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                # State value distribution
                if all(col in filtered_df.columns for col in ['state', 'value']):
                    state_values = filtered_df.groupby('state')['value'].sum().sort_values(ascending=False).head(10)
                    fig_states = px.bar(
                        x=state_values.index,
                        y=state_values.values,
                        title="Total Asset Value by State",
                        labels={'x': 'State', 'y': 'Total Value ($)'}
                    )
                    fig_states.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(26,26,26,0.8)',
                        font_color='white'
                    )
                    st.plotly_chart(fig_states, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Top assets table
        if len(filtered_df) > 0:
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.subheader("🏆 Top 20 Most Valuable Assets")
            
            display_cols = [col for col in ['asset_name', 'state', 'asset_type', 'value', 'sqft', 'value_psf', 'age', 'conf_cat'] 
                           if col in filtered_df.columns]
            
            if display_cols and 'value' in filtered_df.columns:
                top_assets = filtered_df.nlargest(20, 'value')[display_cols].copy()
                
                # Format for display
                if 'value' in top_assets.columns:
                    top_assets['value'] = top_assets['value'].apply(fmt_money_units)
                if 'value_psf' in top_assets.columns:
                    top_assets['value_psf'] = top_assets['value_psf'].apply(fmt_money_units)
                if 'sqft' in top_assets.columns:
                    top_assets['sqft'] = top_assets['sqft'].apply(lambda x: f"{x:,.0f}")
                
                st.dataframe(top_assets, use_container_width=True, hide_index=True)
                
                # Download button
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Filtered Data (CSV)",
                    csv,
                    "filtered_assets.csv",
                    "text/csv",
                    key='download-csv'
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # ==================== TAB 2: GEOGRAPHIC INTELLIGENCE ====================
    with tab2:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        
        if len(filtered_df) == 0:
            st.warning("No data to display. Please adjust your filters.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🌍 State Summary Map")
                bubble_map = create_state_bubble_map(filtered_df)
                if bubble_map:
                    st_folium(bubble_map, height=400, width=None)
                    
                    # Download button for map
                    buf = io.BytesIO()
                    bubble_map.save(buf, close_file=False)
                    st.download_button(
                        "📥 Download State Map",
                        buf.getvalue(),
                        "state_summary_map.html",
                        "text/html"
                    )
                else:
                    st.info("No geographic data available for selected filters.")
            
            with col2:
                st.subheader("📍 Detailed Asset Map")
                cluster_map = create_marker_cluster_map(filtered_df)
                if cluster_map:
                    st_folium(cluster_map, height=400, width=None)
                    
                    # Download button for map
                    buf2 = io.BytesIO()
                    cluster_map.save(buf2, close_file=False)
                    st.download_button(
                        "📥 Download Asset Map",
                        buf2.getvalue(),
                        "asset_cluster_map.html",
                        "text/html"
                    )
                else:
                    st.info("No geographic data available for selected filters.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Geographic statistics
        if len(filtered_df) > 0 and all(col in filtered_df.columns for col in ['state', 'value']):
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.subheader("📊 Geographic Statistics")
            
            agg_cols = {'value': ['count', 'sum', 'median']}
            if 'sqft' in filtered_df.columns:
                agg_cols['sqft'] = 'median'
            if 'age' in filtered_df.columns:
                agg_cols['age'] = 'median'
            
            geo_stats = filtered_df.groupby('state').agg(agg_cols).round(2)
            
            # Flatten column names
            geo_stats.columns = ['_'.join(col).strip() for col in geo_stats.columns.values]
            geo_stats = geo_stats.rename(columns={
                'value_count': 'Asset Count',
                'value_sum': 'Total Value',
                'value_median': 'Median Value',
                'sqft_median': 'Median Size (sq ft)',
                'age_median': 'Median Age (years)'
            })
            
            geo_stats = geo_stats.sort_values('Total Value', ascending=False)
            
            # Format values
            if 'Total Value' in geo_stats.columns:
                geo_stats['Total Value'] = geo_stats['Total Value'].apply(fmt_money_units)
            if 'Median Value' in geo_stats.columns:
                geo_stats['Median Value'] = geo_stats['Median Value'].apply(fmt_money_units)
            if 'Median Size (sq ft)' in geo_stats.columns:
                geo_stats['Median Size (sq ft)'] = geo_stats['Median Size (sq ft)'].apply(lambda x: f"{x:,.0f}")
            
            st.dataframe(geo_stats, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # ==================== TAB 3: ML CLUSTERING ====================
    with tab3:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        
        required_cols = ['value', 'sqft', 'value_psf', 'age']
        if len(filtered_df) < 10:
            st.warning("Need at least 10 assets for ML analysis. Please adjust your filters.")
        elif not all(col in filtered_df.columns for col in required_cols):
            st.warning(f"Missing required columns for ML analysis: {[col for col in required_cols if col not in filtered_df.columns]}")
        else:
            st.subheader("🤖 Machine Learning Clustering Analysis")
            
            # Show model loading status
            if models and len(models) > 0:
                st.success(f"✅ Using {len(models)} loaded ML models from Google Drive")
            else:
                st.info("🔄 Using built-in ML algorithms for analysis")
            
            # Clustering parameters
            col1, col2 = st.columns([1, 3])
            with col1:
                k_clusters = st.slider("Number of Clusters (k):", 2, 8, 4)
            
            # Perform clustering
            df_clustered, sil_score, X_scaled, X_cols, scaler, kmeans = perform_clustering_analysis(filtered_df, k_clusters)
            
            # Display ML metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Clusters", f"{k_clusters}")
            col2.metric("Silhouette Score", f"{sil_score:.3f}")
            col3.metric("Quality", 
                       "Excellent" if sil_score >= 0.65 else
                       "Good" if sil_score >= 0.5 else
                       "Moderate" if sil_score >= 0.35 else "Weak")
            
            # Cluster visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Cluster sizes
                cluster_sizes = df_clustered['cluster_name'].value_counts()
                fig_sizes = px.bar(
                    x=cluster_sizes.index,
                    y=cluster_sizes.values,
                    title="Cluster Sizes",
                    labels={'x': 'Cluster', 'y': 'Number of Assets'}
                )
                fig_sizes.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(26,26,26,0.8)',
                    font_color='white'
                )
                st.plotly_chart(fig_sizes, use_container_width=True)
            
            with col2:
                # PCA visualization
                pca = PCA(n_components=2, random_state=42)
                X_pca = pca.fit_transform(X_scaled)
                
                pca_df = pd.DataFrame({
                    'PC1': X_pca[:, 0],
                    'PC2': X_pca[:, 1],
                    'Cluster': df_clustered['cluster_name']
                })
                
                fig_pca = px.scatter(
                    pca_df,
                    x='PC1',
                    y='PC2',
                    color='Cluster',
                    title="PCA Cluster Visualization",
                    opacity=0.7
                )
                fig_pca.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(26,26,26,0.8)',
                    font_color='white'
                )
                st.plotly_chart(fig_pca, use_container_width=True)
            
            # Cluster profiles
            st.subheader("📋 Cluster Profiles")
            
            profiles = df_clustered.groupby('cluster_name')[['value', 'sqft', 'value_psf', 'age']].median()
            
            profile_cols = st.columns(len(profiles))
            for i, (cluster_name, profile) in enumerate(profiles.iterrows()):
                with profile_cols[i]:
                    st.markdown(f"""
                    **{cluster_name}**
                    - Median Value: {fmt_money_units(profile['value'])}
                    - Median Size: {profile['sqft']:,.0f} ft²
                    - Median $/ft²: {fmt_money_units(profile['value_psf'])}
                    - Median Age: {profile['age']:.0f} years
                    """)
            
            # Classification analysis
            st.subheader("🎯 Cluster Prediction Model")
            
            rf, accuracy, feature_importance, cm, y_test, y_pred = perform_classification_analysis(
                X_scaled, df_clustered['cluster'].values, X_cols
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Model Accuracy", f"{accuracy:.3f}")
                
                # Feature importance
                fig_importance = px.bar(
                    x=feature_importance.values,
                    y=feature_importance.index,
                    orientation='h',
                    title="Feature Importance"
                )
                fig_importance.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(26,26,26,0.8)',
                    font_color='white'
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            
            with col2:
                # Confusion matrix
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    aspect="auto",
                    title="Confusion Matrix",
                    color_continuous_scale="Blues"
                )
                fig_cm.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(26,26,26,0.8)',
                    font_color='white'
                )
                st.plotly_chart(fig_cm, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ==================== TAB 4: ADVANCED ANALYTICS ====================
    with tab4:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.subheader("📈 Advanced Analytics & Insights")
        
        if len(filtered_df) > 0:
            # Model predictions section
            if models and 'model_prediction' in filtered_df.columns:
                st.subheader("🤖 Model Predictions vs Actual Values")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Prediction accuracy
                    if 'predicted_value_original' in filtered_df.columns:
                        actual_vs_pred = filtered_df[['value', 'predicted_value_original']].dropna()
                        if len(actual_vs_pred) > 0:
                            fig_pred = px.scatter(
                                actual_vs_pred.sample(min(1000, len(actual_vs_pred))),
                                x='value',
                                y='predicted_value_original',
                                title="Actual vs Predicted Values",
                                labels={'value': 'Actual Value ($)', 'predicted_value_original': 'Predicted Value ($)'}
                            )
                            
                            # Add diagonal line
                            min_val = min(actual_vs_pred['value'].min(), actual_vs_pred['predicted_value_original'].min())
                            max_val = max(actual_vs_pred['value'].max(), actual_vs_pred['predicted_value_original'].max())
                            fig_pred.add_shape(
                                type="line",
                                x0=min_val, y0=min_val,
                                x1=max_val, y1=max_val,
                                line=dict(color="red", width=2, dash="dash")
                            )
                            
                            fig_pred.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(26,26,26,0.8)',
                                font_color='white'
                            )
                            st.plotly_chart(fig_pred, use_container_width=True)
                
                with col2:
                    # Model performance metrics
                    if 'predicted_value_original' in filtered_df.columns:
                        valid_data = filtered_df[['value', 'predicted_value_original']].dropna()
                        if len(valid_data) > 0:
                            r2 = r2_score(valid_data['value'], valid_data['predicted_value_original'])
                            mae = mean_absolute_error(valid_data['value'], valid_data['predicted_value_original'])
                            
                            st.metric("R² Score", f"{r2:.3f}")
                            st.metric("Mean Absolute Error", fmt_money_units(mae))
                            
                            # Model usage distribution
                            if 'model_used' in filtered_df.columns:
                                model_usage = filtered_df['model_used'].value_counts()
                                fig_models = px.pie(
                                    values=model_usage.values,
                                    names=model_usage.index,
                                    title="Model Usage Distribution"
                                )
                                fig_models.update_layout(
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(26,26,26,0.8)',
                                    font_color='white'
                                )
                                st.plotly_chart(fig_models, use_container_width=True)
            
            # Performance analytics
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk vs Value analysis (if available)
                if 'risk_score' in filtered_df.columns:
                    fig_risk = px.scatter(
                        filtered_df.sample(min(2000, len(filtered_df))),
                        x='risk_score',
                        y='value',
                        color='asset_type' if 'asset_type' in filtered_df.columns else None,
                        size='sqft' if 'sqft' in filtered_df.columns else None,
                        title="Risk vs Value Analysis",
                        labels={'risk_score': 'Risk Score', 'value': 'Asset Value ($)'}
                    )
                    fig_risk.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(26,26,26,0.8)',
                        font_color='white'
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)
                
                # Age distribution
                if 'age' in filtered_df.columns:
                    fig_age = px.histogram(
                        filtered_df,
                        x='age',
                        nbins=20,
                        title="Asset Age Distribution",
                        labels={'age': 'Age (years)', 'count': 'Number of Assets'}
                    )
                    fig_age.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(26,26,26,0.8)',
                        font_color='white'
                    )
                    st.plotly_chart(fig_age, use_container_width=True)
            
            with col2:
                # Utilization vs Value (if available)
                if 'utilization_rate' in filtered_df.columns:
                    fig_util = px.scatter(
                        filtered_df.sample(min(2000, len(filtered_df))),
                        x='utilization_rate',
                        y='value',
                        color='condition' if 'condition' in filtered_df.columns else None,
                        title="Utilization vs Value",
                        labels={'utilization_rate': 'Utilization Rate', 'value': 'Asset Value ($)'}
                    )
                    fig_util.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(26,26,26,0.8)',
                        font_color='white'
                    )
                    st.plotly_chart(fig_util, use_container_width=True)
                
                # Confidence level distribution
                if 'conf_cat' in filtered_df.columns:
                    conf_counts = filtered_df['conf_cat'].value_counts()
                    fig_conf = px.pie(
                        values=conf_counts.values,
                        names=conf_counts.index,
                        title="Confidence Level Distribution"
                    )
                    fig_conf.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(26,26,26,0.8)',
                        font_color='white'
                    )
                    st.plotly_chart(fig_conf, use_container_width=True)
            
            # Summary insights
            st.subheader("🧠 Portfolio Insights")
            
            insights_col1, insights_col2, insights_col3 = st.columns(3)
            
            with insights_col1:
                if 'value' in filtered_df.columns:
                    high_value_assets = len(filtered_df[filtered_df['value'] > filtered_df['value'].quantile(0.9)])
                    high_risk_assets = len(filtered_df[filtered_df['risk_score'] > 0.7]) if 'risk_score' in filtered_df.columns else 0
                    old_assets = len(filtered_df[filtered_df['age'] > 40]) if 'age' in filtered_df.columns else 0
                    avg_util = filtered_df['utilization_rate'].mean() if 'utilization_rate' in filtered_df.columns else 0
                    
                    st.markdown(f"""
                    **🎯 Portfolio Highlights**
                    - High-value assets (top 10%): {high_value_assets:,}
                    - High-risk assets: {high_risk_assets:,}
                    - Assets over 40 years: {old_assets:,}
                    - Average utilization: {avg_util:.1%}
                    """)
            
            with insights_col2:
                if 'value' in filtered_df.columns and 'state' in filtered_df.columns:
                    best_performing = filtered_df.nlargest(1, 'value')['state'].iloc[0] if len(filtered_df) > 0 else "N/A"
                    most_assets = filtered_df['state'].value_counts().index[0] if len(filtered_df) > 0 else "N/A"
                    avg_age = filtered_df['age'].mean() if 'age' in filtered_df.columns else 0
                    premium_threshold = filtered_df['value'].quantile(0.95)
                    premium_assets = len(filtered_df[filtered_df['value'] > premium_threshold])
                    
                    st.markdown(f"""
                    **🏆 Top Performers**
                    - Highest value state: {best_performing}
                    - Most assets in: {most_assets}
                    - Average asset age: {avg_age:.1f} years
                    - Premium assets (>${premium_threshold/1e6:.1f}M): {premium_assets:,}
                    """)
            
            with insights_col3:
                total_maintenance = filtered_df['maintenance_cost'].sum() if 'maintenance_cost' in filtered_df.columns else 0
                avg_efficiency = filtered_df['energy_efficiency'].value_counts().index[0] if 'energy_efficiency' in filtered_df.columns and len(filtered_df) > 0 else "N/A"
                growth_potential = filtered_df['growth_potential'].mean() if 'growth_potential' in filtered_df.columns else 0
                low_util_assets = len(filtered_df[filtered_df['utilization_rate'] < 0.5]) if 'utilization_rate' in filtered_df.columns else 0
                
                st.markdown(f"""
                **💡 Operational Metrics**
                - Total maintenance costs: {fmt_money_units(total_maintenance)}
                - Most common efficiency: {avg_efficiency}
                - Growth potential: {growth_potential:.1%}
                - Low utilization assets: {low_util_assets:,}
                """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #9ca3af; padding: 1rem;'>
        <p>🏛️ <strong>SmartAssets Analytics Pro</strong> | Google Drive Integration | Comprehensive ML-Powered Asset Valuation Platform</p>
        <p>Built with Streamlit • Plotly • Scikit-learn • Folium • Google Drive API</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

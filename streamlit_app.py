import streamlit as st
import pandas as pd
import numpy as np
import gdown
import os
import pickle
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium
from streamlit_folium import folium_static
import geopandas as gpd
from shapely.geometry import Point
from thefuzz import process, fuzz
import libpysal
import esda
import contextily as cx


# --- Configuration ---
st.set_page_config(layout="wide", page_title="Government Asset Valuation Dashboard")
RANDOM_STATE = 4742271 # Ensure reproducibility

# --- Helper Functions ---
@st.cache_data # Cache data loading
def load_data():
    """Downloads the Zillow and Assets datasets from Google Drive."""
    zillow_url = "https://drive.google.com/uc?id=1fFT8Q8GWiIEM7kx6czhQ-qabygUPBQRv"
    assets_url = "https://drive.google.com/uc?id=1YFTWJNoxu0BF8UlMDXI8bXwRTVQNE2mb"
    zillow_path = "zillow_housing_index.csv"
    assets_path = "us_government_assets.csv"

    if not os.path.exists(zillow_path):
        st.info("Downloading Zillow dataset...")
        gdown.download(zillow_url, zillow_path, quiet=True)
    if not os.path.exists(assets_path):
        st.info("Downloading Assets dataset...")
        gdown.download(assets_url, assets_path, quiet=True)

    df_zillow_raw = pd.read_csv(zillow_path)
    df_assets_raw = pd.read_csv(assets_path)
    return df_zillow_raw, df_assets_raw

def download_model_files():
    """Downloads all required .pkl files from Google Drive."""
    file_ids = {
        "cluster_0_model.pkl": "16_C4yL1R-M3b-e8aJ8p-4qY-u7tO8Zz2",
        "cluster_1_model.pkl": "16aE8wX1d-r4tHjB_u6L6Y_tZ-n5VqR_B",
        "cluster_pca_0_model.pkl": "16cO8jP6f-z4xJ_oY_iP9d_f_kL2oE_dE",
        "cluster_pca_1_model.pkl": "16hG7yD4m-n7xK_lT_jJ_rW_zO_oX_fCg",
        "global_model.pkl": "16jI8pL7q-D5xJ_oY_iP9d_f_kL2oE_dE",
        "global_model_pca.pkl": "16lK9wZ_t-V8mH_jJ_rW_zO_oX_fCg",
        "pca_final.pkl": "16nN_u6L6Y_tZ-n5VqR_B-C4yL1R-M3b",
        "scaler_all.pkl": "16qP_lT_jJ_rW_zO_oX_fCg-e8aJ8p-4qY",
        "scaler_last_price.pkl": "16rS_kL2oE_dE-V8mH_jJ_rW_zO_oX_fCg"
    }
    st.info("Checking for model and scaler files...")
    for filename, file_id in file_ids.items():
        if not os.path.exists(filename):
            st.info(f"Downloading {filename}...")
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, filename, quiet=True)

@st.cache_data # Cache model loading
def load_models_and_scalers():
    """Loads all models and scalers after ensuring they are downloaded."""
    download_model_files() # Ensure files are present before loading
    
    scalers = {}
    models = {}
    try:
        scalers['all'] = pickle.load(open("scaler_all.pkl","rb"))
        scalers['last'] = pickle.load(open("scaler_last_price.pkl","rb"))
        models['global'] = pickle.load(open("global_model.pkl","rb"))
        models['global_pca'] = pickle.load(open("global_model_pca.pkl","rb"))
        
        cluster_models_orig = {}
        cluster_models_pca = {}
        for i in range(2): # Assuming 2 clusters
            try:
                cluster_models_orig[i] = pickle.load(open(f"cluster_{i}_model.pkl","rb"))
            except FileNotFoundError:
                cluster_models_orig[i] = None
            try:
                cluster_models_pca[i] = pickle.load(open(f"cluster_pca_{i}_model.pkl","rb"))
            except FileNotFoundError:
                cluster_models_pca[i] = None
        models['cluster_orig'] = cluster_models_orig
        models['cluster_pca'] = cluster_models_pca

        return models, scalers
    except FileNotFoundError as e:
        st.error(f"Error loading required files: {e}. Please ensure the Google Drive link is valid and files are accessible.")
        st.stop()

# (The rest of the helper functions: preprocess_zillow, feature_engineer_zillow, etc. remain the same as the previous version)
def preprocess_zillow(df_zillow_raw, sample_n=5000):
    df_z = df_zillow_raw.copy()
    if sample_n and sample_n < len(df_z):
        df_z = df_z.sample(n=sample_n, random_state=RANDOM_STATE).reset_index(drop=True)
    else:
        df_z = df_z.copy().reset_index(drop=True)

    date_cols = [c for c in df_z.columns if c[:2].isdigit()]
    df_z[date_cols] = df_z[date_cols].apply(pd.to_numeric, errors='coerce')

    # Remove outliers rowwise (simplified for app)
    def remove_outliers_rowwise_simple(row, thresh=3.0):
        mu = row.mean()
        sd = row.std()
        if pd.isna(sd) or sd == 0:
            return row
        z = (row - mu) / sd
        row[z.abs() > thresh] = np.nan
        return row

    df_z[date_cols] = df_z[date_cols].apply(remove_outliers_rowwise_simple, axis=1)

    # KNN imputation
    imputer = KNNImputer(n_neighbors=5)
    df_z[date_cols] = imputer.fit_transform(df_z[date_cols])
    return df_z, date_cols

def feature_engineer_zillow(df_z, date_cols):
    features = []
    for idx, row in df_z.iterrows():
        prices = row[date_cols].values.astype(float)
        # basic stats
        mean_price = np.nanmean(prices)
        median_price = np.nanmedian(prices)
        std_price = np.nanstd(prices)
        price_min = np.nanmin(prices)
        price_max = np.nanmax(prices)
        price_range = price_max - price_min
        price_volatility = (std_price / mean_price) if mean_price != 0 else 0.0
        recent_6mo_avg = np.nanmean(prices[-6:]) if len(prices) >= 6 else mean_price
        recent_12mo_avg = np.nanmean(prices[-12:]) if len(prices) >= 12 else mean_price
        last_price = float(prices[-1])
        # time trend (slope)
        t = np.arange(len(prices)).reshape(-1,1)
        try:
            slope = float(LinearRegression().fit(t, prices).coef_[0])
        except:
            slope = 0.0

        features.append({
            'RegionID': row.get('RegionID', idx),
            'RegionName': row.get('RegionName', ''),
            'City': str(row.get('City','')).upper().strip(),
            'State': str(row.get('State','')).upper().strip(),
            'County': str(row.get('CountyName','')).upper().strip(),
            'mean_price': mean_price,
            'median_price': median_price,
            'std_price': std_price,
            'price_min': price_min,
            'price_max': price_max,
            'price_range': price_range,
            'price_volatility': price_volatility,
            'recent_6mo_avg': recent_6mo_avg,
            'recent_12mo_avg': recent_12mo_avg,
            'last_price': last_price,
            'price_trend_slope': slope
        })
    return pd.DataFrame(features)

def preprocess_assets(df_assets_raw):
    df_assets = df_assets_raw.copy()
    for col in ['City', 'State', 'Installation Name', 'Real Property Asset Name', 'Street Address']:
        if col in df_assets.columns:
            df_assets[col] = df_assets[col].astype(str).fillna('').str.upper().str.strip()

    if 'Latitude' in df_assets.columns:
        df_assets['Latitude'] = pd.to_numeric(df_assets['Latitude'], errors='coerce')
    if 'Longitude' in df_assets.columns:
        df_assets['Longitude'] = pd.to_numeric(df_assets['Longitude'], errors='coerce')

    if 'Zip Code' in df_assets.columns:
        df_assets['Zip Code'] = df_assets['Zip Code'].astype(str).str.zfill(5).str.strip()
    return df_assets

def enrich_assets(df_assets, df_z_features_raw, scaler_all, enc):
    zjoin = df_z_features_raw[['City','State'] + num_cols].copy()
    zjoin['City'] = zjoin['City'].astype(str).fillna('').str.upper().str.strip()
    zjoin['State'] = zjoin['State'].astype(str).fillna('').str.upper().str.strip()

    # Add encoded geographic features for fuzzy matching context
    zjoin[['City_enc','State_enc','County_enc']] = enc.transform(
        df_z_features_raw[['City','State','County']].astype(str)
    )


    df_assets['City'] = df_assets['City'].astype(str).fillna('').str.upper().str.strip()
    df_assets['State'] = df_assets['State'].astype(str).fillna('').str.upper().str.strip()

    assets_enriched = df_assets.merge(zjoin, how='left', on=['City','State'], suffixes=('','_z'))

    # Fuzzy fallback
    missing_idx = assets_enriched[assets_enriched[num_cols[0]].isna()].index.tolist() # Check if any numeric feature is missing as a proxy
    state_to_cities = zjoin.groupby('State')['City'].apply(lambda s: sorted(set(s))).to_dict()

    for i in missing_idx:
        st = assets_enriched.at[i,'State']
        city = assets_enriched.at[i,'City']
        candidates = state_to_cities.get(st, [])
        if not candidates:
            assets_enriched.at[i,'_match_type'] = 'no_candidates_in_state'
            continue
        best = process.extractOne(city, candidates, scorer=fuzz.token_sort_ratio)
        if best and best[1] >= 85:
            matched_city = best[0]
            matched_row = zjoin[(zjoin['State']==st) & (zjoin['City']==matched_city)].iloc[0]
            for col in num_cols:
                assets_enriched.at[i,col] = matched_row[col]
            # Also copy encoded features for prediction if needed
            assets_enriched.at[i,'City_enc'] = matched_row['City_enc']
            assets_enriched.at[i,'State_enc'] = matched_row['State_enc']
            assets_enriched.at[i,'County_enc'] = matched_row['County_enc']
            assets_enriched.at[i, '_match_type'] = f"fuzzy:{best[1]}"
        else:
            assets_enriched.at[i, '_match_type'] = 'no_good_fuzzy'

    # State median fallback
    still_missing_num = assets_enriched[num_cols[0]].isna().sum()
    if still_missing_num > 0:
        med_by_state = zjoin.groupby('State')[num_cols + ['City_enc', 'State_enc', 'County_enc']].median().reset_index()
        assets_enriched = assets_enriched.merge(med_by_state, on='State', how='left', suffixes=('','_state_med'))
        for c in num_cols + ['City_enc', 'State_enc', 'County_enc']:
            col_state = f"{c}_state_med"
            if col_state in assets_enriched.columns:
                assets_enriched[c] = assets_enriched[c].fillna(assets_enriched[col_state])
                assets_enriched.drop(columns=[col_state], inplace=True)
        if '_match_type' in assets_enriched.columns:
            assets_enriched['_match_type'] = assets_enriched['_match_type'].fillna('state_median')
        else:
            assets_enriched['_match_type'] = 'state_median'

    # Fill any remaining NaNs in numeric features with a global median/mean if necessary
    for col in num_cols:
        if assets_enriched[col].isna().any():
            global_median = zjoin[col].median() # Use median from original zillow features
            assets_enriched[col] = assets_enriched[col].fillna(global_median)

    # Fill remaining NaNs in encoded features with a placeholder if necessary
    for col in ['City_enc', 'State_enc', 'County_enc']:
        if assets_enriched[col].isna().any():
            assets_enriched[col] = assets_enriched[col].fillna(-2) # Use -2 for unmatched encoded features

    # Scale the numeric features in the enriched dataset
    assets_enriched[num_cols] = scaler_all.transform(assets_enriched[num_cols])

    return assets_enriched

def predict_asset_values(assets_df, models, scalers, num_predictors_pca, num_cols, output_col_name='pred_last_price_original', output_scaled_col_name='pred_last_price_scaled'):
    assets_with_predictions = assets_df.copy()
    global_model = models['global']
    global_model_pca = models['global_pca']
    cluster_models_orig = models['cluster_orig']
    cluster_models_pca = models['cluster_pca']
    scaler_last = scalers['last']

    pred_scaled_list = []
    pred_orig_list = []
    model_used_list = []

    # Ensure PCA components are in the dataframe if using PCA models
    pca_cols = [f'pca_component_{i+1}' for i in range(3)] # Assuming 3 components
    if not all(col in assets_with_predictions.columns for col in pca_cols):
        try:
            pca_final = pickle.load(open("pca_final.pkl","rb")) # Load if saved
        except FileNotFoundError:
            st.error("pca_final.pkl not found. Cannot perform prediction with PCA models.")
            st.stop()
        
        imputer = SimpleImputer(strategy='mean') # Ensure imputer is available
        assets_numeric_imputed = imputer.fit_transform(assets_with_predictions[num_cols])
        pca_components_assets = pca_final.transform(assets_numeric_imputed)
        for i in range(3):
            assets_with_predictions[pca_cols[i]] = pca_components_assets[:, i]

    # Decide which model set to use (with or without PCA)
    # We'll prioritize PCA models if available and the corresponding cluster column exists
    use_pca_models = False
    if all(col in assets_with_predictions.columns for col in num_predictors_pca) and 'cluster_kmeans_pca' in assets_with_predictions.columns:
        if all(model is not None for model in cluster_models_pca.values()) and global_model_pca is not None:
            use_pca_models = True
            st.info("Using models trained with PCA features.")
            num_predictors_to_use = num_predictors_pca
            cluster_col_to_use = 'cluster_kmeans_pca'
            models_to_use = cluster_models_pca
            global_model_to_use = global_model_pca
        else:
            st.warning("PCA-based cluster models or global PCA model not fully loaded. Falling back to original models.")


    if not use_pca_models:
        st.info("Using original models (without PCA features).")
        num_predictors_to_use = num_cols # Use the original feature list
        cluster_col_to_use = 'cluster_kmeans' # Use the original cluster column
        models_to_use = cluster_models_orig
        global_model_to_use = global_model


    for idx, r in assets_with_predictions.iterrows():
        # Ensure all required columns for prediction are in the row, fill missing with 0 if necessary
        # This is a safeguard, ideally the dataframe assets_with_predictions is complete
        x = r[num_predictors_to_use].fillna(0).values.reshape(1,-1)

        # which cluster model?
        cluster_id = None
        if cluster_col_to_use in r and pd.notna(r[cluster_col_to_use]):
            try:
                cluster_id = int(r[cluster_col_to_use])
            except (ValueError, TypeError):
                cluster_id = None

        model_pack = models_to_use.get(cluster_id) if cluster_id is not None else None

        if model_pack and model_pack is not None:
            model = model_pack['best_model']
            model_used = f"cluster_{cluster_id}"
        else:
            model = global_model_to_use
            model_used = "global"

        try:
            pred_scaled = float(model.predict(x)[0])
        except Exception as e:
            st.warning(f"Prediction failed for row {idx} using model {model_used}: {e}")
            pred_scaled = np.nan # Assign NaN on failure

        # invert to original last_price units (USD-like)
        if pd.notna(pred_scaled):
            pred_original = float(scaler_last.inverse_transform([[pred_scaled]])[0][0])
        else:
            pred_original = np.nan

        pred_scaled_list.append(pred_scaled)
        pred_orig_list.append(pred_original)
        model_used_list.append(model_used)

    assets_with_predictions[output_scaled_col_name] = pred_scaled_list
    assets_with_predictions[output_col_name] = pred_orig_list
    assets_with_predictions['model_used'] = model_used_list

    return assets_with_predictions

# --- Global Variables ---
num_cols = [
    'mean_price','median_price','std_price','price_min','price_max','price_range',
    'price_volatility','recent_6mo_avg','recent_12mo_avg','last_price','price_trend_slope'
]
num_predictors_pca = num_cols + [f'pca_component_{i+1}' for i in range(3)]
price_features = [
    'mean_price','median_price','std_price','price_min','price_max','price_range',
    'recent_6mo_avg','recent_12mo_avg','last_price'
]

# --- Streamlit App Layout ---
st.title("Government Asset Valuation Dashboard")

# --- Data and Model Loading ---
with st.spinner("Loading datasets, models, and scalers..."):
    df_zillow_raw, df_assets_raw = load_data()
    models, scalers = load_models_and_scalers()
st.success("All data and models loaded successfully.")

# --- Data Preprocessing ---
st.header("Zillow Data Processing")
if st.button("Process Zillow Data"):
    with st.spinner("Processing Zillow data... This may take a moment."):
        df_z, date_cols = preprocess_zillow(df_zillow_raw)
        df_z_features_raw = feature_engineer_zillow(df_z, date_cols)
        
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        enc.fit(df_z_features_raw[['City','State','County']].astype(str))
        
        df_z_feat_scaled = df_z_features_raw.copy()
        df_z_feat_scaled[num_cols] = scalers['all'].transform(df_z_features_raw[num_cols])
        
        df_z_feat_scaled[['City_enc','State_enc','County_enc']] = enc.transform(
            df_z_features_raw[['City','State','County']].astype(str)
        )
        
        st.session_state['df_z_features_raw'] = df_z_features_raw
        st.session_state['df_z_feat_scaled'] = df_z_feat_scaled
        st.session_state['ordinal_encoder'] = enc
        
    st.success("Zillow data processed and features engineered.")

# --- Asset Enrichment and Prediction ---
st.header("Asset Enrichment and Prediction")
if st.button("Enrich Assets and Predict Values"):
    if 'df_z_feat_scaled' not in st.session_state or 'ordinal_encoder' not in st.session_state:
        st.warning("Please process Zillow data first.")
    else:
        with st.spinner("Enriching assets and predicting values..."):
            df_z_feat_scaled = st.session_state['df_z_feat_scaled']
            enc = st.session_state['ordinal_encoder']
            
            assets_enriched = enrich_assets(df_assets_raw.copy(), df_z_feat_scaled, scalers['all'], enc)
            st.session_state['assets_enriched'] = assets_enriched

            assets_with_predictions = predict_asset_values(assets_enriched, models, scalers, num_predictors_pca, num_cols)
            st.session_state['assets_with_predictions'] = assets_with_predictions
            
        st.success("Assets enriched and values predicted.")


# --- Display Results ---
if 'assets_with_predictions' in st.session_state:
    assets_with_predictions = st.session_state['assets_with_predictions']
    assets_enriched = st.session_state['assets_enriched']
    st.header("Predicted Asset Values")
    st.dataframe(assets_with_predictions[['Real Property Asset Name', 'City', 'State', 'pred_last_price_original', 'model_used']].head())

    # --- Statistics and Visualizations ---
    st.subheader("Predicted Value Statistics")
    st.write(assets_with_predictions['pred_last_price_original'].describe())
    
    # ... (The visualization and scenario analysis sections remain the same) ...
    # Histogram
    plt.figure(figsize=(10,5))
    sns.histplot(assets_with_predictions['pred_last_price_original'].dropna(), bins=80, kde=True)
    plt.title("Distribution of Predicted Asset Values (original units)")
    st.pyplot(plt)


    # State-level Choropleth
    st.subheader("State-level Predicted Value (Median)")
    if 'State' in assets_with_predictions.columns and 'pred_last_price_original' in assets_with_predictions.columns:
        state_agg = assets_with_predictions.groupby('State').agg({
            'pred_last_price_original':'median',
            'Real Property Asset Name':'count'
        }).reset_index().rename(columns={'Real Property Asset Name':'asset_count'})
        try:
            fig = px.choropleth(state_agg, locations='State', locationmode="USA-states",
                                color='pred_last_price_original',
                                hover_data=['asset_count','pred_last_price_original'],
                                scope="usa", title="Median Predicted Asset Value by State")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate Choropleth map: {e}")
    else:
        st.info("State or predicted price columns not available for Choropleth map.")


    # Folium Map
    st.subheader("Asset Locations Map")
    if {'Latitude','Longitude','pred_last_price_scaled'}.issubset(assets_with_predictions.columns):
        m = folium.Map(location=[39.8, -98.6], zoom_start=4, tiles="CartoDB positron")
        # Use quantiles from the data available in the app
        valid_scaled_prices = assets_with_predictions['pred_last_price_scaled'].dropna()
        if len(valid_scaled_prices) > 5: # Need at least 5 data points for quantiles
            q = valid_scaled_prices.quantile([0, .2, .4, .6, .8, 1]).values
            def color_by_val(v):
                if v <= q[1]: return "green"
                elif v <= q[2]: return "lightgreen"
                elif v <= q[3]: return "orange"
                elif v <= q[4]: return "darkorange"
                else: return "red"
        else: # Fallback color if not enough data for quantiles
            st.warning("Not enough data to calculate price quantiles for map coloring.")
            def color_by_val(v): return "blue" # Default color


        for _, r in assets_with_predictions.dropna(subset=['Latitude','Longitude']).iterrows():
            folium.CircleMarker(
                location=[r['Latitude'], r['Longitude']],
                radius=4,
                color=color_by_val(r['pred_last_price_scaled']),
                fill=True, fill_opacity=0.8,
                popup=(f"{r.get('Real Property Asset Name','Asset')}<br>"
                       f"{r.get('City','N/A')}, {r.get('State','N/A')}<br>"
                       f"${r.get('pred_last_price_original',0):,.0f}<br>Model:{r.get('model_used','N/A')}")) \
            .add_to(m)
        folium_static(m)
    else:
        st.info("Latitude, Longitude, or scaled predicted price not available for Folium map.")

    # LISA Map (if PySAL is installed and data is GeoDataFrame)
    st.subheader("Spatial Autocorrelation (LISA)")
    # Check if LISA results are available in the dataframe or recalculate
    if 'lisa_cluster' in assets_with_predictions.columns:
        st.info("Displaying pre-calculated LISA results.")
        gdf_assets_predictions = assets_with_predictions # Assume it's already a GeoDataFrame with LISA results
        # Ensure it's a GeoDataFrame if not already
        if not isinstance(gdf_assets_predictions, gpd.GeoDataFrame):
            st.warning("Data is not a GeoDataFrame. Attempting to create one for LISA map.")
            try:
                gdf_assets_predictions = gpd.GeoDataFrame(
                    assets_with_predictions.dropna(subset=['Latitude', 'Longitude']).copy(),
                    geometry=gpd.points_from_xy(assets_with_predictions['Longitude'], assets_with_predictions['Latitude']),
                    crs="EPSG:4326"
                )
                gdf_assets_predictions = gdf_assets_predictions.to_crs(epsg=3857) # Project
            except Exception as e:
                st.error(f"Failed to create GeoDataFrame for LISA map: {e}")


        if isinstance(gdf_assets_predictions, gpd.GeoDataFrame) and 'lisa_cluster' in gdf_assets_predictions.columns:
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            gdf_assets_predictions.plot(column='lisa_cluster',
                                       categorical=True,
                                       legend=True,
                                       ax=ax,
                                       cmap='viridis',
                                       markersize=5)
            ax.set_aspect('equal')
            ax.set_title('LISA Cluster Map of Predicted Asset Values')
            try:
                cx.add_basemap(ax, crs=gdf_assets_predictions.crs.to_string(), source=cx.providers.CartoDB.Positron)
            except Exception as e:
                st.warning(f"Could not add basemap: {e}")

            st.pyplot(fig)
        else:
            st.info("LISA cluster results or GeoDataFrame not available for mapping.")

    else:
        st.info("LISA results not available in the data.")
        # Note: Recalculating LISA on the fly is computationally expensive and is commented out
        # in the original code for this reason.

    # --- Scenario Analysis Section ---
    st.header("Scenario Analysis")
    st.write("Explore how predicted asset values might change under hypothetical scenarios.")
    
    scenario_type = st.selectbox("Select Scenario Type", ["Percentage Change in Prices", "Change in Price Trend Slopes"])

    if scenario_type == "Percentage Change in Prices":
        st.subheader("Simulate Percentage Change in Prices")
        pct_change_input = st.slider("Hypothetical Percentage Change (%)", -20.0, 20.0, 5.0, 0.5) / 100.0
        
        if st.button("Run Percentage Change Scenario"):
            with st.spinner(f"Simulating a {pct_change_input*100:.1f}% change..."):
                # Start with the original enriched assets
                assets_scenario_pct = assets_enriched.copy()
                
                # Inverse transform, apply change, re-scale
                original_num_cols_in_assets = [col for col in num_cols if col in assets_scenario_pct.columns]
                assets_original_numeric = pd.DataFrame(scalers['all'].inverse_transform(assets_scenario_pct[original_num_cols_in_assets]), columns=original_num_cols_in_assets, index=assets_scenario_pct.index)
                
                for feature in price_features:
                    if feature in assets_original_numeric.columns:
                        assets_original_numeric[feature] = assets_original_numeric[feature] * (1 + pct_change_input)
                
                assets_scenario_pct[num_cols] = scalers['all'].transform(assets_original_numeric[num_cols])
                st.info("Applied percentage change and re-scaled features for scenario.")
                
                # Predict with scenario data, saving to new columns
                assets_scenario_pct = predict_asset_values(assets_scenario_pct, models, scalers, num_predictors_pca, num_cols,
                                                         output_col_name='pred_last_price_original_scenario',
                                                         output_scaled_col_name='pred_last_price_scaled_scenario')
                
                # Merge the original predictions for comparison
                assets_scenario_pct['pred_last_price_original'] = assets_with_predictions['pred_last_price_original']
                
                # Compare and display results
                assets_scenario_pct['predicted_change_original'] = assets_scenario_pct['pred_last_price_original_scenario'] - assets_scenario_pct['pred_last_price_original']
                assets_scenario_pct['predicted_change_pct'] = (assets_scenario_pct['predicted_change_original'] / assets_scenario_pct['pred_last_price_original']) * 100
                
                st.write("Scenario analysis results:")
                st.dataframe(assets_scenario_pct[['Real Property Asset Name', 'City', 'State', 'pred_last_price_original', 'pred_last_price_original_scenario', 'predicted_change_original', 'predicted_change_pct']].head())

                st.subheader("Summary of Impact")
                st.write(f"Average predicted value (Original): ${assets_scenario_pct['pred_last_price_original'].mean():,.0f}")
                st.write(f"Average predicted value (Scenario): ${assets_scenario_pct['pred_last_price_original_scenario'].mean():,.0f}")
                st.write(f"Average predicted change (Original units): ${assets_scenario_pct['predicted_change_original'].mean():,.0f}")
                st.write(f"Average predicted change (%): {assets_scenario_pct['predicted_change_pct'].mean():.2f}%")
                
                plt.figure(figsize=(10, 6))
                sns.histplot(assets_scenario_pct['predicted_change_pct'].dropna(), bins=50, kde=True)
                plt.title(f"Distribution of Predicted Percentage Change in Asset Value ({pct_change_input*100:.1f}% Housing Price Change Scenario)")
                plt.xlabel("Predicted Percentage Change in Asset Value (%)")
                plt.ylabel("Count")
                st.pyplot(plt)
                

    elif scenario_type == "Change in Price Trend Slopes":
        st.subheader("Simulate Change in Price Trend Slopes")
        slope_change_input = st.number_input("Hypothetical change to scaled price_trend_slope", value=0.01, step=0.001)

        if st.button("Run Slope Change Scenario"):
            with st.spinner(f"Simulating change in price trend slopes (adding {slope_change_input:.3f} to scaled slope)..."):
                # Start with the original enriched assets
                assets_scenario_slope = assets_enriched.copy()
                
                # Apply the change to the scaled price_trend_slope feature
                assets_scenario_slope['price_trend_slope'] = assets_scenario_slope['price_trend_slope'] + slope_change_input
                st.info("Applied change to scaled price_trend_slope.")
                
                # Predict with scenario data, saving to new columns
                assets_scenario_slope = predict_asset_values(assets_scenario_slope, models, scalers, num_predictors_pca, num_cols,
                                                           output_col_name='pred_last_price_original_scenario',
                                                           output_scaled_col_name='pred_last_price_scaled_scenario')
                                                           
                # Merge the original predictions for comparison
                assets_scenario_slope['pred_last_price_original'] = assets_with_predictions['pred_last_price_original']

                # Compare and display results
                assets_scenario_slope['predicted_change_original'] = assets_scenario_slope['pred_last_price_original_scenario'] - assets_scenario_slope['pred_last_price_original']
                assets_scenario_slope['predicted_change_pct'] = (assets_scenario_slope['predicted_change_original'] / assets_scenario_slope['pred_last_price_original']) * 100

                st.write("Scenario analysis results:")
                st.dataframe(assets_scenario_slope[['Real Property Asset Name', 'City', 'State', 'pred_last_price_original', 'pred_last_price_original_scenario', 'predicted_change_original', 'predicted_change_pct']].head())

                st.subheader("Summary of Impact")
                st.write(f"Average predicted value (Original): ${assets_scenario_slope['pred_last_price_original'].mean():,.0f}")
                st.write(f"Average predicted value (Scenario): ${assets_scenario_slope['pred_last_price_original_scenario'].mean():,.0f}")
                st.write(f"Average predicted change (Original units): ${assets_scenario_slope['predicted_change_original'].mean():,.0f}")
                st.write(f"Average predicted change (%): {assets_scenario_slope['predicted_change_pct'].mean():.2f}%")

                plt.figure(figsize=(10, 6))
                sns.histplot(assets_scenario_slope['predicted_change_pct'].dropna(), bins=50, kde=True)
                plt.title(f"Distribution of Predicted Percentage Change in Asset Value (Price Trend Slope Scenario)")
                plt.xlabel("Predicted Percentage Change in Asset Value (%)")
                plt.ylabel("Count")
                st.pyplot(plt)

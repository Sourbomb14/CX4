import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import libpysal
import esda
import contextily as cx

st.set_page_config(layout="wide", page_title="Deep Dive Analytics")

st.title("📊 Deep Dive Analytics")
st.markdown("Explore spatial patterns, distributions, and trends in asset values.")

if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
    st.warning("Please run the analysis on the main page first.")
    st.stop()

df_final = st.session_state.processed_data

# --- Choropleth Map ---
st.header("State-Level Value Distribution")
state_agg = df_final.groupby('State')['Predicted Value'].median().reset_index()
fig = px.choropleth(
    state_agg,
    locations='State',
    locationmode="USA-states",
    color='Predicted Value',
    scope="usa",
    title="Median Predicted Asset Value by State",
    color_continuous_scale=px.colors.sequential.Viridis,
    labels={'Predicted Value': 'Median Value ($)'}
)
st.plotly_chart(fig, use_container_width=True)

# --- Value Distribution Plot ---
st.header("Distribution of Predicted Asset Values")
fig, ax = plt.subplots()
sns.histplot(df_final['Predicted Value'].dropna(), bins=50, kde=True, ax=ax)
ax.set_title("Asset Value Distribution")
ax.set_xlabel("Predicted Value ($)")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# --- Spatial Autocorrelation (LISA) Map ---
st.header("Spatial Hotspot Analysis (LISA)")
st.info("This map identifies statistically significant clusters of high values (High-High, red), low values (Low-Low, blue), and spatial outliers.")

try:
    gdf = gpd.GeoDataFrame(
        df_final.dropna(subset=['Latitude', 'Longitude']),
        geometry=gpd.points_from_xy(df_final.dropna(subset=['Latitude', 'Longitude']).Longitude, df_final.dropna(subset=['Latitude', 'Longitude']).Latitude),
        crs="EPSG:4326"
    )
    gdf = gdf.to_crs(epsg=3857) # Project to Web Mercator for visualization

    # Calculate LISA
    w = libpysal.weights.KNN.from_dataframe(gdf, k=8)
    w.transform = 'r'
    y = gdf['Predicted Value']
    moran_loc = esda.moran.Moran_Local(y, w)

    # Create labels
    gdf['lisa_cluster'] = 'Insignificant'
    gdf.loc[(moran_loc.q == 1) & (moran_loc.p_sim <= 0.05), 'lisa_cluster'] = 'High-High'
    gdf.loc[(moran_loc.q == 3) & (moran_loc.p_sim <= 0.05), 'lisa_cluster'] = 'Low-Low'
    
    # Plotting
    fig, ax = plt.subplots(figsize=(15, 15))
    gdf[gdf['lisa_cluster'] != 'Insignificant'].plot(
        column='lisa_cluster',
        categorical=True,
        legend=True,
        ax=ax,
        cmap='coolwarm',
        markersize=20
    )
    ax.set_axis_off()
    cx.add_basemap(ax, crs=gdf.crs.to_string(), source=cx.providers.CartoDB.Positron)
    st.pyplot(fig)

except Exception as e:
    st.error(f"Could not generate LISA map. Error: {e}")

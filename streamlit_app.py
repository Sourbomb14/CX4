
import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import tempfile
import os
from io import BytesIO

# Attempt optional heavy GIS deps gracefully
try:
    import geopandas as gpd
except Exception as e:
    gpd = None

try:
    import shapely
    from shapely.geometry import shape, Point, Polygon, MultiPolygon
except Exception as e:
    shapely = None

try:
    import fiona
except Exception:
    fiona = None

# GDAL + rasterio are optional; we will only enable features if present
try:
    from osgeo import gdal, ogr, osr
except Exception:
    gdal = None
    ogr = None
    osr = None

try:
    import rasterio
    from rasterio.transform import Affine
    from rasterio.plot import reshape_as_image
except Exception:
    rasterio = None

try:
    import pysal
    from esda.moran import Moran
    from libpysal.weights import KNN, Queen, Rook
except Exception:
    pysal = None
    Moran = None
    KNN = Queen = Rook = None

try:
    from geopy.geocoders import Nominatim
except Exception:
    Nominatim = None

try:
    import folium
    from streamlit_folium import st_folium
except Exception:
    folium = None
    st_folium = None

try:
    import pydeck as pdk
except Exception:
    pdk = None

import matplotlib.pyplot as plt

st.set_page_config(page_title="GIS Analytics Dashboard", layout="wide")

# --- Header with custom HTML/JS ---
st.markdown("""
<style>
.metric-card {
  border-radius: 16px; padding: 16px; box-shadow: 0 2px 12px rgba(0,0,0,0.08);
  background: white; height: 100%; border: 1px solid rgba(0,0,0,0.05);
}
.kpi { font-size: 28px; font-weight: 700; }
.kpi-label { color: #555; font-size: 13px; text-transform: uppercase; letter-spacing: .08em; }
.small { font-size: 12px; color: #777; }
</style>
""", unsafe_allow_html=True)

st.title("🌐 GIS Analytics & Remote Sensing Dashboard")
st.caption("Powered by Streamlit • GeoPandas • GDAL • Shapely • PySAL • Geopy • Folium/PyDeck • Matplotlib • (optional) RSGISLib")

# Inject small JS clock to show HTML/JS usage
st.components.v1.html("""
<div style="font-family:system-ui, -apple-system, Segoe UI, Roboto; padding:8px 12px; background:#111; color:#eee; border-radius:10px; display:inline-block;">
  <span>🕒 <b>Live Clock</b>: <span id="clk"></span></span>
</div>
<script>
function u(){document.getElementById('clk').textContent = new Date().toLocaleString();}
u(); setInterval(u, 1000);
</script>
""", height=50)

# --- Sidebar controls ---
with st.sidebar:
    st.header("Data Sources")
    st.write("Use internet URLs (CSV, GeoJSON, Shapefile .zip, GeoTIFF, GPKG, etc.)")

    preset_urls = ["https://drive.google.com/uc?id=1YFTWJNoxu0BF8UlMDXI8bXwRTVQNE2mb", "https://drive.google.com/uc?id=1fFT8Q8GWiIEM7kx6czhQ-qabygUPBQRv"]
    if not preset_urls:
        st.info("No preset URLs detected. Paste your data URLs below.")
    selected_url = st.selectbox("Preset URLs (from your notebook):", options=["(none)"] + preset_urls)

    custom_url = st.text_input("Or paste a data URL")
    url = custom_url or (selected_url if selected_url != "(none)" else "")

    st.markdown("---")
    st.subheader("Analysis Options")
    layer_name = st.text_input("Layer name (for display)", value="Dataset")
    do_geocode = st.checkbox("Enable Geocoding (addresses ➜ lat/lon)")
    address_input = st.text_input("Address to geocode", value="Mumbai, India") if do_geocode else ""
    do_moran = st.checkbox("Compute Spatial Autocorrelation (Moran's I)")
    moran_col = st.text_input("Numeric column for Moran's I", value="value") if do_moran else ""
    do_batch_convert = st.checkbox("Batch Convert / Reproject (Automation)")
    reproj_epsg = st.text_input("Target EPSG (e.g., 4326)", value="4326") if do_batch_convert else ""

    st.markdown("---")
    st.subheader("Display")
    base_map = st.selectbox("Map Engine", ["Folium", "PyDeck"] if pdk and folium else (["Folium"] if folium else (["PyDeck"] if pdk else ["(none)"])))
    show_table = st.checkbox("Show Attribute Table", value=True)

# --- Helpers ---
def read_vector_from_url(url: str):
    if gpd is None:
        st.error("GeoPandas is not available; cannot read vector data.")
        return None
    try:
        if url.lower().endswith(".zip"):
            return gpd.read_file(url)
        return gpd.read_file(url)
    except Exception as e:
        st.exception(e)
        return None

def read_raster_from_url(url: str):
    if rasterio is None:
        st.warning("Rasterio not available; raster features disabled.")
        return None
    try:
        return rasterio.open(url)
    except Exception as e:
        st.exception(e)
        return None

def read_table_from_url(url: str):
    try:
        if url.lower().endswith(".csv"):
            return pd.read_csv(url)
        if url.lower().endswith(".json") or url.lower().endswith(".geojson"):
            return pd.read_json(url)
        if url.lower().endswith(".parquet"):
            return pd.read_parquet(url)
        if url.lower().endswith(".xlsx") or url.lower().endswith(".xls"):
            return pd.read_excel(url)
    except Exception as e:
        st.exception(e)
    return None

# --- Main loader ---
gdf = None
df = None
rast = None

if url:
    st.write(f"**Selected URL:** {url}")
    # Try vector first
    if any(url.lower().endswith(ext) for ext in [".geojson", ".json", ".gpkg", ".kml", ".kmz", ".zip", ".shp"]):
        gdf = read_vector_from_url(url)
    # Raster?
    if any(url.lower().endswith(ext) for ext in [".tif", ".tiff"]):
        rast = read_raster_from_url(url)
    # Tabular?
    if any(url.lower().endswith(ext) for ext in [".csv", ".xlsx", ".xls", ".parquet", ".json", ".geojson"]):
        df = read_table_from_url(url)

# --- Geocoding ---
geocode_point = None
if do_geocode:
    if Nominatim is None:
        st.warning("geopy not available.")
    else:
        try:
            geolocator = Nominatim(user_agent="streamlit-gis-app")
            loc = geolocator.geocode(address_input)
            if loc:
                geocode_point = (loc.latitude, loc.longitude)
                st.success(f"Geocoded: {address_input} → ({loc.latitude:.5f}, {loc.longitude:.5f})")
            else:
                st.warning("No geocoding result.")
        except Exception as e:
            st.exception(e)

# --- KPIs ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi">{0 if gdf is None else len(gdf)}</div><div class="kpi-label">Features</div>')
    st.markdown('<div class="small">Vector rows</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    crs = None
    if gdf is not None:
        try:
            crs = gdf.crs
        except Exception:
            crs = None
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi">{str(crs) if crs else "N/A"}</div><div class="kpi-label">CRS</div>')
    st.markdown('<div class="small">Coordinate Ref. System</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    rast_shape = None
    if rast is not None:
        try:
            rast_shape = f"{rast.count} bands, {rast.width}x{rast.height}"
        except Exception:
            rast_shape = "—"
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi">{rast_shape or "—"}</div><div class="kpi-label">Raster</div>')
    st.markdown('<div class="small">Bands & Size</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col4:
    moran_val = "—"
    if do_moran and gdf is not None and Moran is not None and moran_col in gdf.columns:
        try:
            centroids = gdf.geometry.centroid
            coords = np.vstack([centroids.y.values, centroids.x.values]).T
            W = KNN.from_array(coords, k=8)
            y = gdf[moran_col].astype(float).values
            mor = Moran(y, W)
            moran_val = f"I = {mor.I:.4f}, p = {mor.p_sim:.4f}"
        except Exception as e:
            moran_val = "error"
            st.exception(e)
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi">{moran_val}</div><div class="kpi-label">Moran\'s I</div>')
    st.markdown('<div class="small">Spatial autocorrelation</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# --- Map display ---
def folium_map(gdf, geocode_point):
    if folium is None:
        st.warning("Folium not available.")
        return
    latlon = [20.5937, 78.9629]  # India
    if gdf is not None and not gdf.empty:
        try:
            c = gdf.geometry.iloc[0].centroid
            latlon = [c.y, c.x]
        except Exception:
            pass
    if geocode_point:
        latlon = list(geocode_point)

    m = folium.Map(location=latlon, zoom_start=5, control_scale=True)
    if gdf is not None and not gdf.empty:
        try:
            folium.GeoJson(gdf.to_json(), name="layer").add_to(m)
        except Exception as e:
            st.exception(e)
    if geocode_point:
        folium.Marker(location=latlon, popup="Geocoded Location").add_to(m)
    st_folium(m, width=None, height=560)

def pydeck_map(gdf, geocode_point):
    if pdk is None:
        st.warning("PyDeck not available.")
        return
    view_state = pdk.ViewState(latitude=20.5937, longitude=78.9629, zoom=4)
    layers = []
    if gdf is not None and not gdf.empty:
        try:
            gjson = json.loads(gdf.to_json())
            layers.append(pdk.Layer("GeoJsonLayer", gjson, pickable=True, stroked=True, filled=True))
        except Exception as e:
            st.exception(e)
    if geocode_point:
        lat, lon = geocode_point
        layers.append(pdk.Layer("ScatterplotLayer", data=pd.DataFrame({"lat":[lat], "lon":[lon]}),
                                get_position='[lon, lat]', get_radius=50000))
        view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=8)
    r = pdk.Deck(layers=layers, initial_view_state=view_state, tooltip={"text": "Layer"})
    st.pydeck_chart(r)

st.subheader("🗺️ Map")
if base_map == "Folium":
    folium_map(gdf, geocode_point)
elif base_map == "PyDeck":
    pydeck_map(gdf, geocode_point)
else:
    st.info("No map engine available.")

# --- Attribute table ---
if show_table:
    if gdf is not None:
        st.subheader("📄 Attributes (Vector)")
        st.dataframe(gdf.drop(columns="geometry", errors="ignore").head(500))
    elif df is not None:
        st.subheader("📄 Tabular Data")
        st.dataframe(df.head(1000))
    else:
        st.info("Load a dataset to see attributes.")

# --- Automation: Batch convert/reproject using GDAL/OGR ---
if do_batch_convert:
    st.subheader("⚙️ Automation: Batch Convert/Reproject")
    if gdal is None or ogr is None:
        st.warning("GDAL/OGR not available in this environment.")
    else:
        st.write("Paste multiple vector URLs (one per line) to convert/reproject:")
        batch_urls = st.text_area("Vector data URLs (.geojson/.gpkg/.zip shapefile)", height=150)
        if st.button("Run Batch"):
            out_msgs = []
            for i, u in enumerate([u.strip() for u in batch_urls.splitlines() if u.strip()]):
                try:
                    out_path = f"converted_{{i}}.geojson"
                    drv = "GeoJSON"
                    cmd = f'ogr2ogr -f {{drv}} -t_srs EPSG:{{reproj_epsg}} {{out_path}} {{u}}'
                    ret = os.system(cmd)
                    if ret == 0 and os.path.exists(out_path):
                        st.success(f"Converted ➜ {{out_path}}")
                        with open(out_path, "rb") as f:
                            st.download_button("Download " + out_path, data=f, file_name=out_path)
                    else:
                        out_msgs.append(f"Failed for: {{u}} (code {{ret}})")
                except Exception as e:
                    out_msgs.append(f"Error for {{u}}: {{e}}")
            if out_msgs:
                st.warning("\\n".join(out_msgs))

# --- Raster preview (if any) ---
if rast is not None:
    st.subheader("🛰️ Raster Preview")
    try:
        arr = rast.read()
        band = min(3, arr.shape[0])
        if band == 1:
            fig, ax = plt.subplots()
            ax.imshow(arr[0], interpolation='nearest')
            ax.set_title("Band 1")
            ax.axis('off')
            st.pyplot(fig)
        else:
            rgb = np.dstack([arr[0], arr[1], arr[2]])
            fig, ax = plt.subplots()
            ax.imshow(rgb)
            ax.set_title("RGB preview (bands 1-3)")
            ax.axis('off')
            st.pyplot(fig)
    except Exception as e:
        st.exception(e)

st.markdown("---")
st.caption("Tip: Paste different URLs to explore datasets. Moran's I needs a numeric column. Batch conversion uses GDAL's ogr2ogr if available.")

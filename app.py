# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
from urllib.error import URLError
import requests
import os

st.set_page_config(page_title="Housing Maps Dashboard", layout="wide")

# ---------------------------
# Styling and helper methods
# ---------------------------
# chart colors and theme variables
COLORS = ["#00C767", "#0B3D0B"]
PIE_COLORS = ["#1B5E20", "#2E7D32", "#66BB6A", "#A5D6A7"]
BG_PAPER = "rgba(0,0,0,0)"
BG_PLOT = "rgba(0,0,0,0)"
FONT_COL = "#FFFFFF"
GRID_COL = "rgba(102,187,106,0.15)"

def apply_dark_layout(fig, show_legend=True):
    """Apply the dashboard dark theme to a Plotly figure."""
    fig.update_layout(
        template="simple_white",
        paper_bgcolor=BG_PAPER,
        plot_bgcolor=BG_PLOT,
        font=dict(color=FONT_COL),
        legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0) if show_legend else dict(),
        margin=dict(t=60, r=20, b=40, l=60)
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COL, zeroline=False, linecolor="#2E7D32")
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COL, zeroline=False, linecolor="#2E7D32")
    return fig

st.markdown("""
<style>
.stApp, [data-testid="stSidebar"] { background-color: #000000; color: #FFFFFF; }
a { color: #66BB6A !important; }
[data-testid="stSlider"] [data-baseweb="slider"] > div:nth-child(1) { background-color: rgba(11,61,11,0.25) !important; }
[data-testid="stSlider"] [data-baseweb="slider"] > div [class*="inner"] { background-color: #0B3D0B !important; }
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background-color: #0B3D0B !important; border: 2px solid #0B3D0B !important; box-shadow: 0 0 0 4px rgba(11,61,11,0.25) !important;
}
[data-testid="stMetric"] { background: transparent; border: none; padding: 14px 16px; }
[data-testid="stMetric"] [data-testid="stMetricLabel"],
[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #FFFFFF !important; }
</style>
""", unsafe_allow_html=True)

# human-readable number format
def human_format(num):
    for unit in ["", "K", "M", "B", "T"]:
        if abs(num) < 1000.0:
            return f"{num:3.1f}{unit}"
        num /= 1000.0
    return f"{num:.1f}P"

# sampling helper to avoid plotting too many points
def sample_df_for_map(df, max_points=5000, random_state=42):
    if len(df) > max_points:
        return df.sample(max_points, random_state=random_state)
    return df

# ---------------------------
# Load data (cached)
# ---------------------------

def load_data_from_drive():
    """Load and preprocess the cleaned housing dataset directly from Google Drive."""
    
    # Direct download link from Google Drive
    DATA_URL = "https://drive.google.com/uc?id=1uqbolYGFffYAdKU9J8d5ZRBh8Pmk8aSl"
    
    # Download the CSV into memory
    r = requests.get(DATA_URL)
    r.raise_for_status()  # Raise error if download failed
    
    # Read CSV directly from bytes
    df = pd.read_csv(BytesIO(r.content))
    
    # Clean column names
    df.columns = df.columns.str.strip().str.lower()  # remove spaces & lowercase
    
    # Ensure numeric columns
    numeric_cols = ["price", "living_space", "land_space", "price_per_unit", "bedroom_number", "bathroom_number"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Ensure is_owned_by_zillow numeric (0/1)
    if "is_owned_by_zillow" in df.columns:
        df["is_owned_by_zillow"] = pd.to_numeric(df["is_owned_by_zillow"], errors="coerce").fillna(0).astype(int)
    else:
        df["is_owned_by_zillow"] = 0
    
    # Parse RunDate if present
    if "rundate" in df.columns:
        df["rundate"] = pd.to_datetime(df["rundate"], errors="coerce")
    
    # Normalize postcode to string
    if "postcode" in df.columns:
        df["postcode"] = df["postcode"].astype(str).str.zfill(5)
    
    return df

# Load dataset
df = load_data_from_drive()
st.success(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")
# ---------------------------
# Sidebar filters
# ---------------------------
st.sidebar.header("Filters")

# state filter selector
states = sorted(df["state"].dropna().unique().tolist())
state_options = ["All"] + states
selected_state = st.sidebar.selectbox("State", state_options, index=0)

# listing age filter (use listing_age if exists)
if "listing_age" in df.columns:
    max_age = int(df["listing_age"].dropna().max()) if df["listing_age"].notna().any() else 1000
    age_range = st.sidebar.slider("Listing age (days)", 0, max(365, max_age), (0, 365))
else:
    age_range = (0, 365)

# price range filter
min_price, max_price = int(df["price"].min(skipna=True) or 0), int(df["price"].max(skipna=True) or 1)
price_range = st.sidebar.slider("Price Range ($)", min_value=min_price, max_value=max_price, value=(min_price, max_price), step=10000)

# apply filters to the base dataframe used across plots
mask = (
    df["price"].between(price_range[0], price_range[1])
)
if selected_state != "All":
    mask &= (df["state"] == selected_state)
if "listing_age" in df.columns:
    mask &= df["listing_age"].between(age_range[0], age_range[1])

filtered = df[mask].copy()

# ---------------------------
# Header and KPIs
# ---------------------------
st.title("Housing Maps Dashboard")

st.markdown("### Overview metrics for the selected filters")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Listings", human_format(filtered["property_id"].nunique()))
c2.metric("Avg Price", f"${human_format(filtered['price'].mean() if filtered['price'].notna().any() else 0)}")
c3.metric("Median Price", f"${human_format(filtered['price'].median() if filtered['price'].notna().any() else 0)}")
c4.metric("Listings with coords", human_format(filtered.dropna(subset=["latitude", "longitude"]).shape[0]))

# ---------------------------
# Row: State-level average price choropleth
# ---------------------------
st.subheader("Average Price by State")
state_avg = df.groupby("state", as_index=False)["price"].mean().reset_index(drop=True)
fig_state = px.choropleth(
    state_avg,
    locations="state",
    locationmode="USA-states",
    color="price",
    scope="usa",
    color_continuous_scale="OrRd",
    labels={"price": "Average Price"}
)
fig_state.update_layout(title_text="Average Price per State")
st.plotly_chart(apply_dark_layout(fig_state, show_legend=False), use_container_width=True)

# ---------------------------
# Row: Scatter map for selected state (or full USA if All)
# ---------------------------
st.subheader("Properties Map (color by price, size by living space)")
gdf_clean = filtered.dropna(subset=["latitude", "longitude", "living_space"]).copy()

# ensure living_space and price numeric
gdf_clean["living_space"] = pd.to_numeric(gdf_clean["living_space"], errors="coerce")
gdf_clean["price"] = pd.to_numeric(gdf_clean["price"], errors="coerce")

# sample for performance
gdf_plot = sample_df_for_map(gdf_clean, max_points=6000)

map_zoom = 6 if selected_state != "All" else 3
center = {"lat": gdf_plot["latitude"].mean() if not gdf_plot["latitude"].isna().all() else 39,
          "lon": gdf_plot["longitude"].mean() if not gdf_plot["longitude"].isna().all() else -98}

fig_props = px.scatter_mapbox(
    gdf_plot,
    lat="latitude",
    lon="longitude",
    color="price",
    size="living_space",
    hover_data=["address", "price", "bedroom_number", "bathroom_number", "property_type"],
    zoom=map_zoom,
    center=center,
    mapbox_style="carto-positron",
    color_continuous_scale="Viridis",
    size_max=18,
    title=f"Properties in {'All States' if selected_state=='All' else selected_state}"
)
st.plotly_chart(apply_dark_layout(fig_props), use_container_width=True)

# ---------------------------
# Row: ZIP choropleth for selected state (if geojson available)
# ---------------------------
st.subheader("Average Price by ZIP code (state-level)")

if selected_state != "All":
    state_data = filtered[filtered["state"] == selected_state]
    if not state_data.empty:
        zip_avg = state_data.groupby("postcode", as_index=False)["price"].mean()
        zip_avg["postcode"] = zip_avg["postcode"].astype(str).str.zfill(5)
        
        # فقط Illinois أو إذا توفر GeoJSON آخر
        if selected_state == "IL":
            geojson_url = "https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON/master/il_illinois_zip_codes_geo.min.json"
            
            fig_zip = px.choropleth(
                zip_avg,
                geojson=geojson_url,
                locations="postcode",
                featureidkey="properties.ZCTA5CE10",
                color="price",
                hover_name="postcode",
                hover_data={"price": True, "postcode": True},
                color_continuous_scale="OrRd",
                title=f"Average Price by ZIP code in {selected_state}"
            )
            fig_zip.update_geos(fitbounds="locations", visible=False)
            st.plotly_chart(apply_dark_layout(fig_zip), use_container_width=True)
        else:
            st.info("ZIP-level GeoJSON not available for this state.")
    else:
        st.info("No data for selected state.")
else:
    st.info("Select a specific state to show ZIP-level prices.")


# ---------------------------
# Row: Property types and status maps (state-filterable)
# ---------------------------
st.subheader("Property Type and Status Distribution (map)")

if selected_state != "All":
    state_df = filtered[filtered["state"] == selected_state].copy()
else:
    state_df = filtered.copy()

# property types scatter
if not state_df.dropna(subset=["latitude", "longitude"]).empty:
    samp_types = sample_df_for_map(state_df.dropna(subset=["latitude", "longitude"]), max_points=5000)
    fig_type = px.scatter_mapbox(
        samp_types,
        lat="latitude", lon="longitude",
        color="property_type",
        hover_name="address",
        size="price",
        zoom=map_zoom,
        center=center,
        mapbox_style="open-street-map",
        title=f"Property Types Distribution in {'All States' if selected_state=='All' else selected_state}"
    )
    st.plotly_chart(apply_dark_layout(fig_type), use_container_width=True)

    # property status scatter
    fig_status = px.scatter_mapbox(
        samp_types,
        lat="latitude", lon="longitude",
        color="property_status",
        hover_name="address",
        size="price",
        zoom=map_zoom,
        center=center,
        mapbox_style="open-street-map",
        title=f"Property Status Distribution in {'All States' if selected_state=='All' else selected_state}"
    )
    st.plotly_chart(apply_dark_layout(fig_status), use_container_width=True)
else:
    st.info("No geo-located properties available for Property Type/Status maps for the selected filters.")

# ---------------------------
# Row: Bedrooms bubble map
# ---------------------------
st.subheader("Bedroom Count Bubble Map")
if "bedroom_number" in state_df.columns:
    state_df["bedroom_number"] = pd.to_numeric(state_df["bedroom_number"], errors="coerce")
    beds_df = state_df.dropna(subset=["latitude", "longitude", "bedroom_number"]).copy()
    if not beds_df.empty:
        beds_sample = sample_df_for_map(beds_df, max_points=5000)
        fig_beds = px.scatter_mapbox(
            beds_sample,
            lat="latitude",
            lon="longitude",
            size="bedroom_number",
            color="bedroom_number",
            hover_name="address",
            mapbox_style="open-street-map",
            zoom=map_zoom,
            center=center,
            title=f"Number of Bedrooms in {'All States' if selected_state=='All' else selected_state} (Bubble Map)"
        )
        st.plotly_chart(apply_dark_layout(fig_beds), use_container_width=True)
    else:
        st.info("No bedroom data with coordinates for the current selection.")
else:
    st.info("No bedroom_number column available in dataset.")

# ---------------------------
# Row: Average Price & Bedrooms by State (dual-choropleth style)
# ---------------------------
st.subheader("Average Price and Bedrooms by State")

# compute state-level stats from full dataset (not filtered) to keep choropleth consistent
state_stats = df.groupby("state").agg(
    avg_bedrooms=("bedroom_number", "mean"),
    avg_price=("price", "mean")
).reset_index()

fig_dual = go.Figure()
fig_dual.add_trace(go.Choropleth(
    locations=state_stats["state"],
    z=state_stats["avg_price"],
    locationmode="USA-states",
    colorscale="OrRd",
    colorbar=dict(title="Avg Price", x=0.9),
    name="Average Price",
    hovertext=state_stats.apply(
        lambda x: f"State: {x['state']}<br>Avg Price: {x['avg_price']:.0f}<br>Avg Bedrooms: {x['avg_bedrooms']:.2f}",
        axis=1
    ),
    hoverinfo="text"
))
fig_dual.add_trace(go.Choropleth(
    locations=state_stats["state"],
    z=state_stats["avg_bedrooms"],
    locationmode="USA-states",
    colorscale="Blues",
    colorbar=dict(title="Avg Bedrooms", x=1.05),
    name="Average Bedrooms",
    showscale=True,
    hoverinfo="skip"
))
fig_dual.update_layout(title="Average Price & Bedrooms by State", geo_scope="usa")
st.plotly_chart(apply_dark_layout(fig_dual), use_container_width=True)

# ---------------------------
# Row: Top Agency by State (excluding Unknown)
# ---------------------------
st.subheader("Top Agency by State ")

df_valid_agency = df[df["agency_name"].notna() & (df["agency_name"] != "Unknown")].copy()
if not df_valid_agency.empty:
    agency_counts = (
        df_valid_agency.groupby(["state", "agency_name"])
        .size()
        .reset_index(name="count")
    )
    idx = agency_counts.groupby("state")["count"].idxmax()
    top_agencies = agency_counts.loc[idx].reset_index(drop=True)

    fig_agency = px.choropleth(
        top_agencies,
        locations="state",
        locationmode="USA-states",
        color="agency_name",
        hover_name="agency_name",
        hover_data=["count"],
        scope="usa",
        title="Top Agency by State"
    )
    st.plotly_chart(apply_dark_layout(fig_agency), use_container_width=True)
else:
    st.info("No valid agency_name data (non-Unknown) to compute top agencies.")

# ---------------------------
# Row: City-level Zillow dominance
# ---------------------------
st.subheader("City-level Zillow Ownership Dominance")

city_stats = (
    df.groupby(["city", "state"])
    .agg(
        total=("property_id", "count"),
        zillow_owned=("is_owned_by_zillow", "sum"),
        lat=("latitude", "mean"),
        lon=("longitude", "mean")
    )
    .reset_index()
)
city_stats = city_stats.dropna(subset=["lat", "lon"])  # need coords for map
city_stats["zillow_ratio"] = city_stats["zillow_owned"] / city_stats["total"]
# threshold set at 0.1 (10%) as you used earlier
city_stats["dominance"] = city_stats["zillow_ratio"].apply(lambda x: "Zillow Dominant" if x > 0.1 else "Non-Zillow Dominant")

# apply state filter to city-level map if a state is selected
if selected_state != "All":
    city_plot = city_stats[city_stats["state"] == selected_state].copy()
else:
    city_plot = city_stats.copy()

if not city_plot.empty:
    fig_city = px.scatter_mapbox(
        city_plot,
        lat="lat",
        lon="lon",
        color="dominance",
        hover_name="city",
        hover_data=["state", "total", "zillow_owned", "zillow_ratio"],
        zoom=map_zoom,
        center=center,
        mapbox_style="carto-positron",
        color_discrete_map={"Zillow Dominant": "blue", "Non-Zillow Dominant": "red"},
        title="City-level Zillow Ownership Dominance"
    )
    st.plotly_chart(apply_dark_layout(fig_city), use_container_width=True)
else:
    st.info("No city-level Zillow data for the selected state/filters.")

# ---------------------------
# Small vs Large Living Spaces Map
# ---------------------------

st.subheader("Small vs Large Living Spaces Map")

if "living_space" in filtered.columns:
    df_space = filtered.dropna(subset=["latitude", "longitude", "living_space"]).copy()
    if not df_space.empty:
        df_space["space_category"] = np.where(df_space["living_space"] < 1000, "Small (<1000 sqft)", "Large (>=1000 sqft)")
        df_space_plot = sample_df_for_map(df_space, max_points=5000)

        fig_space = px.scatter_mapbox(
            df_space_plot,
            lat="latitude",
            lon="longitude",
            color="space_category",
            hover_name="address",
            hover_data=["living_space", "price", "bedroom_number"],
            size="price",
            zoom=map_zoom,
            center=center,
            mapbox_style="carto-positron",
            color_discrete_map={"Small (<1000 sqft)": "orange", "Large (>=1000 sqft)": "green"},
            title=f"Small vs Large Living Spaces in {'All States' if selected_state=='All' else selected_state}"
        )
        st.plotly_chart(apply_dark_layout(fig_space), use_container_width=True)
    else:
        st.info("No properties with living_space and coordinates for the current selection.")


# ---------------------------
# Footer note
# ---------------------------
st.markdown("---")
st.markdown("Dashboard built with Streamlit, Plotly and Geo data. Use the sidebar to filter state, listing age and price range.")





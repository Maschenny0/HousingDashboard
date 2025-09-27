import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import glob

st.set_page_config(page_title="Housing Maps Dashboard", layout="wide")

COLOR_SCALE = "YlGnBu"
COLOR_CATEG = px.colors.qualitative.Set2
BG_PAPER = "rgba(0,0,0,0)"
BG_PLOT = "rgba(0,0,0,0)"
FONT_COL = "#FFFFFF"
GRID_COL = "rgba(255,255,255,0.15)"

def apply_dark_layout(fig, show_legend=True):
    fig.update_layout(
        template="simple_white",
        paper_bgcolor=BG_PAPER,
        plot_bgcolor=BG_PLOT,
        font=dict(color=FONT_COL),
        margin=dict(t=60, r=40, b=40, l=60),
        legend=dict(
            orientation="v",
            y=0.5,
            x=1.05,
            xanchor="left",
            yanchor="middle",
            font=dict(color="white")
        ) if show_legend else dict(),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COL, zeroline=False, linecolor="#444")
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COL, zeroline=False, linecolor="#444")
    return fig

st.markdown("""
<style>
.stApp, [data-testid="stSidebar"] { background-color: #000000; color: #FFFFFF; }
a { color: #42A5F5 !important; }
[data-testid="stMetric"] [data-testid="stMetricLabel"],
[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #FFFFFF !important; }
</style>
""", unsafe_allow_html=True)

def human_format(num):
    for unit in ["", "K", "M", "B", "T"]:
        if abs(num) < 1000.0:
            return f"{num:3.1f}{unit}"
        num /= 1000.0
    return f"{num:.1f}P"

def sample_df_for_map(df, max_points=5000, random_state=42):
    if len(df) > max_points:
        return df.sample(max_points, random_state=random_state)
    return df

@st.cache_data(show_spinner=False)
def load_data(folder="data"):
    files = sorted(glob.glob(f"{folder}/cleaned_part_*.csv"))
    if not files:
        return pd.DataFrame()
    df_list = [pd.read_csv(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    if "RunDate" in df.columns:
        df["RunDate"] = pd.to_datetime(df["RunDate"], errors="coerce")
    numeric_cols = ["price", "living_space", "land_space", "price_per_unit", "bedroom_number", "bathroom_number"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "is_owned_by_zillow" in df.columns:
        df["is_owned_by_zillow"] = pd.to_numeric(df["is_owned_by_zillow"], errors="coerce").fillna(0).astype(int)
    return df

df = load_data()

if df.empty:
    st.error("No data files found in 'data' folder. Please add CSV files like 'cleaned_part_1.csv'.")
    st.stop()

st.success(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
st.sidebar.header("Filters")

if "state" in df.columns:
    states = sorted(df["state"].dropna().unique().tolist())
    state_options = ["All"] + states
    selected_state = st.sidebar.selectbox("State", state_options, index=0)
else:
    selected_state = "All"

if "listing_age" in df.columns:
    max_age = int(df["listing_age"].dropna().max()) if df["listing_age"].notna().any() else 1000
    age_range = st.sidebar.slider("Listing age (days)", 0, max_age, (0, max_age))
else:
    age_range = (0, 1000)

price_min = int(df["price"].dropna().min()) if "price" in df.columns and df["price"].notna().any() else 0
price_max = int(df["price"].dropna().max()) if "price" in df.columns and df["price"].notna().any() else 1000000
price_range = st.sidebar.slider("Price range", price_min, price_max, (price_min, price_max))

df_filtered = df.copy()
if selected_state != "All" and "state" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["state"] == selected_state]
if "listing_age" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["listing_age"].between(age_range[0], age_range[1])]
if "price" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["price"].between(price_range[0], price_range[1])]

st.title("Housing Maps Dashboard")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Listings", human_format(df_filtered["property_id"].nunique() if "property_id" in df_filtered.columns else 0))
c2.metric("Avg Price", f"${human_format(df_filtered['price'].mean() if 'price' in df_filtered.columns and df_filtered['price'].notna().any() else 0)}")
c3.metric("Median Price", f"${human_format(df_filtered['price'].median() if 'price' in df_filtered.columns and df_filtered['price'].notna().any() else 0)}")
c4.metric("Listings with coords", human_format(df_filtered.dropna(subset=["latitude", "longitude"]).shape[0] if "latitude" in df_filtered.columns else 0))

if not df_filtered.empty:
    center = {"lat": df_filtered["latitude"].mean(), "lon": df_filtered["longitude"].mean()} if "latitude" in df_filtered.columns else {"lat": 37.8, "lon": -96}
    map_zoom = 4 if selected_state == "All" else 6

    if "state" in df.columns and "price" in df.columns:
        st.header("Average Price by State")
        state_avg = df.groupby("state", as_index=False)["price"].mean()
        fig_state = px.choropleth(
            state_avg, locations="state", locationmode="USA-states",
            color="price", scope="usa", color_continuous_scale=COLOR_SCALE
        )
        st.plotly_chart(apply_dark_layout(fig_state, show_legend=False), use_container_width=True)

    if all(col in df_filtered.columns for col in ["latitude", "longitude", "living_space", "price"]):
        st.header("Properties Map")
        gdf_clean = df_filtered.dropna(subset=["latitude", "longitude", "living_space"]).copy()
        gdf_clean["living_space"] = pd.to_numeric(gdf_clean["living_space"], errors="coerce")
        gdf_clean["price"] = pd.to_numeric(gdf_clean["price"], errors="coerce")
        gdf_plot = sample_df_for_map(gdf_clean, max_points=6000)
        fig_props = px.scatter_mapbox(
            gdf_plot, lat="latitude", lon="longitude", color="price", size="living_space",
            hover_data=["address", "price", "bedroom_number", "bathroom_number", "property_type"] if "address" in gdf_plot.columns else None,
            zoom=map_zoom, center=center, mapbox_style="carto-positron",
            color_continuous_scale=COLOR_SCALE, size_max=18
        )
        st.plotly_chart(apply_dark_layout(fig_props), use_container_width=True)

    if all(col in df_filtered.columns for col in ["latitude", "longitude", "property_type"]):
        st.header("Property Types Distribution")
        samp_types = sample_df_for_map(df_filtered.dropna(subset=["latitude", "longitude"]))
        if not samp_types.empty:
            fig_type = px.scatter_mapbox(
                samp_types, lat="latitude", lon="longitude", color="property_type",
                hover_name="address" if "address" in samp_types.columns else None, size="price" if "price" in samp_types.columns else None,
                zoom=map_zoom, center=center, mapbox_style="open-street-map", color_discrete_sequence=COLOR_CATEG
            )
            st.plotly_chart(apply_dark_layout(fig_type, show_legend=True), use_container_width=True)

    if all(col in df_filtered.columns for col in ["latitude", "longitude", "property_status"]):
        st.header("Property Status Distribution")
        samp_status = sample_df_for_map(df_filtered.dropna(subset=["latitude", "longitude"]))
        if not samp_status.empty:
            fig_status = px.scatter_mapbox(
                samp_status, lat="latitude", lon="longitude", color="property_status",
                hover_name="address" if "address" in samp_status.columns else None, size="price" if "price" in samp_status.columns else None,
                zoom=map_zoom, center=center, mapbox_style="open-street-map", color_discrete_sequence=COLOR_CATEG
            )
            st.plotly_chart(apply_dark_layout(fig_status, show_legend=True), use_container_width=True)

    if all(col in df_filtered.columns for col in ["latitude", "longitude", "bedroom_number"]):
        st.header("Bedroom Count Bubble Map")
        df_filtered["bedroom_number"] = pd.to_numeric(df_filtered["bedroom_number"], errors="coerce")
        beds_df = df_filtered.dropna(subset=["latitude", "longitude", "bedroom_number"]).copy()
        if not beds_df.empty:
            beds_sample = sample_df_for_map(beds_df, max_points=5000)
            fig_beds = px.scatter_mapbox(
                beds_sample, lat="latitude", lon="longitude", size="bedroom_number", color="bedroom_number",
                hover_name="address" if "address" in beds_sample.columns else None, mapbox_style="open-street-map", zoom=map_zoom, center=center,
                color_continuous_scale=COLOR_SCALE
            )
            st.plotly_chart(apply_dark_layout(fig_beds), use_container_width=True)

    if all(col in df.columns for col in ["state", "bedroom_number", "price"]):
        st.header("Average Price and Bedrooms by State")
        state_stats = df.groupby("state").agg(
            avg_bedrooms=("bedroom_number", "mean"),
            avg_price=("price", "mean")
        ).reset_index()
        fig_dual = go.Figure()
        fig_dual.add_trace(go.Choropleth(
            locations=state_stats["state"],
            z=state_stats["avg_price"],
            locationmode="USA-states",
            colorscale=COLOR_SCALE,
            colorbar=dict(title="Avg Price", x=1.1, y=0.5, len=0.75),
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
            colorbar=dict(title="Avg Bedrooms", x=1.25, y=0.5, len=0.75),
            name="Average Bedrooms",
            showscale=True,
            hoverinfo="skip"
        ))
        fig_dual.update_layout(title="Average Price & Bedrooms by State", geo_scope="usa")
        st.plotly_chart(apply_dark_layout(fig_dual), use_container_width=True)

    if all(col in df.columns for col in ["state", "agency_name"]):
        st.header("Top Agency by State")
        df_valid_agency = df[df["agency_name"].notna() & (df["agency_name"] != "Unknown")].copy()
        if not df_valid_agency.empty:
            agency_counts = df_valid_agency.groupby(["state", "agency_name"]).size().reset_index(name="count")
            idx = agency_counts.groupby("state")["count"].idxmax()
            top_agencies = agency_counts.loc[idx].reset_index(drop=True)
            fig_agency = px.choropleth(
                top_agencies, locations="state", locationmode="USA-states",
                color="agency_name", hover_name="agency_name", hover_data=["count"],
                scope="usa", color_discrete_sequence=COLOR_CATEG
            )
            st.plotly_chart(apply_dark_layout(fig_agency), use_container_width=True)

    if all(col in df.columns for col in ["city", "state", "is_owned_by_zillow", "property_id", "latitude", "longitude"]):
        st.header("City-level Zillow Ownership Dominance")
        city_stats = df.groupby(["city", "state"]).agg(
            total=("property_id", "count"),
            zillow_owned=("is_owned_by_zillow", "sum"),
            lat=("latitude", "mean"),
            lon=("longitude", "mean")
        ).reset_index()
        city_stats = city_stats.dropna(subset=["lat", "lon"])
        city_stats["zillow_ratio"] = city_stats["zillow_owned"] / city_stats["total"]
        city_stats["dominance"] = city_stats["zillow_ratio"].apply(
            lambda x: "Zillow Dominant" if x > 0.1 else "Non-Zillow Dominant"
        )
        city_plot = city_stats[city_stats["state"] == selected_state].copy() if selected_state != "All" else city_stats.copy()
        if not city_plot.empty:
            fig_city = px.scatter_mapbox(
                city_plot, lat="lat", lon="lon", color="dominance",
                hover_name="city", hover_data=["state", "total", "zillow_owned", "zillow_ratio"],
                zoom=map_zoom, center=center, mapbox_style="carto-positron",
                color_discrete_map={"Zillow Dominant": "#1f77b4", "Non-Zillow Dominant": "#ff7f0e"}
            )
            st.plotly_chart(apply_dark_layout(fig_city), use_container_width=True)

    if all(col in df_filtered.columns for col in ["latitude", "longitude", "living_space", "price"]):
        st.header("Small vs Large Living Spaces Map")
        df_space = df_filtered.dropna(subset=["latitude", "longitude", "living_space"]).copy()
        if not df_space.empty:
            df_space["space_category"] = np.where(
                df_space["living_space"] < 1000, "Small (<1000 sqft)", "Large (>=1000 sqft)"
            )
            df_space_plot = sample_df_for_map(df_space, max_points=5000)
            fig_space = px.scatter_mapbox(
                df_space_plot, lat="latitude", lon="longitude", color="space_category",
                hover_name="address" if "address" in df_space_plot.columns else None,
                hover_data=["living_space", "price", "bedroom_number"] if "bedroom_number" in df_space_plot.columns else None,
                size="price", zoom=map_zoom, center=center, mapbox_style="carto-positron",
                color_discrete_map={"Small (<1000 sqft)": "#66c2a5", "Large (>=1000 sqft)": "#fc8d62"}
            )
            st.plotly_chart(apply_dark_layout(fig_space), use_container_width=True)

st.markdown("---")
st.markdown("Dashboard built with Streamlit, Plotly and Geo data. Source: Kaggle")

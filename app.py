"""
================================================================================
UK WATER QUALITY DASHBOARD - METAL CONCENTRATION & RESOURCE RECOVERY ANALYSIS
================================================================================
Company: HydroStar Europe Ltd.
Purpose: Analyze water quality parameters across UK water sources to identify 
         locations with elevated levels of economically valuable metals and compounds
Version: 3.2 - Production Ready with Enhanced Regional Analysis
Date: January 2025
Dataset: 26 years of UK water quality monitoring (2000-2025)
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import polars as pl
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION SECTION - CUSTOMIZE THESE VALUES
# ============================================================================

# ---------------------------------------------------------------------
# PAGE SETTINGS - Change these to customize the app appearance
# ---------------------------------------------------------------------
PAGE_TITLE = "UK Water Quality Dashboard - HydroStar"
PAGE_ICON = "logo.png"  # Using emoji for deployment reliability
LAYOUT = "wide"  # Options: "wide" or "centered"

# ---------------------------------------------------------------------
# BRAND COLORS - Update these with your company colors
# ---------------------------------------------------------------------
# These colors match HydroStar brand guidelines
BRAND_COLORS = {
    'primary_green': '#a7d730',      # Light green - main brand color
    'secondary_green': '#499823',    # Dark green - for text and accents
    'dark_grey': '#30343c',          # Dark grey - for backgrounds
    'light_grey': '#8c919a',         # Light grey - for supporting elements
    'background': '#0d1116'           # Dark background color
}

# ---------------------------------------------------------------------
# METALS & VALUABLE COMPOUNDS - Based on actual data availability
# ---------------------------------------------------------------------
METALS_OF_INTEREST = [
    'Iron',        # Most abundant, 463K measurements
    'Copper',      # Industrial value, 280K measurements
    'Nickel',      # Battery technology, 228K measurements
    'Lead',        # Toxic metal monitoring, 223K measurements
    'Cadmium',     # Toxic metal, valuable recovery, 231K measurements
    'Manganese',   # Steel production, 50K measurements
    'Mercury',     # Toxic metal monitoring, 101K measurements
    'Barium',      # Heavy metal, 10K measurements
    'Calcium',     # Industrial mineral, 35K measurements
    'Magnesium',   # Essential for hydrogen production, 35K measurements
    'Strontium',   # Trace element, 7.6K measurements
]

# ---------------------------------------------------------------------
# NUTRIENTS & CHEMICALS - High volume parameters
# ---------------------------------------------------------------------
NUTRIENTS_CHEMICALS = [
    'Ammoniacal Nitrogen',  # 1.15M measurements - most common
    'Orthophosphate',       # 423K measurements - fertilizer value
    'Chloride',            # 375K measurements - salinity indicator
    'Nitrate',             # 217K measurements
    'Nitrite',             # 227K measurements
    'Sulphate',            # 50K measurements
    'Alkalinity',          # 21K measurements
    'Cyanide',             # 15K measurements
    'Sulphide',            # 1.9K measurements
    'Bicarbonate',         # 688 measurements
    'Bromide',             # 663 measurements
    'Iodide',              # 79 measurements
]

# ---------------------------------------------------------------------
# PHYSICAL CHEMISTRY PARAMETERS - Environmental measurements
# ---------------------------------------------------------------------
PHYSICAL_CHEMISTRY = ['pH', 'Temperature', 'Conductivity', 'Turbidity']

# ---------------------------------------------------------------------
# UK REGIONS - Geographical divisions
# ---------------------------------------------------------------------
UK_REGIONS = {
    'NE': {'lat_min': 54.0, 'lat_max': 56.0, 'lon_min': -2.0, 'lon_max': 0.5},
    'NW': {'lat_min': 53.0, 'lat_max': 56.0, 'lon_min': -5.0, 'lon_max': -2.0},
    'SE': {'lat_min': 50.0, 'lat_max': 52.5, 'lon_min': -1.0, 'lon_max': 2.0},
    'SW': {'lat_min': 49.5, 'lat_max': 52.0, 'lon_min': -6.0, 'lon_max': -2.0}
}

# Units for parameters (based on actual data)
PARAM_UNITS = {
    'pH': 'phunits',
    'Temperature of Water': 'cel',
    'Conductivity at 20 C': 'uS/cm',
    'Conductivity at 25 C': 'uS/cm',
    'Turbidity': 'ntu',
    'default': 'mg/l'
}

# ---------------------------------------------------------------------
# UK MAP SETTINGS - Adjust to focus on different regions
# ---------------------------------------------------------------------
UK_MAP_CONFIG = {
    'center': {'lat': 52.5, 'lon': -1.5},  # Centered on UK midlands
    'zoom': 5.2,                           # Zoom level for full UK view
    'mapbox_style': 'carto-positron'       # Map style
}

# ============================================================================
# PAGE CONFIGURATION - Sets up the Streamlit app
# ============================================================================
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"  # Sidebar starts open
)

# ============================================================================
# CUSTOM STYLING - Applies HydroStar branding to the dashboard
# ============================================================================
st.markdown(f"""
<style>
    /* Main background color */
    .main {{
        background-color: {BRAND_COLORS['background']};
    }}
    
    /* Button styling with brand colors */
    .stButton>button {{
        background-color: {BRAND_COLORS['secondary_green']};
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }}
    
    /* Button hover effect */
    .stButton>button:hover {{
        background-color: {BRAND_COLORS['primary_green']};
        color: {BRAND_COLORS['dark_grey']};
    }}
    
    /* Metric container styling */
    .metric-container {{
        background-color: white;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid {BRAND_COLORS['primary_green']};
        margin-bottom: 1rem;
    }}
    
    /* Header styling with brand colors */
    h1 {{
        color: #ffffff;
        font-family: 'Hind', sans-serif;
        border-bottom: 3px solid {BRAND_COLORS['primary_green']};
        padding-bottom: 10px;
    }}
    
    h2, h3 {{
        color: {BRAND_COLORS['primary_green']};
        font-family: 'Hind', sans-serif;
    }}
    
    /* Dataframe styling */
    .dataframe {{
        font-size: 12px;
    }}

    /* Tag styling for multiselect and filters */
    div[data-testid="stMultiSelect"] span[data-baseweb="tag"] {{
        background-color: {BRAND_COLORS['primary_green']} !important;
    }}

    /* Close button (x) in tags */
    div[data-testid="stMultiSelect"] span[data-baseweb="tag"]:hover {{
        background-color: {BRAND_COLORS['secondary_green']} !important;
    }}

    /* Hover state for clickable elements */
    .element-container:hover button,
    .stButton > button:hover,
    div[data-testid="stMultiSelect"]:hover,
    div[role="listbox"] div[role="option"]:hover {{
        border-color: {BRAND_COLORS['primary_green']} !important;
        color: {BRAND_COLORS['primary_green']} !important;
    }}

    /* Radio buttons active state */
    .stRadio > div[role="radiogroup"] label[data-checked="true"] {{
        color: {BRAND_COLORS['primary_green']} !important;
    }}
    
    /* Checkbox active state */
    .stCheckbox > label > div[data-checked="true"] {{
        background-color: {BRAND_COLORS['primary_green']} !important;
        border-color: {BRAND_COLORS['primary_green']} !important;
    }}

    /* Slider handle and track */
    .stSlider input[type="range"] {{
        accent-color: {BRAND_COLORS['primary_green']};
    }}

    /* Custom styling for multiselect pills/tags */
    div[data-testid="stMultiSelect"] div[data-baseweb="tag"] {{
        background-color: {BRAND_COLORS['primary_green']} !important;
    }}

    /* Style for the delete/close button in tags */
    div[data-testid="stMultiSelect"] div[data-baseweb="tag"] span[role="button"] {{
        color: white !important;
        background-color: transparent !important;
    }}

    /* Hover effect for delete/close button */
    div[data-testid="stMultiSelect"] div[data-baseweb="tag"] span[role="button"]:hover {{
        color: #e6e6e6 !important;
    }}

    /* Style for selectbox and multiselect when focused/active */
    div[data-testid="stMultiSelect"] div[aria-expanded="true"] {{
        border-color: {BRAND_COLORS['primary_green']} !important;
    }}

    /* Override default red color for various interactive elements */
    .stSelectbox [data-baseweb="select"] div[aria-selected="true"],
    .stMultiSelect [data-baseweb="select"] div[aria-selected="true"],
    .stMultiSelect [data-baseweb="tag"] {{
        background-color: {BRAND_COLORS['primary_green']} !important;
        color: white !important;
    }}

    /* Style for the expander arrow */
    .streamlit-expanderHeader:hover {{
        color: {BRAND_COLORS['primary_green']} !important;
    }}

    /* Style for buttons */
    .stButton > button {{
        background-color: {BRAND_COLORS['primary_green']};
        color: white;
    }}

    /* Style for toggle buttons */
    .stToggleButton > button:hover {{
        background-color: {BRAND_COLORS['primary_green']} !important;
        color: white !important;
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS - Utility functions for the dashboard
# ============================================================================

def assign_region(lat, lon):
    """
    Assigns UK region based on latitude and longitude.
    
    Args:
        lat: Latitude
        lon: Longitude
    
    Returns:
        Region code (NE, NW, SE, SW) or 'Other'
    """
    for region, bounds in UK_REGIONS.items():
        if (bounds['lat_min'] <= lat <= bounds['lat_max'] and 
            bounds['lon_min'] <= lon <= bounds['lon_max']):
            return region
    return 'Other'

def get_display_unit(param_name):
    """
    Returns the appropriate unit for display based on parameter name.
    
    Args:
        param_name: The parameter name from the dataset
    
    Returns:
        String with the unit (e.g., "mg/l", "°C", "pH")
    """
    if 'pH' in param_name:
        return ''  # pH has no units
    elif 'Temperature' in param_name:
        return '°C'
    elif 'Conductivity' in param_name:
        return 'µS/cm'
    elif 'Turbidity' in param_name:
        return 'NTU'
    else:
        return 'mg/l'  # Default for most parameters

def format_number(num):
    """Format large numbers with K, M suffixes"""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return str(int(num))

# ============================================================================
# DATA LOADING FUNCTIONS - Handle data import and caching
# ============================================================================

@st.cache_data(show_spinner=False)
def load_data():
    """
    Loads the water quality data with caching for performance.
    Returns:
        pandas DataFrame with the water quality data
    """
    try:
        # Load via Polars (no Arrow build), then to pandas
        df = pl.read_parquet('ea_26_years_beta_only.parquet').to_pandas()
        # Ensure datetime column is properly formatted
        df['Date'] = pd.to_datetime(df['Date'])
        # Add YearMonth if not present
        if 'YearMonth' not in df.columns:
            df['YearMonth'] = df['Date'].dt.to_period('M')
        # Add Region column based on coordinates
        df['Region'] = df.apply(lambda row: assign_region(row['Latitude'], row['Longitude']), axis=1)
        return df
    except FileNotFoundError:
        st.error("Data file 'ea_26_years_beta_only.parquet' not found!")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

@st.cache_data(show_spinner=False)
def get_parameter_list(df):
    """
    Gets unique parameters and categorizes them.
    Returns:
        Dictionary with categorized parameters
    """
    all_params = df['Parameter'].unique()
    analytes = []
    physical = []
    for param in all_params:
        param_lower = param.lower()
        is_analyte = False
        for metal in METALS_OF_INTEREST:
            if metal.lower() in param_lower:
                if param not in analytes:
                    analytes.append(param)
                is_analyte = True
                break
        if not is_analyte:
            for nutrient in NUTRIENTS_CHEMICALS:
                if nutrient.lower() in param_lower:
                    if param not in analytes:
                        analytes.append(param)
                    is_analyte = True
                    break
        if not is_analyte:
            if 'ph' == param_lower:
                physical.append(param)
            elif 'temperature' in param_lower:
                physical.append(param)
            elif 'conductivity' in param_lower:
                physical.append(param)
            elif 'turbidity' in param_lower:
                physical.append(param)
    return {
        'analytes': sorted(analytes),
        'physical': sorted(physical),
        'all': sorted(all_params)
    }

# =========================================================================
# DATA ANALYSIS FUNCTIONS - Process and analyze the data
# =========================================================================

@st.cache_data(show_spinner=False)
def get_top_sampling_points(df, selected_param, n=10, regions=None):
    """
    Identifies the top N sampling points with highest concentrations.
    Now properly filters to get top N from selected regions only.
    Args:
        df: Filtered dataframe
        selected_param: Parameter to analyze
        n: Number of top points to return
        regions: List of regions to include (None or contains 'All' = all regions)
    Returns:
        DataFrame with top sampling points
    """
    param_data = df[df['Parameter'] == selected_param].copy()
    if param_data.empty:
        return pd.DataFrame()
    if regions and len(regions) > 0 and 'All' not in regions:
        param_data = param_data[param_data['Region'].isin(regions)]
    if param_data.empty:
        return pd.DataFrame()
    grouped = param_data.groupby(['Sampling Point', 'Water_Source', 'Latitude', 'Longitude', 'Region']).agg({
        'result': ['mean', 'median', 'std', 'count', 'max']
    }).reset_index()
    grouped.columns = ['Sampling Point', 'Water_Source', 'Latitude', 'Longitude', 'Region',
                       'Mean_Concentration', 'Median_Concentration', 'Std_Dev', 'Sample_Count', 'Max_Concentration']
    grouped = grouped[grouped['Sample_Count'] >= 10]
    top_points = grouped.nlargest(n, 'Mean_Concentration')
    top_points['Rank'] = range(1, len(top_points) + 1)
    return top_points

@st.cache_data(show_spinner=False)
def get_top_sites_temporal_data(df, selected_param, top_sites_df):
    """
    Gets temporal data for the top sampling sites.
    Args:
        df: Full dataframe
        selected_param: Selected parameter
        top_sites_df: DataFrame with top sampling sites
    Returns:
        DataFrame with temporal data for top sites
    """
    if top_sites_df.empty:
        return pd.DataFrame()
    top_sites = top_sites_df['Sampling Point'].tolist()
    temporal_data = df[(df['Parameter'] == selected_param) & 
                       (df['Sampling Point'].isin(top_sites))].copy()
    if temporal_data.empty:
        return pd.DataFrame()
    temporal_data = temporal_data.sort_values('Date')
    return temporal_data

@st.cache_data(show_spinner=False)
def get_time_series_data(df, selected_param, aggregation='monthly'):
    """
    Prepares time series data for visualization.
    Args:
        df: Filtered dataframe
        selected_param: Parameter to analyze
        aggregation: 'monthly' or 'yearly'
    Returns:
        DataFrame with time series data
    """
    # Filter for selected parameter
    ts_data = df[df['Parameter'] == selected_param].copy()
    
    if ts_data.empty:
        return pd.DataFrame()
    
    # Set appropriate time period
    if aggregation == 'monthly':
        ts_data['Period'] = ts_data['Date'].dt.to_period('M')
    else:  # yearly
        ts_data['Period'] = ts_data['Date'].dt.to_period('Y')
    
    # Aggregate data
    agg_data = ts_data.groupby('Period').agg({
        'result': ['mean', 'median', 'count']
    }).reset_index()
    
    # Flatten columns
    agg_data.columns = ['Period', 'Mean', 'Median', 'Count']
    
    # Convert Period back to datetime for plotting
    agg_data['Date'] = agg_data['Period'].dt.to_timestamp()
    
    return agg_data

@st.cache_data(show_spinner=False)
def get_source_distribution(df, selected_param, top_sites=None):
    """
    Analyzes distribution across different water sources.
    
    Args:
        df: Dataframe
        selected_param: Selected parameter
        top_sites: Optional list of top sampling sites to filter by
    
    Returns:
        DataFrame with source distribution statistics
    """
    param_data = df[df['Parameter'] == selected_param].copy()
    
    if param_data.empty:
        return pd.DataFrame()
    
    # If top_sites provided, filter to only those sites
    if top_sites is not None and len(top_sites) > 0:
        param_data = param_data[param_data['Sampling Point'].isin(top_sites)]
    
    if param_data.empty:
        return pd.DataFrame()
    
    # Group by water source
    source_stats = param_data.groupby(['Water_Source', 'Source_Category']).agg({
        'result': ['mean', 'median', 'count', 'std', 'max']
    }).reset_index()
    
    source_stats.columns = ['Water_Source', 'Source_Category', 'Mean', 'Median', 
                           'Sample_Count', 'Std_Dev', 'Max']
    
    return source_stats.sort_values('Mean', ascending=False)

# ============================================================================
# VISUALIZATION FUNCTIONS - Create charts and maps
# ============================================================================

def create_top_sites_temporal_chart(temporal_data, selected_param, unit):
    """
    Creates a temporal chart showing concentration over time for top sites.
    Similar to the user's example image showing individual site trends.
    
    Args:
        temporal_data: DataFrame with temporal data for top sites
        selected_param: Selected parameter name
        unit: Unit of measurement
    
    Returns:
        Plotly figure object
    """
    if temporal_data.empty:
        return go.Figure().add_annotation(
            text="No temporal data available for top sites",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    fig = go.Figure()
    
    # Color palette
    colors = px.colors.qualitative.Set2 + px.colors.qualitative.Set3
    
    # Get unique sampling points
    sites = temporal_data['Sampling Point'].unique()
    
    # Add scatter trace for each site
    for i, site in enumerate(sites):
        site_data = temporal_data[temporal_data['Sampling Point'] == site]
        
        fig.add_trace(go.Scatter(
            x=site_data['Date'],
            y=site_data['result'],
            mode='markers+lines',
            name=site[:30] + '...' if len(site) > 30 else site,  # Truncate long names
            marker=dict(
                size=6,
                color=colors[i % len(colors)],
                opacity=0.7
            ),
            line=dict(
                width=1,
                color=colors[i % len(colors)],
                dash='solid'
            ),
            hovertemplate=f"<b>{site}</b><br>Date: %{{x|%Y-%m-%d}}<br>Concentration: %{{y:.3f}} {unit}<extra></extra>"
        ))
    
    # Update layout
    fig.update_layout(
        title=f"Concentration over Time — {selected_param} ({unit})",
        xaxis_title="Time",
        yaxis_title=f"Detected Concentration",
        height=500,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            title=dict(
                text="Sampling Point",
                font=dict(color='white', size=10)
            ),
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(13,17,22,0.8)',  # Matching dark background
            bordercolor=BRAND_COLORS['primary_green'],
            font=dict(color='white', size=10),
            borderwidth=1
        ),
        template="plotly_white",
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray'
        )
    )
    
    return fig

def create_time_series_chart(ts_data, title, unit):
    """
    Creates an interactive time series chart for overall trends.
    
    Args:
        ts_data: Time series dataframe
        title: Chart title
        unit: Unit of measurement
    
    Returns:
        Plotly figure object
    """
    if ts_data.empty:
        return go.Figure().add_annotation(
            text="No time series data available",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    fig = go.Figure()
    
    # Add mean line
    fig.add_trace(go.Scatter(
        x=ts_data['Date'],
        y=ts_data['Mean'],
        mode='lines+markers',
        name='Mean',
        line=dict(width=3, color=BRAND_COLORS['secondary_green']),
        marker=dict(size=6),
        hovertemplate=f"Date: %{{x|%Y-%m}}<br>Mean: %{{y:.3f}} {unit}<br>Samples: %{{customdata}}<extra></extra>",
        customdata=ts_data['Count']
    ))
    
    # Add median line
    fig.add_trace(go.Scatter(
        x=ts_data['Date'],
        y=ts_data['Median'],
        mode='lines+markers',
        name='Median',
        line=dict(width=2, color=BRAND_COLORS['primary_green'], dash='dash'),
        marker=dict(size=4),
        opacity=0.7,
        hovertemplate=f"Date: %{{x|%Y-%m}}<br>Median: %{{y:.3f}} {unit}<extra></extra>"
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=f"Concentration ({unit})",
        height=500,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,  # Moved further down
            xanchor="center",
            x=0.5,
            bgcolor='rgba(13,17,22,0.8)',  # Matching dark background
            bordercolor=BRAND_COLORS['primary_green'],
            font=dict(color='white'),  # White text for better visibility
            borderwidth=1
        ),
        template="plotly_white"
    )
    
    return fig

def create_map_visualization(top_points, selected_param, unit, selected_regions):
    """
    Creates an interactive map showing top sampling locations with HydroStar colors.
    Circle sizes are dramatically different based on concentration ranking.
    
    Args:
        top_points: DataFrame with top sampling points
        selected_param: Selected parameter for title
        unit: Unit of measurement
        selected_regions: Selected regions for display
    
    Returns:
        Plotly figure object
    """
    if top_points.empty:
        return go.Figure().add_annotation(
            text="No location data available for selected parameters",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    # Create map figure
    fig = go.Figure()
    
    # Sort by concentration to ensure proper layering (smallest on top)
    top_points_sorted = top_points.sort_values('Mean_Concentration', ascending=True)
    
    # Calculate size based on ranking
    # Rank 1 (highest conc) gets biggest size, last rank gets smallest
    max_rank = top_points_sorted['Rank'].max()
    
    # Create dramatic size differences based on rank
    def get_marker_size(rank, max_rank):
        # Inverse relationship - lower rank (1) gets bigger size
        # Size ranges from 15 to 70 pixels (increased from 8-40)
        size_range = 55
        min_size = 15
        # Using exponential scaling for more dramatic size differences
        size = min_size + (size_range * ((max_rank - rank) / max_rank) ** 1.5)
        return size
    
    # Apply size calculation
    top_points_sorted['MarkerSize'] = top_points_sorted.apply(
        lambda row: get_marker_size(row['Rank'], max_rank), axis=1
    )
    
    # Sort by size in descending order so larger markers are plotted first (underneath)
    top_points_sorted = top_points_sorted.sort_values('MarkerSize', ascending=False)
    
    # Add small random jitter to help visualize overlapping points
    jitter_scale = 0.002  # Adjust this value to control jitter amount
    top_points_sorted['Latitude_Jittered'] = top_points_sorted['Latitude'] + np.random.normal(0, jitter_scale, len(top_points_sorted))
    top_points_sorted['Longitude_Jittered'] = top_points_sorted['Longitude'] + np.random.normal(0, jitter_scale, len(top_points_sorted))
    
    # Create a single trace with HydroStar colors
    fig.add_trace(go.Scattermapbox(
        lat=top_points_sorted['Latitude_Jittered'],
        lon=top_points_sorted['Longitude_Jittered'],
        mode='markers',
        marker=dict(
            size=top_points_sorted['MarkerSize'],
            color=top_points_sorted['Mean_Concentration'],
            colorscale=[
                [0, BRAND_COLORS['dark_grey']],      # Lowest concentration
                [0.33, BRAND_COLORS['light_grey']],  # Low-medium
                [0.66, BRAND_COLORS['secondary_green']], # Medium-high
                [1, BRAND_COLORS['primary_green']]   # Highest concentration
            ],
            showscale=True,
            colorbar=dict(
                title=dict(
                    text=f"Mean<br>{unit}",
                    font=dict(color='white')
                ),
                thickness=15,
                len=0.7,
                x=1.02,
                bgcolor='rgba(13,17,22,0.8)',  # Matching your dark background
                bordercolor=BRAND_COLORS['primary_green'],
                borderwidth=2,
                tickfont=dict(color='white'),  # Making tick labels white
                outlinecolor=BRAND_COLORS['primary_green']
            ),
            opacity=0.7,  # Reduced opacity to better show overlapping
            symbol='circle',  # Use circle markers for better visibility
            allowoverlap=True  # Ensure all markers are visible even when overlapping
        ),
        text=top_points_sorted.apply(lambda row: 
            f"<b>Rank #{row['Rank']}: {row['Sampling Point']}</b><br>"
            f"Region: {row['Region']}<br>"
            f"Source: {row['Water_Source']}<br>"
            f"Mean: {row['Mean_Concentration']:.3f} {unit}<br>"
            f"Max: {row['Max_Concentration']:.3f} {unit}<br>"
            f"Samples: {int(row['Sample_Count'])}", axis=1),
        hovertemplate='%{text}<extra></extra>',
        name="Sampling Points"
    ))
    
    # Adjust zoom based on selected regions
    if selected_regions and 'All' not in selected_regions and len(selected_regions) > 0:
        # Calculate center and zoom for selected regions
        filtered_points = top_points_sorted[top_points_sorted['Region'].isin(selected_regions)]
        if not filtered_points.empty:
            center_lat = filtered_points['Latitude'].mean()
            center_lon = filtered_points['Longitude'].mean()
            lat_range = filtered_points['Latitude'].max() - filtered_points['Latitude'].min()
            lon_range = filtered_points['Longitude'].max() - filtered_points['Longitude'].min()
            zoom = 6.5 if max(lat_range, lon_range) < 2 else 5.5 if max(lat_range, lon_range) < 3 else 4.5
        else:
            center_lat = UK_MAP_CONFIG['center']['lat']
            center_lon = UK_MAP_CONFIG['center']['lon']
            zoom = UK_MAP_CONFIG['zoom']
    else:
        center_lat = UK_MAP_CONFIG['center']['lat']
        center_lon = UK_MAP_CONFIG['center']['lon']
        zoom = UK_MAP_CONFIG['zoom']
    
    # Update map layout
    fig.update_layout(
        mapbox=dict(
            style='carto-positron',
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom
        ),
        height=600,
        title=f"Top Sampling Locations - {selected_param}",
        margin=dict(r=0, t=40, l=0, b=0),
        showlegend=False,
        font=dict(family='Hind')
    )
    
    return fig

def create_source_distribution_chart(source_data, unit, title_suffix=""):
    """
    Creates a bar chart showing distribution across water sources.
    
    Args:
        source_data: DataFrame with source statistics
        unit: Unit of measurement
        title_suffix: Additional text for title
    
    Returns:
        Plotly figure object
    """
    if source_data.empty:
        return go.Figure().add_annotation(
            text="No source distribution data available",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    # Limit to top 10 sources
    top_sources = source_data.head(10)
    
    fig = go.Figure()
    
    # Add bars for mean concentrations
    fig.add_trace(go.Bar(
        x=top_sources['Water_Source'],
        y=top_sources['Mean'],
        name='Mean Concentration',
        marker_color=BRAND_COLORS['secondary_green'],
        text=top_sources['Mean'].round(3),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Mean: %{y:.3f} ' + unit + '<br>Samples: %{customdata}<extra></extra>',
        customdata=top_sources['Sample_Count']
    ))
    
    # Add max values as scatter
    fig.add_trace(go.Scatter(
        x=top_sources['Water_Source'],
        y=top_sources['Max'],
        mode='markers',
        name='Max Recorded',
        marker=dict(
            size=10,
            color=BRAND_COLORS['primary_green'],
            symbol='diamond'
        ),
        hovertemplate='<b>%{x}</b><br>Max: %{y:.3f} ' + unit + '<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Concentration by Water Source{title_suffix}",
        xaxis_title="Water Source",
        yaxis_title=f"Concentration ({unit})",
        height=600,  # Increased from 400 to 600 for better vertical spacing
        showlegend=True,
        template="plotly_white",
        xaxis_tickangle=-45,
        legend=dict(
            bgcolor='rgba(13,17,22,0.8)',  # Matching dark background
            bordercolor=BRAND_COLORS['primary_green'],
            font=dict(color='white'),
            borderwidth=1,
            y=1.1  # Moved slightly up to avoid overlap with bars
        )
    )
    
    return fig

# ============================================================================
# MAIN APPLICATION - The main dashboard logic
# ============================================================================

def main():
    """
    Main function that runs the Streamlit dashboard.
    """
    
    # ------------------------------------------------------------------------
    # HEADER SECTION - Dashboard title and branding
    # ------------------------------------------------------------------------
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("Beta Water Quality Dashboard")
        st.markdown("**HydroStar Europe Ltd.**")
        st.markdown("*26 Years of Environmental Monitoring Data (2000-2025)*")
    with col2:
        # Using emoji for better deployment reliability
        st.markdown("### ⭐ HYDROSTAR")
    
    st.markdown("---")
    
    # ------------------------------------------------------------------------
    # DATA LOADING - Load and prepare the data
    # ------------------------------------------------------------------------
    with st.spinner("Loading 5.5 million water quality measurements..."):
        df = load_data()
        params_dict = get_parameter_list(df)
    
    # ------------------------------------------------------------------------
    # SIDEBAR CONTROLS - User input controls
    # ------------------------------------------------------------------------
    st.sidebar.title("Dashboard Controls")
    st.sidebar.markdown("---")
    
    # ====================================================================
    # PARAMETER SELECTION - Simplified to Analytes vs Physical Chemistry
    # ====================================================================
    st.sidebar.subheader("Parameter Selection")
    param_type = st.sidebar.radio(
        "Select Parameter Type:",
        ["Analytes", "Physical Chemistry"],
        help="Choose between chemical analytes or physical measurements"
    )
    
    # Single parameter selection based on type
    if param_type == "Analytes":
        available_params = params_dict['analytes']
        selected_param = st.sidebar.selectbox(
            "Select Analyte:",
            available_params,
            index=available_params.index('Chloride') if 'Chloride' in available_params else 0,
            help="Choose a specific analyte to analyze"
        )
    else:  # Physical Chemistry
        available_params = params_dict['physical']
        selected_param = st.sidebar.selectbox(
            "Select Physical Parameter:",
            available_params,
            index=0 if available_params else None,
            help="Choose a physical chemistry parameter"
        )
    
    # ====================================================================
    # TIME RANGE SLIDER - Improved date selection
    # ====================================================================
    st.sidebar.subheader("Time Range")
    
    # Get date range from data
    min_year = df['Date'].dt.year.min()
    max_year = df['Date'].dt.year.max()
    
    # Year range slider
    year_range = st.sidebar.slider(
        "Select Year Range:",
        min_value=int(min_year),
        max_value=int(max_year),
        value=(int(min_year), int(max_year)),
        step=1,
        help="Drag to select the time period for analysis"
    )
    
    # Convert year range to dates for filtering
    start_date = pd.Timestamp(year=year_range[0], month=1, day=1)
    end_date = pd.Timestamp(year=year_range[1], month=12, day=31)
    
    # ====================================================================
    # REGION FILTER - UK geographical regions with checkboxes
    # ====================================================================
    st.sidebar.subheader("Regions")
    
    # Create checkbox layout similar to user's image
    all_regions_selected = st.sidebar.checkbox("All Regions", value=True)
    
    selected_regions = []
    
    if not all_regions_selected:
        # Individual region checkboxes
        if st.sidebar.checkbox("NE", value=False):
            selected_regions.append("NE")
        if st.sidebar.checkbox("NW", value=False):
            selected_regions.append("NW")
        if st.sidebar.checkbox("SE", value=False):
            selected_regions.append("SE")
        if st.sidebar.checkbox("SW", value=False):
            selected_regions.append("SW")
    else:
        # If "All Regions" is selected, include all regions
        selected_regions = ["All"]
    
    # If no regions selected and "All" not checked, default to all
    if not selected_regions and not all_regions_selected:
        st.sidebar.warning("Please select at least one region")
        selected_regions = ["All"]
    
    # ====================================================================
    # WATER SOURCE FILTER
    # ====================================================================
    st.sidebar.subheader("Water Source Filter")
    all_sources = df['Water_Source'].unique()
    selected_sources = st.sidebar.multiselect(
        "Select Water Sources:",
        all_sources,
        default=all_sources,
        help="Filter by specific water sources"
    )
    
    # ====================================================================
    # ANALYSIS OPTIONS
    # ====================================================================
    st.sidebar.subheader("Analysis Options")
    n_top_sites = st.sidebar.slider(
        "Number of Top Sites:",
        min_value=5,
        max_value=25,
        value=5,  # Changed default value from 10 to 5
        step=5,
        help="Number of top concentration sites to display"
    )
    
    time_aggregation = st.sidebar.radio(
        "Time Series Aggregation:",
        ["Monthly", "Yearly"],
        help="Choose time aggregation for overall trend analysis"
    )
    
    # ------------------------------------------------------------------------
    # DATA FILTERING - Apply user selections
    # ------------------------------------------------------------------------
    
    # Apply date filter
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df_filtered = df[mask].copy()
    
    # Filter by water sources
    df_filtered = df_filtered[df_filtered['Water_Source'].isin(selected_sources)]
    
    # ------------------------------------------------------------------------
    # MAIN CONTENT - Display analysis results
    # ------------------------------------------------------------------------
    
    if selected_param:
        # Get data for selected parameter
        param_data = df_filtered[df_filtered['Parameter'] == selected_param]
        
        if not param_data.empty:
            # Get unit for display
            unit = get_display_unit(selected_param)
            
            # ================================================================
            # SUMMARY METRICS - Key statistics
            # ================================================================
            st.header("Summary Statistics")
            
            # Filter param_data by regions if specific regions selected
            if selected_regions and 'All' not in selected_regions:
                param_data_regional = param_data[param_data['Region'].isin(selected_regions)]
            else:
                param_data_regional = param_data
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "Total Measurements",
                    format_number(len(param_data_regional)),
                    help="Total number of measurements for selected parameter and regions"
                )
            
            with col2:
                st.metric(
                    "Sampling Points",
                    format_number(param_data_regional['Sampling Point'].nunique()),
                    help="Number of unique sampling locations"
                )
            
            with col3:
                mean_val = param_data_regional['result'].mean()
                st.metric(
                    "Mean Concentration",
                    f"{mean_val:.3f} {unit}",
                    help="Average concentration across all measurements"
                )
            
            with col4:
                median_val = param_data_regional['result'].median()
                st.metric(
                    "Median",
                    f"{median_val:.3f} {unit}",
                    help="Median concentration (50th percentile)"
                )
            
            with col5:
                max_val = param_data_regional['result'].max()
                st.metric(
                    "Maximum",
                    f"{max_val:.2f} {unit}",
                    help="Highest recorded concentration"
                )
            
            st.markdown("---")
            
            # ================================================================
            # TOP SAMPLING POINTS MAP - Full width map at top
            # ================================================================
            st.header(f"Top {n_top_sites} Sampling Locations")
            
            # Get top points from selected regions
            top_points = get_top_sampling_points(
                df_filtered, 
                selected_param, 
                n_top_sites, 
                selected_regions if 'All' not in selected_regions else None
            )
            
            if not top_points.empty:
                # Display map full width
                map_fig = create_map_visualization(top_points, selected_param, unit, selected_regions)
                st.plotly_chart(map_fig, use_container_width=True)
                
                # Display table below the map
                st.subheader("Top Sites Details")
                display_df = top_points[['Rank', 'Sampling Point', 'Water_Source', 'Region',
                                        'Mean_Concentration', 'Max_Concentration', 'Sample_Count']].copy()
                display_df['Mean_Concentration'] = display_df['Mean_Concentration'].round(3)
                display_df['Max_Concentration'] = display_df['Max_Concentration'].round(3)
                display_df['Sample_Count'] = display_df['Sample_Count'].astype(int)
                display_df.columns = ['Rank', 'Location', 'Source Type', 'Region', f'Mean ({unit})', 
                                     f'Max ({unit})', 'Samples']
                
                st.dataframe(
                    display_df,
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.info("No data available for the selected parameters and filters.")
            
            st.markdown("---")
            
            # ================================================================
            # TOP SITES TEMPORAL TRENDS - New section for top sites over time
            # ================================================================
            if not top_points.empty:
                st.header(f"Temporal Trends - Top {n_top_sites} Sites")
                
                # Get temporal data for top sites
                temporal_data = get_top_sites_temporal_data(df, selected_param, top_points)
                
                if not temporal_data.empty:
                    # Create and display the temporal chart
                    temporal_fig = create_top_sites_temporal_chart(temporal_data, selected_param, unit)
                    st.plotly_chart(temporal_fig, use_container_width=True)
                else:
                    st.info("No temporal data available for the selected top sites.")
                
                st.markdown("---")
                
                # ============================================================
                # WATER SOURCE ANALYSIS FOR TOP SITES
                # ============================================================
                st.header(f"Water Source Distribution - Top {n_top_sites} Sites")
                
                # Get water source distribution for top sites only
                top_sites_list = top_points['Sampling Point'].tolist()
                top_sites_source_data = get_source_distribution(df_filtered, selected_param, top_sites_list)
                
                if not top_sites_source_data.empty:
                    source_fig_top = create_source_distribution_chart(
                        top_sites_source_data, 
                        unit, 
                        f" (Top {n_top_sites} Sites)"
                    )
                    st.plotly_chart(source_fig_top, use_container_width=True)
                
                st.markdown("---")
            
            # ================================================================
            # OVERALL TIME SERIES ANALYSIS - General trends
            # ================================================================
            st.header("Overall Temporal Trends")
            
            ts_data = get_time_series_data(
                df_filtered if 'All' in selected_regions else df_filtered[df_filtered['Region'].isin(selected_regions)], 
                selected_param, 
                time_aggregation.lower()
            )
            
            if not ts_data.empty:
                ts_fig = create_time_series_chart(
                    ts_data,
                    f"Overall Concentration Trends ({time_aggregation})",
                    unit
                )
                st.plotly_chart(ts_fig, use_container_width=True)
                
                # Additional time series insights - Fixed year formatting
                with st.expander("Temporal Analysis Details"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Yearly statistics with proper formatting
                        yearly_stats = param_data_regional.groupby(param_data_regional['Date'].dt.year)['result'].agg(['mean', 'count']).reset_index()
                        yearly_stats.columns = ['Date', 'Mean Concentration', 'Sample Count']
                        yearly_stats['Mean Concentration'] = yearly_stats['Mean Concentration'].round(3)
                        yearly_stats['Date'] = yearly_stats['Date'].astype(str)  # Convert to string to avoid comma formatting
                        st.subheader("Yearly Statistics")
                        st.dataframe(yearly_stats.tail(10), hide_index=True, use_container_width=True)
                    
                    with col2:
                        # Seasonal patterns
                        if 'Season' in param_data_regional.columns:
                            seasonal_stats = param_data_regional.groupby('Season')['result'].agg(['mean', 'median', 'count']).reset_index()
                            seasonal_stats.columns = ['Season', 'Mean', 'Median', 'Count']
                            seasonal_stats[['Mean', 'Median']] = seasonal_stats[['Mean', 'Median']].round(3)
                            st.subheader("Seasonal Patterns")
                            st.dataframe(seasonal_stats, hide_index=True, use_container_width=True)
                        else:
                            st.info("Seasonal data not available")
            else:
                st.info("Insufficient data for time series analysis.")
            
            st.markdown("---")
            
            # ================================================================
            # OVERALL WATER SOURCE DISTRIBUTION
            # ================================================================
            st.header("Overall Water Source Analysis")
            
            # Get overall source distribution
            overall_source_data = get_source_distribution(
                df_filtered if 'All' in selected_regions else df_filtered[df_filtered['Region'].isin(selected_regions)], 
                selected_param
            )
            
            if not overall_source_data.empty:
                source_fig = create_source_distribution_chart(overall_source_data, unit, " (All Data)")
                st.plotly_chart(source_fig, use_container_width=True)
                
                # Detailed source statistics
                with st.expander("Detailed Source Statistics"):
                    display_source = overall_source_data[['Water_Source', 'Source_Category', 'Mean', 
                                                         'Median', 'Max', 'Sample_Count']].head(15)
                    display_source[['Mean', 'Median', 'Max']] = display_source[['Mean', 'Median', 'Max']].round(3)
                    display_source['Sample_Count'] = display_source['Sample_Count'].astype(int)
                    display_source.columns = ['Water Source', 'Category', f'Mean ({unit})', 
                                            f'Median ({unit})', f'Max ({unit})', 'Samples']
                    st.dataframe(display_source, hide_index=True, use_container_width=True)
            
            st.markdown("---")
            
    # ------------------------------------------------------------------------
    # FOOTER - Copyright and information
    # ------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #8c919a; font-size: 14px; padding: 20px;'>
    © 2025 HydroStar Europe Ltd.<br>
    Data Source: UK Environment Agency | 26 Years of Monthly Water Quality Monitoring (2000-2025)<br>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()

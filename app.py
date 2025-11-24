import streamlit as st
import pandas as pd
import numpy as np
import os

# Try to import Plotly, with fallback error handling
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.error("⚠️ Plotly is not installed. Please add 'plotly>=5.10.0' to your requirements.txt file")

# Set the title of the web app
st.set_page_config(layout="wide", page_title="Optical Data Viewer")
st.title("Optical Data Colormap Viewer 🔬")

# --- Data Loading ---
@st.cache_data
def load_data(polarization):
    # Define filenames based on polarization
    if polarization == "TE":
        filename = "pvk_0_TE_desp_0_interval_4_sim.txt"
    else:  # TM
        filename = "pvk_0_TM_desp_0_interval_4_sim.txt"
    
    # Check if file exists to prevent crashing
    if not os.path.exists(filename):
        st.error(f"File not found: `{filename}`. Please ensure the .txt file is in the same folder.")
        return None

    try:
        # We manually list the 6 columns found in your file
        # The 3rd one is 'lambda0' which you asked to ignore
        column_names = [
            'h_fib', 
            'lda0', 
            'lambda0_duplicate', # This is the duplicate column we will drop
            'Reflectance_port_1', 
            'Transmittance_port_2', 
            'Absorvance'
        ]

        # --- KEY FIX FOR .TXT FILES ---
        # pd.read_csv works for .txt files too!
        # sep=r'\s+' tells it to separate columns by SPACES, not commas.
        # comment='%' tells it to ignore the text headers at the top of your file.
        df = pd.read_csv(
            filename, 
            sep=r'\s+',          # Handle space-separated values
            comment='%',         # Skip lines starting with %
            header=None,         # Do not try to read the first line as header
            names=column_names   # Use our manual column names
        )

        # Drop the column you asked to ignore
        df = df.drop(columns=['lambda0_duplicate'])

        return df

    except Exception as e:
        st.error(f"Error loading {polarization} data from {filename}: {e}")
        return None

if not PLOTLY_AVAILABLE:
    st.warning("Plotly is required. Please add `plotly>=5.10.0` to requirements.txt")
    st.stop()

# --- Polarization Selection ---
st.sidebar.header("Polarization Selection")
polarization = st.sidebar.radio(
    "Select Polarization:",
    ('TE', 'TM')
)

# Load data
df = load_data(polarization)

if df is not None:
    # --- User Controls (Sidebar) ---
    st.sidebar.header("Plot Controls")

    # Map readable names to your internal column names
    display_options = {
        'Reflectance (Port 1)': 'Reflectance_port_1',
        'Transmittance (Port 2)': 'Transmittance_port_2', 
        'Absorbance': 'Absorvance'
    }
    
    selected_display_name = st.sidebar.selectbox(
        "Select data to plot:",
        list(display_options.keys())
    )
    
    selected_column = display_options[selected_display_name]

    scale_choice = st.sidebar.radio(
        "Select color scale:",
        ('Normal Scale', 'Log Scale')
    )

    # --- Data Processing ---
    try:
        # Rounding ensures floating point numbers match up perfectly for the heatmap grid
        df['h_fib'] = df['h_fib'].round(5)
        df['lda0'] = df['lda0'].round(5)

        # Create the matrix (pivot table)
        df_pivot = df.pivot(
            index='h_fib', 
            columns='lda0', 
            values=selected_column
        )
        
        # Sort to ensure axes are in order
        df_pivot.sort_index(axis=0, inplace=True) 
        df_pivot.sort_index(axis=1, inplace=True)
        
        z_data = df_pivot.values
        color_label = selected_display_name
        zmin_val = None
        zmax_val = None

        # --- Log Scale Logic ---
        if scale_choice == 'Log Scale':
            z_data_safe = np.where(z_data <= 0, 1e-12, z_data) # Prevent log(0) error
            z_data = np.log10(z_data_safe)
            color_label = f"Log10({selected_display_name})"
            zmin_val = -5 # Adjust these bounds if image is too dark/bright
            zmax_val = 0

        # --- Sliders ---
        st.sidebar.header("Cross-Section Controls")
        
        wavelength_min = float(df_pivot.columns.min())
        wavelength_max = float(df_pivot.columns.max())
        height_min = float(df_pivot.index.min())
        height_max = float(df_pivot.index.max())
        
        selected_wavelength = st.sidebar.slider(
            "Select Wavelength (nm)",
            min_value=wavelength_min,
            max_value=wavelength_max,
            value=(wavelength_min + wavelength_max) / 2,
            step=(wavelength_max - wavelength_min) / 100
        )
        
        selected_height = st.sidebar.slider(
            "Select Fiber Height (nm)",
            min_value=height_min,
            max_value=height_max,
            value=(height_min + height_max) / 2,
            step=(height_max - height_min) / 100
        )

        # Find nearest real data points
        wavelength_idx = np.abs(df_pivot.columns - selected_wavelength).argmin()
        height_idx = np.abs(df_pivot.index - selected_height).argmin()
        
        actual_wavelength = df_pivot.columns[wavelength_idx]
        actual_height = df_pivot.index[height_idx]

        # --- Plotting ---
        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.8, 0.2],
            row_heights=[0.2, 0.8],
            vertical_spacing=0.05,
            horizontal_spacing=0.05,
            shared_xaxes=True,
            shared_yaxes=True,
            subplot_titles=(
                f'Horizontal Cut (h={actual_height:.1f}nm)', 
                '', 
                f'{polarization} Map: {color_label}', 
                f'Vertical Cut (λ={actual_wavelength:.1f}nm)'
            )
        )

        # 1. Heatmap
        heatmap = go.Heatmap(
            z=z_data,
            x=df_pivot.columns,
            y=df_pivot.index,
            colorscale='Jet',
            colorbar=dict(title=color_label, len=0.75, y=0.15, yanchor='bottom'),
            zmin=zmin_val,
            zmax=zmax_val,
            hovertemplate="Wavelength: %{x} nm<br>Height: %{y} nm<br>Value: %{z:.4e}<extra></extra>"
        )
        fig.add_trace(heatmap, row=2, col=1)

        # Crosshair lines on Heatmap
        fig.add_hline(y=actual_height, line=dict(color='white', width=1, dash='dash'), row=2, col=1)
        fig.add_vline(x=actual_wavelength, line=dict(color='white', width=1, dash='dash'), row=2, col=1)

        # 2. Top Plot (Horizontal Cut)
        x_cross_section = z_data[height_idx, :]
        fig.add_trace(go.Scatter(
            x=df_pivot.columns, y=x_cross_section,
            mode='lines', line=dict(color='red', width=2), name="H-Cut"
        ), row=1, col=1)
        
        # Dot on Top Plot
        fig.add_trace(go.Scatter(
            x=[actual_wavelength], y=[x_cross_section[wavelength_idx]],
            mode='markers', marker=dict(color='blue', size=8), showlegend=False
        ), row=1, col=1)

        # 3. Right Plot (Vertical Cut)
        y_cross_section = z_data[:, wavelength_idx]
        fig.add_trace(go.Scatter(
            x=y_cross_section, y=df_pivot.index,
            mode='lines', line=dict(color='blue', width=2), name="V-Cut"
        ), row=2, col=2)

        # Dot on Right Plot
        fig.add_trace(go.Scatter(
            x=[y_cross_section[height_idx]], y=[actual_height],
            mode='markers', marker=dict(color='red', size=8), showlegend=False
        ), row=2, col=2)

        # Final Layout
        fig.update_layout(
            height=700,
            showlegend=False,
            template="plotly_white",
            xaxis2=dict(title='Wavelength (nm)'),
            yaxis2=dict(title='Fiber Height (nm)'),
            xaxis1=dict(showticklabels=True), # Top axis
            yaxis3=dict(showticklabels=True)  # Right axis
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while plotting: {e}")
        st.write("First 5 rows of loaded data for debugging:")
        st.dataframe(df.head())

else:
    st.info("Please ensure the file 'pvk_0_TE_desp_0_interval_4_sim.txt' is in the same folder.")

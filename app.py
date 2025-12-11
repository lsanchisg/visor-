
# --- DEBUG: Show me all files in the current folder ---
import os



import streamlit as st
import pandas as pd
import numpy as np

st.write("📂 **Files found in this folder:**")
st.write(os.listdir('.')) # This prints the list of files to the screen
st.write("---")


# Try to import Plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.error("⚠️ Plotly is not installed. Please add 'plotly>=5.10.0' to your requirements.txt file")

# Set page config
st.set_page_config(layout="wide", page_title="Cluster Optical Viewer")
st.title("Optical Data Colormap Viewer (Cluster) 🔬")

# --- Configuration: Available Data Parameters ---
# Update these lists based on your actual data files
thickness_options = [0, 2, 15] 
separation_options = [0]      

# --- Data Loading ---
@st.cache_data
def load_data(thickness, polarization, separation, is_symmetric):
    # --- NEW FILENAME LOGIC ---
    # Base pattern: cluster_pvk_{thickness}_{polarization}_desp_{separation}
    base_name = f"cluster_pvk_{thickness}_{polarization}_desp_{separation}"
    
    # Add '_sim' suffix if Symmetric mode is selected
    if is_symmetric:
        filename = f"{base_name}_sim.txt"
    else:
        filename = f"{base_name}.txt"
    
    # Check if file exists
    if not os.path.exists(filename):
        return None, filename

    try:
        # Define columns (ignoring lambda0_duplicate later)
        column_names = [
            'h_fib', 
            'lda0', 
            'lambda0_duplicate', 
            'Reflectance_port_1', 
            'Transmittance_port_2', 
            'Absorvance'
        ]

        # Read the text file (Space-separated)
        df = pd.read_csv(
            filename, 
            sep=r'\s+',          # Handle space separation
            comment='%',         # Skip comments
            header=None,         # No header row
            names=column_names   # Manual column names
        )

        # Drop the duplicate column
        df = df.drop(columns=['lambda0_duplicate'])

        return df, filename

    except Exception as e:
        st.error(f"Error reading file {filename}: {e}")
        return None, filename

if not PLOTLY_AVAILABLE:
    st.stop()

# --- Sidebar: Data Selection ---
st.sidebar.header("Data Configuration")

# 1. Thickness
selected_thickness = st.sidebar.selectbox(
    "Thickness (nm):",
    thickness_options
)

# 2. Polarization
selected_polarization = st.sidebar.radio(
    "Polarization:",
    ('TE', 'TM')
)

# 3. Separation
selected_separation = st.sidebar.selectbox(
    "Separation (nm):",
    separation_options
)

# 4. NEW: Symmetry Selection
symmetry_mode = st.sidebar.radio(
    "Simulation Type:",
    ('Standard', 'Symmetric (_sim)')
)
# Convert text choice to boolean
is_symmetric = (symmetry_mode == 'Symmetric (_sim)')

# Load data based on ALL selected parameters
df, current_filename = load_data(selected_thickness, selected_polarization, selected_separation, is_symmetric)

if df is not None:
    # --- User Controls (Plotting) ---
    st.sidebar.markdown("---")
    st.sidebar.header("Plot Controls")

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
        # Rounding for clean pivoting
        df['h_fib'] = df['h_fib'].round(5)
        df['lda0'] = df['lda0'].round(5)

        # Create Matrix (Pivot)
        df_pivot = df.pivot(
            index='h_fib', 
            columns='lda0', 
            values=selected_column
        )
        
        # Sort axes
        df_pivot.sort_index(axis=0, inplace=True) 
        df_pivot.sort_index(axis=1, inplace=True)
        
        z_data = df_pivot.values
        color_label = selected_display_name
        zmin_val = None
        zmax_val = None

        # Log Scale Logic
        if scale_choice == 'Log Scale':
            z_data_safe = np.where(z_data <= 0, 1e-12, z_data)
            z_data = np.log10(z_data_safe)
            color_label = f"Log10({selected_display_name})"
            zmin_val = -5 
            zmax_val = 0

        # --- Sliders ---
        st.sidebar.header("Cross-Section Controls")
        
        # Get ranges
        w_min, w_max = float(df_pivot.columns.min()), float(df_pivot.columns.max())
        h_min, h_max = float(df_pivot.index.min()), float(df_pivot.index.max())
        
        # Sliders
        sel_wave = st.sidebar.slider("Wavelength (nm)", w_min, w_max, (w_min + w_max)/2)
        sel_height = st.sidebar.slider("Fiber Height (nm)", h_min, h_max, (h_min + h_max)/2)

        # Find nearest indices
        w_idx = np.abs(df_pivot.columns - sel_wave).argmin()
        h_idx = np.abs(df_pivot.index - sel_height).argmin()
        
        act_wave = df_pivot.columns[w_idx]
        act_height = df_pivot.index[h_idx]

        # --- Plotting ---
        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.8, 0.2],
            row_heights=[0.2, 0.8],
            vertical_spacing=0.05, horizontal_spacing=0.05,
            shared_xaxes=True, shared_yaxes=True,
            subplot_titles=(
                f'Horizontal Cut (h={act_height:.3f})', 
                f'File: {current_filename}', 
                f'{selected_polarization} Map ({symmetry_mode})', 
                f'Vertical Cut (λ={act_wave:.3f})'
            )
        )

        # 1. Heatmap
        heatmap = go.Heatmap(
            z=z_data, x=df_pivot.columns, y=df_pivot.index,
            colorscale='Jet',
            colorbar=dict(title=color_label, len=0.75, y=0.15, yanchor='bottom'),
            zmin=zmin_val, zmax=zmax_val,
            hovertemplate="λ: %{x}<br>h: %{y}<br>Val: %{z:.4e}<extra></extra>"
        )
        fig.add_trace(heatmap, row=2, col=1)

        # Crosshairs
        fig.add_hline(y=act_height, line=dict(color='white', width=1, dash='dash'), row=2, col=1)
        fig.add_vline(x=act_wave, line=dict(color='white', width=1, dash='dash'), row=2, col=1)

        # 2. Horizontal Cut (Top)
        x_cross = z_data[h_idx, :]
        fig.add_trace(go.Scatter(x=df_pivot.columns, y=x_cross, mode='lines', line=dict(color='red'), name="H-Cut"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[act_wave], y=[x_cross[w_idx]], mode='markers', marker=dict(color='blue', size=8), showlegend=False), row=1, col=1)

        # 3. Vertical Cut (Right)
        y_cross = z_data[:, w_idx]
        fig.add_trace(go.Scatter(x=y_cross, y=df_pivot.index, mode='lines', line=dict(color='blue'), name="V-Cut"), row=2, col=2)
        fig.add_trace(go.Scatter(x=[y_cross[h_idx]], y=[act_height], mode='markers', marker=dict(color='red', size=8), showlegend=False), row=2, col=2)

        # Layout
        fig.update_layout(
            height=700, showlegend=False, template="plotly_white",
            xaxis2=dict(title='Wavelength (nm)'),
            yaxis2=dict(title='Fiber Height (nm)'),
            xaxis1=dict(showticklabels=True), 
            yaxis3=dict(showticklabels=True)  
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error plotting data: {e}")
        st.write("Data Head:", df.head())

else:
    # --- File Not Found Message ---
    st.warning(f"File not found: `{current_filename}`")
    st.info(f"""
    **Looking for file:**
    `{current_filename}`
    
    **Current Configuration:**
    - Prefix: `cluster_pvk_`
    - Thickness: `{selected_thickness}`
    - Polarization: `{selected_polarization}`
    - Separation: `{selected_separation}`
    - Mode: `{symmetry_mode}`
    
    *Please ensure the file matches this naming pattern exactly and is in the same folder.*
    """)



import streamlit as st
import pandas as pd
import numpy as np
import os
import re

# --- 1. Page Configuration ---
st.set_page_config(layout="wide", page_title="Cluster Optical Viewer")

# --- 2. Imports and Setup for Plotly ---
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.error("⚠️ Plotly is not installed. Please add 'plotly>=5.10.0' to your requirements.txt file")

st.title("Optical Data Colormap Viewer (Cluster) 🔬")

# --- 3. Configuration Lists ---
thickness_options = [0, 2, 15] 
separation_options = [0]       

# --- 4. Helper Functions ---

@st.cache_data
def scan_profile_images(directory="profile_images"):
    """
    Scans the directory for PNG files matching the naming convention:
    {Type}_field_pvk_{thick}_{pol}_desp_{sep}_{h}_{lam}.png
    Example: E_field_pvk_0_TE_desp_0_1100_52375.png
    """
    if not os.path.exists(directory):
        return pd.DataFrame()

    data = []
    # Regex to parse the filename structure
    # Groups: 1=Type, 2=Thickness, 3=Pol, 4=Separation, 5=Height, 6=LambdaCode
    pattern = re.compile(r"([EM])_field_pvk_(\d+)_([A-Z]+)_desp_(\d+)_(\d+)_(\d+)\.png")

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            f_type, thick, pol, sep, h_raw, lam_raw = match.groups()
            
            # Convert Lambda (e.g., 52375 -> 523.75)
            # Assuming format is XX.XXX (div by 100 based on screenshot example 52375)
            # Adjust divisor if your files use a different logic (e.g. 1000)
            lam_val = float(lam_raw) / 100.0 
            
            entry = {
                "filename": filename,
                "type": f_type, # E or M
                "thickness": int(thick),
                "polarization": pol,
                "separation": int(sep),
                "h_fib": float(h_raw),
                "lda0": lam_val,
                "path": os.path.join(directory, filename)
            }
            data.append(entry)

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    return df

@st.cache_data
def load_data(thickness, polarization, separation, is_symmetric):
    # Base pattern
    base_name = f"cluster_pvk_{thickness}_{polarization}_desp_{separation}"
    
    if is_symmetric:
        filename = f"{base_name}_sim.txt"
    else:
        filename = f"{base_name}.txt"
    
    if not os.path.exists(filename):
        return None, filename

    try:
        column_names = [
            'h_fib', 'lda0', 'lambda0_duplicate', 
            'Reflectance_port_1', 'Transmittance_port_2', 'Absorvance'
        ]
        df = pd.read_csv(
            filename, sep=r'\s+', comment='%', header=None, names=column_names
        )
        df = df.drop(columns=['lambda0_duplicate'])
        return df, filename
    except Exception as e:
        return None, filename

# --- 5. Dialog for Fullscreen Image ---
@st.dialog("Field Map Detail", width="large")
def show_full_image(image_path, title):
    st.subheader(title)
    st.image(image_path, use_container_width=True)

# --- 6. Main App Logic ---

# Check dependencies
if not PLOTLY_AVAILABLE:
    st.stop()

# Layout: Sidebar
st.sidebar.header("Data Configuration")
sel_thick = st.sidebar.selectbox("Thickness (nm):", thickness_options)
sel_pol = st.sidebar.radio("Polarization:", ('TE', 'TM'))
sel_sep = st.sidebar.selectbox("Separation (nm):", separation_options)
sym_mode = st.sidebar.radio("Simulation Type:", ('Standard', 'Symmetric (_sim)'))
is_sym = (sym_mode == 'Symmetric (_sim)')

# Load Main Data
df, current_filename = load_data(sel_thick, sel_pol, sel_sep, is_sym)

# Load Image Database
df_imgs = scan_profile_images("profile_images")

# Filter Image Database for Current Settings
available_points = pd.DataFrame()
if not df_imgs.empty:
    # Filter by Thickness, Polarization, Separation
    mask = (
        (df_imgs['thickness'] == sel_thick) & 
        (df_imgs['polarization'] == sel_pol) & 
        (df_imgs['separation'] == sel_sep)
    )
    filtered_imgs = df_imgs[mask]
    
    # We need unique (h, lda) points that have at least one image
    if not filtered_imgs.empty:
        available_points = filtered_imgs[['h_fib', 'lda0']].drop_duplicates()

# --- Layout: Main Content ---
col_main, col_side = st.columns([0.75, 0.25])

with col_main:
    if df is not None:
        # User Controls
        st.write(f"### 📊 Analysis: `{current_filename}`")
        
        display_opts = {
            'Reflectance (Port 1)': 'Reflectance_port_1',
            'Transmittance (Port 2)': 'Transmittance_port_2', 
            'Absorbance': 'Absorvance'
        }
        
        # Plot Controls in a horizontal row
        c1, c2, c3 = st.columns(3)
        with c1:
            sel_display = st.selectbox("Metric:", list(display_opts.keys()))
        with c2:
            scale_choice = st.radio("Scale:", ('Normal', 'Log'), horizontal=True)
        with c3:
            st.info("💡 **Click on white circles** in the plot to see field maps.")

        sel_col = display_opts[sel_display]

        # Data Pivoting
        try:
            df['h_fib'] = df['h_fib'].round(5)
            df['lda0'] = df['lda0'].round(5)
            df_pivot = df.pivot(index='h_fib', columns='lda0', values=sel_col)
            df_pivot.sort_index(axis=0, inplace=True)
            df_pivot.sort_index(axis=1, inplace=True)
            
            z_data = df_pivot.values
            
            # Log Scale Logic
            if scale_choice == 'Log':
                z_data_safe = np.where(z_data <= 0, 1e-12, z_data)
                z_data = np.log10(z_data_safe)
                color_label = f"Log10({sel_display})"
                zmin, zmax = -5, 0
            else:
                color_label = sel_display
                zmin, zmax = None, None

            # --- PLOTTING ---
            fig = make_subplots(rows=1, cols=1) # Simplified to 1 main plot for clarity

            # 1. Heatmap Trace
            heatmap = go.Heatmap(
                z=z_data, x=df_pivot.columns, y=df_pivot.index,
                colorscale='Jet',
                colorbar=dict(title=color_label),
                zmin=zmin, zmax=zmax,
                hovertemplate="λ: %{x}<br>h: %{y}<br>Val: %{z:.4e}<extra></extra>"
            )
            fig.add_trace(heatmap)

            # 2. Marker Trace (Available Images)
            if not available_points.empty:
                fig.add_trace(go.Scatter(
                    x=available_points['lda0'],
                    y=available_points['h_fib'],
                    mode='markers',
                    marker=dict(
                        color='white',
                        size=8,
                        line=dict(width=2, color='black'),
                        symbol='circle'
                    ),
                    name='Field Map Available',
                    hovertemplate="<b>Field Map Available</b><br>λ: %{x}<br>h: %{y}<extra></extra>"
                ))

            fig.update_layout(
                height=650, 
                title_text=f"{sel_pol} Polarization - {sel_display}",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Fiber Height (nm)",
                # Enable click selection
                clickmode='event+select',
                dragmode='zoom' # Default tool
            )

            # RENDER PLOT with Selection Event
            # 'selection_mode="points"' ensures we just get the clicked point
            event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")

        except Exception as e:
            st.error(f"Plot Error: {e}")

    else:
        st.warning(f"Data file not found: {current_filename}")


# --- Right Sidebar: Image Viewer ---
with col_side:
    st.header("🖼️ Field Maps")
    
    selected_h = None
    selected_lam = None

    # 1. Handle Click Event
    if event and event["selection"]["points"]:
        point = event["selection"]["points"][0]
        # Coordinates from the click
        click_x = point["x"]
        click_y = point["y"]
        
        # Find closest match in available_points to handle slight float precision differences
        if not available_points.empty:
            # Simple distance check to snap to nearest valid point
            distances = ((available_points['lda0'] - click_x)**2 + (available_points['h_fib'] - click_y)**2)
            nearest_idx = distances.idxmin()
            
            # If closest point is very close (avoid random clicks on background)
            if distances[nearest_idx] < 1.0: # Tolerance
                selected_h = available_points.loc[nearest_idx, 'h_fib']
                selected_lam = available_points.loc[nearest_idx, 'lda0']

    # 2. Display Images if Selection is Valid
    if selected_h is not None:
        st.success(f"Selected:\nλ={selected_lam} nm\nh={selected_h} nm")
        
        # Find the specific file paths in df_imgs
        # We look for entries matching our parameters + the clicked coordinates
        subset = df_imgs[
            (df_imgs['thickness'] == sel_thick) &
            (df_imgs['polarization'] == sel_pol) &
            (df_imgs['separation'] == sel_sep) &
            (df_imgs['h_fib'] == selected_h) &
            # Use small epsilon for float comparison on lambda
            (np.abs(df_imgs['lda0'] - selected_lam) < 0.001)
        ]
        
        # Get E-field and M-field paths
        e_row = subset[subset['type'] == 'E']
        m_row = subset[subset['type'] == 'M']
        
        # --- E-Field ---
        st.markdown("---")
        st.write("**Electric Field (|E|)**")
        if not e_row.empty:
            e_path = e_row.iloc[0]['path']
            st.image(e_path, use_container_width=True)
            if st.button("🔍 Zoom E-Field"):
                show_full_image(e_path, f"Electric Field (h={selected_h}, λ={selected_lam})")
        else:
            st.info("No E-field image.")

        # --- H-Field ---
        st.markdown("---")
        st.write("**Magnetic Field (|H|)**")
        if not m_row.empty:
            m_path = m_row.iloc[0]['path']
            st.image(m_path, use_container_width=True)
            if st.button("🔍 Zoom H-Field"):
                show_full_image(m_path, f"Magnetic Field (h={selected_h}, λ={selected_lam})")
        else:
            st.info("No M-field image.")

    else:
        st.caption("Click a white circle on the graph to view field profiles.")
        
        # Debugging: Show raw list if needed
        # st.write("Available points:", available_points)

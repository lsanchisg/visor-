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
thickness_options = [0, 2] 
separation_options = [0]       

# --- 4. Helper Functions ---

@st.cache_data
def scan_profile_images(directory="profile_images"):
    if not os.path.exists(directory):
        return pd.DataFrame()

    data = []
    pattern = re.compile(r"([EM])_field_pvk_(\d+)_([A-Z]+)_desp_(\d+)_(\d+)_(\d+)\.png")

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            f_type, thick, pol, sep, h_raw, lam_raw = match.groups()
            lam_val = float(lam_raw) / 100.0 
            
            entry = {
                "filename": filename,
                "type": f_type, 
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
    base_name = f"cluster_pvk_{thickness}_{polarization}_desp_{separation}"
    filename = f"{base_name}_sim.txt" if is_symmetric else f"{base_name}.txt"
    
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

@st.dialog("Field Map Detail", width="large")
def show_full_image(image_path, title):
    st.subheader(title)
    st.image(image_path, use_container_width=True)

# --- 5. Main App Logic ---

if not PLOTLY_AVAILABLE:
    st.stop()

# Layout: Sidebar
st.sidebar.header("Data Configuration")
sel_thick = st.sidebar.selectbox("Thickness (nm):", thickness_options)
sel_pol = st.sidebar.radio("Polarization:", ('TE', 'TM'))
sel_sep = st.sidebar.selectbox("Separation (nm):", separation_options)
sym_mode = st.sidebar.radio("Simulation Type:", ('Standard', 'Symmetric (_sim)'))
is_sym = (sym_mode == 'Symmetric (_sim)')

# Load Data
df, current_filename = load_data(sel_thick, sel_pol, sel_sep, is_sym)
df_imgs = scan_profile_images("profile_images")

available_points = pd.DataFrame()
if not df_imgs.empty:
    mask = (
        (df_imgs['thickness'] == sel_thick) & 
        (df_imgs['polarization'] == sel_pol) & 
        (df_imgs['separation'] == sel_sep)
    )
    filtered_imgs = df_imgs[mask]
    if not filtered_imgs.empty:
        available_points = filtered_imgs[['h_fib', 'lda0']].drop_duplicates()

# --- MAIN LAYOUT ---
col_main, col_side = st.columns([0.75, 0.25])

with col_main:
    if df is not None:
        st.write(f"### 📊 Analysis: `{current_filename}`")
        
        display_opts = {
            'Reflectance (Port 1)': 'Reflectance_port_1',
            'Transmittance (Port 2)': 'Transmittance_port_2', 
            'Absorbance': 'Absorvance'
        }
        
        c1, c2, c3 = st.columns(3)
        with c1:
            sel_display = st.selectbox("Metric:", list(display_opts.keys()))
        with c2:
            scale_choice = st.radio("Scale:", ('Normal', 'Log'), horizontal=True)
        with c3:
            st.info("💡 **Click white circles** to see Field Maps.")

        sel_col = display_opts[sel_display]

        try:
            # Data Prep
            df['h_fib'] = df['h_fib'].round(5)
            df['lda0'] = df['lda0'].round(5)
            df_pivot = df.pivot(index='h_fib', columns='lda0', values=sel_col)
            df_pivot.sort_index(axis=0, inplace=True)
            df_pivot.sort_index(axis=1, inplace=True)
            z_data = df_pivot.values
            
            # Log Scale
            if scale_choice == 'Log':
                z_data_safe = np.where(z_data <= 0, 1e-12, z_data)
                z_data = np.log10(z_data_safe)
                color_label = f"Log10({sel_display})"
                zmin, zmax = -5, 0
            else:
                color_label = sel_display
                zmin, zmax = None, None

            # --- SESSION STATE & SLIDERS ---
            w_min = float(df_pivot.columns.min())
            w_max = float(df_pivot.columns.max())
            h_min = float(df_pivot.index.min())
            h_max = float(df_pivot.index.max())

            # Initialize state variables if not present
            if "current_wave" not in st.session_state:
                st.session_state.current_wave = (w_min + w_max) / 2.0
            if "current_height" not in st.session_state:
                st.session_state.current_height = (h_min + h_max) / 2.0

            # Define callbacks to update state from sliders
            def update_wave_from_slider():
                st.session_state.current_wave = st.session_state.slider_wave
            
            def update_height_from_slider():
                st.session_state.current_height = st.session_state.slider_height

            st.sidebar.markdown("---")
            st.sidebar.header("Cross-Section Controls")
            
            # SLIDERS with explicit floats to prevent type errors
            st.sidebar.slider(
                "Wavelength (nm)", 
                min_value=w_min, 
                max_value=w_max, 
                value=float(st.session_state.current_wave), 
                step=0.01,
                key="slider_wave", 
                on_change=update_wave_from_slider
            )
            st.sidebar.slider(
                "Fiber Height (nm)", 
                min_value=h_min, 
                max_value=h_max, 
                value=float(st.session_state.current_height), 
                step=1.0,
                key="slider_height", 
                on_change=update_height_from_slider
            )

            # Use the values from state
            act_wave = st.session_state.current_wave
            act_height = st.session_state.current_height

            # Indices
            w_idx = np.abs(df_pivot.columns - act_wave).argmin()
            h_idx = np.abs(df_pivot.index - act_height).argmin()
            real_wave = df_pivot.columns[w_idx]
            real_height = df_pivot.index[h_idx]

            # --- PLOT ---
            fig_final = make_subplots(
                rows=2, cols=2,
                column_widths=[0.8, 0.2],
                row_heights=[0.2, 0.8],
                vertical_spacing=0.05, horizontal_spacing=0.05,
                shared_xaxes=True, shared_yaxes=True,
                subplot_titles=(
                    f'H-Cut (h={real_height:.3f})', '', 
                    f'{sel_pol} Map', f'V-Cut (λ={real_wave:.3f})'
                )
            )

            # Heatmap
            heatmap = go.Heatmap(
                z=z_data, x=df_pivot.columns, y=df_pivot.index,
                colorscale='Jet',
                colorbar=dict(title=color_label, len=0.75, y=0.15, yanchor='bottom'),
                zmin=zmin, zmax=zmax,
                hovertemplate="λ: %{x}<br>h: %{y}<br>Val: %{z:.4e}<extra></extra>"
            )
            fig_final.add_trace(heatmap, row=2, col=1)

            # Markers
            if not available_points.empty:
                fig_final.add_trace(go.Scatter(
                    x=available_points['lda0'],
                    y=available_points['h_fib'],
                    mode='markers',
                    marker=dict(color='white', size=8, line=dict(width=2, color='black')),
                    name='Images',
                    hovertemplate="<b>Image Available</b><br>λ: %{x}<br>h: %{y}<extra></extra>"
                ), row=2, col=1)

            # Cuts
            x_cross = z_data[h_idx, :]
            y_cross = z_data[:, w_idx]
            
            fig_final.add_trace(go.Scatter(x=df_pivot.columns, y=x_cross, line=dict(color='red'), name="H-Cut"), row=1, col=1)
            fig_final.add_trace(go.Scatter(x=[real_wave], y=[x_cross[w_idx]], mode='markers', marker=dict(color='blue', size=8), showlegend=False), row=1, col=1)
            
            fig_final.add_trace(go.Scatter(x=y_cross, y=df_pivot.index, line=dict(color='blue'), name="V-Cut"), row=2, col=2)
            fig_final.add_trace(go.Scatter(x=[y_cross[h_idx]], y=[real_height], mode='markers', marker=dict(color='red', size=8), showlegend=False), row=2, col=2)

            # Crosshairs
            fig_final.add_hline(y=real_height, line=dict(color='white', width=1, dash='dash'), row=2, col=1)
            fig_final.add_vline(x=real_wave, line=dict(color='white', width=1, dash='dash'), row=2, col=1)

            fig_final.update_layout(height=700, clickmode='event+select', template="plotly_white", showlegend=False, dragmode='zoom')

            # --- RENDER & CAPTURE EVENTS ---
            event = st.plotly_chart(fig_final, use_container_width=True, on_select="rerun", selection_mode="points", key="main_plot")

            # --- HANDLE CLICK ---
            if event and event["selection"]["points"]:
                point = event["selection"]["points"][0]
                click_x = point["x"]
                click_y = point["y"]
                
                # Check if values changed significantly
                if (abs(click_x - st.session_state.current_wave) > 0.0001 or 
                    abs(click_y - st.session_state.current_height) > 0.0001):
                    
                    # UPDATE STATE AND RERUN IMMEDIATELY
                    st.session_state.current_wave = float(click_x)
                    st.session_state.current_height = float(click_y)
                    st.rerun()

        except Exception as e:
            st.error(f"Plot Error: {e}")
    else:
        st.warning(f"File not found: {current_filename}")

# --- SIDEBAR IMAGES ---
with col_side:
    st.header("🖼️ Field Maps")
    
    # Use values from Session State
    cur_h = st.session_state.get('current_height', None)
    cur_w = st.session_state.get('current_wave', None)
    
    found_any = False

    if cur_h is not None and not available_points.empty:
        # Distance Check (Squared Euclidean)
        distances = ((available_points['lda0'] - cur_w)**2 + (available_points['h_fib'] - cur_h)**2)
        nearest_idx = distances.idxmin()
        min_dist = distances[nearest_idx]
        
        # Tolerance: 10.0 units squared allows easy clicking
        if min_dist < 10.0:
            img_h = available_points.loc[nearest_idx, 'h_fib']
            img_lam = available_points.loc[nearest_idx, 'lda0']
            
            st.success(f"Selected:\nλ={img_lam} nm\nh={img_h} nm")
            
            subset = df_imgs[
                (df_imgs['thickness'] == sel_thick) &
                (df_imgs['polarization'] == sel_pol) &
                (df_imgs['separation'] == sel_sep) &
                (df_imgs['h_fib'] == img_h) &
                (np.abs(df_imgs['lda0'] - img_lam) < 0.001)
            ]
            
            e_rows = subset[subset['type'] == 'E']
            m_rows = subset[subset['type'] == 'M']
            
            st.markdown("---")
            st.write("**Electric Field (|E|)**")
            if not e_rows.empty:
                e_path = e_rows.iloc[0]['path']
                st.image(e_path, use_container_width=True)
                if st.button("🔍 Zoom E"):
                    show_full_image(e_path, f"E-Field (h={img_h}, λ={img_lam})")
            else:
                st.info("No E-field image")

            st.markdown("---")
            st.write("**Magnetic Field (|H|)**")
            if not m_rows.empty:
                m_path = m_rows.iloc[0]['path']
                st.image(m_path, use_container_width=True)
                if st.button("🔍 Zoom H"):
                    show_full_image(m_path, f"H-Field (h={img_h}, λ={img_lam})")
            else:
                st.info("No M-field image")
            
            found_any = True

    if not found_any:
        st.caption("Click a white circle to view fields.")


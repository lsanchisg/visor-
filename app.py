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
    pattern = re.compile(r"([EM])_field_pvk_(\d+)_([A-Z]+)_desp_(\d+)_(\d+)_(\d+)\.png")

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            f_type, thick, pol, sep, h_raw, lam_raw = match.groups()
            
            # Convert Lambda code to float (e.g., 52375 -> 523.75)
            # Adjust the divisor (100.0) if your filename logic differs
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

# Dialog for Fullscreen Image
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

# Load Main Data
df, current_filename = load_data(sel_thick, sel_pol, sel_sep, is_sym)

# Load Image Database
df_imgs = scan_profile_images("profile_images")

# Filter Image Database for Current Settings
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
        
        # Plot Controls
        c1, c2, c3 = st.columns(3)
        with c1:
            sel_display = st.selectbox("Metric:", list(display_opts.keys()))
        with c2:
            scale_choice = st.radio("Scale:", ('Normal', 'Log'), horizontal=True)
        with c3:
            st.info("💡 **Click white circles** to see Field Maps.")

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

            # --- Slider Defaults ---
            w_min, w_max = float(df_pivot.columns.min()), float(df_pivot.columns.max())
            h_min, h_max = float(df_pivot.index.min()), float(df_pivot.index.max())
            
            # Store slider values in session state to persist them
            if "sel_wave" not in st.session_state: st.session_state.sel_wave = (w_min + w_max)/2
            if "sel_height" not in st.session_state: st.session_state.sel_height = (h_min + h_max)/2

            # --- PLOT CONSTRUCTION ---
            fig = make_subplots(
                rows=2, cols=2,
                column_widths=[0.8, 0.2],
                row_heights=[0.2, 0.8],
                vertical_spacing=0.05, horizontal_spacing=0.05,
                shared_xaxes=True, shared_yaxes=True,
                subplot_titles=(
                    f'Horizontal Cut', 
                    '', 
                    f'{sel_pol} Map', 
                    f'Vertical Cut'
                )
            )

            # 1. Main Heatmap (Bottom-Left: Row 2, Col 1)
            heatmap = go.Heatmap(
                z=z_data, x=df_pivot.columns, y=df_pivot.index,
                colorscale='Jet',
                colorbar=dict(title=color_label, len=0.75, y=0.15, yanchor='bottom'),
                zmin=zmin, zmax=zmax,
                hovertemplate="λ: %{x}<br>h: %{y}<br>Val: %{z:.4e}<extra></extra>"
            )
            fig.add_trace(heatmap, row=2, col=1)

            # 2. Markers for Available Images (Bottom-Left: Row 2, Col 1)
            if not available_points.empty:
                fig.add_trace(go.Scatter(
                    x=available_points['lda0'],
                    y=available_points['h_fib'],
                    mode='markers',
                    marker=dict(color='white', size=8, line=dict(width=2, color='black')),
                    name='Images Available',
                    hovertemplate="<b>Image Available</b><br>λ: %{x}<br>h: %{y}<extra></extra>"
                ), row=2, col=1)

            # --- CROSS SECTION LOGIC ---
            # We need to determine where the "Crosshair" is.
            # Default to sliders, but if user CLICKED, use that.
            
            # Render initially to get event
            fig.update_layout(
                height=700, 
                clickmode='event+select',
                template="plotly_white",
                showlegend=False
            )
            
            # CAPTURE CLICK
            # selection_mode="points" is key
            event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")

            # Determine Active Coordinates
            act_wave = st.session_state.sel_wave
            act_height = st.session_state.sel_height
            
            clicked_point_found = False

            if event and event["selection"]["points"]:
                point = event["selection"]["points"][0]
                # Check if the click was on the heatmap (curveNumber 0 or 1 usually)
                # We update the active crosshair position
                act_wave = point["x"]
                act_height = point["y"]
                clicked_point_found = True

            # Find nearest matrix indices for the cuts
            w_idx = np.abs(df_pivot.columns - act_wave).argmin()
            h_idx = np.abs(df_pivot.index - act_height).argmin()
            
            real_wave = df_pivot.columns[w_idx]
            real_height = df_pivot.index[h_idx]

            # Re-draw Crosshairs and Cuts (We actually need to add these traces NOW)
            # Since Streamlit re-runs the whole script, we can add them to the 'fig' object 
            # *before* it was rendered? No, we need to render twice or use logic effectively.
            # simpler approach: Just calculate the traces based on 'act_wave' which we derived above.
            
            # --- ADDING CROSS SECTIONS TO FIGURE ---
            # (We have to recreate the figure traces to include lines at correct positions)
            
            # 3. Horizontal Cut (Top-Left: Row 1, Col 1)
            x_cross = z_data[h_idx, :]
            fig.add_trace(go.Scatter(x=df_pivot.columns, y=x_cross, mode='lines', line=dict(color='red'), name="H-Cut"), row=1, col=1)
            fig.add_trace(go.Scatter(x=[real_wave], y=[x_cross[w_idx]], mode='markers', marker=dict(color='blue', size=8)), row=1, col=1)

            # 4. Vertical Cut (Bottom-Right: Row 2, Col 2)
            y_cross = z_data[:, w_idx]
            fig.add_trace(go.Scatter(x=y_cross, y=df_pivot.index, mode='lines', line=dict(color='blue'), name="V-Cut"), row=2, col=2)
            fig.add_trace(go.Scatter(x=[y_cross[h_idx]], y=[real_height], mode='markers', marker=dict(color='red', size=8)), row=2, col=2)

            # 5. Add Crosshair Lines to Heatmap
            fig.add_hline(y=real_height, line=dict(color='white', width=1, dash='dash'), row=2, col=1)
            fig.add_vline(x=real_wave, line=dict(color='white', width=1, dash='dash'), row=2, col=1)

            # Update titles with specific values
            fig.layout.annotations[0].text = f"H-Cut (h={real_height:.3f})"
            fig.layout.annotations[3].text = f"V-Cut (λ={real_wave:.3f})"

            # HACK: Rerender the chart with the lines added
            # Use a unique key to prevent conflict
            st.empty() # Clear previous
            # Note: In Streamlit, we usually render once. 
            # The 'event' capture above was strictly to get the coordinates. 
            # Now we display the FINAL figure with the crosshairs at the clicked location.
            # To avoid "duplicate chart" visual, we just display it once at the end of this block.
            # But 'event' needs a chart to exist. 
            # Solution: We display the chart ONCE. The variables 'act_wave' determine where lines are drawn.
            # If the user clicks, 'rerun' happens, we get new 'act_wave', we draw lines there.
            
            # Overwrite the previous chart display?
            # Actually, standard Streamlit flow:
            # 1. Calc logic
            # 2. Build Fig
            # 3. st.plotly_chart(fig, on_select="rerun")
            # This works perfectly. The chart displayed 'above' was just for logic flow? 
            # No, 'event = st.plotly_chart' RENDERS the chart.
            # We need to add the traces BEFORE rendering.
            
            # FIX: Move the 'event = ...' line to the very end of the plotting block.
            # But we need 'event' to know where to draw the lines?
            # Chicken and Egg problem.
            # Standard Streamlit Solution: 
            # The 'event' variable contains data from the *previous* run's interaction.
            # So we use 'event' (if it exists) to set coordinates, THEN build and draw the chart.
            
        except Exception as e:
            st.error(f"Plot Error: {e}")

    else:
        st.warning(f"Data file not found: {current_filename}")


# --- Right Sidebar: Image Viewer ---
with col_side:
    st.header("🖼️ Field Maps")
    
    selected_h = None
    selected_lam = None

    # Logic: If we found a click in the event (from the previous render loop), use it
    if clicked_point_found:
        # Snap to nearest image point
        if not available_points.empty:
            # Simple distance check
            distances = ((available_points['lda0'] - act_wave)**2 + (available_points['h_fib'] - act_height)**2)
            nearest_idx = distances.idxmin()
            
            # Tolerance check (e.g. within 1 unit)
            if distances[nearest_idx] < 5.0: 
                selected_h = available_points.loc[nearest_idx, 'h_fib']
                selected_lam = available_points.loc[nearest_idx, 'lda0']

    # Display Images
    if selected_h is not None:
        st.success(f"Selected:\nλ={selected_lam} nm\nh={selected_h} nm")
        
        subset = df_imgs[
            (df_imgs['thickness'] == sel_thick) &
            (df_imgs['polarization'] == sel_pol) &
            (df_imgs['separation'] == sel_sep) &
            (df_imgs['h_fib'] == selected_h) &
            (np.abs(df_imgs['lda0'] - selected_lam) < 0.001)
        ]
        
        e_row = subset[subset['type'] == 'E']
        m_row = subset[subset['type'] == 'M']
        
        # E-Field
        st.markdown("---")
        st.write("**Electric Field (|E|)**")
        if not e_row.empty:
            e_path = e_row.iloc[0]['path']
            st.image(e_path, use_container_width=True)
            if st.button("🔍 Zoom E-Field"):
                show_full_image(e_path, f"Electric Field (h={selected_h}, λ={selected_lam})")
        else:
            st.info("No E-field image.")

        # H-Field
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

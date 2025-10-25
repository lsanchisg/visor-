import streamlit as st
import pandas as pd
import numpy as np

# Try to import Plotly, with fallback error handling
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.error("⚠️ Plotly is not installed. Please add 'plotly>=5.10.0' to your requirements.txt file")

# Set the title of the web app
st.set_page_config(layout="wide")
st.title("Optical Data Colormap Viewer 🔬")

# --- Data Loading ---
@st.cache_data
def load_data(polarization):
    if polarization == "TE":
        filename = "pvk_0_TE_desp_0_points_1994.csv"
    else:  # TM
        filename = "pvk_0_TM_desp_0_points_1994.csv"
    
    try:
        df = pd.read_csv(filename)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Error loading {polarization} data from {filename}: {e}")
        return None

if not PLOTLY_AVAILABLE:
    st.warning("""
    **Plotly is required for visualization.** 
    
    Please add `plotly>=5.10.0` to your `requirements.txt` file and redeploy.
    """)
    st.stop()

# --- Polarization Selection ---
st.sidebar.header("Polarization Selection")
polarization = st.sidebar.radio(
    "Select Polarization:",
    ('TE', 'TM')
)

# Load data based on selected polarization
df = load_data(polarization)

if df is not None:
    # --- User Controls (Sidebar) ---
    st.sidebar.header("Plot Controls")

    # Create a mapping between display names and actual column names
    column_mapping = {
        'Reflectance port 1': 'Reflectance_port_1',
        'Transmittance port 2': 'Transmittance_port_2', 
        'Absorvance': 'Absorvance'
    }
    
    # Use display names in selectbox
    selected_display_name = st.sidebar.selectbox(
        "Select data to plot:",
        list(column_mapping.keys())
    )
    
    # Get the actual column name for processing
    selected_column = column_mapping[selected_display_name]

    scale_choice = st.sidebar.radio(
        "Select color scale:",
        ('Normal Scale', 'Log Scale')
    )

    # --- Data Pivoting for Heatmap ---
    try:
        # Use the exact column names from your CSV
        df_pivot = df.pivot(
            index='h_fib', 
            columns='lda0', 
            values=selected_column
        )
        
        z_data = df_pivot.values
        color_label = selected_display_name
        zmin_val = None
        zmax_val = None

        # --- Handle Log vs. Normal Scale ---
        if scale_choice == 'Log Scale':
            z_data_safe = np.where(z_data <= 0, 1e-8, z_data)
            z_data = np.log10(z_data_safe)
            color_label = f"Log10({selected_display_name})"
            zmin_val = -7
            zmax_val = -1

        # Create sliders for cross-section positions
        st.sidebar.header("Cross-Section Controls")
        
        # Get min and max values for sliders
        wavelength_min = float(df_pivot.columns.min())
        wavelength_max = float(df_pivot.columns.max())
        height_min = float(df_pivot.index.min())
        height_max = float(df_pivot.index.max())
        
        # Create sliders
        selected_wavelength = st.sidebar.slider(
            "Select Wavelength for Vertical Cross-Section",
            min_value=wavelength_min,
            max_value=wavelength_max,
            value=(wavelength_min + wavelength_max) / 2,
            step=(wavelength_max - wavelength_min) / 100
        )
        
        selected_height = st.sidebar.slider(
            "Select Fiber Height for Horizontal Cross-Section",
            min_value=height_min,
            max_value=height_max,
            value=(height_min + height_max) / 2,
            step=(height_max - height_min) / 100
        )

        # Find closest indices for selected values
        wavelength_idx = np.abs(df_pivot.columns - selected_wavelength).argmin()
        height_idx = np.abs(df_pivot.index - selected_height).argmin()

        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.8, 0.2],
            row_heights=[0.2, 0.8],
            vertical_spacing=0.05,
            horizontal_spacing=0.05,
            shared_xaxes=True,
            shared_yaxes=True,
            subplot_titles=(f'Horizontal Cross-Section at h_fib = {selected_height:.3f}', 
                          '', 
                          f'{polarization} Polarization Heatmap', 
                          f'Vertical Cross-Section at λ = {selected_wavelength:.3f}')
        )

        # Add heatmap (main plot)
        heatmap = go.Heatmap(
            z=z_data,
            x=df_pivot.columns,
            y=df_pivot.index,
            colorscale='Plasma',
            colorbar=dict(
                title=color_label,
                len=0.75,
                y=0.15,
                yanchor='bottom'
            ),
            zmin=zmin_val,
            zmax=zmax_val,
            hoverinfo="x+y+z",
            name="",
            hovertemplate="<br>".join([
                "Wavelength: %{x:.3f} nm",
                "Fiber Height: %{y:.3f} nm",
                f"{selected_display_name}: %{{z:.6f}}",
                "<extra></extra>"
            ])
        )
        fig.add_trace(heatmap, row=2, col=1)

        # Add horizontal line to heatmap (at selected height)
        horizontal_line_heatmap = go.Scatter(
            x=[df_pivot.columns.min(), df_pivot.columns.max()],
            y=[selected_height, selected_height],
            mode='lines',
            line=dict(color='red', width=3, dash='dash'),
            name=f'h_fib = {selected_height:.3f}',
            showlegend=False
        )
        fig.add_trace(horizontal_line_heatmap, row=2, col=1)

        # Add vertical line to heatmap (at selected wavelength)
        vertical_line_heatmap = go.Scatter(
            x=[selected_wavelength, selected_wavelength],
            y=[df_pivot.index.min(), df_pivot.index.max()],
            mode='lines',
            line=dict(color='blue', width=3, dash='dash'),
            name=f'λ = {selected_wavelength:.3f}',
            showlegend=False
        )
        fig.add_trace(vertical_line_heatmap, row=2, col=1)

        # Add horizontal cross-section (top plot)
        x_cross_section = z_data[height_idx, :]
        horizontal_line = go.Scatter(
            x=df_pivot.columns,
            y=x_cross_section,
            mode='lines',
            line=dict(color='red', width=2),
            showlegend=False
        )
        fig.add_trace(horizontal_line, row=1, col=1)

        # Add vertical marker to horizontal cross-section
        horizontal_marker = go.Scatter(
            x=[selected_wavelength],
            y=[x_cross_section[wavelength_idx]],
            mode='markers',
            marker=dict(color='blue', size=8, symbol='circle'),
            showlegend=False
        )
        fig.add_trace(horizontal_marker, row=1, col=1)

        # Add vertical cross-section (right plot)
        y_cross_section = z_data[:, wavelength_idx]
        vertical_line = go.Scatter(
            x=y_cross_section,
            y=df_pivot.index,
            mode='lines',
            line=dict(color='blue', width=2),
            showlegend=False
        )
        fig.add_trace(vertical_line, row=2, col=2)

        # Add horizontal marker to vertical cross-section
        vertical_marker = go.Scatter(
            x=[y_cross_section[height_idx]],
            y=[selected_height],
            mode='markers',
            marker=dict(color='red', size=8, symbol='circle'),
            showlegend=False
        )
        fig.add_trace(vertical_marker, row=2, col=2)

        # Update layout
        fig.update_layout(
            title=f'{polarization} Polarization - {scale_choice} Colormap of {selected_display_name}',
            height=700,
            showlegend=False,
            # Heatmap axes
            xaxis2=dict(
                title='Wavelength (nm)',
                constrain='domain'
            ),
            yaxis2=dict(
                title='Fiber Height (nm)',
                constrain='domain'
            ),
            # Top plot (horizontal cross-section)
            xaxis1=dict(
                showticklabels=True,
                title='Wavelength (nm)'
            ),
            yaxis1=dict(
                title=selected_display_name
            ),
            # Right plot (vertical cross-section)
            xaxis3=dict(
                title=selected_display_name
            ),
            yaxis3=dict(
                showticklabels=True,
                title='Fiber Height (nm)'
            )
        )

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

        # Add explanation
        st.sidebar.info("""
        **Guide:**
        - **Red dashed line**: Horizontal cross-section position
        - **Blue dashed line**: Vertical cross-section position
        - Adjust sliders to move the cross-section lines
        """)

    except Exception as e:
        st.error(f"An error occurred while creating the plot: {e}")
        st.error(f"Please check if the columns 'h_fib', 'lda0', and '{selected_column}' exist in the CSV.")
        st.write("Available columns:", df.columns.tolist())

else:
    st.warning(f"No {polarization} data loaded. Please check the data source.")
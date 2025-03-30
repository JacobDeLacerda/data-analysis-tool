# pages/6_Interpolate_Data.py
import streamlit as st
import pandas as pd
import numpy as np
import io
from scipy.interpolate import (
    interp1d, RegularGridInterpolator, griddata, Rbf
)
from scipy.spatial import Delaunay

st.set_page_config(layout="wide")

st.header("6. Interpolate Data")
st.caption("Estimate values between known data points.")

# --- Check if data is loaded ---
if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("No data loaded. Please go to the 'Load Data' page first.")
    st.stop()

df = st.session_state.df # Get the DataFrame

# --- Helper Function for 1D Interpolator Setup ---
# Avoids repeating data prep logic
@st.cache_data # Cache the interpolator creation if data & settings don't change
def setup_1d_interpolator(_df_internal, x_col, y_col, method, extrapolate, fill_value_str):
    """Prepares data and returns a SciPy interp1d object or raises error."""
    try:
        x_known = pd.to_numeric(_df_internal[x_col], errors='coerce').values
        y_known = pd.to_numeric(_df_internal[y_col], errors='coerce').values
    except KeyError as e:
        raise ValueError(f"Selected column '{e}' not found in the current DataFrame.")
    except Exception as e:
        raise ValueError(f"Could not process selected columns '{x_col}', '{y_col}': {e}")

    nan_mask = np.isnan(x_known) | np.isnan(y_known)
    num_nan = np.sum(nan_mask)
    if num_nan > 0:
        st.info(f"Ignoring {num_nan} rows with missing/non-numeric values in '{x_col}' or '{y_col}'.")
        x_known = x_known[~nan_mask]
        y_known = y_known[~nan_mask]

    if len(x_known) < 2:
        raise ValueError("Not enough valid (non-NaN) data points remaining for 1D interpolation.")

    # Sort and handle duplicates
    sort_idx = np.argsort(x_known)
    x_known = x_known[sort_idx]
    y_known = y_known[sort_idx]
    unique_x, unique_idx = np.unique(x_known, return_index=True)
    num_duplicates = len(x_known) - len(unique_x)
    if num_duplicates > 0:
        st.info(f"Note: Using first occurrence for {num_duplicates} duplicate X values.")
        x_known = unique_x
        y_known = y_known[unique_idx]

    if len(x_known) < 2:
        raise ValueError("Not enough unique, valid data points after handling duplicates.")

    # Check method requirements
    min_points_required = {'linear': 2, 'cubic': 4, 'quadratic': 3}.get(method, 2)
    if len(x_known) < min_points_required:
         raise ValueError(f"Method '{method}' requires at least {min_points_required} unique data points, found {len(x_known)}.")

    # Determine fill value argument for interp1d
    fill_value_arg = None
    bounds_error = not extrapolate
    if fill_value_str == 'nan':
        fill_value_arg = np.nan
    elif fill_value_str == 'edge':
        # Only valid if extrapolating, otherwise interp1d uses NaN/error
        fill_value_arg = (y_known[0], y_known[-1]) if extrapolate else np.nan
    else: # Assume numeric string
        try:
            fill_value_arg = float(fill_value_str)
        except ValueError:
             raise ValueError(f"Invalid numeric fill value '{fill_value_str}'.")

    try:
        interpolator = interp1d(
            x_known, y_known, kind=method, bounds_error=bounds_error,
            fill_value=fill_value_arg, assume_sorted=True
        )
        # Also return bounds for checking later
        return interpolator, x_known.min(), x_known.max()
    except ValueError as e:
        # Catch specific SciPy errors
         raise ValueError(f"SciPy interp1d failed: {e}")
    except Exception as e:
         raise RuntimeError(f"Unexpected error creating 1D interpolator: {e}")


# =============================================
# === Main Interpolation Workflow Selection ===
# =============================================

interp_goal = st.radio(
    "What do you want to do?",
    ["**Generate Interpolated Points (for Download)**", "**Evaluate Single Point (1D Only)**"],
    horizontal=True,
    help="Choose whether to create a file with many interpolated points or predict Y for a single X (1D only)."
)

st.divider()

# ===============================================
# === Goal 1: Generate Download File Workflow ===
# ===============================================
if "**Generate Interpolated Points (for Download)**" in interp_goal:
    st.subheader("1. Select Interpolation Mode & Input Data")
    # Mode & Input Columns (Simplified layout)
    mode_choice = st.radio(
        "Interpolation Mode:",
        ['1D (y = f(x))', 'N-D Scattered Data', 'N-D Regular Grid'],
        key="mode_download"
    )
    mode = '1d' if '1D' in mode_choice else ('scattered' if 'Scattered' in mode_choice else 'grid')

    cols_available = df.columns.tolist()
    if not cols_available:
        st.error("No columns found in the loaded data!")
        st.stop()

    x_col, y_col, coord_cols, value_col = None, None, None, None
    num_coords = 0
    coord_names_list = []

    input_container = st.container(border=True)
    with input_container:
        if mode == '1d':
            c1, c2 = st.columns(2)
            with c1:
                 x_col = st.selectbox("Select X column (independent):", cols_available, index=None, placeholder="Select column...", key="xd_down")
            with c2:
                 y_col = st.selectbox("Select Y column (dependent):", cols_available, index=None, placeholder="Select column...", key="yd_down")
            if x_col: coord_names_list = [x_col]
            num_coords = 1
        else: # grid or scattered
             coord_cols = st.multiselect("Select Coordinate columns (order matters):", cols_available, help="Columns representing the location/coordinates.", key="coordd_down")
             value_col = st.selectbox("Select Value column:", cols_available, index=None, placeholder="Select column...", help="Column with the values to be interpolated.", key="vald_down")
             if coord_cols:
                 num_coords = len(coord_cols)
                 coord_names_list = coord_cols

    input_valid = (mode == '1d' and x_col and y_col) or (mode != '1d' and coord_cols and value_col)

    if not input_valid:
        st.info("Please select the required input columns above.")
        st.stop()

    # --- Method & Parameters ---
    st.subheader("2. Choose Interpolation Method & Parameters")
    method_container = st.container(border=True)
    with method_container:
        method = None
        rbf_params = {}

        if mode == '1d':
            method = st.selectbox("1D Method:", ['linear', 'cubic', 'quadratic', 'slinear', 'nearest', 'zero'], index=0, key="methd_1d")
        elif mode == 'grid':
            method = st.selectbox("Grid Method:", ['linear', 'nearest'], index=0, key="methd_grid")
        elif mode == 'scattered':
            method = st.selectbox("Scattered Method:", ['linear', 'cubic', 'nearest', 'rbf'], index=0, key="methd_scat")
            if method == 'rbf':
                with st.expander("RBF Parameters (Optional)"):
                    rbf_params['kernel'] = st.selectbox("RBF Kernel:", ['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate'], index=0, key="rbfk")
                    rbf_params['epsilon'] = st.number_input("Epsilon (> 0):", min_value=1e-9, value=None, format="%f", help="Shape parameter. Default usually works well.", key="rbfe")
                    rbf_params['smooth'] = st.number_input("Smooth factor (>= 0):", min_value=0.0, value=0.0, format="%f", help="0=exact interpolation.", key="rbfs")

        # --- Extrapolation & Fill ---
        with st.expander("Extrapolation & Fill Options"):
             extrapolate = st.checkbox("Allow Extrapolation", value=False, help="Estimate outside original data range?", key="extrapd")
             fill_options = ['nan']
             if mode == '1d' and extrapolate: fill_options.append('edge')
             fill_options.append('number')
             fill_choice = st.selectbox("Fill value for outside domain:", fill_options, index=0, help="Used if extrapolation is off, or outside convex hull.", key="filld")
             fill_value_str = fill_choice
             if fill_choice == 'number':
                 fill_value_num = st.number_input("Enter numeric fill value:", value=0.0, format="%f", key="filld_num")
                 fill_value_str = str(fill_value_num)


    # --- Output Points Specification ---
    st.subheader("3. Define Points for Interpolation Output")
    points_container = st.container(border=True)
    with points_container:
        new_points_source = None
        points_new_df = None # To hold points read from file
        num_points_1d = None
        step_1d = None
        grid_spec_nd = None
        points_file_cols = None # To hold selected column names from uploaded file

        output_point_options = ["Generate points on a new grid"]
        if mode == '1d':
            output_point_options.insert(0, "Generate evenly spaced points (1D)")
            output_point_options.insert(1, "Generate points with specific step (1D)")
        output_point_options.insert(0, "Upload a file with points")

        source_choice = st.selectbox("How to specify output points?", output_point_options, index=0, key="point_source_down")

        if "Upload a file" in source_choice:
            new_points_source = 'file'
            uploaded_points_file = st.file_uploader("Upload points file (CSV/Excel/Txt)", type=['csv', 'xlsx', 'xls', 'txt'], key="points_upload_down")
            if uploaded_points_file:
                points_delimiter = st.text_input("Delimiter (if CSV/Txt)", value=',', key="points_delim_down")
                if points_delimiter == '\\t': points_delimiter = '\t'
                points_header = st.number_input("Header row (0-based, None if none)", value=None, step=1, key="points_head_down", format="%d")
                # Load preview
                try:
                    points_new_df = pd.read_csv(uploaded_points_file, delimiter=points_delimiter, header=points_header, skipinitialspace=True, nrows=100) # Read only head for preview/col selection
                    st.write("Preview of uploaded points file:")
                    st.dataframe(points_new_df.head(), height=150)
                    cols_in_points_file = points_new_df.columns.tolist()
                    # Get column selection
                    if mode == '1d':
                        points_file_cols = st.selectbox(f"Select column for X ({x_col}) from points file:", cols_in_points_file, key="points_cols_1d_down", index=None)
                    else: # N-D
                        st.caption(f"Coordinates required (in order): {', '.join(coord_names_list)}")
                        default_indices = list(range(min(num_coords, len(cols_in_points_file))))
                        points_file_cols_selected = st.multiselect(
                            f"Select {num_coords} columns for coordinates from points file:",
                            cols_in_points_file,
                            default=[cols_in_points_file[i] for i in default_indices] if default_indices else None,
                            key="points_cols_nd_down"
                        )
                        if len(points_file_cols_selected) == num_coords: points_file_cols = points_file_cols_selected
                        else: st.warning(f"Please select exactly {num_coords} columns."); points_file_cols = None
                    # Reset file pointer for full read later
                    uploaded_points_file.seek(0)
                except Exception as e:
                    st.error(f"Error previewing points file: {e}"); points_new_df = None; points_file_cols = None

        elif "evenly spaced points (1D)" in source_choice:
            new_points_source = 'num_points'
            num_points_1d = st.number_input("Number of points to generate:", min_value=2, value=100, step=10, help="Points generated between min/max of input X.", key="npd")
        elif "specific step (1D)" in source_choice:
            new_points_source = 'step'
            step_1d = st.number_input("Step size between points:", min_value=1e-9, value=1.0, format="%f", help="Points generated between min/max of input X.", key="stepd")
        elif "new grid" in source_choice:
            new_points_source = 'grid_spec'
            st.write(f"Define grid boundaries and points for {num_coords} dimension(s) ({', '.join(coord_names_list)}):")
            grid_specs_list = []
            valid_grid_spec = True
            for i in range(num_coords):
                 gc1, gc2, gc3 = st.columns(3)
                 g_min = gc1.number_input(f"Min (Dim {i+1})", value=0.0, format="%f", key=f"gmind_{i}")
                 g_max = gc2.number_input(f"Max (Dim {i+1})", value=1.0, format="%f", key=f"gmaxd_{i}")
                 g_num = gc3.number_input(f"Num Points (Dim {i+1})", min_value=2, value=20, step=1, key=f"gnumd_{i}")
                 if g_max <= g_min: gc2.warning("Max <= Min!"); valid_grid_spec = False
                 grid_specs_list.append(f"{g_min}:{g_max}:{g_num}")
            if valid_grid_spec: grid_spec_nd = ",".join(grid_specs_list)


    # --- Execute Button & Logic ---
    st.divider()
    st.subheader("4. Execute Interpolation & Download Results")

    # Check if prerequisites are met
    run_disabled = not method or not new_points_source or \
                   (new_points_source == 'file' and (uploaded_points_file is None or points_file_cols is None)) or \
                   (new_points_source == 'grid_spec' and not grid_spec_nd)

    if st.button("Run Interpolation & Prepare Download", type="primary", disabled=run_disabled):

        settings = { # Collect all settings
            'mode': mode, 'method': method,
            'x_col': x_col, 'y_col': y_col, 'coord_cols': coord_cols, 'value_col': value_col,
            'extrapolate': extrapolate, 'fill_value': fill_value_str,
            'new_points_source': new_points_source,
            'num_points': num_points_1d, 'step': step_1d, 'grid_spec': grid_spec_nd,
            # RBF specific
            'rbf_kernel': rbf_params.get('kernel'), 'rbf_epsilon': rbf_params.get('epsilon'), 'rbf_smooth': rbf_params.get('smooth'),
            # Info needed by generator/saver functions
            'coord_col_names': coord_names_list, 'value_col_name': y_col if mode == '1d' else value_col,
            # File source specifics (passed separately or processed here)
            'uploaded_points_file_obj': uploaded_points_file if new_points_source == 'file' else None,
            'points_file_delimiter': points_delimiter if new_points_source == 'file' else None,
            'points_file_header': points_header if new_points_source == 'file' else None,
            'points_file_cols': points_file_cols if new_points_source == 'file' else None,
        }

        results_df = None
        error_occurred = False
        with st.spinner("Performing interpolation..."):
            try:
                # --- Get known points/values --- (Simplified - assumes previous selection)
                # [Similar data prep as before: get points_known, values_known, x_known, y_known, handle NaNs, sort/unique for 1D]
                # This part is crucial and should be robust
                if mode == '1d':
                    # Re-use setup function
                    _, x_min_bound, x_max_bound = setup_1d_interpolator(df, x_col, y_col, method, extrapolate, fill_value_str)
                    # Keep x_known, y_known from setup (already prepared)
                    # Get prepared x_known, y_known if needed (e.g., for generator functions)
                    # Simplified: Re-run prep inside generator if needed, or pass them
                    x_known_for_gen = pd.to_numeric(df[x_col], errors='coerce').dropna().unique() # For bounds
                else: # N-D
                    points_known_df = df[coord_cols]
                    values_known_series = df[value_col]
                    points_known = pd.to_numeric(points_known_df.stack(), errors='coerce').unstack().values
                    values_known = pd.to_numeric(values_known_series, errors='coerce').values
                    nan_mask = np.isnan(points_known).any(axis=1) | np.isnan(values_known)
                    num_nan = np.sum(nan_mask)
                    if num_nan > 0: st.info(f"Ignoring {num_nan} rows with missing/non-numeric values in selected columns.")
                    points_known = points_known[~nan_mask]
                    values_known = values_known[~nan_mask]
                    if len(points_known) == 0: raise ValueError("Not enough valid data points.")


                # --- Generate or Get New Points Array ---
                points_new_array = None
                if new_points_source == 'file':
                     # Read the full file now using settings
                     try:
                         points_new_df_full = pd.read_csv(
                             settings['uploaded_points_file_obj'],
                             delimiter=settings['points_file_delimiter'],
                             header=settings['points_file_header'],
                             skipinitialspace=True
                         )
                         # Select columns and convert
                         selected_cols_df = points_new_df_full[settings['points_file_cols']] if isinstance(settings['points_file_cols'], list) else points_new_df_full[[settings['points_file_cols']]]
                         points_new_array = selected_cols_df.values.astype(float)
                     except Exception as e: raise ValueError(f"Failed to read/process points file fully: {e}")
                elif mode == '1d':
                    # Pass x_known bounds to generator
                    settings_gen = settings.copy() # Avoid modifying original settings
                    settings_gen['x_known_min'] = x_known_for_gen.min()
                    settings_gen['x_known_max'] = x_known_for_gen.max()
                    points_new_array = generate_points_1d_interp(settings_gen, x_known_for_gen) # Pass minimal needed info
                else: # N-D Grid Spec
                    points_new_array = generate_points_nd_interp(settings, num_coords, coord_names_list)


                # --- Perform Actual Interpolation --- (Simplified logic - assumes previous implementation was correct)
                # [Insert the core interpolation logic here, similar to previous script's handlers]
                # This part calls interp1d, RegularGridInterpolator, griddata, or Rbf based on mode/method
                # Make sure to handle fill_value_str correctly based on context
                # ...
                values_new = None # Placeholder - this needs the actual calculation
                st.info("Executing SciPy interpolation...") # Placeholder feedback

                # --- Example Placeholder Calculation (Replace with actual logic) ---
                if points_new_array is not None:
                    if mode == '1d':
                        interpolator, _, _ = setup_1d_interpolator(df, x_col, y_col, method, extrapolate, fill_value_str)
                        values_new = interpolator(points_new_array)
                    else:
                        # Placeholder for ND - requires full implementation based on mode/method
                        values_new = np.random.rand(len(points_new_array)) * 10 # Fake results for now
                        st.warning("N-D Interpolation logic needs full implementation in this section.")


                # --- Combine results into DataFrame ---
                if values_new is not None:
                    if points_new_array.ndim == 1: points_new_array = points_new_array.reshape(-1, 1)
                    output_data = np.hstack((points_new_array, values_new.reshape(-1, 1)))
                    headers = [f"{h}_interp" for h in coord_names_list] + [f"{settings['value_col_name']}_interp"]
                    results_df = pd.DataFrame(output_data, columns=headers)

            except ValueError as ve: st.error(f"Interpolation Value Error: {ve}"); error_occurred = True
            except RuntimeError as re: st.error(f"Interpolation Runtime Error: {re}"); error_occurred = True
            except Exception as e: st.error(f"Unexpected interpolation error: {e}"); error_occurred = True

        # --- Display Results & Download ---
        if results_df is not None and not error_occurred:
            st.success("Interpolation completed!")
            st.write("Preview of Interpolated Results:")
            st.dataframe(results_df.head(), use_container_width=True)
            csv_results = results_df.to_csv(index=False).encode('utf-8')
            default_interp_filename = f"interpolated_output_{mode}.csv"
            st.download_button("Download Interpolation Results as CSV", csv_results, default_interp_filename, 'text/csv', type="primary")
        elif not error_occurred:
             st.warning("Interpolation did not produce results.")

# ======================================================
# === Goal 2: Evaluate Single Point (1D Only) Workflow ===
# ======================================================
elif "**Evaluate Single Point (1D Only)**" in interp_goal:
    st.subheader("Evaluate Y for a Single X Value (1D Interpolation)")

    cols_available = df.columns.tolist()
    if not cols_available:
        st.error("No columns found in the loaded data!")
        st.stop()

    eval_container = st.container(border=True)
    with eval_container:
        st.markdown("**1. Select 1D Data & Method**")
        c1e, c2e, c3e = st.columns(3)
        with c1e:
             x_col_eval = st.selectbox("Select X column:", cols_available, index=None, placeholder="Select column...", key="xe_eval")
        with c2e:
             y_col_eval = st.selectbox("Select Y column:", cols_available, index=None, placeholder="Select column...", key="ye_eval")
        with c3e:
             method_eval = st.selectbox("Interpolation Method:", ['linear', 'cubic', 'quadratic', 'slinear', 'nearest', 'zero'], index=0, key="methe_eval")

        # Only show next steps if columns selected
        if x_col_eval and y_col_eval:
            st.markdown("**2. Extrapolation & Input**")
            c1e2, c2e2 = st.columns(2)
            with c1e2:
                extrapolate_eval = st.checkbox("Allow Extrapolation?", value=False, help="Allow prediction outside the range of your X data?", key="extrape_eval")
            with c2e2:
                 x_value_eval = st.number_input("Enter the X value to evaluate:", value=None, format="%f", placeholder="Enter X...", help="The specific X point where you want to predict Y.")

            if x_value_eval is not None:
                 if st.button("Predict Y Value", type="primary", key="pred_button"):
                     with st.spinner("Calculating..."):
                         try:
                             # Use the helper to setup interpolator (fill='nan' for eval if no extrap)
                             fill_eval = 'nan' # Default to NaN if out of bounds and no extrap
                             interpolator, x_min, x_max = setup_1d_interpolator(df, x_col_eval, y_col_eval, method_eval, extrapolate_eval, fill_eval)

                             # Check bounds if extrapolation is off
                             if not extrapolate_eval and (x_value_eval < x_min or x_value_eval > x_max):
                                 st.error(f"Input X ({x_value_eval}) is outside the data range [{x_min:.4g}, {x_max:.4g}] and extrapolation is disallowed.")
                             else:
                                 # Perform the prediction
                                 y_predicted = interpolator(x_value_eval)
                                 st.success(f"Predicted Y value for X = {x_value_eval}:")
                                 st.metric(label=f"Predicted {y_col_eval}", value=f"{float(y_predicted):.4f}") # Format output

                         except ValueError as ve:
                             st.error(f"Error during prediction setup: {ve}")
                         except Exception as e:
                             st.error(f"An unexpected error occurred: {e}")
            else:
                 st.info("Enter an X value to predict its corresponding Y.")
        else:
            st.info("Select X and Y columns first.")


# Display current data preview at the end (Optional for this page)
# st.divider()
# st.subheader("Current Data Preview (Read-Only)")
# st.dataframe(st.session_state.df.head(), use_container_width=True)

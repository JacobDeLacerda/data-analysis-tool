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
st.caption("Estimate values between known data points. Results are generated for download.")

# --- Check if data is loaded ---
if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("No data loaded. Please go to the 'Load Data' page first.")
    st.stop()

df = st.session_state.df # Get the DataFrame

# --- Interpolation Settings Gathering ---
st.subheader("1. Select Interpolation Mode & Input Data")

# Use columns for layout
col_mode, col_input = st.columns(2)

with col_mode:
    mode_choice = st.radio(
        "Interpolation Mode:",
        ['1D (y = f(x))', 'N-D Scattered Data', 'N-D Regular Grid'],
        help="Choose based on your data structure and interpolation needs."
    )
    mode = '1d' if '1D' in mode_choice else ('scattered' if 'Scattered' in mode_choice else 'grid')

with col_input:
    x_col, y_col, coord_cols, value_col = None, None, None, None
    num_coords = 0
    coord_names_list = []

    cols_available = df.columns.tolist()
    if not cols_available:
        st.error("No columns found in the loaded data!")
        st.stop()

    if mode == '1d':
        x_col = st.selectbox("Select X column (independent):", cols_available, index=None, placeholder="Select column...")
        y_col = st.selectbox("Select Y column (dependent):", cols_available, index=None, placeholder="Select column...")
        if x_col: coord_names_list = [x_col]
        num_coords = 1
    else: # grid or scattered
        coord_cols = st.multiselect("Select Coordinate columns (order matters):", cols_available, help="Columns representing the location/coordinates.")
        value_col = st.selectbox("Select Value column:", cols_available, index=None, placeholder="Select column...", help="Column with the values to be interpolated.")
        if coord_cols:
             num_coords = len(coord_cols)
             coord_names_list = coord_cols


# --- Check if input columns are selected ---
input_valid = False
if mode == '1d' and x_col and y_col: input_valid = True
if mode != '1d' and coord_cols and value_col: input_valid = True

if not input_valid:
    st.info("Please select the required input columns above.")
    st.stop()

st.divider()
st.subheader("2. Choose Interpolation Method & Parameters")

method = None
rbf_params = {}
extrapolate = False
fill_value_str = 'nan'

col_meth, col_param = st.columns(2)

with col_meth:
    # Method selection based on mode
    if mode == '1d':
        method = st.selectbox(
            "1D Method:",
            ['linear', 'cubic', 'quadratic', 'slinear', 'nearest', 'zero'], # Common useful ones
            index=0, help="Linear=straight lines, Cubic=smooth curves"
        )
    elif mode == 'grid':
         method = st.selectbox(
             "Grid Method:",
             ['linear', 'nearest'], # Only these are guaranteed in older scipy
             index=0, help="Linear=multilinear, Nearest=piecewise constant"
         )
    elif mode == 'scattered':
         method = st.selectbox(
             "Scattered Method:",
             ['linear', 'cubic', 'nearest', 'rbf'],
             index=0, help="Linear/Cubic use triangulation, RBF uses basis functions"
         )

    # Extrapolation & Fill Value
    extrapolate = st.checkbox("Allow Extrapolation", value=False, help="Estimate values outside the range of original data? Can be unstable.")

    fill_options = ['nan (missing value)']
    if mode == '1d' and extrapolate: # 'edge' only for interp1d extrapolation
         fill_options.append('edge (boundary value)')
    fill_options.append('a specific number')
    fill_choice = st.selectbox(
        "Fill value for outside domain:",
        fill_options, index=0,
        help="Value used if extrapolation is off, or outside convex hull (scattered)."
    )
    if 'number' in fill_choice:
        fill_value_num = st.number_input("Enter numeric fill value:", value=0.0, format="%f")
        fill_value_str = str(fill_value_num)
    elif 'edge' in fill_choice:
        fill_value_str = 'edge'
    else:
        fill_value_str = 'nan'


with col_param:
    # RBF specific parameters
    if mode == 'scattered' and method == 'rbf':
        with st.expander("RBF Parameters (Optional)"):
            rbf_params['kernel'] = st.selectbox(
                "RBF Kernel:",
                ['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate'],
                index=0
            )
            rbf_params['epsilon'] = st.number_input(
                "Epsilon (shape parameter, > 0):",
                min_value=1e-9, value=None, format="%f", # Default handled by Rbf based on data
                help="Adjusts smoothness/influence. Leave blank for default (recommended)."
            )
            rbf_params['smooth'] = st.number_input(
                "Smooth factor (>= 0):",
                min_value=0.0, value=0.0, format="%f",
                help="0 = exact interpolation, > 0 = approximation (smoothing)."
            )


st.divider()
st.subheader("3. Define Points for Interpolation Output")

new_points_source = None
points_new_df = None # To hold points read from file
num_points_1d = None
step_1d = None
grid_spec_nd = None

output_point_options = ["Generate points on a new grid"] # Always offer grid
if mode == '1d':
    output_point_options.insert(0, "Generate evenly spaced points (1D)")
    output_point_options.insert(1, "Generate points with specific step (1D)")
output_point_options.insert(0, "Upload a file with points") # Always offer file upload

source_choice = st.selectbox("How to specify output points?", output_point_options, index=0)

if "Upload a file" in source_choice:
    new_points_source = 'file'
    uploaded_points_file = st.file_uploader(
        "Upload file containing coordinates for interpolation",
        type=['csv', 'xlsx', 'xls', 'txt'],
        accept_multiple_files=False,
        key="points_uploader"
    )
    if uploaded_points_file:
        st.caption(f"File: {uploaded_points_file.name}")
        # Add options for delimiter, header etc for this file
        # Simplified for now: Assume CSV, no header by default
        points_delimiter = st.text_input("Delimiter for points file", value=',', key="points_delim")
        if points_delimiter == '\\t': points_delimiter = '\t'
        points_header = st.number_input("Header row in points file (0-based, None if none)", value=None, step=1, key="points_head", format="%d")

        try:
            points_new_df = pd.read_csv(uploaded_points_file, delimiter=points_delimiter, header=points_header, skipinitialspace=True)
            st.write("Preview of uploaded points file:")
            st.dataframe(points_new_df.head(), height=150)

            # Get column selection from this *new* DataFrame
            st.caption("Select columns from the uploaded file corresponding to the interpolation coordinates:")
            cols_in_points_file = points_new_df.columns.tolist()
            if mode == '1d':
                points_file_cols = st.selectbox(f"Select column for X ({x_col}):", cols_in_points_file, key="points_cols_1d", index=None)
            else: # N-D
                st.caption(f"Coordinates required (in order): {', '.join(coord_names_list)}")
                # Simple approach: assume columns are in correct order by index
                default_indices = list(range(min(num_coords, len(cols_in_points_file))))
                points_file_cols_selected = st.multiselect(
                    f"Select {num_coords} columns for coordinates:",
                    cols_in_points_file,
                    default=[cols_in_points_file[i] for i in default_indices] if default_indices else None,
                    key="points_cols_nd"
                )
                if len(points_file_cols_selected) == num_coords:
                     points_file_cols = points_file_cols_selected # Use the selected list
                else:
                     st.warning(f"Please select exactly {num_coords} columns from the points file.")
                     points_file_cols = None # Invalid selection

        except Exception as e:
            st.error(f"Error reading points file: {e}")
            points_new_df = None # Ensure it's None on error
            points_file_cols = None


elif "evenly spaced points (1D)" in source_choice:
    new_points_source = 'num_points'
    num_points_1d = st.number_input("Number of points to generate:", min_value=2, value=100, step=10, help="Points generated between min/max of input X column.")
elif "specific step (1D)" in source_choice:
    new_points_source = 'step'
    step_1d = st.number_input("Step size between points:", min_value=1e-9, value=1.0, format="%f", help="Points generated between min/max of input X column.")
elif "new grid" in source_choice:
    new_points_source = 'grid_spec'
    st.write(f"Define grid boundaries and number of points for {num_coords} dimension(s):")
    st.caption(f"Coordinate order: {', '.join(coord_names_list)}")
    grid_specs_list = []
    valid_grid_spec = True
    for i in range(num_coords):
        col_grid_min, col_grid_max, col_grid_num = st.columns(3)
        with col_grid_min:
            g_min = st.number_input(f"Min (Dim {i+1}: {coord_names_list[i]})", value=0.0, format="%f", key=f"gmin_{i}")
        with col_grid_max:
            g_max = st.number_input(f"Max (Dim {i+1}: {coord_names_list[i]})", value=1.0, format="%f", key=f"gmax_{i}")
        with col_grid_num:
            g_num = st.number_input(f"Num Points (Dim {i+1})", min_value=2, value=20, step=1, key=f"gnum_{i}")
        if g_max <= g_min:
             st.warning(f"Max must be greater than Min for Dim {i+1}", icon="⚠️")
             valid_grid_spec = False
        grid_specs_list.append(f"{g_min}:{g_max}:{g_num}")
    if valid_grid_spec:
        grid_spec_nd = ",".join(grid_specs_list)


st.divider()
st.subheader("4. Execute Interpolation & Download Results")

if st.button("Run Interpolation", type="primary", disabled=not method):

    # --- Prepare settings dict for internal handlers ---
    settings = {
        'mode': mode, 'method': method,
        'x_col': x_col, 'y_col': y_col, 'coord_cols': coord_cols, 'value_col': value_col,
        'extrapolate': extrapolate, 'fill_value': fill_value_str,
        'new_points_source': new_points_source,
        'num_points': num_points_1d, 'step': step_1d, 'grid_spec': grid_spec_nd,
        # RBF specific
        'rbf_kernel': rbf_params.get('kernel'),
        'rbf_epsilon': rbf_params.get('epsilon'),
        'rbf_smooth': rbf_params.get('smooth'),
        # Info needed by generator/saver functions
        'coord_col_names': coord_names_list,
        'value_col_name': y_col if mode == '1d' else value_col,
    }

    # --- Handle File Upload Source ---
    points_new_array = None
    if new_points_source == 'file':
        if points_new_df is not None and points_file_cols is not None:
            try:
                # Select the specified columns and convert to numpy array
                selected_cols_df = points_new_df[points_file_cols] if isinstance(points_file_cols, list) else points_new_df[[points_file_cols]]
                points_new_array = selected_cols_df.values.astype(float)
                # Need to pass points_new_array directly, or modify generate_points helpers
                # For now, let's pass it and skip the generate_points helpers for file case
            except KeyError as e:
                st.error(f"Column '{e}' not found in the uploaded points file.")
                st.stop()
            except ValueError as e:
                st.error(f"Could not convert selected columns in points file to numeric: {e}")
                st.stop()
            except Exception as e:
                st.error(f"Error processing uploaded points file columns: {e}")
                st.stop()
        else:
            st.error("Points file was uploaded but columns are not correctly specified or file not read.")
            st.stop()

    # --- Run Interpolation Logic ---
    results_df = None
    error_occurred = False
    with st.spinner("Performing interpolation..."):
        try:
            # --- Get known points/values from main DataFrame ---
            if mode == '1d':
                x_known = pd.to_numeric(df[x_col], errors='coerce').values
                y_known = pd.to_numeric(df[y_col], errors='coerce').values
                nan_mask = np.isnan(x_known) | np.isnan(y_known)
                x_known = x_known[~nan_mask]
                y_known = y_known[~nan_mask]
                # Sort for interp1d
                sort_idx = np.argsort(x_known)
                x_known = x_known[sort_idx]
                y_known = y_known[sort_idx]
                # Handle duplicates for interp1d
                unique_x, unique_idx = np.unique(x_known, return_index=True)
                if len(unique_x) < len(x_known):
                    st.info(f"Note: Using first occurrence for {len(x_known) - len(unique_x)} duplicate X values.")
                    x_known = unique_x
                    y_known = y_known[unique_idx]
                if len(x_known) < 2: raise ValueError("Not enough unique valid data points.")

            else: # N-D
                points_known_df = df[coord_cols]
                values_known_series = df[value_col]
                points_known = pd.to_numeric(points_known_df.stack(), errors='coerce').unstack().values # Faster numeric conversion try
                values_known = pd.to_numeric(values_known_series, errors='coerce').values

                nan_mask = np.isnan(points_known).any(axis=1) | np.isnan(values_known)
                points_known = points_known[~nan_mask]
                values_known = values_known[~nan_mask]
                if len(points_known) == 0: raise ValueError("Not enough valid data points after removing NaNs.")


            # --- Generate or Get New Points Array ---
            if points_new_array is None: # If not loaded from file
                 if mode == '1d':
                     points_new_array = generate_points_1d_interp(settings, x_known)
                 else: # N-D
                     points_new_array = generate_points_nd_interp(settings, num_coords, coord_names_list)

            # --- Perform Actual Interpolation ---
            values_new = None
            if mode == '1d':
                # Simplified fill_value logic here (from original handler)
                fill_value_arg = None
                if fill_value_str == 'nan': fill_value_arg = np.nan
                elif fill_value_str == 'edge' and extrapolate: fill_value_arg = (y_known[0], y_known[-1])
                elif fill_value_str != 'edge': fill_value_arg = float(fill_value_str)

                interpolator = interp1d(
                    x_known, y_known, kind=method, bounds_error=not extrapolate,
                    fill_value=fill_value_arg, assume_sorted=True
                )
                values_new = interpolator(points_new_array) # points_new_array is 1D here

            elif mode == 'grid':
                 # Simplified Grid logic - assumes `handle_grid_interp` logic here
                 # Need to reconstruct grid_axes and values_grid
                 grid_axes = []
                 for i in range(num_coords): grid_axes.append(np.unique(points_known[:, i]))
                 expected_points = np.prod([len(ax) for ax in grid_axes])
                 if expected_points != len(values_known): raise ValueError("Data size mismatch for regular grid.")
                 # Simplified reshape (assumes data is perfectly ordered) - Robust version needed
                 try:
                    multi_index = pd.MultiIndex.from_product(grid_axes)
                    mapping_df = pd.DataFrame(values_known, index=pd.MultiIndex.from_arrays(points_known.T)).reindex(multi_index)
                    values_grid = mapping_df[0].values.reshape([len(ax) for ax in grid_axes])
                 except Exception: raise ValueError("Could not reshape data onto grid - ensure input is perfectly ordered.")
                 if np.isnan(values_grid).any(): raise ValueError("NaNs found after reshaping - grid likely incomplete.")

                 fill_value_arg = None
                 if fill_value_str == 'nan': fill_value_arg = np.nan
                 else: fill_value_arg = float(fill_value_str)

                 interpolator = RegularGridInterpolator(tuple(grid_axes), values_grid, method=method, bounds_error=not extrapolate, fill_value=fill_value_arg)
                 values_new = interpolator(points_new_array) # points_new_array is N-D here

            elif mode == 'scattered':
                 fill_value_arg = np.nan if fill_value_str == 'nan' else float(fill_value_str)
                 if method in ['linear', 'cubic', 'nearest']:
                      # Check degeneracy for triangulation methods
                      if method != 'nearest' and len(points_known) <= num_coords:
                          raise ValueError(f"Need > {num_coords} points for {method} triangulation.")
                      try:
                          if method != 'nearest': _ = Delaunay(points_known)
                      except Exception as qe: raise ValueError(f"Input points degenerate/insufficient for {method}. QHull error: {qe}")

                      values_new = griddata(points_known, values_known, points_new_array, method=method, fill_value=fill_value_arg)
                 elif method == 'rbf':
                      coord_args_known = [points_known[:, i] for i in range(num_coords)]
                      coord_args_new = [points_new_array[:, i] for i in range(num_coords)]
                      # Pass RBF params correctly
                      rbf_options = {'function': rbf_params.get('kernel', 'multiquadric')}
                      if rbf_params.get('epsilon') is not None: rbf_options['epsilon'] = rbf_params['epsilon']
                      if rbf_params.get('smooth') is not None: rbf_options['smooth'] = rbf_params['smooth']

                      interpolator = Rbf(*coord_args_known, values_known, **rbf_options)
                      values_new = interpolator(*coord_args_new)
                      # Manual bounds check if extrapolate is False
                      if not extrapolate and len(points_known) > num_coords:
                          try:
                              hull = Delaunay(points_known)
                              in_hull = hull.find_simplex(points_new_array) >= 0
                              values_new[~in_hull] = fill_value_arg
                          except Exception as hull_e:
                               st.warning(f"Could not perform convex hull check for RBF bounds: {hull_e}")

            # --- Combine results into DataFrame ---
            if values_new is not None:
                if points_new_array.ndim == 1: points_new_array = points_new_array.reshape(-1, 1)
                output_data = np.hstack((points_new_array, values_new.reshape(-1, 1)))
                headers = [f"{h}_interp" for h in coord_names_list] + [f"{settings['value_col_name']}_interp"]
                results_df = pd.DataFrame(output_data, columns=headers)

        except ValueError as ve:
            st.error(f"Interpolation Value Error: {ve}")
            error_occurred = True
        except RuntimeError as re:
             st.error(f"Interpolation Runtime Error: {re}")
             error_occurred = True
        except Exception as e:
            st.error(f"An unexpected error occurred during interpolation: {e}")
            error_occurred = True

    # --- Display Results & Download ---
    if results_df is not None and not error_occurred:
        st.success("Interpolation completed successfully!")
        st.write("Preview of Interpolated Results:")
        st.dataframe(results_df.head(), use_container_width=True)

        # Download Button
        csv_results = results_df.to_csv(index=False).encode('utf-8')
        default_interp_filename = f"interpolated_output_{mode}.csv"
        st.download_button(
            label="Download Interpolation Results as CSV",
            data=csv_results,
            file_name=default_interp_filename,
            mime='text/csv',
        )
    elif not error_occurred:
         st.warning("Interpolation did not produce results (check inputs and method compatibility).")

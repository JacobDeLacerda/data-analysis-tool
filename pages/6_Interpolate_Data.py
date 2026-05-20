import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, griddata, Rbf
from utils import require_data, sidebar_status

st.set_page_config(layout="wide")
sidebar_status()
require_data()

st.header("6. Interpolate Data")

df = st.session_state.df


# ── Helper functions ──────────────────────────────────────────────────────────

@st.cache_data
def build_1d_interpolator(_df, x_col, y_col, method, extrapolate, fill_str):
    """Prepare data and return a SciPy interp1d object plus (x_min, x_max)."""
    x = pd.to_numeric(_df[x_col], errors='coerce').values
    y = pd.to_numeric(_df[y_col], errors='coerce').values

    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]

    sort_idx = np.argsort(x)
    x, y = x[sort_idx], y[sort_idx]
    _, uniq = np.unique(x, return_index=True)
    x, y = x[uniq], y[uniq]

    if len(x) < 2:
        raise ValueError("Not enough valid data points for interpolation.")

    min_pts = {'cubic': 4, 'quadratic': 3}.get(method, 2)
    if len(x) < min_pts:
        raise ValueError(f"'{method}' requires at least {min_pts} unique points; found {len(x)}.")

    if fill_str == 'nan':
        fill_arg = np.nan
    elif fill_str == 'edge':
        fill_arg = (y[0], y[-1]) if extrapolate else np.nan
    else:
        fill_arg = float(fill_str)

    f = interp1d(x, y, kind=method, bounds_error=not extrapolate,
                 fill_value=fill_arg, assume_sorted=True)
    return f, float(x.min()), float(x.max())


def _nd_known_points(df, coord_cols, value_col):
    pts = df[coord_cols].apply(pd.to_numeric, errors='coerce').values
    vals = pd.to_numeric(df[value_col], errors='coerce').values
    valid = ~(np.isnan(pts).any(axis=1) | np.isnan(vals))
    return pts[valid], vals[valid]


def _generate_1d_points(source, x_min, x_max, num_pts, step):
    if source == 'num_points':
        return np.linspace(x_min, x_max, int(num_pts))
    return np.arange(x_min, x_max + step * 1e-9, step)


def _generate_nd_grid(grid_spec):
    """Parse 'min:max:n,...' and return (M, D) array of all grid points."""
    grids = []
    for spec in grid_spec.split(','):
        g_min, g_max, g_n = spec.split(':')
        grids.append(np.linspace(float(g_min), float(g_max), int(g_n)))
    mesh = np.meshgrid(*grids, indexing='ij')
    return np.column_stack([g.ravel() for g in mesh])


def _parse_fill(fill_str):
    if fill_str in ('nan', 'edge'):
        return np.nan
    try:
        return float(fill_str)
    except ValueError:
        return np.nan


# ── Workflow selection ────────────────────────────────────────────────────────

goal = st.radio(
    "Goal:",
    ["Generate points (download CSV)", "Evaluate single point (1D)"],
    horizontal=True,
)
st.divider()

# ── Goal 1: Generate & download ───────────────────────────────────────────────
if goal == "Generate points (download CSV)":

    st.subheader("1. Input data")
    mode_label = st.radio(
        "Mode:", ["1D — y = f(x)", "N-D Scattered", "N-D Grid"], horizontal=True
    )
    mode = '1d' if '1D' in mode_label else ('scattered' if 'Scattered' in mode_label else 'grid')

    cols = df.columns.tolist()
    x_col = y_col = coord_cols = value_col = None
    coord_names = []

    with st.container(border=True):
        if mode == '1d':
            c1, c2 = st.columns(2)
            x_col = c1.selectbox("X column (independent):", cols, index=None,
                                  placeholder="Select…", key="x_down")
            y_col = c2.selectbox("Y column (dependent):", cols, index=None,
                                  placeholder="Select…", key="y_down")
            coord_names = [x_col] if x_col else []
        else:
            coord_cols = st.multiselect("Coordinate columns (order matters):", cols, key="coord_down")
            value_col = st.selectbox("Value column:", cols, index=None,
                                      placeholder="Select…", key="val_down")
            coord_names = coord_cols or []

    input_ok = (mode == '1d' and x_col and y_col) or (mode != '1d' and coord_cols and value_col)
    if not input_ok:
        st.info("Select the required input columns above.")
        st.stop()

    st.subheader("2. Method")
    rbf_params = {}
    with st.container(border=True):
        if mode == '1d':
            method = st.selectbox("Method:", ['linear', 'cubic', 'quadratic', 'slinear', 'nearest', 'zero'], key="m_1d")
        elif mode == 'grid':
            method = st.selectbox("Method:", ['linear', 'nearest'], key="m_grid")
            st.caption("N-D Grid mode uses griddata; 'linear' recommended for most datasets.")
        else:
            method = st.selectbox("Method:", ['linear', 'cubic', 'nearest', 'rbf'], key="m_scat")
            if method == 'rbf':
                with st.expander("RBF options"):
                    rbf_params['function'] = st.selectbox(
                        "Kernel:",
                        ['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate'],
                    )
                    rbf_params['epsilon'] = st.number_input("Epsilon (> 0):", min_value=1e-9, value=1.0, format="%f")
                    rbf_params['smooth'] = st.number_input("Smooth (0 = exact fit):", min_value=0.0, value=0.0, format="%f")

        with st.expander("Extrapolation & fill"):
            extrapolate = st.checkbox("Allow extrapolation", value=False, key="extrap_down")
            fill_opts = ['nan']
            if mode == '1d' and extrapolate:
                fill_opts.append('edge')
            fill_opts.append('number')
            fill_choice = st.selectbox("Fill value for out-of-range points:", fill_opts, key="fill_down")
            fill_str = fill_choice
            if fill_choice == 'number':
                fill_str = str(st.number_input("Numeric fill value:", value=0.0, format="%f", key="fill_num_down"))

    st.subheader("3. Output points")
    n_coords = 1 if mode == '1d' else len(coord_cols)
    src_options = ["Upload file"]
    if mode == '1d':
        src_options += ["Evenly spaced", "Fixed step size"]
    else:
        src_options.append("Define grid")

    num_pts = step_val = grid_spec = None
    upload_obj = None
    pts_file_cols = None
    pts_delimiter = ','
    pts_header = 0

    with st.container(border=True):
        source = st.selectbox("Point source:", src_options, key="src_down")

        if source == "Upload file":
            upload_obj = st.file_uploader("Points file (CSV/Excel/Txt)",
                                           type=['csv', 'xlsx', 'xls', 'txt'], key="pts_upload")
            if upload_obj:
                c1, c2 = st.columns(2)
                pts_delimiter = c1.text_input("Delimiter", value=',', key="pts_delim")
                if pts_delimiter == '\\t':
                    pts_delimiter = '\t'
                pts_header = c2.number_input("Header row", min_value=0, value=0, step=1, key="pts_hdr")
                try:
                    preview = pd.read_csv(upload_obj, delimiter=pts_delimiter,
                                          header=pts_header, skipinitialspace=True, nrows=5)
                    st.dataframe(preview, use_container_width=True)
                    file_cols = preview.columns.tolist()
                    if mode == '1d':
                        pts_file_cols = st.selectbox(f"Column for X ({x_col}):", file_cols, key="pts_xcol")
                    else:
                        sel = st.multiselect(
                            f"Select {n_coords} coordinate column(s):", file_cols,
                            default=file_cols[:n_coords], key="pts_ndcols",
                        )
                        pts_file_cols = sel if len(sel) == n_coords else None
                        if len(sel) != n_coords:
                            st.warning(f"Select exactly {n_coords} column(s).")
                    upload_obj.seek(0)
                except Exception as e:
                    st.error(f"Error reading file: {e}")

        elif source == "Evenly spaced":
            num_pts = st.number_input("Number of points:", min_value=2, value=100, step=10, key="npts")

        elif source == "Fixed step size":
            step_val = st.number_input("Step size:", min_value=1e-9, value=1.0, format="%f", key="step")

        elif source == "Define grid":
            st.write(f"Grid for {n_coords} dimension(s) ({', '.join(coord_names)}):")
            specs, valid = [], True
            for i in range(n_coords):
                gc1, gc2, gc3 = st.columns(3)
                g_min = gc1.number_input(f"Min (dim {i+1})", value=0.0, format="%f", key=f"gmin_{i}")
                g_max = gc2.number_input(f"Max (dim {i+1})", value=1.0, format="%f", key=f"gmax_{i}")
                g_n = gc3.number_input(f"Points (dim {i+1})", min_value=2, value=20, step=1, key=f"gn_{i}")
                if g_max <= g_min:
                    gc2.warning("Max must be > Min")
                    valid = False
                specs.append(f"{g_min}:{g_max}:{g_n}")
            grid_spec = ','.join(specs) if valid else None

    st.subheader("4. Run")
    run_disabled = (
        not method
        or (source == "Upload file" and (not upload_obj or not pts_file_cols))
        or (source == "Define grid" and not grid_spec)
    )

    if st.button("Run Interpolation & Download", type="primary", disabled=run_disabled):
        with st.spinner("Interpolating…"):
            try:
                if mode == '1d':
                    f, x_min, x_max = build_1d_interpolator(
                        df, x_col, y_col, method, extrapolate, fill_str
                    )
                    if source == "Upload file":
                        full = pd.read_csv(upload_obj, delimiter=pts_delimiter,
                                           header=pts_header, skipinitialspace=True)
                        x_new = pd.to_numeric(full[pts_file_cols], errors='coerce').values
                    else:
                        x_new = _generate_1d_points(
                            'num_points' if source == "Evenly spaced" else 'step',
                            x_min, x_max, num_pts, step_val,
                        )
                    y_new = f(x_new)
                    result_df = pd.DataFrame({f"{x_col}_interp": x_new, f"{y_col}_interp": y_new})

                else:  # N-D scattered or grid
                    pts_known, vals_known = _nd_known_points(df, coord_cols, value_col)
                    if len(pts_known) == 0:
                        raise ValueError("No valid data points after removing NaN rows.")

                    if source == "Upload file":
                        full = pd.read_csv(upload_obj, delimiter=pts_delimiter,
                                           header=pts_header, skipinitialspace=True)
                        pts_new = full[pts_file_cols].apply(pd.to_numeric, errors='coerce').values
                    else:
                        pts_new = _generate_nd_grid(grid_spec)

                    fill_num = _parse_fill(fill_str)

                    if method == 'rbf':
                        rbf_fn = Rbf(*pts_known.T, vals_known, **rbf_params)
                        vals_new = rbf_fn(*pts_new.T)
                    else:
                        vals_new = griddata(pts_known, vals_known, pts_new,
                                            method=method, fill_value=fill_num)

                    coord_hdrs = [f"{c}_interp" for c in coord_names]
                    result_df = pd.DataFrame(pts_new, columns=coord_hdrs)
                    result_df[f"{value_col}_interp"] = vals_new

                st.success(f"Done — {len(result_df):,} interpolated points.")
                st.dataframe(result_df.head(), use_container_width=True)
                st.download_button(
                    "Download CSV",
                    data=result_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"interpolated_{mode}.csv",
                    mime='text/csv',
                    type="primary",
                )

            except ValueError as e:
                st.error(f"Interpolation error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")


# ── Goal 2: Evaluate single point (1D) ───────────────────────────────────────
else:
    st.subheader("Evaluate Y for a single X value (1D)")

    cols = df.columns.tolist()
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        x_col = c1.selectbox("X column:", cols, index=None, placeholder="Select…", key="x_eval")
        y_col = c2.selectbox("Y column:", cols, index=None, placeholder="Select…", key="y_eval")
        method = c3.selectbox("Method:", ['linear', 'cubic', 'quadratic', 'slinear', 'nearest', 'zero'], key="m_eval")

    if x_col and y_col:
        c1, c2 = st.columns(2)
        extrapolate = c1.checkbox("Allow extrapolation", value=False, key="extrap_eval")
        x_val = c2.number_input("X value:", value=None, format="%f",
                                  placeholder="Enter X…", key="xval_eval")
        if x_val is not None:
            if st.button("Predict Y", type="primary", key="predict_btn"):
                with st.spinner("Calculating…"):
                    try:
                        f, x_min, x_max = build_1d_interpolator(
                            df, x_col, y_col, method, extrapolate, 'nan'
                        )
                        if not extrapolate and (x_val < x_min or x_val > x_max):
                            st.error(
                                f"X={x_val} is outside the data range "
                                f"[{x_min:.4g}, {x_max:.4g}]. "
                                "Enable extrapolation to predict outside this range."
                            )
                        else:
                            st.metric(f"Predicted {y_col}", f"{float(f(x_val)):.4f}")
                    except ValueError as e:
                        st.error(f"Error: {e}")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")
        else:
            st.info("Enter an X value to predict.")
    else:
        st.info("Select X and Y columns.")

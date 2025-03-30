# pages/3_Clean_Data.py
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

st.header("3. Clean Data")
st.caption("Handle missing values and duplicate rows.")

# --- Check if data is loaded ---
if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("No data loaded. Please go to the 'Load Data' page first.")
    st.stop()

df = st.session_state.df # Get the DataFrame
df_shape_before = df.shape # Store shape for comparison

# --- Display current missing values summary ---
nan_counts_before = df.isna().sum()
nan_cols_before = nan_counts_before[nan_counts_before > 0]
if not nan_cols_before.empty:
    st.info("Summary of columns with missing values currently:")
    st.dataframe(nan_cols_before.reset_index().rename(columns={'index':'Column', 0:'Missing Count'}), use_container_width=True)
else:
    st.success("No missing values detected.")

st.divider()

# --- Cleaning Options ---
st.subheader("Select Cleaning Action")

col1, col2 = st.columns(2)

with col1:
    cleaning_action = st.radio(
        "Choose Action:",
        ["Handle Missing Values", "Remove Duplicate Rows", "Drop Columns with High Missing %"],
        help="Select how you want to clean the data."
    )

# --- Action Specific Controls ---

if cleaning_action == "Handle Missing Values":
    st.markdown("**Handle Missing Values (NaNs/NaTs)**")
    nan_handling_method = st.radio(
        "Method:",
        ["Drop Rows with missing values", "Fill missing values"]
    )

    if nan_handling_method == "Drop Rows with missing values":
        cols_to_check_nan = st.multiselect(
            "Check for missing values only in these columns (optional):",
            df.columns.tolist(),
            help="If left empty, rows with a missing value in *any* column will be dropped."
        )
        if st.button("Apply Drop Rows", type="primary"):
            df_cleaned = df.dropna(subset=cols_to_check_nan if cols_to_check_nan else None)
            rows_dropped = len(df) - len(df_cleaned)
            st.session_state.df = df_cleaned
            st.session_state.df_modified = True
            st.success(f"Applied: Dropped {rows_dropped} rows.")
            st.rerun() # Rerun to update the display

    elif nan_handling_method == "Fill missing values":
        cols_to_fill = st.multiselect(
            "Select columns to fill missing values in:",
            df.columns.tolist(),
            help="Choose one or more columns where you want to fill missing values."
        )
        if cols_to_fill:
            fill_strategy = st.selectbox(
                "Fill Strategy:",
                ["Specific Value", "Mean", "Median", "Mode", "Forward Fill (ffill)", "Backward Fill (bfill)"]
            )

            fill_value_input = None
            limit_fill = None
            if fill_strategy == "Specific Value":
                fill_value_input = st.text_input("Value to fill with:", help="Enter the value (numeric or text). Type compatibility will be attempted.")
            elif fill_strategy in ["Forward Fill (ffill)", "Backward Fill (bfill)"]:
                 limit_fill = st.number_input("Limit consecutive fills (optional, 0 for no limit)", min_value=0, value=0, step=1)
                 if limit_fill == 0: limit_fill = None # Use None for no limit

            if st.button("Apply Fill Values", type="primary"):
                df_cleaned = df.copy() # Work on a copy
                skipped_cols = []
                filled_count_total = 0

                for col in cols_to_fill:
                    original_na = df_cleaned[col].isna().sum()
                    if original_na == 0:
                        skipped_cols.append(f"{col} (no NaNs)")
                        continue # Skip if no NaNs in this column

                    try:
                        if fill_strategy == "Specific Value":
                            if fill_value_input is not None:
                                # Try converting fill value based on column type
                                try:
                                    col_dtype = df_cleaned[col].dtype
                                    if pd.api.types.is_numeric_dtype(col_dtype):
                                        fill_val = float(fill_value_input)
                                    elif pd.api.types.is_datetime64_any_dtype(col_dtype):
                                        fill_val = pd.to_datetime(fill_value_input)
                                    else: # Assume string/object
                                        fill_val = str(fill_value_input)
                                    df_cleaned[col].fillna(value=fill_val, inplace=True)
                                except (ValueError, TypeError):
                                     st.warning(f"Could not convert '{fill_value_input}' for column '{col}'. Filling with string representation.", icon="⚠️")
                                     df_cleaned[col].fillna(value=str(fill_value_input), inplace=True)
                            else:
                                st.warning("Please enter a specific value to fill with.")
                                skipped_cols.append(f"{col} (no value entered)")

                        elif fill_strategy == "Mean":
                            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                                mean_val = df_cleaned[col].mean()
                                df_cleaned[col].fillna(value=mean_val, inplace=True)
                            else: skipped_cols.append(f"{col} (not numeric)")
                        elif fill_strategy == "Median":
                             if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                                median_val = df_cleaned[col].median()
                                df_cleaned[col].fillna(value=median_val, inplace=True)
                             else: skipped_cols.append(f"{col} (not numeric)")
                        elif fill_strategy == "Mode":
                            mode_val = df_cleaned[col].mode()
                            if not mode_val.empty:
                                df_cleaned[col].fillna(value=mode_val[0], inplace=True)
                            else: skipped_cols.append(f"{col} (no mode)")
                        elif fill_strategy == "Forward Fill (ffill)":
                             df_cleaned[col].ffill(limit=limit_fill, inplace=True)
                        elif fill_strategy == "Backward Fill (bfill)":
                             df_cleaned[col].bfill(limit=limit_fill, inplace=True)

                        filled_count_col = original_na - df_cleaned[col].isna().sum()
                        if filled_count_col > 0: filled_count_total += filled_count_col

                    except Exception as e:
                        skipped_cols.append(f"{col} (Error: {e})")

                st.session_state.df = df_cleaned
                st.session_state.df_modified = True
                st.success(f"Applied Fill: Filled {filled_count_total} missing values.")
                if skipped_cols:
                    st.warning(f"Skipped/Issues with columns: {', '.join(skipped_cols)}")
                st.rerun() # Update display
        else:
            st.info("Select columns to fill.")


elif cleaning_action == "Remove Duplicate Rows":
    st.markdown("**Remove Duplicate Rows**")
    cols_to_check_duplicates = st.multiselect(
        "Check duplicates based on these columns only (optional):",
        df.columns.tolist(),
        help="If empty, rows are considered duplicates only if *all* columns match."
    )
    subset_arg = cols_to_check_duplicates if cols_to_check_duplicates else None

    num_duplicates = df.duplicated(subset=subset_arg).sum()

    if num_duplicates == 0:
        st.info("No duplicate rows found" + (f" based on selected columns." if subset_arg else "."))
    else:
        st.warning(f"Found {num_duplicates} duplicate rows" + (f" based on selected columns." if subset_arg else "."))

        keep_choice = st.radio("Which occurrence to keep?", ["first", "last"], index=0)

        if st.button("Apply Remove Duplicates", type="primary"):
            df_cleaned = df.drop_duplicates(subset=subset_arg, keep=keep_choice)
            rows_dropped = len(df) - len(df_cleaned)
            st.session_state.df = df_cleaned
            st.session_state.df_modified = True
            st.success(f"Applied: Removed {rows_dropped} duplicate rows (kept '{keep_choice}').")
            st.rerun() # Update display

elif cleaning_action == "Drop Columns with High Missing %":
    st.markdown("**Drop Columns with High Missing %**")
    threshold_percent = st.slider(
        "Drop column if MORE than this percentage of values are missing:",
        min_value=0, max_value=100, value=50, step=5,
        format="%d%%",
        help="Columns exceeding this threshold will be dropped."
    )
    min_non_nan_perc = 100.0 - threshold_percent
    min_non_nan_count = int(len(df) * (min_non_nan_perc / 100.0))

    cols_to_drop_auto = df.columns[df.isna().sum() > (len(df) - min_non_nan_count)]

    if not cols_to_drop_auto.empty:
        st.warning(f"Columns to be dropped ({len(cols_to_drop_auto)}): {', '.join(cols_to_drop_auto)}")
        if st.button(f"Apply Drop Columns (> {threshold_percent}% missing)", type="primary"):
            df_cleaned = df.dropna(axis=1, thresh=min_non_nan_count)
            cols_actually_dropped = list(set(df.columns) - set(df_cleaned.columns))
            st.session_state.df = df_cleaned
            st.session_state.df_modified = True
            st.success(f"Applied: Dropped {len(cols_actually_dropped)} columns.")
            st.rerun()
    else:
        st.info(f"No columns found with more than {threshold_percent}% missing values.")


st.divider()
st.subheader("Current Data Preview (First 5 Rows)")
st.dataframe(st.session_state.df.head(), use_container_width=True)
st.caption(f"Shape after last action: {st.session_state.df.shape}")

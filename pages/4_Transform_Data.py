# pages/4_Transform_Data.py
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

st.header("4. Transform Data")
st.caption("Filter rows, select/drop/rename columns, sort, calculate, change types.")

# --- Check if data is loaded ---
if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("No data loaded. Please go to the 'Load Data' page first.")
    st.stop()

df = st.session_state.df # Get the DataFrame

# --- Transformation Options ---
st.subheader("Select Transformation Action")

transform_action = st.selectbox(
    "Choose Action:",
    [
        "Filter Rows (by condition/query)",
        "Select Columns (Keep)",
        "Drop Columns",
        "Rename Column",
        "Add Column (Basic Calculation)",
        "Sort Data",
        "Change Column Type"
    ],
    index=None, # Require selection
    placeholder="Select transformation..."
)

st.divider()

# --- Action Specific Controls ---

if transform_action == "Filter Rows (by condition/query)":
    st.markdown("**Filter Rows using Pandas Query**")
    st.markdown("""
    Enter a condition using column names, operators (`==`, `!=`, `>`, `<`, `>=`, `<=`), and logicals (`&`, `|`, `~`).
    - String values need quotes: `Country == 'Canada'`
    - Column names with spaces need backticks: `` `Column Name` > 10 ``
    - Check for nulls: `Value.isnull()` or `Value.notnull()`
    - Check if in list: `Category.isin(['A', 'B'])`
    """)
    st.code("Example: (`Temperature` > 25.0) & (`Status` == 'Active') | `Value`.isnull()")

    # Display available columns for reference
    st.caption(f"Available columns: {', '.join(df.columns)}")

    query_string = st.text_area("Enter Query Condition:", height=100)

    if st.button("Apply Filter", type="primary", disabled=not query_string):
        try:
            df_filtered = df.query(query_string, engine='python') # Python engine often more flexible
            rows_before = len(df)
            rows_after = len(df_filtered)
            st.session_state.df = df_filtered
            st.session_state.df_modified = True
            st.success(f"Applied Filter: Kept {rows_after} out of {rows_before} rows.")
            if rows_after == 0:
                st.warning("Filter resulted in an empty dataset!")
            st.rerun()
        except Exception as e:
            st.error(f"Error applying query '{query_string}': {e}")
            st.error("Check syntax, column names (use backticks `` if needed), quotes, and data types.")

elif transform_action == "Select Columns (Keep)":
    st.markdown("**Select Columns to Keep**")
    cols_to_keep = st.multiselect(
        "Choose columns to keep (others will be dropped):",
        df.columns.tolist(),
        help="Select one or more columns."
    )
    if st.button("Apply Column Selection", type="primary", disabled=not cols_to_keep):
        try:
            df_transformed = df[cols_to_keep]
            st.session_state.df = df_transformed
            st.session_state.df_modified = True
            st.success(f"Applied: Kept {len(cols_to_keep)} columns.")
            st.rerun()
        except KeyError as e:
            st.error(f"Error: Column '{e}' not found. This shouldn't happen with multiselect.")
        except Exception as e:
            st.error(f"Error selecting columns: {e}")

elif transform_action == "Drop Columns":
    st.markdown("**Drop Columns**")
    cols_to_drop = st.multiselect(
        "Choose columns to drop:",
        df.columns.tolist(),
        help="Select one or more columns to remove."
    )
    if st.button("Apply Drop Columns", type="primary", disabled=not cols_to_drop):
         try:
            df_transformed = df.drop(columns=cols_to_drop)
            st.session_state.df = df_transformed
            st.session_state.df_modified = True
            st.success(f"Applied: Dropped {len(cols_to_drop)} columns: {', '.join(cols_to_drop)}.")
            st.rerun()
         except KeyError as e:
             st.error(f"Error: Column '{e}' not found during drop operation.")
         except Exception as e:
            st.error(f"Error dropping columns: {e}")

elif transform_action == "Rename Column":
    st.markdown("**Rename Column**")
    col_old = st.selectbox(
        "Select column to rename:",
        [""] + df.columns.tolist(), # Add empty option
        index=0,
        help="Choose the column you want to give a new name."
    )
    if col_old:
        col_new = st.text_input(
            f"Enter new name for '{col_old}':",
            help="Provide the desired new column name."
        )
        if st.button("Apply Rename", type="primary", disabled=not col_new or col_new == col_old):
            if col_new in df.columns:
                st.error(f"Error: Column name '{col_new}' already exists.")
            else:
                try:
                    df_transformed = df.rename(columns={col_old: col_new})
                    st.session_state.df = df_transformed
                    st.session_state.df_modified = True
                    st.success(f"Applied: Renamed '{col_old}' to '{col_new}'.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error renaming column: {e}")

elif transform_action == "Add Column (Basic Calculation)":
    st.markdown("**Add Column (Basic Calculation)**")
    numeric_cols = df.select_dtypes(include=pd.np.number).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Need at least two numeric columns to perform basic calculations.")
    else:
        new_col_name = st.text_input("Enter name for the new column:", help="Choose a unique name for the result column.")
        op_choice = st.selectbox("Select operation:", ["Add (+)", "Subtract (-)", "Multiply (*)", "Divide (/)"])
        col1 = st.selectbox("Select FIRST numeric column:", numeric_cols, index=0)
        col2 = st.selectbox("Select SECOND numeric column:", numeric_cols, index=1)

        overwrite = False
        if new_col_name in df.columns:
             overwrite = st.checkbox(f"Column '{new_col_name}' exists. Overwrite?", value=False)

        if st.button("Apply Calculation", type="primary", disabled=not new_col_name or (new_col_name in df.columns and not overwrite)):
            try:
                 num_col1 = pd.to_numeric(df[col1], errors='coerce')
                 num_col2 = pd.to_numeric(df[col2], errors='coerce')
                 result = None
                 op_symbol = op_choice.split()[1][1] # Get '+', '-', '*', '/'

                 if op_choice == "Add (+)": result = num_col1 + num_col2
                 elif op_choice == "Subtract (-)": result = num_col1 - num_col2
                 elif op_choice == "Multiply (*)": result = num_col1 * num_col2
                 elif op_choice == "Divide (/)":
                      with np.errstate(divide='ignore', invalid='ignore'): result = num_col1 / num_col2
                      result.replace([np.inf, -np.inf], np.nan, inplace=True)

                 df_transformed = df.copy()
                 df_transformed[new_col_name] = result
                 st.session_state.df = df_transformed
                 st.session_state.df_modified = True
                 st.success(f"Applied: {'Updated' if overwrite else 'Added'} column '{new_col_name}' ({col1} {op_symbol} {col2}).")
                 if result.isna().sum() > num_col1.isna().sum() + num_col2.isna().sum():
                      st.warning("NaNs were introduced, possibly due to non-numeric inputs or division by zero.")
                 st.rerun()

            except KeyError:
                 st.error("Selected column(s) not found.")
            except Exception as e:
                 st.error(f"Error performing calculation: {e}")

elif transform_action == "Sort Data":
    st.markdown("**Sort Data**")
    sort_cols = st.multiselect(
        "Select column(s) to sort by (order matters):",
        df.columns.tolist(),
        help="Choose one or more columns. Sorting precedence is based on selection order."
    )
    if sort_cols:
        ascending_list = []
        st.write("Select sort order for each column:")
        for col in sort_cols:
             is_ascending = st.radio(f"Sort '{col}':", ["Ascending", "Descending"], index=0, horizontal=True, key=f"sort_{col}")
             ascending_list.append(is_ascending == "Ascending")

        na_pos = st.radio("Place missing values:", ["first", "last"], index=1, horizontal=True, help="Where should NaNs/NaTs appear?")

        if st.button("Apply Sort", type="primary"):
            try:
                df_sorted = df.sort_values(by=sort_cols, ascending=ascending_list, na_position=na_pos)
                st.session_state.df = df_sorted
                st.session_state.df_modified = True # Sorting is arguably a modification
                st.success(f"Applied: Data sorted by {', '.join(sort_cols)}.")
                st.rerun()
            except Exception as e:
                st.error(f"Error sorting data: {e}")

elif transform_action == "Change Column Type":
    st.markdown("**Change Column Type**")
    col_to_change = st.selectbox(
        "Select column to change type:",
        [""] + df.columns.tolist(), index=0,
        help="Choose the column whose data type you want to modify."
    )
    if col_to_change:
        st.write(f"Current type of '{col_to_change}': **{df[col_to_change].dtype}**")
        type_options = ["numeric (float/int)", "string (object)", "datetime", "boolean"]
        new_type = st.selectbox(
            "Select new data type:",
            type_options,
            help="Choose the target data type. Conversion errors will result in missing values (NaN/NaT/NA)."
        )
        if st.button("Apply Type Change", type="primary"):
            try:
                original_series = df[col_to_change]
                converted_series = None
                target_dtype_str = "Unknown"

                if new_type == "numeric (float/int)":
                    converted_series = pd.to_numeric(original_series, errors='coerce')
                    if converted_series.notna().all() and (converted_series == converted_series.astype(int)).all():
                         converted_series = converted_series.astype(int)
                    target_dtype_str = str(converted_series.dtype)
                elif new_type == "string (object)":
                    converted_series = original_series.astype(str)
                    target_dtype_str = "object/string"
                elif new_type == "datetime":
                    converted_series = pd.to_datetime(original_series, errors='coerce', infer_datetime_format=True)
                    target_dtype_str = str(converted_series.dtype)
                elif new_type == "boolean":
                     bool_map_true = {'true', '1', 'yes', 't', 'y'}
                     bool_map_false = {'false', '0', 'no', 'f', 'n'}
                     def to_bool_safe(x):
                         if pd.isna(x): return pd.NA
                         s = str(x).lower().strip()
                         if s in bool_map_true: return True
                         if s in bool_map_false: return False
                         return pd.NA
                     converted_series = original_series.apply(to_bool_safe).astype('boolean')
                     target_dtype_str = "boolean (nullable)"

                df_transformed = df.copy()
                df_transformed[col_to_change] = converted_series
                st.session_state.df = df_transformed
                st.session_state.df_modified = True
                st.success(f"Applied: Changed '{col_to_change}' type to {target_dtype_str}.")

                original_na = original_series.isna().sum()
                converted_na = converted_series.isna().sum()
                if converted_na > original_na:
                     st.warning(f"{converted_na - original_na} values could not be converted and became missing (NaN/NaT/NA).")
                st.rerun()

            except Exception as e:
                 st.error(f"Error changing type of column '{col_to_change}' to {new_type}: {e}")


st.divider()
st.subheader("Current Data Preview (First 5 Rows)")
st.dataframe(st.session_state.df.head(), use_container_width=True)
st.caption(f"Shape after last action: {st.session_state.df.shape}")

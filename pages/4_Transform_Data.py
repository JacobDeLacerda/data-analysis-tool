import streamlit as st
import pandas as pd
import numpy as np
from utils import require_data, sidebar_status

st.set_page_config(layout="wide")
sidebar_status()
require_data()

st.header("4. Transform Data")

df = st.session_state.df

action = st.selectbox(
    "Action:",
    [
        "Filter Rows",
        "Keep Columns",
        "Drop Columns",
        "Rename Column",
        "Create Column",
        "Sort Data",
        "Cast Column Type",
    ],
    index=None,
    placeholder="Select a transformation…",
)

st.divider()

if action == "Filter Rows":
    st.markdown("""
    Filter using Pandas query syntax. Examples:
    - `Age > 30 & Status == 'Active'`
    - `` `Column Name` < 100 ``
    - `Value.isnull()` or `Score.isin([1, 2, 3])`
    """)
    st.caption(f"Available columns: {', '.join(df.columns)}")
    query = st.text_area("Condition:", height=80)
    if st.button("Apply Filter", type="primary", disabled=not query):
        try:
            filtered = df.query(query, engine='python')
            st.session_state.df = filtered
            st.session_state.df_modified = True
            st.success(f"Kept {len(filtered):,} of {len(df):,} rows.")
            if len(filtered) == 0:
                st.warning("Filter produced an empty dataset.")
            st.rerun()
        except Exception as e:
            st.error(f"Query error: {e}")

elif action == "Keep Columns":
    cols = st.multiselect("Columns to keep (all others will be dropped):", df.columns.tolist())
    if st.button("Apply", type="primary", disabled=not cols):
        st.session_state.df = df[cols]
        st.session_state.df_modified = True
        st.success(f"Kept {len(cols)} column(s).")
        st.rerun()

elif action == "Drop Columns":
    cols = st.multiselect("Columns to drop:", df.columns.tolist())
    if st.button("Apply", type="primary", disabled=not cols):
        st.session_state.df = df.drop(columns=cols)
        st.session_state.df_modified = True
        st.success(f"Dropped: {', '.join(cols)}.")
        st.rerun()

elif action == "Rename Column":
    col_old = st.selectbox("Column to rename:", [""] + df.columns.tolist(), index=0)
    if col_old:
        col_new = st.text_input(f"New name for '{col_old}':")
        if st.button("Apply Rename", type="primary", disabled=not col_new or col_new == col_old):
            if col_new in df.columns:
                st.error(f"Column '{col_new}' already exists.")
            else:
                st.session_state.df = df.rename(columns={col_old: col_new})
                st.session_state.df_modified = True
                st.success(f"Renamed '{col_old}' → '{col_new}'.")
                st.rerun()

elif action == "Create Column":
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Need at least two numeric columns.")
    else:
        new_name = st.text_input("New column name:")
        op = st.selectbox("Operation:", ["Add (+)", "Subtract (-)", "Multiply (*)", "Divide (/)"])
        c1 = st.selectbox("First column:", numeric_cols, index=0)
        c2 = st.selectbox("Second column:", numeric_cols, index=min(1, len(numeric_cols) - 1))

        overwrite = False
        if new_name and new_name in df.columns:
            overwrite = st.checkbox(f"'{new_name}' already exists — overwrite?")

        disabled = not new_name or (new_name in df.columns and not overwrite)
        if st.button("Create Column", type="primary", disabled=disabled):
            try:
                a = pd.to_numeric(df[c1], errors='coerce')
                b = pd.to_numeric(df[c2], errors='coerce')
                results = {
                    "Add (+)": a + b,
                    "Subtract (-)": a - b,
                    "Multiply (*)": a * b,
                    "Divide (/)": a / b.replace(0, np.nan),
                }
                result = results[op]
                transformed = df.copy()
                transformed[new_name] = result
                st.session_state.df = transformed
                st.session_state.df_modified = True
                label = "Updated" if overwrite else "Added"
                st.success(f"{label} column '{new_name}'.")
                if result.isna().sum() > a.isna().sum() + b.isna().sum():
                    st.warning("Some NaNs introduced (non-numeric input or division by zero).")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

elif action == "Sort Data":
    sort_cols = st.multiselect("Sort by (order matters):", df.columns.tolist())
    if sort_cols:
        ascending = []
        for col in sort_cols:
            direction = st.radio(f"'{col}':", ["Ascending", "Descending"], horizontal=True, key=f"sort_{col}")
            ascending.append(direction == "Ascending")
        na_pos = st.radio("Place NaNs:", ["last", "first"], horizontal=True)
        if st.button("Apply Sort", type="primary"):
            try:
                st.session_state.df = df.sort_values(by=sort_cols, ascending=ascending, na_position=na_pos)
                st.session_state.df_modified = True
                st.success(f"Sorted by {', '.join(sort_cols)}.")
                st.rerun()
            except Exception as e:
                st.error(f"Sort error: {e}")

elif action == "Cast Column Type":
    col = st.selectbox("Column to cast:", [""] + df.columns.tolist(), index=0)
    if col:
        st.write(f"Current type: **{df[col].dtype}**")
        new_type = st.selectbox("Target type:", ["numeric", "string", "datetime", "boolean"])
        if st.button("Apply Cast", type="primary"):
            try:
                original = df[col]
                if new_type == "numeric":
                    converted = pd.to_numeric(original, errors='coerce')
                    if converted.notna().all() and (converted % 1 == 0).all():
                        converted = converted.astype(int)
                elif new_type == "string":
                    converted = original.astype(str)
                elif new_type == "datetime":
                    converted = pd.to_datetime(original, errors='coerce')
                elif new_type == "boolean":
                    true_vals = {'true', '1', 'yes', 't', 'y'}
                    false_vals = {'false', '0', 'no', 'f', 'n'}

                    def to_bool(x):
                        if pd.isna(x):
                            return pd.NA
                        s = str(x).lower().strip()
                        return True if s in true_vals else (False if s in false_vals else pd.NA)

                    converted = original.apply(to_bool).astype('boolean')

                result = df.copy()
                result[col] = converted
                st.session_state.df = result
                st.session_state.df_modified = True
                st.success(f"Cast '{col}' to {converted.dtype}.")
                new_na = converted.isna().sum() - original.isna().sum()
                if new_na > 0:
                    st.warning(f"{new_na} value(s) could not be converted and became NaN/NA.")
                st.rerun()
            except Exception as e:
                st.error(f"Cast error: {e}")

st.divider()
st.caption(f"Shape: {st.session_state.df.shape}")
st.dataframe(st.session_state.df.head(), use_container_width=True)

import streamlit as st
import pandas as pd
from utils import require_data, sidebar_status

st.set_page_config(layout="wide")
sidebar_status()
require_data()

st.header("3. Clean Data")

df = st.session_state.df

missing = df.isna().sum()
missing = missing[missing > 0]
if not missing.empty:
    st.info("Columns with missing values:")
    st.dataframe(
        missing.rename_axis('Column').reset_index(name='Missing'),
        use_container_width=True,
    )
else:
    st.success("No missing values detected.")

st.divider()

action = st.radio(
    "Action:",
    ["Handle Missing Values", "Remove Duplicates", "Drop High-Missing Columns"],
    horizontal=True,
)

st.divider()

if action == "Handle Missing Values":
    method = st.radio("Method:", ["Drop rows", "Fill values"], horizontal=True)

    if method == "Drop rows":
        subset = st.multiselect(
            "Check only these columns (leave empty = check all):",
            df.columns.tolist(),
        )
        if st.button("Drop Rows", type="primary"):
            cleaned = df.dropna(subset=subset or None)
            st.session_state.df = cleaned
            st.session_state.df_modified = True
            st.success(f"Dropped {len(df) - len(cleaned):,} rows.")
            st.rerun()

    else:
        cols = st.multiselect("Columns to fill:", df.columns.tolist())
        if cols:
            strategy = st.selectbox(
                "Strategy:",
                ["Specific Value", "Mean", "Median", "Mode", "Forward Fill", "Backward Fill"],
            )
            fill_input = None
            limit = None
            if strategy == "Specific Value":
                fill_input = st.text_input("Fill with:")
            elif strategy in ("Forward Fill", "Backward Fill"):
                n = st.number_input("Max consecutive fills (0 = unlimited)", min_value=0, value=0, step=1)
                limit = n or None

            if st.button("Fill Values", type="primary"):
                cleaned = df.copy()
                skipped, filled_total = [], 0

                for col in cols:
                    orig_na = cleaned[col].isna().sum()
                    if orig_na == 0:
                        skipped.append(f"{col} (no missing)")
                        continue
                    try:
                        if strategy == "Specific Value":
                            if not fill_input:
                                skipped.append(f"{col} (no value entered)")
                                continue
                            dtype = cleaned[col].dtype
                            if pd.api.types.is_numeric_dtype(dtype):
                                val = float(fill_input)
                            elif pd.api.types.is_datetime64_any_dtype(dtype):
                                val = pd.to_datetime(fill_input)
                            else:
                                val = str(fill_input)
                            cleaned[col] = cleaned[col].fillna(val)
                        elif strategy == "Mean":
                            if pd.api.types.is_numeric_dtype(cleaned[col]):
                                cleaned[col] = cleaned[col].fillna(cleaned[col].mean())
                            else:
                                skipped.append(f"{col} (not numeric)")
                                continue
                        elif strategy == "Median":
                            if pd.api.types.is_numeric_dtype(cleaned[col]):
                                cleaned[col] = cleaned[col].fillna(cleaned[col].median())
                            else:
                                skipped.append(f"{col} (not numeric)")
                                continue
                        elif strategy == "Mode":
                            mode = cleaned[col].mode()
                            if not mode.empty:
                                cleaned[col] = cleaned[col].fillna(mode[0])
                            else:
                                skipped.append(f"{col} (no mode)")
                                continue
                        elif strategy == "Forward Fill":
                            cleaned[col] = cleaned[col].ffill(limit=limit)
                        elif strategy == "Backward Fill":
                            cleaned[col] = cleaned[col].bfill(limit=limit)

                        filled_total += orig_na - cleaned[col].isna().sum()
                    except Exception as e:
                        skipped.append(f"{col} ({e})")

                st.session_state.df = cleaned
                st.session_state.df_modified = True
                st.success(f"Filled {filled_total:,} missing values.")
                if skipped:
                    st.warning(f"Skipped: {', '.join(skipped)}")
                st.rerun()
        else:
            st.info("Select columns to fill.")

elif action == "Remove Duplicates":
    subset = st.multiselect(
        "Check only these columns (leave empty = check all):",
        df.columns.tolist(),
    )
    subset_arg = subset or None
    n_dups = df.duplicated(subset=subset_arg).sum()

    if n_dups == 0:
        st.info("No duplicate rows found.")
    else:
        st.warning(f"Found {n_dups:,} duplicate rows.")
        keep = st.radio("Keep:", ["first", "last"], horizontal=True)
        if st.button("Remove Duplicates", type="primary"):
            cleaned = df.drop_duplicates(subset=subset_arg, keep=keep)
            st.session_state.df = cleaned
            st.session_state.df_modified = True
            st.success(f"Removed {len(df) - len(cleaned):,} duplicate rows (kept '{keep}').")
            st.rerun()

elif action == "Drop High-Missing Columns":
    threshold = st.slider("Drop column if missing >", 0, 100, 50, 5, format="%d%%")
    to_drop = [c for c in df.columns if df[c].isna().mean() * 100 > threshold]

    if to_drop:
        st.warning(f"Columns to drop ({len(to_drop)}): {', '.join(to_drop)}")
        if st.button(f"Drop {len(to_drop)} Column(s)", type="primary"):
            st.session_state.df = df.drop(columns=to_drop)
            st.session_state.df_modified = True
            st.success(f"Dropped {len(to_drop)} column(s).")
            st.rerun()
    else:
        st.info(f"No columns exceed {threshold}% missing.")

st.divider()
st.caption(f"Shape: {st.session_state.df.shape}")
st.dataframe(st.session_state.df.head(), use_container_width=True)

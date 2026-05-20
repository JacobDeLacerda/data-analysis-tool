import streamlit as st
import pandas as pd
import numpy as np
import io
from utils import require_data, sidebar_status

st.set_page_config(layout="wide")
sidebar_status()
require_data()

st.header("2. Inspect Data")

df = st.session_state.df

OPTIONS = [
    "Head",
    "Tail",
    "Column Info",
    "Statistics",
    "Column Types",
    "Unique Values",
    "Value Counts",
    "Missing Values",
]

col_left, col_right = st.columns([1, 2])

with col_left:
    choice = st.radio("Inspect:", OPTIONS)

    n_rows = 5
    selected_col = None

    if choice in ("Head", "Tail"):
        n_rows = st.number_input("Rows to show", min_value=1, value=5, step=1)
    elif choice in ("Unique Values", "Value Counts"):
        selected_col = st.selectbox("Column", df.columns.tolist())
    elif choice == "Statistics":
        stats_scope = st.radio("Include:", ["Numeric only", "All columns"], index=0)

with col_right:
    st.subheader("Results")
    try:
        if choice == "Head":
            st.dataframe(df.head(n_rows), use_container_width=True)

        elif choice == "Tail":
            st.dataframe(df.tail(n_rows), use_container_width=True)

        elif choice == "Column Info":
            buf = io.StringIO()
            df.info(buf=buf, verbose=True, show_counts=True)
            st.text(buf.getvalue())

        elif choice == "Statistics":
            desc = df.describe(include='all') if stats_scope == "All columns" else df.describe()
            st.dataframe(desc, use_container_width=True)

        elif choice == "Column Types":
            st.dataframe(
                df.dtypes.rename_axis('Column').reset_index(name='Type'),
                use_container_width=True,
            )

        elif choice == "Unique Values":
            if selected_col:
                uniques = df[selected_col].unique()
                st.write(f"**{len(uniques)}** unique values in `{selected_col}`:")
                st.write(uniques[:1000])
                if len(uniques) > 1000:
                    st.caption("(showing first 1,000)")

        elif choice == "Value Counts":
            if selected_col:
                counts = (
                    df[selected_col]
                    .value_counts()
                    .rename_axis(selected_col)
                    .reset_index(name='count')
                )
                st.dataframe(counts, use_container_width=True)

        elif choice == "Missing Values":
            missing = df.isna().sum()
            missing = missing[missing > 0]
            if missing.empty:
                st.success("No missing values found.")
            else:
                summary = missing.rename_axis('Column').reset_index(name='Missing')
                summary['% of rows'] = (summary['Missing'] / len(df) * 100).round(2)
                st.dataframe(summary, use_container_width=True)
                st.metric(
                    "Total missing cells",
                    int(summary['Missing'].sum()),
                    delta=f"{summary['Missing'].sum() / df.size * 100:.2f}% of all cells",
                    delta_color="inverse",
                )

    except Exception as e:
        st.error(f"Error: {e}")

st.divider()
st.dataframe(df.head(), use_container_width=True)

# pages/2_Inspect_Data.py
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

st.header("2. Inspect Data")
st.caption("Explore the structure and content of your loaded data.")

# --- Check if data is loaded ---
if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("No data loaded. Please go to the 'Load Data' page first.")
    st.stop() # Stop execution if no data

df = st.session_state.df # Get the DataFrame

# --- Inspection Options ---
st.subheader("Select Inspection Action")

options = [
    "View Head (first N rows)",
    "View Tail (last N rows)",
    "Show Info (column types, non-null counts)",
    "Show Descriptive Statistics",
    "List Column Names and Types",
    "Show Unique Values in a Column",
    "Count Values in a Column",
    "Check for Missing Values (NaNs/NaTs)"
]

# Use columns for layout
col1, col2 = st.columns([1, 2]) # Give more space to the output

with col1:
    choice = st.radio(
        "Choose what to inspect:",
        options,
        help="Select an option to see details about your data."
    )

    # --- Action-specific inputs ---
    n_rows = 5 # Default for head/tail
    selected_column = None
    stats_include = 'Numeric only'

    if choice == "View Head (first N rows)" or choice == "View Tail (last N rows)":
        n_rows = st.number_input("Number of rows to show", min_value=1, value=5, step=1)
    elif choice == "Show Unique Values in a Column" or choice == "Count Values in a Column":
        if not df.columns.empty:
            selected_column = st.selectbox("Select Column", df.columns.tolist())
        else:
            st.warning("No columns available in the DataFrame.")
    elif choice == "Show Descriptive Statistics":
        stats_include = st.radio("Include statistics for:", ["Numeric only", "All columns"], index=0)


with col2:
    st.subheader("Results")
    try:
        if choice == "View Head (first N rows)":
            st.dataframe(df.head(n_rows), use_container_width=True)
        elif choice == "View Tail (last N rows)":
            st.dataframe(df.tail(n_rows), use_container_width=True)
        elif choice == "Show Info (column types, non-null counts)":
            # Capture info() output
            buffer = io.StringIO()
            df.info(buf=buffer, verbose=True, show_counts=True)
            info_str = buffer.getvalue()
            st.text(info_str)
        elif choice == "Show Descriptive Statistics":
            if stats_include == "All columns":
                st.dataframe(df.describe(include='all', datetime_is_numeric=True), use_container_width=True)
            else:
                st.dataframe(df.describe(datetime_is_numeric=True), use_container_width=True)
        elif choice == "List Column Names and Types":
            st.dataframe(df.dtypes.reset_index().rename(columns={'index': 'Column', 0: 'DataType'}), use_container_width=True)
        elif choice == "Show Unique Values in a Column":
            if selected_column:
                uniques = df[selected_column].unique()
                st.write(f"Unique values in '{selected_column}' ({len(uniques)} total):")
                # Display smartly, avoid crashing browser with millions of uniques
                if len(uniques) > 1000:
                    st.write(uniques[:1000])
                    st.caption("...(Output truncated for brevity)")
                else:
                    st.write(uniques)
            else:
                 st.info("Select a column.")
        elif choice == "Count Values in a Column":
             if selected_column:
                st.write(f"Value counts for '{selected_column}':")
                # Use st.dataframe for better formatting of counts
                counts_df = df[selected_column].value_counts().reset_index()
                counts_df.columns = [selected_column, 'count']
                st.dataframe(counts_df, use_container_width=True)
             else:
                 st.info("Select a column.")
        elif choice == "Check for Missing Values (NaNs/NaTs)":
             nan_counts = df.isna().sum()
             nan_cols = nan_counts[nan_counts > 0].reset_index()
             if not nan_cols.empty:
                 nan_cols.columns = ['Column', 'Missing Count']
                 st.write("Columns with missing values (NaNs/NaTs):")
                 st.dataframe(nan_cols, use_container_width=True)
                 total_nans = nan_cols['Missing Count'].sum()
                 total_cells = np.prod(df.shape)
                 percent_nans = (total_nans / total_cells) * 100 if total_cells > 0 else 0
                 st.metric(label="Total Missing Values", value=f"{total_nans}", delta=f"{percent_nans:.2f}% of total cells", delta_color="inverse")
             else:
                 st.success("No missing values (NaNs/NaTs) found in the dataset.")
    except Exception as e:
        st.error(f"An error occurred during inspection: {e}")

st.divider()
st.subheader("Current Data Preview (First 5 Rows)")
st.dataframe(df.head(), use_container_width=True)

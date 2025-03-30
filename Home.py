# cdat_app.py
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="CDAT Streamlit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Session State ---
# This dictionary-like object persists across script reruns and pages
if 'df' not in st.session_state:
    st.session_state.df = None # Holds the main DataFrame
if 'original_filename' not in st.session_state:
    st.session_state.original_filename = None
if 'df_modified' not in st.session_state:
    st.session_state.df_modified = False # Track if changes were made

# --- Main Page Content ---
st.title("📊 Comprehensive Data Analysis Tool (CDAT)")
st.caption("Navigate through the steps using the sidebar.")

st.markdown("""
Welcome to CDAT! This tool allows you to interactively load, clean, transform, analyze, and interpolate your tabular data.

**Getting Started:**

1.  Go to the **Load Data** page in the sidebar to upload your dataset (CSV or Excel).
2.  Once loaded, proceed through the other pages (Inspect, Clean, etc.) to work with your data.
3.  The currently loaded data is preserved as you move between pages.
4.  Use the **Save Data** page to download your processed data.

""")

# Display current data status
st.sidebar.header("Current Status")
if st.session_state.df is not None:
    st.sidebar.success(f"Data loaded: {st.session_state.df.shape[0]} rows, {st.session_state.df.shape[1]} columns")
    if st.session_state.original_filename:
        st.sidebar.caption(f"From: {st.session_state.original_filename}")
    if st.session_state.df_modified:
        st.sidebar.warning("Data has been modified.")
    # Optional: Add a button to clear data?
    # if st.sidebar.button("Clear Loaded Data"):
    #     st.session_state.df = None
    #     st.session_state.original_filename = None
    #     st.session_state.df_modified = False
    #     st.rerun() # Rerun to update UI
else:
    st.sidebar.info("No data loaded yet.")
    st.sidebar.caption("Go to 'Load Data' page.")


st.info("Use the sidebar on the left to navigate between different functions.")

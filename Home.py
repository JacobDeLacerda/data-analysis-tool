import streamlit as st
from utils import init_session_state, sidebar_status

st.set_page_config(
    page_title="Data Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
sidebar_status()

st.title("📊 Data Analysis Tool")
st.markdown("""
Use the sidebar to navigate through the workflow:

| Step | Page | Purpose |
|------|------|---------|
| 1 | **Load Data** | Upload a CSV or Excel file |
| 2 | **Inspect** | Explore structure and content |
| 3 | **Clean** | Fix missing values and duplicates |
| 4 | **Transform** | Filter, reshape, and engineer features |
| 5 | **Analyze** | Correlations and group statistics |
| 6 | **Interpolate** | Estimate values between known points |
| 7 | **Save** | Download the processed data |
""")

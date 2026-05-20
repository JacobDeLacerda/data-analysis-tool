import streamlit as st
import pandas as pd
from utils import init_session_state, sidebar_status

st.set_page_config(layout="wide")
init_session_state()
sidebar_status()

st.header("1. Load Data")

uploaded_file = st.file_uploader(
    "Upload a CSV, Excel (.xlsx/.xls), or delimited text file",
    type=['csv', 'xlsx', 'xls', 'txt'],
)

if uploaded_file is not None:
    ext = uploaded_file.name.rsplit('.', 1)[-1].lower()

    if ext in ('csv', 'txt') or 'text' in uploaded_file.type:
        with st.expander("CSV / Text options", expanded=True):
            c1, c2, c3 = st.columns(3)
            delimiter = c1.text_input("Delimiter", value=',', help="Use \\t for tab")
            if delimiter == '\\t':
                delimiter = '\t'
            header_row = c2.number_input("Header row (0-based)", min_value=0, value=0, step=1)
            comment_char = c3.text_input("Comment char (optional)", value='') or None

        if st.button("Load Data", type="primary"):
            try:
                df = pd.read_csv(
                    uploaded_file, delimiter=delimiter,
                    header=header_row, comment=comment_char, skipinitialspace=True,
                )
                st.session_state.df = df
                st.session_state.original_filename = uploaded_file.name
                st.session_state.df_modified = False
                st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns.")
                st.dataframe(df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Error loading file: {e}")

    elif ext in ('xlsx', 'xls'):
        try:
            xls = pd.ExcelFile(uploaded_file)
            if not xls.sheet_names:
                st.warning("No sheets found in this Excel file.")
            else:
                with st.expander("Excel options", expanded=True):
                    c1, c2 = st.columns(2)
                    sheet = c1.selectbox("Sheet", xls.sheet_names)
                    header_row = c2.number_input("Header row (0-based)", min_value=0, value=0, step=1)

                if st.button("Load Data", type="primary"):
                    try:
                        df = pd.read_excel(uploaded_file, sheet_name=sheet, header=header_row)
                        st.session_state.df = df
                        st.session_state.original_filename = uploaded_file.name
                        st.session_state.df_modified = False
                        st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns.")
                        st.dataframe(df.head(), use_container_width=True)
                    except Exception as e:
                        st.error(f"Error loading Excel file: {e}")
                        st.error("Ensure openpyxl is installed: pip install openpyxl")
        except Exception as e:
            st.error(f"Could not read Excel file: {e}")

    else:
        st.warning(f"Unsupported file type: .{ext}")

elif st.session_state.get('df') is not None:
    st.subheader("Current data (first 5 rows)")
    st.dataframe(st.session_state.df.head(), use_container_width=True)
else:
    st.info("Upload a file to get started.")

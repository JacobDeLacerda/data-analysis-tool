import streamlit as st
import pandas as pd
import io
import os
from utils import require_data, sidebar_status

st.set_page_config(layout="wide")
sidebar_status()
require_data()

st.header("7. Save Data")

df = st.session_state.df
st.info(f"{df.shape[0]:,} rows × {df.shape[1]} columns")
if st.session_state.get('df_modified'):
    st.warning("This data has been modified from the original.")

st.divider()

original = st.session_state.get('original_filename', '')
base = os.path.splitext(original)[0] if original else 'processed_data'
default_name = f"{base}_processed.csv"

fmt = st.radio("Format:", ["CSV", "Excel (.xlsx)"], horizontal=True)
filename = st.text_input("Filename:", value=default_name)

if fmt == "CSV":
    c1, c2 = st.columns(2)
    delimiter = c1.text_input("Delimiter:", value=',', help="Use \\t for tab")
    if delimiter == '\\t':
        delimiter = '\t'
    include_index = c2.checkbox("Include row index?", value=False)

    @st.cache_data
    def to_csv(df, sep, idx):
        return df.to_csv(sep=sep, index=idx).encode('utf-8')

    data = to_csv(df, delimiter, include_index)
    mime = 'text/csv'

else:
    c1, c2 = st.columns(2)
    sheet = c1.text_input("Sheet name:", value="Sheet1")
    include_index = c2.checkbox("Include row index?", value=False)
    if not filename.lower().endswith('.xlsx'):
        filename += '.xlsx'

    @st.cache_data
    def to_excel(df, sheet, idx):
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as w:
            df.to_excel(w, sheet_name=sheet, index=idx)
        return buf.getvalue()

    data = to_excel(df, sheet, include_index)
    mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

if filename:
    st.download_button(
        f"Download as {fmt}",
        data=data,
        file_name=filename,
        mime=mime,
        type="primary",
    )
else:
    st.warning("Enter a filename.")

st.divider()
st.subheader("Preview (first 5 rows)")
st.dataframe(df.head(), use_container_width=True)

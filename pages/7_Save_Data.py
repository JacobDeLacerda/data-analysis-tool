# pages/7_Save_Data.py
import streamlit as st
import pandas as pd
import io # For BytesIO
import os # For path manipulation

st.set_page_config(layout="wide")

st.header("7. Save Current Data")
st.caption("Download the currently processed data (in its current state) to a file.")

# --- Check if data is loaded ---
if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("No data loaded to save. Please load and process data first.")
    st.stop()

df_to_save = st.session_state.df

st.info(f"Data to be saved has {df_to_save.shape[0]} rows and {df_to_save.shape[1]} columns.")
if st.session_state.get('df_modified', False):
    st.warning("Note: This data has been modified from the originally loaded file.")

st.divider()
st.subheader("Choose Output Format and Options")

# --- Determine Default Filename ---
default_filename = "processed_data.csv"
original_filename = st.session_state.get('original_filename', None)
if original_filename:
    base, ext = os.path.splitext(original_filename)
    processed_ext = ext.lower() if ext.lower() in ['.csv', '.xlsx', '.xls'] else '.csv'
    default_filename = f"{base}_processed{processed_ext}"


# --- Format Selection ---
file_format = st.radio("Select output format:", ["CSV", "Excel (.xlsx)"], index=0, horizontal=True)
output_filename = st.text_input("Enter output filename:", value=default_filename)

# --- Format Specific Options ---
file_data = None
mime_type = None

if file_format == "CSV":
    delimiter = st.text_input("Delimiter:", value=',', help="Common delimiters: ',' (comma), '\\t' (tab), ';' (semicolon).")
    if delimiter == '\\t': delimiter = '\t'
    include_index_csv = st.checkbox("Include DataFrame index?", value=False)
    mime_type = 'text/csv'

    @st.cache_data # Cache the conversion
    def convert_df_to_csv(df, sep, index):
        return df.to_csv(sep=sep, index=index).encode('utf-8')

    if output_filename:
         file_data = convert_df_to_csv(df_to_save, delimiter, include_index_csv)

elif file_format == "Excel (.xlsx)":
    sheet_name = st.text_input("Sheet name:", value="Processed Data")
    include_index_excel = st.checkbox("Include DataFrame index?", value=False)
    mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

    @st.cache_data # Cache the conversion
    def convert_df_to_excel(df, sheet, index):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet, index=index)
        processed_data = output.getvalue()
        return processed_data

    if output_filename:
        # Ensure filename has .xlsx extension
        if not output_filename.lower().endswith('.xlsx'):
            output_filename += '.xlsx'
        file_data = convert_df_to_excel(df_to_save, sheet_name, include_index_excel)

# --- Download Button ---
if output_filename and file_data:
    st.download_button(
        label=f"Download as {file_format}",
        data=file_data,
        file_name=output_filename,
        mime=mime_type,
        type="primary"
    )
elif not output_filename:
    st.warning("Please enter an output filename.")


st.divider()
st.subheader("Current Data Preview (First 5 Rows)")
st.dataframe(st.session_state.df.head(), use_container_width=True)

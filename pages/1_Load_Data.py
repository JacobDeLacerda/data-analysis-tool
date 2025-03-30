# pages/1_Load_Data.py
import streamlit as st
import pandas as pd
import io # For handling uploaded file bytes

st.set_page_config(layout="wide") # Ensure wide layout for pages too

st.header("1. Load Data")
st.caption("Upload your CSV or Excel file here.")

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Drag and drop your file or click to browse",
    type=['csv', 'xlsx', 'xls', 'txt'],  # Allow common extensions
    accept_multiple_files=False,
    help="Upload a CSV, Excel (.xlsx, .xls), or other delimited text file."
)

if uploaded_file is not None:
    st.success(f"File '{uploaded_file.name}' uploaded successfully!")

    # --- Determine File Type and Get Loading Options ---
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
    st.write("File Details:", file_details)

    file_extension = uploaded_file.name.split('.')[-1].lower()

    # --- CSV/Text Options ---
    if file_extension in ['csv', 'txt'] or 'text' in uploaded_file.type:
        st.subheader("CSV / Text File Options")
        delimiter = st.text_input(
            "Delimiter (separator)",
            value=',',
            help="Common delimiters: ',' (comma), '\\t' (tab), ';' (semicolon), '|' (pipe). Use '\\t' for tab."
        )
        # Interpret \t correctly
        if delimiter == '\\t':
            delimiter = '\t'

        header_row_option = st.number_input(
            "Header Row Number (0-based)",
            min_value=0,
            value=0,
            step=1,
            help="Row containing column names (usually 0 for the first row). If no header, set high and clean later or load without."
        )
        # Option for no header? Could set header=None if needed.

        comment_char = st.text_input(
            "Comment Character (optional)",
            value='#',
            help="Lines starting with this character will be ignored (e.g., '#'). Leave blank if none."
        )
        if not comment_char:
            comment_char = None

        # --- Load Button for CSV ---
        if st.button("Load CSV/Text Data", type="primary"):
            try:
                # To read directly from uploaded file object:
                dataframe = pd.read_csv(
                    uploaded_file,
                    delimiter=delimiter,
                    header=header_row_option,
                    comment=comment_char,
                    skipinitialspace=True
                )
                # Store in session state
                st.session_state.df = dataframe
                st.session_state.original_filename = uploaded_file.name
                st.session_state.df_modified = False # Reset modified flag
                st.success("Data loaded successfully!")
                st.info("Proceed to 'Inspect Data' or other pages.")
                st.dataframe(dataframe.head(), use_container_width=True) # Show preview
            except Exception as e:
                st.error(f"Error loading CSV/Text file: {e}")
                st.error("Please check delimiter, header row, file encoding, and file integrity.")

    # --- Excel Options ---
    elif file_extension in ['xlsx', 'xls']:
        st.subheader("Excel File Options")
        try:
             # Need to read the file to get sheet names
             excel_file = pd.ExcelFile(uploaded_file)
             sheet_names = excel_file.sheet_names
             if not sheet_names:
                  st.warning("This Excel file appears to have no sheets.")
             else:
                  sheet_name = st.selectbox(
                      "Select Sheet",
                      sheet_names,
                      help="Choose the sheet containing the data you want to load."
                  )
                  header_row_option = st.number_input(
                      "Header Row Number (0-based)",
                      min_value=0,
                      value=0,
                      step=1,
                      help="Row containing column names (usually 0 for the first row)."
                  )

                  # --- Load Button for Excel ---
                  if st.button("Load Excel Data", type="primary"):
                      try:
                          dataframe = pd.read_excel(
                              uploaded_file, # Can read directly from uploaded file
                              sheet_name=sheet_name,
                              header=header_row_option
                          )
                          # Store in session state
                          st.session_state.df = dataframe
                          st.session_state.original_filename = uploaded_file.name
                          st.session_state.df_modified = False # Reset modified flag
                          st.success("Data loaded successfully!")
                          st.info("Proceed to 'Inspect Data' or other pages.")
                          st.dataframe(dataframe.head(), use_container_width=True) # Show preview
                      except Exception as e:
                          st.error(f"Error loading Excel file: {e}")
                          st.error("Ensure the selected sheet and header row are correct. You might need 'openpyxl' or 'xlrd' installed (`pip install openpyxl xlrd`).")

        except Exception as e:
            st.error(f"Could not process Excel file: {e}")
            st.error("Make sure the file is a valid Excel file and you have 'openpyxl'/'xlrd' installed.")

    else:
        st.warning(f"Unsupported file extension: '{file_extension}'. Please upload CSV or Excel files.")

# --- Display Status ---
if st.session_state.df is not None:
    st.divider()
    st.subheader("Current Data Preview (First 5 Rows)")
    st.dataframe(st.session_state.df.head(), use_container_width=True)
else:
    st.info("Upload a file to begin.")

import streamlit as st


def init_session_state():
    for key, default in [('df', None), ('original_filename', None), ('df_modified', False)]:
        if key not in st.session_state:
            st.session_state[key] = default


def require_data():
    if st.session_state.get('df') is None:
        st.warning("No data loaded. Go to **Load Data** first.")
        st.stop()


def sidebar_status():
    st.sidebar.header("Data Status")
    df = st.session_state.get('df')
    if df is not None:
        st.sidebar.success(f"{df.shape[0]:,} rows × {df.shape[1]} columns")
        fname = st.session_state.get('original_filename')
        if fname:
            st.sidebar.caption(f"Source: {fname}")
        if st.session_state.get('df_modified'):
            st.sidebar.warning("Modified from original")
    else:
        st.sidebar.info("No data loaded.")
        st.sidebar.caption("Start at **Load Data**.")

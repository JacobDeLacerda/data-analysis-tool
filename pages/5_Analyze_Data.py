import streamlit as st
import pandas as pd
import numpy as np
from utils import require_data, sidebar_status

st.set_page_config(layout="wide")
sidebar_status()
require_data()

st.header("5. Analyze Data")

df = st.session_state.df

action = st.selectbox(
    "Action:",
    ["Correlations", "Group & Aggregate"],
    index=None,
    placeholder="Select analysis…",
)

st.divider()

if action == "Correlations":
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        st.warning("Need at least two numeric columns for correlation analysis.")
    else:
        cols = st.multiselect(
            "Columns to include:",
            numeric_df.columns.tolist(),
            default=numeric_df.columns.tolist(),
        )
        if len(cols) < 2:
            st.warning("Select at least two columns.")
        else:
            corr = numeric_df[cols].corr()
            try:
                st.dataframe(
                    corr.style.background_gradient(cmap='coolwarm', axis=None).format("{:.2f}"),
                    use_container_width=True,
                )
            except Exception:
                st.dataframe(corr.map("{:.2f}".format), use_container_width=True)

elif action == "Group & Aggregate":
    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    group_cols = st.multiselect("Group by:", all_cols)
    if group_cols:
        suggested = [c for c in numeric_cols if c not in group_cols]
        agg_cols = st.multiselect("Aggregate columns:", all_cols, default=suggested)

        if agg_cols:
            agg_funcs = st.multiselect(
                "Functions:",
                ['mean', 'median', 'sum', 'count', 'std', 'var', 'min', 'max',
                 'first', 'last', 'nunique'],
                default=['mean', 'count'],
            )

            if agg_funcs:
                with st.expander("Advanced"):
                    drop_na_groups = st.checkbox("Exclude groups with NaN keys", value=True)

                if st.button("Calculate", type="primary"):
                    try:
                        grouped = df.groupby(group_cols, dropna=drop_na_groups).agg(
                            {col: agg_funcs for col in agg_cols}
                        )
                        if isinstance(grouped.columns, pd.MultiIndex):
                            grouped.columns = ['_'.join(map(str, c)) for c in grouped.columns]
                        result = grouped.reset_index()

                        st.success("Aggregation complete.")
                        st.dataframe(result, use_container_width=True)
                        st.download_button(
                            "Download as CSV",
                            data=result.to_csv(index=False).encode('utf-8'),
                            file_name=f"aggregated_by_{'_'.join(group_cols)}.csv",
                            mime='text/csv',
                        )

                        if st.checkbox("Replace current data with these results?"):
                            if st.button("Confirm Replace"):
                                st.session_state.df = result
                                st.session_state.df_modified = True
                                st.success("Dataset replaced with aggregation results.")
                                st.rerun()
                    except Exception as e:
                        st.error(f"Aggregation error: {e}")

st.divider()
st.dataframe(st.session_state.df.head(), use_container_width=True)

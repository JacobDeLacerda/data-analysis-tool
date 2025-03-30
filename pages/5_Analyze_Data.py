# pages/5_Analyze_Data.py
import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(layout="wide")

st.header("5. Analyze Data")
st.caption("Calculate correlations and perform group-by aggregations.")

# --- Check if data is loaded ---
if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("No data loaded. Please go to the 'Load Data' page first.")
    st.stop()

df = st.session_state.df # Get the DataFrame

# --- Analysis Options ---
st.subheader("Select Analysis Action")

analysis_action = st.selectbox(
    "Choose Action:",
    [
        "Correlations (numeric columns)",
        "Group By and Aggregate"
    ],
    index=None,
    placeholder="Select analysis..."
)

st.divider()

# --- Action Specific Controls & Output ---

if analysis_action == "Correlations (numeric columns)":
    st.markdown("**Correlation Matrix**")
    st.caption("Calculates the pairwise correlation between numeric columns (-1 to +1).")
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        st.warning("Need at least two numeric columns for correlation analysis.")
    else:
        st.write("Select numeric columns for correlation (optional, default is all):")
        cols_for_corr = st.multiselect(
            "Columns:",
            numeric_df.columns.tolist(),
            default=numeric_df.columns.tolist(), # Default to all numeric
            help="Choose the numeric columns to include in the correlation matrix."
        )
        if len(cols_for_corr) < 2:
             st.warning("Please select at least two numeric columns.")
        else:
             corr_matrix = numeric_df[cols_for_corr].corr()
             st.write("Correlation Matrix:")
             # Display with heatmap styling
             try:
                 st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None).format("{:.2f}"), use_container_width=True)
             except Exception as e:
                  st.warning(f"Could not apply heatmap styling: {e}. Displaying plain table.")
                  st.dataframe(corr_matrix.applymap("{:.2f}".format), use_container_width=True) # Apply formatting for consistency


elif analysis_action == "Group By and Aggregate":
    st.markdown("**Group By and Aggregate**")
    st.caption("Group data by one or more columns and calculate summary statistics for others.")

    # Ensure columns are available
    if df.columns.empty:
        st.warning("No columns available in the DataFrame.")
        st.stop()

    cols_all = df.columns.tolist()
    group_cols = st.multiselect(
        "Select column(s) to GROUP BY:",
        cols_all,
        help="Choose the columns whose unique combinations will form the groups."
    )

    if group_cols:
        # Suggest numeric cols for aggregation but allow others
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        suggested_agg_cols = [c for c in numeric_cols if c not in group_cols]
        other_cols = [c for c in cols_all if c not in group_cols and c not in numeric_cols]

        agg_cols = st.multiselect(
            "Select column(s) to AGGREGATE:",
            cols_all, # Allow selecting any column
            default=suggested_agg_cols, # Suggest numeric ones
            help="Choose the columns you want to calculate statistics for within each group."
        )

        if agg_cols:
            agg_funcs_options = ['mean', 'median', 'sum', 'count', 'size', 'std', 'var', 'min', 'max', 'first', 'last', 'nunique']
            agg_funcs = st.multiselect(
                "Select aggregation function(s):",
                agg_funcs_options,
                default=['mean', 'count'], # Sensible defaults
                help="Choose one or more functions to apply to the aggregate columns."
            )

            if agg_funcs:
                # Use an expander for advanced options
                with st.expander("Advanced Options"):
                    dropna_group = st.checkbox("Drop groups where group key is NA?", value=True, help="If unchecked, groups with missing values in group-by columns will be included.")
                    # observed_group = st.checkbox("Use observed combinations only (for categorical groupers)?", value=False) # Less common need

                if st.button("Calculate Aggregation", type="primary"):
                    try:
                         agg_dict = {col: agg_funcs for col in agg_cols}
                         st.write(f"Grouping by `{', '.join(group_cols)}`, aggregating `{', '.join(agg_cols)}` with `{', '.join(agg_funcs)}`...")

                         grouped_df = df.groupby(group_cols, dropna=dropna_group).agg(agg_dict)

                         # Flatten multi-index columns
                         if isinstance(grouped_df.columns, pd.MultiIndex):
                              grouped_df.columns = ['_'.join(map(str, col)).strip() for col in grouped_df.columns.values]

                         grouped_df_reset = grouped_df.reset_index() # Always reset for consistent display/save

                         st.success("Aggregation Complete!")
                         st.dataframe(grouped_df_reset, use_container_width=True)

                         # --- Download Button for Aggregation ---
                         csv_agg = grouped_df_reset.to_csv(index=False).encode('utf-8')
                         default_agg_filename = f"aggregated_by_{'_'.join(group_cols)}.csv"
                         st.download_button(
                             label="Download Aggregation Results as CSV",
                             data=csv_agg,
                             file_name=default_agg_filename,
                             mime='text/csv',
                         )

                         # Option to replace main DataFrame
                         if st.checkbox("Replace current data with these aggregation results? (Use with caution!)"):
                              if st.button("Confirm Replace Data"):
                                   st.session_state.df = grouped_df_reset
                                   st.session_state.df_modified = True
                                   st.success("Main dataset replaced with aggregation results.")
                                   st.rerun() # Rerun to reflect change everywhere

                    except Exception as e:
                         st.error(f"Error during aggregation: {e}")
                         st.error("Check if functions are valid for selected column types.")

# Display current data preview at the end
st.divider()
st.subheader("Current Data Preview (First 5 Rows)")
st.dataframe(st.session_state.df.head(), use_container_width=True)

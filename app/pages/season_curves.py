#show summary table, with predicted and actual values, show any graphs

import streamlit as st


table = st.session_state.table

st.write(table.summary())
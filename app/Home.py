import streamlit as st
import sys
import paths
sys.path.insert(0, str(paths.ROOT))

import control.setup_session as setup_session
setup_session.load_data()

st.write(paths.HISTORICAL)
pe_graphs = setup_session.pred_evolution_graphs()
pw_tables = setup_session.preds_by_week_tables()

st.session_state['pe_graphs'] = pe_graphs
st.session_state['pw_tables'] = pw_tables

st.write(st.session_state['data']['init_kg'].shape)

st.title("Home")
st.write("Welcome to the home page")

st.title("Home")
st.write("Welcome to the home page")

st.title("Home")
st.write("Welcome to the home page")

st.title("Home")
st.write("Welcome to the home page")
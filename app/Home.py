import streamlit as st
import sys
import paths

sys.path.append(str(paths.ROOT))

import control.setup_session as setup_session
setup_session.load_proper_data()

st.write(paths.HISTORICAL)
pe_graphs = setup_session.pred_evolution_graphs()
pw_tables = setup_session.preds_by_week_tables()

st.session_state['pe_graphs'] = pe_graphs
st.session_state['pw_tables'] = pw_tables


st.title("Home")
st.write("Welcome to the home page")

st.title("Pages Overview")
st.write(
    "Historical: Shows current season compared to previous seasons\n \
    This Seasn: Shows current season/projections over time\n \
    This Week: Shows the projections for this upcoming week")


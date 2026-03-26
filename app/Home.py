import streamlit as st
import sys
import paths

sys.path.append(str(paths.ROOT))

import control.setup_session as setup_session

setup_session.load_data()
st.write('loaded data')
setup_session.load_historical_data()
st.write('loaded historical data')
setup_session.load_mappings()
st.write('loaded mappings')


st.title("Home")
st.write("Welcome to the home page")

st.title("Pages Overview")
st.write(
    "Historical: Shows current season compared to previous seasons\n \
    This Season: Shows current season/projections over time\n \
    This Week: Shows the projections for this upcoming week")


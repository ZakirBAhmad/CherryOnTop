import streamlit as st
import src.graphs as graphs
st.set_page_config(layout="wide")

st.title("Historical")
st.write("Welcome to the historical page")

st.title("this season: 2024")
st.write("week slider")

szn_act_kg = st.session_state['data']['szn_act_kg']
szn_adj_kg = st.session_state['data']['szn_adj_kg']
graph = graphs.this_season_graph(szn_act_kg,szn_adj_kg)
st.write("Insert current graph")
st.write("hectares planted")


st.title("2023")
st.write("insert graph for 2023")
st.write("hectares planted")

st.title("2022")
st.write("insert graph for 2022")
st.write("hectares planted")
st.title("etc")
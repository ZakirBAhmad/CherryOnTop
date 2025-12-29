import streamlit as st
import src.graphs as graphs
st.set_page_config(layout="wide")

st.title("This Season")
st.write("Welcome to the  page")

szn_act_kg = st.session_state['data']['szn_act_kg']
szn_adj_kg = st.session_state['data']['szn_adj_kg']

st.write(szn_act_kg.shape)

graph = graphs.this_season_graph(szn_act_kg,szn_adj_kg)
st.plotly_chart(graph)


st.title("this season: 2024")
st.write("week slider")
st.write("Insert current graph")
st.write("add table")



st.write("hectares planted")
st.write("insert hectare graph")
st.write("add table")

st.write("week slider")
st.write("Insert sched insights per week")




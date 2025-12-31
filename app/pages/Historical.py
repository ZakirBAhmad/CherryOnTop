import streamlit as st
import src.graphs as graphs
import src.table as table
st.set_page_config(layout="wide")

st.title("Historical")
st.write("Welcome to the historical page")

st.title("this season: 2024")
st.write("week slider")


h_meta = st.session_state['data']['h_meta']
h_y = st.session_state['data']['h_y']
ht_weeks = st.session_state['data']['ht_weeks']
h_idx_dict = st.session_state['data']['h_idx_dict']
h_total_kg = st.session_state['data']['h_total_kg']

szn_hist_kg = table.season_shift(h_y.values,ht_weeks)

szn_act_kg = st.session_state['data']['szn_act_kg']
szn_adj_kg = st.session_state['data']['szn_adj_kg']
graph = graphs.this_season_graph(szn_act_kg,szn_adj_kg)
st.plotly_chart(graph)

st.write("Insert current graph")
st.write("hectares planted")


st.title("2023")
year = 2023
idx = h_idx_dict['year'][year]

fig = graphs.season_act_graph(h_meta,szn_hist_kg,idx,'2023')
st.plotly_chart(fig)
st.write("insert graph for 2023")
st.write("hectares planted")

st.title("2022")
year = 2022
idx = h_idx_dict['year'][year]
fig = graphs.season_act_graph(h_meta,szn_hist_kg,idx,'2022')
st.plotly_chart(fig)
st.write("insert graph for 2022")
st.write("hectares planted")

st.title("2021")
year = 2021
idx = h_idx_dict['year'][year]
fig = graphs.season_act_graph(h_meta,szn_hist_kg,idx,'2021')
st.plotly_chart(fig)
st.write("insert graph for 2021")
st.write("hectares planted")

st.title("2020")
year = 2020
idx = h_idx_dict['year'][year]
fig = graphs.season_act_graph(h_meta,szn_hist_kg,idx,'2020')
st.plotly_chart(fig)
st.write("insert graph for 2020")
st.write("hectares planted")

st.title("etc")
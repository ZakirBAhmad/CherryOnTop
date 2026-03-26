import streamlit as st
import src.graphs as graphs
import src.table as table
st.set_page_config(layout="wide")

st.title("This Season")
st.write("Welcome to the page")

preds = st.session_state['current_data']['final_preds']
szn_act = st.session_state['current_data']['szn_act']
idx_dict = st.session_state['current_data']['idx_dict']

##### Uva #####
st.write('Uva')
idx = idx_dict['class']['Uva']
uva_fig = graphs.this_season_CI_graph(preds[idx],szn_act[idx],'Uva')
uva_df = table.class_prediction_history(preds, szn_act, 'Uva', idx_dict)
st.plotly_chart(uva_fig,key='uva_fig')
st.dataframe(uva_df.round(0))

#### Cherry #####
st.write('Cherry')
idx = idx_dict['class']['Cherry']
cherry_fig = graphs.this_season_CI_graph(preds[idx],szn_act[idx],'Cherry')
cherry_df = table.class_prediction_history(preds, szn_act, 'Cherry', idx_dict)
st.plotly_chart(cherry_fig,key='cherry_fig')
st.dataframe(cherry_df.round(0))

##### Mix #####
st.write('Mix')
idx = idx_dict['class']['Mix']
mix_fig = graphs.this_season_CI_graph(preds[idx],szn_act[idx],'Mix')
mix_df = table.class_prediction_history(preds, szn_act, 'Mix', idx_dict)
st.plotly_chart(mix_fig,key='mix_fig')
st.dataframe(mix_df.round(0))
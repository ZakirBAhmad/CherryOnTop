import streamlit as st
import src.graphs as graphs
import src.load as load
st.set_page_config(layout="wide")

meta = st.session_state['historical_data']['meta']
idx_dict = st.session_state['historical_data']['idx_dict']
mappings = st.session_state['mappings']

st.title("Insights")
fig = graphs.yield_breakdown_graph(meta,'Class Historical Yields','class',mappings = mappings,categorical = True)
st.write('Class Historical Yields')
st.plotly_chart(fig,key='class_historical_yields')

###### Uva #####
st.write('Uva')
idx = idx_dict['class']['Uva']
fig1 = graphs.yield_breakdown_graph(meta[idx],'Ranch Breakdown for Uva','ranch',mappings = mappings,categorical = True)
fig2 = graphs.yield_breakdown_graph(meta[idx],'Yield Breakdown for Uva','transplant_week')
fig3 = graphs.kg_ha_graph(meta[idx],'kg per ha for Uva')
st.plotly_chart(fig1,key='uva_ranch_breakdown')
st.plotly_chart(fig2,key='uva_yield_breakdown')
st.plotly_chart(fig3,key='uva_kg_ha')

###### Cherry #####
st.write('Cherry')
idx = idx_dict['class']['Cherry']
fig1 = graphs.yield_breakdown_graph(meta[idx],'Ranch Breakdown for Cherry','ranch',mappings = mappings,categorical = True)
fig2 = graphs.yield_breakdown_graph(meta[idx],'Yield Breakdown for Cherry','transplant_week')
fig3 = graphs.kg_ha_graph(meta[idx],'kg per ha for Cherry')
st.plotly_chart(fig1,key='cherry_ranch_breakdown')
st.plotly_chart(fig2,key='cherry_yield_breakdown')
st.plotly_chart(fig3,key='cherry_kg_ha')

###### Mix #####
st.write('Mix')
idx = idx_dict['class']['Mix']
fig1 = graphs.yield_breakdown_graph(meta[idx],'Ranch Breakdown for Mix','ranch',mappings = mappings,categorical = True)
fig2 = graphs.yield_breakdown_graph(meta[idx],'Yield Breakdown for Mix','transplant_week')
fig3 = graphs.kg_ha_graph(meta[idx],'kg per ha for Mix')
st.plotly_chart(fig1,key='mix_ranch_breakdown')
st.plotly_chart(fig2,key='mix_yield_breakdown')
st.plotly_chart(fig3,key='mix_kg_ha')
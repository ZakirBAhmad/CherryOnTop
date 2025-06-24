import streamlit as st
import sys
import os


st.set_page_config(layout="wide")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


st.title('Harvest Curves')

###### Imports #####
import app.demo as demo

###### Season shift, initializations #####
preds = st.session_state['preds']
actuals = st.session_state['actuals']
plan = st.session_state['meta']

graph_dict, summary_class,idx_dict = demo.create_harvest_curves(preds,actuals,plan)




###### display overall shit (season curve, harvest cycle, cumulative sum curve) ######
st.plotly_chart(graph_dict['SeasonCurve'])
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(graph_dict['HarvestCycle'])

with col2:
    st.plotly_chart(graph_dict['CumulativeSum'])


##### tables mayhaps? #####
st.dataframe(summary_class,use_container_width=True)

##### select classes, display curves ######
classes = st.multiselect('Select Classes',plan['Class'].unique(),default=plan['Class'].unique())
if classes:
    fig1, fig2 = demo.graph_classes(classes,preds,actuals,idx_dict,plan)
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)



###### select types, display season curve, harvest cycle, cumulative sum curve #####
types = st.multiselect('Select Types',plan['Type'].unique(),default=plan['Type'].unique())
if types:
    fig1, fig2 = demo.graph_types(types,preds,actuals,idx_dict,plan)
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)

##### tables again #####


#### select ranches, display curves #####
ranches = st.multiselect('Select Ranches',plan['Ranch'].unique(),default=plan['Ranch'].unique())
if ranches:
    fig1, fig2 = demo.graph_ranches(ranches,preds,actuals,idx_dict,plan)
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)

##### show table, summary shown by type #####
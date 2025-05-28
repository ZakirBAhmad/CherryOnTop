import streamlit as st
import src.graphs as graphs

if 'filtered_table' in st.session_state:
    figs,summaries = graphs.graph_transplant_table(st.session_state['filtered_table'],'Harvest Cycle Graphs')
    for i, fig in enumerate(figs):
        st.plotly_chart(fig)
        st.write(summaries[i])
else:
    st.write('No data to display')
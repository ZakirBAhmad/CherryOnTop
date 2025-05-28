import streamlit as st
import src.graphs as graphs
st.title('Projections Summary')

if 'filtered_table' in st.session_state:
    st.write('By Class')
    st.write(st.session_state['filtered_table'].summary(classes=True))
    st.write('By Ranch')
    st.write(st.session_state['filtered_table'].summary(ranches=True))
    
    
else:
    st.write('No data to display')
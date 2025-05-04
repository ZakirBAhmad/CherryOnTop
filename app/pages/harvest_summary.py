#show summary table, with predicted and actual values, show any graphs

import streamlit as st

def show_harvest_summary():
    st.title("Harvest Summary")
    
    # Placeholder for summary table
    st.write("Summary Table:")
    st.write("Table will be displayed here")
    
    # Placeholder for graphs
    st.write("Summary Graphs:")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Predicted vs Actual:")
        st.write("Graph will be displayed here")
    with col2:
        st.write("Yield Distribution:")
        st.write("Graph will be displayed here")
    
    # Placeholder for key metrics
    st.write("Key Metrics:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Predicted", "0 kg")
    with col2:
        st.metric("Total Actual", "0 kg")
    with col3:
        st.metric("Accuracy", "0%")
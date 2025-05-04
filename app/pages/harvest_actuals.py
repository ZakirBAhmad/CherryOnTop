import streamlit as st

def show_harvest_actuals():
    st.title("Harvest Actuals")
    
    # Placeholder for filters
    st.write("Filter Options:")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.selectbox("Select Ranch", ["All", "Ranch 1", "Ranch 2"])
    with col2:
        st.selectbox("Select Class", ["All", "Class 1", "Class 2"])
    with col3:
        st.selectbox("Select Type", ["All", "Type 1", "Type 2"])
    with col4:
        st.selectbox("Select Variety", ["All", "Variety 1", "Variety 2"])
    
    # Placeholder for graphs
    st.write("Predicted Harvest Curve:")
    st.write("Graph will be displayed here")
    
    st.write("Actual Harvest Data:")
    st.write("Bar graph will be displayed here")
    
    # Placeholder for summary statistics
    st.write("Summary Statistics:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cumulative Sum", "0 kg")
    with col2:
        st.metric("Cumulative Proportion", "0%")
    with col3:
        st.metric("Total Harvest", "0 kg")

#have a filtered option in the streamlit

#show the graph of the filtered data predicted harvest curve

#show the actuals in a bar graph, x axis is weeks of the year or weeks after transplant, y axis is kilos

#show cumsum, cumprop, totals

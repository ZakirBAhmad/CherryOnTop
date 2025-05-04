#show the predicted harvest curve for each variety, then for each type, then each class, then each ranch.

import streamlit as st

def show_harvest_curves():
    st.title("Harvest Curves")
    
    # Placeholder for category selection
    category = st.selectbox(
        "Select Category",
        ["Variety", "Type", "Class", "Ranch"]
    )
    
    # Placeholder for specific selection
    st.selectbox(
        f"Select {category}",
        [f"{category} 1", f"{category} 2", f"{category} 3"]
    )
    
    # Placeholder for graph
    st.write("Predicted Harvest Curve:")
    st.write("Graph will be displayed here")
    
    # Placeholder for statistics
    st.write("Statistics:")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Peak Week", "Week 10")
    with col2:
        st.metric("Total Yield", "0 kg")
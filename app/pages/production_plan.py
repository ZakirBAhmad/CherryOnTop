#show planting meta data test in separate year, in form of pivot table, where values are hectares
# use decode method
#columns are weeks of the year rows are ranch, class, type, variety

#filter by ranch, class, type, variety

#have button to run projections, and predict the production plan

#show graph of hectares by week of the year

#show graph of hectares by week of the year for each ranch, class, type, variety

import streamlit as st

st.title("Production Plan")
st.write("This is the production plan page.")

table = st.session_state.table
# Create filter columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    ranches = st.checkbox('Filter by Ranch', value=True)
    if ranches:
        ranch_options = sorted(table.meta['Ranch'].unique())
        selected_ranches = st.multiselect('Select Ranches', ranch_options, default=ranch_options)

with col2:
    types = st.checkbox('Filter by Type', value=True) 
    if types:
        type_options = sorted(table.meta['Type'].unique())
        selected_types = st.multiselect('Select Types', type_options, default=type_options)

with col3:
    classes = st.checkbox('Filter by Class', value=True)
    if classes:
        class_options = sorted(table.meta['Class'].unique())
        selected_classes = st.multiselect('Select Classes', class_options, default=class_options)

with col4:
    varieties = st.checkbox('Filter by Variety', value=True)
    if varieties:
        variety_options = sorted(table.meta['Variety'].unique())
        selected_varieties = st.multiselect('Select Varieties', variety_options, default=variety_options)

filtered_meta = table.meta.copy()
if ranches:
    filtered_meta = filtered_meta[filtered_meta['Ranch'].isin(selected_ranches)]
if types:
    filtered_meta = filtered_meta[filtered_meta['Type'].isin(selected_types)]
if classes:
    filtered_meta = filtered_meta[filtered_meta['Class'].isin(selected_classes)]
if varieties:
    filtered_meta = filtered_meta[filtered_meta['Variety'].isin(selected_varieties)]

st.write("Filtered Data:")
st.dataframe(filtered_meta)

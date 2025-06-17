import streamlit as st
import sys
import os

st.title('Cherry On Top')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import src.preprocessing as pre 
import app.demo as demo


train, test, mappings, meta = demo.initialize_data('data/planting_meta.json','data/y.csv','data/mapping_dict.json')
st.session_state['train'] = train
st.session_state['test'] = test
st.session_state['mappings'] = mappings
st.session_state['meta'] = meta

model = demo.create_model(train)
st.session_state['model'] = model

st.write('Model initialized')


grouped_meta = meta.pivot_table(index=['Ranch','Class','Type'],columns = 'WeekTransplanted',values='Ha').reset_index()

st.write('Production Plan')
# Create filters
col1, col2, col3 = st.columns(3)

with col1:
    ranch_filter = st.multiselect(
        'Ranch',
        options=sorted(grouped_meta['Ranch'].unique()),
        default=sorted(grouped_meta['Ranch'].unique())
    )

with col2:
    class_filter = st.multiselect(
        'Class', 
        options=sorted(grouped_meta['Class'].unique()),
        default=sorted(grouped_meta['Class'].unique())
    )

with col3:
    type_filter = st.multiselect(
        'Type',
        options=sorted(grouped_meta['Type'].unique()),
        default=sorted(grouped_meta['Type'].unique())
    )


# Filter both dataframes
filtered_meta = meta[
    meta['Ranch'].isin(ranch_filter) &
    meta['Class'].isin(class_filter) &
    meta['Type'].isin(type_filter) 
]

filtered_grouped_meta = grouped_meta[
    grouped_meta['Ranch'].isin(ranch_filter) &
    grouped_meta['Class'].isin(class_filter) &
    grouped_meta['Type'].isin(type_filter)
]



st.write(filtered_grouped_meta)

if st.button('Predict'):
    preds = demo.create_predictions(model,test)
    st.session_state['preds'] = preds
    st.session_state['filtered_table'] = st.session_state['preds'].filter(ranch_list=ranch_filter,class_list=class_filter,type_list=type_filter)



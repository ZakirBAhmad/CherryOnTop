import streamlit as st
import pandas as pd
import numpy as np
from src.dataset import HarvestDataset
import src.preprocessing as pre
from src.model import HarvestModel
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from src.utils import train_harvest_model, predict_harvest

# Load and process data
train, test, mappings, test_meta = pre.separate_year('data/planting_meta.json', 'data/y.csv', 'data/mapping_dict.json')

# Create reverse mapping dictionary for each category
reverse_mappings = {}
for category in mappings:
    reverse_mappings[category] = {v: k for k, v in mappings[category].items()}

model = train_harvest_model(train,num_epochs=30)
preds = predict_harvest(model, test)

meta = test_meta[['Ranch','Class','Type','Variety','Ha']].copy()

meta['Ranch'] = meta['Ranch'].map(reverse_mappings['Ranch'])  
meta['Class'] = meta['Class'].map(reverse_mappings['Class']) 
meta['Type'] = meta['Type'].map(reverse_mappings['Type'])
meta['Variety'] = meta['Variety'].map(reverse_mappings['Variety'])

grouped_meta = meta.groupby(['Ranch','Class','Type','Variety']).agg({'Ha':'sum'}).reset_index()

st.title('Cherry On Top')
st.write('Production Plan')
# Create filters
col1, col2, col3, col4 = st.columns(4)

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

with col4:
    variety_filter = st.multiselect(
        'Variety',
        options=sorted(grouped_meta['Variety'].unique()),
        default=sorted(grouped_meta['Variety'].unique())
    )

# Filter both dataframes
filtered_meta = meta[
    meta['Ranch'].isin(ranch_filter) &
    meta['Class'].isin(class_filter) &
    meta['Type'].isin(type_filter) &
    meta['Variety'].isin(variety_filter)
]

filtered_grouped_meta = grouped_meta[
    grouped_meta['Ranch'].isin(ranch_filter) &
    grouped_meta['Class'].isin(class_filter) &
    grouped_meta['Type'].isin(type_filter) &
    grouped_meta['Variety'].isin(variety_filter)
]



st.write(filtered_grouped_meta)
st.write('Individual Batches')
st.write(filtered_meta)
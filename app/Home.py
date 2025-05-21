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

meta = test_meta[['WeekTransplanted','Ranch','Class','Type','Variety','Ha']].copy()

meta['Ranch'] = meta['Ranch'].map(reverse_mappings['Ranch'])  
meta['Class'] = meta['Class'].map(reverse_mappings['Class']) 
meta['Type'] = meta['Type'].map(reverse_mappings['Type'])
meta['Variety'] = meta['Variety'].map(reverse_mappings['Variety'])

st.title('Cherry On Top')

st.write(meta)
st.write(preds)
#imports
import streamlit as st
import src.preprocessing as pre
from src.utils import create_model, predict_harvest
from src.table import CherryTable

@st.cache_resource
def initialize_data(path_meta, path_y, path_mapping_dict):
    train, test, mappings, meta = pre.separate_year(path_meta, path_y, path_mapping_dict)
    meta = pre.decode(meta,mappings)
    return train, test, mappings, meta

@st.cache_resource
def create_model(train_dataset,num_epochs=30):
    model = create_model(train_dataset,num_epochs=num_epochs)
    return model

def create_predictions(model,test,meta,name):
    preds = predict_harvest(model,test)
    table = CherryTable(meta,{name:preds},test.Y.detach().numpy())
    return table
    


    
import numpy as np
import pandas as pd
import json
from src.dataset import HarvestDataset
import streamlit as st
def load_data(path_meta, path_y, path_mapping_dict):
    """
    Load data from JSON and CSV files.
    
    Parameters:
        path_meta (str): Path to JSON file containing metadata
        path_y (str): Path to CSV file containing target variables
        path_mapping_dict (str): Path to JSON file containing mapping dictionary
        
    Returns:
        tuple: (meta DataFrame, y DataFrame, mapping dictionary)
    """
    meta = pd.read_json(path_meta,orient='split')
    y = pd.read_csv(path_y,index_col=0)
    mapping_dict = json.load(open(path_mapping_dict))
    return meta, y, mapping_dict

@st.cache_resource
def separate_year(path_meta, path_y, path_mapping_dict,year = 2024):
    """
    Split data into training and test sets based on year.
    
    Parameters:
        path_meta (str): Path to JSON file containing metadata
        path_y (str): Path to CSV file containing target variables 
        path_mapping_dict (str): Path to JSON file containing mapping dictionary
        year (int): Year to use for test set separation (default: 2024)
        
    Returns:
        tuple: (train_dataset, test_dataset, mapping_dict) where datasets are HarvestDataset objects
    """
    meta, y, mapping_dict = load_data(path_meta, path_y, path_mapping_dict)
    train_meta = meta[meta['Year'] != year]
    test_meta = meta[meta['Year'] == year]
    train_y = y[y.index.isin(train_meta.index)]
    test_y = y[y.index.isin(test_meta.index)]

    train_dataset = make_dataset(train_meta,train_y)
    test_dataset = make_dataset(test_meta,test_y)
    
    return train_dataset, test_dataset, mapping_dict, test_meta

def make_dataset(meta,y):
    """
    Create a HarvestDataset object from metadata and target DataFrames.
    
    Parameters:
        meta (pd.DataFrame): DataFrame containing metadata features
        y (pd.DataFrame): DataFrame containing target variables
        
    Returns:
        HarvestDataset: Dataset object containing processed features and targets
    """
    features = np.column_stack([
    meta['Ha'].to_numpy(),                    # Hectares
    meta['WeekTransplanted_sin'].to_numpy(),  # Week sine
    meta['WeekTransplanted_cos'].to_numpy(),  # Week cosine
    meta['Year'].to_numpy() - 2010,                  # Year
    np.ones(len(meta))                    # Constant feature
    ])
    ranches = meta['Ranch'].to_numpy()
    varieties = meta['Variety'].to_numpy()
    classes = meta['Class'].to_numpy()
    types = meta['Type'].to_numpy()
    climate_data = np.array(meta.ClimateSeries.to_list())
    y_kilos = y.iloc[:,:20].to_numpy()

    return HarvestDataset(
        features,
        ranches,
        classes,
        types,
        varieties,
        climate_data,
        y_kilos
    )

def decode(meta,mapping_dict):
    """
    Decode metadata using mapping dictionary.
    
    Parameters:
        meta (pd.DataFrame): DataFrame containing metadata
        mapping_dict (dict): Mapping dictionary
    """
    meta = meta.copy()
    for key,value in mapping_dict.items():
        meta[key] = meta[key].map(dict(zip(value.values(),value.keys())))
    return meta
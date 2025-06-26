import numpy as np
import pandas as pd
import json
from src.dataset import HarvestDataset

def load_data(folder_path):
    """
    Load data from JSON and CSV files.
    
    Parameters:
        path_meta (str): Path to JSON file containing metadata
        path_y (str): Path to CSV file containing target variables
        path_mapping_dict (str): Path to JSON file containing mapping dictionary
        
    Returns:
        tuple: (meta DataFrame, y DataFrame, mapping dictionary)
    """

    meta = pd.read_json(folder_path + 'meta.json',orient='split')
    meta.TransplantDate = pd.to_datetime(meta.TransplantDate,unit='ms')
    y = pd.read_csv(folder_path + 'y.csv',index_col=0)

    mapping_dict = json.load(open(folder_path + 'mappings.json'))
    reverse_mappings = json.load(open(folder_path + 'reverse_mappings.json'))
    return meta, y, mapping_dict, reverse_mappings

def separate_year(folder_path,year = 2024):
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
    meta, y, mapping_dict, reverse_mappings = load_data(folder_path)
    assert type(meta.TransplantDate) != int, 'TransplantDate is not a datetime object'

    train_meta = meta[meta['Year'] != year]
    test_meta = meta[meta['Year'] == year]
    train_y = y[y.index.isin(train_meta.index)]
    test_y = y[y.index.isin(test_meta.index)]

    train_dataset = make_dataset(train_meta,train_y,mapping_dict)
    test_dataset = make_dataset(test_meta,test_y,mapping_dict)
    
    return train_dataset, test_dataset, mapping_dict, reverse_mappings, test_meta

def make_dataset(meta,y,mappings):
    """
    Create a HarvestDataset object from metadata and target DataFrames.
    
    Parameters:
        meta (pd.DataFrame): DataFrame containing metadata features
        y (pd.DataFrame): DataFrame containing target variables
        
    Returns:
        HarvestDataset: Dataset object containing processed features and targets
    """
    features = np.column_stack([
    meta['Ha'].to_numpy(),  # Hectares
    meta['Week_Sin'].to_numpy(),  # Week sine
    meta['Week_Cos'].to_numpy(),  # Week cosine
    meta['Year'].to_numpy() - 2010,                  # Year
    np.ones(len(meta))                    # Constant feature
    ])
    mapped_arrays = [meta[column].astype(str).map(mappings[column]).to_numpy() for column in ['ProducerCode','Parcel','Lot','Class','Type','Variety']]
    encoded_features = np.column_stack(mapped_arrays)

    climate_data = np.array(meta.ClimateSeries.to_list())
    y_kilos = y.to_numpy()


    return HarvestDataset(
        features,
        encoded_features,
        climate_data,
        y_kilos
    )


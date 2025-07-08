import numpy as np
import pandas as pd
import json
import torch
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

    meta = pd.read_json(folder_path + 'meta.json',orient='split').reset_index(drop=True)
    meta.TransplantDate = pd.to_datetime(meta.TransplantDate,unit='ms')
    y = pd.read_csv(folder_path + 'y.csv',index_col=0).reset_index(drop=True)
    schedule = pd.read_csv(folder_path + 'schedule.csv',index_col=0).reset_index(drop=True)

    mapping_dict = json.load(open(folder_path + 'mappings.json'))
    reverse_mappings = json.load(open(folder_path + 'reverse_mappings.json'))
    return meta, y, schedule, mapping_dict, reverse_mappings

def separate_year(folder_path, year=2024):
    """
    Split data into training and test sets based on year.
    
    Parameters:
        folder_path (str): Path to folder containing data files
        year (int): Year to use for test set separation (default: 2024)
        device (str or torch.device): Device for tensor placement (default: 'cpu')
        
    Returns:
        tuple: (train_dataset, test_dataset, mapping_dict, reverse_mappings, test_meta) 
               where datasets are HarvestDataset objects
    """
    meta, y, schedule, mapping_dict, reverse_mappings = load_data(folder_path)
    assert type(meta.TransplantDate) != int, 'TransplantDate is not a datetime object'

    train_meta = meta[meta['Year'] != year]
    test_meta = meta[meta['Year'] == year]
    train_y = y[y.index.isin(train_meta.index)]
    test_y = y[y.index.isin(test_meta.index)]

    train_dataset = make_dataset(train_meta, train_y, schedule, mapping_dict)
    test_dataset = make_dataset(test_meta, test_y, schedule, mapping_dict)
    
    return train_dataset, test_dataset, mapping_dict, reverse_mappings, test_meta

def separate_prop(folder_path, p=0.8):
    """
    Split data into training and test sets based on proportion.
    
    Parameters:
        folder_path (str): Path to folder containing data files
        p (float): Proportion for training set (default: 0.8)
        device (str or torch.device): Device for tensor placement (default: 'cpu')
        
    Returns:
        tuple: (train_dataset, test_dataset, mapping_dict, reverse_mappings, test_meta)
    """
    meta, y, schedule, mapping_dict, reverse_mappings = load_data(folder_path)
    train_meta = meta.sample(frac=p)
    test_meta = meta[~meta.index.isin(train_meta.index)]

    train_schedule = schedule[schedule.index.isin(train_meta.index)]
    test_schedule = schedule[schedule.index.isin(test_meta.index)]

    train_y = y[y.index.isin(train_meta.index)]
    test_y = y[y.index.isin(test_meta.index)]

    train_dataset = make_dataset(train_meta, train_y, train_schedule, mapping_dict)
    test_dataset = make_dataset(test_meta, test_y, test_schedule, mapping_dict)
    
    return train_dataset, test_dataset, mapping_dict, reverse_mappings, test_meta

def make_dataset(meta, y, schedule, mappings):
    features = np.column_stack([
    meta['Ha'].to_numpy(),  # Hectares
    meta['WeekSin'].to_numpy(),  # Week sine
    meta['WeekCos'].to_numpy(),  # Week cosine
    meta['Year'].to_numpy() - 2010,                  # Year
    np.ones(len(meta))                    # Constant feature
])
    mapped_arrays = [meta[column].astype(str).map(mappings[column]).to_numpy() for column in ['ProducerCode','Parcel','Class','Type','Variety']]
    encoded_features = np.column_stack(mapped_arrays)

    climate_data = np.array(meta.ClimateSeries.to_list())
    schedule_data = schedule.values
    kilo_dist = (y.to_numpy() / y.to_numpy().sum(axis=1, keepdims=True)).cumsum(axis=1)
    yield_dist = np.log1p(y.to_numpy() / meta['Ha'].to_numpy()[:, np.newaxis])
    yield_log = np.log1p(y.to_numpy().sum(axis=1) / meta['Ha'].to_numpy())

    dataset = HarvestDataset(
        features,
        encoded_features,
        climate_data,
        yield_dist,
        kilo_dist,
        yield_log,
        schedule_data
    )
    return dataset

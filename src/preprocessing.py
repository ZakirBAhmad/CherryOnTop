#imports
import pandas as pd
import numpy as np
from src.dataset import HarvestDataset
import torch
from datetime import date

#funcs
def load_tomato(planting_meta_path, weekly_summary_path):
    """
    desc: Loads data from the tomato_data folder

    params: 

    returns: 
        planting_meta (df): contains meta information for each planting, including climate info, hectares
        weekly_summary (df):
    """
    planting_meta = pd.read_json(planting_meta_path,orient='split')
    planting_meta['TransplantDate'] = pd.to_datetime(planting_meta['TransplantDate'])
    planting_meta['ClimateSeries'] = planting_meta['ClimateSeries'].apply(lambda x: pd.Series([np.array(x)]))
    weekly_summary = pd.read_csv(weekly_summary_path)

    weekly_summary = weekly_summary.set_index('PlantingID')
    mapping_dict = construct_mapping_dict(planting_meta)
    return planting_meta, weekly_summary, mapping_dict

def reverse_sin_cos_week(sin, cos,year):
    """
    desc: Reverses the sin and cos transformation of the week

    params: sin, cos

    returns: week
    """
    
    weeks_in_year = date(year, 12, 28).isocalendar()[1]  # Dec 28 is always in last week
    week = (np.degrees(np.arctan2(sin, cos)) * weeks_in_year/360 + weeks_in_year/2) % weeks_in_year
    return week

def separate_year(planting_meta_path, weekly_summary_path, year=2024):
    """
    desc: Separate a year from the dataset, for comparison

    params: year

    returns: planting_meta_train df (excluding year), planting_meta_test df (just data from year)
    """
    planting_meta, weekly_summary, mapping_dict = load_tomato(planting_meta_path, weekly_summary_path)
    production_plan = planting_meta[planting_meta['Year'] == year]
    training_data = planting_meta.drop(production_plan.index)
    

    train_dataset = construct_dataset(training_data, weekly_summary,mapping_dict)
    test_dataset = construct_dataset(production_plan, weekly_summary,mapping_dict)

    return train_dataset, test_dataset, mapping_dict

def construct_mapping_dict(planting_meta):
    """
    desc: Constructs a mapping dictionary for the planting meta data

    params: planting_meta

    returns: mapping_dict
    """
    mapping_dict = {}
    for col in ['Variety', 'Class', 'Type', 'Ranch']:
        mapping_dict[col] = {label: idx for idx, label in enumerate(planting_meta[col].unique())}

    return mapping_dict

def construct_dataset(planting_meta, weekly_summary,mapping_dict):
    """
    desc: Encodes data, for use for training/predicting

    params:

    returns: encoded dataframe
    """

    features = np.column_stack([
        planting_meta['Ha'].values,                    # Hectares
        planting_meta['WeekTransplanted_sin'].values,  # Week sine
        planting_meta['WeekTransplanted_cos'].values,  # Week cosine
        planting_meta['Year'].values,                  # Year
        np.ones(len(planting_meta))                    # Constant feature
    ])
    ranch_ids = planting_meta['Ranch'].map(mapping_dict['Ranch']).values
    class_ids = planting_meta['Class'].map(mapping_dict['Class']).values
    type_ids = planting_meta['Type'].map(mapping_dict['Type']).values
    variety_ids = planting_meta['Variety'].map(mapping_dict['Variety']).values

    climate_data = np.stack(planting_meta['ClimateSeries'].values)

    kilos = weekly_summary.loc[planting_meta.index].pivot(columns='WeeksAfterTransplant',  values='Kilos').fillna(0)
    kilos = kilos.reindex(columns=range(1, 21), fill_value=0)

    dataset = HarvestDataset(
        features=features,
        ranch_ids=ranch_ids,
        class_ids=class_ids,
        type_ids=type_ids,
        variety_ids=variety_ids,
        climate_data=climate_data,
        Y_kilos=  kilos.values 
    )
    return dataset


def decode(dataset, mapping_dict):
    """
    desc: Decodes data from encoded format into readable format

    params:
        dataset: HarvestDataset object containing encoded data
        mapping_dict: Dictionary mapping encoded IDs back to original labels

    returns: DataFrame with decoded categorical variables and features
    """
    # Create reverse mappings
    reverse_mappings = {
        key: {idx: label for label, idx in mapping.items()}
        for key, mapping in mapping_dict.items()
    }

    # Convert tensors to numpy arrays if needed
    features = dataset.features.numpy() if torch.is_tensor(dataset.features) else dataset.features
    ranch_ids = dataset.ranch_ids.numpy() if torch.is_tensor(dataset.ranch_ids) else dataset.ranch_ids
    class_ids = dataset.class_ids.numpy() if torch.is_tensor(dataset.class_ids) else dataset.class_ids
    type_ids = dataset.type_ids.numpy() if torch.is_tensor(dataset.type_ids) else dataset.type_ids
    variety_ids = dataset.variety_ids.numpy() if torch.is_tensor(dataset.variety_ids) else dataset.variety_ids

    # Create DataFrame
    df = pd.DataFrame({
        'Ha': features[:, 0],
        'WeekTransplanted_sin': features[:, 1],
        'WeekTransplanted_cos': features[:, 2],
        'Year': features[:, 3].astype(int),
        'Ranch': [reverse_mappings['Ranch'][id] for id in ranch_ids],
        'Class': [reverse_mappings['Class'][id] for id in class_ids],
        'Type': [reverse_mappings['Type'][id] for id in type_ids],
        'Variety': [reverse_mappings['Variety'][id] for id in variety_ids]
    })
    df['WeekTransplanted'] = df.apply(
        lambda row: reverse_sin_cos_week(
            row['WeekTransplanted_sin'], 
            row['WeekTransplanted_cos'], 
            row['Year']), 
            axis=1
        ).astype(int)

    return df
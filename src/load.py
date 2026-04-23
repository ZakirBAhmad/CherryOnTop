import numpy as np
import pandas as pd
import app.paths as paths
import src.table as table
import json

def load_preds():
    final_preds = np.load(paths.PREDS / 'final_preds.npz')['arr_0']
    return final_preds

def load_actuals():
    meta = pd.read_csv(paths.PREDS / 'meta.csv')
    y = np.loadtxt(paths.PREDS / 'y.csv', delimiter=',').astype(np.int32)
    return meta, y

def load_mappings():
    with open(paths.PREDS / 'mappings.json', 'r') as f:
        mappings = json.load(f)
    return mappings

def load_historical_data():
    meta = pd.read_csv(paths.HISTORICAL / 'meta.csv')
    y = np.loadtxt(paths.HISTORICAL / 'y.csv', delimiter=',').astype(np.int32)
    t_weeks = meta.transplant_week.values
    idx_dict = create_idx_dict(meta)
    total_kg = y.sum(axis=1)
    meta['total_kg'] = total_kg
    return meta, y, t_weeks, idx_dict, total_kg

def collapse_table(table,idx_dict,col = 'class'):
    return np.stack(
        [
            table[idx_dict[col][c]].sum(axis=0) for c in idx_dict[col]
        ])

def create_idx_dict(production_plan):
    indices_dict = {'class':{},'ranch':{},'transplant_week':{}}
    # Populate indices_dict with the indices of production_plan for Class, Ranch, Type
    for column in ['class', 'ranch', 'transplant_week','year']:
        if column in production_plan.columns:
            unique_values = production_plan[column].unique()
            indices_dict[column] = {value: production_plan[column] == value for value in unique_values}
    return indices_dict

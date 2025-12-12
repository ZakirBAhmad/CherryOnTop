import numpy as np
import pandas as pd
import control.paths as paths

def load_init_preds():
    init_kg = np.loadtxt(paths.PREDS / 'initial_kilos_cleaned.csv', delimiter=',')
    init_sched = np.loadtxt(paths.PREDS / 'initial_scheds_cleaned.csv', delimiter=',').astype(np.int32)
    return init_kg, init_sched

def load_actuals():
    actual_kg = np.loadtxt(paths.PREDS / 'y.csv', delimiter=',').astype(np.int32)
    actual_sched = np.loadtxt(paths.PREDS / 'sched.csv', delimiter=',', skiprows=1).astype(np.int32)
    meta = pd.read_csv(paths.PREDS / 'meta.csv')
    return actual_kg, actual_sched, meta

def load_adj_preds():
    adj_kg = np.load(paths.PREDS / 'kilo_reg_preds.npz')
    adj_sched = np.load(paths.PREDS / 'sched_reg_preds.npz')
    adj_kg = {int(key): adj_kg[key].astype(np.int32) for key in adj_kg.keys()}
    adj_sched = {int(key): adj_sched[key].astype(np.int32) for key in adj_sched.keys()}
    return adj_kg, adj_sched

def get_season_init_preds(init_kg, init_sched,transplant_weeks):
    season_init_sched = init_sched.copy()
    season_init_sched[:,1:] = init_sched[:,1:] + transplant_weeks[:,None]

    season_init_kg = season_shift(transplant_weeks,init_kg)
    return season_init_sched, season_init_kg

def get_season_actuals(actual_kg, actual_sched,transplant_weeks):
    season_actual_sched = actual_sched.copy()
    season_actual_sched[:,1:] = actual_sched[:,1:] + transplant_weeks[:,None]
    season_actual_kg = season_shift(transplant_weeks,actual_kg)
    return season_actual_sched, season_actual_kg

def get_season_adj_kg(season_init_kg,adj_kg,transplant_weeks):

    max_shift = transplant_weeks.max()
    season_adj_kg = np.repeat(season_init_kg[None, ...], max_shift + 20, axis=0).transpose(1,0,2)
    for i, season_week in enumerate(transplant_weeks):
        for wat in range(5,20):
            week = season_week + wat
            season_adj_kg[i,week,season_week:season_week+40] = adj_kg[wat][i]
        
        season_adj_kg[i,week+1:,season_week:season_week+40] = adj_kg[20][i]

    return season_adj_kg.astype(np.int32)

def get_season_adj_sched(season_init_sched,adj_sched,transplant_weeks):
    max_shift = transplant_weeks.max()
    season_adj_sched = np.repeat(season_init_sched[None, ...], max_shift + 20, axis=0).transpose(1,0,2)
    for i, season_week in enumerate(transplant_weeks):
        for wat in range(5,20):
            week = season_week + wat
            row = adj_sched[wat][i]
            row[1:] = row[1:] + season_week
            season_adj_sched[i,week] = row
        row = adj_sched[20][i]
        row[1:] = row[1:] + season_week
        season_adj_sched[i,week+1:] = row

    return season_adj_sched.astype(np.int32)


def season_shift(transplant_week, table):
    """
    Shift N x Output Dim table to be Nx SeasonLength
    """
    max_shift = transplant_week.max()
    N,O = table.shape
    out_shape = (N, O + max_shift)
    out = np.zeros(out_shape, dtype=table.dtype)
    indices = (np.arange(N)[:, None], np.arange(O)[None, :] + transplant_week[:, None])
    np.add.at(out, indices, table)
    return out



def collapse_table(table,idx_dict,col = 'class'):
    return np.stack(
        [
            table[idx_dict[col][c]].sum(axis=0) for c in idx_dict[col]
        ])

def create_idx_dict(production_plan):
    indices_dict = {'class':{},'ranch':{},'transplant_week':{}}
    # Populate indices_dict with the indices of production_plan for Class, Ranch, Type
    for column in ['class', 'ranch', 'transplant_week']:
        if column in production_plan.columns:
            unique_values = production_plan[column].unique()
            indices_dict[column] = {value: production_plan[column] == value for value in unique_values}
    return indices_dict

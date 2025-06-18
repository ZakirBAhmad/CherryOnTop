import pandas as pd
import numpy as np

def create_indices_dict(production_plan):
    indices_dict = {'Class':{},'Type':{},'Ranch':{},'Variety':{}}
    # Populate indices_dict with the indices of production_plan for Class, Ranch, Type
    for column in ['Class', 'Ranch', 'Type', 'Variety']:
        if column in production_plan.columns:
            unique_values = production_plan[column].unique()
            indices_dict[column] = {value: production_plan.index[production_plan[column] == value].tolist() for value in unique_values}
    return indices_dict

def get_indices(column,vals,indices_dict):
    return np.unique(np.concatenate([indices_dict[column][val] for val in vals]))

def filter_preds(preds, actuals, column, vals,indices_dict):
    types_idx = [indices_dict[column][t]for t in vals]
    preds_filtered = np.array([preds[idx].sum(axis=0) for idx in types_idx])
    actuals_filtered = np.array([actuals[idx].sum(axis=0) for idx in types_idx])
    return preds_filtered, actuals_filtered

def season_shift(transplant_weeks, preds, actuals):
    N, M, K = preds.shape
    max_shift = transplant_weeks.max()
    out_preds = np.zeros((N, M, K + max_shift), dtype=preds.dtype)
    out_actuals = np.zeros((N, K + max_shift), dtype=actuals.dtype)

    # Create indices for preds
    batch_idx_preds = np.arange(N)[:, None, None]      # (N, 1, 1)
    row_idx_preds = np.arange(M)[None, :, None]        # (1, M, 1)
    col_idx_preds = np.arange(K)[None, None, :]        # (1, 1, K)

    # Compute shifted column index for preds
    shifted_col_idx_preds = col_idx_preds + transplant_weeks[:, None, None]  # (N, 1, K)

    # Use np.add.at to safely add values at shifted positions for preds
    np.add.at(out_preds, (batch_idx_preds, row_idx_preds, shifted_col_idx_preds), preds)

    # Create indices for actuals
    batch_idx_actuals = np.arange(N)[:, None]          # (N, 1)
    col_idx_actuals = np.arange(K)[None, :]            # (1, K)

    # Compute shifted column index for actuals
    shifted_col_idx_actuals = col_idx_actuals + transplant_weeks[:, None]  # (N, K)

    # Use np.add.at to safely add values at shifted positions for actuals
    np.add.at(out_actuals, (batch_idx_actuals, shifted_col_idx_actuals), actuals)

    return out_preds, out_actuals


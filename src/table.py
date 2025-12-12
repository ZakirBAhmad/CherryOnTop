import numpy as np
import pandas as pd
def collapse_table(table,idx_dict):
    return np.stack(
        [
            table[idx_dict['class'][c]].sum(axis=0) for c in idx_dict['class']
        ]
    )

def season_shift(table,transplant_weeks):
    """
    Shift the actuals based on the transplant days.
    """
    max_shift = transplant_weeks.max()
    N,O = table.shape
    out_shape = (N, O + max_shift)
    out = np.zeros(out_shape, dtype=table.dtype)
    indices = (np.arange(N)[:, None], np.arange(O)[None, :] + transplant_weeks
    [:, None])
    np.add.at(out, indices, table)
    return out

def week_class_table(this_week,c,szn_adj_kg,szn_act_kg,idx_dict):
    """
    Shows the predictions based on weeks transplanted.

    Args:
    this_week [int]: week of predictions
    c [str]: class used
    szn_adj_kg [3d np array]:
    szn_act_kg [2d np array]:
    idx_dict [dictionary]: 

    Returns:
    preds_by_week [2d np array]: predictions by week transplanted
    kg_so_far [1d np array]: totaled actuals through [this_week]

    """
    class_idx = idx_dict['class'][c]
    class_and_week = {week:np.logical_and(idx_dict['transplant_week'][week],class_idx) for week in idx_dict['transplant_week']}

    table = szn_adj_kg[:,this_week]

    preds_by_week = np.stack([
            table[class_and_week[week]].sum(axis=0) for week in idx_dict['transplant_week']
        ])

    kg_so_far = np.array([szn_act_kg[class_and_week[week]][:,:this_week].sum() for week in idx_dict['transplant_week']])
    
    return preds_by_week, kg_so_far

def frame_table(preds_by_week, kg_so_far, idx_dict, this_week, window = 5):
    """
    Frames the table for the given week.
    """
    df = pd.DataFrame(preds_by_week, index = idx_dict['transplant_week'].keys())
    df1 = df.iloc[:,this_week-window: this_week+window].copy()
    df1['KgSoFar'] = kg_so_far 
    df1['Total_Projected'] = preds_by_week.sum(axis=1)
    df1.loc['Total'] = df1.sum(axis=0)


    df1['Percent'] = df1['KgSoFar'] / (df1['Total_Projected'] + 1e-8)
    return df1
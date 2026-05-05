import pandas as pd
import src.table as table


def prediction_history(final_preds, szn_act, idx_dict,tomato_class):
    df = table.class_prediction_history(final_preds, szn_act, tomato_class, idx_dict)

    return df.fillna(' ')

    
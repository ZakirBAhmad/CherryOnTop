import os
import torch
import numpy as np
import pandas as pd
from src.model import DistModel,ScheduleModel,KiloModel

def load_models(folder_path):
    models = {}
    for dir in os.listdir(folder_path):
        sub_dir = os.path.join(folder_path, dir)
        models[dir] = {}
        for file in os.listdir(sub_dir):
            if 'dist' in file:
                models[dir][file[:-3]] = DistModel()
            elif 'kilo' in file:
                models[dir][file[:-3]] = KiloModel()
            elif 'sched' in file:
                models[dir][file[:-3]] = ScheduleModel()
            else:
                continue
            models[dir][file[:-3]].load_state_dict(torch.load(os.path.join(sub_dir, file)))
            models[dir][file[:-3]].eval()
    return models

def create_week_preds(models, week, data):
    week_str = f"week_{week}"
    dist_model = models[week_str][f'dist_{week}']
    sched_model = models[week_str][f'sched_{week}']
    kilo_model = models[week_str][f'kilo_{week}']

    features, encoded_features, climate_data, kilo_gru_input, kilo_dist, log1p_kilos, log1p_schedule, Y_kilos, _= data
    kilo_gru_input = kilo_gru_input[:,:week,:]

    dist_output = dist_model(features, encoded_features, climate_data, kilo_gru_input).detach().numpy()
    sched_output = sched_model(features, encoded_features, climate_data, kilo_gru_input).detach().numpy()
    
    kilo_output = kilo_model(features, encoded_features, climate_data, kilo_gru_input).squeeze(-1).detach().numpy()
    # Better clamping to prevent extreme values
    kilo_output = np.clip(kilo_output, 0, 12)
    
    
    predicted_kilos = (np.expm1(kilo_output)).reshape(-1, 1)
    kilo_dist = (dist_output * predicted_kilos) / dist_output.sum(axis=1, keepdims=True)
    kilo_preds = kilo_dist * predicted_kilos
    kilo_preds[:,:week] = Y_kilos[:,:week].detach().numpy()

    sched_preds = np.expm1(sched_output)



    return kilo_preds, sched_preds

def create_preds(models, current_week, week_transplanted, dataset,num_weeks = 40, sched_cols = ['iqr_weeks','weighted_mean_weeks','first_harvest','end_harvest','uncertainty'],kilo_cols = ['iqr_weeks','weighted_mean_weeks','first_harvest','end_harvest','uncertainty']):
    model_ids = np.clip(current_week - week_transplanted, 1, 20)
    model_id_idx = {i: np.where(model_ids == i)[0].tolist() for i in range(1, 21)}

    size = len(dataset)
    kilo_preds = pd.DataFrame(index = range(size), columns = range(1, num_weeks + 1), data = np.zeros((size, num_weeks)))
    sched_preds = pd.DataFrame(index = range(size), columns = sched_cols, data = np.zeros((size, len(sched_cols))))

    for week in model_id_idx:
        idxs = model_id_idx[week]
        kilo_dist, sched_output = create_week_preds(models, week, dataset[idxs])
        kilo_preds.iloc[idxs] = kilo_dist
        sched_preds.iloc[idxs] = sched_output

    return kilo_preds, sched_preds

def populate_preds(folder_path,models,week_transplanted, dataset):
    for week in range(1, 53):
        kilo_preds, sched_preds = create_preds(models, week, week_transplanted, dataset)

        os.makedirs(os.path.join(folder_path, f'week_{week}'), exist_ok=True)

        kilo_preds.to_csv(os.path.join(folder_path, f'week_{week}', f'kilo_preds_{week}.csv'))
        sched_preds.to_csv(os.path.join(folder_path, f'week_{week}', f'sched_preds_{week}.csv'))

def read_preds(folder_path,dataset,num_weeks = 40):
    week_list = os.listdir(folder_path)
    num_preds = len(week_list)
    num_plantings = len(dataset)

    agg_kilo_preds = np.zeros((num_plantings, num_preds,num_weeks))
    agg_sched_preds = np.zeros((num_plantings, num_preds, 5))

    for week in week_list:
        week_path = os.path.join(folder_path, week)
        week_idx = int(week.split('_')[-1]) - 1
        for file in os.listdir(week_path):
            if 'kilo_preds' in file:
                preds = pd.read_csv(os.path.join(week_path, file),index_col=0).values
                agg_kilo_preds[:,week_idx,:] = preds
            elif 'sched_preds' in file:
                preds = pd.read_csv(os.path.join(week_path, file),index_col=0).values
                agg_sched_preds[:,week_idx,:] = preds
            else:
                print('eeker casillas')

    return agg_kilo_preds, agg_sched_preds

    

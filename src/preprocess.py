##### fill this in!!!!

import pandas as pd
import numpy as np
import os
from scipy.stats import zscore
import json
from sklearn.preprocessing import RobustScaler

def create_files(folder_path,normalize=True):

    tomato_path = folder_path + 'raw/tomato/'
    climate_path = folder_path + 'raw/climate/'
    
    meta, y = create_meta_y(tomato_path)
    df, columns, scaler = create_df(climate_path,normalize)

    meta['ClimateSeries'] = meta.apply(get_climate_series, axis=1, df=df, columns=columns)

    schedule = get_schedule(y)

    filtered_meta, filtered_y, filtered_schedule = filter_outliers(meta, y, schedule)

    return filtered_meta, filtered_y, df, filtered_schedule, scaler

def get_climate_series(row,df,columns):
    """returns the climate series from df_normalized for 100 days after transplant date. returns values from columns to be normalized"""
    transplant_date = row['TransplantDate']
    location = row['ProducerCode']
    
    # Filter the dataframe for the specific location
    location_data = df[df['Location'] == location]
    
    # Convert transplant_date to datetime
    transplant_date = pd.to_datetime(transplant_date)
    
    # Filter the data for 100 days after the transplant date
    climate_series = location_data[(location_data['date'] >= transplant_date) & 
                                   (location_data['date'] < transplant_date + pd.Timedelta(days=100))]
    
    # Return the values from columns to be normalized
    return climate_series[columns].values.tolist()

def filter_outliers(meta, y,schedule,z_bad = 3, uncertainty_bad = 1.5):
    outliers = meta.copy()
    outliers.loc[:,'Total_Kilos'] = y.sum(axis=1)
    outliers.loc[:,'Yield'] = outliers.loc[:,'Total_Kilos'] / outliers.loc[:,'Ha']
    outliers.loc[:,'FirstHarvest'] = schedule['first_harvest']
    outliers.loc[:,'Num_Harvests'] = [np.sum(row > 0) if np.any(row > 0) else np.nan for row in y]
    outliers.loc[:,'Coverage'] = outliers.loc[:,'Num_Harvests'] + outliers.loc[:,'FirstHarvest']
    outliers.loc[:,'EndHarvest'] = schedule['end_harvest']
    outliers.loc[:,'IQR_Weeks'] = schedule['iqr_weeks']
    outliers.loc[:,'Weighted_Mean_Weeks'] = schedule['weighted_mean_weeks']
    outliers.loc[:,'Harvest_Duration'] = outliers.loc[:,'EndHarvest'] - outliers.loc[:,'FirstHarvest']
    

    # Convert specified columns to z-scores
    columns_to_convert = ['Total_Kilos', 'Yield', 'FirstHarvest', 'Num_Harvests', 'Coverage', 'EndHarvest', 'Harvest_Duration','IQR_Weeks','Weighted_Mean_Weeks']
    outliers[columns_to_convert] = outliers[columns_to_convert].apply(zscore)
    schedule['uncertainty'] = np.sqrt((outliers[columns_to_convert] ** 2).mean(axis=1))
    # Make the columns 1 if they are outside of [-3, 3]
    outliers[columns_to_convert] = outliers[columns_to_convert].apply(lambda x: (x < -z_bad) | (x > z_bad)).astype(int)
    outliers['outliers'] = outliers[columns_to_convert].sum(axis=1)

    # Create a column called 'outliers' that is the sum of those columns
    
    z_goods = outliers.outliers == 0
    uncertain_goods = schedule.uncertainty < uncertainty_bad
    goods = z_goods & uncertain_goods
    
    return meta[goods], y[goods], schedule[goods]

def get_schedule(y):
    schedule = pd.DataFrame(columns=['iqr_weeks','weighted_mean_weeks','first_harvest','end_harvest','uncertainty'])
    cumsum = np.cumsum(y, axis=1)
    total = y.sum(axis=1, keepdims=True)
    cumulative_percent = cumsum / total  # shape: (n, 40)

    # Function to find first index where condition is met per row
    def first_ge_threshold(arr, threshold):
        return (arr >= threshold).argmax(axis=1)

    week_25 = first_ge_threshold(cumulative_percent, 0.25)
    week_75 = first_ge_threshold(cumulative_percent, 0.75)

    iqr_weeks = week_75 - week_25  # shape: (n,)
    iqr_weeks = pd.Series(iqr_weeks, name="IQR_Weeks")

    weeks = np.arange(1, y.shape[1]+1)
    numerators = (y * weeks).sum(axis=1)
    denominators = y.sum(axis=1)

    weighted_mean_weeks = pd.Series(numerators / denominators, name="Weighted_Mean_Weeks")

    first_harvest = pd.Series([np.argmax(row > 0) if np.any(row > 0) else np.nan for row in y],name="FirstHarvest")

    end_harvest = pd.Series(([np.max(np.nonzero(row)[0]) if np.any(row > 0) else np.nan for row in y]),name = 'EndHarvest')

    schedule['iqr_weeks'] = iqr_weeks
    schedule['weighted_mean_weeks'] = weighted_mean_weeks
    schedule['first_harvest'] = first_harvest
    schedule['end_harvest'] = end_harvest

    return schedule

def create_df(climate_path,normalize=True):
    climate_file_names = os.listdir(climate_path)

    dfs = []
    for file in climate_file_names:
        loc = file.split('_')[2][:-4]
        df = pd.read_csv(os.path.join(climate_path, file))
        df['Location'] = loc
        dfs.append(df)

    df = pd.concat(dfs).drop(columns=['Unnamed: 0','0','weather_code'])
    df.date = pd.to_datetime(df.date).dt.tz_localize(None)
    columns = [
        'temperature_2m_max',
        'temperature_2m_min',
        'precipitation_sum',
        'rain_sum',
        'shortwave_radiation_sum',
        'et0_fao_evapotranspiration',
        'sunshine_duration',
        'precipitation_hours',
        'wind_speed_10m_max'
    ]
    if normalize:
        scaler = RobustScaler()
        df[columns] = scaler.fit_transform(df[columns])
    else:
        scaler = None

    return df, columns, scaler

def create_meta_y(tomato_path):
    production_reports = []
    for file in os.listdir(tomato_path):
        if file.endswith('.csv'):
            df = pd.read_csv(tomato_path + file,encoding = 'latin1')
            production_reports.append(df)

    production_reports = pd.concat(production_reports,axis=0)

    production_reports.columns = ['TransplantDate', 'Fecha TransplanteTxt', 'AÃ±oSiembra', 'ProducerCode',
    'Productor.1', 'Ha', 'DateRecieved', 'FechaRecibo txt', 'AÃ±o Recibo',
    'WeekRecieved', 'Kilos', 'Class', 'Type', 'Variety', 'Parcel',
    'Lot', 'Brix', 'AñoSiembra', 'Año Recibo']

    df = production_reports[['TransplantDate','ProducerCode','Ha','DateRecieved','WeekRecieved','Kilos','Class','Type','Variety','Parcel','Lot','Brix']]
    df['Type'] = df['Type'].str.rstrip()
    df['Parcel'] = df['Parcel'].str.upper()
    df['Parcel'] = df['Parcel'].str.replace(' ','')
    df['Type'] = df['Type'].str.title()
    df['Variety'] = df['Variety'].str.title()
    df['TransplantDate'] = pd.to_datetime(df['TransplantDate'], dayfirst=True)
    df['DateRecieved'] = pd.to_datetime(df['DateRecieved'], dayfirst=True)
    df['Variety'] = df['Variety'].replace({
    'Amaã\xad': 'Amaí',                    # Fix encoding issue
    'Beby Black Plum': 'Baby Black Plum', # Fix typo
    'Top-2204': 'Top 2204',               # Standardize format
    'Top-2245': 'Top 2245'})                # Standardize format

    df = df[df['Type'] != 'Pruebas']
    df['WeekTransplanted'] = df['TransplantDate'].dt.isocalendar().week
    df['WeekAfterTransplant'] = (df['WeekRecieved'].astype(int) - df['WeekTransplanted'].astype(int)) % 52
    
    meta = df.groupby(['TransplantDate','Parcel','Lot','ProducerCode','Variety','Ha']).agg({'Class': 'first', 'Type': 'first','WeekTransplanted':'first'}).reset_index()
    meta['WeekCos'] = np.cos(2 * np.pi * meta['WeekTransplanted'] / 52)
    meta['WeekSin'] = np.sin(2 * np.pi * meta['WeekTransplanted'] / 52)
    meta['Year'] = meta['TransplantDate'].dt.year

    pivoted = df.pivot_table(index=['TransplantDate', 'Parcel', 'Lot', 'ProducerCode', 'Variety','Ha'], columns='WeekAfterTransplant', values='Kilos', aggfunc='sum').fillna(0)
    y = pivoted.iloc[meta.index].values
    return meta, y


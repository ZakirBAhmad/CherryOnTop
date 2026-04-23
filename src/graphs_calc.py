import numpy as np
import pandas as pd
import plotly.colors as pc
import src.table as table

def get_gradient(name,n):
    for module in (pc.sequential, pc.diverging, pc.cyclical):
        if hasattr(module, name):
            color_scale = getattr(module, name)
            indices = np.linspace(0, len(color_scale)-1, n, dtype=int)
            colors = [color_scale[i] for i in indices]
            return colors
    raise ValueError(f"Colorscale '{name}' not found in Plotly colors.")

def sliding_transplant(year,tomato_class,meta, y, idx_dict):
    idx = idx_dict['year'][year] & idx_dict['class'][tomato_class]
    y = y[idx]
    meta = meta[idx]
    transplant_weeks = meta['transplant_week'].values

    szn_kg = table.shift_actuals(y,transplant_weeks)
    total_kg = szn_kg.sum(axis=0) #season shifted y
    szn_len = len(total_kg)
    szn_x = np.arange(szn_len)

    df = pd.DataFrame(szn_kg)
    df['transplant_week'] = meta.transplant_week.values
    kg_by_transplant_week = df.groupby('transplant_week').sum()

    by_t_week = meta.groupby('transplant_week').agg({'ha':'sum','total_kg':'sum'})
    by_t_week['yield'] = by_t_week['total_kg'] / by_t_week['ha']
    return {
        'total_kg': total_kg,
        'szn_x': szn_x,
        'initial_transplant_y': kg_by_transplant_week.iloc[0],
        'szn_transplant_weeks': by_t_week.index.values,
        'szn_ha': by_t_week['ha'].values,
        'szn_yield': by_t_week['yield'].values
    }, kg_by_transplant_week, by_t_week

def sliding_transplant_projected(final_preds,tomato_class,idx_dict,szn_act):
    idx = idx_dict['class'][tomato_class]
    class_preds = final_preds[idx]
    data = {}
    lower_preds = np.nansum(class_preds[...,0],axis=0)
    lower_mid_preds = np.nansum(class_preds[...,1],axis=0)
    mean_preds = np.nansum(class_preds[...,2],axis=0)
    upper_mid_preds = np.nansum(class_preds[...,3],axis=0)
    upper_preds = np.nansum(class_preds[...,4],axis=0)

    O = mean_preds.shape[-1]

    data['mean_preds'] = mean_preds
    data['actuals'] = szn_act[idx].sum(axis=0)
    data['szn_x'] = np.arange(O)
    data['in_CI'] = np.concatenate([upper_mid_preds, lower_mid_preds[...,::-1]],axis=1)
    data['out_CI'] = np.concatenate([upper_preds, lower_preds[...,::-1]],axis=1)
    data['interval_x'] = np.concatenate([data['szn_x'], data['szn_x'][::-1]])

    return data

def kg_ha(meta, tomato_class, idx_dict):
    data = {}
    idx = idx_dict['class'][tomato_class]
    data['meta'] = meta[idx]

    data['total_yield'] = data['meta']['total_kg'].sum() / data['meta']['ha'].sum()
    data['max_kg'] = data['meta']['total_kg'].max()
    data['max_line_ha'] = data['max_kg'] / data['total_yield']

    data['years'] = np.sort(np.unique(data['meta'].year))
    data['num_years'] = len(data['years'])
    data['colors'] = get_gradient('YlGn', data['num_years'])

    return data

def kg_ha_projected(meta, tomato_class, idx_dict, final_preds, szn_act):
    idx = idx_dict['class'][tomato_class]
    data = {}
    data['preds'] = np.nansum(final_preds[...,2],axis=2)[idx]
    data['actuals'] = szn_act[idx]
    data['ha'] = meta['ha'][idx].values
    return data

def transplant_week_yield(meta, tomato_class, idx_dict):
    idx = idx_dict['class'][tomato_class]
    df = meta[idx].copy()
    df['yield'] = df['total_kg'] / df['ha']
    df['x'] = df['transplant_week'] + np.random.uniform(-0.2, 0.2, size=len(df))
    return df

def transplant_week_yield_projected(meta,tomato_class, idx_dict, final_preds, szn_act):
    idx = idx_dict['class'][tomato_class]
    df = meta[idx].copy()
    df['x1'] = df['transplant_week'] + np.random.uniform(-0.2, 0, size=len(df))
    df['x2'] = df['transplant_week'] + np.random.uniform(0, 0.2, size=len(df))
    preds = np.nansum(final_preds[...,2],axis=2)[idx]
    actuals = szn_act[idx]
    return df, preds, actuals

def ranch_yield(meta, tomato_class, idx_dict,mappings):
    idx = idx_dict['class'][tomato_class]
    df = meta[idx].copy()
    df['yield'] = df['total_kg'] / df['ha']
    df['x'] = df['ranch'].map(mappings['ranch']) + np.random.uniform(-0.2, 0.2, size=len(df))
    return df

def ranch_yield_projected(meta, tomato_class, idx_dict,final_preds, szn_act, mappings):
    idx = idx_dict['class'][tomato_class]
    df = meta[idx].copy()
    df['x1'] = df['ranch'].map(mappings['ranch']) + np.random.uniform(-0.2, 0, size=len(df))
    df['x2'] = df['ranch'].map(mappings['ranch']) + np.random.uniform(0, 0.2, size=len(df))
    preds = np.nansum(final_preds[...,2],axis=2)[idx]
    actuals = szn_act[idx]
    return df, preds, actuals

def year_yield(h_meta, tomato_class, h_idx_dict,new_meta, new_idx_dict,final_preds, szn_act):
    h_idx = h_idx_dict['class'][tomato_class]
    new_idx = new_idx_dict['class'][tomato_class]
    df = h_meta[h_idx].copy()
    df['yield'] = df['total_kg'] / df['ha']
    df['x'] = df['year'] + np.random.uniform(-0.2, 0.2, size=len(df))

    df_by_year = df.groupby('year').agg({'total_kg': 'sum', 'ha': 'sum'})
    df_by_year['yield'] = df_by_year['total_kg'] / df_by_year['ha']

    historical_yield = df_by_year['total_kg'].sum() / df_by_year['ha'].sum()

    preds = np.nansum(final_preds[...,2],axis=2)[new_idx]
    actuals = szn_act[new_idx]

    df1 = new_meta[new_idx].copy()
    df1['x1'] = df1['year'] + np.random.uniform(-0.2, 0, size=len(df1))
    df1['x2'] = df1['year'] + np.random.uniform(0, 0.2, size=len(df1))
    return df, df_by_year, preds, actuals, df1, historical_yield
    

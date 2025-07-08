#imports
import streamlit as st
import src.load as load
import pandas as pd
import src.utils as utils
import torch
from src.model import KiloModel,ScheduleModel
import src.table as table
import src.graphs as graphs
from src.encoder import ClimateEncoder
import plotly.graph_objects as go
import numpy as np

@st.cache_resource
def initialize_data(path_meta, path_y, path_mapping_dict):
    train, test, mappings, reverse_mappings, meta  = load.separate_prop('../data/processed/')
    return train, test, mappings, reverse_mappings, meta

def display_production_plan(plan):
    st.write('Production Plan:')
    agg_cols = {'Ha':'sum',
     'WeekTransplanted':['min','max','nunique']}

    summary_ranch = plan.groupby(['Ranch','Class','Type']).agg(agg_cols)
    summary_class = plan.groupby(['Class','Type']).agg(agg_cols)
    summary_class_ranch = plan.groupby(['Class','Type','Ranch']).agg(agg_cols)

    st.dataframe(summary_ranch,use_container_width=True)
    st.dataframe(summary_class,use_container_width=True)
    st.dataframe(summary_class_ranch,use_container_width=True)

    ranch_pivot = plan.pivot_table(
    index=['Ranch', 'Class', 'Type'],
    columns='WeekTransplanted',
    values='Ha',
    aggfunc='sum',
    fill_value=0,
)

    class_pivot = plan.pivot_table(
        index=['Class', 'Type'],
        columns='WeekTransplanted',
        values='Ha',
        aggfunc='sum',
        fill_value=0,
    )

    

    data = []
    for class_type in class_pivot.index:
        data.append(go.Bar(
            name=f'{class_type}',
            x=class_pivot.columns,
            y=class_pivot.loc[class_type]
        ))

    st.write('Planting Schedule over the Season')    
    # Create the figure
    fig = go.Figure(data=data)

    # Update layout for the plot
    fig.update_layout(
        barmode='stack',
        title='Stacked Bar Plot of Hectares by Week Transplanted',
        xaxis_title='Week Transplanted',
        yaxis_title='Hectares (Ha)',
        legend_title='Class and Type',
        xaxis=dict(tickangle=45, categoryorder='total descending'),
        legend=dict(x=1.05, y=1)
    )
    st.write('Hectares by Class and Type')
    st.plotly_chart(fig)
    st.dataframe(class_pivot,use_container_width=True)

    ranch_pivot = plan.pivot_table(
    index='Ranch',
    columns='WeekTransplanted',
    values='Ha',
    aggfunc='sum',
    fill_value=0,
)

    # Prepare data for the stacked bar plot
    ranch_data = []
    for ranch in ranch_pivot.index:
        ranch_data.append(go.Bar(
            name=f'{ranch}',
            x=ranch_pivot.columns,
            y=ranch_pivot.loc[ranch]
        ))

    # Create the figure
    ranch_fig = go.Figure(data=ranch_data)

    # Update layout for the plot
    ranch_fig.update_layout(
        barmode='stack',
        title='Stacked Bar Plot of Hectares by Week Transplanted (Ranch)',
        xaxis_title='Week Transplanted',
        yaxis_title='Hectares (Ha)',
        legend_title='Ranch',
        xaxis=dict(tickangle=45, categoryorder='total descending'),
        legend=dict(x=1.05, y=1)
    )
    st.write('Hectares by Ranch')
    st.plotly_chart(ranch_fig)
    st.dataframe(ranch_pivot,use_container_width=True)

@st.cache_resource
def create_models(_kilo_model_path, _schedule_model_path, _final_model_path):
    kilo_encoder = ClimateEncoder()
    schedule_encoder = ClimateEncoder()
    kilo_model = KiloModel(kilo_encoder)
    kilo_model.load_state_dict(torch.load(_kilo_model_path))
    kilo_model.eval()

    schedule_model = ScheduleModel(schedule_encoder)
    schedule_model.load_state_dict(torch.load(_schedule_model_path))
    schedule_model.eval()

    return kilo_model, schedule_model

def create_predictions(kilo_model, schedule_model, test):
    preds = utils.predict_gridded_harvest(kilo_model,schedule_model,test)
    actuals = test.Y.detach().numpy()
    return preds, actuals
    
def create_harvest_curves(preds,actuals,plan):
    graph_dict = {}
    idx_dict = table.create_indices_dict(plan)
    transplant_weeks = plan.WeekTransplanted.to_numpy()
    year_preds, year_actuals = table.season_shift(transplant_weeks,preds,actuals)

    graph_dict['SeasonCurve'] = graphs.graph_harvest(year_preds,year_actuals)

    graph_dict['HarvestCycle'] = graphs.graph_harvest(preds,actuals)
    graph_dict['CumulativeSum'] = graphs.graph_harvest_cumsum(preds,actuals)


    summary_class = plan.groupby(['Class','Type']).agg(
    {'Ha':'sum',
     'WeekTransplanted':['min','max','nunique'],
     'Ranch':'nunique'
})
    
    types = summary_class.index.get_level_values('Type')
    preds_all_types, actuals_all_types = table.filter_preds(preds,actuals,'Type',types,idx_dict)

    actuals_received = actuals_all_types.sum(axis=1)
    final_preds = preds_all_types.sum(axis=2)[:,-1]
    initial_preds = preds_all_types.sum(axis=2)[:,6]

    summary_class['KilosReceived'] = actuals_received
    summary_class['FinalPreds'] = final_preds
    summary_class['InitialPreds'] = initial_preds
    


    return graph_dict, summary_class,idx_dict


def graph_classes(classes, preds, actuals, idx_dict, plan):
    transplant_weeks = plan.WeekTransplanted.to_numpy()
    year_preds, year_actuals = table.season_shift(transplant_weeks,preds,actuals)

    
    classes_idx = table.get_indices('Class',classes,idx_dict)

    year_preds_by_class, year_actuals_by_class = table.filter_preds(year_preds,year_actuals,'Class',classes,idx_dict)
    preds_by_class, actuals_by_class = table.filter_preds(preds,actuals,'Class',classes,idx_dict)
    fig1 = graphs.graph_harvest_stacked(year_preds_by_class,year_actuals_by_class,classes)
    fig2 = graphs.graph_harvest_stacked(preds_by_class,actuals_by_class,classes)


    return fig1, fig2

def graph_types(types, preds, actuals, idx_dict, plan):
    transplant_weeks = plan.WeekTransplanted.to_numpy()
    year_preds, year_actuals = table.season_shift(transplant_weeks,preds,actuals)

    
    types_idx = table.get_indices('Type',types,idx_dict)

    year_preds_by_type, year_actuals_by_type = table.filter_preds(year_preds,year_actuals,'Type',types,idx_dict)
    preds_by_type, actuals_by_type = table.filter_preds(preds,actuals,'Type',types,idx_dict)
    fig1 = graphs.graph_harvest_stacked(year_preds_by_type,year_actuals_by_type,types)
    fig2 = graphs.graph_harvest_stacked(preds_by_type,actuals_by_type,types)


    return fig1, fig2

def graph_ranches(ranches, preds, actuals, idx_dict, plan):
    transplant_weeks = plan.WeekTransplanted.to_numpy()
    year_preds, year_actuals = table.season_shift(transplant_weeks,preds,actuals)

    
    ranch_idx = table.get_indices('Ranch',ranches,idx_dict)

    year_preds_by_ranch, year_actuals_by_ranch = table.filter_preds(year_preds,year_actuals,'Ranch',ranches,idx_dict)
    preds_by_ranch, actuals_by_ranch = table.filter_preds(preds,actuals,'Ranch',ranches,idx_dict)
    fig1 = graphs.graph_harvest_stacked(year_preds_by_ranch,year_actuals_by_ranch,ranches)
    fig2 = graphs.graph_harvest_stacked(preds_by_ranch,actuals_by_ranch,ranches)


    return fig1, fig2
#imports
import streamlit as st
import src.preprocessing as pre
import pandas as pd
import src.utils as utils
from src.table import CherryTable
import plotly.graph_objects as go

@st.cache_resource
def initialize_data(path_meta, path_y, path_mapping_dict):
    train, test, mappings, meta = pre.separate_year(path_meta, path_y, path_mapping_dict)
    meta = pre.decode(meta,mappings)
    return train, test, mappings, meta

def display_production_plan(plan,preds=None):
    st.write('Production Plan:')
    agg_cols = {'Ha':'sum',
     'WeekTransplanted':['min','max','nunique']}

    if preds is not None:
        agg_cols['PredictedKilos'] = 'sum'
        plan['PredictedKilos'] = preds.sum(axis=1)


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
def create_model(_train_dataset,num_epochs=30):
    model = utils.create_model(_train_dataset,num_epochs=num_epochs)
    return model

def create_predictions(model,test,meta,name):
    preds = utils.predict_harvest(model,test)
    table = CherryTable(meta,{name:preds},test.Y.detach().numpy())
    return table
    


    
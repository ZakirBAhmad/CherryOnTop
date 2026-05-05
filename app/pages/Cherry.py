import streamlit as st
import pandas as pd
import src.graphs_new as graphs
import src.graph_update as update
import src.table as table
import src.graphs_calc as calc
import src.display_table as display_table
import src.update_table as update_table
st.set_page_config(layout="wide")

tomato_class = 'Cherry'    
st.title(tomato_class)

final_preds = st.session_state['current_data']['final_preds']
meta = st.session_state['current_data']['meta']
szn_act = st.session_state['current_data']['szn_act']
idx_dict = st.session_state['current_data']['idx_dict']

mappings = st.session_state['mappings']

historical_meta = st.session_state['historical_data']['meta']
historical_idx_dict = st.session_state['historical_data']['idx_dict']
historical_y = st.session_state['historical_data']['y']
historical_transplant_weeks = st.session_state['historical_data']['t_weeks']


if f'{tomato_class}' not in st.session_state:
    stp_data = calc.sliding_transplant_projected(final_preds,tomato_class,idx_dict,szn_act)
    stp_fig = graphs.sliding_transplant_projected(stp_data)

    df, df_by_year, preds, actuals, df1, historical_yield = calc.year_yield(historical_meta, tomato_class, historical_idx_dict,meta, idx_dict,final_preds, szn_act)
    yb_fig = graphs.yield_breakdown(df, df_by_year, preds, actuals,df1,historical_yield)

    kh_data = calc.kg_ha(historical_meta,tomato_class,historical_idx_dict)
    kh_fig = graphs.kg_ha(kh_data)

    khp_data = calc.kg_ha_projected(meta,tomato_class,idx_dict,final_preds,szn_act)
    khp_fig = graphs.kg_ha_projected(khp_data)

    twy_df = calc.transplant_week_yield(historical_meta,tomato_class,historical_idx_dict)
    twy_fig = graphs.transplant_week_yield(twy_df)

    twyp_df, twyp_preds, twyp_actuals = calc.transplant_week_yield_projected(meta,tomato_class,idx_dict,final_preds,szn_act)
    twyp_fig = graphs.transplant_week_yield_projected(twyp_df, twyp_preds, twyp_actuals)

    ry_df = calc.ranch_yield(historical_meta,tomato_class,historical_idx_dict,mappings)
    ry_fig = graphs.ranch_yield(ry_df,mappings)

    ryp_df, ryp_preds, ryp_actuals = calc.ranch_yield_projected(meta,tomato_class,idx_dict,final_preds,szn_act,mappings)
    ryp_fig = graphs.ranch_yield_projected(ryp_df,ryp_preds,ryp_actuals,mappings)
    year_figs = {}
    year_data = {}
    for year in range(2013, 2024):
        st_data, st_kg_by_transplant_week, st_by_t_week = calc.sliding_transplant(year,tomato_class,historical_meta, historical_y, historical_idx_dict)
        year_data[str(year)] = st_data,st_kg_by_transplant_week, st_by_t_week
        year_figs[str(year)] = graphs.sliding_transplant(st_data)

week = st.slider(
    "Select Transplant Week",
    min_value=1,
    max_value=53,
    value=1,
    step=1
)
update.sliding_transplant_projected(stp_fig,stp_data,week)
update.kg_ha_projected(khp_fig,khp_data,week)
update.transplant_week_yield_projected(twyp_fig,twyp_df,twyp_preds,twyp_actuals,week)
update.ranch_yield_projected(ryp_fig,ryp_df,ryp_preds,ryp_actuals,week)
update.yield_breakdown(yb_fig,df,df_by_year,preds,actuals,df1,historical_yield,week)
df = display_table.prediction_history(final_preds,szn_act,idx_dict,tomato_class)
styled_df = update_table.pred_history(df,week)
st.dataframe(styled_df) 
st.plotly_chart(stp_fig,key='stp_fig')

st.plotly_chart(twyp_fig,key='twyp_fig')
st.plotly_chart(twy_fig,key='twy_fig')

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(kh_fig, key='kh_fig')
    st.plotly_chart(ry_fig, key='ry_fig')

with col2:
    st.plotly_chart(khp_fig, key='khp_fig')
    st.plotly_chart(ryp_fig, key='ryp_fig')

st.plotly_chart(yb_fig,key='yb_fig')


selected_year = st.selectbox(
    'Select year',
    options=[str(year) for year in range(2013, 2022)],
    index=8  # Default to '2021'
)

st_data, st_kg_by_transplant_week, st_by_t_week = year_data[selected_year]

week = st.slider(
    "Select Transplant Week",
    min_value=int(st_by_t_week.index.values.min()),
    max_value=int(st_by_t_week.index.values.max()),
    value=int(st_by_t_week.index.values.min()),
    step=1
)
fig = year_figs[selected_year]
if week  in st_by_t_week.index.values:
    update.sliding_transplant(fig,st_by_t_week,st_kg_by_transplant_week,week)

st.plotly_chart(fig,key=f'st_fig_{selected_year}')
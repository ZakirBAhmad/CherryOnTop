import streamlit as st
import pandas as pd
import src.graphs as graphs
import src.table as table
st.set_page_config(layout="wide")

st.title("Historical")
st.write("Welcome to the historical page")

st.title("this season: 2024")
st.write("week slider")

meta = st.session_state['historical_data']['meta']
y = st.session_state['historical_data']['y']
t_weeks = st.session_state['historical_data']['t_weeks']
idx_dict = st.session_state['historical_data']['idx_dict']
total_kg = st.session_state['historical_data']['total_kg']

##### 2023 #####
st.title("2023")

szn_act = table.shift_actuals(y,t_weeks)
year = 2023
idx = idx_dict['year'][year]
fig2023 = graphs.season_act_graph(meta,szn_act,idx,'2023')

df = pd.DataFrame(szn_act[idx].sum(axis=0)[:53], columns=['kg'])
df.index.name = 'week of season'
ha_table = meta[idx].groupby('transplant_week').agg({'ha':'sum'}).T

st.dataframe(df.T.astype(int))
st.plotly_chart(fig2023,key='2023')
st.dataframe(ha_table)

##### 2022 #####
st.title("2022")

szn_act = table.shift_actuals(y,t_weeks)
year = 2022
idx = idx_dict['year'][year]
fig2022 = graphs.season_act_graph(meta,szn_act,idx,'2022')

df = pd.DataFrame(szn_act[idx].sum(axis=0)[:53], columns=['kg'])
df.index.name = 'week of season'
ha_table = meta[idx].groupby('transplant_week').agg({'ha':'sum'}).T

st.dataframe(df.T.astype(int))
st.plotly_chart(fig2022,key='2022')
st.dataframe(ha_table)

##### 2021 #####
st.title("2021")

szn_act = table.shift_actuals(y,t_weeks)
year = 2021
idx = idx_dict['year'][year]
fig2021 = graphs.season_act_graph(meta,szn_act,idx,'2021')

df = pd.DataFrame(szn_act[idx].sum(axis=0)[:53], columns=['kg'])
df.index.name = 'week of season'
ha_table = meta[idx].groupby('transplant_week').agg({'ha':'sum'}).T

st.dataframe(df.T.astype(int))
st.plotly_chart(fig2021,key='2021')
st.dataframe(ha_table)

##### 2020 #####
st.title("2020")

szn_act = table.shift_actuals(y,t_weeks)
year = 2020
idx = idx_dict['year'][year]
fig2020 = graphs.season_act_graph(meta,szn_act,idx,'2020')

df = pd.DataFrame(szn_act[idx].sum(axis=0)[:53], columns=['kg'])
df.index.name = 'week of season'
ha_table = meta[idx].groupby('transplant_week').agg({'ha':'sum'}).T

st.dataframe(df.T.astype(int))
st.plotly_chart(fig2020,key='2020')
st.dataframe(ha_table)


##### 2019 #####
st.title("2019")

szn_act = table.shift_actuals(y,t_weeks)
year = 2019
idx = idx_dict['year'][year]
fig2019 = graphs.season_act_graph(meta,szn_act,idx,'2019')

df = pd.DataFrame(szn_act[idx].sum(axis=0)[:53], columns=['kg'])
df.index.name = 'week of season'
ha_table = meta[idx].groupby('transplant_week').agg({'ha':'sum'}).T

st.dataframe(df.T.astype(int))
st.plotly_chart(fig2019,key='2019')
st.dataframe(ha_table)

##### 2018 #####
st.title("2018")

szn_act = table.shift_actuals(y,t_weeks)
year = 2018
idx = idx_dict['year'][year]
fig2018  = graphs.season_act_graph(meta,szn_act,idx,'2018')

df = pd.DataFrame(szn_act[idx].sum(axis=0)[:53], columns=['kg'])
df.index.name = 'week of season'
ha_table = meta[idx].groupby('transplant_week').agg({'ha':'sum'}).T

st.dataframe(df.T.astype(int))
st.plotly_chart(fig2018,key='2018')
st.dataframe(ha_table)

##### 2017 #####
st.title("2017")

szn_act = table.shift_actuals(y,t_weeks)
year = 2017
idx = idx_dict['year'][year]
fig2017 = graphs.season_act_graph(meta,szn_act,idx,'2017')

df = pd.DataFrame(szn_act[idx].sum(axis=0)[:53], columns=['kg'])
df.index.name = 'week of season'
ha_table = meta[idx].groupby('transplant_week').agg({'ha':'sum'}).T

st.dataframe(df.T.astype(int))
st.plotly_chart(fig2017,key='2017')
st.dataframe(ha_table)

##### 2016 #####
st.title("2016")

szn_act = table.shift_actuals(y,t_weeks)
year = 2016
idx = idx_dict['year'][year]
fig2016 = graphs.season_act_graph(meta,szn_act,idx,'2016')

df = pd.DataFrame(szn_act[idx].sum(axis=0)[:53], columns=['kg'])
df.index.name = 'week of season'
ha_table = meta[idx].groupby('transplant_week').agg({'ha':'sum'}).T

st.dataframe(df.T.astype(int))
st.plotly_chart(fig2016,key='2016')
st.dataframe(ha_table)

##### 2015 #####
st.title("2015")

szn_act = table.shift_actuals(y,t_weeks)
year = 2015
idx = idx_dict['year'][year]
fig2015 = graphs.season_act_graph(meta,szn_act,idx,'2015')

df = pd.DataFrame(szn_act[idx].sum(axis=0)[:53], columns=['kg'])
df.index.name = 'week of season'
ha_table = meta[idx].groupby('transplant_week').agg({'ha':'sum'}).T

st.dataframe(df.T.astype(int))
st.plotly_chart(fig2015,key='2015')
st.dataframe(ha_table)
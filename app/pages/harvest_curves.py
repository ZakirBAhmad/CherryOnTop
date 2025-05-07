#show the predicted harvest curve for each variety, then for each type, then each class, then each ranch.

import streamlit as st
import src.graphs as graphs


#have a filtered option in the streamlit

#show the graph of the filtered data predicted harvest curve

#show the actuals in a bar graph, x axis is weeks of the year or weeks after transplant, y axis is kilos

#show cumsum, cumprop, totals

table = st.session_state.filtered_data
a,d,ha_info = table.graph_ready(ranches=True,classes=True)
df = d['predictions_summed'].T

fig = graphs.graph_preds(df, "Harvest Predictions vs Actuals")

st.write(fig)
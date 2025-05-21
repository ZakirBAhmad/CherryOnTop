import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath('..'))

import pandas as pd
import numpy as np
from plotly import graph_objects as go
from plotly import colors
from plotly.colors import sample_colorscale

import streamlit as st
import demo


train, test = demo.initialize_datasets()
model = demo.train_model(train)

print('Model trained')

st.set_page_config(
    page_title="Harvest Forecasting",
    page_icon="🌱",
    layout="wide"
)

st.title("Harvest Forecasting")

page = st.sidebar.selectbox(
    "Select a page",
    ["Home", "Harvest Curves", "Harvest Summary", "Production Plan", "Harvest Actuals"]
)

if page == "Home":
    st.write("This is the home page.")
elif page == "Harvest Curves":
    st.write("Harvest curves visualization will be here.")
elif page == "Harvest Summary":
    st.write("Harvest summary will be here.")
elif page == "Production Plan":
    st.write("Production planning tools will be here.")
elif page == "Harvest Actuals":
    st.write("Harvest actuals data will be here.") 

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
from src.table import CherryTable
import src.utils as utils
import src.preprocessing as pre
import src.graphs as graphs

train, test, mapping_dict = pre.separate_year(planting_meta_path='../data/planting_meta.json', weekly_summary_path='../data/weekly_summary.csv')
model =utils.train_harvest_model(train)
predictions = utils.predict_harvest(model, test)

meta = pre.decode(test, mapping_dict)
actuals = test.Y_kilos.detach().numpy()

st.session_state.table = CherryTable(meta, {'predictions':predictions}, actuals)

st.set_page_config(
    page_title="Cherry On Top",
    page_icon="🍒",
    layout="wide"
)

st.title("Cherry On Top")
st.write("Welcome to the Cherry On Top application!")
st.write("This is a work in progress, please be patient.")

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Harvest Curves", "Harvest Summary", "Production Plan", "Harvest Actuals"])

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
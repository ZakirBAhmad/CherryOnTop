import streamlit as st
import sys
import os


st.set_page_config(layout="wide")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


###### Imports #####


###### list of things this will include #####
"""
- which types perform better at which ranches
- test batches
- trends in harvest
- % box yield
- historical yield, type breakdown
- small batches vs larger batches
- matching?
- plots over performing/underperforming
- temperature insights
- comparison to other years
"""

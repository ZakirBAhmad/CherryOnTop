import streamlit as st
import sys
import os

st.set_page_config(layout="wide")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

###### Imports #####
import app.demo as demo

###### Title #####
st.title('Cherry On Top')

###### Initialize Production Plan, mapping dict, model, and predictions#####
train, test, mappings, meta = demo.initialize_data('data/meta.json','data/y.csv','data/mappings.json')
model = demo.create_model('app/model1.pth')
preds, actuals = demo.create_predictions(model,test)

####### Display Production Plan #######

demo.display_production_plan(meta)

##### Save shit to session state #######

st.session_state['model'] = model
st.session_state['preds'] = preds
st.session_state['actuals'] = actuals
st.session_state['meta'] = meta
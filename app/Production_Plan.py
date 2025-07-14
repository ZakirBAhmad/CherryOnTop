import streamlit as st
import sys
import os

st.set_page_config(layout="wide")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

###### Imports #####
import app.demo as demo
import app.utils as utils

###### Title #####
st.title('Cherry On Top')
data_path = 'data/processed/'
model_path = 'app/models'
preds_path = 'app/preds'

###### Initialize Production Plan, mapping dict, model, and predictions#####
train_dataset, test_dataset, mappings, reverse_mappings, train_meta, test_meta  = demo.initialize_data(data_path)
demo.display_production_plan(test_meta)

models = utils.load_models(model_path)
preds_kilo, preds_sched = demo.create_predictions(preds_path,models,test_dataset)

####### Display Production Plan #######



##### Save shit to session state #######

st.session_state['models'] = models
st.session_state['preds_kilo'] = preds_kilo
st.session_state['preds_sched'] = preds_sched
st.session_state['train_meta'] = train_meta.reset_index(drop=True)
st.session_state['test_meta'] = test_meta.reset_index(drop=True)
st.session_state['mappings'] = mappings
st.session_state['reverse_mappings'] = reverse_mappings
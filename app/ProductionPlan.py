import streamlit as st
import sys
import os


st.set_page_config(layout="wide")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import app.demo as demo
import src.preprocessing as pre

st.title('Cherry On Top')


train, test, mappings, meta = demo.initialize_data('data/meta.json','data/y.csv','data/mappings.json')
production_plan_slot = st.empty()



model = demo.create_model(train)
st.write('Created model')
pred_table = demo.create_predictions(model,test,meta,'test')
st.write('Created predictions')

demo.display_production_plan(meta,pred_table.predictions['test'])
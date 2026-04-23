import src.load as load
import streamlit as st
import src.table as table
import src.graphs_new as graphs
import src.graphs_calc as calc

def load_data():
    batch_preds = load.load_preds()
    
    meta,y = load.load_actuals()
    idx_dict = load.create_idx_dict(meta)
    transplant_weeks = meta['transplant_week'].values
    final_preds = table.shift_preds(batch_preds,transplant_weeks)
    szn_act = table.shift_actuals(y,transplant_weeks)

    st.session_state['current_data'] = {
        'meta': meta,
        'y': y,
        'idx_dict': idx_dict,
        'final_preds': final_preds,
        'szn_act': szn_act}

def load_historical_data():
    meta, y, t_weeks, idx_dict, total_kg = load.load_historical_data()
    meta['yield'] = meta['total_kg'] / meta['ha']

    st.session_state['historical_data'] = {
        'meta': meta,
        'y': y,
        't_weeks': t_weeks,
        'idx_dict': idx_dict,
        'total_kg': total_kg
        }

def load_mappings():
    mappings = load.load_mappings()
    st.session_state['mappings'] = mappings

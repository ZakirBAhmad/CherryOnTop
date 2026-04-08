import src.load as load
import streamlit as st
import src.table as table

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
# def pred_evolution_graphs():
#     szn_adj_kg = st.session_state['data']['szn_adj_kg']
#     szn_act_kg = st.session_state['data']['szn_act_kg']
#     szn_init_kg = st.session_state['data']['szn_init_kg']
#     idx_dict = st.session_state['data']['idx_dict']

#     graphs = {str(week): graph.pred_evolution(week,szn_act_kg,szn_init_kg,szn_adj_kg,idx_dict) for week in range(52)}
#     return graphs

# def preds_by_week_tables():
#     szn_adj_kg = st.session_state['data']['szn_adj_kg']
#     szn_act_kg = st.session_state['data']['szn_act_kg']
#     idx_dict = st.session_state['data']['idx_dict']

#     tables = {}
#     for i in range(51):
#         this_week = i
#         row = {}
#         for c in idx_dict['class'].keys():
#             preds_by_week, kg_so_far = table.week_class_table(this_week,c,szn_adj_kg,szn_act_kg,idx_dict)
#             row[c] = preds_by_week, kg_so_far
#         tables[str(i)] = row
#     return tables
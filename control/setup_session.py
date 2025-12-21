import src.load as load
import streamlit as st
import src.graphs as graph
import src.table as table

def load_data():
    init_kg, init_sched = load.load_init_preds()
    act_kg, act_sched, meta = load.load_actuals()
    t_weeks = meta.transplant_week.values
    adj_kg, adj_sched = load.load_adj_preds()
    szn_init_sched, szn_init_kg = load.get_season_init_preds(init_kg, init_sched,t_weeks)
    szn_act_sched, szn_act_kg = load.get_season_actuals(act_kg, act_sched,t_weeks)
    szn_adj_kg = load.get_season_adj_kg(szn_init_kg,adj_kg,t_weeks)
    szn_adj_sched = load.get_season_adj_sched(szn_init_sched,adj_sched,t_weeks)
    idx_dict = load.create_idx_dict(meta)

    st.session_state['data'] = {
        'init_kg': init_kg,
        'init_sched': init_sched,
        'act_kg': act_kg,
        'act_sched': act_sched,
        'meta': meta,
        't_weeks': t_weeks,
        'adj_kg': adj_kg,
        'adj_sched': adj_sched,
        'szn_init_sched': szn_init_sched,
        'szn_init_kg': szn_init_kg,
        'szn_act_sched': szn_act_sched,
        'szn_act_kg': szn_act_kg,
        'szn_adj_kg': szn_adj_kg,
        'szn_adj_sched': szn_adj_sched,
        'idx_dict': idx_dict}

def pred_evolution_graphs():
    szn_adj_kg = st.session_state['data']['szn_adj_kg']
    szn_act_kg = st.session_state['data']['szn_act_kg']
    szn_init_kg = st.session_state['data']['szn_init_kg']
    idx_dict = st.session_state['data']['idx_dict']

    graphs = {str(week): graph.pred_evolution(week,szn_act_kg,szn_init_kg,szn_adj_kg,idx_dict) for week in range(52)}
    return graphs

def preds_by_week_tables():
    szn_adj_kg = st.session_state['data']['szn_adj_kg']
    szn_act_kg = st.session_state['data']['szn_act_kg']
    idx_dict = st.session_state['data']['idx_dict']

    tables = {}
    for i in range(51):
        this_week = i
        row = {}
        for c in idx_dict['class'].keys():
            preds_by_week, kg_so_far = table.week_class_table(this_week,c,szn_adj_kg,szn_act_kg,idx_dict)
            row[c] = preds_by_week, kg_so_far
        tables[str(i)] = row
    return tables
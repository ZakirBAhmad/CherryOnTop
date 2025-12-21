import streamlit as st
import src.table as table
st.set_page_config(layout="wide")

st.write("to do: prediction adjustment graphs for each week, mask for what batches are actually active. group table by week transplanted")
value = st.slider("Current Weeks", min_value=0, max_value=51, value=20)

st.title(f"Week {value}")
st.write("[this week/last week/next week/x weeks ago/x weeks from now]")
idx_dict = st.session_state['data']['idx_dict']
col1, col2, col3, col4 = st.columns(4)

with col1:
    week = value - 1
    st.write(f"{week} prediction history")
    graph = st.session_state['pe_graphs'][str(week)]
    st.plotly_chart(graph)
with col2:
    week = value
    st.write("this week prediction history")
    graph = st.session_state['pe_graphs'][str(value)]
    st.plotly_chart(graph)
with col3:
    week = value + 1
    st.write(f"{week} prediction history")
    graph = st.session_state['pe_graphs'][str(week)]
    st.plotly_chart(graph)
with col4:
    week = value + 2
    st.write(f"{week} prediction history")
    graph = st.session_state['pe_graphs'][str(week)]
    st.plotly_chart(graph)


st.title("breakdown by class")
row = st.session_state['pw_tables'][str(value)]


st.write("cherry")
preds_by_week, kg_so_far = row['Cherry']
df = table.frame_table(preds_by_week, kg_so_far, idx_dict, value)
# Highlight the "actual" column if present
if value in df.columns:
    df = df.style.highlight_between(subset=[value], left=None, right=None, color='yellow')
st.dataframe(df)


st.write("uva")
preds_by_week, kg_so_far = row['Uva']
df = table.frame_table(preds_by_week, kg_so_far, idx_dict, value)
st.dataframe(df)

st.write("mix")
preds_by_week, kg_so_far = row['Mix']
df = table.frame_table(preds_by_week, kg_so_far, idx_dict, value)
st.dataframe(df)


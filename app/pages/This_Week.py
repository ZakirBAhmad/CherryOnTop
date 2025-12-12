import streamlit as st
st.set_page_config(layout="wide")

st.write("to do: prediction adjustment graphs for each week, mask for what batches are actually active. group table by week transplanted")

st.write("Insert week Slider")
st.title("Week [insert week number]")
st.write("[this week/last week/next week/x weeks ago/x weeks from now]")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.write("last week prediction history")
with col2:
    st.write("this week prediction history")
with col3:
    st.write("next week prediction history")
with col4:
    st.write("2 weeks from now prediction history")
with col5:
    st.write("3 weeks from now prediction history")

st.title("breakdown by class")
col1, col2, col3= st.columns(3)
with col1:
    st.write("cherry")
    st.write("breakdown by ranch")
    st.write("breakdown by week transplanted")
    st.write("insert table, w/ kg/ha so far, total kg/ha projected. rows will be weeks transplant, columns will be weeks of the season")
with col2:
    st.write("uva")
    st.write("cherry")
    st.write("breakdown by ranch")
    st.write("breakdown by week transplanted")
    st.write("insert table, w/ kg/ha so far, total kg/ha projected. rows will be weeks transplant, columns will be weeks of the season")
with col3:
    st.write("mix")
    st.write("cherry")
    st.write("breakdown by ranch")
    st.write("breakdown by week transplanted")
    st.write("insert table, w/ kg/ha so far, total kg/ha projected. rows will be weeks transplant, columns will be weeks of the season")

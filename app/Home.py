import streamlit as st
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath('..'))

# Page configuration
st.set_page_config(
    page_title="Cherry Harvest Dashboard",
    page_icon="🍒",
    layout="wide"
)

# Title
st.title("🍒 Cherry Harvest Dashboard")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Production Plan", "Harvest Actuals", "Harvest Curves", "Harvest Summary"]
)

# Page routing
if page == "Home":
    st.write("Welcome to the Cherry Harvest Dashboard!")
    st.write("Use the sidebar to navigate between different sections.")
    st.write("This dashboard helps you analyze and predict cherry harvest data.")
    
elif page == "Production Plan":
    from pages.production_plan import show_production_plan
    show_production_plan()
    
elif page == "Harvest Actuals":
    from pages.harvest_actuals import show_harvest_actuals
    show_harvest_actuals()
    
elif page == "Harvest Curves":
    from pages.harvest_curves import show_harvest_curves
    show_harvest_curves()
    
elif page == "Harvest Summary":
    from pages.harvest_summary import show_harvest_summary
    show_harvest_summary() 
import streamlit as st
import importlib

st.set_page_config(page_title="AI Decision Platform", layout="wide")

st.title("🤖 AI Decision Support & What-If Simulator")

# Sidebar Navigation
tab_selection = st.sidebar.radio("Navigate", ["1. Data & Training", "2. Prediction & Insights", "3. What-If Simulator"])

# Dynamic Import of Tabs to keep code clean
if "Data" in tab_selection:
    module = importlib.import_module("tabs.1_upload")
    module.render()
elif "Prediction" in tab_selection:
    module = importlib.import_module("tabs.2_analysis")
    module.render()
elif "Simulator" in tab_selection:
    module = importlib.import_module("tabs.3_simulation")
    module.render()
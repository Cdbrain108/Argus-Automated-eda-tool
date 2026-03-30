import pandas as pd
import streamlit as st
import os

# Create dummy session state
st.session_state["dataset_name"] = "Titanic"

# Load data
df = pd.read_csv("demo_data.csv")
from home import _run_discovery_engine

try:
    _run_discovery_engine(df, "Titanic")
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()

"""
app.py — Entry point for Glimpse Automated EDA Tool.
Routes between the auth page and the main home page.
"""

import streamlit as st

from PIL import Image
try:
    icon = Image.open("argus_logo.png")
except Exception:
    icon = "📊"

st.set_page_config(
    page_title="Argus – Automated EDA Tool",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="collapsed",
)

from auth import show_auth_page
from home import show_home_page

if not st.session_state.get("logged_in"):
    show_auth_page()
else:
    show_home_page()

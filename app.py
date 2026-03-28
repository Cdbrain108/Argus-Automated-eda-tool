"""
app.py — Entry point for Glimpse Automated EDA Tool.
Routes between the auth page and the main home page.
"""

import streamlit as st

import os

icon_path = "argus_logo.png" if os.path.exists("argus_logo.png") else "📊"

st.set_page_config(
    page_title="Argus – An AI based Automated EDA Tool",
    page_icon=icon_path,
    layout="wide",
    initial_sidebar_state="collapsed",
)

from auth import show_auth_page, authenticate_from_cookie
from home import show_home_page

# Try to authenticate the user from a potential JWT cookie
if not st.session_state.get("logged_in"):
    authenticate_from_cookie()

if not st.session_state.get("logged_in"):
    show_auth_page()
else:
    show_home_page()

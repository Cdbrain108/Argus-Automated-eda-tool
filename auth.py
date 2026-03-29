"""
auth.py — Redesigned Argus landing page + login/signup.
Full-page dark design with sections: Nav, Hero, Features, How it works,
Gallery, About, FAQ, CTA, Footer — followed by the auth form.
"""

import streamlit as st
import hashlib, os, re, time
import jwt
from datetime import datetime, timedelta
from pymongo import MongoClient
from streamlit_cookies_controller import CookieController

def get_cookie_controller():
    if "cookie_controller" not in st.session_state:
        st.session_state["cookie_controller"] = CookieController(key="cookies")
    return st.session_state["cookie_controller"]

@st.cache_resource
def get_db_collection():
    uri = st.secrets.get("MONGO_URI", "")
    if not uri:
        return None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        return client["argus_db"]["users"]
    except Exception as e:
        print("MongoDB Auth Error:", e)
        return None

def _mint_jwt(email: str, name: str) -> str:
    secret = st.secrets.get("JWT_SECRET", "super-secret-fallback-key")
    payload = {
        "email": email,
        "name": name,
        "exp": datetime.utcnow() + timedelta(days=7)
    }
    return jwt.encode(payload, secret, algorithm="HS256")

def authenticate_from_cookie():
    cc = get_cookie_controller()
    token = cc.get("argus_jwt")
    if token:
        secret = st.secrets.get("JWT_SECRET", "super-secret-fallback-key")
        try:
            decoded = jwt.decode(token, secret, algorithms=["HS256"])
            st.session_state.logged_in = True
            st.session_state.user_email = decoded["email"]
            st.session_state.user_name = decoded["name"]
        except Exception:
            try:
                cc.remove("argus_jwt")
            except:
                pass

def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def _valid_email(e: str) -> bool:
    return bool(re.match(r"^[\w\.-]+@[\w\.-]+\.\w{2,}$", e))


# ── Main ───────────────────────────────────────────────────────────────────────

def show_auth_page():
    _inject_global_css()
    _render_navbar()
    _render_hero()
    _render_features()
    _render_how_it_works()
    _render_gallery()
    _render_about()
    _render_faq()
    _render_cta()
    st.markdown("<br><br>", unsafe_allow_html=True)
    _render_auth_form()
    _render_footer()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════

def _inject_global_css():
    st.markdown("""<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container {padding-top: 0 !important;padding-left: 0 !important;padding-right: 0 !important;max-width: 100% !important;}
.arg-nav-link {font-size: 13px;color: #9ca3af;cursor: pointer;padding: 6px 14px;border-radius: 6px;text-decoration: none;transition: color 0.15s;}
.arg-nav-link:hover { color: #ffffff; }
.arg-section-tag {display: inline-block;font-size: 10px;padding: 3px 10px;border-radius: 20px;font-weight: 700;letter-spacing: 0.06em;margin-bottom: 10px;}
.arg-card {background: #1a1f2e;border: 1px solid #2d3748;border-radius: 12px;padding: 16px 18px;}
.arg-stat {background: rgba(255,255,255,0.05);border: 1px solid rgba(255,255,255,0.08);border-radius: 12px;padding: 14px 18px;text-align: center;}
.arg-pill {display: inline-flex;align-items: center;gap: 6px;font-size: 11px;padding: 4px 12px;border-radius: 20px;margin: 3px;}
.arg-step {width: 32px;height: 32px;border-radius: 50%;display: inline-flex;align-items: center;justify-content: center;font-size: 13px;font-weight: 700;}
.arg-insight {border-radius: 10px;padding: 14px 16px;border-left: 3px solid;}
.arg-faq {border-bottom: 1px solid rgba(255,255,255,0.07);padding: 14px 0;}
.arg-btn-primary {background: #f97316;color: #000;border: none;border-radius: 10px;padding: 13px 28px;font-size: 14px;font-weight: 700;cursor: pointer;text-decoration: none;display: inline-block;}
.arg-btn-secondary {background: transparent;color: #fff;border: 1px solid rgba(255,255,255,0.15);border-radius: 10px;padding: 13px 24px;font-size: 14px;cursor: pointer;text-decoration: none;display: inline-block;}
@keyframes pulse {0%, 100% { opacity: 1; }50% { opacity: 0.4; }}
.arg-pulse {width: 7px; height: 7px;border-radius: 50%;background: #f97316;display: inline-block;animation: pulse 1.5s infinite;margin-right: 6px;}
.stApp {background: #0a0e1a !important;}
[data-testid="stTabs"] {background: rgba(255,255,255,0.04);border: 1px solid rgba(249,115,22,0.22);border-radius: 20px; padding: 24px 28px 20px;backdrop-filter: blur(14px); position: relative; z-index: 1;}
[data-testid="stTab"] { color: #94A3B8 !important; font-weight: 500; }
[data-testid="stTab"][aria-selected="true"] { color: #F97316 !important; }
[data-testid="stTextInput"] input {background: rgba(255,255,255,0.05) !important;border: 1px solid rgba(249,115,22,0.28) !important;border-radius: 10px !important; color: #E2E8F0 !important; padding: 11px 14px !important;}
[data-testid="stTextInput"] input:focus {border-color: #F97316 !important;box-shadow: 0 0 0 3px rgba(249,115,22,0.18) !important;outline: none !important;}
[data-testid="stFormSubmitButton"] > button {background: linear-gradient(135deg, #F97316, #EA580C) !important;color: #fff !important; border: none !important;border-radius: 10px !important; font-size: 1rem !important;font-weight: 700 !important; padding: 12px !important;box-shadow: 0 0 24px rgba(249,115,22,0.35) !important;transition: all 0.25s !important;}
[data-testid="stFormSubmitButton"] > button:hover {box-shadow: 0 0 42px rgba(249,115,22,0.6) !important;transform: translateY(-2px) !important;}
</style>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: STICKY NAVIGATION BAR
# ══════════════════════════════════════════════════════════════════════════════

def _render_navbar():
    import base64
    import os
    logo_path = os.path.join(os.path.dirname(__file__), "argus_logo.png")
    try:
        with open(logo_path, "rb") as f:
            b64_logo = base64.b64encode(f.read()).decode()
    except Exception:
        b64_logo = ""

    logo_img = f'<img src="data:image/png;base64,{b64_logo}" alt="Argus" style="width:28px;height:28px;border-radius:50%;box-shadow:0 0 8px rgba(249,115,22,0.4)">'

    st.markdown(
        f'<div style="position:sticky;top:0;z-index:999;background:rgba(10,14,26,0.96);backdrop-filter:blur(12px);border-bottom:1px solid rgba(255,255,255,0.06);padding:12px 40px;display:flex;align-items:center;justify-content:space-between">'
        f'<div style="display:flex;align-items:center;gap:10px">'
        f'{logo_img}'
        f'<span style="font-size:16px;font-weight:800;color:#f97316">Argus</span>'
        f'</div>'
        f'<div style="display:flex;gap:4px;align-items:center">'
        f'<a href="#" class="arg-nav-link">Home</a>'
        f'<a href="#" class="arg-nav-link">Features</a>'
        f'<a href="#" class="arg-nav-link">How it works</a>'
        f'<a href="#" class="arg-nav-link">Gallery</a>'
        f'<a href="#" class="arg-nav-link">About</a>'
        f'<a href="#" class="arg-nav-link">FAQ</a>'
        f'</div>'
        f'<a href="#" style="background:#f97316;color:#000;border:none;border-radius:8px;padding:8px 18px;font-size:13px;font-weight:700;cursor:pointer;text-decoration:none">Get started free</a>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: HERO SECTION
# ══════════════════════════════════════════════════════════════════════════════

def _render_hero():
    st.markdown(
        '<div id="home" style="text-align:center;padding:60px 40px 40px;background:#0a0e1a">'
        '<div style="display:inline-flex;align-items:center;background:rgba(249,115,22,0.12);border:1px solid rgba(249,115,22,0.25);border-radius:20px;padding:5px 16px;font-size:11px;color:#f97316;margin-bottom:22px">'
        '<span class="arg-pulse"></span>Now with AI-powered column descriptions</div>'
        '<h1 style="font-size:46px;font-weight:800;line-height:1.15;margin-bottom:14px;color:#fff">'
        'Understand your data<br>'
        '<span style="background:linear-gradient(90deg,#f97316,#7F77DD);-webkit-background-clip:text;-webkit-text-fill-color:transparent">in under 30 seconds</span></h1>'
        '<p style="font-size:15px;color:#9ca3af;max-width:500px;margin:0 auto 28px;line-height:1.75">'
        'Upload any CSV or Excel file. Argus automatically generates charts, detects anomalies, and explains every column in plain English — no code required.</p>'
        '<div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-bottom:28px">'
        '<a href="#" class="arg-btn-primary">Upload your dataset →</a>'
        '<a href="#" class="arg-btn-secondary">See sample insights</a></div>'
        '<div style="margin-bottom:36px">'
        '<span class="arg-pill" style="background:rgba(127,119,221,0.12);color:#AFA9EC;border:1px solid rgba(127,119,221,0.2)">No signup needed</span>'
        '<span class="arg-pill" style="background:rgba(29,158,117,0.12);color:#5DCAA5;border:1px solid rgba(29,158,117,0.2)">Works with any dataset</span>'
        '<span class="arg-pill" style="background:rgba(55,138,221,0.12);color:#85B7EB;border:1px solid rgba(55,138,221,0.2)">AI descriptions included</span>'
        '<span class="arg-pill" style="background:rgba(239,159,39,0.12);color:#FAC775;border:1px solid rgba(239,159,39,0.2)">10+ charts auto-generated</span></div>'
        '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;max-width:620px;margin:0 auto">'
        '<div class="arg-stat"><p style="font-size:24px;font-weight:800;color:#f97316;margin:0">13+</p><p style="font-size:11px;color:#6b7280;margin:4px 0 0">Datasets analyzed</p></div>'
        '<div class="arg-stat"><p style="font-size:24px;font-weight:800;color:#f97316;margin:0">★ 4.5</p><p style="font-size:11px;color:#6b7280;margin:4px 0 0">User rating</p></div>'
        '<div class="arg-stat"><p style="font-size:24px;font-weight:800;color:#f97316;margin:0">110+</p><p style="font-size:11px;color:#6b7280;margin:4px 0 0">Charts generated</p></div>'
        '<div class="arg-stat"><p style="font-size:24px;font-weight:800;color:#f97316;margin:0">9+</p><p style="font-size:11px;color:#6b7280;margin:4px 0 0">Questions answered</p></div>'
        '</div></div>'
        '<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.08),transparent);margin:0 40px"></div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: FEATURES SECTION
# ══════════════════════════════════════════════════════════════════════════════

def _render_features():
    st.markdown(
        '<div id="features" style="padding:48px 40px;background:#0a0e1a">'
        '<div style="text-align:center;margin-bottom:28px">'
        '<span class="arg-section-tag" style="background:rgba(127,119,221,0.15);color:#7F77DD">FEATURES</span>'
        '<h2 style="font-size:26px;font-weight:700;color:#fff;margin-bottom:6px">Everything EDA, automated</h2>'
        '<p style="font-size:13px;color:#6b7280">What Argus does the moment you upload a file</p></div>'
        '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px">'
        # Card 1
        '<div class="arg-card">'
        '<div style="width:38px;height:38px;border-radius:10px;background:rgba(249,115,22,0.15);display:flex;align-items:center;justify-content:center;margin-bottom:12px;font-size:18px">⚡</div>'
        '<p style="font-size:13px;font-weight:600;color:#fff;margin:0 0 6px">Instant EDA</p>'
        '<p style="font-size:12px;color:#6b7280;line-height:1.7;margin:0">10+ charts auto-generated in under 3 seconds — histograms, bar charts, heatmaps, trends, all in one click</p></div>'
        # Card 2
        '<div class="arg-card">'
        '<div style="width:38px;height:38px;border-radius:10px;background:rgba(127,119,221,0.15);display:flex;align-items:center;justify-content:center;margin-bottom:12px;font-size:18px">🧠</div>'
        '<p style="font-size:13px;font-weight:600;color:#fff;margin:0 0 6px">AI Column Descriptions</p>'
        '<p style="font-size:12px;color:#6b7280;line-height:1.7;margin:0">Every column explained in plain English — distribution shape, outliers, missing data, and what to do next</p></div>'
        # Card 3
        '<div class="arg-card">'
        '<div style="width:38px;height:38px;border-radius:10px;background:rgba(29,158,117,0.15);display:flex;align-items:center;justify-content:center;margin-bottom:12px;font-size:18px">💬</div>'
        '<p style="font-size:13px;font-weight:600;color:#fff;margin:0 0 6px">AI Chat</p>'
        '<p style="font-size:12px;color:#6b7280;line-height:1.7;margin:0">Ask anything about your data in plain English. &quot;Which column has the most outliers?&quot; — just ask.</p></div>'
        # Card 4
        '<div class="arg-card">'
        '<div style="width:38px;height:38px;border-radius:10px;background:rgba(226,75,74,0.15);display:flex;align-items:center;justify-content:center;margin-bottom:12px;font-size:18px">🔍</div>'
        '<p style="font-size:13px;font-weight:600;color:#fff;margin:0 0 6px">Anomaly Detection</p>'
        '<p style="font-size:12px;color:#6b7280;line-height:1.7;margin:0">Outliers flagged with IQR method. Negative values, zero prices, and impossible values all caught automatically</p></div>'
        # Card 5
        '<div class="arg-card">'
        '<div style="width:38px;height:38px;border-radius:10px;background:rgba(55,138,221,0.15);display:flex;align-items:center;justify-content:center;margin-bottom:12px;font-size:18px">📊</div>'
        '<p style="font-size:13px;font-weight:600;color:#fff;margin:0 0 6px">Dataset Health Score</p>'
        '<p style="font-size:12px;color:#6b7280;line-height:1.7;margin:0">A single 0-100 score summarising missing data, duplicates, and outlier severity — at a glance</p></div>'
        # Card 6
        '<div class="arg-card">'
        '<div style="width:38px;height:38px;border-radius:10px;background:rgba(239,159,39,0.15);display:flex;align-items:center;justify-content:center;margin-bottom:12px;font-size:18px">🎯</div>'
        '<p style="font-size:13px;font-weight:600;color:#fff;margin:0 0 6px">Smart Next Steps</p>'
        '<p style="font-size:12px;color:#6b7280;line-height:1.7;margin:0">AI recommends exactly what to do — impute missing values, log-transform skewed cols, encode categoricals</p></div>'
        '</div></div>'
        '<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.08),transparent);margin:0 40px"></div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: HOW IT WORKS SECTION
# ══════════════════════════════════════════════════════════════════════════════

def _render_how_it_works():
    st.markdown(
        '<div id="how-it-works" style="padding:48px 40px;background:rgba(255,255,255,0.02)">'
        '<div style="text-align:center;margin-bottom:28px">'
        '<span class="arg-section-tag" style="background:rgba(239,159,39,0.15);color:#EF9F27">HOW IT WORKS</span>'
        '<h2 style="font-size:26px;font-weight:700;color:#fff;margin-bottom:6px">Upload to insight in 4 steps</h2>'
        '<p style="font-size:13px;color:#6b7280">No setup. No code. No learning curve.</p></div>'
        '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px">'
        '<div style="text-align:center;padding:20px 12px"><div class="arg-step" style="background:#f97316;color:#000;margin:0 auto 14px">1</div><p style="font-size:13px;font-weight:600;color:#fff;margin:0 0 6px">Upload your file</p><p style="font-size:11px;color:#6b7280;line-height:1.7">CSV, Excel or JSON — drag and drop, no schema or config needed</p></div>'
        '<div style="text-align:center;padding:20px 12px"><div class="arg-step" style="background:#7F77DD;color:#fff;margin:0 auto 14px">2</div><p style="font-size:13px;font-weight:600;color:#fff;margin:0 0 6px">AI reads your data</p><p style="font-size:11px;color:#6b7280;line-height:1.7">Detects column types, domain, structure, and quality issues automatically</p></div>'
        '<div style="text-align:center;padding:20px 12px"><div class="arg-step" style="background:#1D9E75;color:#fff;margin:0 auto 14px">3</div><p style="font-size:13px;font-weight:600;color:#fff;margin:0 0 6px">Charts generated</p><p style="font-size:11px;color:#6b7280;line-height:1.7">10+ interactive charts with AI description per column, instantly</p></div>'
        '<div style="text-align:center;padding:20px 12px"><div class="arg-step" style="background:#378ADD;color:#fff;margin:0 auto 14px">4</div><p style="font-size:13px;font-weight:600;color:#fff;margin:0 0 6px">Chat with your data</p><p style="font-size:11px;color:#6b7280;line-height:1.7">Ask anything in plain English and get instant AI answers</p></div>'
        '</div></div>'
        '<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.08),transparent);margin:0 40px"></div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: GALLERY SECTION
# ══════════════════════════════════════════════════════════════════════════════

def _render_gallery():
    st.markdown(
        '<div id="gallery" style="padding:48px 40px;background:#0a0e1a">'
        '<div style="text-align:center;margin-bottom:28px">'
        '<span class="arg-section-tag" style="background:rgba(29,158,117,0.15);color:#1D9E75">GALLERY</span>'
        '<h2 style="font-size:26px;font-weight:700;color:#fff;margin-bottom:6px">Sample AI insights</h2>'
        '<p style="font-size:13px;color:#6b7280">Real descriptions Argus generates — automatically</p></div>'
        '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px">'
        '<div class="arg-insight" style="background:#1a1f2e;border-left-color:#7F77DD"><p style="font-size:10px;color:#7F77DD;margin:0 0 8px;font-weight:700;letter-spacing:.04em">AGE — HEART DISEASE DATASET</p><p style="font-size:12px;color:#e5e7eb;line-height:1.7;margin:0">&ldquo;Roughly symmetric, mean 47.8 yrs. Most patients aged 40-60. No missing values — clean and ready to use.&rdquo;</p></div>'
        '<div class="arg-insight" style="background:#1a1f2e;border-left-color:#1D9E75"><p style="font-size:10px;color:#1D9E75;margin:0 0 8px;font-weight:700;letter-spacing:.04em">COUNTRY — RETAIL DATASET</p><p style="font-size:12px;color:#e5e7eb;line-height:1.7;margin:0">&ldquo;UK dominates at 91% of records. 37 other countries share the rest — high cardinality, consider grouping rare values.&rdquo;</p></div>'
        '<div class="arg-insight" style="background:#1a1f2e;border-left-color:#EF9F27"><p style="font-size:10px;color:#EF9F27;margin:0 0 8px;font-weight:700;letter-spacing:.04em">SALARY — HR DATASET</p><p style="font-size:12px;color:#e5e7eb;line-height:1.7;margin:0">&ldquo;Strongly right-skewed (skew=2.1). Most earn $40K-$70K but outliers up to $500K pull the mean. Consider log transform.&rdquo;</p></div>'
        '<div class="arg-insight" style="background:#1a1f2e;border-left-color:#E24B4A"><p style="font-size:10px;color:#E24B4A;margin:0 0 8px;font-weight:700;letter-spacing:.04em">CHOLESTEROL — MEDICAL DATASET</p><p style="font-size:12px;color:#e5e7eb;line-height:1.7;margin:0">&ldquo;154 unique values, max 603 — likely data entry error. 7.8% missing. Filter extreme values before modelling.&rdquo;</p></div>'
        '<div class="arg-insight" style="background:#1a1f2e;border-left-color:#378ADD"><p style="font-size:10px;color:#378ADD;margin:0 0 8px;font-weight:700;letter-spacing:.04em">QUANTITY — RETAIL DATASET</p><p style="font-size:12px;color:#e5e7eb;line-height:1.7;margin:0">&ldquo;10,624 negative values — likely returns. 75% of orders are 10 units or fewer. Filter negatives before revenue analysis.&rdquo;</p></div>'
        '<div class="arg-insight" style="background:#1a1f2e;border-left-color:#D4537E"><p style="font-size:10px;color:#D4537E;margin:0 0 8px;font-weight:700;letter-spacing:.04em">RATING — E-COMMERCE DATASET</p><p style="font-size:12px;color:#e5e7eb;line-height:1.7;margin:0">&ldquo;Discrete values 1-5. Rating 4 dominates at 38%. Left-skewed — customers rate high more than low overall.&rdquo;</p></div>'
        '</div></div>'
        '<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.08),transparent);margin:0 40px"></div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: ABOUT SECTION
# ══════════════════════════════════════════════════════════════════════════════

def _render_about():
    st.markdown(
        '<div id="about" style="padding:48px 40px;background:rgba(255,255,255,0.02)">'
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:32px;align-items:center">'
        '<div>'
        '<span class="arg-section-tag" style="background:rgba(55,138,221,0.15);color:#378ADD">ABOUT</span>'
        '<h2 style="font-size:26px;font-weight:700;color:#fff;margin-bottom:12px">Why Argus?</h2>'
        '<p style="font-size:13px;color:#9ca3af;line-height:1.85;margin-bottom:14px">'
        'Exploratory data analysis is the most time-consuming and undervalued step in any data project. Argus was built to eliminate that bottleneck — giving analysts, students, and domain experts instant AI-powered insight into any dataset without writing a single line of code.</p>'
        '<p style="font-size:13px;color:#9ca3af;line-height:1.85">'
        'The name <span style="color:#f97316;font-weight:700">Argus</span> comes from the all-seeing giant of Greek mythology — a watcher with a hundred eyes. That\'s exactly what this tool does: it sees everything in your data that you might miss.</p></div>'
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">'
        '<div class="arg-card" style="text-align:center;padding:20px"><p style="font-size:28px;font-weight:800;color:#f97316;margin:0">100%</p><p style="font-size:11px;color:#6b7280;margin-top:5px">No-code EDA</p></div>'
        '<div class="arg-card" style="text-align:center;padding:20px"><p style="font-size:28px;font-weight:800;color:#7F77DD;margin:0">Any</p><p style="font-size:11px;color:#6b7280;margin-top:5px">Domain dataset</p></div>'
        '<div class="arg-card" style="text-align:center;padding:20px"><p style="font-size:28px;font-weight:800;color:#1D9E75;margin:0">&lt;30s</p><p style="font-size:11px;color:#6b7280;margin-top:5px">Full EDA time</p></div>'
        '<div class="arg-card" style="text-align:center;padding:20px"><p style="font-size:28px;font-weight:800;color:#378ADD;margin:0">AI</p><p style="font-size:11px;color:#6b7280;margin-top:5px">Powered insights</p></div>'
        '</div></div></div>'
        '<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.08),transparent);margin:0 40px"></div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: FAQ SECTION
# ══════════════════════════════════════════════════════════════════════════════

def _render_faq():
    st.markdown(
        '<div id="faq" style="padding:48px 40px;background:#0a0e1a">'
        '<div style="text-align:center;margin-bottom:28px">'
        '<span class="arg-section-tag" style="background:rgba(226,75,74,0.15);color:#E24B4A">FAQ</span>'
        '<h2 style="font-size:26px;font-weight:700;color:#fff;margin-bottom:6px">Common questions</h2>'
        '<p style="font-size:13px;color:#6b7280">Everything you want to know before you upload</p></div>'
        '<div style="max-width:640px;margin:0 auto">'
        '<div class="arg-faq"><p style="font-size:13px;font-weight:600;color:#fff;margin:0 0 6px">Do I need to know Python or SQL?</p><p style="font-size:12px;color:#6b7280;line-height:1.7;margin:0">No. Argus is fully no-code. Upload your file and everything happens automatically — no commands, no scripts.</p></div>'
        '<div class="arg-faq"><p style="font-size:13px;font-weight:600;color:#fff;margin:0 0 6px">What file types are supported?</p><p style="font-size:12px;color:#6b7280;line-height:1.7;margin:0">CSV, Excel (.xlsx, .xls), and JSON. Argus auto-detects column types and data domains without any configuration.</p></div>'
        '<div class="arg-faq"><p style="font-size:13px;font-weight:600;color:#fff;margin:0 0 6px">How do the AI descriptions work?</p><p style="font-size:12px;color:#6b7280;line-height:1.7;margin:0">Argus computes statistics per column — mean, skewness, outliers, missing % — then uses an LLM to write plain-English explanations covering distribution shape, data quality, and next steps.</p></div>'
        '<div class="arg-faq"><p style="font-size:13px;font-weight:600;color:#fff;margin:0 0 6px">Will it work on my specific dataset?</p><p style="font-size:12px;color:#6b7280;line-height:1.7;margin:0">Yes. Argus has been tested on healthcare, retail, finance, HR, IoT, and marketing datasets. It auto-detects the domain and adjusts descriptions accordingly.</p></div>'
        '<div class="arg-faq"><p style="font-size:13px;font-weight:600;color:#fff;margin:0 0 6px">Is my data stored anywhere?</p><p style="font-size:12px;color:#6b7280;line-height:1.7;margin:0">No. Your data is processed in-session only and never saved to any server or database.</p></div>'
        '<div class="arg-faq" style="border-bottom:none"><p style="font-size:13px;font-weight:600;color:#fff;margin:0 0 6px">How is this different from pandas profiling?</p><p style="font-size:12px;color:#6b7280;line-height:1.7;margin:0">Pandas profiling gives you statistics. Argus gives you understanding — AI-written descriptions, smart chart titles, anomaly flags, health scores, and a chat interface. It\'s EDA plus interpretation.</p></div>'
        '</div></div>'
        '<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.08),transparent);margin:0 40px"></div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9: FINAL CTA
# ══════════════════════════════════════════════════════════════════════════════

def _render_cta():
    st.markdown(
        '<div id="upload" style="text-align:center;padding:56px 40px;background:linear-gradient(180deg,#0a0e1a,rgba(249,115,22,0.06))">'
        '<h2 style="font-size:30px;font-weight:800;color:#fff;margin-bottom:10px">Ready to see your data differently?</h2>'
        '<p style="font-size:14px;color:#9ca3af;margin-bottom:24px">Login or create an account below to get started.</p>'
        '<p style="font-size:11px;color:#4b5563;margin-top:16px">CSV · Excel · JSON &nbsp;|&nbsp; No code required &nbsp;|&nbsp; Free to use</p></div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

def _render_footer():
    st.markdown(
        '<div style="padding:20px 40px;border-top:1px solid rgba(255,255,255,0.06);display:flex;justify-content:space-between;align-items:center;background:#0a0e1a">'
        '<span style="font-size:13px;font-weight:700;color:#f97316">Argus</span>'
        '<span style="font-size:11px;color:#4b5563">An Automated EDA Tool · Built with Streamlit &amp; Groq</span>'
        '<span style="font-size:11px;color:#4b5563">2025</span></div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# AUTH FORM (preserved from original)
# ══════════════════════════════════════════════════════════════════════════════

def _render_auth_form():
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown(
            '<div style="text-align:center;margin-bottom:16px">'
            '<h3 style="color:#fff;font-size:20px;font-weight:700;margin:0 0 4px">Login or Create Account</h3>'
            '<p style="color:#6b7280;font-size:12px;margin:0">Access the full Argus dashboard</p></div>',
            unsafe_allow_html=True,
        )

        tab_login, tab_signup = st.tabs(["🔑  Login", "✨  Create Account"])

        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.form("login_form"):
                email = st.text_input("Email", placeholder="you@example.com", key="li_email")
                pw    = st.text_input("Password", type="password", placeholder="••••••••", key="li_pw")
                ok    = st.form_submit_button("Login →", use_container_width=True)
            if ok:
                _handle_login(email.strip(), pw)

        with tab_signup:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.form("signup_form"):
                name    = st.text_input("Full Name",  placeholder="Anuj Kesharwani", key="su_name")
                email_s = st.text_input("Email",      placeholder="you@example.com", key="su_email")
                pw_s    = st.text_input("Password",   type="password", placeholder="Min 6 characters", key="su_pw")
                ok_s    = st.form_submit_button("Create Account →", use_container_width=True)
            if ok_s:
                _handle_signup(name.strip(), email_s.strip(), pw_s)

    st.markdown('<p style="font-size:0.72rem;color:#334155;text-align:center;margin-top:16px">'
                'By continuing you agree to our Terms of Service</p>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# AUTH HANDLERS (preserved from original)
# ══════════════════════════════════════════════════════════════════════════════

def _handle_login(email, pw):
    if not email or not pw:
        st.error("Please fill in all fields."); return

    users_col = get_db_collection()
    if users_col is None:
        st.error("⚠️ Database not configured: `MONGO_URI` missing in Streamlit secrets. Auth disabled."); return

    user = users_col.find_one({"email": email})
    if not user:
        st.error("No account found. Please sign up first."); return
    if user.get("password") != _hash(pw):
        st.error("Incorrect password."); return

    _set_session_and_cookie(user.get("name", "User"), email)

def _handle_signup(name, email, pw):
    if not name or not email or not pw:
        st.error("Please fill in all fields."); return
    if not _valid_email(email):
        st.error("Invalid email address."); return
    if len(pw) < 6:
        st.error("Password must be at least 6 characters."); return

    users_col = get_db_collection()
    if users_col is None:
        st.error("⚠️ Database not configured: `MONGO_URI` missing in Streamlit secrets. Auth disabled."); return

    if users_col.find_one({"email": email}):
        st.error("Account already exists. Please log in."); return

    users_col.insert_one({
        "name": name,
        "email": email,
        "password": _hash(pw),
        "created_at": datetime.utcnow()
    })

    st.success(f"Welcome aboard, {name}! 🎉")
    _set_session_and_cookie(name, email)

def _set_session_and_cookie(name, email):
    token = _mint_jwt(email, name)
    get_cookie_controller().set("argus_jwt", token)
    st.session_state.update(logged_in=True, user_name=name, user_email=email)
    time.sleep(0.7)
    st.rerun()

def logout():
    try:
        get_cookie_controller().remove("argus_jwt")
    except Exception:
        pass
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    time.sleep(0.5)
    st.rerun()

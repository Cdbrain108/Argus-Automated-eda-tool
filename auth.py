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

def _live_rating() -> str:
    """Return live avg rating string from MongoDB, fallback to 4.5."""
    try:
        from ratings import get_avg_rating
        avg, count = get_avg_rating()
        return str(avg)
    except Exception:
        return "4.5"


# ── Main ───────────────────────────────────────────────────────────────────────

def show_auth_page():
    _inject_global_css()

    # ── Detect guest mode via URL query param (?demo=1) ───────────────────
    if st.query_params.get("demo") == "1":
        # Clear existing data to force demo load
        for key in ["df", "eda_result", "w"]:
            if key in st.session_state:
                del st.session_state[key]
        
        st.session_state["guest_mode"] = True
        st.session_state["logged_in"] = True
        # Clear param to avoid loops on refresh
        st.query_params.clear()
        st.rerun()

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
.arg-nav-link {font-size:18px;color: #9ca3af;cursor: pointer;padding: 8px 16px;border-radius: 50px;text-decoration: none !important;font-weight: 600;transition: all 0.3s ease;border: 1px solid transparent;}
.arg-nav-link:hover { color: #fff !important; background: rgba(255, 255, 255, 0.05) !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; box-shadow: inset 0 0 10px rgba(255,255,255,0.02) !important; text-decoration: none !important; }
.arg-section-tag {display: inline-block;font-size:16px;padding: 3px 10px;border-radius: 20px;font-weight: 700;letter-spacing: 0.06em;margin-bottom: 10px;}
.arg-card {background: #1a1f2e;border: 1px solid #2d3748;border-radius: 12px;padding: 16px 18px;}
.arg-stat {background: rgba(255,255,255,0.05);border: 1px solid rgba(255,255,255,0.08);border-radius: 12px;padding: 14px 18px;text-align: center;}
.arg-pill {display: inline-flex;align-items: center;gap: 6px;font-size:17px;padding: 4px 12px;border-radius: 20px;margin: 3px;}
.arg-step {width: 32px;height: 32px;border-radius: 50%;display: inline-flex;align-items: center;justify-content: center;font-size:19px;font-weight: 700;}
.arg-insight {border-radius: 10px;padding: 14px 16px;border-left: 3px solid;}
.arg-faq {border-bottom: 1px solid rgba(255,255,255,0.07);padding: 14px 0;}
.arg-btn-primary {background: #f97316;color: #000 !important;border: none;border-radius: 10px;padding: 13px 28px;font-size:20px;font-weight: 700;cursor: pointer;text-decoration: none !important;display: inline-block;transition: all 0.25s ease;}
.arg-btn-primary:hover {transform: translateY(-2px); box-shadow: 0 10px 20px rgba(249,115,22,0.4); text-decoration: none !important;}
.arg-btn-secondary {background: transparent;color: #fff !important;border: 1px solid rgba(255,255,255,0.15);border-radius: 10px;padding: 13px 24px;font-size:20px;cursor: pointer;text-decoration: none !important;display: inline-block;transition: all 0.25s ease;}
.arg-btn-secondary:hover {border: 1px solid rgba(255,255,255,0.4); text-decoration: none !important; background: rgba(255,255,255,0.05);}
.arg-nav-cta {background:#f97316;color:#000 !important;border:none;border-radius:10px;padding:8px 22px;font-size:18px;font-weight:700;cursor:pointer;text-decoration:none !important;transition: all 0.25s ease;}
.arg-nav-cta:hover {text-decoration:none !important; transform:translateY(-1px); box-shadow:0 4px 12px rgba(249,115,22,0.3);}
@keyframes pulse {0%, 100% { opacity: 1; }50% { opacity: 0.4; }}
.arg-pulse {width: 7px; height: 7px;border-radius: 50%;background: #f97316;display: inline-block;animation: pulse 1.5s infinite;margin-right: 6px;}

/* Typing Animation */
.typing-container { display: inline-block; position: relative; }
.typing-text::after {
    content: "in under 30 seconds";
    animation: typing-cycle 9s infinite;
    background: linear-gradient(90deg,#f97316,#7F77DD);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
@keyframes typing-cycle {
    0%, 30% { content: "in under 30 seconds"; }
    33%, 63% { content: "with AI intelligence"; }
    66%, 96% { content: "with zero code"; }
}
.typing-cursor {
    display: inline-block;
    width: 3px;
    height: 1em;
    background: #f97316;
    margin-left: 4px;
    animation: blink 0.8s infinite;
    vertical-align: middle;
}
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }

.stApp {background: #0a0e1a !important;}
[data-testid="stTabs"] {background: rgba(255,255,255,0.04);border: 1px solid rgba(249,115,22,0.22);border-radius: 20px; padding: 24px 28px 20px;backdrop-filter: blur(14px); position: relative; z-index: 1;}
[data-testid="stTab"] { color: #94A3B8 !important; font-weight: 500; }
[data-testid="stTab"][aria-selected="true"] { color: #F97316 !important; }
[data-testid="stTextInput"] input {background: rgba(255,255,255,0.05) !important;border: 1px solid rgba(249,115,22,0.28) !important;border-radius: 10px !important; color: #E2E8F0 !important; padding: 11px 14px !important;}
[data-testid="stTextInput"] input:focus {border-color: #F97316 !important;box-shadow: 0 0 0 3px rgba(249,115,22,0.18) !important;outline: none !important;}
[data-testid="stFormSubmitButton"] > button {background: linear-gradient(135deg, #F97316, #EA580C) !important;color: #fff !important; border: none !important;border-radius: 10px !important; font-size: 1rem !important;font-weight: 700 !important; padding: 12px !important;box-shadow: 0 0 24px rgba(249,115,22,0.35) !important;transition: all 0.25s !important;}
[data-testid="stFormSubmitButton"] > button:hover {box-shadow: 0 0 42px rgba(249,115,22,0.6) !important;transform: translateY(-2px) !important;}

/* ══════════════════════════════════════════════
   RESPONSIVE — Tablet (≤ 900px)
══════════════════════════════════════════════ */
@media (max-width: 900px) {
    .arg-nav-link { display: none !important; }
    .nav-links-container { display: none !important; }
    .arg-nav-cta { padding: 6px 14px !important; font-size: 14px !important; }
    .arg-stat { padding: 10px !important; }
    .arg-stat p:first-child { font-size: 24px !important; }
}

/* ══════════════════════════════════════════════
   RESPONSIVE — Mobile (≤ 600px)
══════════════════════════════════════════════ */
@media (max-width: 600px) {
    /* Navbar */
    .arg-nav-link { display: none !important; }
    .nav-links-container { display: none !important; } /* Hide completely to save space */
    .arg-nav-cta { padding: 5px 10px !important; font-size: 12px !important; }

    /* Shrink large inline logo in navbar */
    .nav-header-container { height: 60px !important; padding: 0 10px !important; }
    .nav-logo-text { font-size: 26px !important; }
    .nav-logo-img { width: 44px !important; height: 44px !important; }

    /* Hero section */
    .hero-container { padding: 24px 8px 16px !important; }
    .hero-title { 
        font-size: 32px !important; 
        line-height: 1.2 !important; 
        word-break: keep-all !important; 
        overflow-wrap: normal !important; 
    }
    .hero-subtitle { font-size: 15px !important; padding: 0 4px !important; }

    /* Hero CTA buttons — stack vertically */
    .hero-btns-container {
        flex-direction: column !important;
        align-items: stretch !important;
    }
    .arg-btn-primary, .arg-btn-secondary {
        font-size: 15px !important;
        padding: 10px 16px !important;
        text-align: center !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }

    /* Pills row — smaller on mobile */
    .arg-pill { font-size: 12px !important; padding: 3px 8px !important; }

    /* Hero stats: 2-col instead of 4-col */
    .stats-grid-container {
        grid-template-columns: repeat(2, 1fr) !important;
        max-width: 100% !important;
    }
    .arg-stat p:first-child { font-size: 20px !important; }
    .arg-stat p:last-child { font-size: 13px !important; }

    /* Section headings */
    h2 { font-size: 22px !important; }
    .arg-section-tag { font-size: 12px !important; }

    /* Features section: 1 column */
    div[id="features"] { padding: 24px 12px !important; }
    .features-grid-container {
        grid-template-columns: 1fr !important;
    }

    /* Gallery section: 1 column */
    div[id="gallery"] { padding: 24px 12px !important; }
    .gallery-grid-container {
        grid-template-columns: 1fr !important;
    }

    /* About section: stack to 1 col */
    div[id="about"] { padding: 24px 12px !important; }
    .about-main-grid {
        grid-template-columns: 1fr !important;
        gap: 16px !important;
    }
    .about-stats-grid {
        grid-template-columns: repeat(2, 1fr) !important;
    }

    /* FAQ */
    div[id="faq"] { padding: 24px 12px !important; }
    div[id="faq"] div[style*="max-width:640px"] { max-width: 100% !important; }

    /* CTA */
    div[id="upload"] { padding: 24px 12px !important; }

    /* Footer: single column */
    .footer-grid-container {
        grid-template-columns: 1fr !important;
        text-align: center !important;
        gap: 6px !important;
        padding: 16px 12px !important;
    }
    div[style*="text-align:left"] > span { display: block !important; text-align: center !important; }
    div[style*="text-align:right"] > span { display: block !important; text-align: center !important; }

    /* Auth form */
    [data-testid="stTabs"] { padding: 12px 8px 10px !important; }
    .block-container { padding-left: 6px !important; padding-right: 6px !important; }

    /* Streamlit buttons on mobile — ensure proper sizing & no joining */
    [data-testid="stButton"] > button {
        min-height: 48px !important;
        font-size: 0.95rem !important;
        padding: 12px 16px !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }
    [data-testid="stButton"] {
        width: 100% !important;
        margin-bottom: 2px !important;
    }
    /* Streamlit columns on mobile — add gap */
    [data-testid="stHorizontalBlock"] { gap: 8px !important; }
    [data-testid="column"] { min-width: 0 !important; }
}
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

    logo_img = f'<img src="data:image/png;base64,{b64_logo}" alt="Argus" class="nav-logo-img" style="width:80px;height:80px;border-radius:50%;box-shadow:0 0 12px rgba(249,115,22,0.4); border: 2px solid rgba(249,115,22,0.3); transition: transform 0.3s ease;">'

    st.markdown(
        f'<div class="nav-header-container" style="position:sticky;top:0;z-index:999;background:rgba(10,14,26,0.98);backdrop-filter:blur(15px);border-bottom:1px solid rgba(255,255,255,0.06);padding:0 40px;display:flex;align-items:center;justify-content:space-between; height: 100px;">'
        f'<div style="display:flex;align-items:center;gap:16px">'
        f'{logo_img}'
        f'<span class="nav-logo-text" style="font-size:48px;font-weight:900;color:#f97316;letter-spacing:-0.02em; line-height: 1;">Argus</span>'
        f'</div>'
        f'<div class="nav-links-container" style="display:flex;gap:4px;align-items:center">'
        f'<a href="#home" class="arg-nav-link">Home</a>'
        f'<a href="#features" class="arg-nav-link">Features</a>'
        f'<a href="#how-it-works" class="arg-nav-link">How it works</a>'
        f'<a href="#gallery" class="arg-nav-link">Gallery</a>'
        f'<a href="#about" class="arg-nav-link">About</a>'
        f'<a href="#faq" class="arg-nav-link">FAQ</a>'
        f'</div>'
        f'<a href="#upload" class="arg-nav-cta">Get started free</a>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: HERO SECTION
# ══════════════════════════════════════════════════════════════════════════════

def _render_hero():
    st.markdown(
        '<div id="home" class="hero-container" style="text-align:center;padding:60px 40px 40px;background:#0a0e1a">'
        '<div style="display:inline-flex;align-items:center;background:rgba(249,115,22,0.12);border:1px solid rgba(249,115,22,0.25);border-radius:20px;padding:5px 16px;font-size:17px;color:#f97316;margin-bottom:22px">'
        '<span class="arg-pulse"></span>Now with AI-powered categorical column intelligence</div>'
        '<h1 class="hero-title" style="font-size:54px;font-weight:800;line-height:1.15;margin-bottom:14px;color:#fff">'
        'Understand your data<br>'
        '<div class="typing-container">'
        '<span class="typing-text"></span>'
        '<span class="typing-cursor"></span>'
        '</div></h1>'
        '<p class="hero-subtitle" style="font-size:21px;color:#9ca3af;max-width:500px;margin:0 auto 28px;line-height:1.75">'
        'Upload any CSV or Excel file. Argus automatically generates charts, detects anomalies, and explains every column in plain English — no code required.</p>'
        '<div class="hero-btns-container" style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-bottom:28px">'
        '<a href="#upload" class="arg-btn-primary">Upload your dataset \u2192</a>'
        '<a href="?demo=1" target="_self" class="arg-btn-secondary" style="text-decoration:none !important;">\u26a1 Try Live Demo</a></div>'
        '<div style="margin-bottom:36px">'
        '<span class="arg-pill" style="background:rgba(127,119,221,0.12);color:#AFA9EC;border:1px solid rgba(127,119,221,0.2)">Free Signup</span>'
        '<span class="arg-pill" style="background:rgba(29,158,117,0.12);color:#5DCAA5;border:1px solid rgba(29,158,117,0.2)">Works with any dataset</span>'
        '<span class="arg-pill" style="background:rgba(55,138,221,0.12);color:#85B7EB;border:1px solid rgba(55,138,221,0.2)">AI descriptions included</span>'
        '<span class="arg-pill" style="background:rgba(249,115,22,0.12);color:#FAC775;border:1px solid rgba(249,115,22,0.2)">Smart Categorical Analysis</span>'
        '<span class="arg-pill" style="background:rgba(239,159,39,0.12);color:#FAC775;border:1px solid rgba(239,159,39,0.2)">10+ charts auto-generated</span></div>'
        '<div class="stats-grid-container" style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;max-width:620px;margin:0 auto">'
        '<div class="arg-stat"><p style="font-size:32px;font-weight:800;color:#f97316;margin:0">13+</p><p style="font-size:17px;color:#6b7280;margin:4px 0 0">Datasets analyzed</p></div>'
        f'<div class="arg-stat"><p style="font-size:32px;font-weight:800;color:#f97316;margin:0">\u2605 {_live_rating()}</p><p style="font-size:17px;color:#6b7280;margin:4px 0 0">User rating</p></div>'
        '<div class="arg-stat"><p style="font-size:32px;font-weight:800;color:#f97316;margin:0">110+</p><p style="font-size:17px;color:#6b7280;margin:4px 0 0">Charts generated</p></div>'
        '<div class="arg-stat"><p style="font-size:32px;font-weight:800;color:#f97316;margin:0">9+</p><p style="font-size:17px;color:#6b7280;margin:4px 0 0">Questions answered</p></div>'
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
        '<h2 style="font-size:34px;font-weight:700;color:#fff;margin-bottom:6px">Everything EDA, automated</h2>'
        '<p style="font-size:19px;color:#6b7280">What Argus does the moment you upload a file</p></div>'
        '<div class="features-grid-container" style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px">'
        # Card 1
        '<div class="arg-card">'
        '<div style="width:38px;height:38px;border-radius:10px;background:rgba(249,115,22,0.15);display:flex;align-items:center;justify-content:center;margin-bottom:12px;font-size:24px">⚡</div>'
        '<p style="font-size:19px;font-weight:600;color:#fff;margin:0 0 6px">Instant EDA</p>'
        '<p style="font-size:18px;color:#6b7280;line-height:1.7;margin:0">10+ charts auto-generated in under 3 seconds — histograms, bar charts, heatmaps, trends, all in one click</p></div>'
        # Card 2
        '<div class="arg-card">'
        '<div style="width:38px;height:38px;border-radius:10px;background:rgba(127,119,221,0.15);display:flex;align-items:center;justify-content:center;margin-bottom:12px;font-size:24px">🧠</div>'
        '<p style="font-size:19px;font-weight:600;color:#fff;margin:0 0 6px">AI Column Descriptions</p>'
        '<p style="font-size:18px;color:#6b7280;line-height:1.7;margin:0">Every column explained in plain English — distribution shape, outliers, missing data, and what to do next</p></div>'
        # Card 3
        '<div class="arg-card">'
        '<div style="width:38px;height:38px;border-radius:10px;background:rgba(29,158,117,0.15);display:flex;align-items:center;justify-content:center;margin-bottom:12px;font-size:24px">💬</div>'
        '<p style="font-size:19px;font-weight:600;color:#fff;margin:0 0 6px">AI Chat</p>'
        '<p style="font-size:18px;color:#6b7280;line-height:1.7;margin:0">Ask anything about your data in plain English. &quot;Which column has the most outliers?&quot; — just ask.</p></div>'
        # Card 4
        '<div class="arg-card">'
        '<div style="width:38px;height:38px;border-radius:10px;background:rgba(226,75,74,0.15);display:flex;align-items:center;justify-content:center;margin-bottom:12px;font-size:24px">🔍</div>'
        '<p style="font-size:19px;font-weight:600;color:#fff;margin:0 0 6px">Anomaly Detection</p>'
        '<p style="font-size:18px;color:#6b7280;line-height:1.7;margin:0">Outliers flagged with IQR method. Negative values, zero prices, and impossible values all caught automatically</p></div>'
        # Card 5
        '<div class="arg-card">'
        '<div style="width:38px;height:38px;border-radius:10px;background:rgba(55,138,221,0.15);display:flex;align-items:center;justify-content:center;margin-bottom:12px;font-size:24px">📊</div>'
        '<p style="font-size:19px;font-weight:600;color:#fff;margin:0 0 6px">Dataset Health Score</p>'
        '<p style="font-size:18px;color:#6b7280;line-height:1.7;margin:0">A single 0-100 score summarising missing data, duplicates, and outlier severity — at a glance</p></div>'
        # Card 6
        '<div class="arg-card">'
        '<div style="width:38px;height:38px;border-radius:10px;background:rgba(239,159,39,0.15);display:flex;align-items:center;justify-content:center;margin-bottom:12px;font-size:24px">🎯</div>'
        '<p style="font-size:19px;font-weight:600;color:#fff;margin:0 0 6px">Smart Next Steps</p>'
        '<p style="font-size:18px;color:#6b7280;line-height:1.7;margin:0">AI recommends exactly what to do — impute missing values, log-transform skewed cols, encode categoricals</p></div>'
        '</div></div>'
        '<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.08),transparent);margin:0 40px"></div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: HOW IT WORKS SECTION
# ══════════════════════════════════════════════════════════════════════════════

def _render_how_it_works():
    st.markdown('<div id="how-it-works" style="padding:60px 40px;background:rgba(255,255,255,0.02)">', unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align:center;margin-bottom:30px">
            <span class="arg-section-tag" style="background:rgba(239,159,39,0.15);color:#EF9F27">PROCESS FLOW</span>
            <h2 style="font-size:38px;font-weight:800;color:#fff;margin-bottom:6px">Data to Insight in 4 Steps</h2>
            <p style="font-size:19px;color:#6b7280">No maintenance. No setup. No learning curve.</p>
        </div>
    """, unsafe_allow_html=True)

    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    :root {
        --s1: #f97316; --s2: #7F77DD; --s3: #1D9E75; --s4: #378ADD;
        --bg: transparent; --card: #111524; --border: #2d3748;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: var(--bg); color: #fff; font-family: 'Inter', system-ui, -apple-system, sans-serif; overflow: hidden; }
    
    .cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 30px; }
    .card {
        background: var(--card); border: 2px solid var(--border); border-radius: 16px;
        padding: 20px 18px; transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        min-height: 380px; display: flex; flex-direction: column; position: relative;
    }
    .card.active { border-color: var(--active-color); box-shadow: 0 0 25px rgba(255,255,255,0.05); transform: translateY(-6px); }
    .card.done { border-color: #1D9E75; background: rgba(29, 158, 117, 0.03); }
    
    .step-badge {
        width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
        font-size: 16px; font-weight: 800; color: #fff; margin-bottom: 14px;
    }
    .card-title { font-size: 17px; font-weight: 700; color: #fff; margin-bottom: 12px; line-height: 1.3; }
    .card-body { flex: 1; display: flex; flex-direction: column; gap: 12px; }
    .card-desc { font-size: 13px; color: #94a3b8; line-height: 1.6; margin-top: auto; }
    
    .screen {
        background: #0a0e1a; border-radius: 10px; border: 1px solid rgba(255,255,255,0.08);
        padding: 12px; min-height: 160px; display: flex; flex-direction: column; gap: 8px;
        overflow: hidden; position: relative;
    }
    
    .file-row { display: flex; align-items: center; gap: 8px; background: rgba(255,255,255,0.03); border-radius: 8px; padding: 10px; }
    .file-icon { width: 24px; height: 30px; background: var(--s1); border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 8px; color: #000; font-weight: 800; }
    .file-name { font-size: 12px; color: #e2e8f0; font-weight: 600; }
    .progress-bar { height: 5px; background: #1e293b; border-radius: 3px; overflow: hidden; margin-top: 6px; }
    .progress-fill { height: 100%; background: var(--s1); width: 0%; transition: width 0.1s ease; }
    
    .col-row { display: flex; align-items: center; gap: 6px; background: rgba(255,255,255,0.03); border-radius: 6px; padding: 6px 10px; opacity: 0; transform: translateX(-10px); transition: all 0.3s; }
    .col-dot { width: 8px; height: 8px; border-radius: 50%; }
    .col-label { font-size: 11px; color: #94a3b8; }
    .col-type { font-size: 10px; padding: 2px 8px; border-radius: 10px; font-weight: 700; margin-left: auto; }
    
    .chart-container { display: flex; align-items: flex-end; gap: 5px; height: 90px; padding-top: 10px; }
    .bar { flex: 1; border-radius: 4px 4px 1px 1px; background: var(--s3); min-width: 0; height: 0%; transition: height 0.6s cubic-bezier(0.34, 1.56, 0.64, 1); }
    
    .chat-bubble { border-radius: 12px; padding: 8px 12px; font-size: 12px; line-height: 1.5; max-width: 90%; opacity: 0; transform: translateY(8px); transition: all 0.4s; }
    .chat-user { background: rgba(255,255,255,0.08); color: #e2e8f0; align-self: flex-end; margin-left: auto; }
    .chat-ai { background: rgba(55, 138, 221, 0.1); border: 1px solid rgba(55, 138, 221, 0.2); color: #fff; align-self: flex-start; }
    
    .typing { display: flex; gap: 4px; padding: 8px 12px; background: rgba(255,255,255,0.04); border-radius: 12px; width: fit-content; opacity: 0; transition: opacity 0.3s; }
    .typing span { width: 6px; height: 6px; border-radius: 50%; background: var(--s4); animation: dotBounce 1.4s infinite; }
    .typing span:nth-child(2) { animation-delay: 0.2s; }
    .typing span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes dotBounce { 0%, 80%, 100% { transform: scale(0); opacity: 0.3; } 40% { transform: scale(1); opacity: 1; } }
    
    .status-bar { text-align: center; color: #64748b; font-size: 15px; min-height: 30px; margin-top: 20px; transition: color 0.4s; }
    .status-bar.done { color: var(--s3); font-weight: 700; animation: pulseSuccess 2s infinite; }
    @keyframes pulseSuccess { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
    
    .replay-btn {
        display: none; margin: 24px auto 0; background: transparent; border: 2px solid var(--s1);
        color: var(--s1); border-radius: 30px; padding: 10px 28px; font-size: 14px; font-weight: 700;
        cursor: pointer; transition: all 0.3s; letter-spacing: 0.05em; text-transform: uppercase;
    }
    .replay-btn:hover { background: rgba(249, 115, 22, 0.1); transform: scale(1.05); }
    
    .insight-pill { background: rgba(255,255,255,0.03); border-left: 3px solid; border-radius: 4px; padding: 7px 10px; font-size: 11px; color: #cbd5e1; line-height: 1.4; opacity: 0; transform: translateX(-8px); transition: all 0.4s; }
    
    @media (max-width: 900px) {
        /* Tablet: 2x2 grid, but aggressively shrink heights to stay under the 600px iframe vertical limit */
        .cards { grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 16px; }
        .card { min-height: 230px; padding: 12px; }
        .step-badge { width: 28px; height: 28px; font-size: 13px; margin-bottom: 10px; }
        .card-title { font-size: 14px; margin-bottom: 8px; }
        .card-desc { font-size: 10px; line-height: 1.4; }
        .screen { padding: 8px; min-height: 110px; gap: 6px; }
        .file-icon { width: 18px; height: 24px; font-size: 6px; }
        .file-name { font-size: 10px; }
        .file-row { padding: 6px; gap: 8px; }
        #file-stats { margin-top: 8px !important; gap: 6px !important; }
        #file-stats > div { padding: 6px; }
        #file-stats div { font-size: 9px !important; }
        #file-stats div:last-child { font-size: 13px !important; }
        .col-row { padding: 4px 6px; gap: 6px; }
        .col-label { font-size: 10px; }
        .col-type { font-size: 8px; padding: 2px 6px; }
        .chart-container { height: 50px; padding-top: 6px; }
        .insight-pill { font-size: 9.5px; padding: 5px 8px; }
        .chat-bubble { font-size: 10px; padding: 5px 8px; }
        .status-bar { font-size: 13px; margin-top: 10px; min-height: 24px; }
        .replay-btn { margin: 16px auto 0; padding: 8px 20px; font-size: 12px; }
    }
    @media (max-width: 600px) {
        /* Mobile: Keep 2x2 grid but shrink even further for ultra-narrow screens */
        .cards { gap: 8px; margin-bottom: 10px; }
        .card { min-height: 200px; padding: 10px; }
        .step-badge { width: 24px; height: 24px; font-size: 12px; margin-bottom: 8px; }
        .card-title { font-size: 13px; margin-bottom: 6px; line-height: 1.2; }
        .card-desc { font-size: 9px; line-height: 1.3; }
        .screen { padding: 6px; min-height: 90px; gap: 4px; }
        .file-icon { width: 14px; height: 18px; font-size: 5px; }
        .file-name { font-size: 9px; }
        .file-row { padding: 4px; gap: 6px; }
        #file-stats { margin-top: 6px !important; gap: 4px !important; }
        #file-stats > div { padding: 4px; }
        #file-stats div { font-size: 8px !important; }
        #file-stats div:last-child { font-size: 11px !important; }
        .col-row { padding: 3px 4px; gap: 4px; }
        .col-label { font-size: 9px; }
        .col-type { font-size: 7px; padding: 2px 4px; }
        .chart-container { height: 40px; padding-top: 4px; }
        .insight-pill { font-size: 8.5px; padding: 4px 6px; }
        .chat-bubble { font-size: 8px; padding: 4px 6px; }
        .status-bar { font-size: 12px; margin-top: 8px; min-height: 20px; }
        .replay-btn { margin: 10px auto 0; padding: 6px 16px; font-size: 11px; }
    }
    </style>
    </head>
    <body>
    <div class="cards">
        <div class="card" id="c1" style="--active-color: var(--s1)">
            <div class="step-badge" style="background:var(--s1);color:#000">1</div>
            <div class="card-title">Upload your file</div>
            <div class="card-body">
                <div class="screen">
                    <div class="file-row" id="frow" style="opacity:0">
                        <div class="file-icon">CSV</div>
                        <div style="flex:1">
                            <div class="file-name">heart_data.csv</div>
                            <div class="progress-bar"><div class="progress-fill" id="prog"></div></div>
                            <div style="font-size:10px;color:var(--s1);margin-top:4px" id="pct">0%</div>
                        </div>
                    </div>
                    <div id="file-stats" style="margin-top:10px;display:grid;grid-template-columns:1fr 1fr;gap:6px;opacity:0;transition:opacity 0.5s">
                        <div style="background:rgba(255,255,255,0.03);border-radius:6px;padding:8px">
                            <div style="font-size:10px;color:#64748b">Rows</div>
                            <div style="font-size:16px;font-weight:800;color:var(--s1)">1,025</div>
                        </div>
                        <div style="background:rgba(255,255,255,0.03);border-radius:6px;padding:8px">
                            <div style="font-size:10px;color:#64748b">Columns</div>
                            <div style="font-size:16px;font-weight:800;color:var(--s1)">14</div>
                        </div>
                    </div>
                </div>
                <div class="card-desc">CSV, Excel or JSON — Drag & drop, no schema definition needed</div>
            </div>
        </div>
        <div class="card" id="c2" style="--active-color: var(--s2)">
            <div class="step-badge" style="background:var(--s2)">2</div>
            <div class="card-title">AI reads your data</div>
            <div class="card-body">
                <div class="screen" id="cols" style="display:flex;flex-direction:column;gap:6px"></div>
                <div class="card-desc">Automatically detects column types, domain, and quality issues</div>
            </div>
        </div>
        <div class="card" id="c3" style="--active-color: var(--s3)">
            <div class="step-badge" style="background:var(--s3)">3</div>
            <div class="card-title">Charts generated</div>
            <div class="card-body">
                <div class="screen" style="justify-content:flex-end">
                    <div class="chart-container" id="bars"></div>
                </div>
                <div id="insights-area" style="display:flex;flex-direction:column;gap:6px"></div>
                <div class="card-desc">10+ interactive charts with AI-generated narratives</div>
            </div>
        </div>
        <div class="card" id="c4" style="--active-color: var(--s4)">
            <div class="step-badge" style="background:var(--s4)">4</div>
            <div class="card-title">Chat with data</div>
            <div class="card-body">
                <div class="screen" id="chat" style="display:flex;flex-direction:column;gap:8px"></div>
                <div class="card-desc">Ask anything in plain English, get instant AI-powered answers</div>
            </div>
        </div>
    </div>
    <div class="status-bar" id="status">Starting analysis...</div>
    <button class="replay-btn" id="replay-btn" onclick="replay()">↻ Replay Animation</button>

    <script>
    const COLORS = { s1: '#f97316', s2: '#7F77DD', s3: '#1D9E75', s4: '#378ADD' };
    const COLS = [
        {n:'Age',t:'numeric',c:COLORS.s2},
        {n:'Sex',t:'category',c:COLORS.s1},
        {n:'Cholesterol',t:'numeric',c:COLORS.s2},
        {n:'HeartDisease',t:'target',c:'#E24B4A'},
        {n:'MaxHR',t:'numeric',c:COLORS.s2}
    ];
    const MSGS = [
        {role:'user',text:'Find missing values in health data?'},
        {role:'ai',text:'Cholesterol has 172 missing values (16.8%). Imputation recommended.'},
        {role:'user',text:'Is Age a risk factor?'},
        {role:'ai',text:'Yes—patients with heart disease average 5.4 years older.'}
    ];
    const INSIGHTS = [
        {c:'#E24B4A',t:'Extreme Outliers in Cholesterol (max 603)'},
        {c:'#EF9F27',t:'Strong Skew in Oldpeak column'}
    ];

    let timers = [];
    let intervals = [];

    function clr() {
        timers.forEach(clearTimeout);
        intervals.forEach(clearInterval);
        timers = [];
        intervals = [];
    }

    function after(ms, fn) { let t = setTimeout(fn, ms); timers.push(t); return t; }

    function setStatus(txt, done) {
        const el = document.getElementById('status');
        el.textContent = txt;
        el.className = 'status-bar' + (done ? ' done' : '');
    }

    function setActive(id) {
        ['c1','c2','c3','c4'].forEach(c => document.getElementById(c).classList.remove('active'));
        if(id) document.getElementById(id).classList.add('active');
    }

    function setDone(id) {
        const el = document.getElementById(id);
        el.classList.remove('active');
        el.classList.add('done');
    }

    function resetUI() {
        clr();
        document.getElementById('replay-btn').style.display = 'none';
        ['c1','c2','c3','c4'].forEach(c => document.getElementById(c).classList.remove('active','done'));
        document.getElementById('frow').style.opacity = 0;
        document.getElementById('prog').style.width = '0%';
        document.getElementById('pct').textContent = '0%';
        document.getElementById('file-stats').style.opacity = 0;
        document.getElementById('cols').innerHTML = '';
        document.getElementById('bars').innerHTML = '';
        document.getElementById('insights-area').innerHTML = '';
        document.getElementById('chat').innerHTML = '';
    }

    function replay() { resetUI(); run(); }

    function run() {
        setStatus('Initializing secure upload...');
        setActive('c1');
        
        after(500, () => { document.getElementById('frow').style.opacity = 1; });
        
        let pct = 0;
        const pInt = setInterval(() => {
            pct = Math.min(pct + 4, 100);
            document.getElementById('prog').style.width = pct+'%';
            document.getElementById('pct').textContent = pct+'%';
            if(pct >= 100) clearInterval(pInt);
        }, 40);
        intervals.push(pInt);

        after(1600, () => { document.getElementById('file-stats').style.opacity = 1; });
        
        after(2500, () => {
            setDone('c1'); setActive('c2');
            setStatus('AI parsing schema & domain context...');
            COLS.forEach((col, i) => {
                after(i*300, () => {
                    const row = document.createElement('div');
                    row.className = 'col-row';
                    row.innerHTML = `<div class="col-dot" style="background:${col.c}"></div>
                        <span class="col-label">${col.n}</span>
                        <span class="col-type" style="background:rgba(255,255,255,0.05);color:${col.c}">${col.t}</span>`;
                    document.getElementById('cols').appendChild(row);
                    setTimeout(() => { row.style.opacity = 1; row.style.transform = 'translateX(0)'; }, 10);
                });
            });
        });

        after(4800, () => {
            setDone('c2'); setActive('c3');
            setStatus('Rendering 12 interactive AI-described charts...');
            const values = [40, 90, 60, 85, 45, 70];
            values.forEach((h, i) => {
                const bar = document.createElement('div');
                bar.className = 'bar';
                document.getElementById('bars').appendChild(bar);
                after(i*100, () => { bar.style.height = h+'%'; });
            });
            INSIGHTS.forEach((ins, i) => {
                after(800 + i*500, () => {
                    const dr = document.createElement('div');
                    dr.className = 'insight-pill';
                    dr.style.borderLeftColor = ins.c;
                    dr.textContent = ins.t;
                    document.getElementById('insights-area').appendChild(dr);
                    setTimeout(() => { dr.style.opacity = 1; dr.style.transform = 'translateX(0)'; }, 10);
                });
            });
        });

        after(7500, () => {
            setDone('c3'); setActive('c4');
            setStatus('Knowledge engine active. Ask Argus anything.');
            let delay = 0;
            MSGS.forEach((msg) => {
                after(delay, () => {
                    const chat = document.getElementById('chat');
                    if(msg.role === 'ai') {
                        const t = document.createElement('div'); t.className = 'typing'; t.innerHTML = '<span></span><span></span><span></span>';
                        chat.appendChild(t); t.style.opacity = 1;
                        after(1000, () => {
                            t.remove();
                            const m = document.createElement('div'); m.className = 'chat-bubble chat-ai'; m.textContent = msg.text;
                            chat.appendChild(m);
                            setTimeout(() => { m.style.opacity = 1; m.style.transform = 'translateY(0)'; chat.scrollTop = chat.scrollHeight; }, 10);
                        });
                    } else {
                        const m = document.createElement('div'); m.className = 'chat-bubble chat-user'; m.textContent = msg.text;
                        chat.appendChild(m);
                        setTimeout(() => { m.style.opacity = 1; m.style.transform = 'translateY(0)'; chat.scrollTop = chat.scrollHeight; }, 10);
                    }
                });
                delay += (msg.role === 'user' ? 800 : 2500);
            });
        });

        after(14000, () => {
            setDone('c4');
            setStatus('Analysis complete. 8 findings ready for review.', true);
            document.getElementById('replay-btn').style.display = 'block';
        });
    }

    run();
    </script>
    </body>
    </html>
    """
    import streamlit.components.v1 as components
    components.html(html_code, height=660)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: GALLERY SECTION
# ══════════════════════════════════════════════════════════════════════════════

def _render_gallery():
    st.markdown(
        '<div id="gallery" style="padding:48px 40px;background:#0a0e1a">'
        '<div style="text-align:center;margin-bottom:28px">'
        '<span class="arg-section-tag" style="background:rgba(29,158,117,0.15);color:#1D9E75">GALLERY</span>'
        '<h2 style="font-size:34px;font-weight:700;color:#fff;margin-bottom:6px">Sample AI insights</h2>'
        '<p style="font-size:19px;color:#6b7280">Real descriptions Argus generates — automatically</p></div>'
        '<div class="gallery-grid-container" style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px">'
        '<div class="arg-insight" style="background:#1a1f2e;border-left-color:#7F77DD"><p style="font-size:16px;color:#7F77DD;margin:0 0 8px;font-weight:700;letter-spacing:.04em">AGE — HEART DISEASE DATASET</p><p style="font-size:18px;color:#e5e7eb;line-height:1.7;margin:0">&ldquo;Roughly symmetric, mean 47.8 yrs. Most patients aged 40-60. No missing values — clean and ready to use.&rdquo;</p></div>'
        '<div class="arg-insight" style="background:#1a1f2e;border-left-color:#1D9E75"><p style="font-size:16px;color:#1D9E75;margin:0 0 8px;font-weight:700;letter-spacing:.04em">COUNTRY — RETAIL DATASET</p><p style="font-size:18px;color:#e5e7eb;line-height:1.7;margin:0">&ldquo;UK dominates at 91% of records. 37 other countries share the rest — high cardinality, consider grouping rare values.&rdquo;</p></div>'
        '<div class="arg-insight" style="background:#1a1f2e;border-left-color:#EF9F27"><p style="font-size:16px;color:#EF9F27;margin:0 0 8px;font-weight:700;letter-spacing:.04em">SALARY — HR DATASET</p><p style="font-size:18px;color:#e5e7eb;line-height:1.7;margin:0">&ldquo;Strongly right-skewed (skew=2.1). Most earn $40K-$70K but outliers up to $500K pull the mean. Consider log transform.&rdquo;</p></div>'
        '<div class="arg-insight" style="background:#1a1f2e;border-left-color:#E24B4A"><p style="font-size:16px;color:#E24B4A;margin:0 0 8px;font-weight:700;letter-spacing:.04em">CHOLESTEROL — MEDICAL DATASET</p><p style="font-size:18px;color:#e5e7eb;line-height:1.7;margin:0">&ldquo;154 unique values, max 603 — likely data entry error. 7.8% missing. Filter extreme values before modelling.&rdquo;</p></div>'
        '<div class="arg-insight" style="background:#1a1f2e;border-left-color:#378ADD"><p style="font-size:16px;color:#378ADD;margin:0 0 8px;font-weight:700;letter-spacing:.04em">QUANTITY — RETAIL DATASET</p><p style="font-size:18px;color:#e5e7eb;line-height:1.7;margin:0">&ldquo;10,624 negative values — likely returns. 75% of orders are 10 units or fewer. Filter negatives before revenue analysis.&rdquo;</p></div>'
        '<div class="arg-insight" style="background:#1a1f2e;border-left-color:#D4537E"><p style="font-size:16px;color:#D4537E;margin:0 0 8px;font-weight:700;letter-spacing:.04em">RATING — E-COMMERCE DATASET</p><p style="font-size:18px;color:#e5e7eb;line-height:1.7;margin:0">&ldquo;Discrete values 1-5. Rating 4 dominates at 38%. Left-skewed — customers rate high more than low overall.&rdquo;</p></div>'
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
        '<div class="about-main-grid" style="display:grid;grid-template-columns:1fr 1fr;gap:32px;align-items:center">'
        '<div>'
        '<span class="arg-section-tag" style="background:rgba(55,138,221,0.15);color:#378ADD">ABOUT</span>'
        '<h2 style="font-size:34px;font-weight:700;color:#fff;margin-bottom:12px">Why Argus?</h2>'
        '<p style="font-size:19px;color:#9ca3af;line-height:1.85;margin-bottom:14px">'
        'Exploratory data analysis is the most time-consuming and undervalued step in any data project. Argus was built to eliminate that bottleneck — giving analysts, students, and domain experts instant AI-powered insight into any dataset without writing a single line of code.</p>'
        '<p style="font-size:19px;color:#9ca3af;line-height:1.85">'
        'The name <span style="color:#f97316;font-weight:700">Argus</span> comes from the all-seeing giant of Greek mythology — a watcher with a hundred eyes. That\'s exactly what this tool does: it sees everything in your data that you might miss.</p></div>'
        '<div class="about-stats-grid" style="display:grid;grid-template-columns:1fr 1fr;gap:10px">'
        '<div class="arg-card" style="text-align:center;padding:20px"><p style="font-size:36px;font-weight:800;color:#f97316;margin:0">100%</p><p style="font-size:17px;color:#6b7280;margin-top:5px">No-code EDA</p></div>'
        '<div class="arg-card" style="text-align:center;padding:20px"><p style="font-size:36px;font-weight:800;color:#7F77DD;margin:0">Any</p><p style="font-size:17px;color:#6b7280;margin-top:5px">Domain dataset</p></div>'
        '<div class="arg-card" style="text-align:center;padding:20px"><p style="font-size:36px;font-weight:800;color:#1D9E75;margin:0">&lt;30s</p><p style="font-size:17px;color:#6b7280;margin-top:5px">Full EDA time</p></div>'
        '<div class="arg-card" style="text-align:center;padding:20px"><p style="font-size:36px;font-weight:800;color:#378ADD;margin:0">AI</p><p style="font-size:17px;color:#6b7280;margin-top:5px">Powered insights</p></div>'
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
        '<h2 style="font-size:34px;font-weight:700;color:#fff;margin-bottom:6px">Common questions</h2>'
        '<p style="font-size:19px;color:#6b7280">Everything you want to know before you upload</p></div>'
        '<div style="max-width:640px;margin:0 auto">'
        '<div class="arg-faq"><p style="font-size:19px;font-weight:600;color:#fff;margin:0 0 6px">Do I need to know Python or SQL?</p><p style="font-size:18px;color:#6b7280;line-height:1.7;margin:0">No. Argus is fully no-code. Upload your file and everything happens automatically — no commands, no scripts.</p></div>'
        '<div class="arg-faq"><p style="font-size:19px;font-weight:600;color:#fff;margin:0 0 6px">What file types are supported?</p><p style="font-size:18px;color:#6b7280;line-height:1.7;margin:0">CSV, Excel (.xlsx, .xls), and JSON. Argus auto-detects column types and data domains without any configuration.</p></div>'
        '<div class="arg-faq"><p style="font-size:19px;font-weight:600;color:#fff;margin:0 0 6px">How do the AI descriptions work?</p><p style="font-size:18px;color:#6b7280;line-height:1.7;margin:0">Argus computes statistics per column — mean, skewness, outliers, missing % — then uses an LLM to write plain-English explanations covering distribution shape, data quality, and next steps.</p></div>'
        '<div class="arg-faq"><p style="font-size:19px;font-weight:600;color:#fff;margin:0 0 6px">Will it work on my specific dataset?</p><p style="font-size:18px;color:#6b7280;line-height:1.7;margin:0">Yes. Argus has been tested on healthcare, retail, finance, HR, IoT, and marketing datasets. It auto-detects the domain and adjusts descriptions accordingly.</p></div>'
        '<div class="arg-faq"><p style="font-size:19px;font-weight:600;color:#fff;margin:0 0 6px">Is my data stored anywhere?</p><p style="font-size:18px;color:#6b7280;line-height:1.7;margin:0">No. Your data is processed in-session only and never saved to any server or database.</p></div>'
        '<div class="arg-faq" style="border-bottom:none"><p style="font-size:19px;font-weight:600;color:#fff;margin:0 0 6px">How is this different from pandas profiling?</p><p style="font-size:18px;color:#6b7280;line-height:1.7;margin:0">Pandas profiling gives you statistics. Argus gives you understanding — AI-written descriptions, smart chart titles, anomaly flags, health scores, and a chat interface. It\'s EDA plus interpretation.</p></div>'
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
        '<h2 style="font-size:38px;font-weight:800;color:#fff;margin-bottom:10px">Ready to see your data differently?</h2>'
        '<p style="font-size:20px;color:#9ca3af;margin-bottom:24px">Login or create an account below to get started.</p>'
        '<p style="font-size:17px;color:#4b5563;margin-top:16px">CSV · Excel · JSON &nbsp;|&nbsp; No code required &nbsp;|&nbsp; Free to use</p></div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

def _render_footer():
    st.markdown(
        '<div class="footer-grid-container" style="padding:20px 40px;border-top:1px solid rgba(255,255,255,0.06);display:grid;grid-template-columns:1fr auto 1fr;align-items:center;background:#0a0e1a">'
        '<div style="text-align:left"><span style="font-size:19px;font-weight:700;color:#f97316">Argus</span></div>'
        '<div style="text-align:center"><span style="font-size:17px;color:#4b5563">An Automated EDA Tool · Built with Streamlit &amp; Groq</span></div>'
        '<div style="text-align:right"><span style="font-size:17px;color:#4b5563">2025</span></div>'
        '</div>',
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
            '<h3 style="color:#fff;font-size:26px;font-weight:700;margin:0 0 4px">Login or Create Account</h3>'
            '<p style="color:#6b7280;font-size:18px;margin:0">Access the full Argus dashboard</p></div>',
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

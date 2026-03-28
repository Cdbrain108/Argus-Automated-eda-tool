"""
auth.py — Animated login/signup page for Glimpse.
Features: particle canvas, animated stat counters, floating feature cards, glowing form.
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
    _inject_css_and_canvas()
    _render_hero()
    _render_stat_counters()
    _render_feature_cards()
    st.markdown("<br>", unsafe_allow_html=True)
    _render_auth_form()
    st.markdown('<p class="tos-note">By continuing you agree to our Terms of Service</p>',
                unsafe_allow_html=True)


# ── Sections ───────────────────────────────────────────────────────────────────

def _render_hero():
    import base64
    import os
    logo_path = os.path.join(os.path.dirname(__file__), "argus_logo.png")
    try:
        with open(logo_path, "rb") as f:
            b64_logo = base64.b64encode(f.read()).decode()
    except Exception:
        b64_logo = ""

    # Animated Argus logo
    argus_logo_html = f"""
<div class="argus-logo" title="Argus EDA" style="margin-bottom:12px;">
  <img src="data:image/png;base64,{b64_logo}" alt="Argus Logo" style="width:100px; height:100px; border-radius:50%; box-shadow:0 0 24px rgba(249,115,22,0.35); animation:logoPulse 3s ease-in-out infinite;">
</div>
"""

    st.markdown(f"""
<div class="hero-wrap">
{argus_logo_html}
<div class="brand-title">Argus</div>
<div class="brand-typewriter">
<span class="tw-text" id="tw"></span><span class="tw-cursor">|</span>
</div>
</div>
""", unsafe_allow_html=True)
    
    import streamlit.components.v1 as components
    components.html("""
    <script>
    (function(){
        var phrases = [
            "Create a report from your data.",
            "Ask anything about your dataset.",
            "Discover hidden patterns instantly.",
            "Detect outliers with one click.",
            "Chat with your data using AI.",
            "Generate 10+ charts in seconds."
        ];
        var el = window.parent.document.getElementById('tw');
        if(!el) return;
        if(el.getAttribute('data-typing') === 'true') return;
        el.setAttribute('data-typing', 'true');
        
        var pi=0, ci=0, deleting=false;
        function tick(){
            var phrase = phrases[pi];
            el.textContent = deleting ? phrase.substring(0,ci--) : phrase.substring(0,ci++);
            var speed = deleting ? 40 : 70;
            if(!deleting && ci>phrase.length){ speed=1400; deleting=true; }
            if(deleting && ci<0){ deleting=false; pi=(pi+1)%phrases.length; ci=0; speed=300; }
            setTimeout(tick, speed);
        }
        tick();
    })();
    </script>
    """, height=0, width=0)


def _render_stat_counters():
    st.markdown("""
    <style>
    .stat-stars { 
        font-size: 1.8rem; 
        position: relative; 
        display: inline-block; 
        line-height: 1;
        margin-bottom: 5px;
    }
    .stat-stars-bg { color: rgba(255,255,255,0.1); }
    .stat-stars-fill {
        color: #FCD34D; 
        position: absolute; 
        top: 0; left: 0; 
        white-space: nowrap; 
        overflow: hidden;
        width: 0%; 
        animation: fillStars 1.5s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
    }
    @keyframes fillStars { from { width: 0%; } to { width: 90%; } }
    </style>
    <div class="stats-row">
        <div class="stat-box">
            <div class="stat-num" id="stat-datasets">5+</div>
            <div class="stat-label">Datasets Analyzed</div>
        </div>
        <div class="stat-box">
            <div class="stat-stars">
                <span class="stat-stars-bg">★★★★★</span>
                <div class="stat-stars-fill">★★★★★</div>
            </div>
            <div class="stat-label">User Rating (from 3+ users)</div>
        </div>
        <div class="stat-box">
            <div class="stat-num" id="stat-charts">30+</div>
            <div class="stat-label">Charts Generated</div>
        </div>
        <div class="stat-box">
            <div class="stat-num" id="stat-questions">9+</div>
            <div class="stat-label">Questions Answered</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    import streamlit.components.v1 as components
    components.html("""
    <script>
    (function(){
        try {
            const ls = window.parent.localStorage;
            if (!ls.getItem('argus_datasets')) ls.setItem('argus_datasets', '5');
            if (!ls.getItem('argus_charts')) ls.setItem('argus_charts', '30');
            if (!ls.getItem('argus_questions')) ls.setItem('argus_questions', '9');

            function animate(id, target) {
                const el = window.parent.document.getElementById(id);
                if (!el) return;
                let start = 0;
                const duration = 1500;
                const startTime = performance.now();
                
                function update() {
                    let progress = Math.min((performance.now() - startTime) / duration, 1);
                    let ease = 1 - (1 - progress) * (1 - progress);
                    let current = Math.floor(ease * target);
                    el.innerText = current + "+";
                    if (progress < 1) window.parent.requestAnimationFrame(update);
                    else el.innerText = target + "+";
                }
                window.parent.requestAnimationFrame(update);
            }

            animate('stat-datasets', parseInt(ls.getItem('argus_datasets')));
            animate('stat-charts', parseInt(ls.getItem('argus_charts')));
            animate('stat-questions', parseInt(ls.getItem('argus_questions')));
        } catch(e) { console.error('LocalStorage error:', e); }
    })();
    </script>
    """, height=0, width=0)


def _render_feature_cards():
    st.markdown("""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:0 auto 28px;max-width:900px;position:relative;z-index:1;">

  <!-- CAPABILITIES -->
  <div style="background:#12182b;border:1px solid #1e2a45;border-radius:16px;padding:28px;">
    <span style="font-size:0.65rem;font-weight:700;letter-spacing:0.1em;color:#a78bfa;background:rgba(167,139,250,0.12);border:1px solid rgba(167,139,250,0.3);border-radius:20px;padding:3px 10px;">CAPABILITIES</span>
    <h3 style="color:#fff;font-size:1.1rem;font-weight:700;margin:14px 0 18px;">What Argus auto-detects</h3>
    <div style="display:flex;flex-direction:column;gap:14px;">
      <div style="display:flex;align-items:flex-start;gap:12px;">
        <div style="width:36px;height:36px;border-radius:8px;background:rgba(56,189,248,0.15);display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:1rem;">⊞</div>
        <div><p style="color:#e2e8f0;font-weight:600;font-size:0.88rem;margin:0 0 2px;">Missing values &amp; duplicates</p><p style="color:#475569;font-size:0.76rem;margin:0;">flagged per column with severity</p></div>
      </div>
      <div style="display:flex;align-items:flex-start;gap:12px;">
        <div style="width:36px;height:36px;border-radius:8px;background:rgba(52,211,153,0.15);display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:1rem;">↗</div>
        <div><p style="color:#e2e8f0;font-weight:600;font-size:0.88rem;margin:0 0 2px;">Outliers by IQR method</p><p style="color:#475569;font-size:0.76rem;margin:0;">count, % and affected columns</p></div>
      </div>
      <div style="display:flex;align-items:flex-start;gap:12px;">
        <div style="width:36px;height:36px;border-radius:8px;background:rgba(249,115,22,0.15);display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:1rem;">⏱</div>
        <div><p style="color:#e2e8f0;font-weight:600;font-size:0.88rem;margin:0 0 2px;">Skewed distributions</p><p style="color:#475569;font-size:0.76rem;margin:0;">log-transform suggestions included</p></div>
      </div>
      <div style="display:flex;align-items:flex-start;gap:12px;">
        <div style="width:36px;height:36px;border-radius:8px;background:rgba(248,113,113,0.15);display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:1rem;">∿</div>
        <div><p style="color:#e2e8f0;font-weight:600;font-size:0.88rem;margin:0 0 2px;">Strong correlations</p><p style="color:#475569;font-size:0.76rem;margin:0;">top pairs ranked by |r| value</p></div>
      </div>
    </div>
  </div>

  <!-- COMPATIBILITY -->
  <div style="background:#12182b;border:1px solid #1e2a45;border-radius:16px;padding:28px;">
    <span style="font-size:0.65rem;font-weight:700;letter-spacing:0.1em;color:#34d399;background:rgba(52,211,153,0.12);border:1px solid rgba(52,211,153,0.3);border-radius:20px;padding:3px 10px;">COMPATIBILITY</span>
    <h3 style="color:#fff;font-size:1.1rem;font-weight:700;margin:14px 0 4px;">Works with any dataset</h3>
    <p style="color:#475569;font-size:0.78rem;margin:0 0 18px;">Drop in your file and Argus figures out the rest</p>
    <div style="display:flex;gap:10px;margin-bottom:18px;">
      <div style="flex:1;background:#1a2340;border:1px solid #2d3a55;border-radius:10px;padding:12px;text-align:center;">
        <p style="color:#38bdf8;font-weight:700;font-size:0.95rem;margin:0 0 2px;">.CSV</p><p style="color:#475569;font-size:0.68rem;margin:0;">any size</p>
      </div>
      <div style="flex:1;background:#1a2340;border:1px solid #2d3a55;border-radius:10px;padding:12px;text-align:center;">
        <p style="color:#38bdf8;font-weight:700;font-size:0.95rem;margin:0 0 2px;">.XLSX</p><p style="color:#475569;font-size:0.68rem;margin:0;">multi-sheet</p>
      </div>
      <div style="flex:1;background:#1a2340;border:1px solid #2d3a55;border-radius:10px;padding:12px;text-align:center;">
        <p style="color:#38bdf8;font-weight:700;font-size:0.95rem;margin:0 0 2px;">.JSON</p><p style="color:#475569;font-size:0.68rem;margin:0;">nested ok</p>
      </div>
    </div>
    <p style="color:#64748b;font-size:0.76rem;margin:0 0 10px;">Tested on domains:</p>
    <div style="display:flex;flex-wrap:wrap;gap:7px;">
      <span style="font-size:0.72rem;padding:3px 10px;border-radius:20px;background:rgba(56,189,248,0.1);border:1px solid rgba(56,189,248,0.25);color:#7dd3fc;">Healthcare</span>
      <span style="font-size:0.72rem;padding:3px 10px;border-radius:20px;background:rgba(52,211,153,0.1);border:1px solid rgba(52,211,153,0.25);color:#6ee7b7;">Retail &amp; Sales</span>
      <span style="font-size:0.72rem;padding:3px 10px;border-radius:20px;background:rgba(249,115,22,0.1);border:1px solid rgba(249,115,22,0.25);color:#fdba74;">Finance</span>
      <span style="font-size:0.72rem;padding:3px 10px;border-radius:20px;background:rgba(167,139,250,0.1);border:1px solid rgba(167,139,250,0.25);color:#c4b5fd;">HR &amp; People</span>
      <span style="font-size:0.72rem;padding:3px 10px;border-radius:20px;background:rgba(248,113,113,0.1);border:1px solid rgba(248,113,113,0.25);color:#fca5a5;">IoT / Sensor</span>
      <span style="font-size:0.72rem;padding:3px 10px;border-radius:20px;background:rgba(250,204,21,0.1);border:1px solid rgba(250,204,21,0.25);color:#fde68a;">Marketing</span>
    </div>
  </div>

  <!-- HOW IT WORKS -->
  <div style="background:#12182b;border:1px solid #1e2a45;border-radius:16px;padding:28px;">
    <span style="font-size:0.65rem;font-weight:700;letter-spacing:0.1em;color:#fb923c;background:rgba(249,115,22,0.12);border:1px solid rgba(249,115,22,0.3);border-radius:20px;padding:3px 10px;">HOW IT WORKS</span>
    <h3 style="color:#fff;font-size:1.1rem;font-weight:700;margin:14px 0 18px;">From upload to insight in 4 steps</h3>
    <div style="display:flex;flex-direction:column;gap:16px;">
      <div style="display:flex;align-items:flex-start;gap:14px;">
        <div style="width:28px;height:28px;border-radius:50%;background:#F97316;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.82rem;color:#fff;flex-shrink:0;">1</div>
        <div><p style="color:#e2e8f0;font-weight:600;font-size:0.88rem;margin:0 0 2px;">Upload your CSV or Excel</p><p style="color:#475569;font-size:0.76rem;margin:0;">No setup, no schema definition needed</p></div>
      </div>
      <div style="display:flex;align-items:flex-start;gap:14px;">
        <div style="width:28px;height:28px;border-radius:50%;background:#F97316;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.82rem;color:#fff;flex-shrink:0;">2</div>
        <div><p style="color:#e2e8f0;font-weight:600;font-size:0.88rem;margin:0 0 2px;">AI reads &amp; summarises your data</p><p style="color:#475569;font-size:0.76rem;margin:0;">Domain, structure, quality — in plain English</p></div>
      </div>
      <div style="display:flex;align-items:flex-start;gap:14px;">
        <div style="width:28px;height:28px;border-radius:50%;background:#F97316;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.82rem;color:#fff;flex-shrink:0;">3</div>
        <div><p style="color:#e2e8f0;font-weight:600;font-size:0.88rem;margin:0 0 2px;">10+ charts auto-generated</p><p style="color:#475569;font-size:0.76rem;margin:0;">Distributions, correlations, trends, outliers</p></div>
      </div>
      <div style="display:flex;align-items:flex-start;gap:14px;">
        <div style="width:28px;height:28px;border-radius:50%;background:#F97316;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.82rem;color:#fff;flex-shrink:0;">4</div>
        <div><p style="color:#e2e8f0;font-weight:600;font-size:0.88rem;margin:0 0 2px;">Chat with your data</p><p style="color:#475569;font-size:0.76rem;margin:0;">Ask anything, get instant AI answers</p></div>
      </div>
    </div>
  </div>

  <!-- PREVIEW -->
  <div style="background:#12182b;border:1px solid #1e2a45;border-radius:16px;padding:28px;">
    <span style="font-size:0.65rem;font-weight:700;letter-spacing:0.1em;color:#f87171;background:rgba(248,113,113,0.12);border:1px solid rgba(248,113,113,0.3);border-radius:20px;padding:3px 10px;">PREVIEW</span>
    <h3 style="color:#fff;font-size:1.1rem;font-weight:700;margin:14px 0 4px;">Sample AI insights</h3>
    <p style="color:#475569;font-size:0.78rem;margin:0 0 18px;">The kind of descriptions Argus generates automatically</p>
    <div style="display:flex;flex-direction:column;gap:12px;">
      <div style="border-left:3px solid #7c3aed;background:rgba(124,58,237,0.07);border-radius:0 8px 8px 0;padding:12px 14px;">
        <p style="color:#64748b;font-size:0.7rem;margin:0 0 5px;">Age column — heart disease dataset</p>
        <p style="color:#e2e8f0;font-size:0.82rem;font-weight:600;margin:0;">&ldquo;Roughly symmetric, mean 47.8 yrs. Most patients are 40&ndash;60. No missing values &mdash; clean and ready to use.&rdquo;</p>
      </div>
      <div style="border-left:3px solid #7c3aed;background:rgba(124,58,237,0.07);border-radius:0 8px 8px 0;padding:12px 14px;">
        <p style="color:#64748b;font-size:0.7rem;margin:0 0 5px;">Country column — retail dataset</p>
        <p style="color:#e2e8f0;font-size:0.82rem;font-weight:600;margin:0;">&ldquo;UK dominates at 91% of records. 37 other countries share the rest &mdash; high cardinality, group rare ones.&rdquo;</p>
      </div>
      <div style="border-left:3px solid #7c3aed;background:rgba(124,58,237,0.07);border-radius:0 8px 8px 0;padding:12px 14px;">
        <p style="color:#64748b;font-size:0.7rem;margin:0 0 5px;">Salary column — HR dataset</p>
        <p style="color:#e2e8f0;font-size:0.82rem;font-weight:600;margin:0;">&ldquo;Strongly right-skewed (skew=2.1). Most earn $40K&ndash;$70K but outliers up to $500K pull the mean. Consider log transform.&rdquo;</p>
      </div>
    </div>
  </div>

</div>
""", unsafe_allow_html=True)


def _render_auth_form():
    _, col, _ = st.columns([1, 2, 1])
    with col:
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
    # Set the JWT token securely into cookies
    token = _mint_jwt(email, name)
    get_cookie_controller().set("argus_jwt", token)
    
    # Authenticate locally and delay rerun allowing cookie to inject 
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


# ── CSS + Canvas ───────────────────────────────────────────────────────────────

def _inject_css_and_canvas():
    st.markdown("""
    <style>
    #MainMenu, footer, header { visibility: hidden; }
    /* ── Body */
    .stApp { background: #0A0F1E; overflow-x: hidden; }
    .block-container { padding-top: 0 !important; max-width: 100% !important; }
    /* ── Particle Canvas */
    #particle-canvas {
        position: fixed; top: 0; left: 0;
        width: 100%; height: 100%;
        pointer-events: none; z-index: 0;
    }
    /* ── Hero */
    .hero-wrap { text-align: center; padding: 52px 0 20px; position: relative; z-index: 1; }
    .brand-icon  { font-size: 64px; animation: float 3s ease-in-out infinite; }
    .argus-logo  { display:inline-flex; justify-content:center; }
    .argus-logo svg { filter:drop-shadow(0 0 14px rgba(249,115,22,0.6)); animation:logoPulse 3s ease-in-out infinite; }
    .argus-ring  { animation:rotateSlow 12s linear infinite; transform-origin:32px 32px; }
    .argus-eye   { animation:eyePulse 2.5s ease-in-out infinite; }
    .argus-scan  { animation:scanFade 2s ease-in-out infinite alternate; }
    @keyframes rotateSlow { to { transform: rotate(360deg); } }
    @keyframes eyePulse   { 0%,100%{opacity:0.9} 50%{opacity:1} }
    @keyframes scanFade   { from{opacity:0.2} to{opacity:0.8} }
    @keyframes logoPulse  { 0%,100%{filter:drop-shadow(0 0 10px rgba(249,115,22,0.45))} 50%{filter:drop-shadow(0 0 22px rgba(249,115,22,0.85))} }
    .brand-title {
        font-size: 3.2rem; font-weight: 900; letter-spacing:-1px;
        background: linear-gradient(90deg, #F97316 0%, #FB923C 50%, #FCD34D 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-size: 200%; animation: shimmer 3s linear infinite;
    }
    .brand-typewriter { font-size: 1.05rem; color: #94A3B8; min-height: 28px; margin-top: 6px; }
    .tw-cursor { color: #F97316; animation: blink 1s step-end infinite; }
    @keyframes float   { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-10px)} }
    @keyframes shimmer { 0%{background-position:0%} 100%{background-position:200%} }
    @keyframes blink   { 0%,100%{opacity:1} 50%{opacity:0} }
    /* ── Stats Row */
    .stats-row {
        display: flex; justify-content: center; gap: 20px;
        margin: 28px auto 0; max-width: 760px; position: relative; z-index: 1;
        flex-wrap: wrap;
    }
    .stat-box {
        background: rgba(249,115,22,0.08); border: 1px solid rgba(249,115,22,0.25);
        border-radius: 16px; padding: 18px 28px; text-align: center; min-width: 140px;
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .stat-box:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 28px rgba(249,115,22,0.25);
    }
    .stat-num   { font-size: 1.9rem; font-weight: 800; color: #F97316; }
    .stat-label { font-size: 0.78rem; color: #64748B; margin-top: 4px; }
    /* ── Feature Cards */
    .feature-row {
        display: flex; justify-content: center; gap: 16px;
        margin: 32px auto; max-width: 860px; position: relative; z-index: 1;
        flex-wrap: wrap;
    }
    .feat-card {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px; padding: 22px 20px; min-width: 168px; max-width: 200px;
        text-align: center; transition: all 0.35s; cursor: default;
        position: relative; overflow: hidden;
    }
    .feat-card::before {
        content: ''; position: absolute; inset: 0;
        background: radial-gradient(circle at 50% 0%, rgba(249,115,22,0.12), transparent 70%);
        opacity: 0; transition: opacity 0.35s;
    }
    .feat-card:hover { transform: translateY(-6px) scale(1.03);
                       border-color: rgba(249,115,22,0.4);
                       box-shadow: 0 12px 36px rgba(249,115,22,0.18); }
    .feat-card:hover::before { opacity: 1; }
    .feat-icon  { font-size: 1.9rem; margin-bottom: 8px; }
    .feat-title { font-weight: 700; color: #E2E8F0; font-size: 0.95rem; margin-bottom: 4px; }
    .feat-desc  { font-size: 0.75rem; color: #64748B; line-height: 1.5; }
    /* ── Auth Tabs */
    [data-testid="stTabs"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(249,115,22,0.22);
        border-radius: 20px; padding: 24px 28px 20px;
        backdrop-filter: blur(14px); position: relative; z-index: 1;
    }
    [data-testid="stTab"] { color: #94A3B8 !important; font-weight: 500; }
    [data-testid="stTab"][aria-selected="true"] { color: #F97316 !important; }
    /* ── Inputs */
    [data-testid="stTextInput"] input {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(249,115,22,0.28) !important;
        border-radius: 10px !important; color: #E2E8F0 !important; padding: 11px 14px !important;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: #F97316 !important;
        box-shadow: 0 0 0 3px rgba(249,115,22,0.18) !important;
        outline: none !important;
    }
    /* ── Submit buttons */
    [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #F97316, #EA580C) !important;
        color: #fff !important; border: none !important;
        border-radius: 10px !important; font-size: 1rem !important;
        font-weight: 700 !important; padding: 12px !important;
        box-shadow: 0 0 24px rgba(249,115,22,0.35) !important;
        transition: all 0.25s !important;
    }
    [data-testid="stFormSubmitButton"] > button:hover {
        box-shadow: 0 0 42px rgba(249,115,22,0.6) !important;
        transform: translateY(-2px) !important;
    }
    .tos-note { font-size:0.72rem; color:#334155; text-align:center; margin-top:16px; }
    </style>
    <!-- Particle Canvas -->
    <canvas id="particle-canvas"></canvas>
    """, unsafe_allow_html=True)
    import streamlit.components.v1 as components
    components.html("""
    <script>
    (function() {
        const doc = window.parent.document;
        const win = window.parent;
        const canvas = doc.getElementById("particle-canvas");
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        canvas.width  = win.innerWidth;
        canvas.height = win.innerHeight;
        win.addEventListener("resize", () => {
            canvas.width  = win.innerWidth;
            canvas.height = win.innerHeight;
        });
        const COLORS = ["#F97316","#FB923C","#FCD34D","#94A3B8","#38BDF8"];
        const particles = Array.from({length: 90}, () => ({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            r: Math.random() * 2 + 0.4,
            dx: (Math.random() - 0.5) * 0.55,
            dy: (Math.random() - 0.5) * 0.55,
            color: COLORS[Math.floor(Math.random() * COLORS.length)],
            alpha: Math.random() * 0.5 + 0.15
        }));
        function drawLines() {
            for (let i = 0; i < particles.length; i++) {
                for (let j = i+1; j < particles.length; j++) {
                    const dx = particles[i].x - particles[j].x;
                    const dy = particles[i].y - particles[j].y;
                    const d  = Math.sqrt(dx*dx + dy*dy);
                    if (d < 110) {
                        ctx.beginPath();
                        ctx.strokeStyle = `rgba(249,115,22,${0.12 * (1 - d/110)})`;
                        ctx.lineWidth = 0.6;
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.stroke();
                    }
                }
            }
        }
        function animate() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            drawLines();
            particles.forEach(p => {
                p.x += p.dx; p.y += p.dy;
                if (p.x < 0 || p.x > canvas.width)  p.dx *= -1;
                if (p.y < 0 || p.y > canvas.height) p.dy *= -1;
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r, 0, Math.PI*2);
                ctx.fillStyle = p.color.replace(")", `,${p.alpha})`).replace("rgb","rgba");
                // simple hex to rgba
                ctx.globalAlpha = p.alpha;
                ctx.fillStyle = p.color;
                ctx.fill();
                ctx.globalAlpha = 1;
            });
            win.requestAnimationFrame(animate);
        }
        animate();
    })();
    </script>
    """, height=0, width=0)

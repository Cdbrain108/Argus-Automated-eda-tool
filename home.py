"""
home.py — Unified Argus EDA Dashboard.
Combines: data overview, cleaning, univariate, bivariate, feature importance, AI chat.
Generated PDFs and images are displayed inline as scrollable content.
"""

import os, base64, time, warnings, json, io, re as _re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_file, run_eda, chat_response, get_groq_client


# ── Shared helpers ────────────────────────────────────────────────────────────


def apply_dark_layout(fig):
    """Apply consistent dark theme to all Plotly figures."""
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9ca3af", size=12),
        xaxis=dict(showgrid=False, color="#9ca3af"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#9ca3af"),
        margin=dict(l=10, r=10, t=30, b=10),
    )


def groq_call(prompt: str, max_tokens: int = 300) -> str:
    """Single centralised Groq API call with key rotation."""
    client = get_groq_client()
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.3,
        stream=False,
    )
    return resp.choices[0].message.content.strip()


def show_ai_insight(prompt_text: str, cache_key: str):
    """Show an AI insight block, cached in session_state['ai_cache']."""
    if "ai_cache" not in st.session_state:
        st.session_state["ai_cache"] = {}
    if cache_key in st.session_state["ai_cache"]:
        text = st.session_state["ai_cache"][cache_key]
    else:
        try:
            text = groq_call(prompt_text, max_tokens=200)
        except Exception:
            text = "Analysis unavailable."
        st.session_state["ai_cache"][cache_key] = text
    st.markdown(
        f'<div style="border-left:3px solid #7F77DD;padding:10px 16px;'
        f"border-radius:0 8px 8px 0;background:rgba(127,119,221,0.08);"
        f'font-size:13px;line-height:1.7;color:#9ca3af;margin-top:10px">'
        f"{text}</div>",
        unsafe_allow_html=True,
    )


warnings.filterwarnings("ignore")


# ── dark_card helper ────────────────────────────────────────────────────────────


def dark_card(content_html, badge_text=None, badge_color=None):
    badge_html = ""
    if badge_text:
        badge_html = (
            f'<span style="display:inline-flex;align-items:center;'
            f"font-size:14px;padding:3px 8px;border-radius:20px;font-weight:600;"
            f"letter-spacing:.04em;background:{badge_color}22;"
            f'color:{badge_color};margin-bottom:8px">{badge_text}</span><br>'
        )
    return (
        f'<div style="background:#1a1f2e;border:1px solid #2d3748;'
        f"border-radius:14px;padding:18px 20px;margin-bottom:12px;"
        f'min-height:200px">'
        f"{badge_html}{content_html}"
        f"</div>"
    )


# ── compute_all_widgets (cached per-upload) ──────────────────────────────────────


@st.cache_data(show_spinner=False)
def compute_all_widgets(df_json_hash: str, df_json: str) -> dict:
    """Compute all Overview widget data once and cache by dataframe hash."""
    import hashlib

    try:
        import scipy.stats as scipy_stats

        _has_scipy = True
    except ImportError:
        _has_scipy = False

    df = pd.read_json(df_json)
    results = {}

    # --- health score ---
    missing_penalty = min(40, round(df.isnull().mean().mean() * 100 * 2))
    dup_penalty = min(20, round(df.duplicated().sum() / max(len(df), 1) * 100 * 2))
    num_cols = df.select_dtypes(include="number").columns.tolist()
    outlier_penalties = []
    for col in num_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            pct = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).mean() * 100
            outlier_penalties.append(pct)
    outlier_penalty = (
        min(40, round(sum(outlier_penalties) / len(outlier_penalties)))
        if outlier_penalties
        else 0
    )
    results["health_score"] = max(
        0, 100 - missing_penalty - dup_penalty - outlier_penalty
    )
    results["missing_penalty"] = missing_penalty
    results["dup_penalty"] = dup_penalty
    results["outlier_penalty"] = outlier_penalty

    # --- column breakdown ---
    results["n_numeric"] = len(df.select_dtypes(include="number").columns)
    results["n_categorical"] = len(
        df.select_dtypes(include=["object", "category"]).columns
    )
    results["n_datetime"] = len(df.select_dtypes(include="datetime").columns)
    results["n_cols"] = len(df.columns)
    results["n_rows"] = len(df)

    # --- missing heatmap ---
    results["missing_pct_per_col"] = (df.isnull().mean() * 100).round(1).to_dict()

    # --- skewness ---
    skew_scores = {}
    for col in num_cols:
        clean = df[col].dropna()
        if len(clean) > 10:
            try:
                val = (
                    float(scipy_stats.skew(clean))
                    if _has_scipy
                    else float(clean.skew())
                )
                skew_scores[col] = round(val, 2)
            except Exception:
                pass
    results["skew_scores"] = dict(
        sorted(skew_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    )

    # --- outliers ---
    outlier_details = {}
    for col in num_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        count = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        if count > 0:
            outlier_details[col] = count
    results["outlier_details"] = dict(
        sorted(outlier_details.items(), key=lambda x: x[1], reverse=True)
    )
    if num_cols:
        mask = (
            df[num_cols]
            .apply(
                lambda c: (
                    (c < c.quantile(0.25) - 1.5 * (c.quantile(0.75) - c.quantile(0.25)))
                    | (
                        c
                        > c.quantile(0.75) + 1.5 * (c.quantile(0.75) - c.quantile(0.25))
                    )
                )
            )
            .any(axis=1)
        )
        results["total_outlier_rows"] = int(mask.sum())
        results["outlier_pct"] = round(mask.mean() * 100, 1)
    else:
        results["total_outlier_rows"] = 0
        results["outlier_pct"] = 0.0
    results["cols_affected"] = len(outlier_details)

    # --- correlations ---
    results["top_pairs"] = []
    if len(num_cols) >= 2:
        try:
            corr = df[num_cols].corr()
            pairs = []
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    r = corr.iloc[i, j]
                    if not pd.isna(r):
                        pairs.append((num_cols[i], num_cols[j], round(float(r), 2)))
            results["top_pairs"] = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[
                :5
            ]
        except Exception:
            pass

    return results


# ── Optional backend module imports ───────────────────────────────────────────
try:
    from data_cleaning import SmartDataCleaner

    _HAS_CLEANING = True
except Exception:
    _HAS_CLEANING = False

try:
    from text_generation import AI as GlimpseAI

    _HAS_AI = True
except Exception:
    _HAS_AI = False

try:
    from univariate import uni_analyze_and_visualize

    _HAS_UNIVARIATE = True
except Exception:
    _HAS_UNIVARIATE = False

# ── Constants ─────────────────────────────────────────────────────────────────
from utils import get_groq_client

UNI_PDF_KEY = "uni_pdf_bytes"
BI_PDF_KEY = "bi_pdf_bytes"

SUGGESTIONS = [
    "Summarize all AI insights generated so far",
    "Which columns have the most outliers?",
    "Show correlation between columns",
    "Identify missing values",
]


# ── Main entry ────────────────────────────────────────────────────────────────


def show_home_page(guest: bool = False):
    _inject_css()
    _render_header()

    # ── Rating modal for real logged-in users (after 45 seconds) ──────────────
    if st.session_state.get("logged_in"):
        if "session_start" not in st.session_state:
            st.session_state["session_start"] = time.time()
        elapsed = time.time() - st.session_state.get("session_start", time.time())
        if (
            elapsed > 45
            and not st.session_state.get("rating_shown")
            and not st.session_state.get("rating_dismissed")
        ):
            if not st.session_state.get("show_rating_modal"):
                st.toast("⭐ Help us out by rating Argus in the sidebar!", icon="🚀")
            st.session_state["show_rating_modal"] = True

    if st.session_state.get("show_rating_modal"):
        _render_sidebar_rating()

    if "df" not in st.session_state:
        if guest:
            st.markdown('<div class="section-title">👋 Welcome to Trial Mode</div>', unsafe_allow_html=True)
            st.info("Login to upload your own custom datasets. For now, explore Argus using our natively integrated datasets!")
            # ── Mobile-friendly: stack buttons vertically ──
            st.markdown('<div class="demo-btn-stack">', unsafe_allow_html=True)
            if st.button("🚢 Load Titanic Dataset", use_container_width=True, type="primary", key="guest_titanic_btn"):
                _process_demo_selection("Titanic Dataset", "demo_data.csv", is_guest=True)
            if st.button("🚲 Load Daily Bike Share", use_container_width=True, type="primary", key="guest_bike_btn"):
                _process_demo_selection("Daily Bike Share", "daily-bike-share.csv", is_guest=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Prevent tabs rendering until user selects a dataset
            st.stop()
        else:
            st.markdown('<div class="section-title">📂 Upload Your Data</div>', unsafe_allow_html=True)
            _render_upload_widget()
    else:
        if not guest:
            st.markdown('<div class="section-title">📂 Upload Your Data</div>', unsafe_allow_html=True)
            _render_upload_widget()
        # ── One-time legacy cleanup ──
        for f in ["Uni_variate_output1.pdf", "Bi_variate_output.pdf"]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass

        # ── Instant data summary right after upload
        _render_data_summary(st.session_state["df"])
        st.markdown("---")
        _render_main_tabs(guest=guest)

    if st.session_state.pop("just_uploaded", False):
        import streamlit.components.v1 as components

        components.html(
            """<script>
        try {
            const ls = window.parent.localStorage;
            ls.setItem('argus_datasets', parseInt(ls.getItem('argus_datasets') || '5') + 1);
            ls.setItem('argus_charts', parseInt(ls.getItem('argus_charts') || '30') + 10);
        } catch(e) {}
        </script>""",
            height=0,
            width=0,
        )

    if st.session_state.pop("just_asked_question", False):
        import streamlit.components.v1 as components

        components.html(
            """<script>
        try {
            const ls = window.parent.localStorage;
            ls.setItem('argus_questions', parseInt(ls.getItem('argus_questions') || '9') + 1);
        } catch(e) {}
        </script>""",
            height=0,
            width=0,
        )


# ── Demo dataset loader ───────────────────────────────────────────────────────


def _process_demo_selection(dataset_name: str, file_name: str, is_guest: bool):
    """Load the user-selected demo dataset and seamlessly inject it into the app."""
    demo_path = os.path.join(os.path.dirname(__file__), file_name)
    if not os.path.exists(demo_path):
        st.error(f"Demo dataset '{file_name}' not found on the server.")
        return
    try:
        with st.spinner(f"Loading {dataset_name} & running AI discovery..."):
            df_raw = pd.read_csv(demo_path)
            from ai_encoder import ai_encode_dataframe

            df, final_name = ai_encode_dataframe(df_raw, dataset_name)
            eda = run_eda(df)
            
            chat_msg = (
                f"👋 Welcome to the **{dataset_name} Demo**! Explore the Overview and Univariate tabs. Login to unlock all features safely with your own dataset!"
                if is_guest else
                f"✅ **EDA complete for `{dataset_name}`!**  \nFound **{eda['rows']:,} rows** and **{eda['columns']} columns**. Explore the tabs below or ask me anything!"
            )
            
            st.session_state.update(
                df=df,
                eda_result=eda,
                file_name=file_name,
                dataset_name=final_name,
                just_uploaded=True,
                chat_history=[{"role": "assistant", "content": chat_msg}],
            )
            import hashlib

            df_json = df.to_json()
            df_hash = hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()
            st.session_state["w"] = compute_all_widgets(df_hash, df_json)
        st.rerun()
    except Exception as ex:
        # Trim the error to avoid dumping raw JSON / huge tracebacks
        err_str = str(ex)
        short_err = err_str[:120] + "..." if len(err_str) > 120 else err_str
        st.error(f"❌ Failed to load **{dataset_name}**. Please try again or choose another dataset.\n\n_Details: {short_err}_")


# ── Upload widget ─────────────────────────────────────────────────────────────


def _render_upload_widget():
    uploaded = st.file_uploader(
        label="Drop your Excel or CSV file here",
        type=["xlsx", "xls", "csv"],
        label_visibility="collapsed",
        help="Supported: .xlsx, .xls, .csv (max 200 MB)",
        key="file_uploader",
    )
    if uploaded and "df" not in st.session_state:
        with st.spinner("Processing file & AI decoding numeric categories..."):
            _show_processing_bar()
            df_raw = load_file(uploaded)
            dataset_name = uploaded.name.rsplit(".", 1)[0]

            # ── AI Encoding Layer ──
            from ai_encoder import ai_encode_dataframe

            df, dataset_name = ai_encode_dataframe(df_raw, dataset_name)

            eda = run_eda(df)
            st.session_state.update(
                df=df,
                eda_result=eda,
                file_name=uploaded.name,
                dataset_name=dataset_name,
                just_uploaded=True,
                chat_history=[
                    {
                        "role": "assistant",
                        "content": (
                            f"✅ **EDA complete for `{dataset_name}`!**  \n"
                            f"Found **{eda['rows']:,} rows** and **{eda['columns']} columns**. "
                            "Explore the tabs below or ask me anything!"
                        ),
                    }
                ],
            )
            # ── Pre-compute all Overview widget data once ──
            import hashlib

            df_json = df.to_json()
            df_hash = hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()
            st.session_state["w"] = compute_all_widgets(df_hash, df_json)
        st.rerun()

    if "df" not in st.session_state:
        st.markdown('<p style="text-align:center; color:#94A3B8; margin-top:35px; font-size: 0.9rem; font-weight:600; letter-spacing: 0.5px;">— OR EXPLORE A DEMO DATASET —</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚢 Load Titanic Dataset", use_container_width=True):
                _process_demo_selection("Titanic Dataset", "demo_data.csv", is_guest=False)
        with col2:
            if st.button("🚲 Load Daily Bike Share", use_container_width=True):
                _process_demo_selection("Daily Bike Share", "daily-bike-share.csv", is_guest=False)


# ── Tabbed dashboard (post-upload) ────────────────────────────────────────────


def _render_main_tabs(guest: bool = False):
    df = st.session_state["df"]
    eda = st.session_state["eda_result"]

    tab_labels = [
        "📊  Overview",
        "🧹  Data Cleaning",
        "📈  Univariate",
        "📊  Categorical",
        "🔗  Bivariate",
        "🌐  Multivariate",
        "🌟  Feature Importance",
        "💬  AI Chat",
    ]
    t1, t2, t3_num, t3_cat, t4, t_multi, t5, t6 = st.tabs(tab_labels)

    with t1:
        _render_overview_tab(df, eda)
    with t2:
        _render_cleaning_tab(df)
    with t3_num:
        _render_univariate_tab(mode="numeric")
    with t3_cat:
        _render_univariate_tab(mode="categorical")
    with t4:
        if guest:
            _render_guest_lock_overlay("Bivariate Analysis")
        else:
            _render_bivariate_tab()
    with t_multi:
        if guest:
            _render_guest_lock_overlay("Multivariate Analysis")
        else:
            _render_multivariate_tab()
    with t5:
        if guest:
            _render_guest_lock_overlay("Feature Importance")
        else:
            _render_feature_tab()
    with t6:
        if guest:
            _render_guest_lock_overlay("AI Chat")
        else:
            _render_chat_tab(eda)

    st.markdown("<br>", unsafe_allow_html=True)
    if guest:
        # Guest: show exit demo button instead of clear
        if st.button("🏠  Exit Demo — Go to Login", key="exit_guest_btn"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
    else:
        if st.button("🗑  Clear — upload a new file", key="clear_btn"):
            keep = {"logged_in", "user_name", "user_email"}
            for k in [k for k in st.session_state if k not in keep]:
                del st.session_state[k]
            st.rerun()


# ── Guest Lock Overlay ────────────────────────────────────────────────────────


def _render_guest_lock_overlay(feature_name: str):
    """Render a robot chatbot lock screen for guest-mode locked tabs."""
    if st.button("🏠  Exit Demo — Go to Login", key=f"exit_guest_{feature_name}"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
    st.markdown(
        f"""
        <div style="
            display:flex;flex-direction:column;align-items:center;justify-content:center;
            padding:60px 40px;text-align:center;
            background:linear-gradient(135deg,rgba(127,119,221,0.06),rgba(249,115,22,0.04));
            border:1px solid rgba(127,119,221,0.15);border-radius:20px;margin:20px 0;
        ">
            <div style="font-size:72px;margin-bottom:16px;animation:bob 2s ease-in-out infinite;">🤖</div>
            <h2 style="color:#fff;font-size:26px;font-weight:700;margin:0 0 10px">
                {feature_name} is locked
            </h2>
            <p style="color:#9ca3af;font-size:17px;max-width:440px;line-height:1.8;margin:0 0 24px">
                You're currently in <strong style="color:#f97316">Guest Demo Mode</strong> exploring the Titanic dataset.
                Login or create a free account to unlock <strong>{feature_name}</strong> and run all analyses on your own dataset!
            </p>
            <div style="display:flex;gap:12px;flex-wrap:wrap;justify-content:center;">
                <a href="/" onclick="window.parent.location.reload();" style="
                    background:linear-gradient(135deg,#f97316,#ea580c);
                    color:#fff;padding:12px 28px;border-radius:12px;
                    font-weight:700;font-size:16px;text-decoration:none;
                    box-shadow:0 8px 20px rgba(249,115,22,0.4);
                ">🔑 Login / Sign Up</a>
            </div>
            <p style="color:#4b5563;font-size:14px;margin-top:20px">
                Free forever · No credit card · Upload any CSV or Excel
            </p>
        </div>
        <style>
        @keyframes bob {{
            0%, 100% {{ transform: translateY(0); }}
            50% {{ transform: translateY(-10px); }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Rating Modal ──────────────────────────────────────────────────────────────


def _render_sidebar_rating():
    """Render a clean rating experience in the sidebar to ensure full interactivity."""
    with st.sidebar:
        st.markdown(
            """
            <div style="background:rgba(249,115,22,0.1); border:1px solid rgba(249,115,22,0.3); padding:1.5rem; border-radius:15px; margin-bottom:1.5rem;">
                <h3 style="color:#F97316; margin:0; font-size:1.1rem;">⭐ Enjoying Argus?</h3>
                <p style="color:#9CA3AF; font-size:0.9rem; margin-top:0.5rem; margin-bottom:0;">Rate your experience below!</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        stars = st.select_slider(
            "Your rating:",
            options=[1, 2, 3, 4, 5],
            value=5,
            format_func=lambda x: "⭐" * x,
            key="rating_stars_input",
        )
        feedback = st.text_input(
            "Optional feedback",
            key="rating_feedback_input",
            placeholder="e.g. Love the AI insights!",
        )

        if st.button("Submit ✨", key="rating_submit_btn", use_container_width=True):
            try:
                from ratings import save_rating
                save_rating(
                    email=st.session_state.get("user_email", "guest"),
                    name=st.session_state.get("user_name", "User"),
                    stars=stars,
                    feedback=feedback,
                )
                st.success("Thanks! 🎉")
                time.sleep(1.5)
            except Exception:
                pass
            st.session_state["rating_shown"] = True
            st.session_state["show_rating_modal"] = False
            st.rerun()

        # ── Subtle dismiss button at the bottom
        st.markdown("<div style='text-align:center; margin-top:12px;'>", unsafe_allow_html=True)
        if st.button("Maybe later", key="rating_dismiss_btn", use_container_width=False):
            st.session_state["rating_dismissed"] = True
            st.session_state["show_rating_modal"] = False
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# ── Tab 1 · Overview ──────────────────────────────────────────────────────────


def _render_overview_tab(df: pd.DataFrame, eda: dict):
    w = st.session_state.get("w", {})
    if not w:
        st.info("Upload a file first")
        return

    col_a, col_b = st.columns(2)

    # ──────────────────────── LEFT COLUMN ────────────────────────
    with col_a:
        # ── WIDGET 1: Health Score ──────────────────────────────────────────
        score = w["health_score"]
        score_color = (
            "#1D9E75" if score >= 75 else "#EF9F27" if score >= 50 else "#E24B4A"
        )
        dash = round(score / 100 * 176)

        def penalty_bar(label, value, max_val=40):
            pct = min(100, round(value / max_val * 100))
            bar_color = (
                "#1D9E75" if value == 0 else "#EF9F27" if value < 20 else "#E24B4A"
            )
            return (
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:7px">'
                f'<span style="font-size:14px;color:#9ca3af;min-width:100px">{label}</span>'
                f'<div style="flex:1;height:8px;border-radius:4px;background:#2d3748">'
                f'<div style="width:{pct}%;height:100%;border-radius:4px;background:{bar_color}"></div></div>'
                f'<span style="font-size:14px;color:{bar_color};min-width:30px;text-align:right">-{value}</span>'
                f"</div>"
            )

        st.markdown(
            dark_card(
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px">'
                f"<div>"
                f'<p style="font-size:18px;font-weight:600;color:#fff;margin:0">Dataset Health Score</p>'
                f'<p style="font-size:14px;color:#6b7280;margin:6px 0 0">auto-computed on upload</p>'
                f"</div>"
                f'<div style="position:relative;width:90px;height:90px">'
                f'<svg width="90" height="90" viewBox="0 0 72 72">'
                f'<circle cx="36" cy="36" r="28" fill="none" stroke="#2d3748" stroke-width="7"/>'
                f'<circle cx="36" cy="36" r="28" fill="none" stroke="{score_color}" stroke-width="7"'
                f' stroke-dasharray="{dash} 176" stroke-dashoffset="44" stroke-linecap="round"'
                f' transform="rotate(-90 36 36)"/>'
                f"</svg>"
                f'<div style="position:absolute;inset:0;display:flex;flex-direction:column;'
                f"align-items:center;justify-content:center;font-size:24px;font-weight:700;"
                f'color:{score_color}">{score}</div>'
                f"</div></div>"
                + penalty_bar("Missing", w["missing_penalty"])
                + penalty_bar("Duplicates", w["dup_penalty"], 20)
                + penalty_bar("Outliers", w["outlier_penalty"])
                + '<p style="font-size:13px;color:#6b7280;margin:12px 0 0;font-style:italic">'
                "Score = 100 minus missing, duplicate and outlier penalties</p>",
                "HEALTH",
                "#1D9E75",
            ),
            unsafe_allow_html=True,
        )

        # ── WIDGET 3: Missing Value Heatmap ──────────────────────────────────
        missing_dict = w["missing_pct_per_col"]

        def miss_color(pct):
            if pct == 0:
                return "#1D9E75"
            elif pct < 10:
                return "#FAC775"
            elif pct < 50:
                return "#EF9F27"
            else:
                return "#E24B4A"

        squares = "".join(
            [
                f'<div style="display:flex;flex-direction:column;align-items:center;gap:4px" title="{col} — {pct}% missing">'
                f'<div style="width:36px;height:36px;border-radius:6px;background:{miss_color(pct)}"></div>'
                f'<span style="font-size:12px;color:#6b7280">{col[:5]}</span>'
                f"</div>"
                for col, pct in missing_dict.items()
            ]
        )

        all_clean = all(v == 0 for v in missing_dict.values())
        if all_clean:
            chart_html = (
                '<div style="display:flex;align-items:center;justify-content:center;'
                'height:90px;font-size:16px;color:#1D9E75">All columns are clean — no missing values</div>'
            )
        else:
            chart_html = f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px">{squares}</div>'

        legend = (
            '<div style="display:flex;gap:12px;align-items:center;margin-top:10px">'
            '<span style="font-size:13px;color:#6b7280">Legend:</span>'
            '<span style="font-size:13px;color:#1D9E75"><span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:#1D9E75;margin-right:4px"></span>Clean</span>'
            '<span style="font-size:13px;color:#FAC775"><span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:#FAC775;margin-right:4px"></span>&lt;10%</span>'
            '<span style="font-size:13px;color:#EF9F27"><span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:#EF9F27;margin-right:4px"></span>10-50%</span>'
            '<span style="font-size:13px;color:#E24B4A"><span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:#E24B4A;margin-right:4px"></span>&gt;50%</span>'
            "</div>"
        )

        st.markdown(
            dark_card(
                '<p style="font-size:18px;font-weight:600;color:#fff;margin:0 0 6px">Missing Value Map</p>'
                '<p style="font-size:14px;color:#6b7280;margin:0 0 16px">per-column severity at a glance</p>'
                + chart_html
                + legend,
                "QUALITY",
                "#E24B4A",
            ),
            unsafe_allow_html=True,
        )

        # ── WIDGET 5: Outlier Summary ─────────────────────────────────────────
        od = w["outlier_details"]
        top5_outliers = list(od.items())[:5]

        def out_color(count, total):
            pct = count / max(total, 1) * 100
            return "#E24B4A" if pct > 20 else "#EF9F27" if pct > 10 else "#FAC775"

        rows_html = (
            "".join(
                [
                    f'<div style="display:flex;justify-content:space-between;font-size:14px;margin-bottom:8px">'
                    f'<span style="color:#9ca3af">{col}</span>'
                    f'<span style="color:{out_color(cnt, w["n_rows"])}"> {cnt:,} outliers</span>'
                    f"</div>"
                    for col, cnt in top5_outliers
                ]
            )
            if top5_outliers
            else '<p style="font-size:15px;color:#1D9E75;margin:0">No outliers detected</p>'
        )

        st.markdown(
            dark_card(
                '<p style="font-size:18px;font-weight:600;color:#fff;margin:0 0 6px">Outlier Detection</p>'
                '<p style="font-size:14px;color:#6b7280;margin:0 0 16px">IQR method across all numeric columns</p>'
                '<div style="display:flex;gap:12px;margin-bottom:16px">'
                f'<div style="flex:1;background:#E24B4A11;border:1px solid #E24B4A33;border-radius:8px;padding:14px;text-align:center">'
                f'<p style="font-size:28px;font-weight:700;color:#E24B4A;margin:0">{w["total_outlier_rows"]:,}</p>'
                f'<p style="font-size:14px;color:#6b7280;margin:4px 0 0">Outlier rows</p></div>'
                f'<div style="flex:1;background:#EF9F2711;border:1px solid #EF9F2733;border-radius:8px;padding:14px;text-align:center">'
                f'<p style="font-size:28px;font-weight:700;color:#EF9F27;margin:0">{w["outlier_pct"]}%</p>'
                f'<p style="font-size:14px;color:#6b7280;margin:4px 0 0">Of dataset</p></div>'
                f'<div style="flex:1;background:#7F77DD11;border:1px solid #7F77DD33;border-radius:8px;padding:14px;text-align:center">'
                f'<p style="font-size:28px;font-weight:700;color:#7F77DD;margin:0">{w["cols_affected"]}</p>'
                f'<p style="font-size:14px;color:#6b7280;margin:4px 0 0">Cols affected</p></div>'
                "</div>" + rows_html,
                "OUTLIERS",
                "#E24B4A",
            ),
            unsafe_allow_html=True,
        )

    # ─────────────────────── RIGHT COLUMN ───────────────────────
    with col_b:
        # ── WIDGET 2: Column Breakdown ──────────────────────────────────────
        nn = w["n_numeric"]
        nc2 = w["n_categorical"]
        nd = w["n_datetime"]
        total = w["n_cols"]
        pn = round(nn / max(total, 1) * 100)
        pc2 = round(nc2 / max(total, 1) * 100)
        pd_ = round(nd / max(total, 1) * 100)

        st.markdown(
            dark_card(
                '<p style="font-size:18px;font-weight:600;color:#fff;margin:0 0 16px">Column Breakdown</p>'
                '<div style="display:flex;gap:10px;margin-bottom:16px">'
                f'<div style="flex:1;background:#7F77DD22;border:1px solid #7F77DD44;border-radius:8px;padding:16px;text-align:center">'
                f'<p style="font-size:32px;font-weight:700;color:#7F77DD;margin:0">{nn}</p>'
                f'<p style="font-size:14px;color:#6b7280;margin:6px 0 0">Numeric</p></div>'
                f'<div style="flex:1;background:#1D9E7522;border:1px solid #1D9E7544;border-radius:8px;padding:16px;text-align:center">'
                f'<p style="font-size:32px;font-weight:700;color:#1D9E75;margin:0">{nc2}</p>'
                f'<p style="font-size:14px;color:#6b7280;margin:6px 0 0">Categorical</p></div>'
                f'<div style="flex:1;background:#378ADD22;border:1px solid #378ADD44;border-radius:8px;padding:16px;text-align:center">'
                f'<p style="font-size:32px;font-weight:700;color:#378ADD;margin:0">{nd}</p>'
                f'<p style="font-size:14px;color:#6b7280;margin:6px 0 0">Datetime</p></div>'
                "</div>"
                f'<div style="display:flex;height:12px;border-radius:6px;overflow:hidden;gap:3px">'
                f'<div style="flex:{max(nn, 1)};background:#7F77DD;border-radius:6px 0 0 6px"></div>'
                f'<div style="flex:{max(nc2, 1)};background:#1D9E75"></div>'
                f'<div style="flex:{max(nd, 1)};background:#378ADD;border-radius:0 6px 6px 0"></div>'
                f"</div>"
                f'<div style="display:flex;gap:16px;margin-top:10px">'
                f'<span style="font-size:14px;color:#7F77DD">{pn}% numeric</span>'
                f'<span style="font-size:14px;color:#1D9E75">{pc2}% categorical</span>'
                f'<span style="font-size:14px;color:#378ADD">{pd_}% datetime</span>'
                f"</div>"
                '<div style="margin-top:16px;padding-top:16px;border-top:1px solid #2d3748">'
                f'<div style="display:flex;justify-content:space-between;font-size:15px;margin-bottom:8px">'
                f'<span style="color:#9ca3af">Total rows</span>'
                f'<span style="color:#fff;font-weight:500">{w["n_rows"]:,}</span></div>'
                f'<div style="display:flex;justify-content:space-between;font-size:15px">'
                f'<span style="color:#9ca3af">Total columns</span>'
                f'<span style="color:#fff;font-weight:500">{total}</span></div>'
                "</div>",
                "STRUCTURE",
                "#7F77DD",
            ),
            unsafe_allow_html=True,
        )

        # ── WIDGET 4: Most Skewed Columns ────────────────────────────────────
        skew_dict = w.get("skew_scores", {})

        def skew_bar(col, val):
            width = min(int(abs(val) / 3 * 100), 100)
            bc = (
                "#E24B4A"
                if abs(val) > 1
                else "#EF9F27"
                if abs(val) > 0.5
                else "#1D9E75"
            )
            direction = "right" if val > 0 else "left" if val < 0 else "ok"
            return (
                f'<div style="margin-bottom:12px">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:4px">'
                f'<span style="font-size:14px;color:#9ca3af">{col}</span>'
                f'<div style="display:flex;gap:10px">'
                f'<span style="font-size:14px;color:{bc}">{val}</span>'
                f'<span style="font-size:13px;color:#6b7280;min-width:32px">{direction}</span>'
                f"</div></div>"
                f'<div style="height:8px;border-radius:4px;background:#2d3748">'
                f'<div style="width:{width}%;height:100%;border-radius:4px;background:{bc}"></div></div>'
                f"</div>"
            )

        skew_rows = (
            "".join([skew_bar(c, v) for c, v in skew_dict.items()])
            if skew_dict
            else '<p style="font-size:15px;color:#6b7280">No numeric columns found</p>'
        )

        st.markdown(
            dark_card(
                '<p style="font-size:18px;font-weight:600;color:#fff;margin:0 0 6px">Most Skewed Columns</p>'
                '<p style="font-size:14px;color:#6b7280;margin:0 0 16px">needs attention before modelling</p>'
                + skew_rows
                + '<p style="font-size:13px;color:#6b7280;margin:12px 0 0;font-style:italic">Skewness &gt;1 = consider log transform</p>',
                "DISTRIBUTION",
                "#EF9F27",
            ),
            unsafe_allow_html=True,
        )

        # ── WIDGET 6: Strongest Correlations ─────────────────────────────────
        pairs = w.get("top_pairs", [])

        def corr_bar(col1, col2, r):
            width = int(abs(r) * 100)
            bc = "#E24B4A" if abs(r) > 0.7 else "#EF9F27" if abs(r) > 0.4 else "#1D9E75"
            return (
                f'<div style="margin-bottom:12px">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:4px">'
                f'<span style="font-size:14px;color:#9ca3af">{col1} ↔ {col2}</span>'
                f'<span style="font-size:14px;color:{bc}">r = {r}</span>'
                f"</div>"
                f'<div style="height:8px;border-radius:4px;background:#2d3748">'
                f'<div style="width:{width}%;height:100%;border-radius:4px;background:{bc}"></div></div>'
                f"</div>"
            )

        corr_rows = (
            "".join([corr_bar(c1, c2, r) for c1, c2, r in pairs])
            if pairs
            else '<p style="font-size:15px;color:#6b7280">Need 2+ numeric columns</p>'
        )

        st.markdown(
            dark_card(
                '<p style="font-size:18px;font-weight:600;color:#fff;margin:0 0 6px">Strongest Correlations</p>'
                '<p style="font-size:14px;color:#6b7280;margin:0 0 16px">top pairs by absolute r value</p>'
                + corr_rows
                + '<p style="font-size:13px;color:#6b7280;margin:12px 0 0;font-style:italic">|r| &gt; 0.7 = strong &nbsp;|&nbsp; 0.4–0.7 = moderate</p>',
                "CORRELATION",
                "#378ADD",
            ),
            unsafe_allow_html=True,
        )


# ── Tab 2 · Data Cleaning ─────────────────────────────────────────────────────


def _render_cleaning_tab(df: pd.DataFrame):
    st.markdown(
        '<div class="section-title">🧹 Data Cleaning Pipeline</div>',
        unsafe_allow_html=True,
    )

    if "df_cleaned" in st.session_state:
        df_c = st.session_state["df_cleaned"]
        st.success(
            f"✅ Cleaned dataset ready — **{df_c.shape[0]:,} rows × {df_c.shape[1]} columns**"
        )

        # Download buttons
        import datetime

        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"argus_cleaned_{st.session_state.get('dataset_name', 'data')}_{now}.csv"
        )
        csv = df_c.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Download Cleaned CSV",
            data=csv,
            file_name=filename,
            mime="text/csv",
            help="Download your cleaned dataset as CSV",
        )

        with st.expander("🗃 Cleaned Data Preview", expanded=False):
            st.dataframe(
                df_c.head(10)
                .style.background_gradient(cmap="YlOrRd")
                .format(precision=2),
                use_container_width=True,
            )
        return

    if not _HAS_CLEANING:
        st.warning("⚠️ Data cleaning module not available.")
        return

    if "cleaning_scan_report" not in st.session_state:
        st.markdown(
            """
        <div class="info-card">
            <b>Pipeline Steps:</b><br>
            ✅ Scan for missing values → show summary<br>
            ✅ Replace junk string values with NaN<br>
            ✅ Sanitize special characters<br>
            ✅ Call Groq AI for ambiguous columns<br>
            ✅ Apply imputation strategy per column<br>
            ✅ Re-scan to confirm 0 nulls
        </div>
        """,
            unsafe_allow_html=True,
        )

        if st.button("🚀 Analyze & Scan Dataset for Cleaning", key="run_scan"):
            with st.spinner("Scanning for missing values and junk characters..."):
                from data_cleaning import SmartDataCleaner

                cleaner = SmartDataCleaner(df)
                report = cleaner.scan_missing_values()
                st.session_state["cleaning_scan_report"] = report
            st.rerun()
    else:
        st.info("🔍 **Pre-Clean Summary Report:**")
        st.dataframe(
            st.session_state["cleaning_scan_report"],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✨ Proceed to Smart Imputation", key="run_impute"):
            _run_smart_imputation()


def _run_smart_imputation():
    df = st.session_state["df"]
    from data_cleaning import SmartDataCleaner
    import time

    status_container = st.empty()
    progress_bar = st.progress(0)

    cleaner = SmartDataCleaner(df.copy())

    # Pipeline Step 1
    status_container.info(
        "✅ Replace junk string values with NaN & Sanitize special characters..."
    )
    time.sleep(0.6)
    cleaner.sanitize_data()
    progress_bar.progress(15)

    # Pipeline Step 2
    status_container.info(
        "✅ Call Groq AI for ambiguous columns & Apply imputation strategy per column..."
    )

    def prog_cb(current, total, col_name):
        pct = 15 + int(70 * (current / total))
        progress_bar.progress(pct)
        status_container.info(
            f"✅ Processing column '{col_name}'... ({current}/{total})"
        )

    df_clean, warnings_list = cleaner.smart_impute(progress_callback=prog_cb)

    # Re-scan to confirm 0 nulls
    status_container.info("✅ Re-scan to confirm 0 nulls...")
    final_scan = cleaner.scan_missing_values()
    rem = final_scan["Missing Count"].sum()
    if rem == 0:
        status_container.success("🎉 All clear! 0 missing values remain.")
    else:
        status_container.warning(
            f"⚠️ Imputation finished, but {rem} missing values remain (likely ignored due to >80% rule)."
        )

    for w in warnings_list:
        st.warning(w)

    progress_bar.progress(100)
    time.sleep(1.2)

    st.session_state["df_cleaned"] = df_clean
    st.session_state["just_uploaded"] = (
        True  # To trigger Dataset Analyzed increment on dashboard reload
    )
    st.rerun()


# ── Tab 3 · Univariate Analysis ───────────────────────────────────────────────


def _render_univariate_tab(mode: str = "numeric"):
    title_icon = "📈" if mode == "numeric" else "📊"
    st.markdown(
        f'<div class="section-title">{title_icon} {mode.capitalize()} Univariate Analysis</div>',
        unsafe_allow_html=True,
    )

    df_work = st.session_state.get("df_cleaned", st.session_state.get("df"))
    dataset_name = st.session_state.get("dataset_name", "dataset")
    target_var = st.session_state.get("target_variable")

    pdf_key = f"UNI_PDF_{mode.upper()}"

    if st.session_state.get(pdf_key):
        st.success(f"✅ {mode.capitalize()} univariate analysis ready below.")
        _show_pdf_scrollable(st.session_state[pdf_key], f"📄 {mode.capitalize()} Univariate Report")
        if st.button(f"🔄 Re-run {mode.capitalize()} Analysis", key=f"rerun_uni_{mode}"):
            del st.session_state[pdf_key]
            st.rerun()
        return

    if not _HAS_UNIVARIATE:
        st.warning("⚠️ Univariate analysis module not available.")
        return

    st.markdown(
        """
    <div class="info-card">
        AI-powered per-column analysis — histograms, count plots and natural-language
        descriptions generated by LLaMA 3.3. Output is rendered as a scrollable PDF below.
    </div>
    """,
        unsafe_allow_html=True,
    )



    if st.button(f"{title_icon} Run {mode.capitalize()} Univariate Analysis", key=f"run_uni_{mode}"):
        with st.spinner(f"Running {mode} univariate analysis…"):
            try:
                ctx_items, pdf_bytes = uni_analyze_and_visualize(
                    df_work, dataset_name, target_var or "", mode=mode
                )
                if not pdf_bytes:
                    st.warning(f"No {mode} columns found in the dataset.")
                else:
                    # ── Store in LLM context
                    ctx = st.session_state.setdefault("llm_context", [])
                    if isinstance(ctx_items, list):
                        ctx.extend(ctx_items)
                    st.session_state[pdf_key] = pdf_bytes
                    st.rerun()
            except Exception as e:
                st.error(f"Univariate analysis error: {e}")


# ── Instant Data Summary (post-upload) ─────────────────────────────────────


def _get_groq_ai_summary(df: pd.DataFrame) -> str:
    """Call Groq synchronously (no streaming) with a lean prompt and a thread timeout."""
    import threading

    result_holder = [None]

    def _call():
        try:
            from utils import get_groq_client

            client = get_groq_client()

            dataset_name = st.session_state.get("dataset_name", "dataset")
            num_cols = df.select_dtypes(include="number").columns.tolist()
            cat_cols = df.select_dtypes(exclude="number").columns.tolist()
            missing_total = int(df.isnull().sum().sum())
            mem_kb = round(df.memory_usage(deep=True).sum() / 1024, 1)

            # Keep prompt short to avoid rate-limit / timeout
            col_summary = ", ".join(f"{c}({df[c].dtype})" for c in df.columns[:20]) + (
                f" ...+{len(df.columns) - 20} more" if len(df.columns) > 20 else ""
            )

            num_stats_parts = []
            if num_cols:
                desc = df[num_cols[:8]].describe().T[["min", "max", "mean"]].round(2)
                for col_name, row in desc.iterrows():
                    num_stats_parts.append(
                        f"{col_name}: min={row['min']}, max={row['max']}, mean={row['mean']:.2f}"
                    )
            num_stats = "; ".join(num_stats_parts)

            prompt = (
                f"Dataset: '{dataset_name}', {df.shape[0]} rows × {df.shape[1]} cols, {mem_kb} KB.\n"
                f"Columns: {col_summary}\n"
                f"Missing cells: {missing_total}\n"
                + (f"Numeric stats: {num_stats}\n" if num_stats else "")
                + f"Sample (5 rows):\n{df.head(5).to_string(index=False, max_cols=10)}\n\n"
                "As a data analyst, write a short friendly summary (≤150 words, 3-4 paragraphs) covering: "
                "(1) what this dataset is about, (2) what key columns mean, "
                "(3) data quality notes (nulls/issues), (4) best analysis use case. "
                "Plain English for a beginner. Respond in plain paragraph text only. Do NOT use any markdown formatting (like **), no HTML tags, no bullet points, and no hashtags."
            )

            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=280,
                stream=False,
            )
            result_holder[0] = resp.choices[0].message.content.strip()
        except Exception as e:
            result_holder[0] = f"__ERROR__: {e}"

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(timeout=30)  # hard 30-second cap

    if result_holder[0] is None:
        return "__ERROR__: Request timed out after 30 seconds."
    return result_holder[0]


def _render_data_summary(df: pd.DataFrame):
    """Full redesigned upload page summary section."""
    total_cells = df.shape[0] * df.shape[1]
    missing_pct = round(int(df.isnull().sum().sum()) / max(total_cells, 1) * 100, 1)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    num_pct = round(len(num_cols) / max(df.shape[1], 1) * 100, 1)
    cat_pct = round(len(cat_cols) / max(df.shape[1], 1) * 100, 1)
    mem_kb = round(df.memory_usage(deep=True).sum() / 1024, 1)
    missing_cols_count = int((df.isnull().sum() > 0).sum())

    # Build LLM context
    summary_ctx = (
        f"Dataset '{st.session_state.get('dataset_name', 'dataset')}' has "
        f"{df.shape[0]} rows and {df.shape[1]} columns. "
        f"{len(num_cols)} numeric columns: {', '.join(num_cols[:8])}{'...' if len(num_cols) > 8 else ''}. "
        f"{len(cat_cols)} categorical columns: {', '.join(cat_cols[:8])}{'...' if len(cat_cols) > 8 else ''}. "
        f"Missing values: {df.isnull().sum().sum()} ({missing_pct}%). Memory: {mem_kb} KB."
    )
    ctx = st.session_state.setdefault("llm_context", [])
    if not any("Dataset '" in c and "rows and" in c for c in ctx):
        ctx.append(summary_ctx)

    # ── Header card ──────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="summary-header-card">'
        f'<span class="sum-icon">📊</span> '
        f"<b>{st.session_state.get('file_name', 'Dataset')}</b>"
        f" &nbsp;&mdash;&nbsp; "
        f"<b>{df.shape[0]:,}</b> rows &times; <b>{df.shape[1]}</b> columns"
        f' &nbsp;&bull;&nbsp; <span class="sum-mem">{mem_kb} KB</span>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Progress bars ─────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="sum-bars">
          <div class="sum-bar-row">
            <span class="sum-bar-label">⚠️ Missing</span>
            <div class="sum-bar-track"><div class="sum-bar-fill" style="width:{missing_pct}%;background:#EF4444"></div></div>
            <span class="sum-bar-val">{missing_pct}%</span>
          </div>
          <div class="sum-bar-row">
            <span class="sum-bar-label">🔢 Numeric</span>
            <div class="sum-bar-track"><div class="sum-bar-fill" style="width:{num_pct}%;background:#38BDF8"></div></div>
            <span class="sum-bar-val">{num_pct}% ({len(num_cols)} cols)</span>
          </div>
          <div class="sum-bar-row">
            <span class="sum-bar-label">🏷 Categorical</span>
            <div class="sum-bar-track"><div class="sum-bar-fill" style="width:{cat_pct}%;background:#A78BFA"></div></div>
            <span class="sum-bar-val">{cat_pct}% ({len(cat_cols)} cols)</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Missing values mini chart ─────────────────────────────────────────────
    missing_s = df.isnull().sum()
    missing_s = missing_s[missing_s > 0].sort_values(ascending=False).head(8)
    if not missing_s.empty:
        fig = px.bar(
            x=missing_s.values,
            y=missing_s.index,
            orientation="h",
            title="Top Missing-Value Columns",
            labels={"x": "Missing Count", "y": "Column"},
            color=missing_s.values,
            color_continuous_scale=["#F97316", "#EF4444"],
        )
        fig.update_coloraxes(showscale=False)
        fig.update_layout(
            **_chart_layout(), 
            height=max(200, len(missing_s) * 36 + 80),
            dragmode=False
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ══════════════════════════════════════════════════════════════════════════
    # TASK 1 — DATA PREVIEW TABLE (first 15 rows, styled)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="section-title">📋 Data Preview — First 15 Rows</div>',
        unsafe_allow_html=True,
    )

    preview_df = df.head(15)
    # Build header row with dtype badges
    dtype_badges = "".join(
        f'<th style="background:#1a2340;color:#F97316;font-weight:700;'
        f'position:sticky;top:0;z-index:2;padding:8px 12px;white-space:nowrap;border-bottom:2px solid #F97316;">'
        f'{col}<br><span style="font-size:0.68rem;font-weight:400;color:#94a3b8;'
        f'background:rgba(249,115,22,0.12);padding:1px 5px;border-radius:4px;">'
        f"{str(df[col].dtype)}</span></th>"
        for col in preview_df.columns
    )

    # Build data rows with alternating colors and amber highlights for NaN
    data_rows_html = ""
    for i, (_, row) in enumerate(preview_df.iterrows()):
        bg = "rgba(15,23,42,0.9)" if i % 2 == 0 else "rgba(26,35,64,0.6)"
        cells = ""
        for col in preview_df.columns:
            val = row[col]
            is_null = pd.isnull(val)
            cell_style = (
                "background:#78350f;color:#FCD34D;font-weight:600;"
                if is_null
                else f"color:#e2e8f0;"
            )
            display_val = "⚠ null" if is_null else str(val)
            cells += (
                f'<td style="{cell_style}padding:6px 12px;white-space:nowrap;'
                f'font-size:0.82rem;border-bottom:1px solid rgba(255,255,255,0.04);">'
                f"{display_val}</td>"
            )
        data_rows_html += f'<tr style="background:{bg};">{cells}</tr>'

    preview_table_html = f"""
    <div style="overflow-x:auto;max-height:400px;overflow-y:auto;
        border:1px solid rgba(249,115,22,0.3);border-radius:12px;
        background:#0d1526;margin-bottom:24px;width:100%;text-align:left;">
      <table style="width:100%;min-width:max-content;margin:0;border-collapse:collapse;font-family:monospace;text-align:left;">
        <thead style="text-align:left;"><tr>{dtype_badges}</tr></thead>
        <tbody>{data_rows_html}</tbody>
      </table>
    </div>
    """
    st.markdown(preview_table_html, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # AI DATA INTELLIGENCE SUMMARY — 100% Python-native rendering
    # ══════════════════════════════════════════════════════════════════════════

    # Style block (CSS only, zero JS)
    st.markdown(
        """
    <style>
    @keyframes borderSpin {
      0%   { border-color:#3b82f6; box-shadow:0 0 18px #3b82f680; }
      33%  { border-color:#a855f7; box-shadow:0 0 18px #a855f780; }
      66%  { border-color:#06b6d4; box-shadow:0 0 18px #06b6d480; }
      100% { border-color:#3b82f6; box-shadow:0 0 18px #3b82f680; }
    }
    .ai-box-wrap {
      border:2px solid #3b82f6; border-radius:16px; padding:22px 24px 18px;
      background:rgba(10,15,30,0.95); margin:8px 0 20px 0;
      animation:borderSpin 2.5s linear infinite;
    }
    .ai-box-wrap.done {
      animation:none; border-color:#38BDF8;
      box-shadow:0 0 20px rgba(56,189,248,0.12);
    }
    .ai-box-hdr {
      display:flex; justify-content:space-between; align-items:center;
      margin-bottom:14px;
    }
    .ai-box-badge {
      font-size:0.68rem; background:rgba(56,189,248,0.12); color:#38BDF8;
      border:1px solid rgba(56,189,248,0.28); border-radius:20px; padding:2px 10px;
    }
    .ai-typed-text {
      font-family:'Inter','Segoe UI',sans-serif; color:#e2e8f0;
      font-size:0.93rem; line-height:1.78; white-space:pre-wrap;
    }
    .ai-stat-pills { display:flex; flex-wrap:wrap; gap:9px; margin-top:16px; }
    .ai-pill {
      background:rgba(249,115,22,0.1); border:1px solid rgba(249,115,22,0.3);
      border-radius:20px; padding:5px 14px; font-size:0.79rem; color:#e2e8f0;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # ── Generate the text (cached in session_state) ───────────────────────────
    file_key = st.session_state.get("file_name", "")
    if (
        "ai_summary_text" not in st.session_state
        or st.session_state.get("ai_summary_file") != file_key
        or st.session_state.get("ai_summary_regen")
    ):
        st.session_state["ai_summary_regen"] = False
        st.session_state["ai_summary_file"] = file_key

        # Header first while we wait
        st.markdown(
            '<div class="ai-box-wrap">'
            '<div class="ai-box-hdr">'
            '<span style="font-size:1.05rem;font-weight:700;color:#f1f5f9;">🤖 AI Data Intelligence Summary</span>'
            '<span class="ai-box-badge">✦ Powered by Groq AI</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        loading_slot = st.empty()
        loading_slot.info("⚡ Argus is analysing your dataset with Groq AI…")

        raw = _get_groq_ai_summary(df)

        loading_slot.empty()
        st.markdown("</div>", unsafe_allow_html=True)  # close temp box

        if raw.startswith("__ERROR__"):
            st.session_state["ai_summary_text"] = "__ERROR__"
            st.session_state["ai_summary_error"] = raw.replace("__ERROR__: ", "")
        else:
            # Strip any stray markdown symbols that LLM occasionally adds
            import re as _re

            clean = _re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", raw)  # **bold** → bold
            clean = _re.sub(r"#{1,6}\s+", "", clean)  # ## headings
            clean = _re.sub(r"^\s*[-*•]\s+", "", clean, flags=_re.M)  # bullet symbols
            clean = clean.strip()
            st.session_state["ai_summary_text"] = clean
            st.session_state["ai_summary_typed"] = False  # trigger typewriter

            ctx = st.session_state.setdefault("llm_context", [])
            summary_str = f"Dataset Overview AI Summary: {clean}"
            if summary_str not in ctx:
                ctx.append(summary_str)

        st.rerun()

    # ── Render the stored text ────────────────────────────────────────────────
    ai_text = st.session_state.get("ai_summary_text", "")
    is_error = ai_text == "__ERROR__"
    already_typed = st.session_state.get("ai_summary_typed", False)

    # ── Helper: build the complete box HTML ───────────────────────────────────
    PILLS_HTML = (
        f'<div class="ai-stat-pills">'
        f'<span class="ai-pill">📏 Rows: <b>{df.shape[0]:,}</b></span>'
        f'<span class="ai-pill">📊 Columns: <b>{df.shape[1]}</b></span>'
        f'<span class="ai-pill">⚠️ Missing: <b>{missing_cols_count}</b> cols</span>'
        f'<span class="ai-pill">🔢 Numeric: <b>{len(num_cols)}</b></span>'
        f'<span class="ai-pill">🔤 Categorical: <b>{len(cat_cols)}</b></span>'
        f"</div>"
    )

    HDR_HTML = (
        '<div class="ai-box-hdr">'
        '<span style="font-size:1.05rem;font-weight:700;color:#f1f5f9;">🤖 AI Data Intelligence Summary</span>'
        '<span class="ai-box-badge">✦ Powered by Groq AI</span>'
        "</div>"
    )

    CURSOR = (
        '<span style="display:inline-block;width:2px;height:0.95em;'
        "background:#00d4ff;vertical-align:middle;margin-left:2px;"
        'animation:blink-cur 0.6s step-start infinite;"></span>'
    )

    def _box(body_html: str, cls: str = "ai-box-wrap") -> str:
        return (
            "<style>"
            "@keyframes blink-cur{0%,100%{opacity:1}50%{opacity:0}}"
            "</style>"
            f'<div class="{cls}">'
            f"{HDR_HTML}"
            f"{body_html}"
            "</div>"
        )

    box_slot = st.empty()

    if is_error:
        box_slot.markdown(
            _box(
                '<p style="color:#FCD34D;background:rgba(120,53,15,0.4);'
                'border-radius:8px;padding:12px;">⚠️ Summary unavailable. '
                "Please check your API connection.</p>",
                "ai-box-wrap done",
            ),
            unsafe_allow_html=True,
        )
    elif not already_typed:
        # ── Phase 1: word-by-word typewriter with spinning border ─────────────
        import time as _time

        displayed = ""
        words = ai_text.split(" ")
        for w in words:
            displayed += w + " "
            box_slot.markdown(
                _box(
                    f'<div class="ai-typed-text">{displayed}{CURSOR}</div>',
                    "ai-box-wrap",  # spinning glow while typing
                ),
                unsafe_allow_html=True,
            )
            _time.sleep(0.07)  # 70 ms / word  →  medium speed

        # ── Phase 2: done — static glow + stat pills ──────────────────────────
        box_slot.markdown(
            _box(
                f'<div class="ai-typed-text">{ai_text}</div>' + PILLS_HTML,
                "ai-box-wrap done",  # static cyan glow when finished
            ),
            unsafe_allow_html=True,
        )
        st.session_state["ai_summary_typed"] = True

    else:
        # Already typed → instant static render
        box_slot.markdown(
            _box(
                f'<div class="ai-typed-text">{ai_text}</div>' + PILLS_HTML,
                "ai-box-wrap done",
            ),
            unsafe_allow_html=True,
        )

    # ── Single Regenerate button ──────────────────────────────────────────────
    _, regen_col = st.columns([4, 1])
    with regen_col:
        if st.button("🔄 Regenerate", key="regen_ai_summary"):
            for k in ("ai_summary_text", "ai_summary_typed", "ai_summary_file"):
                st.session_state.pop(k, None)
            st.session_state["ai_summary_regen"] = True
            st.rerun()

    # ── Smart Dataset Dashboard (inserted right after AI summary) ────────────
    _render_smart_dashboard(df)


# ── Smart Dataset Dashboard ───────────────────────────────────────────────────


def _build_col_profile(df: pd.DataFrame) -> dict:
    """
    Pre-compute a comprehensive stats profile for every column.
    This is passed to Groq so it can reason about what matters in THIS dataset
    without us imposing any role or keyword restrictions.
    """
    profile = {}
    n = len(df)

    # Detect date columns structurally (not by name)
    date_cols = {}
    for col in df.columns:
        try:
            parsed = pd.to_datetime(
                df[col], errors="coerce", infer_datetime_format=True
            )
            valid = parsed.dropna()
            if len(valid) / max(n, 1) > 0.6:
                date_cols[col] = {
                    "min": valid.min().strftime("%Y-%m-%d"),
                    "max": valid.max().strftime("%Y-%m-%d"),
                    "span_days": int((valid.max() - valid.min()).days),
                }
        except Exception:
            pass

    for col in df.columns:
        dtype_str = str(df[col].dtype)
        missing_n = int(df[col].isnull().sum())
        missing_pct = round(missing_n / max(n, 1) * 100, 1)
        nunique = int(df[col].nunique(dropna=True))
        samples = [str(x) for x in df[col].dropna().head(3).tolist()]

        entry = {
            "dtype": dtype_str,
            "missing_pct": missing_pct,
            "nunique": nunique,
            "samples": samples,
        }

        # Date column
        if col in date_cols:
            entry["is_date"] = True
            entry.update(date_cols[col])

        # Numeric stats
        elif df[col].dtype.kind in ("i", "f", "u"):
            # Try numeric even if stored as object (common in messy CSVs)
            try:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(s) > 0:
                    entry["sum"] = _fmt_num(s.sum())
                    entry["mean"] = _fmt_num(s.mean())
                    entry["max"] = _fmt_num(s.max())
                    entry["min"] = _fmt_num(s.min())
            except Exception:
                pass

        # Categorical/object stats
        else:
            # Try numeric coercion for object columns (often stored as string)
            try:
                s_num = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(s_num) / max(n, 1) > 0.6:
                    entry["sum"] = _fmt_num(s_num.sum())
                    entry["max"] = _fmt_num(s_num.max())
                    entry["min"] = _fmt_num(s_num.min())
                    entry["is_numeric_stored_as_object"] = True
            except Exception:
                pass

            vc = df[col].value_counts()
            if len(vc) > 0:
                entry["top_value"] = str(vc.index[0])
                entry["top_count"] = int(vc.iloc[0])
                entry["top_pct"] = round(vc.iloc[0] / n * 100, 1)

        profile[col] = entry

    return profile


def _fmt_num(v) -> str:
    """Format a number compactly for Groq context (keeps payload small)."""
    try:
        f = float(v)
        if abs(f) >= 1_000_000_000:
            return f"{f / 1e9:.2f}B"
        if abs(f) >= 1_000_000:
            return f"{f / 1e6:.2f}M"
        if abs(f) >= 1_000:
            return f"{f / 1e3:.2f}K"
        return f"{f:,.2f}"
    except Exception:
        return str(v)


@st.cache_data(show_spinner=False)
def _ai_glance_cards(
    cache_key: str, profile_json: str, n_rows: int, n_cols: int
) -> list:
    """
    Send the full column profile to Groq and ask it to produce 5-6
    ready-to-display insight cards for THIS dataset — no predefined roles,
    no fixed slot names.  Works for retail, COVID, HR, sports, finance,
    healthcare, education, etc.

    Returns: list of {"label": str, "value": str, "sub": str}
    """
    import json as _json

    prompt = f"""You are a senior data analyst. You have been given a pre-computed
statistics profile for a dataset with {n_rows:,} rows and {n_cols} columns.

Column profile (name → stats):
{profile_json}

Your task: Generate exactly 5-6 insightful metric cards that give someone an
immediate "at a glance" overview of what this dataset is about.

Rules:
1. USE the pre-computed numbers from the profile directly — do NOT invent values.
2. Each card must reflect a genuinely interesting fact: a total, a count of unique
   entities, a dominant value, a date range, a max value and which entity holds it, etc.
3. The label should describe WHAT the metric is (e.g. "Total Deaths", "Countries",
   "Tests Conducted", "Date Range", "Most Affected Country").
4. The value should be the actual number/text from the profile (e.g. "2.3M", "38",
   "United Kingdom", "Jan 2020 – Dec 2021").
5. The sub should be a single short context phrase (e.g. "across all continents",
   "25% missing IDs", "91% of total revenue", "highest recorded").
6. Always include Total Rows as the very first card.
7. Cover a mix of: scale (totals), diversity (unique counts), extremes (max/argmax),
   data quality (if notable missing % exists), and time range (if dates exist).
8. Output ONLY valid JSON — a list of objects, nothing else.

Example format (do not copy content, only format):
[
  {{"label": "Total Rows", "value": "541,909", "sub": "Jan 2010 – Dec 2011"}},
  {{"label": "Countries", "value": "38", "sub": "91% records from United Kingdom"}},
  {{"label": "Total Deaths", "value": "2.3M", "sub": "across all continents"}},
  {{"label": "Most Affected", "value": "USA", "sub": "highest total cases"}},
  {{"label": "Tests Conducted", "value": "890M", "sub": "worldwide total"}}
]

Return ONLY the JSON array. No markdown, no explanation."""

    try:
        from utils import get_groq_client

        client = get_groq_client()
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.15,
            max_tokens=600,
            stream=False,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences if present
        raw = raw.strip("` \n")
        if raw.startswith("json"):
            raw = raw[4:].strip()
        cards = _json.loads(raw)
        # Validate structure
        if isinstance(cards, list) and all(
            isinstance(c, dict) and "label" in c and "value" in c and "sub" in c
            for c in cards
        ):
            return cards[:6]  # cap at 6
    except Exception:
        pass
    return []


def _fallback_cards(df: pd.DataFrame) -> list:
    """
    Pure-Python fallback when Groq is unavailable.
    Builds basic cards without role mapping — just structural facts.
    """
    n = len(df)
    n_cols = len(df.columns)
    miss = round(df.isnull().mean().mean() * 100, 1)
    cards = [
        {"label": "Total Rows", "value": f"{n:,}", "sub": f"{n_cols} columns"},
        {
            "label": "Numeric Columns",
            "value": str(len(df.select_dtypes(include="number").columns)),
            "sub": "quantitative features",
        },
        {
            "label": "Categorical Columns",
            "value": str(len(df.select_dtypes(include=["object", "category"]).columns)),
            "sub": "qualitative features",
        },
        {"label": "Missing Data", "value": f"{miss}%", "sub": "overall missing rate"},
    ]
    # Add top column by nunique
    top_cat = (
        df.select_dtypes(include=["object", "category"]).nunique().idxmax()
        if len(df.select_dtypes(include=["object", "category"]).columns)
        else None
    )
    if top_cat:
        cards.append(
            {
                "label": f"Unique {top_cat}",
                "value": str(df[top_cat].nunique()),
                "sub": "distinct values",
            }
        )
    # Add largest numeric sum
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        sums = df[num_cols].sum()
        largest_col = sums.idxmax()
        v = sums[largest_col]
        cards.append(
            {
                "label": largest_col.replace("_", " ").title(),
                "value": _fmt_num(v),
                "sub": "total across all rows",
            }
        )
    return cards[:6]


def _render_smart_dashboard(df: pd.DataFrame):
    """
    CHANGE 1 — Enhanced 'Dataset At a Glance' with AI-generated metric cards
    rendered in a 3-column responsive HTML grid, with hover tooltips.
    """
    filename = st.session_state.get("file_name", "dataset")
    n_rows = len(df)
    n_cols = len(df.columns)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    glance_cache_key = f"{filename}_glance_v2_{','.join(df.columns)}_{n_rows}"

    def _get_cards():
        """Try Groq for smart cards, fall back to hardcoded logic."""
        if glance_cache_key in st.session_state.get("ai_cache", {}):
            cached = st.session_state["ai_cache"][glance_cache_key]
            try:
                return json.loads(cached)
            except Exception:
                pass

        # Build a compact numeric summary
        try:
            numeric_summary = df.describe().round(2).to_dict()
        except Exception:
            numeric_summary = {}
        try:
            cat_summary = {
                col: df[col].value_counts().head(3).to_dict() for col in cat_cols[:8]
            }
        except Exception:
            cat_summary = {}

        col_dtypes = {str(c): str(t) for c, t in df.dtypes.items()}

        groq_prompt = f"""You are a data analyst. Given this dataset profile, choose the 6 most
interesting metrics to highlight on a summary dashboard.

Dataset profile:
  filename: {filename}
  shape: {n_rows} rows x {n_cols} columns
  columns and dtypes: {col_dtypes}
  numeric summary: {json.dumps(numeric_summary)}
  categorical top values: {json.dumps(cat_summary)}

For each of the 6 metrics, return a JSON array like this:
[
  {{
    "label": "short label (3-5 words max)",
    "value": "the actual value as a string",
    "sublabel": "1 short context phrase (e.g. 'across all records', 'highest recorded', '22.1% of records')",
    "reason": "why this metric is interesting (1 sentence, used as tooltip)"
  }}
]

Rules:
- Always include Total Rows as the first card
- Pick a mix: at least 1 from numeric (max, mean, or unique count),
  at least 1 from categorical (most common value + its frequency),
  and 1 data quality metric (missing values or duplicate count)
- For categorical most-common: value should be the top value, sublabel should be "X% of records"
- For numeric max/min: sublabel = "highest recorded" or "lowest recorded"
- Be specific to THIS dataset's domain (medical, retail, financial etc)
- Return ONLY valid JSON. No explanation, no markdown, no extra text."""

        try:
            raw = groq_call(groq_prompt, max_tokens=700)
            raw = raw.strip("`\n")
            if raw.startswith("json"):
                raw = raw[4:].strip()
            parsed = json.loads(raw)
            if isinstance(parsed, list) and len(parsed) > 0:
                if "ai_cache" not in st.session_state:
                    st.session_state["ai_cache"] = {}
                st.session_state["ai_cache"][glance_cache_key] = json.dumps(parsed)
                return parsed
        except Exception:
            pass

        # Hardcoded fallback
        miss_total = int(df.isnull().sum().sum())
        dupe_count = int(df.duplicated().sum())
        num_count = len(df.select_dtypes(include="number").columns)
        cat_count = len(df.select_dtypes(include=["object", "category"]).columns)
        fallback = [
            {
                "label": "Total Rows",
                "value": f"{n_rows:,}",
                "sublabel": "entire dataset",
                "reason": "Total number of records in the dataset.",
            },
            {
                "label": "Total Columns",
                "value": str(n_cols),
                "sublabel": "features",
                "reason": "Total number of columns / features.",
            },
            {
                "label": "Missing Values",
                "value": f"{miss_total:,}",
                "sublabel": "total null cells",
                "reason": "Total count of null cells across the dataset.",
            },
            {
                "label": "Duplicate Rows",
                "value": str(dupe_count),
                "sublabel": "exact duplicates",
                "reason": "Exact duplicate row count that should be removed.",
            },
            {
                "label": "Numeric Columns",
                "value": str(num_count),
                "sublabel": "quantitative features",
                "reason": "Count of columns with numeric values.",
            },
            {
                "label": "Categorical Columns",
                "value": str(cat_count),
                "sublabel": "text features",
                "reason": "Count of columns with text/categorical values.",
            },
        ]
        return fallback

    # ── Generate cards ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Dataset At a Glance</div>', unsafe_allow_html=True)
    with st.spinner("Generating dataset overview…"):
        cards = _get_cards()

    # ── Horizontal scrollable card strip ──────────────────────────────────────
    card_html_parts = []
    for card in cards:
        label    = card.get("label", "")
        value    = card.get("value", "")
        sublabel = card.get("sublabel", card.get("sub", ""))
        reason   = card.get("reason", "")
        card_html_parts.append(
            f'<div class="glance-card" title="{reason}">'
            f'<p class="glance-label">{label}</p>'
            f'<p class="glance-value">{value}</p>'
            f'<p class="glance-sub">{sublabel}</p>'
            f'</div>'
        )

    full_html = (
        '<div class="glance-scroll-row">'
        + "".join(card_html_parts)
        + '</div>'
    )
    st.markdown(full_html, unsafe_allow_html=True)


# ── Numeric Distribution ──────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def _groq_numeric_description(
    cache_key: str,
    column: str,
    dataset_context: str,
    mean: float,
    median: float,
    std: float,
    _min: float,
    _max: float,
    q1: float,
    q3: float,
    skewness: float,
    kurtosis: float,
    missing_pct: float,
    negative_count: int,
    outlier_pct: float,
) -> str:
    """Call Groq for numeric column AI description (cached per unique column stats)."""
    prompt = f"""You are a data analyst writing a concise EDA insight.

Column: '{column}'
Dataset context: {dataset_context}

Statistics:
  Mean: {mean}  |  Median: {median}  |  Std Dev: {std}
  Min: {_min}  |  Max: {_max}
  Q1: {q1}  |  Q3: {q3}
  Skewness: {skewness}  |  Kurtosis: {kurtosis}
  Missing: {missing_pct}%  |  Negatives: {negative_count}
  Outliers (IQR rule): {outlier_pct}%

Write exactly 3-4 sentences covering:
1. Distribution shape using skewness value:
     skewness > 1.0  -> "strongly right-skewed"
     skewness > 0.5  -> "moderately right-skewed"
     skewness < -1.0 -> "strongly left-skewed"
     skewness < -0.5 -> "moderately left-skewed"
     else            -> "roughly symmetric"
   Cite mean and median values to explain WHY it is skewed.
2. Where most values concentrate: mention Q1-Q3 range and what
   that means in real-world terms for this dataset.
3. Data quality: mention negatives, outliers, or missing values
   ONLY if significant (negatives > 0, outliers > 5%, missing > 5%).
   Say what it likely means and what to do about it.
4. One specific actionable recommendation for this column
   (log-transform, filter, impute, cap outliers, or "ready to use").

Use actual numbers from the stats. No bullet points. Under 100 words."""
    try:
        from utils import get_groq_client

        client = get_groq_client()
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=220,
            stream=False,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


def _render_numeric_distribution(df: pd.DataFrame):
    """CHANGE 2 — Full scipy stats + histogram + Groq AI description."""
    try:
        import scipy.stats as scipy_stats
    except ImportError:
        scipy_stats = None

    num_cols = df.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()
    if not num_cols:
        st.info("No numeric columns found.")
        return

    selected_col = st.selectbox(
        "Select a numeric column", num_cols, key="dist_col_select"
    )
    if not selected_col:
        return

    col_data = df[selected_col].dropna()
    filename = st.session_state.get("file_name", "dataset")

    # ── STEP 1: compute full stats ──────────────────────────────────────────
    mean_val = round(float(col_data.mean()), 3)
    median_val = round(float(col_data.median()), 3)
    std_val = round(float(col_data.std()), 3)
    min_val = round(float(col_data.min()), 3)
    max_val = round(float(col_data.max()), 3)
    q1_val = round(float(col_data.quantile(0.25)), 3)
    q3_val = round(float(col_data.quantile(0.75)), 3)
    iqr = q3_val - q1_val
    outlier_mask = (col_data < q1_val - 1.5 * iqr) | (col_data > q3_val + 1.5 * iqr)
    outlier_count = int(outlier_mask.sum())
    outlier_pct = round(outlier_count / max(len(col_data), 1) * 100, 1)

    if scipy_stats is not None:
        skewness_val = round(float(scipy_stats.skew(col_data)), 3)
        kurtosis_val = round(float(scipy_stats.kurtosis(col_data)), 3)
    else:
        skewness_val = round(float(col_data.skew()), 3)
        kurtosis_val = round(float(col_data.kurtosis()), 3)

    missing_count = int(df[selected_col].isnull().sum())
    missing_pct = round(df[selected_col].isnull().mean() * 100, 1)
    negative_count = int((col_data < 0).sum())
    unique_count = int(col_data.nunique())
    dataset_context = f"{filename}, all columns: {list(df.columns)}"

    # ── STEP 2: 6 stat cards ────────────────────────────────────────────────
    def _fmt(v):
        if isinstance(v, float):
            return f"{v:,.2f}"
        return f"{v:,}"

    stat_cards = [
        ("Mean", _fmt(mean_val), "#F97316"),
        ("Median", _fmt(median_val), "#38BDF8"),
        ("Std Dev", _fmt(std_val), "#A78BFA"),
        ("Min", _fmt(min_val), "#34D399"),
        ("Max", _fmt(max_val), "#FCD34D"),
        ("Outliers%", f"{outlier_pct}%", "#EF4444"),
    ]
    cols6 = st.columns(6)
    for i, (label, value, color) in enumerate(stat_cards):
        with cols6[i]:
            st.markdown(
                f'<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);'
                f'border-radius:12px;padding:14px 10px;text-align:center;animation:fadeUp 0.5s ease both;">'
                f'<div style="font-size:0.72rem;color:#94A3B8;margin-bottom:4px;">{label}</div>'
                f'<div style="font-size:1.35rem;font-weight:800;color:{color};">{value}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── STEP 3: histogram (no box plot) ─────────────────────────────────────
    fig = px.histogram(
        df,
        x=selected_col,
        nbins=30,
        marginal="rug",
        labels={selected_col: selected_col, "count": "Frequency"},
    )
    fig.update_traces(marker_color="#7F77DD", marker_line_width=0)
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="#E24B4A",
        annotation_text=f"Mean: {mean_val}",
        annotation_position="top right",
    )
    fig.add_vline(
        x=median_val,
        line_dash="dash",
        line_color="#1D9E75",
        annotation_text=f"Median: {median_val}",
        annotation_position="top left",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#E2E8F0",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        bargap=0.05,
        margin=dict(l=0, r=0, t=20, b=0),
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── STEP 4: Groq AI description (cached) ────────────────────────────────
    cache_key = f"{filename}_{selected_col}_{mean_val}_{std_val}"
    with st.spinner("Argus AI is analysing this column…"):
        ai_desc = _groq_numeric_description(
            cache_key,
            selected_col,
            dataset_context,
            mean_val,
            median_val,
            std_val,
            min_val,
            max_val,
            q1_val,
            q3_val,
            skewness_val,
            kurtosis_val,
            missing_pct,
            negative_count,
            outlier_pct,
        )

    if ai_desc is None:
        # rule-based fallback
        sk = skewness_val
        if sk > 1.0:
            shape = "strongly right-skewed"
        elif sk > 0.5:
            shape = "moderately right-skewed"
        elif sk < -1.0:
            shape = "strongly left-skewed"
        elif sk < -0.5:
            shape = "moderately left-skewed"
        else:
            shape = "roughly symmetric"
        direction = "above" if mean_val > median_val else "below"
        ai_desc = (
            f"{selected_col} is {shape} — mean ({mean_val}) {direction} "
            f"median ({median_val}). "
            f"75% of values fall between {q1_val} and {q3_val}. "
        )
        if negative_count > 0:
            ai_desc += f"Contains {negative_count:,} negative values — likely errors or returns, filter before analysis. "
        if outlier_pct > 5:
            ai_desc += f"{outlier_pct}% outliers detected by IQR rule. "
        if missing_pct > 5:
            ai_desc += f"{missing_pct}% missing — consider imputing. "

    # ── STEP 5: render description ──────────────────────────────────────────
    st.markdown(
        f'<div style="border-left:3px solid #7F77DD;padding:12px 16px;'
        f"border-radius:0 8px 8px 0;background:rgba(127,119,221,0.08);"
        f'font-size:14px;line-height:1.7;color:#ccc;margin-top:12px">'
        f"{ai_desc}</div>",
        unsafe_allow_html=True,
    )

    if ai_desc:
        ctx = st.session_state.setdefault("llm_context", [])
        insight_str = f"Numeric Column Analysis for '{selected_col}': {ai_desc}"
        if insight_str not in ctx:
            ctx.append(insight_str)


# ── Categorical Insights helpers (cached Groq calls) ─────────────────────────


@st.cache_data(show_spinner=False)
def _groq_cat_title(cache_key: str, col: str, top_10_str: str, context: str) -> str:
    """Groq call for bar-chart title (cached)."""
    prompt = (
        f"Write a 5-8 word chart title for a bar chart showing the distribution "
        f"of the column '{col}' in a dataset with these top values: {top_10_str}. "
        f"Dataset context: {context}. "
        "Examples of good titles: "
        "'Top countries by record count', "
        "'Most common product categories', "
        "'Distribution of customer segments'. "
        "Return ONLY the title. No quotes, no punctuation at end, nothing else."
    )
    try:
        from utils import get_groq_client

        client = get_groq_client()
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=40,
            stream=False,
        )
        return resp.choices[0].message.content.strip().strip('"').rstrip(".")
    except Exception:
        return f"Distribution of {col}"


@st.cache_data(show_spinner=False)
def _groq_cat_description(
    cache_key: str,
    col: str,
    dataset_context: str,
    cardinality: int,
    top_value: str,
    top_pct: float,
    top_10_str: str,
    missing_pct: float,
) -> str:
    """Groq call for categorical column description (cached). Returns None on failure."""
    prompt = (
        f"You are a data analyst. Given this column profile, write 2-3 sentences "
        f"describing the distribution for a non-technical audience.\n\n"
        f"Column: '{col}'\n"
        f"Dataset: {dataset_context}\n"
        f"Stats:\n"
        f"  - {cardinality} unique values\n"
        f"  - Top value: '{top_value}' appears in {top_pct}% of rows\n"
        f"  - Top 10 values with counts: {top_10_str}\n"
        f"  - Missing: {missing_pct}%\n\n"
        "Cover:\n"
        "1. Whether one value dominates or values are spread evenly "
        "(if top value > 50%: dominated, if top value < 10%: diverse)\n"
        "2. What the top values suggest about this dataset in plain English\n"
        "3. Any data quality concern only if missing_pct > 10%\n\n"
        "Be specific. Use actual value names and percentages from the stats. "
        "No bullet points. No markdown. Under 60 words."
    )
    try:
        from utils import get_groq_client

        client = get_groq_client()
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=130,
            stream=False,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


def _render_categorical_insights(df: pd.DataFrame):
    """
    Smart Categorical Insights — dropdown UI.
    All detected categorical columns are listed in a selectbox ranked by
    informativeness. The bar chart and AI description are generated ONLY
    after the user picks a column — no auto-loop, no per-row rendering.
    """
    import math

    filename = st.session_state.get("file_name", "dataset")

    # ── 1. Detect categorical columns ────────────────────────────────────────
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Hidden categoricals: numeric with low cardinality + keyword hint
    HINTS = [
        "year",
        "month",
        "grade",
        "rating",
        "category",
        "type",
        "status",
        "class",
        "group",
        "level",
        "rank",
        "code",
        "flag",
    ]
    for c in df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns:
        if c not in cat_cols and df[c].nunique() < 30:
            if any(hint in c.lower() for hint in HINTS):
                cat_cols.append(c)

    if not cat_cols:
        return

    # ── 2. Rank by informativeness (entropy × cardinality penalty) ────────────
    def _info_score(c):
        vc = df[c].value_counts(normalize=True)
        entropy = -sum(p * math.log2(p) for p in vc if p > 0)
        card = df[c].nunique()
        penalty = 1.0 if 2 <= card <= 200 else 0.4
        return entropy * penalty

    cat_cols_ranked = sorted(cat_cols, key=_info_score, reverse=True)

    # ── 3. Section header ─────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-title" style="margin-top:24px">'
        "\U0001f3f7 Smart Categorical Insights</div>",
        unsafe_allow_html=True,
    )

    # ── 4. Dropdown ───────────────────────────────────────────────────────────
    options = ["\u2014 select a column \u2014"] + cat_cols_ranked
    chosen = st.selectbox(
        "Choose a categorical column to explore",
        options,
        index=0,
        key="cat_insight_col",
        help="Columns are ordered by informativeness. Select one to generate the chart and AI insight.",
    )

    if chosen == "\u2014 select a column \u2014":
        st.markdown(
            '<div class="info-card" style="text-align:center;color:#64748B;">'
            "\U0001f446 Pick a column above to see its distribution chart and AI insight."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    col = chosen

    # ── 5. Compute stats ──────────────────────────────────────────────────────
    vc = df[col].value_counts()
    top10 = vc.head(10)
    cardinality = int(df[col].nunique())
    top_pct = round(float(top10.iloc[0]) / len(df) * 100, 1)
    top_value = str(top10.index[0])
    missing_pct = round(df[col].isnull().mean() * 100, 1)
    top_10_dict = {str(k): int(v) for k, v in top10.items()}
    top_10_str = str(top_10_dict)
    context = f"{filename}, columns: {list(df.columns)}"
    cache_key = f"{filename}_{col}_{top_10_str}"

    # Mini stat pills
    st.markdown(
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin:10px 0 16px;">'
        f'<span style="background:rgba(249,115,22,0.1);border:1px solid rgba(249,115,22,0.3);'
        f'border-radius:20px;padding:4px 14px;font-size:0.78rem;color:#e2e8f0;">'
        f"🔢 Unique values: <b>{cardinality}</b></span>"
        f'<span style="background:rgba(56,189,248,0.1);border:1px solid rgba(56,189,248,0.3);'
        f'border-radius:20px;padding:4px 14px;font-size:0.78rem;color:#e2e8f0;">'
        f"🥇 Top value: <b>{top_value}</b> ({top_pct}%)</span>"
        f'<span style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);'
        f'border-radius:20px;padding:4px 14px;font-size:0.78rem;color:#e2e8f0;">'
        f"⚠️ Missing: <b>{missing_pct}%</b></span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── 6. Groq title ─────────────────────────────────────────────────────────
    with st.spinner("Generating AI chart title…"):
        title = _groq_cat_title(cache_key, col, top_10_str, context)
    st.subheader(title)

    # ── 7. Bar chart (CHANGE 3 — fixed y-axis labels) ────────────────────────
    plot_data = vc if cardinality <= 15 else top10
    if cardinality > 15:
        st.caption(f"Showing top 10 of {cardinality} unique values")

    top10_df = pd.DataFrame(
        {
            "value": plot_data.index.astype(str),
            "count": plot_data.values,
        }
    )

    # Groq-generated axis labels (CHANGE 3)
    ax_cache_key = f"{filename}_{col}_axlabels"
    if ax_cache_key not in st.session_state.get("ai_cache", {}):
        try:
            y_label_prompt = (
                f"Write a 2-4 word y-axis label for a bar chart where each bar represents "
                f"a unique value of the column '{col}' in a dataset about '{context}'. "
                "Examples: col='country' → 'Country', col='chol' → 'Cholesterol Level', "
                "col='product_category' → 'Product Category'. Return ONLY the label. Nothing else."
            )
            x_label_prompt = (
                f"Write a 2-4 word x-axis label for a bar chart counting occurrences of '{col}' "
                f"values in a dataset about '{context}'. "
                "Examples: col='country' → 'Number of Records', col='chol' → 'Patient Count', "
                "col='diagnosis' → 'Case Count'. Return ONLY the label text. Nothing else."
            )
            y_label = groq_call(y_label_prompt, max_tokens=15).strip().strip('"')
            x_label = groq_call(x_label_prompt, max_tokens=15).strip().strip('"')
        except Exception:
            y_label = col.replace("_", " ").title()
            x_label = "Count"
        if "ai_cache" not in st.session_state:
            st.session_state["ai_cache"] = {}
        st.session_state["ai_cache"][ax_cache_key] = json.dumps(
            {"y": y_label, "x": x_label}
        )
    else:
        try:
            ldata = json.loads(st.session_state["ai_cache"][ax_cache_key])
            y_label = ldata.get("y", col.replace("_", " ").title())
            x_label = ldata.get("x", "Count")
        except Exception:
            y_label = col.replace("_", " ").title()
            x_label = "Count"

    fig = px.bar(
        top10_df,
        x="count",
        y="value",
        orientation="h",
        labels={"count": x_label, "value": y_label},
    )
    fig.update_traces(
        marker_color="#1D9E75",
        marker_line_width=0,
        texttemplate="%{x}",
        textposition="outside",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="#9ca3af", title_font_size=13),
        yaxis=dict(
            showgrid=False,
            color="#9ca3af",
            title_font_size=13,
            autorange="reversed",
            tickfont=dict(size=12),
        ),
        margin=dict(l=10, r=60, t=30, b=10),
        height=max(220, len(top10_df) * 34 + 60),
        uniformtext_minsize=10,
        uniformtext_mode="hide",
        font=dict(color="#9ca3af"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── 8. Groq description ───────────────────────────────────────────────────
    with st.spinner("Generating AI insight…"):
        description = _groq_cat_description(
            cache_key,
            col,
            context,
            cardinality,
            top_value,
            top_pct,
            top_10_str,
            missing_pct,
        )
    if description is None:
        if top_pct > 60:
            description = (
                f"{col} is heavily dominated by '{top_value}' ({top_pct}% of rows). "
                f"The remaining {cardinality - 1} values share the rest. "
                "Low diversity — may not be useful for segmentation."
            )
        elif cardinality > 100:
            description = (
                f"{col} has high cardinality ({cardinality} unique values). "
                f"Top value is '{top_value}' at {top_pct}%. "
                "Consider grouping rare values before modelling."
            )
        else:
            description = (
                f"{col} has {cardinality} unique values. "
                f"'{top_value}' is the most frequent ({top_pct}%). "
                "Values appear reasonably distributed across categories."
            )

    st.markdown(
        f'<div style="border-left:3px solid #7F77DD;padding:10px 16px;'
        f"border-radius:0 8px 8px 0;background:rgba(127,119,221,0.08);"
        f'font-size:14px;line-height:1.7;color:#ccc;margin-top:8px">'
        f"{description}</div>",
        unsafe_allow_html=True,
    )

    if description:
        ctx = st.session_state.setdefault("llm_context", [])
        insight_str = f"Categorical Column Analysis for '{col}' (Top value: {top_value}): {description}"
        if insight_str not in ctx:
            ctx.append(insight_str)


# ── Tab 4 · Bivariate Analysis ────────────────────────────────────────────────


def _render_bivariate_tab():
    st.markdown(
        '<div class="section-title">🔗 Bivariate Analysis</div>', unsafe_allow_html=True
    )

    df_work = st.session_state.get("df_cleaned", st.session_state.get("df"))
    target_var = st.session_state.get("target_variable", "")

    @st.dialog("Bivariate Analysis Report", width="large")
    def _show_bi_report_dialog():
        _show_pdf_scrollable(st.session_state[BI_PDF_KEY], "📄 Bivariate Analysis Report")

    if st.session_state.get(BI_PDF_KEY):
        st.success("✅ Bivariate analysis ready.")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("👁️ View Bivariate Report", key="view_bi_rep", use_container_width=True):
                _show_bi_report_dialog()
        with col2:
            if st.button("🔄 Re-run Bivariate Analysis", key="rerun_bi", use_container_width=True):
                del st.session_state[BI_PDF_KEY]
                st.rerun()
        return
    
    st.markdown(
        """
    <div class="info-card" style="border-left-color: #EF4444; background: rgba(239, 68, 68, 0.05);">
        <b>📝 Note</b>: Click <b>'Run Analysis'</b> to generate the bivariate report. 
        Reports are <b>memory-resident</b> and are not saved to the system. 
        They will be permanently removed when you log out or clear the data.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="info-card">
        <b>Intelligence Layer</b>: Groq evaluates every column's statistics, unique values, and
        distribution to identify the most important features in your data. It then automatically
        selects the best pairwise combinations for deep bivariate analysis.
        Plots include <b>scatter with trendlines</b>, <b>mean-bar charts</b>, and a
        <b>correlation heatmap</b> — all accompanied by AI-generated insights.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Optional target variable (AI will auto-select otherwise)
    if not target_var:
        tv = st.text_input(
            "🎯 Target variable (Optional)",
            key="bi_target_input",
            placeholder="e.g. SalePrice, Survived, Churn…",
            help="The column you want to predict or focus on. Leave blank and AI will auto-discover the most important relationships.",
        )
        if tv:
            st.session_state["target_variable"] = tv
            target_var = tv
    else:
        st.info(f"🎯 Analysis Focus: **{target_var}**")

    if st.button("🚀 Run Bivariate Analysis", key="run_bi"):
        with st.spinner("Running bivariate analysis…"):
            try:
                from bivariate_analysis import bi_visualize_analyze
                dataset_name = st.session_state.get("dataset_name", "dataset")
                target_variable = st.session_state.get("target_variable", "")
                
                context_items, pdf_bytes = bi_visualize_analyze(df_work, dataset_name, target_variable)
                
                # ── Store in LLM context
                ctx = st.session_state.setdefault("llm_context", [])
                ctx.extend(context_items)
                st.session_state[BI_PDF_KEY] = pdf_bytes
                st.rerun()
            except Exception as e:
                st.error(f"Bivariate analysis error: {e}")


# ── Tab 5 · Multivariate Analysis ─────────────────────────────────────────────

def _render_multivariate_tab():
    st.markdown(
        '<div class="section-title">🌐 Multivariate Analysis</div>', unsafe_allow_html=True
    )

    df_work = st.session_state.get("df_cleaned", st.session_state.get("df"))
    target_var = st.session_state.get("target_variable", "")
    MULTI_PDF_KEY = "multi_analysis_pdf"

    @st.dialog("Multivariate Analysis Report", width="large")
    def _show_multi_report_dialog():
        _show_pdf_scrollable(st.session_state[MULTI_PDF_KEY], "📄 Multivariate Analysis Report")

    if st.session_state.get(MULTI_PDF_KEY):
        st.success("✅ Multivariate analysis ready.")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("👁️ View Multivariate Report", key="view_multi_rep", use_container_width=True):
                _show_multi_report_dialog()
        with col2:
            if st.button("🔄 Re-run Multivariate Analysis", key="rerun_multi", use_container_width=True):
                del st.session_state[MULTI_PDF_KEY]
                st.rerun()
        return

    st.markdown(
        '''
    <div class="info-card" style="border-left-color: #EF4444; background: rgba(239, 68, 68, 0.05);">
        <b>📝 Note</b>: Click <b>'Run Analysis'</b> to generate the multivariate report. 
        Reports are <b>memory-resident</b> and are not saved to the system. 
        They will be permanently removed when you log out or clear the data.
    </div>
    ''',
        unsafe_allow_html=True,
    )

    st.markdown(
        '''
    <div class="info-card">
        <b>Intelligence Layer</b>: Groq evaluates the entire dataset simultaneously. It discovers complex relationships 
        like interconnected clusters, multicollinearity, and natural group behaviors across all features.
        Plots include <b>Connection Maps</b>, <b>Top Influencers</b>, and <b>Group Profiles</b> 
        — all dynamically generated using AI.
    </div>
    ''',
        unsafe_allow_html=True,
    )

    if not target_var:
        st.info("🎯 **Discovery Mode Active**: No specific target selected. The AI will autonomously map clusters and interactions across the entire dataset.")
    else:
        st.info(f"🎯 **Analysis Focus**: **{target_var}**")

    if st.button("🚀 Run Multivariate Analysis", key="run_multi"):
        with st.spinner("Running deep multivariate analysis…"):
            try:
                from multivariate_analysis import multi_visualize_analyze
                dataset_name = st.session_state.get("dataset_name", "dataset")
                target_variable = st.session_state.get("target_variable", "")
                
                context_items, pdf_bytes = multi_visualize_analyze(df_work, dataset_name, target_variable)
                
                ctx = st.session_state.setdefault("llm_context", [])
                ctx.extend(context_items)
                st.session_state[MULTI_PDF_KEY] = pdf_bytes
                st.rerun()
            except Exception as e:
                st.error(f"Multivariate analysis error: {e}")


# ── Tab 6 · Feature Importance ────────────────────────────────────────────────


def _render_feature_tab():
    st.markdown(
        '<div class="section-title">🌟 AI Discovery Engineering</div>',
        unsafe_allow_html=True,
    )

    df_work = st.session_state.get("df_cleaned", st.session_state.get("df"))
    dataset_name = st.session_state.get("dataset_name", "dataset")

    # State: Use st.session_state["discovery_results"] to store everything
    results = st.session_state.get("discovery_results")

    if results:
        # ── Toggle for switching targets (AI-discovered or manual override) ──
        col_t1, col_t2 = st.columns([1, 4])
        with col_t1:
            if st.button("🔄 New Discovery", key="re-run_discovery"):
                del st.session_state["discovery_results"]
                st.rerun()
        with col_t2:
            st.success(f"Discovered Target: **{results['target']}**")

        st.markdown("---")

        # Layout: Row 1 (Importance + Heatmap)
        r1_c1, r1_c2 = st.columns([6, 4])
        with r1_c1:
            st.plotly_chart(results["importance_fig"], use_container_width=True, config={"displayModeBar": False})
        with r1_c2:
            st.plotly_chart(results["heatmap_fig"], use_container_width=True, config={"displayModeBar": False})

        # Layout: Row 2 (Top Relationship + AI Insight)
        r2_c1, r2_c2 = st.columns([5, 5])
        with r2_c1:
            st.plotly_chart(results["relationship_fig"], use_container_width=True, config={"displayModeBar": False})
        with r2_c2:
            st.markdown(
                f"""
                <div class="info-card" style="height:440px; overflow-y:auto; border-left-color: #F97316;">
                    <h4 style="color:#F97316; margin-top:0;">💡 Discovery Story: {results['target']}</h4>
                    <p style="font-size:0.92rem; line-height:1.6; color:#E2E8F0;">
                        {results['ai_insight']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return

    # Discovery Mode UI
    st.markdown(
        """
    <div class="info-card">
        <b>Intelligence Layer</b>: Argus will now autonomously identify the most logical 'Target' variable 
        and analyze which features have the strongest predictive influence. Click below to begin discovery.
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Auto-Discover Target & Features", key="auto_run_feat", use_container_width=True):
            _run_discovery_engine(df_work, dataset_name)
    with col2:
        num_cols = df_work.select_dtypes(include="number").columns.tolist()
        manual_target = st.selectbox("Or Focus Manually", [""] + num_cols, key="feat_manual_target")
        if manual_target:
            _run_discovery_engine(df_work, dataset_name, target=manual_target)


def _run_discovery_engine(df: pd.DataFrame, dataset_name: str, target: str = None):
    """Main automated discovery pipeline."""
    with st.spinner("AI is engineering discovery — analyzing column semantics…"):
        try:
            from univariate import UnivariateAnalyzer1
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

            # 1. AI Discovery (Target Selection)
            if not target:
                analyzer = UnivariateAnalyzer1(df)
                stats = analyzer.analyze()
                target = _ai_suggest_target(dataset_name, stats)
                if not target:
                    st.error("AI could not identify a clear target. Please select one manually.")
                    return

            # 2. Algorithmic Importance
            num_cols = df.select_dtypes(include="number").columns.tolist()
            if target in num_cols:
                feature_cols = [c for c in num_cols if c != target]
            else:
                # If target is categorical, try to use it
                feature_cols = num_cols

            if not feature_cols:
                st.error("Not enough numeric columns available for importance ranking.")
                return

            X = df[feature_cols].fillna(df[feature_cols].median() if not df[feature_cols].empty else 0)
            y = df[target].fillna(df[target].mode()[0] if not df[target].empty else 0)

            # Detect model type
            is_class = pd.api.types.is_object_dtype(y) or y.nunique() <= 10
            if is_class:
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            model.fit(X, y)
            imp_df = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": model.feature_importances_,
            }).sort_values("Importance", ascending=False)

            # 3. Visuals: Importance Plot
            fig_imp = px.bar(
                imp_df.head(10), x="Feature", y="Importance",
                title=f"Feature Weights — Target: {target}",
                color="Importance", color_continuous_scale=["#38BDF8", "#F97316"]
            )
            fig_imp.update_layout(**_chart_layout(), height=440)

            # 4. Visuals: Heatmap (top features + target)
            top_features = imp_df["Feature"].head(8).tolist()
            corr_cols = [target] + top_features if pd.api.types.is_numeric_dtype(df[target]) else top_features
            corr = df[corr_cols].corr(numeric_only=True)
            fig_heat = px.imshow(
                corr, text_auto=".1f", color_continuous_scale="RdBu_r",
                title="Correlation Heatmap (Top Features)"
            )
            fig_heat.update_layout(**_chart_layout(), height=440)

            # 5. Visuals: Top Relationship Plot
            top_feat = top_features[0]
            if is_class:
                fig_rel = px.box(df, x=target, y=top_feat, title=f"Influence Analysis: {top_feat} on {target}", color=target)
            else:
                fig_rel = px.scatter(df, x=top_feat, y=target, trendline="ols", title=f"Trend: {top_feat} vs {target}", opacity=0.6)
            fig_rel.update_layout(**_chart_layout(), height=440)

            # 6. AI Insight Story
            ai_insight = _get_discovery_story(dataset_name, target, imp_df.head(5).to_dict('records'))

            # Store in session state
            st.session_state["discovery_results"] = {
                "target": target,
                "importance_fig": fig_imp,
                "heatmap_fig": fig_heat,
                "relationship_fig": fig_rel,
                "ai_insight": ai_insight
            }
            st.rerun()

        except Exception as e:
            st.error(f"Discovery error: {e}")


def _ai_suggest_target(dataset_name: str, stats: dict) -> str:
    """Prompt LLM to pick the most likely target for prediction."""
    col_str = "\n".join([f"- {col} ({s['type']})" for col, s in list(stats.items())[:30]])
    prompt = (
        f"Dataset: '{dataset_name}'\n"
        f"Available Columns:\n{col_str}\n\n"
        f"Identify the SINGLE most logical column to use as a 'Target' variable for machine learning "
        f"(e.g., survival, price, churn, success). "
        f"Return ONLY the exact column name and nothing else."
    )
    try:
        from utils import get_groq_client
        client = get_groq_client()
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=60
        )
        found = resp.choices[0].message.content.strip().split("\n")[0].replace('"', '')
        return found if found in stats else None
    except: return None


def _get_discovery_story(dataset_name: str, target: str, features: list) -> str:
    """Get a detailed AI explanation of the discoveries."""
    feats_str = ", ".join([f"{f['Feature']} ({round(f['Importance']*100, 1)}%)" for f in features])
    prompt = (
        f"Dataset: '{dataset_name}'. Discovered Target: '{target}'.\n"
        f"Top Features found by RandomForest: {feats_str}.\n\n"
        f"As a Lead Data Scientist, write exactly 2 formal analytical paragraphs (about 120 words total) "
        f"explaining the 'Discovery Story' for this dataset. Use professional language. "
        f"Explain why '{target}' was identified as the primary focus, how the top feature relates to it, "
        f"and what this implies about the underlying business/natural process in the '{dataset_name}' data."
    )
    try:
        from utils import get_groq_client
        client = get_groq_client()
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=250
        )
        return resp.choices[0].message.content.strip()
    except:
        return "Discovery insight currently unavailable, but the algorithmic correlations are displayed above."


# ── Tab 6 · AI Chat ───────────────────────────────────────────────────────────


def _render_chat_tab(eda: dict):
    st.markdown(
        '<div class="section-title">💬 EDA Insights Chat</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="chip-label">Quick questions:</div>', unsafe_allow_html=True
    )

    chip_cols = st.columns(len(SUGGESTIONS))
    for i, s in enumerate(SUGGESTIONS):
        with chip_cols[i]:
            if st.button(s, key=f"chip_{i}", use_container_width=True):
                _handle_chat(s, eda)

    for msg in st.session_state.get("chat_history", []):
        cls = "bubble-ai" if msg["role"] == "assistant" else "bubble-user"
        pfx = "🤖 **Glimpse AI**" if msg["role"] == "assistant" else "🧑 **You**"
        st.markdown(
            f'<div class="chat-bubble {cls}">'
            f'<span class="bubble-prefix">{pfx}</span><br>{msg["content"]}'
            f"</div>",
            unsafe_allow_html=True,
        )

    with st.form("chat_form", clear_on_submit=True):
        ci, cb = st.columns([5, 1])
        with ci:
            q = st.text_input(
                "",
                placeholder="Ask a question about your data…",
                label_visibility="collapsed",
            )
        with cb:
            sent = st.form_submit_button("➤", use_container_width=True)
    if sent and q.strip():
        _handle_chat(q.strip(), eda)


def _handle_chat(q: str, eda: dict):
    hist = st.session_state.setdefault("chat_history", [])
    hist.append({"role": "user", "content": q})
    llm_ctx = st.session_state.get("llm_context", [])
    with st.spinner("Argus AI is thinking…"):
        ans = chat_response(q, eda, llm_context=llm_ctx, chat_history=hist)
    hist.append({"role": "assistant", "content": ans})
    st.session_state["just_asked_question"] = True
    st.rerun()


# ── PDF / image scrollable viewer ─────────────────────────────────────────────


def _show_pdf_scrollable(pdf_data: bytes, title: str = "📄 Output"):
    """Render a PDF inline: converts pages to images (PyMuPDF) with iframe fallback."""
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

    try:
        import fitz  # PyMuPDF

        # Open from bytes
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        pages_b64 = []
        for page in doc:
            pix = page.get_pixmap(dpi=140, colorspace=fitz.csRGB)
            pages_b64.append(base64.b64encode(pix.tobytes("png")).decode())
        doc.close()

        imgs_html = "".join(
            f'<img src="data:image/png;base64,{b64}" '
            f'style="width:100%;border-radius:10px;margin-bottom:14px;'
            f'box-shadow:0 4px 20px rgba(0,0,0,0.45);" />'
            for b64 in pages_b64
        )
        st.markdown(
            f'<div style="height:700px;overflow-y:auto;padding:16px;'
            f"background:rgba(0,0,0,0.25);border:1px solid rgba(249,115,22,0.28);"
            f'border-radius:16px;">{imgs_html}</div>',
            unsafe_allow_html=True,
        )
    except Exception:
        # Fallback to base64 iframe
        b64 = base64.b64encode(pdf_data).decode()
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{b64}" '
            f'width="100%" height="700px" '
            f'style="border:1px solid rgba(249,115,22,0.28);border-radius:16px;'
            f'background:#0A0F1E;"></iframe>',
            unsafe_allow_html=True,
        )

    st.download_button(
        f"⬇️ Download {title}.pdf",
        pdf_data,
        file_name=f"{title.replace(' ', '_')}.pdf",
        mime="application/pdf",
        key=f"dl_{title.replace(' ', '_')}",
    )


# ── Pre-upload showcase ───────────────────────────────────────────────────────


def _render_pre_upload_showcase():
    st.markdown(
        """
<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:0 0 28px 0;">

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
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    <div class="upload-placeholder">
        <div class="upload-icon-anim">&#9729;&#65039;</div>
        <div class="upload-text">
            Drag &amp; drop your <b>.xlsx</b> or <b>.csv</b> file above,<br>
            or click <b>Browse files</b> to get started.
        </div>
        <div class="upload-hint">An interactive EDA dashboard + AI chat will appear after upload.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ── Header ────────────────────────────────────────────────────────────────────


def _render_header():
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
<div class="argus-logo" title="Argus EDA">
  <img src="data:image/png;base64,{b64_logo}" alt="Argus Logo" style="width:70px; height:70px; border-radius:50%; box-shadow:0 0 16px rgba(249,115,22,0.3); animation:logoPulse 3s ease-in-out infinite;">
</div>
"""

    cl, ct, cu = st.columns([1, 6, 2])
    with cl:
        st.markdown(argus_logo_html, unsafe_allow_html=True)

    with ct:
        st.markdown(
            '<h1 class="app-title">Argus '
            '<span class="app-sub">An AI based Automated EDA Tool</span></h1>'
            '<div class="typing-wrapper">'
            '<span class="typing-cursor" id="typing-el"></span>'
            '<span class="cursor-bar">|</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        import streamlit.components.v1 as components

        components.html(
            """
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
            var el = window.parent.document.getElementById('typing-el');
            if(!el) return;
            if (el.getAttribute('data-typing') === 'true') return;
            el.setAttribute('data-typing', 'true');
            var pi = 0, ci = 0, deleting = false;
            function tick() {
                var phrase = phrases[pi];
                el.textContent = deleting ? phrase.substring(0, ci--) : phrase.substring(0, ci++);
                var speed = deleting ? 40 : 70;
                if (!deleting && ci > phrase.length) { speed = 1400; deleting = true; }
                if (deleting && ci < 0) { deleting = false; pi = (pi + 1) % phrases.length; ci = 0; speed = 300; }
                setTimeout(tick, speed);
            }
            tick();
        })();
        </script>
        """,
            height=0,
            width=0,
        )

    with cu:
        name = st.session_state.get("user_name", "User")
        email = st.session_state.get("user_email", "")
        st.markdown(
            f'<div class="user-badge">👤 <b>{name}</b><br>'
            f'<span class="user-email">{email}</span></div>',
            unsafe_allow_html=True,
        )
        if st.button("Logout", key="logout_btn"):
            from auth import logout

            logout()

    st.markdown('<hr class="hdr-divider"/>', unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _metric_card(col, icon, label, value, unit):
    with col:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="mc-icon">{icon}</div>'
            f'<div class="mc-label">{label}</div>'
            f'<div class="mc-value">{value}</div>'
            f'<div class="mc-unit">{unit}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )


def _chart_layout() -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E2E8F0",
        font_size=12,
        title_font_size=14,
        title_font_color="#F97316",
        margin=dict(l=16, r=16, t=44, b=16),
    )


def _show_processing_bar():
    bar = st.progress(0, text="Loading file…")
    for pct, msg in [
        (20, "Parsing columns…"),
        (45, "Computing statistics…"),
        (70, "Detecting types…"),
        (90, "Generating insights…"),
        (100, "Done! ✅"),
    ]:
        time.sleep(0.35)
        bar.progress(pct, text=msg)
    time.sleep(0.3)
    bar.empty()


# ── CSS ───────────────────────────────────────────────────────────────────────


def _inject_css():
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    #MainMenu, footer { visibility: hidden; }
    .stApp { background: linear-gradient(135deg,#0A0F1E 0%,#0F1A2E 60%,#1A0A2E 100%);
              font-family:'Inter',sans-serif; }
    .block-container { padding-top: 1.2rem !important; }
    /* ── Argus Logo (SVG inline) */
    .argus-logo { display:flex; align-items:center; justify-content:center; margin-top:4px; }
    .argus-logo svg { filter:drop-shadow(0 0 10px rgba(249,115,22,0.55));
                      animation:logoPulse 3s ease-in-out infinite; }
    .argus-ring { animation:rotateSlow 12s linear infinite; transform-origin:32px 32px; }
    .argus-eye  { animation:eyePulse 2.5s ease-in-out infinite; }
    .argus-scan { animation:scanFade 2s ease-in-out infinite alternate; }
    @keyframes rotateSlow { to { transform: rotate(360deg); } }
    @keyframes eyePulse   { 0%,100%{r:3.5;opacity:0.9} 50%{r:4.5;opacity:1} }
    @keyframes scanFade   { from{opacity:0.2} to{opacity:0.8} }
    @keyframes logoPulse {
        0%,100%{filter:drop-shadow(0 0 10px rgba(249,115,22,0.45))}
        50%    {filter:drop-shadow(0 0 22px rgba(249,115,22,0.85))}
    }
    .logo-icon { font-size:2.8rem; margin-top:10px; }
    /* ── Hide Plotly modebar (zoom/pan toolbar) on all charts */
    .modebar { display: none !important; }
    .modebar-container { display: none !important; }
    [data-testid="stPlotlyChart"] .modebar { display: none !important; }
    .js-plotly-plot .plotly .modebar { display: none !important; }
    /* ── Glance card strip */
    .glance-scroll-row {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        gap: 12px !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        padding-bottom: 10px !important;
        -webkit-overflow-scrolling: touch !important;
        scrollbar-width: thin !important;
        scrollbar-color: rgba(249,115,22,0.4) transparent !important;
        margin-bottom: 20px !important;
    }
    .glance-scroll-row::-webkit-scrollbar { height: 5px; }
    .glance-scroll-row::-webkit-scrollbar-thumb { background: rgba(249,115,22,0.4); border-radius: 4px; }
    .glance-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 18px 20px;
        cursor: default;
        transition: border-color 0.2s, transform 0.2s;
        flex: 0 0 auto;
        min-width: 140px;
        max-width: 180px;
    }
    .glance-card:hover { border-color: rgba(249,115,22,0.5); transform: translateY(-2px); }
    .glance-label { color: #9ca3af; font-size: 12px; margin: 0 0 8px; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .glance-value { color: #ffffff; font-size: 26px; font-weight: 700; margin: 0 0 6px; letter-spacing: -0.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .glance-sub   { color: #6b7280; font-size: 11px; margin: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    /* ── Header */
    .app-title  { font-size:2rem; font-weight:800; margin:0; color:#F97316; }
    .app-sub    { font-size:1rem; font-weight:400; color:#94A3B8; }
    .user-badge { background:rgba(249,115,22,0.08); border:1px solid rgba(249,115,22,0.3);
                  border-radius:10px; padding:8px 12px; font-size:0.82rem;
                  color:#E2E8F0; text-align:right; }
    .user-email { color:#64748B; font-size:0.72rem; }
    .hdr-divider{ border:none; border-top:1px solid rgba(249,115,22,0.2); margin:10px 0 18px 0; }
    /* ── Typing */
    .typing-wrapper { display:inline-flex; align-items:center; gap:2px;
                      min-height:26px; margin-top:4px; }
    .typing-cursor  { font-size:0.95rem; color:#94A3B8; font-weight:500; }
    .cursor-bar     { font-size:1rem; color:#F97316; font-weight:300;
                      animation:blink 0.85s step-start infinite; }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
    /* ── Section titles */
    .section-title { font-size:1.12rem; font-weight:700; color:#F97316;
                     margin:18px 0 10px; letter-spacing:0.03em; }
    /* ── Info card */
    .info-card {
        background:rgba(56,189,248,0.06); border:1px solid rgba(56,189,248,0.25);
        border-radius:14px; padding:16px 20px; color:#94A3B8;
        font-size:0.9rem; line-height:1.8; margin-bottom:18px;
    }
    /* ── Tabs */
    [data-testid="stTabs"] > div:first-child {
        border-bottom: 4px solid #F97316 !important;
        gap: 10px !important;
        padding-bottom: 0 !important;
        margin-bottom: 20px !important;
        flex-wrap: wrap !important;
    }
    /* Inactive tab */
    [data-testid="stTab"] {
        color: #CBD5E1 !important;
        font-weight: 900 !important;
        font-size: 1.05rem !important;
        padding: 13px 22px !important;
        min-width: 120px !important;
        text-align: center !important;
        background: linear-gradient(160deg, rgba(30,42,68,0.9) 0%, rgba(15,26,46,0.95) 100%) !important;
        border: 2px solid rgba(56,189,248,0.35) !important;
        border-bottom: none !important;
        border-radius: 14px 14px 0 0 !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
    }
    /* Hover — blue glow */
    [data-testid="stTab"]:hover {
        background: linear-gradient(160deg, rgba(56,189,248,0.22) 0%, rgba(14,165,233,0.08) 100%) !important;
        border-color: #38BDF8 !important;
        color: #FFFFFF !important;
        box-shadow: 0 -4px 18px rgba(56,189,248,0.35) !important;
        transform: translateY(-3px) !important;
    }
    /* Active tab — bold orange gradient + strong glow */
    [data-testid="stTab"][aria-selected="true"] {
        color: #FFFFFF !important;
        font-size: 1.1rem !important;
        background: linear-gradient(135deg, #FF8C00 0%, #F97316 50%, #EA580C 100%) !important;
        border: 2.5px solid #FDBA74 !important;
        border-bottom: none !important;
        box-shadow: 0 -8px 30px rgba(249,115,22,0.55) !important;
        transform: translateY(-5px) !important;
        text-shadow: 0 1px 4px rgba(0,0,0,0.4) !important;
    }
    /* Hide native Streamlit active tab sweeping underline */
    [data-baseweb="tab-highlight"] {
        display: none !important;
        background-color: transparent !important;
    }
    /* ── Upload zone */
    [data-testid="stFileUploader"] > div:first-child {
        background:rgba(255,255,255,0.03) !important;
        border:2px dashed rgba(249,115,22,0.4) !important;
        border-radius:16px !important; padding:24px !important;
        transition:border-color 0.3s,background 0.3s;
    }
    [data-testid="stFileUploader"] > div:first-child:hover {
        border-color:rgba(249,115,22,0.8) !important;
        background:rgba(249,115,22,0.04) !important;
    }
    /* ── Upload placeholder */
    .upload-placeholder { text-align:center; padding:36px 0 20px; color:#475569; }
    .upload-icon-anim   { font-size:3rem; animation:float 3s ease-in-out infinite; }
    .upload-text  { font-size:1rem; color:#64748B; line-height:1.7; }
    .upload-hint  { font-size:0.82rem; color:#334155; margin-top:8px; }
    @keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }
    /* ── Metric cards */
    .metric-card {
        background:rgba(255,255,255,0.04); border:1px solid rgba(249,115,22,0.2);
        border-radius:16px; padding:18px 14px; text-align:center;
        transition:transform 0.25s,box-shadow 0.25s; animation:fadeUp 0.5s ease both;
    }
    .metric-card:hover { transform:translateY(-4px); box-shadow:0 10px 28px rgba(249,115,22,0.2); }
    @keyframes fadeUp { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:none} }
    .mc-icon  { font-size:1.6rem; margin-bottom:6px; }
    .mc-label { font-size:0.78rem; color:#94A3B8; }
    .mc-value { font-size:2rem; font-weight:800; color:#F97316; }
    .mc-unit  { font-size:0.7rem; color:#475569; }
    /* ── No-missing banner */
    .no-missing {
        background:rgba(52,211,153,0.1); border:1px solid rgba(52,211,153,0.35);
        border-radius:14px; padding:28px; text-align:center;
        color:#34D399; font-size:1rem; font-weight:600;
        height:340px; display:flex; align-items:center; justify-content:center;
    }
    /* ── Chat */
    .chip-label  { font-size:0.8rem; color:#64748B; margin-bottom:6px; }
    .chat-bubble { border-radius:14px; padding:14px 18px; margin:7px 0;
                   font-size:0.92rem; line-height:1.6; animation:fadeUp 0.3s ease; }
    .bubble-ai   { background:rgba(249,115,22,0.08); border:1px solid rgba(249,115,22,0.2); color:#E2E8F0; }
    .bubble-user { background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1);
                   color:#CBD5E1; text-align:right; }
    .bubble-prefix { font-size:0.75rem; color:#64748B; }
    /* ── Inputs */
    [data-testid="stTextInput"] input {
        background:rgba(255,255,255,0.05) !important;
        border:1px solid rgba(249,115,22,0.3) !important;
        border-radius:12px !important; color:#E2E8F0 !important; padding:12px 16px !important;
    }
    [data-testid="stTextInput"] input:focus {
        border-color:#F97316 !important; box-shadow:0 0 0 3px rgba(249,115,22,0.15) !important;
    }
    /* ── Buttons (scoped to avoid tab conflicts) */
    [data-testid="stButton"] > button {
        background: rgba(249,115,22,0.12) !important;
        border: 1px solid rgba(249,115,22,0.35) !important;
        color: #FB923C !important;
        border-radius: 12px !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        min-height: 42px !important;
        transition: all 0.2s !important;
        white-space: normal !important;
        height: auto !important;
        width: 100% !important;
        box-sizing: border-box !important;
        display: block !important;
    }
    [data-testid="stButton"] > button:hover {
        background: rgba(249,115,22,0.25) !important;
        border-color: #F97316 !important;
        color: #fff !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(249,115,22,0.3) !important;
    }
    /* Primary (type="primary") buttons — solid filled */
    [data-testid="stButton"] > button[kind="primary"],
    [data-testid="stButton"] > button[data-testid*="primary"] {
        background: linear-gradient(135deg, #F97316 0%, #EA580C 100%) !important;
        border: none !important;
        color: #fff !important;
        box-shadow: 0 4px 18px rgba(249,115,22,0.35) !important;
    }
    [data-testid="stButton"] > button[kind="primary"]:hover,
    [data-testid="stButton"] > button[data-testid*="primary"]:hover {
        background: linear-gradient(135deg, #FB923C 0%, #F97316 100%) !important;
        box-shadow: 0 6px 24px rgba(249,115,22,0.55) !important;
        transform: translateY(-2px) !important;
    }
    /* Demo dataset button stack — full-width, spaced */
    .demo-btn-stack { display: flex; flex-direction: column; width: 100%; }
    .demo-btn-stack [data-testid="stButton"] { width: 100% !important; }
    /* Form submit buttons (Login / Sign Up) */
    [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, rgba(249,115,22,0.2), rgba(234,88,12,0.15)) !important;
        border: 1px solid rgba(249,115,22,0.5) !important;
        color: #FB923C !important;
        border-radius: 12px !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        padding: 10px 24px !important;
        transition: all 0.2s !important;
    }
    [data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(135deg, rgba(249,115,22,0.4), rgba(234,88,12,0.3)) !important;
        border-color: #F97316 !important;
        color: #fff !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(249,115,22,0.35) !important;
    }
    /* ── Data Summary */
    .summary-header-card {
        background:rgba(249,115,22,0.08); border:1px solid rgba(249,115,22,0.28);
        border-radius:14px; padding:14px 20px; font-size:0.92rem; color:#E2E8F0;
        margin-bottom:14px; display:flex; align-items:center; gap:10px;
        animation:fadeUp 0.4s ease both;
    }
    .sum-icon { font-size:1.4rem; }
    .sum-mem  { font-size:0.8rem; color:#64748B; }
    .sum-bars { display:flex; flex-direction:column; gap:9px; margin-bottom:14px; }
    .sum-bar-row { display:flex; align-items:center; gap:10px; }
    .sum-bar-label { width:100px; font-size:0.78rem; color:#94A3B8; flex-shrink:0; }
    .sum-bar-track { flex:1; height:8px; background:rgba(255,255,255,0.07);
                     border-radius:10px; overflow:hidden; }
    .sum-bar-fill  { height:100%; border-radius:10px;
                     animation:fillBar 1s cubic-bezier(.4,0,.2,1) forwards; }
    .sum-bar-val   { font-size:0.78rem; color:#64748B; width:120px; flex-shrink:0; }
    /* ── DataFrame / Charts */
    [data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; }
    .js-plotly-plot { border-radius:16px; overflow:hidden; }

    /* ═══════════════════════════════════════════════════════════════
       RESPONSIVE — Dashboard (home.py)
       ═══════════════════════════════════════════════════════════════ */

    /* Tablet (≤ 900px) */
    @media (max-width: 900px) {
        .block-container { padding-left: 12px !important; padding-right: 12px !important; }

        /* Tabs: allow wrapping */
        [data-testid="stTabs"] > div:first-child {
            flex-wrap: wrap !important;
            gap: 6px !important;
        }
        [data-testid="stTab"] {
            font-size: 0.82rem !important;
            padding: 8px 12px !important;
            min-width: 80px !important;
        }

        /* Header: shrink title */
        .app-title { font-size: 1.5rem !important; }
        .app-sub   { font-size: 0.85rem !important; }
    }

    /* Mobile (≤ 600px) */
    @media (max-width: 600px) {
        .block-container { padding-left: 6px !important; padding-right: 6px !important; }

        /* ── App header: stack logo/title/badge vertically */
        /* Streamlit wraps st.columns to rows on narrow viewports; we reinforce */
        .argus-logo img { width: 44px !important; height: 44px !important; }
        .app-title { font-size: 1.3rem !important; }
        .app-sub   { font-size: 0.78rem !important; }
        .user-badge { font-size: 0.72rem !important; padding: 5px 8px !important; }

        /* ── Section title */
        .section-title { font-size: 0.95rem !important; }

        /* ── Buttons: full-width, proper touch target, separated */
        [data-testid="stButton"] {
            width: 100% !important;
            margin-bottom: 2px !important;
        }
        [data-testid="stButton"] > button {
            width: 100% !important;
            min-height: 46px !important;
            font-size: 0.95rem !important;
            padding: 10px 16px !important;
            border-radius: 12px !important;
            line-height: 1.4 !important;
        }
        /* Ensure Streamlit column-based layouts don't crash buttons together */
        [data-testid="stHorizontalBlock"] { gap: 8px !important; }
        [data-testid="column"] { min-width: 0 !important; }

        /* ── Tabs: horizontal scroll instead of wrapped cramming */
        [data-testid="stTabs"] > div:first-child {
            flex-wrap: nowrap !important;
            overflow-x: auto !important;
            overflow-y: hidden !important;
            gap: 8px !important;
            padding-bottom: 8px !important;
            -webkit-overflow-scrolling: touch;
        }
        [data-testid="stTabs"] > div:first-child::-webkit-scrollbar { height: 4px; }
        [data-testid="stTabs"] > div:first-child::-webkit-scrollbar-thumb { background: rgba(249,115,22,0.4); border-radius: 4px; }

        [data-testid="stTab"] {
            font-size: 0.85rem !important;
            padding: 8px 14px !important;
            min-width: max-content !important;
            white-space: nowrap !important;
            flex-shrink: 0 !important;
            letter-spacing: 0 !important;
        }
        [data-testid="stTab"][aria-selected="true"] {
            font-size: 0.9rem !important;
        }

        /* ── Override rigid 3-column inline HTML grids on mobile (e.g. Overview target cards) */
        div[style*="grid-template-columns:repeat(3,1fr)"] {
            grid-template-columns: 1fr !important;
            gap: 8px !important;
            margin-bottom: 12px !important;
        }

        /* ── Overview Overview cards: make the 2-col Streamlit layout feel better */
        /* Since Streamlit's st.columns can't be overridden in CSS alone, we make
           the custom HTML cards inside them scroll-friendly */
        div[style*="background:#1a1f2e;border:1px solid #2d3748"] {
            min-height: unset !important;
            margin-bottom: 10px !important;
            padding: 12px 14px !important;
        }

        /* ── Overview: stat mini-boxes inside the cards */
        div[style*="display:flex;gap:12px;margin-bottom:16px"] {
            flex-direction: column !important;
            gap: 8px !important;
        }
        div[style*="flex:1;background:#E24B4A11"],
        div[style*="flex:1;background:#EF9F2711"],
        div[style*="flex:1;background:#7F77DD11"],
        div[style*="flex:1;background:#7F77DD22"],
        div[style*="flex:1;background:#1D9E7522"],
        div[style*="flex:1;background:#378ADD22"] {
            flex: unset !important;
            width: 100% !important;
            box-sizing: border-box !important;
        }

        /* ── Protect glance strip — always horizontal on all screen sizes */
        .glance-scroll-row {
            flex-direction: row !important;
            flex-wrap: nowrap !important;
        }
        .glance-card {
            flex: 0 0 auto !important;
            min-width: 130px !important;
            max-width: 165px !important;
        }
        .glance-value { font-size: 22px !important; }

        /* ── Missing value squares: wrap on mobile */
        div[style*="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px"] {
            gap: 5px !important;
        }

        /* ── Upload placeholder / hint */
        .upload-placeholder { padding: 20px 0 14px !important; }
        .upload-icon-anim { font-size: 2.2rem !important; }
        .upload-text { font-size: 0.85rem !important; }

        /* ── Metric cards row */
        .metric-card { padding: 12px 8px !important; border-radius: 12px !important; }
        .mc-value { font-size: 1.5rem !important; }
        .mc-icon  { font-size: 1.2rem !important; }

        /* ── Chat bubbles */
        .chat-bubble { font-size: 0.82rem !important; padding: 10px 12px !important; }

        /* ── Plotly charts: avoid horizontal overflow */
        .js-plotly-plot { max-width: 100% !important; overflow-x: hidden !important; }

        /* ── AI insight box */
        div[style*="border-left:3px solid #7F77DD"] {
            font-size: 12px !important;
            padding: 8px 10px !important;
        }

        /* ── Data summary bar labels */
        .sum-bar-label { width: 70px !important; font-size: 0.7rem !important; }
        .sum-bar-val   { width: 80px !important; font-size: 0.7rem !important; }

        /* ── Summary header card */
        .summary-header-card { flex-direction: column !important; align-items: flex-start !important; gap: 6px !important; }

        /* ── Streamlit file uploader zone */
        [data-testid="stFileUploader"] > div:first-child { padding: 16px !important; }
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

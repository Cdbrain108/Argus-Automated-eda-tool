"""
home.py — Unified Argus EDA Dashboard.
Combines: data overview, cleaning, univariate, bivariate, feature importance, AI chat.
Generated PDFs and images are displayed inline as scrollable content.
"""

import os, base64, time, warnings, json, re as _re
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
        f'border-radius:0 8px 8px 0;background:rgba(127,119,221,0.08);'
        f'font-size:13px;line-height:1.7;color:#9ca3af;margin-top:10px">'
        f'{text}</div>',
        unsafe_allow_html=True,
    )

warnings.filterwarnings("ignore")


# ── dark_card helper ────────────────────────────────────────────────────────────

def dark_card(content_html, badge_text=None, badge_color=None):
    badge_html = ""
    if badge_text:
        badge_html = (
            f'<span style="display:inline-flex;align-items:center;'
            f'font-size:14px;padding:3px 8px;border-radius:20px;font-weight:600;'
            f'letter-spacing:.04em;background:{badge_color}22;'
            f'color:{badge_color};margin-bottom:8px">{badge_text}</span><br>'
        )
    return (
        f'<div style="background:#1a1f2e;border:1px solid #2d3748;'
        f'border-radius:14px;padding:18px 20px;margin-bottom:12px;'
        f'min-height:200px">'
        f'{badge_html}{content_html}'
        f'</div>'
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
        if outlier_penalties else 0
    )
    results["health_score"]     = max(0, 100 - missing_penalty - dup_penalty - outlier_penalty)
    results["missing_penalty"]  = missing_penalty
    results["dup_penalty"]      = dup_penalty
    results["outlier_penalty"]  = outlier_penalty

    # --- column breakdown ---
    results["n_numeric"]      = len(df.select_dtypes(include="number").columns)
    results["n_categorical"]  = len(df.select_dtypes(include=["object", "category"]).columns)
    results["n_datetime"]     = len(df.select_dtypes(include="datetime").columns)
    results["n_cols"]         = len(df.columns)
    results["n_rows"]         = len(df)

    # --- missing heatmap ---
    results["missing_pct_per_col"] = (df.isnull().mean() * 100).round(1).to_dict()

    # --- skewness ---
    skew_scores = {}
    for col in num_cols:
        clean = df[col].dropna()
        if len(clean) > 10:
            try:
                val = float(scipy_stats.skew(clean)) if _has_scipy else float(clean.skew())
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
        mask = df[num_cols].apply(
            lambda c: (
                (c < c.quantile(0.25) - 1.5 * (c.quantile(0.75) - c.quantile(0.25))) |
                (c > c.quantile(0.75) + 1.5 * (c.quantile(0.75) - c.quantile(0.25)))
            )
        ).any(axis=1)
        results["total_outlier_rows"] = int(mask.sum())
        results["outlier_pct"]        = round(mask.mean() * 100, 1)
    else:
        results["total_outlier_rows"] = 0
        results["outlier_pct"]        = 0.0
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
            results["top_pairs"] = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:5]
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
UNI_PDF  = "Uni_variate_output1.pdf"
BI_PDF   = "Bi_variate_output.pdf"

SUGGESTIONS = [
    "Show distribution of all columns",
    "Identify missing values",
    "Tell me about the column data types",
    "Which columns have the most outliers?",
    "Show correlation between columns",
]


# ── Main entry ────────────────────────────────────────────────────────────────

def show_home_page():
    _inject_css()
    _render_header()

    if "df" not in st.session_state:
        _render_pre_upload_showcase()
        st.markdown('<div class="section-title">📂 Upload Your Data</div>', unsafe_allow_html=True)
        _render_upload_widget()
    else:
        st.markdown('<div class="section-title">📂 Upload Your Data</div>', unsafe_allow_html=True)
        _render_upload_widget()
        # ── Instant data summary right after upload
        _render_data_summary(st.session_state["df"])
        st.markdown("---")
        _render_main_tabs()

    if st.session_state.pop("just_uploaded", False):
        import streamlit.components.v1 as components
        components.html("""<script>
        try {
            const ls = window.parent.localStorage;
            ls.setItem('argus_datasets', parseInt(ls.getItem('argus_datasets') || '5') + 1);
            ls.setItem('argus_charts', parseInt(ls.getItem('argus_charts') || '30') + 10);
        } catch(e) {}
        </script>""", height=0, width=0)
        
    if st.session_state.pop("just_asked_question", False):
        import streamlit.components.v1 as components
        components.html("""<script>
        try {
            const ls = window.parent.localStorage;
            ls.setItem('argus_questions', parseInt(ls.getItem('argus_questions') || '9') + 1);
        } catch(e) {}
        </script>""", height=0, width=0)


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
        with st.spinner(""):
            _show_processing_bar()
            df  = load_file(uploaded)
            eda = run_eda(df)
            st.session_state.update(
                df=df,
                eda_result=eda,
                file_name=uploaded.name,
                dataset_name=uploaded.name.rsplit(".", 1)[0],
                just_uploaded=True,
                chat_history=[{
                    "role": "assistant",
                    "content": (
                        f"✅ **EDA complete for `{uploaded.name}`!**  \n"
                        f"Found **{eda['rows']:,} rows** and **{eda['columns']} columns**. "
                        "Explore the tabs below or ask me anything!"
                    ),
                }],
            )
            # ── Pre-compute all Overview widget data once ──
            import hashlib
            df_json = df.to_json()
            df_hash = hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()
            st.session_state["w"] = compute_all_widgets(df_hash, df_json)
        st.rerun()


# ── Tabbed dashboard (post-upload) ────────────────────────────────────────────

def _render_main_tabs():
    df  = st.session_state["df"]
    eda = st.session_state["eda_result"]

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📊  Overview",
        "🧹  Data Cleaning",
        "📈  Univariate",
        "🔗  Bivariate",
        "🌟  Feature Importance",
        "💬  AI Chat",
    ])

    with t1: _render_overview_tab(df, eda)
    with t2: _render_cleaning_tab(df)
    with t3: _render_univariate_tab()
    with t4: _render_bivariate_tab()
    with t5: _render_feature_tab()
    with t6: _render_chat_tab(eda)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑  Clear — upload a new file", key="clear_btn"):
        keep = {"logged_in", "user_name", "user_email"}
        for k in [k for k in st.session_state if k not in keep]:
            del st.session_state[k]
        for pdf in [UNI_PDF, BI_PDF]:
            if os.path.exists(pdf):
                os.remove(pdf)
        st.rerun()


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
        score_color = "#1D9E75" if score >= 75 else "#EF9F27" if score >= 50 else "#E24B4A"
        dash = round(score / 100 * 176)

        def penalty_bar(label, value, max_val=40):
            pct = min(100, round(value / max_val * 100))
            bar_color = "#1D9E75" if value == 0 else "#EF9F27" if value < 20 else "#E24B4A"
            return (
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:7px">'
                f'<span style="font-size:14px;color:#9ca3af;min-width:100px">{label}</span>'
                f'<div style="flex:1;height:8px;border-radius:4px;background:#2d3748">'
                f'<div style="width:{pct}%;height:100%;border-radius:4px;background:{bar_color}"></div></div>'
                f'<span style="font-size:14px;color:{bar_color};min-width:30px;text-align:right">-{value}</span>'
                f'</div>'
            )

        st.markdown(dark_card(
            f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px">'
            f'<div>'
            f'<p style="font-size:18px;font-weight:600;color:#fff;margin:0">Dataset Health Score</p>'
            f'<p style="font-size:14px;color:#6b7280;margin:6px 0 0">auto-computed on upload</p>'
            f'</div>'
            f'<div style="position:relative;width:90px;height:90px">'
            f'<svg width="90" height="90" viewBox="0 0 72 72">'
            f'<circle cx="36" cy="36" r="28" fill="none" stroke="#2d3748" stroke-width="7"/>'
            f'<circle cx="36" cy="36" r="28" fill="none" stroke="{score_color}" stroke-width="7"'
            f' stroke-dasharray="{dash} 176" stroke-dashoffset="44" stroke-linecap="round"'
            f' transform="rotate(-90 36 36)"/>'
            f'</svg>'
            f'<div style="position:absolute;inset:0;display:flex;flex-direction:column;'
            f'align-items:center;justify-content:center;font-size:24px;font-weight:700;'
            f'color:{score_color}">{score}</div>'
            f'</div></div>'
            + penalty_bar("Missing",   w["missing_penalty"])
            + penalty_bar("Duplicates", w["dup_penalty"], 20)
            + penalty_bar("Outliers",  w["outlier_penalty"])
            + '<p style="font-size:13px;color:#6b7280;margin:12px 0 0;font-style:italic">'
              'Score = 100 minus missing, duplicate and outlier penalties</p>',
            "HEALTH", "#1D9E75"
        ), unsafe_allow_html=True)

        # ── WIDGET 3: Missing Value Heatmap ──────────────────────────────────
        missing_dict = w["missing_pct_per_col"]

        def miss_color(pct):
            if pct == 0:    return "#1D9E75"
            elif pct < 10:  return "#FAC775"
            elif pct < 50:  return "#EF9F27"
            else:           return "#E24B4A"

        squares = "".join([
            f'<div style="display:flex;flex-direction:column;align-items:center;gap:4px" title="{col} — {pct}% missing">'
            f'<div style="width:36px;height:36px;border-radius:6px;background:{miss_color(pct)}"></div>'
            f'<span style="font-size:12px;color:#6b7280">{col[:5]}</span>'
            f'</div>'
            for col, pct in missing_dict.items()
        ])

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
            '</div>'
        )

        st.markdown(dark_card(
            '<p style="font-size:18px;font-weight:600;color:#fff;margin:0 0 6px">Missing Value Map</p>'
            '<p style="font-size:14px;color:#6b7280;margin:0 0 16px">per-column severity at a glance</p>'
            + chart_html + legend,
            "QUALITY", "#E24B4A"
        ), unsafe_allow_html=True)

        # ── WIDGET 5: Outlier Summary ─────────────────────────────────────────
        od = w["outlier_details"]
        top5_outliers = list(od.items())[:5]

        def out_color(count, total):
            pct = count / max(total, 1) * 100
            return "#E24B4A" if pct > 20 else "#EF9F27" if pct > 10 else "#FAC775"

        rows_html = (
            "".join([
                f'<div style="display:flex;justify-content:space-between;font-size:14px;margin-bottom:8px">'
                f'<span style="color:#9ca3af">{col}</span>'
                f'<span style="color:{out_color(cnt, w["n_rows"])}"> {cnt:,} outliers</span>'
                f'</div>'
                for col, cnt in top5_outliers
            ])
            if top5_outliers
            else '<p style="font-size:15px;color:#1D9E75;margin:0">No outliers detected</p>'
        )

        st.markdown(dark_card(
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
            '</div>'
            + rows_html,
            "OUTLIERS", "#E24B4A"
        ), unsafe_allow_html=True)

    # ─────────────────────── RIGHT COLUMN ───────────────────────
    with col_b:

        # ── WIDGET 2: Column Breakdown ──────────────────────────────────────
        nn = w["n_numeric"]; nc2 = w["n_categorical"]; nd = w["n_datetime"]
        total = w["n_cols"]
        pn  = round(nn  / max(total, 1) * 100)
        pc2 = round(nc2 / max(total, 1) * 100)
        pd_ = round(nd  / max(total, 1) * 100)

        st.markdown(dark_card(
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
            '</div>'
            f'<div style="display:flex;height:12px;border-radius:6px;overflow:hidden;gap:3px">'
            f'<div style="flex:{max(nn,1)};background:#7F77DD;border-radius:6px 0 0 6px"></div>'
            f'<div style="flex:{max(nc2,1)};background:#1D9E75"></div>'
            f'<div style="flex:{max(nd,1)};background:#378ADD;border-radius:0 6px 6px 0"></div>'
            f'</div>'
            f'<div style="display:flex;gap:16px;margin-top:10px">'
            f'<span style="font-size:14px;color:#7F77DD">{pn}% numeric</span>'
            f'<span style="font-size:14px;color:#1D9E75">{pc2}% categorical</span>'
            f'<span style="font-size:14px;color:#378ADD">{pd_}% datetime</span>'
            f'</div>'
            '<div style="margin-top:16px;padding-top:16px;border-top:1px solid #2d3748">'
            f'<div style="display:flex;justify-content:space-between;font-size:15px;margin-bottom:8px">'
            f'<span style="color:#9ca3af">Total rows</span>'
            f'<span style="color:#fff;font-weight:500">{w["n_rows"]:,}</span></div>'
            f'<div style="display:flex;justify-content:space-between;font-size:15px">'
            f'<span style="color:#9ca3af">Total columns</span>'
            f'<span style="color:#fff;font-weight:500">{total}</span></div>'
            '</div>',
            "STRUCTURE", "#7F77DD"
        ), unsafe_allow_html=True)

        # ── WIDGET 4: Most Skewed Columns ────────────────────────────────────
        skew_dict = w.get("skew_scores", {})

        def skew_bar(col, val):
            width = min(int(abs(val) / 3 * 100), 100)
            bc = "#E24B4A" if abs(val) > 1 else "#EF9F27" if abs(val) > 0.5 else "#1D9E75"
            direction = "right" if val > 0 else "left" if val < 0 else "ok"
            return (
                f'<div style="margin-bottom:12px">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:4px">'
                f'<span style="font-size:14px;color:#9ca3af">{col}</span>'
                f'<div style="display:flex;gap:10px">'
                f'<span style="font-size:14px;color:{bc}">{val}</span>'
                f'<span style="font-size:13px;color:#6b7280;min-width:32px">{direction}</span>'
                f'</div></div>'
                f'<div style="height:8px;border-radius:4px;background:#2d3748">'
                f'<div style="width:{width}%;height:100%;border-radius:4px;background:{bc}"></div></div>'
                f'</div>'
            )

        skew_rows = (
            "".join([skew_bar(c, v) for c, v in skew_dict.items()])
            if skew_dict
            else '<p style="font-size:15px;color:#6b7280">No numeric columns found</p>'
        )

        st.markdown(dark_card(
            '<p style="font-size:18px;font-weight:600;color:#fff;margin:0 0 6px">Most Skewed Columns</p>'
            '<p style="font-size:14px;color:#6b7280;margin:0 0 16px">needs attention before modelling</p>'
            + skew_rows +
            '<p style="font-size:13px;color:#6b7280;margin:12px 0 0;font-style:italic">Skewness &gt;1 = consider log transform</p>',
            "DISTRIBUTION", "#EF9F27"
        ), unsafe_allow_html=True)

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
                f'</div>'
                f'<div style="height:8px;border-radius:4px;background:#2d3748">'
                f'<div style="width:{width}%;height:100%;border-radius:4px;background:{bc}"></div></div>'
                f'</div>'
            )

        corr_rows = (
            "".join([corr_bar(c1, c2, r) for c1, c2, r in pairs])
            if pairs
            else '<p style="font-size:15px;color:#6b7280">Need 2+ numeric columns</p>'
        )

        st.markdown(dark_card(
            '<p style="font-size:18px;font-weight:600;color:#fff;margin:0 0 6px">Strongest Correlations</p>'
            '<p style="font-size:14px;color:#6b7280;margin:0 0 16px">top pairs by absolute r value</p>'
            + corr_rows +
            '<p style="font-size:13px;color:#6b7280;margin:12px 0 0;font-style:italic">|r| &gt; 0.7 = strong &nbsp;|&nbsp; 0.4–0.7 = moderate</p>',
            "CORRELATION", "#378ADD"
        ), unsafe_allow_html=True)


# ── Tab 2 · Data Cleaning ─────────────────────────────────────────────────────

def _render_cleaning_tab(df: pd.DataFrame):
    st.markdown('<div class="section-title">🧹 Data Cleaning Pipeline</div>', unsafe_allow_html=True)

    if "df_cleaned" in st.session_state:
        df_c = st.session_state["df_cleaned"]
        st.success(f"✅ Cleaned dataset ready — **{df_c.shape[0]:,} rows × {df_c.shape[1]} columns**")
        
        # Download buttons
        import datetime
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"argus_cleaned_{st.session_state.get('dataset_name', 'data')}_{now}.csv"
        csv = df_c.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="⬇ Download Cleaned CSV",
            data=csv,
            file_name=filename,
            mime="text/csv",
            help="Download your cleaned dataset as CSV"
        )
        
        with st.expander("🗃 Cleaned Data Preview", expanded=False):
            st.dataframe(df_c.head(10).style.background_gradient(cmap="YlOrRd").format(precision=2),
                         use_container_width=True)
        return

    if not _HAS_CLEANING:
        st.warning("⚠️ Data cleaning module not available.")
        return

    if "cleaning_scan_report" not in st.session_state:
        st.markdown("""
        <div class="info-card">
            <b>Pipeline Steps:</b><br>
            ✅ Scan for missing values → show summary<br>
            ✅ Replace junk string values with NaN<br>
            ✅ Sanitize special characters<br>
            ✅ Call Groq AI for ambiguous columns<br>
            ✅ Apply imputation strategy per column<br>
            ✅ Re-scan to confirm 0 nulls
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Analyze & Scan Dataset for Cleaning", key="run_scan"):
            with st.spinner("Scanning for missing values and junk characters..."):
                from data_cleaning import SmartDataCleaner
                cleaner = SmartDataCleaner(df)
                report = cleaner.scan_missing_values()
                st.session_state["cleaning_scan_report"] = report
            st.rerun()
    else:
        st.info("🔍 **Pre-Clean Summary Report:**")
        st.dataframe(st.session_state["cleaning_scan_report"], use_container_width=True, hide_index=True)
        
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
    status_container.info("✅ Replace junk string values with NaN & Sanitize special characters...")
    time.sleep(0.6)
    cleaner.sanitize_data()
    progress_bar.progress(15)
    
    # Pipeline Step 2
    status_container.info("✅ Call Groq AI for ambiguous columns & Apply imputation strategy per column...")
    def prog_cb(current, total, col_name):
        pct = 15 + int(70 * (current / total))
        progress_bar.progress(pct)
        status_container.info(f"✅ Processing column '{col_name}'... ({current}/{total})")
        
    df_clean, warnings_list = cleaner.smart_impute(progress_callback=prog_cb)
    
    # Re-scan to confirm 0 nulls
    status_container.info("✅ Re-scan to confirm 0 nulls...")
    final_scan = cleaner.scan_missing_values()
    rem = final_scan["Missing Count"].sum()
    if rem == 0:
        status_container.success("🎉 All clear! 0 missing values remain.")
    else:
        status_container.warning(f"⚠️ Imputation finished, but {rem} missing values remain (likely ignored due to >80% rule).")
        
    for w in warnings_list:
        st.warning(w)
        
    progress_bar.progress(100)
    time.sleep(1.2)
    
    st.session_state["df_cleaned"] = df_clean
    st.session_state["just_uploaded"] = True # To trigger Dataset Analyzed increment on dashboard reload
    st.rerun()

# ── Tab 3 · Univariate Analysis ───────────────────────────────────────────────

def _render_univariate_tab():
    st.markdown('<div class="section-title">📈 Univariate Analysis</div>', unsafe_allow_html=True)

    df_work = st.session_state.get("df_cleaned", st.session_state.get("df"))
    dataset_name = st.session_state.get("dataset_name", "dataset")
    target_var   = st.session_state.get("target_variable")

    if os.path.exists(UNI_PDF):
        st.success("✅ Univariate analysis PDF ready — scroll through it below.")
        _show_pdf_scrollable(UNI_PDF, "📄 Univariate Analysis Report")
        if st.button("🔄 Re-run Univariate Analysis", key="rerun_uni"):
            os.remove(UNI_PDF)
            st.rerun()
        return

    if not _HAS_UNIVARIATE:
        st.warning("⚠️ Univariate analysis module not available.")
        return

    st.markdown("""
    <div class="info-card">
        AI-powered per-column analysis — histograms, count plots and natural-language
        descriptions generated by LLaMA 3.3. Output is rendered as a scrollable PDF below.
    </div>
    """, unsafe_allow_html=True)

    if not target_var:
        tv = st.text_input("Target variable (optional)", key="uni_target_input",
                           placeholder="e.g. SalePrice, Survived…")
        if tv:
            st.session_state["target_variable"] = tv
            target_var = tv

    if st.button("🚀 Run Univariate Analysis", key="run_uni"):
        with st.spinner("Running univariate analysis — this may take a moment…"):
            try:
                ctx_items = uni_analyze_and_visualize(df_work, dataset_name, target_var or "")
                # ── Store in LLM context
                ctx = st.session_state.setdefault("llm_context", [])
                if isinstance(ctx_items, list):
                    ctx.extend(ctx_items)
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
            col_summary = ", ".join(
                f"{c}({df[c].dtype})" for c in df.columns[:20]
            ) + (f" ...+{len(df.columns)-20} more" if len(df.columns) > 20 else "")

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
    num_cols    = df.select_dtypes(include="number").columns.tolist()
    cat_cols    = df.select_dtypes(exclude="number").columns.tolist()
    num_pct     = round(len(num_cols) / max(df.shape[1], 1) * 100, 1)
    cat_pct     = round(len(cat_cols) / max(df.shape[1], 1) * 100, 1)
    mem_kb      = round(df.memory_usage(deep=True).sum() / 1024, 1)
    missing_cols_count = int((df.isnull().sum() > 0).sum())

    # Build LLM context
    summary_ctx = (
        f"Dataset '{st.session_state.get('dataset_name', 'dataset')}' has "
        f"{df.shape[0]} rows and {df.shape[1]} columns. "
        f"{len(num_cols)} numeric columns: {', '.join(num_cols[:8])}{'...' if len(num_cols)>8 else ''}. "
        f"{len(cat_cols)} categorical columns: {', '.join(cat_cols[:8])}{'...' if len(cat_cols)>8 else ''}. "
        f"Missing values: {df.isnull().sum().sum()} ({missing_pct}%). Memory: {mem_kb} KB."
    )
    ctx = st.session_state.setdefault("llm_context", [])
    if not any("Dataset '" in c and "rows and" in c for c in ctx):
        ctx.append(summary_ctx)

    # ── Header card ──────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="summary-header-card">'
        f'<span class="sum-icon">📊</span> '
        f'<b>{st.session_state.get("file_name","Dataset")}</b>'
        f' &nbsp;&mdash;&nbsp; '
        f'<b>{df.shape[0]:,}</b> rows &times; <b>{df.shape[1]}</b> columns'
        f' &nbsp;&bull;&nbsp; <span class="sum-mem">{mem_kb} KB</span>'
        f'</div>',
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
            x=missing_s.values, y=missing_s.index,
            orientation="h", title="Top Missing-Value Columns",
            labels={"x": "Missing Count", "y": "Column"},
            color=missing_s.values, color_continuous_scale=["#F97316", "#EF4444"],
        )
        fig.update_layout(**_chart_layout(), height=max(200, len(missing_s)*36+80))
        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TASK 1 — DATA PREVIEW TABLE (first 15 rows, styled)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">📋 Data Preview — First 15 Rows</div>', unsafe_allow_html=True)

    preview_df = df.head(15)
    # Build header row with dtype badges
    dtype_badges = "".join(
        f'<th style="background:#1a2340;color:#F97316;font-weight:700;'
        f'position:sticky;top:0;z-index:2;padding:8px 12px;white-space:nowrap;border-bottom:2px solid #F97316;">'
        f'{col}<br><span style="font-size:0.68rem;font-weight:400;color:#94a3b8;'
        f'background:rgba(249,115,22,0.12);padding:1px 5px;border-radius:4px;">'
        f'{str(df[col].dtype)}</span></th>'
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
                "background:#78350f;color:#FCD34D;font-weight:600;" if is_null
                else f"color:#e2e8f0;"
            )
            display_val = "⚠ null" if is_null else str(val)
            cells += (
                f'<td style="{cell_style}padding:6px 12px;white-space:nowrap;'
                f'font-size:0.82rem;border-bottom:1px solid rgba(255,255,255,0.04);">'
                f'{display_val}</td>'
            )
        data_rows_html += f'<tr style="background:{bg};">{cells}</tr>'

    preview_table_html = f"""
    <div style="overflow-x:auto;max-height:400px;overflow-y:auto;
        border:1px solid rgba(249,115,22,0.3);border-radius:12px;
        background:#0d1526;margin-bottom:24px;">
      <table style="width:100%;border-collapse:collapse;font-family:monospace;">
        <thead><tr>{dtype_badges}</tr></thead>
        <tbody>{data_rows_html}</tbody>
      </table>
    </div>
    """
    st.markdown(preview_table_html, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # AI DATA INTELLIGENCE SUMMARY — 100% Python-native rendering
    # ══════════════════════════════════════════════════════════════════════════

    # Style block (CSS only, zero JS)
    st.markdown("""
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
    """, unsafe_allow_html=True)

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
            '</div>',
            unsafe_allow_html=True,
        )
        loading_slot = st.empty()
        loading_slot.info("⚡ Argus is analysing your dataset with Groq AI…")

        raw = _get_groq_ai_summary(df)

        loading_slot.empty()
        st.markdown('</div>', unsafe_allow_html=True)   # close temp box

        if raw.startswith("__ERROR__"):
            st.session_state["ai_summary_text"] = "__ERROR__"
            st.session_state["ai_summary_error"] = raw.replace("__ERROR__: ", "")
        else:
            # Strip any stray markdown symbols that LLM occasionally adds
            import re as _re
            clean = _re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', raw)  # **bold** → bold
            clean = _re.sub(r'#{1,6}\s+', '', clean)                # ## headings
            clean = _re.sub(r'^\s*[-*•]\s+', '', clean, flags=_re.M) # bullet symbols
            clean = clean.strip()
            st.session_state["ai_summary_text"] = clean
            st.session_state["ai_summary_typed"] = False   # trigger typewriter

        st.rerun()

    # ── Render the stored text ────────────────────────────────────────────────
    ai_text  = st.session_state.get("ai_summary_text", "")
    is_error = (ai_text == "__ERROR__")
    already_typed = st.session_state.get("ai_summary_typed", False)

    # ── Helper: build the complete box HTML ───────────────────────────────────
    PILLS_HTML = (
        f'<div class="ai-stat-pills">'
        f'<span class="ai-pill">📏 Rows: <b>{df.shape[0]:,}</b></span>'
        f'<span class="ai-pill">📊 Columns: <b>{df.shape[1]}</b></span>'
        f'<span class="ai-pill">⚠️ Missing: <b>{missing_cols_count}</b> cols</span>'
        f'<span class="ai-pill">🔢 Numeric: <b>{len(num_cols)}</b></span>'
        f'<span class="ai-pill">🔤 Categorical: <b>{len(cat_cols)}</b></span>'
        f'</div>'
    )

    HDR_HTML = (
        '<div class="ai-box-hdr">'
        '<span style="font-size:1.05rem;font-weight:700;color:#f1f5f9;">🤖 AI Data Intelligence Summary</span>'
        '<span class="ai-box-badge">✦ Powered by Groq AI</span>'
        '</div>'
    )

    CURSOR = (
        '<span style="display:inline-block;width:2px;height:0.95em;'
        'background:#00d4ff;vertical-align:middle;margin-left:2px;'
        'animation:blink-cur 0.6s step-start infinite;"></span>'
    )

    def _box(body_html: str, cls: str = "ai-box-wrap") -> str:
        return (
            '<style>'
            '@keyframes blink-cur{0%,100%{opacity:1}50%{opacity:0}}'
            '</style>'
            f'<div class="{cls}">'
            f'{HDR_HTML}'
            f'{body_html}'
            '</div>'
        )

    box_slot = st.empty()

    if is_error:
        box_slot.markdown(
            _box('<p style="color:#FCD34D;background:rgba(120,53,15,0.4);'
                 'border-radius:8px;padding:12px;">⚠️ Summary unavailable. '
                 'Please check your API connection.</p>', "ai-box-wrap done"),
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
                    "ai-box-wrap"          # spinning glow while typing
                ),
                unsafe_allow_html=True,
            )
            _time.sleep(0.07)   # 70 ms / word  →  medium speed

        # ── Phase 2: done — static glow + stat pills ──────────────────────────
        box_slot.markdown(
            _box(
                f'<div class="ai-typed-text">{ai_text}</div>' + PILLS_HTML,
                "ai-box-wrap done"          # static cyan glow when finished
            ),
            unsafe_allow_html=True,
        )
        st.session_state["ai_summary_typed"] = True

    else:
        # Already typed → instant static render
        box_slot.markdown(
            _box(
                f'<div class="ai-typed-text">{ai_text}</div>' + PILLS_HTML,
                "ai-box-wrap done"
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
            parsed = pd.to_datetime(df[col], errors="coerce",
                                    infer_datetime_format=True)
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
                    entry["sum"]    = _fmt_num(s.sum())
                    entry["mean"]   = _fmt_num(s.mean())
                    entry["max"]    = _fmt_num(s.max())
                    entry["min"]    = _fmt_num(s.min())
            except Exception:
                pass

        # Categorical/object stats
        else:
            # Try numeric coercion for object columns (often stored as string)
            try:
                s_num = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(s_num) / max(n, 1) > 0.6:
                    entry["sum"]       = _fmt_num(s_num.sum())
                    entry["max"]       = _fmt_num(s_num.max())
                    entry["min"]       = _fmt_num(s_num.min())
                    entry["is_numeric_stored_as_object"] = True
            except Exception:
                pass

            vc = df[col].value_counts()
            if len(vc) > 0:
                entry["top_value"]  = str(vc.index[0])
                entry["top_count"]  = int(vc.iloc[0])
                entry["top_pct"]    = round(vc.iloc[0] / n * 100, 1)

        profile[col] = entry

    return profile


def _fmt_num(v) -> str:
    """Format a number compactly for Groq context (keeps payload small)."""
    try:
        f = float(v)
        if abs(f) >= 1_000_000_000:
            return f"{f/1e9:.2f}B"
        if abs(f) >= 1_000_000:
            return f"{f/1e6:.2f}M"
        if abs(f) >= 1_000:
            return f"{f/1e3:.2f}K"
        return f"{f:,.2f}"
    except Exception:
        return str(v)


@st.cache_data(show_spinner=False)
def _ai_glance_cards(cache_key: str, profile_json: str, n_rows: int,
                     n_cols: int) -> list:
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
        {"label": "Total Rows",         "value": f"{n:,}",    "sub": f"{n_cols} columns"},
        {"label": "Numeric Columns",    "value": str(len(df.select_dtypes(include='number').columns)),
                                         "sub": "quantitative features"},
        {"label": "Categorical Columns","value": str(len(df.select_dtypes(include=['object','category']).columns)),
                                         "sub": "qualitative features"},
        {"label": "Missing Data",       "value": f"{miss}%",  "sub": "overall missing rate"},
    ]
    # Add top column by nunique
    top_cat = df.select_dtypes(include=["object","category"]).nunique().idxmax() \
        if len(df.select_dtypes(include=["object","category"]).columns) else None
    if top_cat:
        cards.append({"label": f"Unique {top_cat}",
                      "value": str(df[top_cat].nunique()),
                      "sub": "distinct values"})
    # Add largest numeric sum
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        sums = df[num_cols].sum()
        largest_col = sums.idxmax()
        v = sums[largest_col]
        cards.append({"label": largest_col.replace("_"," ").title(),
                      "value": _fmt_num(v),
                      "sub": "total across all rows"})
    return cards[:6]


def _render_smart_dashboard(df: pd.DataFrame):
    """
    CHANGE 1 — Enhanced 'Dataset At a Glance' with AI-generated metric cards
    rendered in a 3-column responsive HTML grid, with hover tooltips.
    """
    filename = st.session_state.get("file_name", "dataset")
    n_rows   = len(df)
    n_cols   = len(df.columns)
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
                col: df[col].value_counts().head(3).to_dict()
                for col in cat_cols[:8]
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
        num_count  = len(df.select_dtypes(include="number").columns)
        cat_count  = len(df.select_dtypes(include=["object","category"]).columns)
        fallback = [
            {"label": "Total Rows",          "value": f"{n_rows:,}",         "sublabel": "entire dataset",        "reason": "Total number of records in the dataset."},
            {"label": "Total Columns",        "value": str(n_cols),            "sublabel": "features",              "reason": "Total number of columns / features."},
            {"label": "Missing Values",       "value": f"{miss_total:,}",      "sublabel": "total null cells",      "reason": "Total count of null cells across the dataset."},
            {"label": "Duplicate Rows",       "value": str(dupe_count),        "sublabel": "exact duplicates",      "reason": "Exact duplicate row count that should be removed."},
            {"label": "Numeric Columns",      "value": str(num_count),         "sublabel": "quantitative features", "reason": "Count of columns with numeric values."},
            {"label": "Categorical Columns",  "value": str(cat_count),         "sublabel": "text features",         "reason": "Count of columns with text/categorical values."},
        ]
        return fallback

    # ── Generate cards ────────────────────────────────────────────────────────
    st.markdown("### 📊 Dataset At a Glance")
    with st.spinner("Generating dataset overview…"):
        cards = _get_cards()

    # ── Build 3-column HTML grid ──────────────────────────────────────────────
    card_html_parts = []
    for card in cards:
        label    = card.get("label", "")
        value    = card.get("value", "")
        sublabel = card.get("sublabel", card.get("sub", ""))
        reason   = card.get("reason", "")
        card_html_parts.append(
            f'<div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:12px;'
            f'padding:20px 24px;cursor:default;transition:border-color 0.2s;position:relative" '
            f'title="{reason}">'
            f'<p style="color:#9ca3af;font-size:13px;margin:0 0 8px;font-weight:400">{label}</p>'
            f'<p style="color:#ffffff;font-size:28px;font-weight:700;margin:0 0 6px;letter-spacing:-0.5px">{value}</p>'
            f'<p style="color:#6b7280;font-size:12px;margin:0">{sublabel}</p>'
            f'</div>'
        )

    full_html = (
        '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:24px">'
        + "".join(card_html_parts)
        + "</div>"
    )
    st.markdown(full_html, unsafe_allow_html=True)
    st.caption("Hover over any card for insight")



# ── Numeric Distribution ──────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _groq_numeric_description(cache_key: str, column: str, dataset_context: str,
                               mean: float, median: float, std: float,
                               _min: float, _max: float, q1: float, q3: float,
                               skewness: float, kurtosis: float,
                               missing_pct: float, negative_count: int,
                               outlier_pct: float) -> str:
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

    num_cols = df.select_dtypes(include=["int64", "float64",
                                         "int32", "float32"]).columns.tolist()
    if not num_cols:
        st.info("No numeric columns found.")
        return

    selected_col = st.selectbox("Select a numeric column", num_cols,
                                key="dist_col_select")
    if not selected_col:
        return

    col_data = df[selected_col].dropna()
    filename  = st.session_state.get("file_name", "dataset")

    # ── STEP 1: compute full stats ──────────────────────────────────────────
    mean_val   = round(float(col_data.mean()), 3)
    median_val = round(float(col_data.median()), 3)
    std_val    = round(float(col_data.std()), 3)
    min_val    = round(float(col_data.min()), 3)
    max_val    = round(float(col_data.max()), 3)
    q1_val     = round(float(col_data.quantile(0.25)), 3)
    q3_val     = round(float(col_data.quantile(0.75)), 3)
    iqr        = q3_val - q1_val
    outlier_mask = (col_data < q1_val - 1.5 * iqr) | (col_data > q3_val + 1.5 * iqr)
    outlier_count = int(outlier_mask.sum())
    outlier_pct   = round(outlier_count / max(len(col_data), 1) * 100, 1)

    if scipy_stats is not None:
        skewness_val = round(float(scipy_stats.skew(col_data)), 3)
        kurtosis_val = round(float(scipy_stats.kurtosis(col_data)), 3)
    else:
        skewness_val = round(float(col_data.skew()), 3)
        kurtosis_val = round(float(col_data.kurtosis()), 3)

    missing_count = int(df[selected_col].isnull().sum())
    missing_pct   = round(df[selected_col].isnull().mean() * 100, 1)
    negative_count = int((col_data < 0).sum())
    unique_count  = int(col_data.nunique())
    dataset_context = f"{filename}, all columns: {list(df.columns)}"

    # ── STEP 2: 6 stat cards ────────────────────────────────────────────────
    def _fmt(v):
        if isinstance(v, float):
            return f"{v:,.2f}"
        return f"{v:,}"

    stat_cards = [
        ("Mean",       _fmt(mean_val),    "#F97316"),
        ("Median",     _fmt(median_val),  "#38BDF8"),
        ("Std Dev",    _fmt(std_val),     "#A78BFA"),
        ("Min",        _fmt(min_val),     "#34D399"),
        ("Max",        _fmt(max_val),     "#FCD34D"),
        ("Outliers%",  f"{outlier_pct}%", "#EF4444"),
    ]
    cols6 = st.columns(6)
    for i, (label, value, color) in enumerate(stat_cards):
        with cols6[i]:
            st.markdown(
                f'<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);'
                f'border-radius:12px;padding:14px 10px;text-align:center;animation:fadeUp 0.5s ease both;">'
                f'<div style="font-size:0.72rem;color:#94A3B8;margin-bottom:4px;">{label}</div>'
                f'<div style="font-size:1.35rem;font-weight:800;color:{color};">{value}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── STEP 3: histogram (no box plot) ─────────────────────────────────────
    fig = px.histogram(
        df, x=selected_col, nbins=30,
        marginal="rug",
        labels={selected_col: selected_col, "count": "Frequency"},
    )
    fig.update_traces(marker_color="#7F77DD", marker_line_width=0)
    fig.add_vline(
        x=mean_val, line_dash="dash", line_color="#E24B4A",
        annotation_text=f"Mean: {mean_val}",
        annotation_position="top right",
    )
    fig.add_vline(
        x=median_val, line_dash="dash", line_color="#1D9E75",
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
    st.plotly_chart(fig, use_container_width=True)

    # ── STEP 4: Groq AI description (cached) ────────────────────────────────
    cache_key = f"{filename}_{selected_col}_{mean_val}_{std_val}"
    with st.spinner("Argus AI is analysing this column…"):
        ai_desc = _groq_numeric_description(
            cache_key, selected_col, dataset_context,
            mean_val, median_val, std_val, min_val, max_val,
            q1_val, q3_val, skewness_val, kurtosis_val,
            missing_pct, negative_count, outlier_pct,
        )

    if ai_desc is None:
        # rule-based fallback
        sk = skewness_val
        if sk > 1.0:    shape = "strongly right-skewed"
        elif sk > 0.5:  shape = "moderately right-skewed"
        elif sk < -1.0: shape = "strongly left-skewed"
        elif sk < -0.5: shape = "moderately left-skewed"
        else:           shape = "roughly symmetric"
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
        f'border-radius:0 8px 8px 0;background:rgba(127,119,221,0.08);'
        f'font-size:14px;line-height:1.7;color:#ccc;margin-top:12px">'
        f'{ai_desc}</div>',
        unsafe_allow_html=True,
    )


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
        return resp.choices[0].message.content.strip().strip('"').rstrip('.')
    except Exception:
        return f"Distribution of {col}"


@st.cache_data(show_spinner=False)
def _groq_cat_description(cache_key: str, col: str, dataset_context: str,
                           cardinality: int, top_value: str, top_pct: float,
                           top_10_str: str, missing_pct: float) -> str:
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
    HINTS = ["year", "month", "grade", "rating", "category", "type", "status",
             "class", "group", "level", "rank", "code", "flag"]
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
        '\U0001f3f7 Smart Categorical Insights</div>',
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
            '\U0001f446 Pick a column above to see its distribution chart and AI insight.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    col = chosen

    # ── 5. Compute stats ──────────────────────────────────────────────────────
    vc          = df[col].value_counts()
    top10       = vc.head(10)
    cardinality = int(df[col].nunique())
    top_pct     = round(float(top10.iloc[0]) / len(df) * 100, 1)
    top_value   = str(top10.index[0])
    missing_pct = round(df[col].isnull().mean() * 100, 1)
    top_10_dict = {str(k): int(v) for k, v in top10.items()}
    top_10_str  = str(top_10_dict)
    context     = f"{filename}, columns: {list(df.columns)}"
    cache_key   = f"{filename}_{col}_{top_10_str}"

    # Mini stat pills
    st.markdown(
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin:10px 0 16px;">'
        f'<span style="background:rgba(249,115,22,0.1);border:1px solid rgba(249,115,22,0.3);'
        f'border-radius:20px;padding:4px 14px;font-size:0.78rem;color:#e2e8f0;">'
        f'🔢 Unique values: <b>{cardinality}</b></span>'
        f'<span style="background:rgba(56,189,248,0.1);border:1px solid rgba(56,189,248,0.3);'
        f'border-radius:20px;padding:4px 14px;font-size:0.78rem;color:#e2e8f0;">'
        f'🥇 Top value: <b>{top_value}</b> ({top_pct}%)</span>'
        f'<span style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);'
        f'border-radius:20px;padding:4px 14px;font-size:0.78rem;color:#e2e8f0;">'
        f'⚠️ Missing: <b>{missing_pct}%</b></span>'
        f'</div>',
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

    top10_df = pd.DataFrame({
        "value": plot_data.index.astype(str),
        "count": plot_data.values,
    })

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
        if "ai_cache" not in st.session_state: st.session_state["ai_cache"] = {}
        st.session_state["ai_cache"][ax_cache_key] = json.dumps({"y": y_label, "x": x_label})
    else:
        try:
            ldata = json.loads(st.session_state["ai_cache"][ax_cache_key])
            y_label = ldata.get("y", col.replace("_", " ").title())
            x_label = ldata.get("x", "Count")
        except Exception:
            y_label = col.replace("_", " ").title()
            x_label = "Count"

    fig = px.bar(
        top10_df, x="count", y="value", orientation="h",
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
    st.plotly_chart(fig, use_container_width=True)

    # ── 8. Groq description ───────────────────────────────────────────────────
    with st.spinner("Generating AI insight…"):
        description = _groq_cat_description(
            cache_key, col, context, cardinality,
            top_value, top_pct, top_10_str, missing_pct,
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
        f'border-radius:0 8px 8px 0;background:rgba(127,119,221,0.08);'
        f'font-size:14px;line-height:1.7;color:#ccc;margin-top:8px">'
        f'{description}</div>',
        unsafe_allow_html=True,
    )


# ── Tab 4 · Bivariate Analysis ────────────────────────────────────────────────

def _render_bivariate_tab():
    st.markdown('<div class="section-title">🔗 Bivariate Analysis</div>', unsafe_allow_html=True)

    df_work = st.session_state.get("df_cleaned", st.session_state.get("df"))
    target_var = st.session_state.get("target_variable", "")

    if os.path.exists(BI_PDF):
        st.success("✅ Bivariate analysis PDF ready — scroll through it below.")
        _show_pdf_scrollable(BI_PDF, "📄 Bivariate Analysis Report")
        if st.button("🔄 Re-run Bivariate Analysis", key="rerun_bi"):
            os.remove(BI_PDF)
            st.rerun()
        return

    st.markdown("""
    <div class="info-card">
        Correlation-based pair selection — the target variable is always paired with the
        most relevant features. Plots include <b>scatter with trendline</b> and
        <b>mean-bar charts</b>. Each chart includes an AI-generated plain-English insight.
    </div>
    """, unsafe_allow_html=True)

    # Ensure target variable is set
    if not target_var:
        tv = st.text_input(
            "🎯 Target variable",
            key="bi_target_input",
            placeholder="e.g. SalePrice, Survived, Churn…",
            help="The column you want to predict or analyse. Leave blank to auto-select top correlated pairs.",
        )
        if tv:
            st.session_state["target_variable"] = tv
            target_var = tv
    else:
        st.info(f"🎯 Target variable: **{target_var}**")

    if st.button("🚀 Run Bivariate Analysis", key="run_bi"):
        _run_bivariate(df_work)


def _run_bivariate(df: pd.DataFrame):
    with st.spinner("Running bivariate analysis — selecting pairs by correlation…"):
        try:
            from bivariate_analysis import bi_visualize_analyze
            dataset_name    = st.session_state.get("dataset_name", "dataset")
            target_variable = st.session_state.get("target_variable", "")

            prog = st.progress(0, "Selecting column pairs…")
            context_items = bi_visualize_analyze(df, dataset_name, target_variable)
            prog.progress(100, "Done! ✅")

            # ── Store in LLM context
            ctx = st.session_state.setdefault("llm_context", [])
            ctx.extend(context_items)

            import time as _t; _t.sleep(0.3)
            prog.empty()
            st.rerun()
        except Exception as e:
            st.error(f"Bivariate analysis error: {e}")


# ── Tab 5 · Feature Importance ────────────────────────────────────────────────

def _render_feature_tab():
    st.markdown('<div class="section-title">🌟 Feature Importance</div>', unsafe_allow_html=True)

    df_work = st.session_state.get("df_cleaned", st.session_state.get("df"))
    target  = st.session_state.get("target_variable")
    num_cols = df_work.select_dtypes(include="number").columns.tolist()

    if "feature_importance_fig" in st.session_state:
        st.plotly_chart(st.session_state["feature_importance_fig"], use_container_width=True)
        if st.button("🔄 Recompute", key="recompute_feat"):
            del st.session_state["feature_importance_fig"]
            st.rerun()
        return

    if not target or target not in num_cols:
        target = st.selectbox("Select target variable", num_cols, key="feat_target_select")
        if target:
            st.session_state["target_variable"] = target

    if target and st.button("🚀 Compute Feature Importance", key="run_feat"):
        _run_feature_importance(df_work, target)


def _run_feature_importance(df: pd.DataFrame, target: str):
    with st.spinner("Computing feature importance…"):
        try:
            from sklearn.ensemble import RandomForestRegressor

            num_cols     = df.select_dtypes(include="number").columns.tolist()
            feature_cols = [c for c in num_cols if c != target]

            if not feature_cols:
                st.warning("No numeric feature columns available.")
                return

            X = df[feature_cols].fillna(0)
            y = df[target].fillna(0)

            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X, y)

            imp_df = pd.DataFrame({
                "Feature":    feature_cols,
                "Importance": model.feature_importances_,
            }).sort_values("Importance", ascending=False)

            fig = px.bar(
                imp_df, x="Feature", y="Importance",
                title=f"Feature Importance — Target: {target}",
                color="Importance",
                color_continuous_scale=["#38BDF8", "#F97316"],
            )
            fig.update_layout(**_chart_layout(), height=460)
            st.session_state["feature_importance_fig"] = fig
            st.rerun()
        except Exception as e:
            st.error(f"Feature importance error: {e}")


# ── Tab 6 · AI Chat ───────────────────────────────────────────────────────────

def _render_chat_tab(eda: dict):
    st.markdown('<div class="section-title">💬 EDA Insights Chat</div>', unsafe_allow_html=True)
    st.markdown('<div class="chip-label">Quick questions:</div>', unsafe_allow_html=True)

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
            f'</div>',
            unsafe_allow_html=True,
        )

    with st.form("chat_form", clear_on_submit=True):
        ci, cb = st.columns([5, 1])
        with ci:
            q = st.text_input("", placeholder="Ask a question about your data…",
                              label_visibility="collapsed")
        with cb:
            sent = st.form_submit_button("➤", use_container_width=True)
    if sent and q.strip():
        _handle_chat(q.strip(), eda)


def _handle_chat(q: str, eda: dict):
    hist = st.session_state.setdefault("chat_history", [])
    hist.append({"role": "user", "content": q})
    llm_ctx = st.session_state.get("llm_context", [])
    with st.spinner("Argus AI is thinking…"):
        ans = chat_response(q, eda, llm_context=llm_ctx)
    hist.append({"role": "assistant", "content": ans})
    st.session_state["just_asked_question"] = True
    st.rerun()


# ── PDF / image scrollable viewer ─────────────────────────────────────────────

def _show_pdf_scrollable(pdf_path: str, title: str = "📄 Output"):
    """Render a PDF inline: converts pages to images (PyMuPDF) with iframe fallback."""
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
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
            f'background:rgba(0,0,0,0.25);border:1px solid rgba(249,115,22,0.28);'
            f'border-radius:16px;">{imgs_html}</div>',
            unsafe_allow_html=True,
        )
    except ImportError:
        with open(pdf_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{b64}" '
            f'width="100%" height="700px" '
            f'style="border:1px solid rgba(249,115,22,0.28);border-radius:16px;'
            f'background:#0A0F1E;"></iframe>',
            unsafe_allow_html=True,
        )

    with open(pdf_path, "rb") as f:
        st.download_button(
            f"⬇️ Download {os.path.basename(pdf_path)}",
            f.read(),
            file_name=os.path.basename(pdf_path),
            mime="application/pdf",
            key=f"dl_{os.path.basename(pdf_path)}",
        )


# ── Pre-upload showcase ───────────────────────────────────────────────────────

def _render_pre_upload_showcase():
    st.markdown("""
    <div class="showcase-grid">
        <div class="showcase-card" style="--accent:#F97316">
            <div class="sc-icon">⚡</div>
            <div class="sc-title">Instant EDA</div>
            <div class="sc-desc">10+ charts auto-generated in under 3 seconds</div>
            <div class="sc-bar"><div class="sc-fill" style="width:92%;background:#F97316"></div></div>
        </div>
        <div class="showcase-card" style="--accent:#38BDF8">
            <div class="sc-icon">🧠</div>
            <div class="sc-title">AI-Powered Chat</div>
            <div class="sc-desc">Ask natural language questions about your dataset</div>
            <div class="sc-bar"><div class="sc-fill" style="width:85%;background:#38BDF8"></div></div>
        </div>
        <div class="showcase-card" style="--accent:#A78BFA">
            <div class="sc-icon">📊</div>
            <div class="sc-title">Smart Visuals</div>
            <div class="sc-desc">Interactive histograms, heatmaps, scatter plots</div>
            <div class="sc-bar"><div class="sc-fill" style="width:78%;background:#A78BFA"></div></div>
        </div>
        <div class="showcase-card" style="--accent:#34D399">
            <div class="sc-icon">🔍</div>
            <div class="sc-title">Anomaly Detection</div>
            <div class="sc-desc">Outliers and missing values flagged automatically</div>
            <div class="sc-bar"><div class="sc-fill" style="width:88%;background:#34D399"></div></div>
        </div>
    </div>
    <div class="live-widgets-row">
        <div class="live-widget lw-counter">
            <div class="lw-badge">&#9679; LIVE</div>
            <div class="lw-main-num" id="lw-count">0</div>
            <div class="lw-label">Datasets Analysed Today</div>
            <div class="lw-sparkline">
                <div class="spark-bar" style="height:30%;background:#F97316"></div>
                <div class="spark-bar" style="height:55%;background:#F97316"></div>
                <div class="spark-bar" style="height:42%;background:#F97316"></div>
                <div class="spark-bar" style="height:70%;background:#F97316"></div>
                <div class="spark-bar" style="height:50%;background:#F97316"></div>
                <div class="spark-bar" style="height:88%;background:#FA8C3A"></div>
                <div class="spark-bar" style="height:95%;background:#FA8C3A" id="spark-last"></div>
            </div>
        </div>
        <div class="live-widget lw-donut">
            <div class="lw-badge" style="background:rgba(56,189,248,0.15);color:#38BDF8;border-color:rgba(56,189,248,0.3)">&#9679; INSIGHT</div>
            <div class="donut-wrap">
                <svg viewBox="0 0 80 80" class="donut-svg">
                    <circle cx="40" cy="40" r="32" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="10"/>
                    <circle cx="40" cy="40" r="32" fill="none" stroke="#38BDF8" stroke-width="10"
                        stroke-dasharray="201 201" stroke-dashoffset="30"
                        stroke-linecap="round" transform="rotate(-90 40 40)" class="donut-arc"/>
                </svg>
                <div class="donut-label">85%<br><span>AI Accuracy</span></div>
            </div>
        </div>
        <div class="live-widget lw-feed">
            <div class="lw-badge" style="background:rgba(167,139,250,0.15);color:#A78BFA;border-color:rgba(167,139,250,0.3)">&#9679; ACTIVITY</div>
            <div class="feed-title">Recent Actions</div>
            <ul class="feed-list" id="feed-list">
                <li class="feed-item"><span class="feed-dot" style="background:#F97316"></span>Correlation heatmap generated</li>
                <li class="feed-item"><span class="feed-dot" style="background:#38BDF8"></span>Outlier detection complete</li>
                <li class="feed-item"><span class="feed-dot" style="background:#A78BFA"></span>AI chat ready</li>
                <li class="feed-item"><span class="feed-dot" style="background:#34D399"></span>Missing value scan done</li>
            </ul>
        </div>
        <div class="live-widget lw-gauge">
            <div class="lw-badge" style="background:rgba(52,211,153,0.15);color:#34D399;border-color:rgba(52,211,153,0.3)">&#9679; PERF</div>
            <div class="gauge-wrap">
                <div class="gauge-ring">
                    <div class="gauge-inner">
                        <span class="gauge-val">2.1s</span>
                        <span class="gauge-sub">Avg. EDA Time</span>
                    </div>
                </div>
            </div>
            <div class="gauge-bars">
                <div class="gb" style="--h:40%;--c:#34D399"></div>
                <div class="gb" style="--h:65%;--c:#34D399"></div>
                <div class="gb" style="--h:50%;--c:#34D399"></div>
                <div class="gb" style="--h:80%;--c:#2DD4C0"></div>
                <div class="gb" style="--h:35%;--c:#2DD4C0"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    import streamlit.components.v1 as components
    components.html("""
    <script>
    (function(){
        var doc = window.parent.document;
        var target = 1247;
        var el = doc.getElementById('lw-count');
        if(el){
            var cur = 0, step = Math.ceil(target/80);
            var t = setInterval(function(){
                cur = Math.min(cur+step, target);
                el.textContent = cur.toLocaleString();
                if(cur>=target) clearInterval(t);
            }, 20);
            setInterval(function(){
                var s = doc.getElementById('spark-last');
                if(s){ s.style.height = Math.floor(80+Math.random()*18)+'%'; }
            }, 1200);
        }
        var feedItems = [
            {dot:'#F97316',text:'Correlation heatmap generated'},
            {dot:'#38BDF8',text:'Outlier detection complete'},
            {dot:'#A78BFA',text:'AI chat session started'},
            {dot:'#34D399',text:'Missing value scan done'},
            {dot:'#F97316',text:'Distribution chart rendered'},
            {dot:'#38BDF8',text:'Data types classified'},
            {dot:'#A78BFA',text:'Summary statistics ready'}
        ];
        var feedIdx = 4;
        setInterval(function(){
            var fl = doc.getElementById('feed-list');
            if(!fl) return;
            var li = doc.createElement('li');
            li.className='feed-item feed-new';
            var f = feedItems[feedIdx%feedItems.length];
            li.innerHTML='<span class="feed-dot" style="background:'+f.dot+'"></span>'+f.text;
            fl.insertBefore(li,fl.firstChild);
            if(fl.children.length>4) fl.removeChild(fl.lastChild);
            feedIdx++;
        }, 2500);
    })();
    </script>
    """, height=0, width=0)
    st.markdown("""
    <div class="upload-placeholder">
        <div class="upload-icon-anim">&#9729;&#65039;</div>
        <div class="upload-text">
            Drag &amp; drop your <b>.xlsx</b> or <b>.csv</b> file above,<br>
            or click <b>Browse files</b> to get started.
        </div>
        <div class="upload-hint">An interactive EDA dashboard + AI chat will appear after upload.</div>
    </div>
    """, unsafe_allow_html=True)


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
            '<span class="app-sub">Automated EDA Tool</span></h1>'
            '<div class="typing-wrapper">'
            '<span class="typing-cursor" id="typing-el"></span>'
            '<span class="cursor-bar">|</span>'
            '</div>',
            unsafe_allow_html=True,
        )
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
        """, height=0, width=0)

    with cu:
        name  = st.session_state.get("user_name", "User")
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
            f'</div>',
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
    for pct, msg in [(20,"Parsing columns…"),(45,"Computing statistics…"),
                     (70,"Detecting types…"),(90,"Generating insights…"),(100,"Done! ✅")]:
        time.sleep(0.35)
        bar.progress(pct, text=msg)
    time.sleep(0.3)
    bar.empty()


# ── CSS ───────────────────────────────────────────────────────────────────────

def _inject_css():
    st.markdown("""
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
        border-bottom:1px solid rgba(249,115,22,0.2) !important;
    }
    [data-testid="stTab"] { color:#64748B !important; font-weight:600; font-size:0.88rem; }
    [data-testid="stTab"][aria-selected="true"] {
        color:#F97316 !important;
        border-bottom:2px solid #F97316 !important;
    }
    /* ── Showcase grid */
    .showcase-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin:0 0 24px 0; }
    @media(max-width:900px){ .showcase-grid{grid-template-columns:1fr 1fr;} }
    .showcase-card {
        background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
        border-radius:18px; padding:22px 18px;
        transition:all 0.35s; position:relative; overflow:hidden;
    }
    .showcase-card:hover { border-color:var(--accent); transform:translateY(-5px);
                           box-shadow:0 10px 32px rgba(0,0,0,0.4); }
    .showcase-card::after {
        content:''; position:absolute; bottom:0; left:0; right:0; height:2px;
        background:var(--accent); transform:scaleX(0); transform-origin:left; transition:transform 0.35s;
    }
    .showcase-card:hover::after { transform:scaleX(1); }
    .sc-icon  { font-size:1.9rem; margin-bottom:8px; }
    .sc-title { font-weight:700; color:#E2E8F0; font-size:0.93rem; margin-bottom:4px; }
    .sc-desc  { font-size:0.75rem; color:#64748B; line-height:1.5; margin-bottom:12px; }
    .sc-bar   { height:5px; background:rgba(255,255,255,0.07); border-radius:10px; }
    .sc-fill  { height:5px; border-radius:10px; animation:fillBar 1.4s cubic-bezier(.4,0,.2,1) forwards; }
    @keyframes fillBar { from{width:0!important} }
    /* ── Live Widgets */
    .live-widgets-row { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin:0 0 28px 0; }
    @media(max-width:900px){ .live-widgets-row{grid-template-columns:1fr 1fr;} }
    .live-widget {
        background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
        border-radius:18px; padding:18px 16px; position:relative; overflow:hidden;
        animation:fadeUp 0.6s ease both;
    }
    .live-widget:hover { border-color:rgba(249,115,22,0.3); box-shadow:0 8px 28px rgba(0,0,0,0.35);
                         transform:translateY(-3px); transition:all 0.3s; }
    .lw-badge {
        display:inline-flex; align-items:center; gap:5px; font-size:0.62rem; font-weight:700;
        letter-spacing:0.08em; color:#F97316; background:rgba(249,115,22,0.12);
        border:1px solid rgba(249,115,22,0.3); border-radius:20px; padding:2px 8px; margin-bottom:10px;
        animation:badgePulse 2s ease-in-out infinite;
    }
    @keyframes badgePulse { 0%,100%{opacity:1} 50%{opacity:0.6} }
    .lw-main-num { font-size:2.4rem; font-weight:800; color:#F97316; line-height:1; margin:4px 0 2px; }
    .lw-label    { font-size:0.73rem; color:#64748B; margin-bottom:14px; }
    .lw-sparkline{ display:flex; align-items:flex-end; gap:4px; height:36px; }
    .spark-bar   { flex:1; border-radius:3px 3px 0 0; transition:height 0.6s ease; }
    .donut-wrap  { position:relative; width:110px; height:110px; margin:6px auto 0; }
    .donut-svg   { width:100%; height:100%; }
    .donut-arc   { animation:spinArc 1.6s ease forwards; }
    @keyframes spinArc { from{stroke-dashoffset:201} to{stroke-dashoffset:30} }
    .donut-label { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
                   text-align:center; font-weight:800; font-size:1.1rem; color:#38BDF8; line-height:1.2; }
    .donut-label span { font-size:0.62rem; color:#64748B; font-weight:500; }
    .feed-title { font-size:0.75rem; color:#94A3B8; font-weight:600; margin-bottom:10px; }
    .feed-list  { list-style:none; margin:0; padding:0; }
    .feed-item  { display:flex; align-items:center; gap:8px; font-size:0.72rem; color:#94A3B8;
                  padding:5px 0; border-bottom:1px solid rgba(255,255,255,0.05);
                  animation:feedSlide 0.4s ease; }
    .feed-new   { animation:feedSlide 0.4s ease; }
    @keyframes feedSlide { from{opacity:0;transform:translateX(-10px)} to{opacity:1;transform:none} }
    .feed-dot   { width:7px; height:7px; border-radius:50%; flex-shrink:0; }
    .gauge-wrap { display:flex; justify-content:center; margin:4px 0 12px; }
    .gauge-ring {
        width:86px; height:86px; border-radius:50%;
        background:conic-gradient(#34D399 0% 72%,rgba(255,255,255,0.06) 72% 100%);
        display:flex; align-items:center; justify-content:center;
        box-shadow:0 0 16px rgba(52,211,153,0.25); animation:gaugeReveal 1.8s ease forwards;
    }
    @keyframes gaugeReveal {
        from{background:conic-gradient(#34D399 0% 0%,rgba(255,255,255,0.06) 0% 100%)}
        to  {background:conic-gradient(#34D399 0% 72%,rgba(255,255,255,0.06) 72% 100%)}
    }
    .gauge-inner { width:62px; height:62px; border-radius:50%; background:#0A0F1E;
                   display:flex; flex-direction:column; align-items:center; justify-content:center; }
    .gauge-val   { font-size:1rem; font-weight:800; color:#34D399; }
    .gauge-sub   { font-size:0.52rem; color:#475569; text-align:center; }
    .gauge-bars  { display:flex; align-items:flex-end; gap:4px; height:28px; margin-top:8px; }
    .gb { flex:1; height:var(--h); background:var(--c); border-radius:3px 3px 0 0; opacity:0.7;
          animation:gbAnim 1.2s ease both; }
    @keyframes gbAnim { from{height:0} }
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
    /* ── Buttons */
    [data-testid="stButton"] button,
    [data-testid="stFormSubmitButton"] > button {
        background:rgba(249,115,22,0.12) !important;
        border:1px solid rgba(249,115,22,0.35) !important;
        color:#FB923C !important; border-radius:20px !important;
        font-size:0.78rem !important; transition:all 0.2s !important;
        white-space:normal !important; height:auto !important;
    }
    [data-testid="stButton"] button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        background:rgba(249,115,22,0.25) !important;
        border-color:#F97316 !important; color:#fff !important; transform:translateY(-1px) !important;
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
    </style>
    """, unsafe_allow_html=True)

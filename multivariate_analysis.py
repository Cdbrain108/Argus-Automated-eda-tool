"""
multivariate_analysis.py — AI-powered multivariate analysis for Argus EDA.

Designed for NON-TECHNICAL users. Every chart has:
  - A plain-English title (no stats jargon)
  - Emoji labels and colour-coded zones
  - A "What does this mean?" AI insight box beneath it
  - Axis labels written as questions or plain descriptions

5 beginner-friendly pages
──────────────────────────
  1. "Which things move together?"          → Relationship strength grid
  2. "What drives [target] the most?"       → Top influencers bar chart
  3. "Are there natural groups?"            → Customer / row groups visual
  4. "How does each group behave?"          → Group behaviour profile cards
  5. "How are all columns connected?"       → Bubble connection map
"""

import io
import json
import textwrap
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from utils import get_groq_client

warnings.filterwarnings("ignore")

# ── Palette (matches univariate / bivariate) ──────────────────────────────────
BG_COLOR    = "#0A0F1E"
TEXT_COLOR  = "#E2E8F0"
BAR_COLOR   = "#F97316"   # orange
LINE_COLOR  = "#38BDF8"   # sky blue
SCATTER_COL = "#A78BFA"   # purple
PANEL_COLOR = "#1E2D40"
SPINE_COLOR = "#374151"
AXES_FACE   = "#111827"
GREEN       = "#22C55E"
RED         = "#EF4444"

MAX_COLS        = 12
MAX_ROWS_SAMPLE = 5_000


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _sample(df):
    if len(df) > MAX_ROWS_SAMPLE:
        return df.sample(MAX_ROWS_SAMPLE, random_state=42).reset_index(drop=True)
    return df.copy()


def _numeric_cols(df, min_unique=5):
    return [c for c in df.select_dtypes(include="number").columns
            if df[c].nunique(dropna=True) >= min_unique]


def _cat_cols(df, max_unique=20):
    return [c for c in df.select_dtypes(exclude="number").columns
            if df[c].nunique(dropna=True) <= max_unique]


def _style_ax(ax):
    ax.set_facecolor(AXES_FACE)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_COLOR)


def _insight_box(fig, text, rect=(0.04, 0.03, 0.92, 0.18)):
    """Render the AI insight panel — identical look to bivariate."""
    ax = fig.add_axes(rect)
    ax.set_facecolor(PANEL_COLOR)
    for sp in ax.spines.values():
        sp.set_edgecolor(BAR_COLOR)
        sp.set_linewidth(0.9)
    wrapped = "\n".join(textwrap.wrap(text, width=115))
    ax.text(0.5, 0.58, wrapped, ha="center", va="center",
            fontsize=8, color=TEXT_COLOR, linespacing=1.45,
            transform=ax.transAxes)
    ax.text(0.5, 0.06, "💡 What does this mean?",
            ha="center", va="bottom", fontsize=7.5,
            color=BAR_COLOR, fontweight="bold",
            transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("", labelpad=0)


def _ask_ai(prompt, max_tokens=130):
    """Call Groq LLM. Always ask for plain English — no jargon."""
    try:
        client = get_groq_client()
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(AI insight unavailable: {e})"


def _clean(df, cols):
    return df[cols].dropna()


# ═══════════════════════════════════════════════════════════════════════════════
# AI column selection (internal — user never sees this)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_profiles(df):
    profiles = {}
    n = len(df)
    for col in df.columns:
        entry = {
            "dtype": str(df[col].dtype),
            "nunique": int(df[col].nunique(dropna=True)),
            "missing_pct": round(df[col].isnull().mean() * 100, 1),
            "samples": [str(x) for x in df[col].dropna().head(3)],
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].dropna()
            if len(s):
                entry.update({"mean": round(float(s.mean()), 2),
                               "std":  round(float(s.std()),  2),
                               "min":  round(float(s.min()),  2),
                               "max":  round(float(s.max()),  2)})
        else:
            vc = df[col].value_counts()
            if len(vc):
                entry["top_value"] = str(vc.index[0])
                entry["unique_values"] = [str(v) for v in vc.head(5).index]
        profiles[col] = entry
    return profiles


def _ai_select_columns(df, profiles, target, dataset_name):
    num_cols = _numeric_cols(df)
    cat_cols = _cat_cols(df)
    prompt = (
        f"You are selecting columns for multivariate analysis on '{dataset_name}'.\n"
        f"Column profiles: {json.dumps(profiles, indent=2)}\n"
        f"Numeric: {num_cols}  Categorical: {cat_cols}  Target: '{target or 'None'}'\n"
        f"Pick the {min(MAX_COLS, len(df.columns))} most meaningful columns.\n"
        f"Return ONLY JSON (no markdown): {{\"selected_columns\": [\"col1\", ...]}}"
    )
    try:
        client = get_groq_client()
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=300,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        cols = json.loads(raw).get("selected_columns", [])
        cols = [c for c in cols if c in df.columns]
        if cols:
            return cols[:MAX_COLS]
    except Exception as e:
        print(f"  AI column selection failed ({e}); using fallback.")
    return df[num_cols].var().nlargest(MAX_COLS).index.tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# Page 1 — "Which things move together?"
# Relationship strength grid (simplified correlation heatmap)
# ═══════════════════════════════════════════════════════════════════════════════

def _page_relationships(pdf, df, num_cols, dataset_name):
    cols = num_cols[:10]
    n    = len(cols)
    if n < 2:
        return ""

    corr     = _clean(df, cols).corr(method="pearson")
    fig_size = max(7, n * 0.75 + 2)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.82), facecolor=BG_COLOR)
    fig.subplots_adjust(bottom=0.38, top=0.88, left=0.18, right=0.95)

    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(cols, fontsize=max(6, 9 - n // 3),
                       color=TEXT_COLOR, rotation=40, ha="right")
    ax.set_yticklabels(cols, fontsize=max(6, 9 - n // 3), color=TEXT_COLOR)

    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            if i == j:
                label = "—"
            elif abs(v) >= 0.7:
                label = f"{'🔥' if v > 0 else '❄️'}\n{v:+.2f}"
            else:
                label = f"{v:+.2f}"
            txt_col = "#fff" if abs(v) > 0.5 else "#94a3b8"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=max(5, 8 - n // 4), color=txt_col, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.65, pad=0.02)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.set_ticklabels(["Opposite ❄️", "Slightly\nopposite",
                         "No link", "Slightly\nrelated", "Strongly\nrelated 🔥"],
                        fontsize=7)
    cbar.ax.tick_params(colors=TEXT_COLOR)

    ax.set_title("Which things move together?",
                 fontsize=13, fontweight="bold", color=BAR_COLOR, pad=10)
    ax.text(0.5, 1.02,
            "🔥 = when one goes up, the other goes up too  |  "
            "❄️ = when one goes up, the other goes down  |  blank = no strong link",
            transform=ax.transAxes, ha="center", fontsize=7.5, color="#94A3B8")
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_COLOR)
    ax.tick_params(colors=TEXT_COLOR)

    flat  = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    top_p = flat.idxmax() if len(flat) else ("?", "?")
    top_n = flat.idxmin() if len(flat) else ("?", "?")
    insight = _ask_ai(
        f"Explain this chart to someone with NO data background. "
        f"Dataset: '{dataset_name}'. "
        f"Strongest positive link: '{top_p[0]}' and '{top_p[1]}' ({flat.max():.2f}). "
        f"Strongest opposite link: '{top_n[0]}' and '{top_n[1]}' ({flat.min():.2f}). "
        f"In plain English (no maths, no jargon), 2-3 sentences, "
        f"say what this means in real life."
    )
    _insight_box(fig, insight)
    pdf.savefig(fig, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    print("  [1/5] Relationship grid done.")
    return f"Multivariate [Relationships]: {insight}"


# ═══════════════════════════════════════════════════════════════════════════════
# Page 2 — "What drives [target] the most?"
# Top influencers bar chart (Random Forest importances, shown as %)
# ═══════════════════════════════════════════════════════════════════════════════

def _page_top_influencers(pdf, df, num_cols, target, dataset_name):
    if not target or target not in df.columns:
        return ""

    feat_cols = [c for c in num_cols if c != target]
    if len(feat_cols) < 2:
        return ""

    clean = df[feat_cols + [target]].dropna()
    if len(clean) < 20:
        return ""

    X = clean[feat_cols].values
    y = clean[target].values

    is_regression = (pd.api.types.is_numeric_dtype(clean[target])
                     and clean[target].nunique() > 10)
    model = (RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
             if is_regression else
             RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1))
    model.fit(X, y)

    imp    = model.feature_importances_
    order  = np.argsort(imp)[::-1][:10]
    labels = [feat_cols[i] for i in order]
    pct    = imp[order] / imp[order].sum() * 100

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.55 + 2)),
                           facecolor=BG_COLOR)
    fig.subplots_adjust(bottom=0.38, left=0.02, right=0.88)

    bar_cols = [BAR_COLOR] + [LINE_COLOR] * (len(labels) - 1)
    bars     = ax.barh(labels[::-1], pct[::-1],
                       color=bar_cols[::-1], alpha=0.88, height=0.6)

    for bar, p in zip(bars, pct[::-1]):
        ax.text(bar.get_width() + 0.4,
                bar.get_y() + bar.get_height() / 2,
                f"{p:.1f}%", va="center", fontsize=8.5,
                color=TEXT_COLOR, fontweight="bold")

    # Crown on top bar
    top_bar = bars[-1]
    ax.text(top_bar.get_width() / 2,
            top_bar.get_y() + top_bar.get_height() / 2,
            "👑", ha="center", va="center", fontsize=11)

    ax.set_xlabel("How much influence does this column have?  (%)",
                  fontsize=9, color=TEXT_COLOR)
    ax.set_title(f"What drives  '{target}'  the most?",
                 fontsize=13, fontweight="bold", color=BAR_COLOR, pad=10)
    ax.text(0.5, 1.02,
            "Longer bar = stronger influence on the outcome.  👑 = biggest driver.",
            transform=ax.transAxes, ha="center", fontsize=8, color="#94A3B8")
    _style_ax(ax)
    ax.set_xlim(0, pct.max() * 1.28)

    insight = _ask_ai(
        f"Explain this chart to someone who has never analysed data. "
        f"It shows which columns most influence '{target}' in '{dataset_name}'. "
        f"Top 3 influencers: {labels[:3]} with influence "
        f"{pct[0]:.1f}%, {pct[1]:.1f}%, {pct[2]:.1f}%. "
        f"In plain English (no maths, no jargon), 2-3 sentences, "
        f"say what this means and why it matters."
    )
    _insight_box(fig, insight)
    pdf.savefig(fig, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    print("  [2/5] Top influencers done.")
    return f"Multivariate [Top Influencers → {target}]: {insight}"


# ═══════════════════════════════════════════════════════════════════════════════
# Page 3 — "Are there natural groups in the data?"
# PCA scatter coloured by cluster — "a map of your data"
# ═══════════════════════════════════════════════════════════════════════════════

def _page_natural_groups(pdf, df, num_cols, target, dataset_name):
    cols  = num_cols[:10]
    clean = _clean(df, cols)
    if len(clean) < 15 or len(cols) < 2:
        return ""

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(clean)
    pca    = PCA(n_components=2, random_state=42)
    X_2d   = pca.fit_transform(X_sc)
    ev     = pca.explained_variance_ratio_

    k      = min(4, max(2, len(clean) // 80))
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_sc)
    group_names = [f"Group {i+1}" for i in range(k)]

    # If categorical target available with few values, colour by it instead
    if target and target in df.columns:
        t_ser = df[target]
        if not pd.api.types.is_numeric_dtype(t_ser) and t_ser.nunique() <= 8:
            try:
                t_aligned = df.loc[clean.index, target]
                cats      = t_aligned.astype("category")
                labels    = cats.cat.codes.values
                k         = cats.nunique()
                group_names = [str(c) for c in cats.cat.categories]
            except Exception:
                pass

    PALETTE = ["#F97316", "#38BDF8", "#A78BFA", "#22C55E",
               "#FACC15", "#EC4899", "#14B8A6", "#F43F5E"]

    fig, ax = plt.subplots(figsize=(9, 6.5), facecolor=BG_COLOR)
    fig.subplots_adjust(bottom=0.38, top=0.88)

    for i in range(k):
        mask = labels == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=PALETTE[i % len(PALETTE)],
                   s=20, alpha=0.55, linewidths=0,
                   label=group_names[i])

    ax.set_xlabel(f"← Spread direction 1  ({ev[0]:.0%} of variation)",
                  fontsize=8.5, color=TEXT_COLOR)
    ax.set_ylabel(f"← Spread direction 2  ({ev[1]:.0%} of variation)",
                  fontsize=8.5, color=TEXT_COLOR)
    ax.set_title("Are there natural groups in the data?",
                 fontsize=13, fontweight="bold", color=BAR_COLOR, pad=10)
    ax.text(0.5, 1.02,
            "Each dot = one row.  Same colour = rows that behave similarly.",
            transform=ax.transAxes, ha="center", fontsize=8, color="#94A3B8")

    leg = ax.legend(fontsize=8, facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR,
                    loc="upper right", title="Groups", title_fontsize=8)
    leg.get_title().set_color(BAR_COLOR)
    _style_ax(ax)

    insight = _ask_ai(
        f"Explain this chart to someone with no data knowledge. "
        f"Dataset: '{dataset_name}'. Each dot is a row. "
        f"Similar rows are placed close together. "
        f"There appear to be {k} natural groups. "
        f"The two directions together show {(ev[0]+ev[1]):.0%} of the variation. "
        f"In plain English, 2-3 sentences, explain what 'natural groups' means "
        f"and why a business would care."
    )
    _insight_box(fig, insight)
    pdf.savefig(fig, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    print("  [3/5] Natural groups done.")
    return f"Multivariate [Natural Groups]: {insight}"


# ═══════════════════════════════════════════════════════════════════════════════
# Page 4 — "How does each group behave?"
# Group profile cards — above/below average bars per cluster
# ═══════════════════════════════════════════════════════════════════════════════

def _page_group_profiles(pdf, df, num_cols, dataset_name):
    cols  = num_cols[:8]
    clean = _clean(df, cols)
    if len(clean) < 10 or len(cols) < 2:
        return ""

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(clean)
    k      = min(4, max(2, len(clean) // 80))
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_sc)
    clean  = clean.copy()
    clean["_cluster"] = labels

    gmeans  = clean.groupby("_cluster")[cols].mean()
    overall = clean[cols].mean()
    z       = (gmeans - overall) / (clean[cols].std() + 1e-9)

    PALETTE      = ["#F97316", "#38BDF8", "#A78BFA", "#22C55E"]
    GROUP_EMOJIS = ["🅰️", "🅱️", "🅲️", "🅳️"]

    fig, axes = plt.subplots(1, k, figsize=(3.5 * k, 6.5),
                              facecolor=BG_COLOR, sharey=True)
    fig.subplots_adjust(bottom=0.38, top=0.88, wspace=0.08)
    if k == 1:
        axes = [axes]

    sizes = clean["_cluster"].value_counts().sort_index()

    for ci, ax in enumerate(axes):
        colour = PALETTE[ci % len(PALETTE)]
        z_vals = z.loc[ci].values
        bar_col = [GREEN if v >= 0 else RED for v in z_vals]

        bars = ax.barh(cols, z_vals, color=bar_col, alpha=0.85, height=0.6)
        ax.axvline(0, color=TEXT_COLOR, linewidth=1.0, linestyle="--", alpha=0.4)

        for bar, val in zip(bars, z_vals):
            emoji = "↑" if val >= 0 else "↓"
            ax.text(val + (0.05 if val >= 0 else -0.05),
                    bar.get_y() + bar.get_height() / 2,
                    emoji, va="center",
                    ha="left" if val >= 0 else "right",
                    fontsize=10, color=TEXT_COLOR)

        n_rows = int(sizes.get(ci, 0))
        ax.set_title(f"{GROUP_EMOJIS[ci]}  Group {ci + 1}\n({n_rows} rows)",
                     fontsize=10, fontweight="bold", color=colour, pad=6)
        ax.set_xlabel("← Below avg  |  Above avg →",
                      fontsize=7, color=TEXT_COLOR)
        _style_ax(ax)
        if ci == 0:
            ax.set_yticklabels(cols, fontsize=8, color=TEXT_COLOR)
        ax.set_xlim(-2.5, 2.5)

    fig.suptitle("How does each group behave?",
                 fontsize=13, fontweight="bold", color=BAR_COLOR, y=0.97)
    fig.text(0.5, 0.91,
             "↑ green = higher than average  |  ↓ red = lower than average",
             ha="center", fontsize=8, color="#94A3B8")

    top_col = cols[int(np.argmax(np.abs(z.values).mean(axis=0)))]
    insight = _ask_ai(
        f"Explain this chart to a total beginner. "
        f"Dataset: '{dataset_name}'. "
        f"It shows {k} groups and how each group differs from the average. "
        f"The column that differs most is '{top_col}'. Group sizes: {dict(sizes)}. "
        f"In plain English, 2-3 sentences, explain what this means in practice "
        f"and give a relatable everyday example."
    )
    _insight_box(fig, insight)
    pdf.savefig(fig, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    print("  [4/5] Group profiles done.")
    return f"Multivariate [Group Profiles]: {insight}"


# ═══════════════════════════════════════════════════════════════════════════════
# Page 5 — "How are all columns connected?"
# Bubble connection map — columns as circles, lines = link strength
# ═══════════════════════════════════════════════════════════════════════════════

def _page_connection_map(pdf, df, num_cols, dataset_name):
    cols = num_cols[:10]
    n    = len(cols)
    if n < 3:
        return ""

    corr = _clean(df, cols).corr(method="pearson").values

    fig, ax = plt.subplots(figsize=(9, 7.5), facecolor=BG_COLOR)
    fig.subplots_adjust(bottom=0.36, top=0.90)
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")
    ax.axis("off")

    # Nodes on a circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    R  = 1.0
    xs = R * np.cos(angles)
    ys = R * np.sin(angles)

    # Edges — only draw if abs(corr) > 0.25
    for i in range(n):
        for j in range(i + 1, n):
            v = corr[i, j]
            if abs(v) < 0.25:
                continue
            color = BAR_COLOR if v > 0 else LINE_COLOR
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]],
                    color=color, linewidth=abs(v) * 5,
                    alpha=min(0.85, abs(v) + 0.1), zorder=1)

    # Nodes
    for i in range(n):
        ax.scatter(xs[i], ys[i], s=350, color=SCATTER_COL,
                   zorder=3, edgecolors=TEXT_COLOR, linewidths=1.0)
        lx = xs[i] * 1.22
        ly = ys[i] * 1.22
        ha = "left"   if xs[i] >  0.1 else ("right" if xs[i] < -0.1 else "center")
        va = "bottom" if ys[i] >  0.1 else ("top"   if ys[i] < -0.1 else "center")
        ax.text(lx, ly, cols[i], ha=ha, va=va,
                fontsize=9, color=TEXT_COLOR, fontweight="bold")

    legend_els = [
        mpatches.Patch(color=BAR_COLOR,  label="🔥 Rise together"),
        mpatches.Patch(color=LINE_COLOR, label="❄️ One rises, other falls"),
        mpatches.Patch(color=AXES_FACE,  label="No line = no strong link"),
    ]
    ax.legend(handles=legend_els, loc="lower center", fontsize=8,
              facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR,
              bbox_to_anchor=(0.5, -0.08), ncol=3)

    ax.set_title("How are all columns connected?",
                 fontsize=13, fontweight="bold", color=BAR_COLOR, pad=10)
    ax.text(0.5, 1.02,
            "Each bubble = one column.  Thicker line = stronger connection.",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#94A3B8")

    degree         = (np.abs(corr) > 0.25).sum(axis=1) - 1
    most_connected = cols[int(np.argmax(degree))]
    insight = _ask_ai(
        f"Explain a connection map to someone who has never analysed data. "
        f"Dataset: '{dataset_name}'. "
        f"The most connected column is '{most_connected}'. "
        f"In plain English, 2-3 sentences, explain what it means "
        f"for a column to be 'well connected' and why that matters."
    )
    _insight_box(fig, insight, rect=(0.04, 0.03, 0.92, 0.18))
    pdf.savefig(fig, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    print("  [5/5] Connection map done.")
    return f"Multivariate [Connection Map]: {insight}"


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def multi_visualize_analyze(
    df: pd.DataFrame,
    dataset_name: str,
    target_variable: str = "",
) -> tuple[list[str], bytes]:
    """
    Run beginner-friendly multivariate analysis.

    Parameters
    ----------
    df              : The (cleaned) dataset.
    dataset_name    : Human-readable name of the dataset.
    target_variable : Optional column to use as the prediction target.

    Returns
    -------
    context_store   : Plain-English insight strings for the AI chat.
    pdf_bytes       : In-memory PDF bytes.
    """
    print("── Multivariate Analysis ──────────────────────────────────────────")
    df_s = _sample(df)

    print("  Building column profiles…")
    profiles = _build_profiles(df_s)
    print("  Asking AI to select important columns…")
    selected = _ai_select_columns(df_s, profiles, target_variable, dataset_name)

    num_cols = [c for c in selected if pd.api.types.is_numeric_dtype(df_s[c])]
    cat_cols = [c for c in selected if not pd.api.types.is_numeric_dtype(df_s[c])]

    # Always include target
    if target_variable and target_variable in df_s.columns:
        if target_variable not in num_cols and target_variable not in cat_cols:
            if pd.api.types.is_numeric_dtype(df_s[target_variable]):
                num_cols.insert(0, target_variable)
            else:
                cat_cols.insert(0, target_variable)

    if len(num_cols) < 2:
        print("  Not enough numeric columns for multivariate analysis.")
        return [], b""

    buf           = io.BytesIO()
    context_store = []

    with PdfPages(buf) as pdf:

        # ── Title page ────────────────────────────────────────────────────────
        fig_t = plt.figure(figsize=(9, 4), facecolor=BG_COLOR)
        fig_t.text(0.5, 0.70, "Argus — Multivariate Analysis",
                   ha="center", va="center", fontsize=22,
                   fontweight="bold", color=BAR_COLOR)
        fig_t.text(
            0.5, 0.50,
            f"Dataset: {dataset_name}   |   "
            f"Target: {target_variable or 'None'}   |   "
            f"{len(num_cols)} columns analysed",
            ha="center", va="center", fontsize=10, color="#94A3B8"
        )
        pages_list = (
            "📊 Which things move together?   •   "
            "👑 What drives the outcome?   •   "
            "🗺️ Natural groups   •   "
            "📋 Group behaviour   •   "
            "🕸️ Connection map"
        )
        fig_t.text(0.5, 0.30, pages_list, ha="center", va="center",
                   fontsize=8.5, color="#64748B")
        pdf.savefig(fig_t, facecolor=BG_COLOR)
        plt.close(fig_t)

        # ── 5 story pages ─────────────────────────────────────────────────────
        pages = [
            (_page_relationships,
             dict(num_cols=num_cols, dataset_name=dataset_name)),

            (_page_top_influencers,
             dict(num_cols=num_cols, target=target_variable,
                  dataset_name=dataset_name)),

            (_page_natural_groups,
             dict(num_cols=num_cols, target=target_variable,
                  dataset_name=dataset_name)),

            (_page_group_profiles,
             dict(num_cols=num_cols, dataset_name=dataset_name)),

            (_page_connection_map,
             dict(num_cols=num_cols, dataset_name=dataset_name)),
        ]

        for fn, kwargs in pages:
            try:
                ctx = fn(pdf=pdf, df=df_s, **kwargs)
                if ctx:
                    context_store.append(ctx)
            except Exception as e:
                print(f"  WARNING: {fn.__name__} failed — {e}")

    buf.seek(0)
    pdf_bytes = buf.read()
    print("  Multivariate visualization complete (memory-resident).")
    return context_store, pdf_bytes

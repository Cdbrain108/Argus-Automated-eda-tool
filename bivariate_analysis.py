"""
bivariate_analysis.py — AI-powered bivariate analysis for Argus EDA.

Strategy:
  - Column profiles (stats, unique values, samples) are sent to Groq LLM.
  - AI selects the most important columns and returns pairs to analyse.
  - Plots: scatter (numeric–numeric), bar-mean (numeric–categorical),
           count_bar (categorical–categorical).
  - Groq LLM also generates plain-English descriptions per pair.
  - Returns context strings to feed the AI chat.
"""

import json
import io
import textwrap
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from utils import get_groq_client

MAX_PAIRS = 8
MAX_CAT_UNIQUE = 15
BAR_COLOR = "#F97316"
LINE_COLOR = "#38BDF8"
SCATTER_COL = "#A78BFA"
BG_COLOR = "#0A0F1E"
TEXT_COLOR = "#E2E8F0"


# ── Column profiling ───────────────────────────────────────────────────────────


def _build_column_profiles(df: pd.DataFrame) -> dict:
    """
    Build a compact profile dict for every column with stats, unique values,
    and samples. This is sent to Groq so it can reason about importance.
    """
    profiles = {}
    n = len(df)

    for col in df.columns:
        dtype_str = str(df[col].dtype)
        nunique = int(df[col].nunique(dropna=True))
        missing_pct = round(df[col].isnull().mean() * 100, 1)
        samples = [str(x) for x in df[col].dropna().head(3).tolist()]

        entry = {
            "dtype": dtype_str,
            "nunique": nunique,
            "missing_pct": missing_pct,
            "samples": samples,
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].dropna()
            if len(s) > 0:
                entry["mean"] = round(float(s.mean()), 2)
                entry["std"] = round(float(s.std()), 2)
                entry["min"] = round(float(s.min()), 2)
                entry["max"] = round(float(s.max()), 2)
                entry["skewness"] = round(float(s.skew()), 2)
        else:
            vc = df[col].value_counts()
            if len(vc) > 0:
                entry["top_value"] = str(vc.index[0])
                entry["top_pct"] = round(float(vc.iloc[0]) / n * 100, 1)
                entry["unique_values"] = [str(v) for v in vc.head(5).index.tolist()]

        profiles[col] = entry

    return profiles


# ── AI pair selection ──────────────────────────────────────────────────────────


def _ai_select_pairs(
    df: pd.DataFrame, profiles: dict, target: str, dataset_name: str
) -> list[dict]:
    """
    Send column profiles to Groq AI and ask it to:
      1. Identify the most important columns (with reasoning).
      2. Return bivariate pairs to plot.

    Returns list of dicts: {col1, col2, plot_type}
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [
        c
        for c in df.select_dtypes(exclude="number").columns
        if df[c].nunique() <= MAX_CAT_UNIQUE
    ]

    profile_json = json.dumps(profiles, indent=2)

    if target:
        task3_instructions = f"""
TASK 3: Bivariate Pair Generation
Since a target variable '{target}' has been specified, you MUST specifically pair the most highly correlated (positive or negative) variables WITH this target variable.
- EXPLORE variables which have any correlation (+ve or -ve) with the target variable '{target}'.
- EVERY pair returned MUST include '{target}' as either col1 or col2.
- Choose plot types: 
    - "scatter": for numeric-numeric
    - "bar_mean": for categorical-numeric
    - "count_bar": for categorical-categorical"""
    else:
        task3_instructions = """
TASK 3: Bivariate Pair Generation
From your selected top columns, create insightful bivariate pairs to plot. 
- You should prioritize pairs that show potential correlations, trends, or segments.
- Do NOT just pair every column with the focus variable; find diverse relationships.
- Choose plot types: 
    - "scatter": for numeric-numeric
    - "bar_mean": for categorical-numeric
    - "count_bar": for categorical-categorical"""

    prompt = f"""You are a senior data analyst performing an intelligent bivariate discovery for the '{dataset_name}' dataset.
The dataset has {len(df)} rows and {len(df.columns)} columns.

Here are the detailed column profiles (name → type, uniqueness, samples, stats):
{profile_json}

Numeric columns: {num_cols}
Categorical columns (≤{MAX_CAT_UNIQUE} unique): {cat_cols}
Optional focus variable: '{target or "None"}'

TASK 1: Evaluation of Importance
For EACH column provided in the profiles above, evaluate its importance for understanding the dataset. 
Consider: 
- High variance or interesting distributions (numeric).
- Meaningful categories (not IDs or unique hashes).
- Semantic value (e.g., 'Price' is usually more important than 'RowID').

TASK 2: Column Selection
Identify the top {MAX_PAIRS // 2} to {MAX_PAIRS} most important columns based on your evaluation. 
{task3_instructions}

Return ONLY a valid JSON object:
{{
  "column_importance": [
    {{"column": "col_name", "importance_score": 0.0-1.0, "reason": "brief reasoning"}}
  ],
  "top_columns": ["col1", "col2", ...],
  "pairs": [
    {{"col1": "colA", "col2": "colB", "plot_type": "scatter"}},
    {{"col1": "colC", "col2": "colD", "plot_type": "bar_mean"}}
  ]
}}"""

    try:
        client = get_groq_client()
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1500,
        )
        content = resp.choices[0].message.content.strip()

        # Clean markdown fences
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]

        result = json.loads(content)
        pairs = result.get("pairs", [])

        # Validate and filter pairs
        valid_pairs = []
        seen = set()
        for p in pairs:
            c1 = p.get("col1", "")
            c2 = p.get("col2", "")
            pt = p.get("plot_type", "scatter")
            if c1 in df.columns and c2 in df.columns and c1 != c2:
                key = tuple(sorted([c1, c2]))
                if key not in seen:
                    seen.add(key)
                    valid_pairs.append({"col1": c1, "col2": c2, "plot_type": pt})

        if valid_pairs:
            important_list = result.get("top_columns", [])
            print(
                f"  AI selected {len(valid_pairs)} pairs from the top {len(important_list)} columns."
            )
            return valid_pairs[:MAX_PAIRS]

    except Exception as e:
        print(f"  AI pair selection failed: {e}")

    # Fallback: correlation-based selection
    return _fallback_select_pairs(df, target)


def _fallback_select_pairs(df: pd.DataFrame, target: str) -> list[dict]:
    """Fallback pair selection if AI call fails."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [
        c
        for c in df.select_dtypes(exclude="number").columns
        if df[c].nunique() <= MAX_CAT_UNIQUE
    ]
    pairs = []

    # Top correlated numeric pairs
    if len(num_cols) >= 2:
        corr = df[num_cols].corr().abs()
        np.fill_diagonal(corr.values, 0)
        seen = set()
        for _ in range(MAX_PAIRS // 2):
            idx = corr.stack().idxmax()
            if idx[0] == idx[1]:
                break
            key = tuple(sorted(idx))
            if key not in seen:
                seen.add(key)
                pairs.append({"col1": idx[0], "col2": idx[1], "plot_type": "scatter"})
            corr.loc[idx[0], idx[1]] = 0
            corr.loc[idx[1], idx[0]] = 0

    # Categorical vs numeric
    for cat in cat_cols[:2]:
        for num in num_cols[:2]:
            if {"col1": cat, "col2": num} not in pairs:
                pairs.append({"col1": cat, "col2": num, "plot_type": "bar_mean"})

    return pairs[:MAX_PAIRS]


# ── LLM description ────────────────────────────────────────────────────────────


def _get_description(
    col1: str, col2: str, plot_type: str, df: pd.DataFrame, dataset_name: str
) -> str:
    """Ask Groq for a single plain-English description of the relationship."""
    stats_lines = []
    for c in [col1, col2]:
        s = df[c].dropna()
        if pd.api.types.is_numeric_dtype(s):
            stats_lines.append(
                f"{c}: mean={s.mean():.2f}, std={s.std():.2f}, "
                f"min={s.min():.2f}, max={s.max():.2f}"
            )
        else:
            top = s.value_counts().head(3).index.tolist()
            stats_lines.append(f"{c}: top values = {top}")

    prompt = (
        f"Dataset: {dataset_name}\n"
        f"You are a Senior Data Scientist providing an in-depth bivariate analysis insight "
        f"for the relationship between '{col1}' and '{col2}'.\n"
        f"Stats Overview:\n" + "\n".join(stats_lines) + "\n\n"
        f"TASK:\n"
        f"Write exactly 3-4 highly concise sentences (max 60 words total) explaining the key insight this relationship reveals about '{dataset_name}'. "
        f"If there is any positive or negative correlation, EXPLICITLY state it and briefly explain why. "
        f"Use professional yet accessible language. Do not use markdown. Do not start with 'The chart shows'. Limit your output strictly to fit within a small display box."
    )
    try:
        client = get_groq_client()
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=150,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return f"Relationship between {col1} and {col2}. See chart for details."


# ── Plotting ───────────────────────────────────────────────────────────────────


def _plot_pair(ax_top, col1: str, col2: str, plot_type: str, df: pd.DataFrame) -> str:
    """Draw a single chart on ax_top. Returns the actual plot_type used."""
    try:
        # Auto-correct scatter: needs both numeric
        if plot_type == "scatter":
            if not pd.api.types.is_numeric_dtype(df[col1]) or not pd.api.types.is_numeric_dtype(df[col2]):
                if pd.api.types.is_numeric_dtype(df[col1]) or pd.api.types.is_numeric_dtype(df[col2]):
                    plot_type = "bar_mean"
                else:
                    plot_type = "count_bar"

        # Auto-correct bar_mean: needs at least one numeric
        if plot_type == "bar_mean":
            c_cat, c_num = col1, col2
            if not pd.api.types.is_numeric_dtype(df[col2]) and pd.api.types.is_numeric_dtype(df[col1]):
                c_cat, c_num = col2, col1
            
            if not pd.api.types.is_numeric_dtype(df[c_num]):
                plot_type = "count_bar"

        if plot_type == "scatter":
            x = df[col1].dropna()
            y = df[col2].dropna()
            common_idx = x.index.intersection(y.index)
            x, y = x.loc[common_idx], y.loc[common_idx]

            ax_top.scatter(x, y, alpha=0.45, s=18, color=SCATTER_COL, edgecolors="none")

            if len(x) >= 2:
                m, b = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax_top.plot(
                    x_line,
                    m * x_line + b,
                    color=BAR_COLOR,
                    linewidth=1.5,
                    linestyle="--",
                    label="Trend",
                )
                r = np.corrcoef(x, y)[0, 1]
                ax_top.annotate(
                    f"r = {r:.2f}",
                    xy=(0.04, 0.92),
                    xycoords="axes fraction",
                    fontsize=8,
                    color=LINE_COLOR,
                    fontweight="bold",
                )
            ax_top.set_xlabel(col1, fontsize=9, color=TEXT_COLOR)
            ax_top.set_ylabel(col2, fontsize=9, color=TEXT_COLOR)

        elif plot_type == "bar_mean":
            c_cat, c_num = col1, col2
            if not pd.api.types.is_numeric_dtype(df[col2]) and pd.api.types.is_numeric_dtype(df[col1]):
                c_cat, c_num = col2, col1

            grp = df.groupby(c_cat)[c_num].mean().dropna().sort_values(ascending=False)
            grp = grp.head(MAX_CAT_UNIQUE)
            bars = ax_top.barh(
                grp.index.astype(str), grp.values, color=BAR_COLOR, alpha=0.85
            )
            ax_top.set_xlabel(f"Mean {c_num}", fontsize=9, color=TEXT_COLOR)
            ax_top.set_ylabel(c_cat, fontsize=9, color=TEXT_COLOR)
            for bar in bars:
                w = bar.get_width()
                ax_top.text(
                    w * 1.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{w:.1f}",
                    va="center",
                    fontsize=7,
                    color=TEXT_COLOR,
                )

        elif plot_type == "count_bar":
            # Use col1 for primary grouping; if col1 has too many uniques, use col2
            primary = col1 if df[col1].nunique() <= df[col2].nunique() else col2
            secondary = col2 if primary == col1 else col1

            vc = df[primary].value_counts().head(MAX_CAT_UNIQUE)
            labels_list = [str(v) for v in vc.index]
            counts = vc.values

            bars = ax_top.barh(labels_list[::-1], counts[::-1], color=BAR_COLOR, alpha=0.85)
            ax_top.set_xlabel("Count", fontsize=9, color=TEXT_COLOR)
            ax_top.set_ylabel(primary, fontsize=9, color=TEXT_COLOR)

            # Value labels on bars
            for bar in bars:
                w = bar.get_width()
                ax_top.text(
                    w * 1.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{int(w):,}",
                    va="center",
                    fontsize=7,
                    color=TEXT_COLOR,
                )

        # Common styling
        ax_top.set_facecolor("#111827")
        ax_top.tick_params(colors=TEXT_COLOR, labelsize=7)
        for spine in ax_top.spines.values():
            spine.set_edgecolor("#374151")

    except Exception as e:
        ax_top.text(
            0.5,
            0.5,
            f"Could not render plot.\n{e}",
            ha="center",
            va="center",
            color="red",
            fontsize=8,
            transform=ax_top.transAxes,
        )
    return plot_type


# ── Correlation heatmap page ───────────────────────────────────────────────────


def _plot_correlation_heatmap(pdf, df: pd.DataFrame, important_cols: list[str]):
    """Add a correlation heatmap page for important numeric columns."""
    num_cols = [c for c in important_cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        return

    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, fontsize=7, color=TEXT_COLOR, rotation=45, ha="right")
    ax.set_yticklabels(num_cols, fontsize=7, color=TEXT_COLOR)

    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            val = corr.values[i, j]
            color = "#fff" if abs(val) > 0.5 else "#94a3b8"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=color,
                fontweight="bold",
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    cbar.set_label("Pearson r", color=TEXT_COLOR, fontsize=8)

    ax.set_title(
        "Correlation Heatmap — AI-Selected Important Columns",
        fontsize=11,
        fontweight="bold",
        color=BAR_COLOR,
        pad=10,
    )

    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")

    pdf.savefig(fig, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    print("  Correlation heatmap complete.")


# ── Main entry ─────────────────────────────────────────────────────────────────


def bi_visualize_analyze(
    df: pd.DataFrame, dataset_name: str, target_variable: str = ""
) -> tuple[list[str], bytes]:
    """
    Run AI-powered bivariate analysis:
      1. Build column profiles.
      2. Ask AI to select important columns and pairs.
      3. Plot each pair with AI-generated descriptions.
      4. Save PDF and return context strings for AI chat.
    """
    print("  Building column profiles...")
    profiles = _build_column_profiles(df)

    print("  Asking AI to select important columns & pairs...")
    pairs = _ai_select_pairs(df, profiles, target_variable, dataset_name)

    if not pairs:
        print("  No valid column pairs found.")
        return []

    # Collect all important columns from the pairs for the heatmap
    all_pair_cols = list(set([p["col1"] for p in pairs] + [p["col2"] for p in pairs]))

    buf = io.BytesIO()
    context_store: list[str] = []

    with PdfPages(buf) as pdf:
        # Title page
        fig_title = plt.figure(figsize=(8, 3), facecolor=BG_COLOR)
        fig_title.text(
            0.5,
            0.65,
            "Argus — Bivariate Analysis",
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            color=BAR_COLOR,
        )
        fig_title.text(
            0.5,
            0.42,
            f"Dataset: {dataset_name}   |   "
            f"Target: {target_variable or 'AI-selected'}   |   "
            f"Pairs: {len(pairs)}   |   "
            f"AI-powered pair selection",
            ha="center",
            va="center",
            fontsize=11,
            color="#94A3B8",
        )
        pdf.savefig(fig_title, facecolor=BG_COLOR)
        plt.close(fig_title)

        # Correlation heatmap for important columns
        _plot_correlation_heatmap(pdf, df, all_pair_cols)

        # Plot each AI-selected pair
        for item in pairs:
            col1 = item["col1"]
            col2 = item["col2"]
            plot_type = item["plot_type"]

            if col1 not in df.columns or col2 not in df.columns:
                print(f"  Skipping {col1} vs {col2} — columns not found.")
                continue

            # Get LLM description
            desc = _get_description(col1, col2, plot_type, df, dataset_name)

            # Store for AI chat
            context_store.append(f"Bivariate [{col1} vs {col2}]: {desc}")

            # Build figure with adaptive layout
            # desc box sits at bottom; chart floats above with enough room for x-tick labels
            fig = plt.figure(figsize=(8.5, 7.0), facecolor=BG_COLOR)

            desc_bottom = 0.03
            desc_height = 0.18
            desc_top    = desc_bottom + desc_height   # 0.21

            # 0.14 gap between desc top and chart bottom → clears x-axis tick labels
            chart_bottom = desc_top + 0.14            # ~0.35
            chart_height = 1.0 - chart_bottom - 0.08  # top margin for title

            ax_chart = fig.add_axes([0.10, chart_bottom, 0.82, chart_height])
            ax_desc  = fig.add_axes([0.05, desc_bottom,  0.90, desc_height])

            # Plot
            actual_plot_type = _plot_pair(ax_chart, col1, col2, plot_type, df)

            # Title
            type_label = {
                "scatter": "Scatter + Trend",
                "bar_mean": "Mean Bar",
                "count_bar": "Count Bar",
            }.get(actual_plot_type, actual_plot_type.title())
            ax_chart.set_title(
                f"{col1}  vs  {col2}  ({type_label})",
                fontsize=11,
                fontweight="bold",
                color=BAR_COLOR,
                pad=8,
            )

            # Description panel
            ax_desc.set_facecolor("#1E2D40")
            for spine in ax_desc.spines.values():
                spine.set_edgecolor(BAR_COLOR)
                spine.set_linewidth(0.8)
            wrapped = "\n".join(textwrap.wrap(desc, width=110))
            ax_desc.text(
                0.5,
                0.55,
                wrapped,
                ha="center",
                va="center",
                fontsize=8,
                color=TEXT_COLOR,
                linespacing=1.4,
                transform=ax_desc.transAxes,
                wrap=False,
            )
            ax_desc.text(
                0.5, 0.04,
                "✦ AI Insight",
                ha="center", va="bottom",
                fontsize=7, color=BAR_COLOR,
                transform=ax_desc.transAxes,
                fontweight="bold",
            )
            ax_desc.set_xticks([])
            ax_desc.set_yticks([])
            ax_desc.set_xlabel("", labelpad=0)

            pdf.savefig(fig, facecolor=BG_COLOR, bbox_inches="tight")
            plt.close(fig)
            print(f"  Processed {col1} vs {col2}")

    buf.seek(0)
    pdf_bytes = buf.read()
    print(f"  Bivariate visualization complete (memory-resident).")
    return context_store, pdf_bytes

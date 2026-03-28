"""
bivariate_analysis.py — Smart bivariate analysis for Argus EDA.

Strategy:
  - Column PAIRS are selected via correlation/statistics (not LLM) so the
    target variable always appears in every chart.
  - Groq LLM is called only to generate a plain-English description per pair.
  - Plot types: scatter (numeric–numeric), bar-mean (numeric–categorical).
  - Returns a list of context strings to feed the AI chat.
"""

import json
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from utils import get_groq_client

MAX_PAIRS = 6          # max number of chart pairs
MAX_CAT_UNIQUE = 15    # skip categoricals with more unique values than this
BAR_COLOR   = "#F97316"
LINE_COLOR  = "#38BDF8"
SCATTER_COL = "#A78BFA"
BG_COLOR    = "#0A0F1E"
TEXT_COLOR  = "#E2E8F0"


# ── Pair selection ─────────────────────────────────────────────────────────────

def _select_pairs(df: pd.DataFrame, target: str) -> list[dict]:
    """
    Return a list of dicts:
      {col1, col2, plot_type}  — plot_type in ['scatter','bar_mean','count_bar']
    
    Priority order:
      1. target vs top-N correlated numeric columns (scatter)
      2. target vs top categorical columns by variance ratio  (bar_mean)
      3. If no target, top correlated numeric pairs
    """
    num_cols  = df.select_dtypes(include="number").columns.tolist()
    cat_cols  = [c for c in df.select_dtypes(exclude="number").columns
                 if df[c].nunique() <= MAX_CAT_UNIQUE]

    pairs = []

    if target and target in num_cols:
        feature_nums = [c for c in num_cols if c != target]
        feature_cats = cat_cols

        # Numeric features sorted by abs correlation with target
        if feature_nums:
            corr_vals = (
                df[feature_nums + [target]]
                .corr()[target]
                .drop(target)
                .abs()
                .sort_values(ascending=False)
            )
            for col in corr_vals.index[:MAX_PAIRS // 2 + 1]:
                pairs.append({"col1": target, "col2": col, "plot_type": "scatter"})

        # Categorical features: mean of target per category
        for col in feature_cats[:MAX_PAIRS // 2]:
            pairs.append({"col1": col, "col2": target, "plot_type": "bar_mean"})

    elif target and target in cat_cols:
        # target is categorical – numeric vs target
        for col in num_cols[:MAX_PAIRS]:
            pairs.append({"col1": col, "col2": target, "plot_type": "bar_mean"})

    else:
        # No target: top correlated numeric pairs
        if len(num_cols) >= 2:
            corr = df[num_cols].corr().abs()
            np.fill_diagonal(corr.values, 0)
            seen = set()
            for _ in range(MAX_PAIRS):
                idx = corr.stack().idxmax()
                if idx[0] == idx[1]:
                    break
                key = tuple(sorted(idx))
                if key not in seen:
                    seen.add(key)
                    pairs.append({"col1": idx[0], "col2": idx[1], "plot_type": "scatter"})
                corr.loc[idx[0], idx[1]] = 0
                corr.loc[idx[1], idx[0]] = 0

    return pairs[:MAX_PAIRS]


# ── LLM description ────────────────────────────────────────────────────────────

def _get_description(col1: str, col2: str, plot_type: str,
                     df: pd.DataFrame, dataset_name: str) -> str:
    """Ask Groq for a single plain-English description of the relationship."""
    # Build a compact stats snippet to give context to the LLM
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
        f"You are analysing the relationship between '{col1}' and '{col2}'.\n"
        f"Stats:\n" + "\n".join(stats_lines) + "\n\n"
        f"Write exactly 2 short sentences (under 35 words total) in plain English that "
        f"a non-technical person can understand. Describe what this relationship tells us "
        f"about the data. Do not use jargon. Do not start with 'The chart shows'."
    )
    try:
        client = get_groq_client()
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=80,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Relationship between {col1} and {col2}. See chart for details."


# ── Plotting ───────────────────────────────────────────────────────────────────

def _plot_pair(ax_top, col1: str, col2: str, plot_type: str, df: pd.DataFrame):
    """Draw a single chart on ax_top. No boxplot, no histogram."""
    try:
        if plot_type == "scatter":
            x = df[col1].dropna()
            y = df[col2].dropna()
            common_idx = x.index.intersection(y.index)
            x, y = x.loc[common_idx], y.loc[common_idx]

            ax_top.scatter(x, y, alpha=0.45, s=18, color=SCATTER_COL, edgecolors="none")

            # Trendline
            if len(x) >= 2:
                m, b = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax_top.plot(x_line, m * x_line + b, color=BAR_COLOR,
                            linewidth=1.5, linestyle="--", label="Trend")
                # Pearson r
                r = np.corrcoef(x, y)[0, 1]
                ax_top.annotate(f"r = {r:.2f}", xy=(0.04, 0.92),
                                xycoords="axes fraction", fontsize=8,
                                color=LINE_COLOR, fontweight="bold")
            ax_top.set_xlabel(col1, fontsize=9, color=TEXT_COLOR)
            ax_top.set_ylabel(col2, fontsize=9, color=TEXT_COLOR)

        elif plot_type == "bar_mean":
            # Mean of numeric col2 grouped by categorical col1
            grp = df.groupby(col1)[col2].mean().dropna().sort_values(ascending=False)
            grp = grp.head(MAX_CAT_UNIQUE)
            bars = ax_top.barh(grp.index.astype(str), grp.values,
                               color=BAR_COLOR, alpha=0.85)
            ax_top.set_xlabel(f"Mean {col2}", fontsize=9, color=TEXT_COLOR)
            ax_top.set_ylabel(col1, fontsize=9, color=TEXT_COLOR)
            # value labels
            for bar in bars:
                w = bar.get_width()
                ax_top.text(w * 1.01, bar.get_y() + bar.get_height() / 2,
                            f"{w:.1f}", va="center", fontsize=7, color=TEXT_COLOR)

        elif plot_type == "count_bar":
            grp = df[col1].value_counts().head(MAX_CAT_UNIQUE)
            ax_top.bar(grp.index.astype(str), grp.values, color=LINE_COLOR, alpha=0.85)
            ax_top.set_xlabel(col1, fontsize=9, color=TEXT_COLOR)
            ax_top.set_ylabel("Count", fontsize=9, color=TEXT_COLOR)
            if len(grp) > 6:
                ax_top.tick_params(axis="x", rotation=45)

        # Common styling
        ax_top.set_facecolor("#111827")
        ax_top.tick_params(colors=TEXT_COLOR, labelsize=7)
        for spine in ax_top.spines.values():
            spine.set_edgecolor("#374151")

    except Exception as e:
        ax_top.text(0.5, 0.5, f"Could not render plot.\n{e}",
                    ha="center", va="center", color="red", fontsize=8,
                    transform=ax_top.transAxes)


# ── Main entry ─────────────────────────────────────────────────────────────────

def bi_visualize_analyze(df: pd.DataFrame,
                         dataset_name: str,
                         target_variable: str = "") -> list[str]:
    """
    Run bivariate analysis, save PDF, and return a list of context strings
    suitable for feeding into the AI chat LLM.
    """
    pairs = _select_pairs(df, target_variable)
    if not pairs:
        print("No valid column pairs found.")
        return []

    pdf_path = "Bi_variate_output.pdf"
    context_store: list[str] = []

    with PdfPages(pdf_path) as pdf:
        # Title page
        fig_title = plt.figure(figsize=(8, 3), facecolor=BG_COLOR)
        fig_title.text(0.5, 0.65, "Argus — Bivariate Analysis",
                       ha="center", va="center", fontsize=18,
                       fontweight="bold", color=BAR_COLOR)
        fig_title.text(0.5, 0.42,
                       f"Dataset: {dataset_name}   |   "
                       f"Target: {target_variable or 'Not set'}   |   "
                       f"Pairs: {len(pairs)}",
                       ha="center", va="center", fontsize=11, color="#94A3B8")
        pdf.savefig(fig_title, facecolor=BG_COLOR)
        plt.close(fig_title)

        for item in pairs:
            col1      = item["col1"]
            col2      = item["col2"]
            plot_type = item["plot_type"]

            # Validate columns exist
            if col1 not in df.columns or col2 not in df.columns:
                print(f"Skipping {col1} vs {col2} — columns not found.")
                continue

            # Get LLM description
            desc = _get_description(col1, col2, plot_type, df, dataset_name)

            # Store for AI chat
            context_store.append(
                f"Bivariate [{col1} vs {col2}]: {desc}"
            )

            # Build figure: 70% chart + 30% description panel
            fig = plt.figure(figsize=(8.5, 6.5), facecolor=BG_COLOR)
            ax_chart = fig.add_axes([0.10, 0.30, 0.85, 0.60])  # [left,bot,w,h]
            ax_desc  = fig.add_axes([0.05, 0.02, 0.90, 0.25])

            # Plot
            _plot_pair(ax_chart, col1, col2, plot_type, df)

            # Title
            type_label = {
                "scatter":   "Scatter + Trend",
                "bar_mean":  "Mean Bar",
                "count_bar": "Count Bar",
            }.get(plot_type, plot_type.title())
            ax_chart.set_title(
                f"{col1}  vs  {col2}  ({type_label})",
                fontsize=11, fontweight="bold", color=BAR_COLOR, pad=8
            )

            # Description panel
            ax_desc.set_facecolor("#1E2D40")
            for spine in ax_desc.spines.values():
                spine.set_edgecolor(BAR_COLOR)
                spine.set_linewidth(0.8)
            wrapped = "\n".join(textwrap.wrap(desc, width=90))
            ax_desc.text(
                0.5, 0.5, wrapped,
                ha="center", va="center",
                fontsize=9.5, color=TEXT_COLOR,
                linespacing=1.55,
                transform=ax_desc.transAxes,
                wrap=False,
            )
            ax_desc.set_xticks([]); ax_desc.set_yticks([])
            ax_desc.set_xlabel("💡 AI Insight", fontsize=8,
                               color="#64748B", labelpad=4)

            pdf.savefig(fig, facecolor=BG_COLOR, bbox_inches="tight")
            plt.close(fig)
            print(f"  ✓  {col1} vs {col2}")

    print(f"Saved to {pdf_path} — {len(context_store)} pairs analysed.")
    return context_store
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
#import data_cleaning
from itertools import combinations
import random
from text_generation import AI
from groq import Groq
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import ipywidgets as widgets
from IPython.display import display
from matplotlib.backends.backend_pdf import PdfPages
import importlib
try:
    from utils import get_groq_client
    client = get_groq_client()
except ImportError:
    client = Groq()


# ── FIX 1: Smart column type detection ────────────────────────────────────────

def get_col_type(series):
    """
    Returns one of: 'continuous', 'discrete_numeric', 'categorical_low',
    'categorical_high'
    """
    n_unique = series.nunique()

    if pd.api.types.is_numeric_dtype(series):
        # numeric but very few unique values = treat as category
        if n_unique <= 10:
            return 'discrete_numeric'
        else:
            return 'continuous'
    else:
        # text/object column
        if n_unique <= 15:
            return 'categorical_low'
        else:
            return 'categorical_high'


# ── FIX 2: Updated UnivariateAnalyzer1.analyze_column ─────────────────────────

class UnivariateAnalyzer1:
    def __init__(self, df):
        self.df = df

    def analyze(self):
        analysis_results = {}
        for column in self.df.columns:
            analysis_results[column] = self.analyze_column(column)
        return analysis_results

    def analyze_column(self, column):
        series = self.df[column]
        col_type = get_col_type(series)

        if col_type == 'continuous':
            return {
                'type': 'continuous',
                'mean': round(float(series.mean()), 2),
                'std': round(float(series.std()), 2),
                'min': round(float(series.min()), 2),
                '25%': round(float(series.quantile(0.25)), 2),
                '50%': round(float(series.median()), 2),
                '75%': round(float(series.quantile(0.75)), 2),
                'max': round(float(series.max()), 2),
                'skewness': round(float(series.skew()), 2),
                'missing_pct': round(series.isnull().mean() * 100, 1),
                'n_unique': int(series.nunique()),
            }

        elif col_type == 'discrete_numeric':
            vc = series.value_counts().sort_index()
            return {
                'type': 'discrete_numeric',
                'unique_values': int(series.nunique()),
                'value_counts': {str(k): int(v) for k, v in vc.items()},
                'most_common': str(vc.idxmax()),
                'most_common_count': int(vc.max()),
                'most_common_pct': round(vc.max() / len(series.dropna()) * 100, 1),
                'missing_pct': round(series.isnull().mean() * 100, 1),
            }

        elif col_type in ('categorical_low', 'categorical_high'):
            vc = series.value_counts()
            return {
                'type': col_type,
                'total_count': int(series.count()),
                'unique_values': int(series.nunique()),
                'top_value': str(vc.index[0]),
                'top_value_count': int(vc.iloc[0]),
                'top_value_pct': round(vc.iloc[0] / series.count() * 100, 1),
                'top_10': {str(k): int(v) for k, v in vc.head(10).items()},
                'missing_pct': round(series.isnull().mean() * 100, 1),
            }


# ── FIX 3: Updated UnivariateAnalyzer.visualize ───────────────────────────────

class UnivariateAnalyzer:
    def __init__(self, df, descriptions, uni_columns):
        self.df = df[uni_columns]
        self.descriptions = descriptions
        self.uni_columns = uni_columns

    def visualize(self):
        # ── Theme (matches bivariate_analysis.py) ─────────────────────────────
        BG_COLOR    = "#0A0F1E"
        TEXT_COLOR  = "#E2E8F0"
        BAR_COLOR   = "#F97316"   # orange
        LINE_COLOR  = "#38BDF8"   # sky blue
        SCATTER_COL = "#A78BFA"   # purple
        HIST_COLOR  = "#38BDF8"   # histograms → sky blue
        PANEL_COLOR = "#1E2D40"
        SPINE_COLOR = "#374151"
        AXES_FACE   = "#111827"

        import textwrap
        import matplotlib
        matplotlib.use("Agg")

        with PdfPages('Uni_variate_output1.pdf') as pdf:

            # ── Title page ────────────────────────────────────────────────────
            fig_title = plt.figure(figsize=(8, 3), facecolor=BG_COLOR)
            fig_title.text(0.5, 0.65, "Argus — Univariate Analysis",
                           ha="center", va="center", fontsize=18,
                           fontweight="bold", color=BAR_COLOR)
            fig_title.text(0.5, 0.42,
                           f"Columns: {len(self.df.columns)}   |   "
                           f"Rows: {len(self.df)}",
                           ha="center", va="center",
                           fontsize=11, color="#94A3B8")
            pdf.savefig(fig_title, facecolor=BG_COLOR)
            plt.close(fig_title)

            # ── One page per column ───────────────────────────────────────────
            for column in self.df.columns:
                series = self.df[column]
                col_type = get_col_type(series)

                # Build figure: 65% chart + 30% description panel
                fig = plt.figure(figsize=(8.5, 6.5), facecolor=BG_COLOR)
                ax_chart = fig.add_axes([0.12, 0.32, 0.83, 0.58])
                ax_desc  = fig.add_axes([0.05, 0.02, 0.90, 0.26])

                # ── Chart ─────────────────────────────────────────────────────
                if col_type == 'continuous':
                    ax_chart.hist(series.dropna(), bins=30,
                                  color=HIST_COLOR, alpha=0.75, edgecolor="none")
                    mean_val   = series.mean()
                    median_val = series.median()
                    ax_chart.axvline(mean_val,   color=BAR_COLOR, linestyle='--',
                                     linewidth=1.5, label=f'Mean: {mean_val:.1f}')
                    ax_chart.axvline(median_val, color=SCATTER_COL, linestyle='--',
                                     linewidth=1.5, label=f'Median: {median_val:.1f}')
                    leg = ax_chart.legend(fontsize=8, facecolor=PANEL_COLOR,
                                         edgecolor=BAR_COLOR, labelcolor=TEXT_COLOR)
                    ax_chart.set_title(f"Distribution of {column}",
                                       fontsize=11, fontweight="bold",
                                       color=BAR_COLOR, pad=8)
                    ax_chart.set_xlabel(column, fontsize=9, color=TEXT_COLOR)
                    ax_chart.set_ylabel("Count",  fontsize=9, color=TEXT_COLOR)

                elif col_type == 'discrete_numeric':
                    vc = series.value_counts().sort_index()
                    labels = [str(v) for v in vc.index]
                    counts = vc.values
                    if len(labels) > 15:
                        vc = series.value_counts().head(15)
                        labels = [str(v) for v in vc.index]
                        counts = vc.values
                        ax_chart.set_title(
                            f"Top 15 values of {column} "
                            f"(of {series.nunique()} unique)",
                            fontsize=10, fontweight="bold",
                            color=BAR_COLOR, pad=8)
                    else:
                        ax_chart.set_title(f"Value counts of {column}",
                                           fontsize=11, fontweight="bold",
                                           color=BAR_COLOR, pad=8)
                    bars = ax_chart.bar(range(len(labels)), counts,
                                        color=BAR_COLOR, alpha=0.85)
                    ax_chart.set_xticks(range(len(labels)))
                    ax_chart.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
                    ax_chart.set_xlabel(column, fontsize=9, color=TEXT_COLOR)
                    ax_chart.set_ylabel("Count",  fontsize=9, color=TEXT_COLOR)
                    for bar, count in zip(bars, counts):
                        ax_chart.text(bar.get_x() + bar.get_width() / 2.,
                                      bar.get_height() + 0.3,
                                      str(count), ha='center', va='bottom',
                                      fontsize=7, color=TEXT_COLOR)

                elif col_type == 'categorical_low':
                    vc = series.value_counts()
                    bars = ax_chart.barh(range(len(vc)), vc.values,
                                         color=LINE_COLOR, alpha=0.85)
                    ax_chart.set_yticks(range(len(vc)))
                    ax_chart.set_yticklabels([str(v)[:20] for v in vc.index],
                                             fontsize=8, color=TEXT_COLOR)
                    ax_chart.set_title(f"Count of {column}",
                                       fontsize=11, fontweight="bold",
                                       color=BAR_COLOR, pad=8)
                    ax_chart.set_xlabel("Count",  fontsize=9, color=TEXT_COLOR)
                    ax_chart.set_ylabel(column,   fontsize=9, color=TEXT_COLOR)
                    for i, (val, count) in enumerate(vc.items()):
                        ax_chart.text(count + 0.3, i, str(count),
                                      va='center', fontsize=7, color=TEXT_COLOR)

                elif col_type == 'categorical_high':
                    vc = series.value_counts().head(10)
                    n_total = series.nunique()
                    bars = ax_chart.barh(range(len(vc)), vc.values,
                                         color=SCATTER_COL, alpha=0.85)
                    ax_chart.set_yticks(range(len(vc)))
                    ax_chart.set_yticklabels([str(v)[:20] for v in vc.index],
                                             fontsize=8, color=TEXT_COLOR)
                    ax_chart.set_title(
                        f"Top 10 of {column} ({n_total} unique values total)",
                        fontsize=10, fontweight="bold", color=BAR_COLOR, pad=8)
                    ax_chart.set_xlabel("Count", fontsize=9, color=TEXT_COLOR)
                    ax_chart.invert_yaxis()
                    total = series.count()
                    for i, (val, count) in enumerate(vc.items()):
                        pct = round(count / total * 100, 1)
                        ax_chart.text(count + 0.3, i, f"{count} ({pct}%)",
                                      va='center', fontsize=7, color=TEXT_COLOR)

                # ── Common chart styling (matches bivariate) ──────────────────
                ax_chart.set_facecolor(AXES_FACE)
                ax_chart.tick_params(colors=TEXT_COLOR, labelsize=7)
                for spine in ax_chart.spines.values():
                    spine.set_edgecolor(SPINE_COLOR)

                # ── Description panel (matches bivariate exactly) ─────────────
                ax_desc.set_facecolor(PANEL_COLOR)
                for spine in ax_desc.spines.values():
                    spine.set_edgecolor(BAR_COLOR)
                    spine.set_linewidth(0.8)
                desc_text = self.descriptions.get(column, 'No description available.')
                wrapped = "\n".join(textwrap.wrap(desc_text, width=90))
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

                pdf.savefig(fig, facecolor=BG_COLOR, bbox_inches='tight')
                plt.close(fig)


# ── uni_analyze_and_visualize ─────────────────────────────────────────────────

def uni_analyze_and_visualize(df_5, dataset_name, target_variable):
    """Run univariate analysis, generate PDF, and return context strings for AI chat."""
    text_generation = importlib.import_module("text_generation")
    ai_instance = AI()
    uni_poss_corr = ai_instance.uni_poss_corr
    analyzer1 = UnivariateAnalyzer1(df_5)
    analysis_results = analyzer1.analyze()
    descriptions = {}
    context_store: list = []

    for column, stats in analysis_results.items():
        # ── FIX 4: Column-type-aware AI prompts ──────────────────────────────
        col_type = stats.get('type', 'continuous')

        if col_type == 'continuous':
            prompt = (
                f"Describe the column '{column}' for a non-technical person. "
                f"Stats: mean={stats['mean']}, median={stats['50%']}, "
                f"std={stats['std']}, min={stats['min']}, max={stats['max']}, "
                f"skewness={stats['skewness']}, missing={stats['missing_pct']}%. "
                f"Cover: what the column likely measures, whether values are "
                f"spread out or clustered, and any skew. Under 80 tokens."
            )

        elif col_type == 'discrete_numeric':
            prompt = (
                f"Describe the column '{column}' for a non-technical person. "
                f"It has {stats['unique_values']} unique numeric values. "
                f"Most common value: {stats['most_common']} "
                f"appearing {stats['most_common_count']} times "
                f"({stats['most_common_pct']}% of records). "
                f"Missing: {stats['missing_pct']}%. "
                f"Cover: what this column likely represents and which value "
                f"dominates. Under 80 tokens."
            )

        elif col_type in ('categorical_low', 'categorical_high'):
            top_items = list(stats['top_10'].items())[:3]
            top_str = ', '.join([f"'{k}' ({v} times)" for k, v in top_items])
            prompt = (
                f"Describe the column '{column}' for a non-technical person. "
                f"It has {stats['unique_values']} unique values out of "
                f"{stats['total_count']} records. "
                f"Top value: '{stats['top_value']}' appears {stats['top_value_count']} "
                f"times ({stats['top_value_pct']}%). "
                f"Top 3 values: {top_str}. "
                f"Missing: {stats['missing_pct']}%. "
                f"Cover: what this column represents and whether one value "
                f"dominates or values are spread. Under 80 tokens."
            )
        else:
            prompt = (
                f"Describe the column '{column}' for a non-technical person "
                f"based on these stats: {stats}. Under 80 tokens."
            )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        desc = response.choices[0].message.content.strip()
        descriptions[column] = desc
        context_store.append(f"Univariate [{column}]: {desc}")

    uni_columns = uni_poss_corr(df_5, dataset_name, target_variable)
    uni_columns = [col for col in uni_columns if col in df_5.columns]
    analyzer = UnivariateAnalyzer(df_5, descriptions, uni_columns)
    analyzer.visualize()
    print("Visualization saved to uni_variate_output1.pdf")
    return context_store
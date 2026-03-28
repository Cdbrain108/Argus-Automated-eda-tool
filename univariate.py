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
        with PdfPages('Uni_variate_output1.pdf') as pdf:
            for column in self.df.columns:
                series = self.df[column]
                col_type = get_col_type(series)

                fig, axs = plt.subplots(2, 1, figsize=(8, 5))
                fig.subplots_adjust(hspace=0.5)

                # ── TOP PLOT: chart based on col_type ──────────────────────────

                if col_type == 'continuous':
                    sns.histplot(series.dropna(), kde=True, ax=axs[0],
                                 color='steelblue', bins=30)
                    mean_val = series.mean()
                    median_val = series.median()
                    axs[0].axvline(mean_val, color='red', linestyle='--',
                                   linewidth=1, label=f'Mean: {mean_val:.1f}')
                    axs[0].axvline(median_val, color='green', linestyle='--',
                                   linewidth=1, label=f'Median: {median_val:.1f}')
                    axs[0].legend(fontsize=7)
                    axs[0].set_title(f"Distribution of {column}", fontsize=10)
                    axs[0].set_xlabel(column, fontsize=8)
                    axs[0].set_ylabel("Count", fontsize=8)

                elif col_type == 'discrete_numeric':
                    vc = series.value_counts().sort_index()
                    labels = [str(v) for v in vc.index]
                    counts = vc.values

                    # Trim to top 15 if somehow still too many
                    if len(labels) > 15:
                        vc = series.value_counts().head(15)
                        labels = [str(v) for v in vc.index]
                        counts = vc.values
                        axs[0].set_title(
                            f"Top 15 values of {column} "
                            f"(of {series.nunique()} unique)", fontsize=9
                        )
                    else:
                        axs[0].set_title(f"Value counts of {column}", fontsize=10)

                    bars = axs[0].bar(range(len(labels)), counts, color='steelblue')
                    axs[0].set_xticks(range(len(labels)))
                    axs[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
                    axs[0].set_xlabel(column, fontsize=8)
                    axs[0].set_ylabel("Count", fontsize=8)

                    # Count labels on top of bars
                    for bar, count in zip(bars, counts):
                        axs[0].text(
                            bar.get_x() + bar.get_width() / 2.,
                            bar.get_height() + 0.3,
                            str(count), ha='center', va='bottom', fontsize=7
                        )

                elif col_type == 'categorical_low':
                    vc = series.value_counts()
                    axs[0].barh(range(len(vc)), vc.values, color='steelblue')
                    axs[0].set_yticks(range(len(vc)))
                    axs[0].set_yticklabels(
                        [str(v)[:20] for v in vc.index], fontsize=8
                    )
                    axs[0].set_title(f"Count of {column}", fontsize=10)
                    axs[0].set_xlabel("Count", fontsize=8)
                    axs[0].set_ylabel(column, fontsize=8)
                    for i, (val, count) in enumerate(vc.items()):
                        axs[0].text(count + 0.3, i, str(count),
                                    va='center', fontsize=7)

                elif col_type == 'categorical_high':
                    vc = series.value_counts().head(10)
                    n_total_unique = series.nunique()

                    axs[0].barh(range(len(vc)), vc.values, color='steelblue')
                    axs[0].set_yticks(range(len(vc)))
                    axs[0].set_yticklabels(
                        [str(v)[:20] for v in vc.index], fontsize=8
                    )
                    axs[0].set_title(
                        f"Top 10 of {column} ({n_total_unique} unique values total)",
                        fontsize=9
                    )
                    axs[0].set_xlabel("Count", fontsize=8)
                    axs[0].invert_yaxis()
                    total = series.count()
                    for i, (val, count) in enumerate(vc.items()):
                        pct = round(count / total * 100, 1)
                        axs[0].text(count + 0.3, i, f"{count} ({pct}%)",
                                    va='center', fontsize=7)

                # ── BOTTOM PLOT: AI description text ───────────────────────────
                axs[1].text(
                    0.5, 0.5,
                    self.descriptions.get(column, 'No description available.'),
                    wrap=True,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=8,
                    transform=axs[1].transAxes,
                )
                axs[1].axis('off')

                pdf.savefig(fig, bbox_inches='tight')
                plt.close()


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
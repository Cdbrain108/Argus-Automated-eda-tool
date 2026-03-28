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
client = Groq(api_key="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")
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
        if pd.api.types.is_numeric_dtype(series):
            return {
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                '25%': series.quantile(0.25),
                '50%': series.quantile(0.50),
                '75%': series.quantile(0.75),
                'max': series.max(),
                'describe': series.describe()
            }
        else:
            return {
                'count': series.count(),
                'unique': series.nunique(),
                'top': series.mode()[0] if not series.mode().empty else None,
                'freq': series.value_counts().iloc[0] if not series.value_counts().empty else None,
                'describe': series.describe()
            }

class UnivariateAnalyzer:
    def __init__(self, df, descriptions,uni_columns):
        self.df = df[uni_columns]
        self.descriptions = descriptions
        self.uni_columns = uni_columns

    def visualize(self):
        with PdfPages('Uni_variate_output1.pdf') as pdf:
            for i, column in enumerate(self.df.columns):
                fig, axs = plt.subplots(2, 1, figsize=(6, 4))
                if pd.api.types.is_numeric_dtype(self.df[column]):
                    sns.histplot(self.df[column], kde=True, ax=axs[0])
                    axs[0].set_title(f"Distribution of {column}")
                    axs[0].set_xlabel(column)
                else:
                    if not self.df[column].empty and not pd.isnull(self.df[column]).all():
                        sns.countplot(y=self.df[column], ax=axs[0])
                        axs[0].set_title(f"Count of {column}")
                        axs[0].set_xlabel('Count')
                        axs[0].set_ylabel(column)

                axs[1].text(0.5, 0.5, self.descriptions[column], wrap=True, horizontalalignment='center', verticalalignment='center', fontsize=8)
                axs[1].axis('off')  # Hide the axes

                pdf.savefig(fig)  # saves the current figure into a pdf page
                plt.close()



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
        prompt = (
            f"Please generate a description (univariate analysis) for a dataset column "
            f"with the following statistics in simple words so that a non technical person "
            f"can also understand what's going on — firstly explain about the column and "
            f"then the variation. Complete within 80 tokens only: {stats}"
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
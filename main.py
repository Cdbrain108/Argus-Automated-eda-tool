import pandas as pd
import openai
import data_cleaning
from data_cleaning import DatasetCleaning
import input_file
import os
import difflib
import text_generation
import re
import groq
from groq import Groq
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import warnings
#import bivariate_analysis
from univariate import uni_analyze_and_visualize
#import bivariate_analysis
warnings.filterwarnings("ignore")
try:
    from utils import get_groq_client
    client = get_groq_client()
except ImportError:
    client = Groq()

data, dataset_name, target_variable = input_file.load_data()
cleaner = DatasetCleaning(data)
GPT = text_generation.AI()
if data is not None:
    df = pd.DataFrame(data)
    print("AI Describing Dataset Summary: ")
    print("....................................................")
    summary = GPT.data_summary(df, dataset_name, target_variable)
    df_1 = cleaner.remove_duplicates(df) 
    objects = cleaner.object_columns(df)
    obj_cols = GPT.object_correction(dataset_name, target_variable, objects)
    df_2 = cleaner.convert_dtypes(df_1, obj_cols)
    df_3 = cleaner.onehot_encode(df_2)
    df_4 = cleaner.remove_missing(df_3)
    df_5, outliers_count = cleaner.remove_outliers(df_4)
    features_draw = cleaner.plot_feature_importance(df_5, target_variable)
    uni_columns = GPT.uni_poss_corr(df_5, dataset_name, target_variable)
    print("......................................................")
    uni_analyze_and_visualize(df_5, dataset_name, target_variable)
    #test = GPT.bi_poss_corr(df_5,dataset_name,target_variable)
    print("....................................................")
    #bivariate_analysis.bi_visualize_analyze(df, dataset_name, target_variable)
    #bi_analyze_and_visualize(df_5,dataset_name)
    #Bivariate_analyzer = bivariate_analysis.BivariateAnalyzer(df_5,dataset_name)
    #print("Bivariate Analysis: ")
        #Bivariate_analyzer.bi_analyze_and_visualize()
    
    print("Dataset Name:", dataset_name)
    print("Before Cleaning:", df.shape, "After Preprocessing Operations:", df_5.shape)
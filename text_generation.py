import pandas as pd
import os
import difflib
import re
from groq import Groq
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import random
import importlib
import warnings
warnings.filterwarnings("ignore")
client = Groq(api_key="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")
#from bivariate_analysis import get_analysis_results
class AI:
    def __init__(self):
        pass    
        
    def data_summary(self,df,dataset_name, target_variable):
            # Load the dataset
            # Get basic information about the dataset
            num_records = len(df)
            num_features = len(df.columns)
            feature_names = [str(name) for name in df.columns.tolist()]  # Convert feature names to strings
            data_types = df.dtypes.tolist()
            data_types_str = [str(dtype) for dtype in data_types]
            missing_values = df.isnull().sum().tolist()
            dataset_shape = df.shape
            if target_variable is None:
                 target_variable = "not provided"        
            
             # Generate a brief introduction using GPT API
            prompt = f"this is a dataset of {dataset_name} and it has a target variable {target_variable} overall shape of dataset is {dataset_shape}.The dataset contains {num_records} records and {num_features} features. The features include: {', '.join(feature_names)}. The data types of features are: {', '.join(map(str, data_types_str))}."
            prompt += f" Total missing values in each feature are: {', '.join(map(str, missing_values))}. Here are the summary statistics:\n{df.describe(include='all')}"
            prompt += f" Please provide a Summary of overall dataset in simple words"
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            # Print the generated introduction
            print(response.choices[0].message.content)

    def object_correction(self,dataset_name, target_variable, objects):
        obj = objects
        if target_variable is None:
            target_variable = "not provided"
        prompt = f"This dataset is named {dataset_name} and it includes target variables {target_variable}. The following dictionary {obj} contains column names of Object data types along with their unique data values. Please analyze the correct data type for each column and return a dictionary with the column names(original as given in input set) and their corresponding pandas data types (e.g., category, float, datetime, etc.). RETURN STRICTLY A RAW VALID PYTHON DICTIONARY STARTING WITH '{{' AND ENDING WITH '}}'. DO NOT ADD ANY CONVERSATIONAL TEXT, EXPLANATION, OR MARKDOWN BLOCK."
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        # Parse the generated response into dictionary format
        response_text = response.choices[0].message.content
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            obj_columns = eval(match.group(0))
        else:
            obj_columns = eval(response_text)
        
        return obj_columns
    def uni_poss_corr(self,df,dataset_name,target_variable):
            dataset_columns = df.columns.tolist()
            data_types = df.dtypes.tolist()
            max_unique_values=10
            object_columns={}
            for col in df.columns:
                unique_values = df[col].unique().tolist()
                if len(unique_values) > max_unique_values:
                    unique_values = random.sample(unique_values, max_unique_values)
                    object_columns[col] = unique_values
            prompt = f"This dataset is named {dataset_name} and it includes target variable {target_variable}. here are the data columns {dataset_columns} in dataframe with their {data_types} datatypes i want a list of datacolumns at which graph plotting (for univariate) is possibe means ignore name,id,passengerid,and another unusual data column names and don't forget to include target variable and key important variable for the datasets.Please ensure and return only a list of case sesnsitive df columns. RETURN STRICTLY A RAW VALID PYTHON LIST STARTING WITH '[' AND ENDING WITH ']'. DO NOT ADD ANY CONVERSATIONAL TEXT, EXPLANATION, OR MARKDOWN BLOCK."
            response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.7,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
            uni_poss = (response.choices[0].message.content)    
                # Parse the generated response into dictionary format
            try:   
                match = re.search(r'\[.*\]', uni_poss, re.DOTALL)
                if match:
                    uni_columns = eval(match.group(0))
                else:
                    uni_columns = eval(uni_poss)
                return uni_columns
            except Exception as e:
                print("Error:", e)
                return None
    """def bi_poss_corr(self,df, dataset_name, target_variable):
        bivariate_analysis = importlib.import_module("bivariate_analysis")
        get_analysis_results = bivariate_analysis.get_analysis_results
        analysis_results = get_analysis_results(df, dataset_name)
        dataset_columns = df.columns.tolist()
        data_types = df.dtypes.tolist()
        max_unique_values = 10
        unique_counts = df.nunique()
        object_columns = {}
        length = len(df.columns)
        length = 2 * length
        for col in df.columns:
            unique_values = df[col].unique().tolist()
            if len(unique_values) > max_unique_values:
                unique_values = random.sample(unique_values, max_unique_values)
                object_columns[col] = unique_values
            prompt = (
                f"In the {dataset_name} dataset, perform a bivariate analysis with the target variable '{target_variable}'. "
                f"Return a dictionary of at least {length} column pairs, where each pair of columns should be the target variable '{target_variable}'. "
                f"Use the correlations {analysis_results} to select pairs. "
                f"Include pairs with the most positive, most negative, and neutral correlations. "
                f"Format each pair as 'column1':'column2'. "
                f"Use full, case-sensitive, and unique variable names. "
                f"Avoid syntax errors and do not include the correlation value in the output. "
                f"Focus on correlation (most positive, most negative and balanced) the target variable with as many other variables as possible."
                f"each column will pair with at least 3 other columns except target_variable"
                f"Output format of your will be a dictionary with the format {{\column\ : \column\, \column\ : \column\}} only."
                )
            print(prompt)
            response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.7,
                    top_p=0.7,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                
                # Parse the generated response into dictionary format
            response_content = response.choices[0].message.content
            try:
                # Remove the outer double quotes and newline characters
                response_content = response_content.strip('"\n')
                # Replace curly braces with square brackets to convert dictionary to list
                response_content = response_content.replace('{', '[').replace('}', ']')
                # Split the string into lines
                lines = response_content.split(',')
                bi_columns = [{line.strip().split(':')[0].strip().strip('"\n[').strip("'"): line.strip().split(':')[1].strip().strip('"\n]').strip("'")} for line in lines if ':' in line]            
                for i in range(len(bi_columns)):
                    for key, value in bi_columns[i].items():
                        # Remove the quotes from the key and value
                        new_key = key.strip("'")
                        new_value = value.strip("'")
                        # Update the dictionary with the new key and value
                        bi_columns[i] = {new_key: new_value}            
                return bi_columns
            except Exception as e:
                print("Error:", e)
                return None
"""
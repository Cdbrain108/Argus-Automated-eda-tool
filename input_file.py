import io
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import requests
from io import BytesIO
import os
import difflib
import re
from urllib.parse import urlparse
from groq import Groq
import validators
import gdown
from urllib.parse import parse_qs
def cleaning(data):
        clean_col_names = []
        for col in data.columns:
            clean_col = re.sub(r'[_\-]', ' ', col)
            clean_col = re.sub(r'(?<!^)(?=[A-Z])', ' ', col)
            clean_col = clean_col.strip()
            clean_col = clean_col.replace('_', ' ')
            clean_col = re.sub(r'\s+', ' ', clean_col)
            clean_col = clean_col.lower()# Improved regex for better name cleaning
            clean_col_names.append(clean_col)
        data.columns = clean_col_names
        return data
def extract_file_id(drive_link):
        """Extracts the file ID from a Google Drive link."""
        file_id = None
        try:
            parsed = urlparse(drive_link)
            file_id = drive_link.split("/")[5]
            filename = os.path.basename(parsed.path)
        except IndexError:
            print("Invalid Google Drive link provided.")
        return file_id

def download_file(file_id):
        """Downloads the file from Google Drive."""
        download_url = f"https://drive.google.com/uc?id={file_id}"
        response = requests.get(download_url)
        return response.content

def load_data():
        print("How would you like to input the data?")
        print("1. Browse from system")
        print("2. Provide a link to online data")
        choice = input("Enter your choice (1 or 2): ")

        if choice == '1':
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            file_path = filedialog.askopenfilename()  # Open file dialog

            if file_path:
                try:
                    # Detect file type and read accordingly
                    file_ext = os.path.splitext(file_path)[-1].lower()
                    if file_ext == '.csv' or file_ext in ['.xls', '.xlsx']:
                        if file_ext == '.csv':
                            with open(file_path, 'r') as file:
                                first_line = file.readline()
                                separator = re.search("[,;\t]", first_line).group()
                            data = pd.read_csv(file_path, sep=separator)
                        else:
                            data = pd.read_excel(file_path)
                    else:
                        print("Unsupported file format. Please provide a CSV, XLS, or XLSX file.")
                        return None, None,None
                    # Check if data has any columns
                    if len(data.columns) == 0:
                        print("Error: The file contains no columns.")
                        return None, None,None

                    # Clean column names
                    data = cleaning(data)

                    # Extract filename and get dataset name confirmation
                    filename = os.path.basename(file_path)
                    dataset_name = filename
                    #confirm_name = input(f"Is '{filename}' the intended dataset name? (yes/no): ")
                    #dataset_name = filename if confirm_name.lower() == 'yes' else input("Enter the dataset name: ")
                    #target_variable, data = targeted_variable(data)  # Capture the returned value from target_variable function
                    target_variable = get_target_variables(data, dataset_name)
                    return data, dataset_name, target_variable
                    

                except Exception as e:
                    print("Error:", e)
                    return None, None,None

            else:
                print("No file selected.")
                return None, None,None

        elif choice == '2':
            url = input("Enter the URL of the online data: ")
            if not validators.url(url):
                print("Invalid URL. Please check the URL and try again.")
                return None, None,None
            try:
                data = None
                if 'drive.google.com' in url:
                    # Google Drive URL
                    try:
                        file_id = extract_file_id(url)
                        if file_id is not None:
                            file_content = download_file(file_id)
                            # Try reading the file
                            try:
                                data = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
                            except pd.errors.ParserError:
                                try:
                                    data = pd.read_excel(io.BytesIO(file_content))
                                except Exception as e:
                                    print("Unsupported file format. Please provide a CSV, XLS, or XLSX file.")
                                    return None, None,None
                        else:
                            print("Invalid Google Drive URL. The URL should contain an 'id' parameter.")
                            return None, None,None
                    except Exception as e:
                        print("Error:", e)
                        print("An error occurred. Please check the URL and try again.")
                        return None, None,None

                else:
                    # Other URL
                    response = requests.get(url)
                    if url.endswith('.csv'):
                        data = pd.read_csv(BytesIO(response.content))
                    elif url.endswith('.xls') or url.endswith('.xlsx'):
                        data = pd.read_excel(BytesIO(response.content))
                    else:
                        print("Unsupported file format. Please provide a CSV, XLS, or XLSX file.")
                        return None, None,None
                # Clean column names
                data = cleaning(data)

                # Check if data has any columns
                if len(data.columns) == 0:
                    print("Error: The online data contains no columns.")
                    return None, None,None
                path = urlparse(url).path
                filename = os.path.basename(path)
                dataset_name = filename
                target_variable = get_target_variables(data, dataset_name)
                return data, dataset_name, target_variable
            except Exception as e:
                print("Error:", e)
                print("An error occurred. Please check the URL and try again.")
                return None, None,None
        else:
            print("No file selected.")
            return None, None,None
        
client = Groq(api_key="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")

def get_target_variables(data,dataset_name):
                # Load the dataset
                # Get basic information about the dataset
                df = pd.DataFrame(data)
                num_records = len(df)
                num_features = len(df.columns)
                feature_names = [str(name) for name in df.columns.tolist()]  # Convert feature names to strings
                data_types = df.dtypes.tolist()
                data_types_str = [str(dtype) for dtype in data_types]
                missing_values = df.isnull().sum().tolist()
                dataset_shape = df.shape       
                
                # Generate a brief introduction using GPT API
                prompt = f"this is a dataset of {dataset_name} and overall shape of dataset is {dataset_shape}.The dataset contains {num_records} records and {num_features} features. The features include: {', '.join(feature_names)}. The data types of features are: {', '.join(map(str, data_types_str))}."
                prompt += f"Here are the summary statistics:\n{df.describe(include='all')}"
                prompt += f" Please provide most possible target varaible (only one) of dataset return in a only a single string."
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
                target_variable = (response.choices[0].message.content)
                # If the response is not a string, convert it to a string
                if not isinstance(target_variable, str):
                    target_variable = str(target_variable)
                if target_variable:
                    print("Target variables found:", target_variable)
                else:
                    print("No target variables found.")
                    target_variable = None
                return target_variable
"""def targeted_variable(data):
        print("Columns:", data.columns)
        while True:
            has_target_variable = input("Does the dataset have a target variable? (yes/no): ")
            if has_target_variable.lower() in ['yes', 'no']:
                break
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")

        if has_target_variable.lower() == 'yes':
            target_variable = asking_target(data)
        else:
            print("Target variable is not provided.")
            target_variable = None

        return target_variable, data  # Return data in all cases
        
    def asking_target(data):
        target_variable = []
        while True:
            target_variable = input("Enter the full names of the target variables of your dataset: ").lower()
            if target_variable in data.columns:
                print(f"Target variable {target_variable} confirmed.")
                target_variable.append(target_variable)
                if len(target_variable) >= len(data.columns) // 3:
                    break
                more = input("Does the dataset have more target variables? (yes/no): ").lower()
                if more == 'no':
                    break
            else:
                similar_cols = [col for col in data.columns if target_variable in col]
                if similar_cols:
                    print(f"Target variable not found. Did you mean one of these: {similar_cols}?")
                    confirm_target = input("Enter 'yes' to confirm or 'no' to enter again: ")
                    if confirm_target.lower() == 'yes':
                        for i, col in enumerate(similar_cols, start=1):
                            print(f"{i} for {col}")
                        user_input = input("Enter the index num of the target variable: ")
                        if user_input.isdigit() and int(user_input) <= len(similar_cols):
                            target_variable = similar_cols[int(user_input) - 1]
                            print(f"Target variable {target_variable} confirmed.")
                            target_variable.append(target_variable)
                            more = input("Does the dataset have more target variables? (yes/no): ").lower()
                            if more == 'no':
                                break
                else:
                    print("Target variable not found.")
                    target_variable = None

return target_variable"""
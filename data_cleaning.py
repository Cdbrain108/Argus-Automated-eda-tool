import pandas as pd
import numpy as np
import re
import random
from utils import get_groq_client

class SmartDataCleaner:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

    def scan_missing_values(self):
        """
        Scans for pure nulls, NaNs, empty strings, and whitespace-only cells.
        Returns a pre-clean summary DataFrame: 'Column Name', 'Data Type', 'Missing Count', '% Missing'.
        """
        df = self.data
        report = []
        total_rows = len(df)
        
        for col in df.columns:
            # 1. Standard nulls
            null_mask = df[col].isnull()
            
            # 2. String-specific missingness (empty, whitespace, or generic pandas NaN string)
            if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                # Convert to string and check for pure whitespace/empty
                str_mask = df[col].astype(str).str.strip() == ""
                null_mask = null_mask | str_mask
                
            missing_count = int(null_mask.sum())
            pct_missing = round((missing_count / total_rows) * 100, 2) if total_rows > 0 else 0.0
            
            report.append({
                "Column Name": col,
                "Data Type": str(df[col].dtype),
                "Missing Count": missing_count,
                "% Missing": pct_missing
            })
            
        return pd.DataFrame(report)

    def sanitize_data(self):
        """
        Task 3 - Character / Special Value Handling
        - Strip leading/trailing whitespace
        - Replace junk ('N/A', 'none', 'null', '?', '-', 'NA') with NaN
        - Standardize casing to title-case for strings
        - Remove non-printable / special chars using regex
        """
        df = self.data
        junk_values = ['n/a', 'none', 'null', '?', '-', 'na', '']
        
        for col in df.columns:
            if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                # Strip leading/trailing whitespace
                s = df[col].astype(str).str.strip()
                
                # Replace junk with NaN
                s_lower = s.str.lower()
                mask_junk = s_lower.isin(junk_values) | s_lower.isna()
                
                # Apply title casing to valid strings
                # Remove non-printable / special chars (allow alphanumeric, punctuation, spaces)
                s_clean = s.apply(lambda x: re.sub(r'[^\x20-\x7E]', '', str(x)).title() if pd.notnull(x) else x)
                
                # Assign NaNs back
                df.loc[mask_junk, col] = np.nan
                df.loc[~mask_junk, col] = s_clean[~mask_junk]
                
                # Type inference: Attempt to convert back to numeric after cleaning
                num_converted = pd.to_numeric(df[col], errors='coerce')
                valid_total = df[col].notnull().sum()
                if valid_total > 0:
                    valid_numeric = num_converted.notnull().sum()
                    if valid_numeric / valid_total >= 0.8:
                        df[col] = num_converted
                        continue
                        
                # Type inference: Attempt datetime
                try:
                    dt_converted = pd.to_datetime(df[col], errors='coerce')
                    valid_dt = dt_converted.notnull().sum()
                    if valid_total > 0 and (valid_dt / valid_total >= 0.8):
                        df[col] = dt_converted
                except Exception:
                    pass
                
        self.data = df
        return df

    def call_groq_imputation_strategy(self, col_name, dtype, sample_values):
        """
        Interacts with Groq AI to decide ambiguity imputation strategy.
        """
        prompt = f"""
Given the following column from a pandas DataFrame, what is the most appropriate imputation strategy for its missing values?

Column Name: '{col_name}'
Data Type: {dtype}
Sample values (ignoring nulls): {sample_values}

Please choose exactly ONE from the following strategies:
- median (if it's naturally continuous numeric, regardless of how it's stored)
- mode (if it comprises classes, categories, states, or strings)
- forward fill (if it's sequential or time-series)

If none fit, you may suggest a specific default value strategy in the format "fill: [value]", for example "fill: Unknown".

Return ONLY the strategy name (e.g. "median", "mode", "forward fill", or "fill: Unknown") and nothing else. No explanation.
"""
        try:
            client = get_groq_client()
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an expert data scientist AI. Output strictly the strategy requested without ANY conversational text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=20,
            )
            return completion.choices[0].message.content.strip().lower()
        except Exception as e:
            print(f"Groq API Error for {col_name}: {e}")
            return "mode"  # Fallback gracefully

    def smart_impute(self, progress_callback=None):
        """
        Task 2 - Smart Imputation Logic
        Applies rules column by column.
        Returns the finalized dataframe and a list of warnings (e.g. >80% missing).
        """
        df = self.data
        warnings_list = []
        total_rows = len(df)
        
        # Determine steps for progress bar
        cols_to_process = df.columns.tolist()
        total_cols = len(cols_to_process)
        
        for idx, col in enumerate(cols_to_process):
            # Calculate missing ratio
            missing_mask = df[col].isnull()
            missing_count = missing_mask.sum()
            missing_ratio = missing_count / total_rows if total_rows > 0 else 0
            
            if missing_count == 0:
                if progress_callback: progress_callback(idx + 1, total_cols, col)
                continue
                
            # Flag if >80% missing
            if missing_ratio > 0.80:
                warnings_list.append(f"⚠️ Column '{col}' has {missing_ratio*100:.1f}% missing values. Left intact without dropping.")
                if progress_callback: progress_callback(idx + 1, total_cols, col)
                continue

            dtype = df[col].dtype
            
            # Simple Rules Override
            if pd.api.types.is_numeric_dtype(df[col]):
                # Median imputation for pure numeric
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                # ffill then bfill for datetime
                df[col] = df[col].ffill().bfill()
            else:
                # String / Categorical / Ambiguous - Use Groq if high unique count or pure ambiguity, otherwise mode
                # Let's get up to 10 valid random samples
                valid_data = df[col].dropna().unique().tolist()
                
                # If very few categories, it's definitively categorical -> mode
                if len(valid_data) <= 5:
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        df[col] = df[col].fillna(mode_val[0])
                    else:
                        df[col] = df[col].fillna("Missing")
                else:
                    # Ambiguous -> call Groq!
                    sample_size = min(10, len(valid_data))
                    samples = random.sample(valid_data, sample_size)
                    
                    strategy = self.call_groq_imputation_strategy(col, str(dtype), samples)
                    
                    if "median" in strategy:
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            df[col] = df[col].fillna(df[col].median())
                        except:
                            # fallback to mode
                            mode_val = df[col].mode()
                            df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else "Missing")
                            
                    elif "forward fill" in strategy or "ffill" in strategy:
                        df[col] = df[col].ffill().bfill()
                        
                    elif "fill:" in strategy:
                        # Extract the targeted fill value
                        val = strategy.split("fill:")[-1].strip()
                        val = val.strip("'\"")
                        df[col] = df[col].fillna(val)
                        
                    else:
                        # Default to mode
                        mode_val = df[col].mode()
                        df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else "Missing")

            if progress_callback: 
                progress_callback(idx + 1, total_cols, col)

        self.data = df
        return df, warnings_list

    def get_cleaned_data(self):
        return self.data
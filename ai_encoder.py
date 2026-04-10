import json
import pandas as pd
from utils import get_groq_client

def ai_encode_dataframe(df: pd.DataFrame, dataset_name: str) -> (pd.DataFrame, str):
    """
    1. Renames columns to full, human-readable names based on dataset context.
    2. Scans for numeric columns that might be categorical codes (<= 15 unique values).
    3. Queries Groq LLM to decode them into human-readable labels and new column names.
    4. Identifies a more descriptive name for the dataset itself.
    Returns (new_df, suggested_dataset_name).
    """
    df_encoded = df.copy()
    candidate_cols = {}
    suggested_name = dataset_name
    
    for col in df_encoded.columns:
        if pd.api.types.is_numeric_dtype(df_encoded[col]):
            n_unique = df_encoded[col].nunique()
            if 1 < n_unique <= 15:
                vals = sorted(df_encoded[col].dropna().unique().tolist())
                candidate_cols[col] = vals
                
    prompt = f"""
    Current Filename/Dataset Name: '{dataset_name}'
    
    TASK 1: COLUMN RENAMING
    All columns: {list(df_encoded.columns)}
    Provide full, human-readable names (e.g., 'cp' -> 'Chest Pain Type').
    
    TASK 2: VALUE DECODING
    Candidate numeric columns and unique values:
    {json.dumps(candidate_cols, indent=2) if candidate_cols else "None."}
    Provide short labels for these coded values (e.g., 0: "Male", 1: "Female").
    
    TASK 3: DATASET IDENTITY
    Based on the column names and the data context, what is the most likely official title/name of this dataset? 
    (e.g., if columns are PatientID, thalach, etc. the name might be "Heart Disease Analysis").
    Be concise (2-4 words).

    Respond ONLY with raw JSON:
    {{
      "suggested_dataset_name": "New Dataset Name",
      "column_renames": {{ "orig": "Full Name" }},
      "value_mappings": {{ "orig": {{ "0": "Label" }} }}
    }}
    """
    
    try:
        client = get_groq_client()
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "You are a data preprocessing expert."},
                      {"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2500,
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```json"): content = content[7:-3]
        elif content.startswith("```"): content = content[3:-3]
            
        result = json.loads(content)
        
        # Suggested Name
        suggested_name = result.get("suggested_dataset_name", dataset_name)
        
        # Value mappings
        mappings = result.get("value_mappings", {})
        for col, col_map in mappings.items():
            if col in df_encoded.columns and col_map:
                try:
                    typed_map = {df_encoded[col].dtype.type(k): v for k, v in col_map.items()}
                    df_encoded[col] = df_encoded[col].map(typed_map).fillna(df_encoded[col])
                except Exception:
                    orig_col_str = df_encoded[col].astype(str).str.replace(r'\.0$', '', regex=True)
                    str_map = {str(k): str(v) for k, v in col_map.items()}
                    df_encoded[col] = orig_col_str.map(str_map).fillna(orig_col_str)
                    
        # Rename columns
        renames = result.get("column_renames", {})
        if renames:
            df_encoded.rename(columns=renames, inplace=True)
            
            # Ensure unique column names to prevent to_json() failures
            if df_encoded.columns.duplicated().any():
                new_cols = []
                seen = set()
                for c in df_encoded.columns:
                    base_name = str(c)
                    new_name = base_name
                    counter = 1
                    while new_name in seen:
                        new_name = f"{base_name}_{counter}"
                        counter += 1
                    seen.add(new_name)
                    new_cols.append(new_name)
                df_encoded.columns = new_cols
                
    except Exception as e:
        print(f"AI Encoding failed: {e}")
        return df, dataset_name
        
    return df_encoded, suggested_name


import json
import pandas as pd
from utils import get_groq_client

def ai_encode_dataframe(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    1. Renames columns to full, human-readable names based on dataset context.
    2. Scans for numeric columns that might be categorical codes (<= 15 unique values).
    3. Queries Groq LLM to decode them into human-readable labels and new column names.
    Returns a new DataFrame with mapped values and renamed columns.
    """
    df_encoded = df.copy()
    candidate_cols = {}
    
    for col in df_encoded.columns:
        if pd.api.types.is_numeric_dtype(df_encoded[col]):
            n_unique = df_encoded[col].nunique()
            # If it has between 2 and 15 unique values, it's likely a categorical code
            if 1 < n_unique <= 15:
                vals = sorted(df_encoded[col].dropna().unique().tolist())
                candidate_cols[col] = vals
                
    # Build prompt
    prompt = f"""
    Dataset Name: '{dataset_name}'
    
    TASK 1: COLUMN RENAMING
    Here are all the columns in this dataset: {list(df_encoded.columns)}
    Based on the dataset name, please provide full, human-readable names for these columns. (e.g., 'cp' -> 'Chest Pain Type', 'thalach' -> 'Max Heart Rate').
    If a column is already clear (like 'Age'), just map it to exactly what it is, maybe capitalized (e.g., 'age' -> 'Age').
    
    TASK 2: VALUE DECODING
    I have numeric columns that are likely categorical encodings (like 0/1 for Gender). Here are the candidate columns and their unique values:
    {json.dumps(candidate_cols, indent=2) if candidate_cols else "No candidate columns for decoding."}
    Please provide short, human-readable labels for these coded values based on the likely context.
    If you are completely unsure about a column, map its numbers to their string versions.

    Respond ONLY with a valid JSON object in this exact format. Do NOT add markdown code blocks (like ```json), do not explain your reasoning. Just return the raw JSON:
    {{
      "column_renames": {{
        "col1_original": "Col1 Readable",
        "col2_original": "Col2 Readable"
      }},
      "value_mappings": {{
        "col1_original": {{
          "0": "Label A",
          "1": "Label B"
        }}
      }}
    }}
    """
    
    try:
        client = get_groq_client()
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2500,
        )
        content = resp.choices[0].message.content.strip()
        
        # Clean up markdown if the model ignored instructions
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
            
        result = json.loads(content)
        
        # 1. Apply value mappings first (using original column names)
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
                    
        # 2. Rename columns
        renames = result.get("column_renames", {})
        if renames:
            df_encoded.rename(columns=renames, inplace=True)
            
        print("Successfully applied AI column renaming and value encoding.")
                
    except Exception as e:
        print(f"AI Encoding failed: {e}")
        return df
        
    return df_encoded

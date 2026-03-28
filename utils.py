"""
utils.py — Helpers for Argus EDA.
chat_response now calls Groq LLaMA with full EDA + LLM-generated context.
"""

import pandas as pd
import time
from groq import Groq

import streamlit as st
import random

def get_groq_client():
    """Return a Groq client initialized with a random API key from secrets."""
    try:
        keys = st.secrets.get("GROQ_API_KEYS", [])
        if keys:
            return Groq(api_key=random.choice(keys))
            
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return Groq(api_key=key)
    except Exception:
        pass
    
    import os
    env_key = os.environ.get("GROQ_API_KEY")
    if env_key:
        return Groq(api_key=env_key)

    # Fallback if secrets and explicit env check missing
    # Groq SDK will automatically look for GROQ_API_KEY in the environment
    return Groq()

# We will let other files call `get_groq_client()` directly instead of using a global `_client`
# But for utils internally, we can define one for `chat_response` if needed, although
# calling it fresh avoids reuse limits or rate limit tracking on a single IP/client if Groq tracks it.
MAX_CONTEXT_CHARS = 3000   # trim llm_context to stay within token limits


# ─── File Processing ──────────────────────────────────────────────────────────

def load_file(uploaded_file) -> pd.DataFrame:
    """Read an uploaded Excel or CSV file into a DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)


def run_eda(df: pd.DataFrame) -> dict:
    """Return a summary dict for display on the frontend."""
    time.sleep(0.3)
    missing = int(df.isnull().sum().sum())
    return {
        "rows":           df.shape[0],
        "columns":        df.shape[1],
        "missing_values": missing,
        "column_names":   list(df.columns),
        "dtypes":         df.dtypes.astype(str).to_dict(),
        "preview":        df.head(5).to_dict(orient="records"),
    }


# ─── Chat ─────────────────────────────────────────────────────────────────────

def chat_response(question: str,
                  eda_context: dict,
                  llm_context: list[str] | None = None) -> str:
    """
    Send the user question to Groq LLaMA with:
      - EDA metadata (rows, cols, dtypes, missing)
      - Accumulated LLM-generated analysis (bivariate descriptions, summaries, etc.)
    Returns the LLM answer as a markdown string.
    """
    llm_context = llm_context or []

    # ── Build system prompt ──────────────────────────────────────────────────
    cols_str   = ", ".join(eda_context.get("column_names", []))
    dtypes_str = "\n".join(
        f"  {c}: {t}" for c, t in list(eda_context.get("dtypes", {}).items())[:20]
    )

    # Trim accumulated context to avoid token overflow
    context_blob = "\n".join(llm_context)
    if len(context_blob) > MAX_CONTEXT_CHARS:
        context_blob = context_blob[-MAX_CONTEXT_CHARS:]

    system_msg = f"""You are Argus, an expert AI data analyst embedded in an EDA tool.
Answer the user's question about their dataset clearly and concisely. Use markdown formatting.

=== DATASET METADATA ===
Rows: {eda_context.get('rows', '?')}
Columns: {eda_context.get('columns', '?')}
Missing values: {eda_context.get('missing_values', '?')}
Column names: {cols_str}

Data types:
{dtypes_str}

=== AI-GENERATED ANALYSIS (from previous analyses) ===
{context_blob if context_blob else 'No analysis results available yet — analyses have not been run.'}
========================

Answer in a friendly, insightful way. If the user asks about something not in the data, say so.
Keep answers under 200 words unless more detail is genuinely needed."""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": question},
    ]

    try:
        client = get_groq_client()
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.5,
            max_tokens=400,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        # Graceful fallback
        return (
            f"⚠️ Could not reach Argus AI right now (`{e}`).  \n\n"
            f"Your dataset has **{eda_context.get('rows','?')} rows** and "
            f"**{eda_context.get('columns','?')} columns**."
        )

"""
ratings.py — Argus rating storage and retrieval.
Saves user star-ratings + feedback to MongoDB argus_db.ratings.
Falls back gracefully when MONGO_URI is not configured.
"""

import streamlit as st
from datetime import datetime


@st.cache_resource
def _get_ratings_collection():
    uri = st.secrets.get("MONGO_URI", "")
    if not uri:
        return None
    try:
        from pymongo import MongoClient
        client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        return client["argus_db"]["ratings"]
    except Exception as e:
        print("MongoDB Ratings Error:", e)
        return None


def save_rating(email: str, name: str, stars: int, feedback: str = ""):
    """Store a rating in MongoDB. Silently skips if DB unavailable."""
    col = _get_ratings_collection()
    if col is None:
        return
    try:
        # Upsert: one rating per user email (update if they rate again)
        col.update_one(
            {"email": email},
            {"$set": {
                "name": name,
                "stars": stars,
                "feedback": feedback,
                "timestamp": datetime.utcnow()
            }},
            upsert=True
        )
    except Exception as e:
        print("Rating save error:", e)


def get_avg_rating() -> tuple[float, int]:
    """
    Returns (average_stars, total_count).
    Falls back to (4.5, 0) if DB unavailable.
    """
    col = _get_ratings_collection()
    if col is None:
        return 4.5, 0
    try:
        pipeline = [
            {"$group": {
                "_id": None,
                "avg": {"$avg": "$stars"},
                "count": {"$sum": 1}
            }}
        ]
        result = list(col.aggregate(pipeline))
        if result:
            avg = round(result[0]["avg"], 1)
            count = result[0]["count"]
            return avg, count
        return 4.5, 0
    except Exception as e:
        print("Rating fetch error:", e)
        return 4.5, 0

import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# Lazy-load model (CRITICAL)
# =============================
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

# =============================
# Sample Review Dataset
# =============================
reviews_dataset: List[Dict[str, str]] = [
    {"review_text": "I love this product! It's exactly what I needed.", "label": "positive"},
    {"review_text": "Terrible experience. Completely disappointed.", "label": "negative"},
    {"review_text": "It's okay, nothing special.", "label": "neutral"},
    {"review_text": "Fantastic quality and great value!", "label": "positive"},
    {"review_text": "Waste of money. Do not recommend.", "label": "negative"},
]

# =============================
# Build corpus embeddings once
# =============================
_corpus_embeddings = None

def get_corpus_embeddings():
    global _corpus_embeddings
    if _corpus_embeddings is None:
        model = get_model()
        _corpus_embeddings = model.encode(
            [r["review_text"] for r in reviews_dataset],
            convert_to_numpy=True
        )
    return _corpus_embeddings

# =============================
# Retrieve Similar Reviews
# =============================
def get_similar_reviews(text: str, top_k: int = 3):
    model = get_model()
    corpus_embeddings = get_corpus_embeddings()

    query_embedding = model.encode([text], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]

    top_indices = similarities.argsort()[-top_k:][::-1]
    return [reviews_dataset[i] for i in top_indices]

# =============================
# Predict Sentiment (RAG-style)
# =============================
def predict_sentiment(retrieved_reviews: List[Dict[str, str]]) -> str:
    labels = [r["label"] for r in retrieved_reviews]
    return max(set(labels), key=labels.count)

# =============================
# Main Pipeline
# =============================
def analyze_input_review(text: str):
    retrieved = get_similar_reviews(text)
    sentiment = predict_sentiment(retrieved)

    return {
        "input_review": text,
        "predicted_sentiment": sentiment,
        "similar_reviews": retrieved
    }

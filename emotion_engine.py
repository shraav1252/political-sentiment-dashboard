import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        return_all_scores=True,
        truncation=True,
        max_length=128,
    )

def predict_emotion(text):
    if not isinstance(text, str) or text.strip() == "":
        return "neutral"
    classifier = load_emotion_model()
    result = classifier(text[:512])
    scores = result[0] if isinstance(result[0], list) else result
    return max(scores, key=lambda x: x["score"])["label"]

def predict_emotion_batch(texts, batch_size=64):
    """Run emotion detection in batches for large datasets."""
    classifier = load_emotion_model()
    cleaned = [t[:512] if isinstance(t, str) and t.strip() else "" for t in texts]
    results = []
    for i in range(0, len(cleaned), batch_size):
        batch = cleaned[i:i + batch_size]
        safe_batch = [t if t else "neutral" for t in batch]
        outputs = classifier(safe_batch)
        for j, out in enumerate(outputs):
            if not cleaned[i + j]:
                results.append("neutral")
            else:
                scores = out if isinstance(out, list) else [out]
                results.append(max(scores, key=lambda x: x["score"])["label"])
    return results

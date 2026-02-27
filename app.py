import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import numpy as np

# ---- Page Config ----
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)

# ---- Load Model ----
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained('./saved_bert_model')
    model = BertForSequenceClassification.from_pretrained('./saved_bert_model')
    le = LabelEncoder()
    le.classes_ = np.load('./saved_bert_model/le_classes.npy', allow_pickle=True)
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    return tokenizer, model, le, device

tokenizer, model, le, device = load_model()

# ---- Predict Function ----
def predict(text):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        max_length=64,
        padding='max_length',
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        pred = outputs.logits.argmax(dim=1).item()
        probs = torch.softmax(outputs.logits, dim=1)[0].tolist()

    return le.classes_[pred], probs

# ---- UI ----
st.title("💬 Sentiment Analysis App")
st.markdown("Built with **BERT** | NLP Portfolio Project by Manoj S Annigeri")
st.divider()

# Single prediction
st.subheader("🔍 Analyze a Sentence")
user_input = st.text_area("Type any sentence here:", placeholder="e.g. The flight was amazing!")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        sentiment, probs = predict(user_input)
        color = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}
        st.markdown(f"### Result: {color[sentiment]} **{sentiment.upper()}**")
        st.divider()
        st.subheader("📊 Confidence Scores")
        col1, col2, col3 = st.columns(3)
        col1.metric("Negative", f"{round(probs[0]*100, 1)}%")
        col2.metric("Neutral", f"{round(probs[1]*100, 1)}%")
        col3.metric("Positive", f"{round(probs[2]*100, 1)}%")

st.divider()

# Batch prediction
st.subheader("📋 Analyze Multiple Sentences")
batch_input = st.text_area(
    "Enter multiple sentences (one per line):",
    placeholder="Flight was great!\nTerrible service.\nIt was okay."
)

if st.button("Analyze All"):
    if batch_input.strip() == "":
        st.warning("Please enter some sentences!")
    else:
        sentences = [s.strip() for s in batch_input.strip().split('\n') if s.strip()]
        results = []
        for sentence in sentences:
            sentiment, probs = predict(sentence)
            results.append({
                "Sentence": sentence,
                "Sentiment": sentiment,
                "Confidence": f"{round(max(probs)*100, 1)}%"
            })
        st.dataframe(pd.DataFrame(results), use_container_width=True)

st.divider()
st.caption("🎓 Master's Portfolio Project | Sentiment Analysis NLP Pipeline | Manoj S Annigeri")
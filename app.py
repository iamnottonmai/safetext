import streamlit as st
import torch
import os
import gdown
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "trained_model"
MODEL_FILE = os.path.join(MODEL_DIR, "model.safetensors")

GDRIVE_URL = "https://drive.google.com/uc?id=1LR421nKcA3S7QkIuF4yY6AVB45BistEl"

LABELS = {
    0: "No Risk",
    1: "Insult",
    2: "Defamation"
}

@st.cache_resource
def load_model():
    # download model if missing
    if not os.path.exists(MODEL_FILE):
        os.makedirs(MODEL_DIR, exist_ok=True)
        with st.spinner("Downloading model weights..."):
            gdown.download(GDRIVE_URL, MODEL_FILE, quiet=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    return tokenizer, model

tokenizer, model = load_model()

st.title("SafeText: Thai Legal Risk Classification")

text = st.text_area("Enter Thai text")

if st.button("Classify") and text.strip():
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred = probs.argmax().item()

    st.success(f"Prediction: {LABELS[pred]}")
    st.write("Confidence:")
    for i, p in enumerate(probs):
        st.write(f"{LABELS[i]}: {p:.2%}")

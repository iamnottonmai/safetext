import streamlit as st
import torch
import os
import zipfile
import gdown
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "trained_model"
ZIP_PATH = "safetext_model.zip"

GDRIVE_ZIP_URL = "https://drive.google.com/uc?id=1cyjqbVRAhOAogoUWY1_zhby9uy9VdSPR"

LABELS = {
    0: "ปลอดภัย",
    1: "สุ่มเสี่ยง"
}

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_DIR):
        with st.spinner("Downloading model..."):
            gdown.download(GDRIVE_ZIP_URL, ZIP_PATH, quiet=False)

        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(".")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    return tokenizer, model


tokenizer, model = load_model()

st.title("SafeText 🇹🇭")
st.caption("Thai Defamation & Insult Risk Analyzer")

text = st.text_area("Enter Thai text")
context = st.selectbox(
    "Context",
    ["public_post", "private_dm", "email", "letter"]
)

if st.button("Analyze") and text.strip():
    input_text = f"[CONTEXT] {context} [TEXT] {text}"

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred = probs.argmax().item()

    st.success(f"Prediction: **{LABELS[pred]}**")

    st.write("Confidence:")
    for i, p in enumerate(probs):
        st.write(f"- {LABELS[i]}: {p:.2%}")

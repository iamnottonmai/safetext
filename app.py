import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = {
    0: "No Risk",
    1: "Insult",
    2: "Defamation"
}

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("trained_model")
    model = AutoModelForSequenceClassification.from_pretrained("trained_model")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.title("Thai Text Risk Classification")

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

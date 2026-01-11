import streamlit as st
import torch
import os
import zipfile
import gdown
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------- CONFIG ----------------

MODEL_DIR = "model_runtime"
ZIP_PATH = "model_runtime.zip"

GDRIVE_ZIP_URL = "https://drive.google.com/uc?id=1cyjqbVRAhOAogoUWY1_zhby9uy9VdSPR"

LABELS = {
    0: "ปลอดภัย",
    1: "สุ่มเสี่ยง"
}

# คำที่ถือว่า "เสี่ยงทางภาษาแบบไม่ต้องตีความ"
HIGH_CERTAINTY_ABUSE = {
    "อีดอก", "อีเหี้ย", "อีสัตว์", "อีควาย",
    "อีกระหรี่", "ไอ้สัตว์", "ไอ้ควาย"
}

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_DIR):
        with st.spinner("Downloading model..."):
            gdown.download(GDRIVE_ZIP_URL, ZIP_PATH, quiet=False)
            with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
                zip_ref.extractall(MODEL_DIR)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        use_fast=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    return tokenizer, model


tokenizer, model = load_model()

# ---------------- LOGIC ----------------

def detect_linguistic_risk(text: str):
    matched = [w for w in HIGH_CERTAINTY_ABUSE if w in text]
    return len(matched) > 0, matched


def final_decision(linguistic_risk, legal_label):
    if linguistic_risk and legal_label == "สุ่มเสี่ยง":
        return "สุ่มเสี่ยงสูง", "คำหยาบ + มีลักษณะเข้าข่ายกฎหมาย"

    if linguistic_risk and legal_label == "ปลอดภัย":
        return "คำไม่เหมาะสม", "พบคำหยาบ แต่ไม่เข้าข่ายความผิดทางกฎหมาย"

    if not linguistic_risk and legal_label == "สุ่มเสี่ยง":
        return "สุ่มเสี่ยง", "ไม่ใช้คำหยาบ แต่มีความเสี่ยงทางกฎหมาย"

    return "ปลอดภัย", "ไม่พบความเสี่ยง"

# ---------------- UI ----------------

st.title("SafeText 🇹🇭")
st.caption("Thai Defamation & Insult Risk Analyzer")

text = st.text_area("ข้อความ")
context = st.selectbox(
    "บริบท",
    ["public_post", "private_dm", "email", "letter"]
)

if st.button("วิเคราะห์") and text.strip():

    # 1️⃣ Linguistic signal
    linguistic_risk, matched_terms = detect_linguistic_risk(text)

    # 2️⃣ Model inference
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
        legal_label = LABELS[pred]

    # 3️⃣ Final decision
    verdict, explanation = final_decision(linguistic_risk, legal_label)

    # ---------------- OUTPUT ----------------

    if verdict in ["สุ่มเสี่ยง", "สุ่มเสี่ยงสูง"]:
        st.error(f"ผลการวิเคราะห์: **{verdict}**")
    elif verdict == "คำไม่เหมาะสม":
        st.warning(f"ผลการวิเคราะห์: **{verdict}**")
    else:
        st.success(f"ผลการวิเคราะห์: **{verdict}**")

    st.write(explanation)

    if linguistic_risk:
        st.write("🔎 ตรวจพบคำไม่เหมาะสม:", ", ".join(matched_terms))

    st.write("📊 ความมั่นใจของโมเดล:")
    for i, p in enumerate(probs):
        st.write(f"- {LABELS[i]}: {p:.2%}")

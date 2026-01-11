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

# thresholds (ปรับได้)
ABUSIVE_THRESHOLD = 0.20     # linguistic signal
LEGAL_THRESHOLD = 0.50       # legal risk

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

# ---------------- DECISION LOGIC ----------------

def analyze(probs):
    """
    probs[1] = P(สุ่มเสี่ยง)
    """
    p_risk = probs[1]

    linguistic_risk = p_risk >= ABUSIVE_THRESHOLD
    legal_risk = p_risk >= LEGAL_THRESHOLD

    if linguistic_risk and legal_risk:
        return "สุ่มเสี่ยงสูง", (
            "ข้อความมีลักษณะก้าวร้าวและอาจเข้าข่ายความผิดทางกฎหมาย"
        )

    if linguistic_risk and not legal_risk:
        return "คำไม่เหมาะสม", (
            "ข้อความมีลักษณะก้าวร้าวหรือดูหมิ่น "
            "แต่ยังไม่พบความเสี่ยงทางกฎหมายในบริบทนี้"
        )

    if not linguistic_risk and legal_risk:
        return "สุ่มเสี่ยง", (
            "ข้อความอาจกระทบต่อชื่อเสียงหรือสิทธิของผู้อื่น "
            "แม้ไม่ใช้ถ้อยคำรุนแรง"
        )

    return "ปลอดภัย", "ไม่พบความเสี่ยงจากข้อความนี้"

# ---------------- UI ----------------

st.title("SafeText 🇹🇭")
st.caption("Thai Defamation & Insult Risk Analyzer")

text = st.text_area("ข้อความ")
context = st.selectbox(
    "บริบท",
    ["public_post", "private_dm", "email", "letter"]
)

if st.button("วิเคราะห์") and text.strip():

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
        probs = torch.softmax(outputs.logits, dim=1)[0].tolist()

    verdict, explanation = analyze(probs)

    # -------- OUTPUT --------

    if verdict in ["สุ่มเสี่ยง", "สุ่มเสี่ยงสูง"]:
        st.error(f"ผลการวิเคราะห์: **{verdict}**")
    elif verdict == "คำไม่เหมาะสม":
        st.warning(f"ผลการวิเคราะห์: **{verdict}**")
    else:
        st.success(f"ผลการวิเคราะห์: **{verdict}**")

    st.write(explanation)

    st.write("📊 ความมั่นใจของโมเดล:")
    st.write(f"- ปลอดภัย: {probs[0]:.2%}")
    st.write(f"- สุ่มเสี่ยง: {probs[1]:.2%}")

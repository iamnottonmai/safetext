import streamlit as st
import torch
import os
import gdown
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =====================
# CONFIG
# =====================
MODEL_DIR = "trained_model"

# If you still download from Drive, this should point to a FOLDER zip
# (If you already bundled tokenizer + model, remove gdown logic)
GDRIVE_URL = "https://drive.google.com/uc?id=YOUR_DRIVE_ID"

LABELS = {
    0: "ปลอดภัย",
    1: "สุ่มเสี่ยง"
}

RISK_THRESHOLD = 0.4  # recommended from your evaluation

# =====================
# LOAD MODEL
# =====================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_DIR):
        with st.spinner("Downloading model..."):
            gdown.download_folder(GDRIVE_URL, output=MODEL_DIR, quiet=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    return tokenizer, model

tokenizer, model = load_model()

# =====================
# UI
# =====================
st.title("SafeText — Thai Legal Risk Analyzer")
st.caption("⚠️ การประเมินโดย AI ไม่ใช่คำแนะนำทางกฎหมาย")

context = st.selectbox(
    "Select context",
    ["public_post", "private_dm", "email", "letter"]
)

text = st.text_area(
    "Enter Thai text",
    placeholder="พิมพ์ข้อความภาษาไทยที่ต้องการตรวจสอบ…"
)

# =====================
# INFERENCE
# =====================
if st.button("Analyze") and text.strip():

    input_text = f"[CONTEXT] {context} [TEXT] {text}"

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    risk_prob = probs[1].item()
    pred_label = 1 if risk_prob >= RISK_THRESHOLD else 0

    # =====================
    # OUTPUT
    # =====================
    if pred_label == 1:
        st.error("⚠️ ผลการประเมิน: ข้อความมีความ **สุ่มเสี่ยง**")
    else:
        st.success("✅ ผลการประเมิน: ข้อความ **ปลอดภัย**")

    st.write("### Confidence")
    st.write(f"ปลอดภัย: {probs[0]:.2%}")
    st.write(f"สุ่มเสี่ยง: {probs[1]:.2%}")

    st.caption(
        "หมายเหตุ: ระบบนี้เป็นการประเมินความเสี่ยงเบื้องต้น "
        "ไม่สามารถใช้แทนคำปรึกษาทางกฎหมายได้"
    )

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

# optional heuristic list (NOT decision)
PROFANITY_HINTS = [
    "อี", "ไอ้", "เหี้ย", "ควาย", "สัตว์", "ดอก"
]

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_DIR):
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

# ---------------- UI ----------------
st.title("SafeText 🇹🇭")
st.caption("Thai Legal Risk Analyzer")

text = st.text_area("ข้อความที่ต้องการตรวจสอบ")
context = st.selectbox(
    "บริบทการสื่อสาร",
    ["public_post", "private_dm", "email", "letter"]
)

if st.button("วิเคราะห์") and text.strip():

    # ---------- BOX 1: LANGUAGE NOTE ----------
    found_hints = [w for w in PROFANITY_HINTS if w in text]

    st.subheader("🗣️ การใช้ถ้อยคำ")
    if found_hints:
        st.warning(
            "ตรวจพบถ้อยคำที่อาจไม่เหมาะสม "
            "(ยังไม่ถือว่าเป็นความเสี่ยงทางกฎหมาย)"
        )
    else:
        st.success("ไม่พบถ้อยคำรุนแรงชัดเจน")

    # ---------- BOX 2: AI LEGAL RISK ----------
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

    st.subheader("⚖️ ผลการวิเคราะห์ความเสี่ยง")
    st.info(f"ผลลัพธ์: **{LABELS[pred]}**")

    st.write("ความเชื่อมั่นของโมเดล:")
    for i, p in enumerate(probs):
        st.write(f"- {LABELS[i]}: {p:.2%}")

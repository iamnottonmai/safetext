import streamlit as st
import torch
import os
import zipfile
import gdown
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------- Runtime paths ----------------
MODEL_DIR = "model_runtime"
ZIP_PATH = "model_runtime.zip"

# gdown-compatible Google Drive link (ZIP that contains full model folder)
GDRIVE_ZIP_URL = "https://drive.google.com/uc?id=1cyjqbVRAhOAogoUWY1_zhby9uy9VdSPR"

# Model labels (must match training)
LABELS = {
    0: "ปลอดภัย",
    1: "สุ่มเสี่ยง"
}

# ---------------- Load model ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_DIR):
        with st.spinner("กำลังโหลดโมเดลจาก Google Drive..."):
            gdown.download(GDRIVE_ZIP_URL, ZIP_PATH, quiet=False)

        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(MODEL_DIR)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        use_fast=False  # จำเป็นสำหรับ WangchanBERTa / CamemBERT-based
    )

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    return tokenizer, model


tokenizer, model = load_model()

# ---------------- UI ----------------

st.title("SafeText 🇹🇭")
st.caption("ระบบวิเคราะห์ความเสี่ยงทางกฎหมายจากข้อความภาษาไทย")

text = st.text_area("ข้อความที่ต้องการวิเคราะห์")
context = st.selectbox(
    "บริบทของข้อความ",
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
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred = probs.argmax().item()

    # -------- Output --------
    st.subheader("ผลการวิเคราะห์ความเสี่ยงทางกฎหมาย")
    st.write(f"**ผลลัพธ์:** {LABELS[pred]}")
    st.write(f"**ระดับความเชื่อมั่น:** {probs[pred]:.2%}")

    st.caption(
        "หมายเหตุ: การใช้ถ้อยคำไม่สุภาพเพียงอย่างเดียว "
        "ไม่ถือเป็นความผิดตามกฎหมาย "
        "ระบบประเมินจากลักษณะการพาดพิง บุคคล และบริบทของข้อความ"
    )

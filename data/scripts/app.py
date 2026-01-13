# app.py
import os
import time
import streamlit as st
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel

from ocr_core import process_image

# ============================================================
# AJUSTES VISUAIS
# ============================================================

IMAGE_DISPLAY_WIDTH = 740

FONT_SIZE_BASE = 25
FONT_SIZE_CODE = 25
FONT_SIZE_TITLE = 30

LOGO = "/home/rogerio/PycharmProjects/deepseek-coordinates-ocr/data/logo_azul.jpeg"

st.markdown(
    f"""
    <style>
    html, body, [class*="css"] {{
        font-size: {FONT_SIZE_BASE}px;
    }}
    code {{
        font-size: {FONT_SIZE_CODE}px;
    }}
    h1 {{
        font-size: {FONT_SIZE_TITLE}px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# CONFIGURAÇÕES FIXAS
# ============================================================

IMAGES_DIR = "/home/rogerio/Imagens/fotos_para_treinamento"
OUTPUT_DIR = "/home/rogerio/PycharmProjects/deepseek-coordinates-ocr/data/testes/output"
OUTPUT_TXT = os.path.join(OUTPUT_DIR, "resultado.txt")

MODEL_NAME = "deepseek-ai/DeepSeek-OCR"

BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
PROMPT = "<image>\nFree OCR."

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# STREAMLIT SETUP
# ============================================================

st.set_page_config(
    page_title="Extração de Coordenadas",
    layout="wide"
)

# LOGO + TÍTULO
col_logo, col_title = st.columns([1, 4])
with col_logo:
    if os.path.isfile(LOGO):
        st.image(LOGO, width=380)
with col_title:
    st.title("Extração de Coordenadas - OCR")

# ============================================================
# CARREGAR MODELO
# ============================================================

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "3.1GB", "cpu": "16GB"},
        low_cpu_mem_usage=True,
        use_safetensors=True
    )

    model.eval()
    return tokenizer, model

with st.spinner("Carregando modelo IA - OCR..."):
    tokenizer, model = load_model()

# ============================================================
# VALIDAR IMAGENS
# ============================================================

if not os.path.isdir(IMAGES_DIR):
    st.error(f"Pasta não encontrada:\n{IMAGES_DIR}")
    st.stop()

image_names = sorted(
    [
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    ],
    key=lambda x: os.path.getsize(os.path.join(IMAGES_DIR, x))
)

st.info(f"{len(image_names)} imagens encontradas.")

# ============================================================
# CONTROLE
# ============================================================

if "running" not in st.session_state:
    st.session_state.running = False

if st.button("Iniciar processamento"):
    st.session_state.running = True
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("")

# ============================================================
# PROCESSAMENTO
# ============================================================

if st.session_state.running:
    progress = st.progress(0.0)
    col_img, col_info = st.columns([1, 1])

    image_placeholder = col_img.empty()
    info_placeholder = col_info.empty()

    times = []
    total = len(image_names)

    for idx, image_name in enumerate(image_names, 1):
        image_path = os.path.join(IMAGES_DIR, image_name)

        start_img = time.perf_counter()

        coord = process_image(
            model,
            tokenizer,
            image_path,
            PROMPT,
            BASE_SIZE,
            IMAGE_SIZE,
            CROP_MODE,
            output_path=OUTPUT_DIR
        )

        elapsed = time.perf_counter() - start_img
        times.append(elapsed)

        avg = sum(times) / len(times)
        eta = avg * (total - idx)

        line = f"{image_name}|{coord}"

        with open(OUTPUT_TXT, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        # FRONTEND MOSTRA IMAGEM COLORIDA
        image_placeholder.image(
            Image.open(image_path).convert("RGB"),
            caption=image_name,
            width=IMAGE_DISPLAY_WIDTH
        )

        with info_placeholder.container():
            st.subheader(f"Progresso: {idx}/{total}")
            progress.progress(idx / total)

            st.code(line)
            st.write(f"Tempo da imagem: {elapsed:.1f} s")
            st.write(f"Tempo médio: {avg:.1f} s")
            st.write(f"Estimativa restante: {eta/60:.1f} min")

        time.sleep(0.05)

    st.success("Processamento finalizado.")
    st.session_state.running = False
    st.info(f"Resultado salvo em:\n{OUTPUT_TXT}")
    del model
    del tokenizer
    torch.cuda.empty_cache()
    st.info("GPU liberada")

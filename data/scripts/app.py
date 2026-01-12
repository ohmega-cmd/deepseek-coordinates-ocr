# app.py
import os
import time
import streamlit as st
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel

from ocr_core import process_image

# ============================================================
# ESTILO GLOBAL (TAMANHO DE FONTE)
# ============================================================

IMAGE_DISPLAY_WIDTH = 640   # largura da imagem em pixels

FONT_SIZE_BASE = 20
FONT_SIZE_CODE = 20
FONT_SIZE_TITLE = 24

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

IMAGES_DIR = "/home/rogerio/PycharmProjects/PythonProject3/DeepSeek-OCR/data/scripts/anna/CFTV-VISTORIAS"
OUTPUT_DIR = "/home/rogerio/PycharmProjects/deepseek-coordinates-ocr/data/testes/output"
OUTPUT_TXT = os.path.join(OUTPUT_DIR, "resultado_CFTV.txt")

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

st.title("Extração de Coordenadas - OCR")

# ============================================================
# CARREGAR MODELO (UMA VEZ)
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

with st.spinner("Carregando modelo DeepSeek-OCR..."):
    tokenizer, model = load_model()

# ============================================================
# VALIDAR PASTA DE IMAGENS
# ============================================================

if not os.path.isdir(IMAGES_DIR):
    st.error(f"Pasta de imagens não encontrada:\n{IMAGES_DIR}")
    st.stop()

image_names = sorted(
    [
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    ],
    key=lambda x: os.path.getsize(os.path.join(IMAGES_DIR, x))
)

if not image_names:
    st.warning("Nenhuma imagem encontrada na pasta.")
    st.stop()

st.info(f"{len(image_names)} imagens encontradas.")

# ============================================================
# CONTROLE DE EXECUÇÃO
# ============================================================

if "running" not in st.session_state:
    st.session_state.running = False

if st.button("Iniciar processamento"):
    st.session_state.running = True

    # limpa o arquivo ANTES de começar
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

        # ============================
        # SALVAR TXT (INCREMENTAL)
        # ============================
        with open(OUTPUT_TXT, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        # ESQUERDA – SUBSTITUI A IMAGEM
        image_placeholder.image(
            Image.open(image_path),
            caption=image_name,
            width=IMAGE_DISPLAY_WIDTH
        )

        # DIREITA – INFO
        with info_placeholder.container():
            st.subheader(f"Progresso: {idx}/{total}")
            progress.progress(idx / total)

            st.write("Resultado:")
            st.code(line)

            st.write(f"Tempo da imagem: {elapsed:.1f} s")
            st.write(f"Tempo médio: {avg:.1f} s")
            st.write(f"Estimativa restante: {eta/60:.1f} min")

        time.sleep(0.05)

    st.success("Processamento finalizado.")
    st.session_state.running = False

    st.info(f"Resultado salvo em:\n{OUTPUT_TXT}")

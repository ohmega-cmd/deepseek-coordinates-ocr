# ocr_core.py
import re
import io
import sys
import os
import tempfile
import torch
from PIL import Image

# ============================================================
# REGEX + NORMALIZAÇÃO
# ============================================================

coord_pattern = re.compile(
    r'(\d{1,2}[.,]\d+)\s*([NS])\s*(\d{1,3}[.,]\d+)\s*([WE])'
)

def normalize_text(text: str) -> str:
    text = text.upper()
    for a, b in {
        "O": "0",
        "I": "1",
        "L": "1",
        ",": ".",
        "°": "",
        "º": "",
    }.items():
        text = text.replace(a, b)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()

# ============================================================
# PROCESSAR UMA IMAGEM
# ============================================================

def process_image(
    model,
    tokenizer,
    image_path: str,
    prompt: str,
    base_size: int,
    image_size: int,
    crop_mode: bool,
    output_path: str
) -> str:
    """
    OCR usa imagem PRETO E BRANCO salva temporariamente.
    Frontend continua usando imagem colorida.
    """

    # --------------------------------------------------------
    # CRIAR IMAGEM BW TEMPORÁRIA
    # --------------------------------------------------------
    img_color = Image.open(image_path).convert("RGB")
    img_bw = img_color.convert("L")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        img_bw.save(tmp_path)

    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer

    try:
        with torch.no_grad():
            res = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=tmp_path,  # ✅ path real
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                save_results=False,
                output_path=output_path
            )
    finally:
        sys.stdout = old_stdout
        try:
            os.remove(tmp_path)  # 🧹 limpa arquivo temporário
        except OSError:
            pass

    raw = buffer.getvalue().strip() or str(res)
    text = normalize_text(raw)
    match = coord_pattern.search(text)

    if match:
        lat = f"{match.group(1)}{match.group(2)}"
        lon = f"{match.group(3)}{match.group(4)}"
        return f"{lat} {lon}"

    return "NO_COORD_FOUND"

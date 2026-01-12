import os
import time
import torch
from transformers import AutoTokenizer, AutoModel

from ocr_core import process_image

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ============================================================
# 1) CAMINHOS FIXOS
# ============================================================

IMAGES_DIR = r"/home/rogerio/PycharmProjects/deepseek-coordinates-ocr/data/testes/imgs"
OUTPUT_DIR = r"/home/rogerio/PycharmProjects/deepseek-coordinates-ocr/data/testes/output"
OUTPUT_TXT = os.path.join(OUTPUT_DIR, "resultado.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.isdir(IMAGES_DIR):
    raise FileNotFoundError(f"Pasta não encontrada: {IMAGES_DIR}")

# ============================================================
# 2) CONFIGURAÇÕES
# ============================================================

VALID_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
MODEL_NAME = "deepseek-ai/DeepSeek-OCR"

BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
PROMPT = "<image>\nFree OCR."

# ============================================================
# 3) GPU + OFFLOAD
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("❌ CUDA não disponível")

print("✅ GPU:", torch.cuda.get_device_name(0))
print(
    "VRAM:",
    round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    "GB"
)

print("🔄 Carregando DeepSeek-OCR...")

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

# ============================================================
# 4) IMAGENS (ordenadas por tamanho)
# ============================================================

image_names = sorted(
    [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(VALID_EXT)],
    key=lambda x: os.path.getsize(os.path.join(IMAGES_DIR, x))
)

print(f"📷 {len(image_names)} imagens encontradas.")

# ============================================================
# 5) OCR EM LOTE COM TEMPO
# ============================================================

start_total = time.perf_counter()
times = []
lines = []

for idx, image_name in enumerate(image_names, 1):
    image_path = os.path.join(IMAGES_DIR, image_name)
    print(f"[{idx}/{len(image_names)}] Processando {image_name}...")

    start_img = time.perf_counter()

    coord = process_image(
        model,
        tokenizer,
        image_path,
        PROMPT,
        BASE_SIZE,
        IMAGE_SIZE,
        CROP_MODE
    )

    elapsed = time.perf_counter() - start_img
    times.append(elapsed)

    avg = sum(times) / len(times)
    remaining = avg * (len(image_names) - idx)

    line = f"{image_name}|{coord}"

    print("   ", line)
    print(
        f"    ⏱️ {elapsed:.1f}s | "
        f"média {avg:.1f}s | "
        f"Estimativa para conclusão {remaining/60:.1f} min"
    )

    lines.append(line)

    if idx % 3 == 0:
        torch.cuda.empty_cache()

total_time = time.perf_counter() - start_total

# ============================================================
# 6) SALVAR RESULTADO
# ============================================================

with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    for l in lines:
        f.write(l + "\n")

print("\n⏱️ TEMPO TOTAL:", total_time / 60, "min")
print("📊 IMAGENS:", len(image_names))
print("⏱️ MÉDIA:", total_time / len(image_names), "s")

# ============================================================
# 7) LIMPEZA
# ============================================================

del model
del tokenizer
torch.cuda.empty_cache()
print("🧹 GPU liberada")

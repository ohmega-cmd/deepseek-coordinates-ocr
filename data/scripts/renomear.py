import os

# ============================================================
# CONFIGURAÇÃO
# ============================================================

IMAGES_DIR = "/home/rogerio/Imagens/fotos_para_treinamento"

VALID_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
PREFIX = "img_"

# ============================================================
# COLETAR IMAGENS
# ============================================================

files = sorted(
    f for f in os.listdir(IMAGES_DIR)
    if f.lower().endswith(VALID_EXT)
)

if not files:
    raise RuntimeError("Nenhuma imagem encontrada.")

# ============================================================
# RENOMEAR COM SEGURANÇA
# ============================================================

for idx, filename in enumerate(files, 1):
    ext = os.path.splitext(filename)[1].lower()
    new_name = f"{PREFIX}{idx:03d}{ext}"

    old_path = os.path.join(IMAGES_DIR, filename)
    new_path = os.path.join(IMAGES_DIR, new_name)

    if old_path == new_path:
        continue

    if os.path.exists(new_path):
        raise FileExistsError(
            f"Arquivo já existe: {new_name}\n"
            "Abortando para evitar sobrescrita."
        )

    os.rename(old_path, new_path)
    print(f"{filename} -> {new_name}")

print("Renomeação concluída.")

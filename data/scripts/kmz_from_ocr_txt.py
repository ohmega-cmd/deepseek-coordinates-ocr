import zipfile
import xml.etree.ElementTree as ET
import math
import pandas as pd
import os
import re

# ============================================================
# CONFIGURAÇÃO
# ============================================================

TXT_OCR = "/home/rogerio/PycharmProjects/deepseek-coordinates-ocr/data/resultado_SAT.txt"
KMZ_MARCOS = "/home/rogerio/PycharmProjects/deepseek-coordinates-ocr/data/KM_BR163.kmz"

KMZ_SAIDA = "/home/rogerio/PycharmProjects/deepseek-coordinates-ocr/data/testes/pontos_imagens_referenciados.kmz"
EXCEL_SAIDA = "/home/rogerio/PycharmProjects/deepseek-coordinates-ocr/data/testes/pontos_imagens_referenciados.xlsx"

# ============================================================
# FUNÇÃO HAVERSINE
# ============================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # metros
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2

    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ============================================================
# LER KMZ (MARCOS OU PONTOS)
# ============================================================

def read_kmz_points(kmz_path):
    with zipfile.ZipFile(kmz_path, "r") as kmz:
        kml_file = [f for f in kmz.namelist() if f.endswith(".kml")][0]
        kml_data = kmz.read(kml_file)

    root = ET.fromstring(kml_data)
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    pontos = []
    for pm in root.findall(".//kml:Placemark", ns):
        coord = pm.find(".//kml:coordinates", ns).text.strip()
        lon, lat, *_ = coord.split(",")

        nome = pm.find("kml:name", ns)
        nome = nome.text if nome is not None else ""

        pontos.append({
            "name": nome,
            "lat": float(lat),
            "lon": float(lon)
        })

    return pontos

# ============================================================
# LER TXT DO OCR
# ============================================================

def read_ocr_txt(txt_path):
    pontos = []

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "NO_COORD_FOUND" in line:
                continue

            img, coord = line.split("|")
            lat_str, lon_str = coord.split()

            lat = float(lat_str[:-1])
            if lat_str.endswith("S"):
                lat = -lat

            lon = float(lon_str[:-1])
            if lon_str.endswith("W"):
                lon = -lon

            pontos.append({
                "image": img,
                "lat": lat,
                "lon": lon
            })

    return pontos

# ============================================================
# PROCESSAMENTO
# ============================================================

print("Lendo marcos KM...")
marcos = read_kmz_points(KMZ_MARCOS)

# extrair número do KM do nome
for m in marcos:
    km_num = "".join(filter(str.isdigit, m["name"]))
    if not km_num:
        raise ValueError(f"Não foi possível extrair KM de: {m['name']}")
    m["km"] = int(km_num)

marcos = sorted(marcos, key=lambda x: x["km"])

print("Lendo pontos do OCR...")
pontos = read_ocr_txt(TXT_OCR)

placemarks = []
linhas_excel = []

for p in pontos:
    distancias = []
    for m in marcos:
        d = haversine(p["lat"], p["lon"], m["lat"], m["lon"])
        distancias.append((m["km"], d))

    distancias.sort(key=lambda x: x[1])

    km1, d1 = distancias[0]
    km2, d2 = distancias[1]

    if km1 < km2:
        km_base = km1
        dist_m = int(round(d1))
    else:
        km_base = km2
        dist_m = int(round(d2))

    nome_ponto = f"{p['image']} | KM_{km_base}+{dist_m:03d}"

    # KML
    placemarks.append(f"""
    <Placemark>
        <name>{nome_ponto}</name>
        <description>
            Coordenadas: {p['lat']:.6f}, {p['lon']:.6f}
        </description>
        <Point>
            <coordinates>{p['lon']},{p['lat']},0</coordinates>
        </Point>
    </Placemark>
    """)

    # Excel
    linhas_excel.append({
        "Imagem": p["image"],
        "KM_Base": km_base,
        "Distancia_m": dist_m,
        "Latitude": round(p["lat"], 6),
        "Longitude": round(p["lon"], 6)
    })

# ============================================================
# GERAR KMZ
# ============================================================

kml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
{''.join(placemarks)}
</Document>
</kml>
"""

with open("temp.kml", "w", encoding="utf-8") as f:
    f.write(kml_content)

with zipfile.ZipFile(KMZ_SAIDA, "w", zipfile.ZIP_DEFLATED) as kmz:
    kmz.write("temp.kml", arcname="doc.kml")

# ============================================================
# GERAR EXCEL
# ============================================================

df = pd.DataFrame(linhas_excel)
df.to_excel(EXCEL_SAIDA, index=False)

os.remove("temp.kml")

print("KMZ gerado:", KMZ_SAIDA)
print("Excel gerado:", EXCEL_SAIDA)

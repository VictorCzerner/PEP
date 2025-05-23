from ultralytics import YOLO
import os
import shutil
from glob import glob

# === Caminhos ===
CAMINHO_MODELO = "runs/detect/train2/weights/best.pt"
CAMINHO_IMAGENS_NOVAS = "/root/PEP/novas_fotos"
DESTINO_IMAGENS = "/root/PEP/Dataset/images/train"
DESTINO_LABELS = "/root/PEP/Dataset/labels/train"

os.makedirs(DESTINO_IMAGENS, exist_ok=True)
os.makedirs(DESTINO_LABELS, exist_ok=True)

# === Carrega o modelo treinado ===
modelo = YOLO(CAMINHO_MODELO)

# === Faz predição e salva os rótulos no padrão YOLO ===
results = modelo.predict(
    source=CAMINHO_IMAGENS_NOVAS,
    save_txt=True,
    save_conf=True,
    conf=0.25
)

# === Encontra a última pasta de predição ===
pastas_predict = sorted(glob("runs/detect/predict*"), key=os.path.getmtime)
pasta_predicao_mais_recente = pastas_predict[-1]
pasta_labels = os.path.join(pasta_predicao_mais_recente, "labels")

print(f"🔍 Buscando labels em: {pasta_labels}")

# === Move os labels gerados ===
labels_geradas = glob(os.path.join(pasta_labels, "*.txt"))
for lbl in labels_geradas:
    shutil.move(lbl, os.path.join(DESTINO_LABELS, os.path.basename(lbl)))

# === Move imagens para pasta de treino ===
imagens = glob(os.path.join(CAMINHO_IMAGENS_NOVAS, "*.jpg")) + \
          glob(os.path.join(CAMINHO_IMAGENS_NOVAS, "*.jpeg")) + \
          glob(os.path.join(CAMINHO_IMAGENS_NOVAS, "*.png"))

for img in imagens:
    shutil.move(img, os.path.join(DESTINO_IMAGENS, os.path.basename(img)))

print("✅ Labels e imagens movidos com sucesso!")

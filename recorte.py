import os
import cv2
from ultralytics import YOLO

# Nome do arquivo de entrada
imagem_entrada = "/root/PEP/testes/Buffy_104.jpg"
nome_base = os.path.splitext(os.path.basename(imagem_entrada))[0]

# Carrega o modelo YOLO
model = YOLO("runs/detect/train3/weights/best.pt")
results = model.predict(source=imagem_entrada)[0]
image = cv2.imread(imagem_entrada)

# Cria pasta
os.makedirs("maos_cortadas", exist_ok=True)

# Recorta e salva com nome único
for i, box in enumerate(results.boxes.xyxy):
    x1, y1, x2, y2 = map(int, box)
    mao_cortada = image[y1:y2, x1:x2]
    caminho_saida = f"maos_cortadas/{nome_base}_mao_{i+1}.jpg"
    cv2.imwrite(caminho_saida, mao_cortada)

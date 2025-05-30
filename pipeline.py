import os
import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms
import argparse

# === Argumento de entrada
parser = argparse.ArgumentParser()
parser.add_argument("--imagem", type=str, default="/root/PEP/testes/image.png", help="Caminho da imagem para analisar")
args = parser.parse_args()
caminho_imagem = args.imagem
nome_base = os.path.splitext(os.path.basename(caminho_imagem))[0]

# === Device (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Modelos
modelo_yolo = YOLO("runs/detect/train2/weights/best.pt")
modelo_cnn = torch.load("modelo_classificador.pt", map_location=device)
modelo_cnn = modelo_cnn.to(device)
modelo_cnn.eval()

# === Transforms para classificador
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Pasta de saída
pasta_saida = "maos_pipeline"
os.makedirs(pasta_saida, exist_ok=True)

# === YOLO detecta mãos
print(f"🔍 Analisando {caminho_imagem} com YOLO...")
results = modelo_yolo.predict(source=caminho_imagem)[0]
image = cv2.imread(caminho_imagem)

# === Para cada mão detectada
for i, box in enumerate(results.boxes.xyxy):
    x1, y1, x2, y2 = map(int, box)
    mao_crop = image[y1:y2, x1:x2]

    # Prepara imagem para a CNN

    # Classifica
    entrada = transform(mao_crop).unsqueeze(0).to(device)
    saida = modelo_cnn(entrada)
    classe = torch.argmax(saida).item()
    rotulo = ["Adulto", "Idoso"][classe]

    # Salva a imagem com o nome da classe
    nome_arquivo = f"{nome_base}_mao{i+1}_{rotulo.lower()}.jpg"
    caminho_saida = os.path.join(pasta_saida, nome_arquivo)
    cv2.imwrite(caminho_saida, mao_crop)

    print(f"✋ Mão {i+1}: {rotulo} — salva como {nome_arquivo}")

print("✅ Pipeline concluído.")

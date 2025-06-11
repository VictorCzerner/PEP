import os
import cv2
import torch
import csv
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms

# === Pasta com imagens de entrada
pasta_imagens = "/root/PEP/testes"

# === Pasta de saída
pasta_saida = "maos_pipeline_idade"
os.makedirs(pasta_saida, exist_ok=True)

# === Arquivo CSV de resultados
csv_path = os.path.join(pasta_saida, "resultados_idade.csv")
csv_file = open(csv_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)

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

# === Processa todas as imagens da pasta
for arquivo in sorted(os.listdir(pasta_imagens)):
    if not arquivo.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    caminho_imagem = os.path.join(pasta_imagens, arquivo)
    nome_base = os.path.splitext(arquivo)[0]

    print(f"🔍 Analisando {caminho_imagem} com YOLO...")
    results = modelo_yolo.predict(source=caminho_imagem)[0]
    image = cv2.imread(caminho_imagem)

    if len(results.boxes.xyxy) == 0:
        print(f"⚠️ Nenhuma mão detectada em {arquivo}.")
        continue

    # === Só considera a primeira mão
    box = results.boxes.xyxy[0]
    x1, y1, x2, y2 = map(int, box)
    mao_crop = image[y1:y2, x1:x2]

    entrada = transform(mao_crop).unsqueeze(0).to(device)
    saida = modelo_cnn(entrada)
    classe = torch.argmax(saida).item()
    rotulo = ["Adulto", "Idoso"][classe]

    nome_arquivo = f"{nome_base}_mao1_{rotulo.lower()}.jpg"
    caminho_saida = os.path.join(pasta_saida, nome_arquivo)
    cv2.imwrite(caminho_saida, mao_crop)

    print(f"✋ Mão detectada: {rotulo} — salva como {nome_arquivo}")
    csv_writer.writerow([rotulo])

csv_file.close()
print(f"✅ Pipeline concluído. Resultados salvos em {csv_path}")

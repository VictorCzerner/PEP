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
pasta_saida = "maos_pipeline3"
os.makedirs(pasta_saida, exist_ok=True)

# === Arquivo CSV de resultados
csv_path = os.path.join(pasta_saida, "resultados.csv")
csv_file = open(csv_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["nome_da_imagem", "classe"])

# === Device (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Modelos
modelo_yolo = YOLO("runs/detect/train2/weights/best.pt")
modelo_cnn = torch.load("modelo_classificadorG.pt", map_location=device)
modelo_cnn = modelo_cnn.to(device)
modelo_cnn.eval()

# === Transforms para classificador
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

arquivos = os.listdir(pasta_imagens)
print(f"Total de arquivos na pasta: {len(arquivos)}")

# === Processa todas as imagens na pasta
for arquivo in sorted(os.listdir(pasta_imagens)):
    if not arquivo.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    caminho_imagem = os.path.join(pasta_imagens, arquivo)
    nome_base = os.path.splitext(arquivo)[0]

    print(f"🔍 Analisando {caminho_imagem} com YOLO...")
    results = modelo_yolo.predict(source=caminho_imagem)[0]
    image = cv2.imread(caminho_imagem)

    for i, box in enumerate(results.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        mao_crop = image[y1:y2, x1:x2]

        entrada = transform(mao_crop).unsqueeze(0).to(device)
        saida = modelo_cnn(entrada)
        classe = torch.argmax(saida).item()
        rotulo = ["SemAcessorio", "ComAcessorio"][classe]

        nome_arquivo = f"{nome_base}_mao{i+1}_{rotulo.lower()}.jpg"
        caminho_saida = os.path.join(pasta_saida, nome_arquivo)
        cv2.imwrite(caminho_saida, mao_crop)

        print(f"✋ Mão {i+1}: {rotulo} — salva como {nome_arquivo}")

        # === Salva no CSV
        csv_writer.writerow([nome_arquivo, rotulo])
        break

csv_file.close()
print(f"✅ Pipeline concluído. Resultados salvos em {csv_path}")
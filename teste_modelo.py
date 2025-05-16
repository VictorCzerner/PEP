from ultralytics import YOLO

# Carrega o modelo treinado
model = YOLO("runs/detect/train2/weights/best.pt")

# Faz predição em uma imagem
results = model.predict(source="/root/praticaEmPesquisa/testes/imagem001.jpeg", save=True)

# Ver os resultados se quiser:
print(results[0].boxes)

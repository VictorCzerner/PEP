from ultralytics import YOLO

# Carrega o modelo treinado
model = YOLO("runs/detect/train3/weights/best.pt")

# Faz predição em uma imagem
results = model.predict(source="/root/PEP/testes/imagem004.jpeg", save=True)

# Ver os resultados se quiser:
print(results[0].boxes)

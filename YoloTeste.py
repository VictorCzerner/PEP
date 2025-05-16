from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="/root/praticaEmPesquisa/dataset.yaml" , epochs=152)


from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="/root/PEP/dataset.yaml" , epochs=1095)


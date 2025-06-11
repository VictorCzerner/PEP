import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split

# Caminho do dataset
DATASET_PATH = "/root/PEP/DataIdade"

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

# Carrega dataset
dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)

# Divide em treino e validação
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)

# Modelo pré-treinado
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: Adulto, Idoso

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treinamento
print("🚀 Iniciando o treinamento...")
for epoch in range(10):  # Você pode aumentar
    model.train()
    total, correct = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"📘 Epoch {epoch+1}: Acurácia treino = {acc:.2f}%")

model.eval()
total, correct = 0, 0

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

acc_val = 100 * correct / total
print(f"🔎 Acurácia na validação = {acc_val:.2f}%")

# Salva o modelo
torch.save(model, "modelo_classificador.pt")
print("✅ Modelo salvo como 'modelo_classificador.pt'")
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from Prepare_dataset import train_loader, test_loader, device  # Önceki dosyadan veri setini al

# Sınıf sayısı (8 farklı kan hücresi türü var)
num_classes = 8

# Önceden eğitilmiş ResNet-50 modelini yükle
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Son katmanı 8 sınıfa uygun hale getir

# Modeli GPU'ya taşı
model = model.to(device)

# Kayıp fonksiyonu (Loss Function) ve optimizer seçimi
criterion = nn.CrossEntropyLoss()  # Çok sınıflı sınıflandırma için uygun
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer kullanıyoruz


# Modeli eğitme fonksiyonu
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")


# Modeli eğit
train_model(model, train_loader, criterion, optimizer, epochs=5)

# Eğitilmiş modeli kaydet
torch.save(model.state_dict(), "blood_cell_classifier.pth")
print("Model başarıyla kaydedildi!")

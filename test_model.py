import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models
from Prepare_dataset import test_loader, dataset, device  # Önceki dosyadan veri setini al
from sklearn.metrics import confusion_matrix, classification_report

# Eğitilmiş modeli yükle
num_classes = 8
model = models.resnet50(weights=None)  # Yeni model nesnesi oluştur
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Son katmanı tekrar tanımla
model.load_state_dict(torch.load("blood_cell_classifier.pth"))  # Kaydedilen ağırlıkları yükle
model = model.to(device)
model.eval()  # Modeli değerlendirme moduna al

# Test seti üzerinde tahminler yap
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# **Confusion Matrix Hesapla**
cm = confusion_matrix(y_true, y_pred)
class_names = dataset.classes  # Sınıf isimleri

# **Confusion Matrix Görselleştirme**
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix - Blood Cell Classification")
plt.show()

# **Sınıflandırma Raporunu Yazdır**
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))

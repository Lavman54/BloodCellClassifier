import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

# Cihaz belirleme (GPU varsa kullan)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Veri seti yolu
dataset_path = "C:\\Users\\ARDA\\PycharmProjects\\PythonProject3\\BloodCellsDataset"

# Görüntü ön işleme adımları
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Görüntüleri 224x224 boyutuna getir
    transforms.RandomHorizontalFlip(),  # Rastgele yatay çevirme
    transforms.RandomRotation(10),  # Rastgele 10 derece döndürme
    transforms.ToTensor(),  # Tensor formatına çevir
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizasyon
])

# Veri setini yükle
dataset = ImageFolder(root=dataset_path, transform=transform)

# Veri setini eğitim (%80) ve test (%20) olarak böl
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoader oluştur
batch_size = 32  # Mini-batch boyutu
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Sınıf isimlerini yazdır
print(f"Class names: {dataset.classes}")
print(f"Total images: {len(dataset)}, Train: {train_size}, Test: {test_size}")

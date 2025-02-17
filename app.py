import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import gdown
import os

# **Streamlit sayfa yapılandırmasını EN BAŞTA yap**
st.set_page_config(page_title="Blood Cell Classification AI", layout="wide")

# **Modeli Google Drive’dan indir ve yükle**
def download_model():
    drive_url = "https://drive.google.com/uc?id=1o_Dz60Ucza7HsZsG71kC274RsozwnVym"
    model_path = "blood_cell_classifier.pth"

    # Eğer model zaten varsa, tekrar indirme
    if not os.path.exists(model_path):
        st.write("📥 Model indiriliyor, lütfen bekleyin...")
        gdown.download(drive_url, model_path, quiet=False)
        st.write("✅ Model başarıyla indirildi!")

    return model_path

# **Modeli yükleme fonksiyonu**
@st.cache_resource
def load_model():
    model_file = download_model()
    num_classes = 8
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

# **Modeli yükle**
model = load_model()

# **Sınıf etiketleri**
class_names = ['Basophil', 'Eosinophil', 'Erythroblast', 'IG', 'Lymphocyte', 'Monocyte', 'Neutrophil', 'Platelet']

# **Görüntü işleme fonksiyonu**
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)

# **Arayüz Tasarımı**
st.markdown(
    f"""
    <style>
    body {{
        background-color: #0d0d0d;
        color: white;
        font-family: Arial, sans-serif;
    }}
    .title {{
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }}
    .footer {{
        text-align: center;
        font-size: 14px;
        margin-top: 50px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# **Başlık ve Logo**
st.image("logo.jpg", use_container_width=True)
st.markdown("<div class='title'>🚀 Yapay Zeka ile Kan Hücresi Sınıflandırma</div>", unsafe_allow_html=True)

st.write("Lütfen bir periferik yayma mikroskop görüntüsü yükleyin.")

# **Görüntü yükleme alanı**
uploaded_file = st.file_uploader("Görüntü Seç (JPG, PNG)", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Yüklenen Görüntü", use_container_width=True)

    # **Görüntüyü işle ve modele ver**
    img_tensor = process_image(image)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        prediction = class_names[predicted.item()]

    # **Tahmin sonucunu göster**
    st.subheader(f"📌 Model Tahmini: **{prediction}**")

# **Footer**
st.markdown("<div class='footer'>Written by ArdaBilgili | 2025</div>", unsafe_allow_html=True)

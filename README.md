# 🗑️ Sistem Klasifikasi Sampah dengan Deep Learning

> Proyek Final Deep Learning — Klasifikasi sampah menjadi **Organik**, **Anorganik**, dan **B3** menggunakan CNN, MobileNetV2, dan ResNet50.

---

## 📂 Struktur Proyek

```
deeplearning/
├── garbage_classification.ipynb    ← Notebook training lengkap
├── app.py                          ← Streamlit web app
├── requirements.txt                ← Dependensi Python
├── .streamlit/
│   └── config.toml                 ← Tema warna (#CE7F20, #607938, #533416)
├── models/                         ← (dibuat otomatis saat training)
│   ├── E1_best.keras
│   ├── E2_best.keras
│   ├── E3_best.keras
│   ├── F1_best.keras
│   ├── F2_best.keras
│   ├── F3_best.keras
│   ├── best_model_final.h5         ← Model terbaik untuk deployment
│   └── class_names.json            ← Metadata kelas & preprocessing
├── dataset/split/                  ← (dibuat otomatis saat training)
│   ├── train/
│   ├── val/
│   └── test/
└── histories/                      ← Kurva training & confusion matrix
```

---

## 🎯 Dataset

**Sumber:** [Kaggle — mostafaabla/garbage-classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)

**Pemetaan Kelas (12 → 3):**

| Kelas Asli | Kategori Target |
|------------|-----------------|
| `biological` | 🌱 **Organik** |
| `battery` | ⚠️ **B3** |
| `brown-glass`, `cardboard`, `clothes`, `green-glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`, `white-glass` | ♻️ **Anorganik** |

**Pembagian Data (Stratified):**
- Training: **70%**
- Validation: **15%**
- Testing: **15%**

---

## 🧠 Model & Arsitektur

### 1. Custom CNN (Baseline)
Dibangun dan dilatih dari nol (*from scratch*) tanpa pretrained weights.
```
Input (224×224×3)
→ Conv2D(32) + BN + ReLU + MaxPool
→ Conv2D(64) + BN + ReLU + MaxPool
→ Conv2D(128)+ BN + ReLU + MaxPool
→ Conv2D(256)+ BN + ReLU + MaxPool
→ GlobalAveragePooling2D
→ Dense(256, ReLU) + Dropout(0.5)
→ Dense(3, Softmax)
```

### 2. MobileNetV2 (Transfer Learning)
- **Pretrained:** ImageNet (1.28M gambar, 1000 kelas)
- **Arsitektur:** 17× Inverted Residual Block + Depthwise Separable Conv
- **Strategi:** Feature Extraction — base layer dibekukan (frozen)
- **Head tambahan:** Dense(128) → Dropout(0.4) → Dense(3, Softmax)
- **Feature dim:** 1280-dimensional

### 3. ResNet50 (Transfer Learning)
- **Pretrained:** ImageNet (1.28M gambar, 1000 kelas)
- **Arsitektur:** 50 layer dengan Residual/Skip Connections, 4 Stage Bottleneck
- **Strategi:** Feature Extraction — base layer dibekukan (frozen)
- **Head tambahan:** Dense(128) → Dropout(0.4) → Dense(3, Softmax)
- **Feature dim:** 2048-dimensional

---

## 🔬 Desain Eksperimental

| Kode | Model | Learning Rate | Epoch | Optimizer | Batch |
|------|-------|---------------|-------|-----------|-------|
| **E1** *(Baseline)* | CNN | 0.001 | 15 | Adam | 32 |
| **E2** | MobileNetV2 | 0.001 | 15 | Adam | 32 |
| **E3** | ResNet50 | 0.001 | 15 | Adam | 32 |
| **F1** | CNN | 0.0001 | 15 | Adam | 32 |
| **F2** | MobileNetV2 | 0.0001 | 15 | Adam | 32 |
| **F3** | ResNet50 | 0.0001 | 15 | Adam | 32 |

**Callbacks:** EarlyStopping (patience=5) + ModelCheckpoint + ReduceLROnPlateau

**Augmentasi (training only):**
- Horizontal flip
- Rotation ±20°
- Zoom ±20%
- Width & height shift ±10%

**Metrik Evaluasi:** Accuracy · Precision (macro) · Recall (macro) · F1-Score (macro) · Confusion Matrix

---

## 🚀 Cara Menjalankan

### Langkah 1 — Siapkan Kaggle API
```bash
# Download token dari https://www.kaggle.com/settings → Account → API
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Langkah 2 — Install Dependensi
```bash
pip install -r requirements.txt
```

### Langkah 3 — Jalankan Notebook Training
```bash
jupyter notebook garbage_classification.ipynb
# Jalankan semua sel (Run All) → tunggu hingga selesai (~30-120 menit tergantung GPU)
```

### Langkah 4 — Jalankan Aplikasi Streamlit
```bash
streamlit run app.py
```

Buka browser di `http://localhost:8501`

---

## 📱 Fitur Aplikasi

| Fitur | Deskripsi |
|-------|-----------|
| 📷 **Kamera Live** | Ambil foto sampah langsung dari webcam |
| 📁 **Upload Foto** | Upload gambar JPG/PNG/WEBP |
| 🎯 **Klasifikasi AI** | Prediksi kategori: Organik / Anorganik / B3 |
| 📊 **Confidence Score** | Progress bar + bar chart distribusi keyakinan model |
| 📋 **Panduan Pembuangan** | Tips penanganan spesifik per kategori |
| 🤖 **Info Model** | Sidebar: arsitektur, skenario terbaik, Macro F1 |
| 📊 **Semua Skenario** | Tabel perbandingan semua 6 eksperimen di sidebar |

---

## 🎨 Palet Warna

| Warna | Hex | Penggunaan |
|-------|-----|-----------|
| Orange/Gold | `#CE7F20` | Primary · Anorganik badge |
| Hijau | `#607938` | Organik badge |
| Coklat Tua | `#533416` | Judul · aksen |

---

## 📋 Requirement Akademik yang Dipenuhi

- ✅ **Framework:** TensorFlow 2.x / Keras
- ✅ **Model types:** CNN (from scratch) + Pretrained (MobileNetV2 + ResNet50)
- ✅ **Pretrained model tidak black box:** Dijelaskan arsitektur, pretraining objective, dataset, transfer learning strategy, layer yang digunakan, alasan pemilihan, dan keterbatasan
- ✅ **Proses eksperimen:** 6 skenario (1 baseline + 5 kondisi eksperimen)
- ✅ **Evaluasi:** Accuracy, Precision, Recall, F1-Macro, Confusion Matrix per skenario
- ✅ **Minimal 2 kondisi selain baseline:** E2, E3, F1, F2, F3 (5 kondisi)

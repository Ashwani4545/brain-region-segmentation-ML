# 🧠 NeuroDetect AI — Brain NCCT Hypodense Region Segmentation

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python)
![Django](https://img.shields.io/badge/Django-6.0-green?style=for-the-badge&logo=django)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?style=for-the-badge&logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-lightblue?style=for-the-badge&logo=opencv)
![License](https://img.shields.io/badge/License-Academic-yellow?style=for-the-badge)

**An AI-powered web application for detecting and segmenting hypodense regions (e.g., stroke-affected areas) from Brain Non-Contrast CT (NCCT) scans.**

</div>

---

## 📸 Application Preview

| Landing Page | Scanner App | AI Overlay Result |
|---|---|---|
| Premium dark-theme hero page | Drag-and-drop NCCT upload | Red-highlighted hypodense regions with confidence score |

---

## 🎯 What This Does

NeuroDetect AI takes a Brain NCCT image (DICOM, JPG, or PNG) and:

1. **Detects hypodense regions** — areas darker than surrounding brain tissue, typical of ischemic stroke, oedema, or infarction
2. **Generates a segmentation mask** — binary map of detected lesions
3. **Produces an AI overlay** — original scan with red highlights and orange contour outlines on lesions
4. **Reports a confidence score** — lesion load % (fraction of brain area that is hypodense)
5. **Provides an Explainable AI summary** — plain-language interpretation of findings

---

## 🏗️ Project Structure

```
brain-region-segmentation/
│
├── data/                          # NCCT image data
│   ├── ST000001/                  # DICOM study folders
│   ├── Training/                  # Training image sets
│   ├── testing/                   # Test image sets
│   └── splits/                    # train.csv / val.csv split files
│
├── webapp/                        # 🌐 Django Web Application
│   ├── manage.py
│   ├── core_ml/                   # ML inference engine
│   │   ├── __init__.py
│   │   ├── model.py               # 2.5D U-Net architecture
│   │   └── inference_service.py   # Dual-mode segmentation pipeline
│   ├── segmentation/              # Django app
│   │   ├── views.py               # API + page views
│   │   ├── urls.py                # URL routing
│   │   ├── templates/segmentation/
│   │   │   ├── index.html         # Landing page
│   │   │   └── app.html           # Scanner application page
│   │   └── static/segmentation/
│   │       ├── css/styles.css     # Premium glassmorphism design system
│   │       └── js/app.js          # Drag-and-drop + AJAX inference
│   └── webapp/                    # Django project settings
│       ├── settings.py
│       └── urls.py
│
├── configs/config.json            # Training hyperparameters
├── model.py                       # U-Net model definition (root)
├── dataset.py                     # PyTorch Dataset (NIfTI 2.5D)
├── train.py                       # Segmentation training loop
├── train_classifier.py            # CNN classifier training
├── inference.py                   # Volumetric inference (NIfTI)
├── preprocess.py                  # DICOM → NIfTI preprocessing
├── classifier.py                  # CNN classifier definition
├── report_generator.py            # Radiology-style text report
├── visualize.py                   # Slice visualization utilities
├── requirements.txt               # Python dependencies
└── README.md
```

---

## ⚙️ Segmentation Pipeline

The inference engine uses a **dual-mode physics-based pipeline** — no trained weights required for meaningful results:

### Mode 1 — DICOM (Hounsfield Unit Thresholding)
> Most accurate — uses real clinical HU values

1. Apply DICOM rescale slope/intercept → true HU values
2. Brain window: **WL = 35, WW = 100** (standard brain CT window)
3. Skull strip: approximate brain mask via connected-component analysis
4. Hypodense band: **HU 15–35** (below normal parenchyma ~30–45 HU)
5. Restrict to brain mask only
6. Morphological open/close + remove regions < 50 px

### Mode 2 — JPG / PNG (Adaptive Intensity Thresholding)
> Works on standard images without HU metadata

1. CLAHE contrast enhancement
2. Approximate skull stripping
3. Compute tissue median and MAD inside the brain
4. Flag pixels **> 1.5 × MAD below median** as hypodense
5. Morphological cleanup + remove regions < 30 px

### Mode 3 — Trained UNet (when checkpoint provided)
> Best accuracy when trained weights are available

- 2.5D U-Net with 5-channel pseudo-volume input
- Automatically used if a `.pth` checkpoint is supplied

---

## 🚀 Quick Start — Web Application

### 1. Clone the repository
```bash
git clone https://github.com/Ashwani4545/brain-region-segmentation-ML.git
cd brain-region-segmentation-ML
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Run the web app
```bash
cd webapp
python manage.py migrate
python manage.py runserver
```

### 4. Open in browser
```
http://127.0.0.1:8000/
```

Upload any Brain CT scan (JPG, PNG, or DICOM `.dcm`) and get instant segmentation results.

---

## 🧪 Training the Model (Optional)

For best segmentation accuracy, train the U-Net on labelled NCCT data:

### 1. Prepare data splits
Create `data/splits/train.csv` and `data/splits/val.csv` with columns:
```
image,mask
data/nifti/patient001.nii.gz,data/masks/patient001_mask.nii.gz
...
```

### 2. Preprocess DICOM → NIfTI
```bash
python preprocess.py --input_dir data/raw --output_dir data/nifti --spacing 1.0 1.0 5.0
```

### 3. Train segmentation model
```bash
python train.py --config configs/config.json
# Checkpoint saved to: checkpoints/best.pth
```

### 4. Train classifier (optional)
```bash
python train_classifier.py --config configs/config.json
```

### 5. Activate trained weights in the web app
Edit `webapp/core_ml/inference_service.py` — update the singleton:
```python
_service = InferenceService(
    checkpoint_path=r"path/to/checkpoints/best.pth"
)
```

---

## 💡 Outputs

| Output | Description |
|--------|-------------|
| 🖼️ **Original Scan** | Uploaded brain CT image |
| 🩹 **Segmented Mask** | Binary map — white = hypodense regions |
| 🎨 **AI Overlay** | Scan + semi-transparent red fill + orange contour outlines |
| 📊 **Confidence Score** | Lesion load % = hypodense area ÷ total brain area |
| 📝 **Explainable AI Summary** | Plain-language clinical interpretation |
| ⬇️ **Download** | Export the overlay image directly |

---

## 🧩 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Web Framework** | Django 6.0 + Django REST Framework |
| **ML / DL** | PyTorch 2.x, 2.5D U-Net |
| **Image Processing** | OpenCV, NumPy, scikit-image |
| **Medical Imaging** | pydicom, nibabel, SimpleITK |
| **Frontend** | Vanilla HTML/CSS/JS — glassmorphism dark theme |
| **Language** | Python 3.12+ |

---

## 🌐 Web App Features

- ✅ Drag-and-drop or click-to-browse file upload
- ✅ Supports JPG, PNG, DICOM (`.dcm`) formats
- ✅ File type & size validation (max 20 MB)
- ✅ Loading spinner during inference
- ✅ Side-by-side: Original | Mask | AI Overlay
- ✅ Colour-coded confidence badge (🟢 Normal / 🔴 Anomaly)
- ✅ Explainable AI summary panel
- ✅ Download overlay button
- ✅ UUID-prefixed filenames (no collision on concurrent uploads)
- ✅ Premium glassmorphism UI with animated background

---

## 🗂️ Running Batch Inference (CLI)

For processing full NIfTI volumes from the command line:

```bash
python inference.py \
  --model checkpoints/best.pth \
  --input data/nifti \
  --output_dir results
```

Generate a radiology-style text report:
```bash
python report_generator.py \
  --mask results/patient.mask.nii.gz \
  --image data/nifti/patient.nii.gz \
  --out results/patient_report.txt
```

---

## ⚠️ Disclaimer

> This software is intended for **academic and research purposes only**.
> It is **not a certified medical device** and must not be used for clinical diagnosis without validation by a qualified radiologist.
> This repository does **not** contain patient medical images.

---

## 👨‍💻 Author

```
Ashwani Pandey
Project : NeuroDetect AI — Brain NCCT Hypodense Region Segmentation
Model   : 2.5D U-Net + Physics-based HU Thresholding Pipeline
Stack   : Django · PyTorch · OpenCV · pydicom
Year    : 2025–2026
Use     : Academic & Research Purposes Only
```

---

## 🏁 Summary

NeuroDetect AI demonstrates end-to-end AI + healthcare integration — from raw NCCT data through deep learning segmentation to a production-ready web interface. The dual-mode inference pipeline (HU thresholding for DICOM, adaptive thresholding for JPG/PNG) delivers clinically meaningful results even without a pre-trained model, making it accessible for research and demonstration immediately after setup.

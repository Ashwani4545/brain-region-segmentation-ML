# 🧠 Brain NCCT Hypodense Region Segmentation & Classification Project

## Project Overview
This project automatically detects, segments, and classifies hypodense regions from Brain Non-Contrast Computed Tomography (NCCT) scans.
It uses deep learning models (2.5D U-Net + CNN classifier) to identify abnormalities such as ischemic strokes or lesions and generates a summary report describing the findings.

### Project Objectives
```
• Segment hypodense regions (stroke, edema, etc.) using a 2.5D U-Net model
• Classify NCCT scans as Normal or Abnormal using a CNN classifier
• Generate a descriptive report summarizing lesion characteristics and side
• Provide a Google Colab notebook for easy and automated execution
```

## 📁 Project Structure
```
NCCT_Segmentation_Final/
│
├── data/ # NCCT image files and CSV splits
├── models/ # Saved trained models (segmentation + classification)
├── notebooks/ # Google Colab notebook for end-to-end workflow
├── src/ # Source code for training, inference, reporting
│ ├── train_segmentation.py
│ ├── train_classifier.py
│ ├── inference.py
│ ├── report_generator.py
│ └── utils.py
├── requirements.txt # Python dependencies
└── README.md # Documentation
```

## ⚙️ Workflow Overview

1. Preprocessing — DICOM/NIfTI conversion, normalization, skull stripping
2. Segmentation — 2.5D U-Net detects hypodense regions (e.g., stroke zones)
3. Classification — CNN predicts whether the scan is Normal or Abnormal
4. Report Generation — Produces a readable summary of findings
5. Visualization — Displays overlay of segmented lesions for interpretation


## 🚀 Running on Google Colab

1. Upload the project ZIP to your Google Drive
2. Mount your Drive:
3. from google.colab import drive
4. drive.mount('/content/drive')
5. Navigate to the project:
6. %cd /content/drive/MyDrive/NCCT_Segmentation_Final
7. Install dependencies:
8. !pip install -r requirements.txt
9. Open and run the notebook:
10. notebooks/colab_run.ipynb


## 💡 Outputs
```
• 🧩 Segmentation Mask: Highlights lesion regions
• 🧠 Classification Result: Normal / Abnormal
• 📄 Report: Text summary (lesion volume, type, side)
• 🎨 Visualization: Overlayed images for interpretation
```

# 🧩 Tech Stack
```
• Language: Python 3.12
• Frameworks: PyTorch, torchvision
• Libraries: nibabel, SimpleITK, pydicom, pynrrd, albumentations, scikit-image, matplotlib, numpy, scipy
```

```
**Important:** This repository does NOT contain medical images. Use public datasets (e.g., CQ500) or institutional NCCTs.
```

## Quick run (Colab / local)

1. Place DICOM / NIfTI files under `data/raw/` and corresponding masks (if available) under `data/masks/`.
2. Update `data/splits/train.csv` and `data/splits/val.csv` with two columns: `image,mask`.
3. Install requirements: `pip install -r requirements.txt`
4. Preprocess: `python preprocess.py --input_dir data/raw --output_dir data/nifti --spacing 1.0 1.0 5.0`
5. Train segmentation: `python train.py --config configs/config.json`
6. Train classifier (optional): `python train_classifier.py --config configs/config.json`
7. Inference & report: `python inference.py --model checkpoints/best.pth --input data/nifti --output_dir results`
   Then `python report_generator.py --mask results/patient.mask.nii.gz --image data/nifti/patient.nii.gz --out results/patient_report.txt`


## What's new in this final version
```
- `classifier.py` and `train_classifier.py` (simple encoder-based classifier)
- `report_generator.py` that creates a short radiology-style textual report (template-based)
- Updated `requirements.txt` (uses `pynrrd` instead of `nrrd`)

## 👨‍💻 Author
```
Ashwani Pandey
Project: Brain NCCT Hypodense Region Segmentation and Classification
Model: 2.5D U-Net + CNN | Dataset: CQ500
Year: 2025
Use: Academic & Research Purposes Only
```

## 🏁 Summary

This project demonstrates how AI-powered medical imaging can assist radiologists in early detection of ischemic strokes and lesions from NCCT scans — a step toward faster and more accurate computer-aided diagnosis.

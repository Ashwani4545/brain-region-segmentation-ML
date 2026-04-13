# AI-Based Brain CT Hypodense Region Detection System

## Goal Description

The project aims to develop a complete end-to-end web application that integrates the existing PyTorch 2.5D U-Net segmentation model. The web app will allow users to upload Brain NCCT images (PNG, JPG, DICOM) and view detected hypodense regions (like stroke-affected areas). The solution features a modern front-end built with Vanilla CSS for a premium look, and a Python Django backend for robust handling of image processing, inference, and file saving.

> [!NOTE]
> Since the original model expects 2.5D volumetric inputs (5 slices), and the UI will support single 2D image uploads (PNG/JPG/DICOM), we will adapt the data pipeline to duplicate a single 2D upload into 5 channels, allowing the model to process 2D images as pseudo-2.5D inputs seamlessly.

## User Review Required

> [!WARNING]
> No pre-trained model weights (`.pth` or `.pt` files) were found in the workspace or `checkpoints` directory. The application will be built to instantiate the model and expect a path to the weights. If the weights are missing, it will initialize the model with untrained weights to ensure the pipeline runs for demonstration purposes without crashing. 
> **Are you planning to provide a `.pth` model file, or is random initialization fine for the prototype phase?**

> [!IMPORTANT]
> Since we are creating a whole Django project, I propose scaffolding the Django project directly inside `e:\projects\Advanced ML\python\brain-region-segmentation\webapp`. Is that location acceptable?

## Proposed Changes

### Django Setup and Architecture

We will create a Django project and a main application called `segmentation`.
#### [NEW] [webapp/manage.py](file:///e:/projects/Advanced%20ML/python/brain-region-segmentation/webapp/manage.py)
#### [NEW] [webapp/segmentation/apps.py](file:///e:/projects/Advanced%20ML/python/brain-region-segmentation/webapp/segmentation/apps.py)
#### [NEW] [webapp/segmentation/views.py](file:///e:/projects/Advanced%20ML/python/brain-region-segmentation/webapp/segmentation/views.py)
#### [NEW] [webapp/segmentation/urls.py](file:///e:/projects/Advanced%20ML/python/brain-region-segmentation/webapp/segmentation/urls.py)
#### [NEW] [webapp/core_ml/inference_service.py](file:///e:/projects/Advanced%20ML/python/brain-region-segmentation/webapp/core_ml/inference_service.py)

### Machine Learning Integration

The backend will expose an endpoint that:
1. Receives an image file.
2. Preprocesses the file based on its extension (Pydicom for DICOM, OpenCV/NumPy for JPG/PNG).
3. Adapts the data structurally (stacking 5 times, reshaping to `[1, 5, H, W]`).
4. Invokes the exported U-Net model from `model.py`.
5. Post-processes the output (confidence thresholding, area filtering).
6. Generates an overlay visualization (blending the original image with the mask).

### Frontend UI/UX Design

We will discard the simple `index.html` and craft a rich UI using modern Vanilla CSS.
#### [NEW] [webapp/segmentation/templates/index.html](file:///e:/projects/Advanced%20ML/python/brain-region-segmentation/webapp/segmentation/templates/index.html) (Landing page)
#### [NEW] [webapp/segmentation/templates/app.html](file:///e:/projects/Advanced%20ML/python/brain-region-segmentation/webapp/segmentation/templates/app.html) (Application UI)
#### [NEW] [webapp/static/css/styles.css](file:///e:/projects/Advanced%20ML/python/brain-region-segmentation/webapp/static/css/styles.css) (CSS Design System with premium glassmorphism themes)
#### [NEW] [webapp/static/js/app.js](file:///e:/projects/Advanced%20ML/python/brain-region-segmentation/webapp/static/js/app.js) (AJAX interactions for drag-and-drop, loading spinners, and result rendering)

## Verification Plan

### Automated Tests
- Validate that the inference script can accept a generic image tensor and produce an output string/mask without errors.
- Confirm Django can run locally via `python manage.py check`.

### Manual Verification
- Start the Django server and open the web app.
- Check navigation from Landing Page to Application Page.
- Upload an existing JPG/DICOM from `data/ST000001/SE000001/`.
- Ensure the progress state resolves to a fully completed mask and overlay rendering.

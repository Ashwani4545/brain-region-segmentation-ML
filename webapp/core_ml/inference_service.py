"""
inference_service.py
---------------------
Dual-mode segmentation pipeline:

  1. DICOM  → True Hounsfield-Unit thresholding (clinical standard)
             Brain window → hypodense band (HU 15–35) → morphological cleanup
  2. JPG/PNG → Adaptive intensity thresholding (relative to local tissue brightness)

Both modes produce a binary mask, a colour overlay, and a lesion-load confidence
score without requiring any trained model weights.

The UNet2p5D model is still loaded and used when a checkpoint is available;
otherwise the HU / adaptive pipeline is the primary inference method.
"""

import os
import uuid

import cv2
import numpy as np


# ── Hounsfield-Unit helpers (DICOM only) ─────────────────────────────────────

def _apply_window(hu: np.ndarray, wl: float, ww: float) -> np.ndarray:
    """Map HU values through a CT window to [0, 255] uint8."""
    lo = wl - ww / 2
    hi = wl + ww / 2
    windowed = np.clip(hu, lo, hi)
    return ((windowed - lo) / (ww + 1e-8) * 255).astype(np.uint8)


def _get_brain_mask(grey: np.ndarray) -> np.ndarray:
    """
    Approximate skull-stripping: threshold out skull + background,
    keep only the soft-tissue/brain region.
    Returns a binary mask (uint8, 0/255).
    """
    # Threshold: keep pixels in typical brain-tissue brightness range
    _, thresh = cv2.threshold(grey, 10, 255, cv2.THRESH_BINARY)
    # Close small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # Keep only the largest connected region (the head)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num_labels < 2:
        return closed
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    brain_mask = np.zeros_like(closed)
    brain_mask[labels == largest] = 255
    # Erode slightly to exclude skull ring
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    brain_mask = cv2.erode(brain_mask, erode_k, iterations=2)
    return brain_mask


def _segment_dicom(ds, hu_lo=15.0, hu_hi=35.0, min_area=50) -> tuple[np.ndarray, np.ndarray]:
    """
    DICOM segmentation using real HU values.

    Returns
    -------
    img_display : uint8 grayscale (brain-windowed, for overlay)
    mask        : uint8 binary mask (255 = hypodense)
    """
    raw = ds.pixel_array.astype(np.float32)

    # Apply DICOM rescale if present
    slope     = float(getattr(ds, 'RescaleSlope',     1))
    intercept = float(getattr(ds, 'RescaleIntercept', 0))
    hu = raw * slope + intercept

    # Standard brain CT window: WL=35, WW=100
    img_display = _apply_window(hu, wl=35, ww=100)

    # Brain mask (soft tissue only, skull stripped)
    brain_mask = _get_brain_mask(img_display)

    # Hypodense band: HU hu_lo–hu_hi (lower than normal parenchyma ~30–45 HU)
    # These values correspond to acute infarcts / oedema
    hypo_raw = ((hu >= hu_lo) & (hu <= hu_hi)).astype(np.uint8) * 255

    # Restrict to brain region only
    hypo_in_brain = cv2.bitwise_and(hypo_raw, brain_mask)

    # Morphological cleanup: remove speckle, close small gaps
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(hypo_in_brain, cv2.MORPH_OPEN,  k_open)
    cleaned = cv2.morphologyEx(cleaned,       cv2.MORPH_CLOSE, k_close)

    # Remove very small regions (< min_area px) — likely noise
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    final_mask = np.zeros_like(cleaned)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            final_mask[labels == i] = 255

    return img_display, final_mask


def _segment_image(img_grey: np.ndarray, min_area=30) -> np.ndarray:
    """
    Adaptive segmentation for JPG/PNG images that lack HU information.

    Strategy
    --------
    1. CLAHE enhancement for contrast normalisation
    2. Approximate skull strip (keep largest bright region)
    3. Identify pixels significantly darker than local median inside the brain
       (these correspond to hypodense / CSF-like areas)
    4. Morphological cleanup
    """
    # CLAHE for contrast normalisation
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_grey)

    # Approximate brain mask
    brain_mask = _get_brain_mask(enhanced)

    # Compute local tissue median inside the brain
    brain_pixels = enhanced[brain_mask > 0]
    if len(brain_pixels) == 0:
        return np.zeros_like(img_grey)
    tissue_median = float(np.median(brain_pixels))

    # Pixels significantly darker than median → potential hypodense
    # "significantly" = more than 1.5 × MAD below median (robust threshold)
    mad = float(np.median(np.abs(brain_pixels.astype(float) - tissue_median)))
    lo_thr = max(0, tissue_median - 1.5 * mad - 10)
    hi_thr = tissue_median - 0.5 * mad           # must be darker than median

    if lo_thr >= hi_thr:
        # Degenerate case (very uniform image) — use simple lower-quartile band
        lo_thr = float(np.percentile(brain_pixels, 10))
        hi_thr = float(np.percentile(brain_pixels, 30))

    hypo_raw = (
        (enhanced.astype(float) >= lo_thr) &
        (enhanced.astype(float) <= hi_thr)
    ).astype(np.uint8) * 255

    # Apply brain mask
    hypo_in_brain = cv2.bitwise_and(hypo_raw, brain_mask)

    # Morphological cleanup
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned = cv2.morphologyEx(hypo_in_brain, cv2.MORPH_OPEN,  k_open)
    cleaned = cv2.morphologyEx(cleaned,       cv2.MORPH_CLOSE, k_close)

    # Remove tiny specks
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    final_mask = np.zeros_like(cleaned)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            final_mask[labels == i] = 255

    return final_mask


# ── Optional UNet model (used only if checkpoint exists) ─────────────────────

def _try_load_unet(checkpoint_path: str | None, device: str):
    """Load the UNet2p5D model if a checkpoint exists, else return None."""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None
    try:
        import torch
        from core_ml.model import UNet2p5D
        model = UNet2p5D(in_channels=5, out_channels=1)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device).eval()
        print(f"[InferenceService] Loaded model weights from {checkpoint_path}")
        return model
    except Exception as e:
        print(f"[InferenceService] Could not load checkpoint: {e}")
        return None


def _unet_segment(model, img_grey: np.ndarray, device: str) -> np.ndarray:
    """Run UNet inference on a grayscale image; returns binary mask."""
    import torch
    orig_h, orig_w = img_grey.shape
    resized  = cv2.resize(img_grey, (128, 128))
    norm     = resized.astype('float32') / 255.0
    inp      = np.stack([norm] * 5, axis=0)[None, ...]          # (1,5,128,128)
    inp_t    = torch.from_numpy(inp).to(device)
    with torch.no_grad():
        out = model(inp_t).cpu().numpy()[0, 0]                   # (128,128)
    bin_mask = (out > 0.5).astype(np.uint8) * 255
    return cv2.resize(bin_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


# ── Main service class ────────────────────────────────────────────────────────

class InferenceService:
    def __init__(self, checkpoint_path: str | None = None, device: str = 'cpu'):
        self.device = device
        self.model  = _try_load_unet(checkpoint_path, device)
        if self.model is None:
            print(
                "[InferenceService] No model checkpoint — using HU/adaptive "
                "thresholding pipeline (clinically meaningful without training)."
            )

    def process_image(self, input_path: str, output_dir: str, hu_lo=15.0, hu_hi=35.0, min_area_dicom=50, min_area_image=30) -> dict:
        """
        Segment a brain CT image and produce mask + overlay outputs.

        Supports .dcm (DICOM), .jpg, .jpeg, .png.
        """
        filename  = os.path.basename(input_path)
        base_name, _ = os.path.splitext(filename)

        # ── Load ─────────────────────────────────────────────────────────────
        is_dicom = input_path.lower().endswith('.dcm')

        if is_dicom:
            import pydicom
            ds  = pydicom.dcmread(input_path)
            img_display, mask = _segment_dicom(ds, hu_lo=hu_lo, hu_hi=hu_hi, min_area=min_area_dicom)
        else:
            img_grey = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if img_grey is None:
                raise ValueError(f"Failed to load image: {input_path}")
            img_display = img_grey

            # Prefer trained UNet if available
            if self.model is not None:
                mask = _unet_segment(self.model, img_grey, self.device)
            else:
                mask = _segment_image(img_grey, min_area=min_area_image)

        if img_display is None:
            raise ValueError(f"Failed to decode image pixels from: {input_path}")

        # ── Build outputs ─────────────────────────────────────────────────────
        uid              = uuid.uuid4().hex[:8]
        mask_filename    = f"{uid}_{base_name}_mask.png"
        overlay_filename = f"{uid}_{base_name}_overlay.png"
        mask_path        = os.path.join(output_dir, mask_filename)
        overlay_path     = os.path.join(output_dir, overlay_filename)

        # Save mask
        cv2.imwrite(mask_path, mask)

        # Colour overlay: original in grey, hypodense in semi-transparent red
        orig_bgr   = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)
        color_mask = np.zeros_like(orig_bgr)
        color_mask[mask == 255] = [0, 0, 255]           # red (BGR)
        overlay = cv2.addWeighted(orig_bgr, 0.65, color_mask, 0.35, 0)

        # Draw a subtle contour around detected regions for clarity
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 180, 255), 1)  # orange outline

        cv2.imwrite(overlay_path, overlay)

        # ── Confidence = lesion load % (fraction of brain pixels positive) ───
        brain_area    = float(np.sum(_get_brain_mask(img_display) > 0))
        lesion_pixels = float(np.sum(mask > 0))
        if brain_area > 0:
            conf_score = round(min(lesion_pixels / brain_area * 100, 100), 2)
        else:
            conf_score = round(float(np.mean(mask > 0)) * 100, 2)

        detected = conf_score > 0.5   # >0.5 % lesion load = flag as detected

        return {
            'mask_filename':    mask_filename,
            'overlay_filename': overlay_filename,
            'confidence':       f"{conf_score:.2f}%",
            'detected':         detected,
        }


# ── Module-level singleton ────────────────────────────────────────────────────
_service: InferenceService | None = None


def get_inference_service() -> InferenceService:
    global _service
    if _service is None:
        _service = InferenceService()
    return _service

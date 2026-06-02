import os
import uuid
import torch
import numpy as np
import cv2

# model.py lives alongside this file inside core_ml/
from core_ml.model import UNet2p5D


class InferenceService:
    def __init__(self, checkpoint_path=None, device='cpu'):
        self.device = device
        self.model = UNet2p5D(in_channels=5, out_channels=1)

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"Loaded model weights from {checkpoint_path}")
        else:
            print(
                "WARNING: No valid checkpoint found. "
                "Initializing with RANDOM weights — for demonstration only."
            )

        self.model.to(self.device).eval()

    def process_image(self, input_path, output_dir):
        """
        Process a single 2D image (JPG/PNG/DICOM) and return paths to the
        generated mask and overlay images plus a confidence score.
        """
        filename = os.path.basename(input_path)
        base_name, _ = os.path.splitext(filename)

        # ── Load image (grayscale) ──────────────────────────────────────────
        if input_path.lower().endswith('.dcm'):
            import pydicom
            ds = pydicom.dcmread(input_path)
            img = ds.pixel_array
            img = (
                (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8) * 255
            ).astype(np.uint8)
        else:
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Failed to load image: {input_path}")

        # ── Pre-process ─────────────────────────────────────────────────────
        orig_h, orig_w = img.shape
        img_resized = cv2.resize(img, (128, 128))
        img_norm = img_resized.astype('float32') / 255.0

        # Stack single slice into 5 channels (pseudo-2.5D)
        inp = np.stack([img_norm] * 5, axis=0)[None, ...]   # (1, 5, 128, 128)
        inp_t = torch.from_numpy(inp).to(self.device)

        # ── Inference ───────────────────────────────────────────────────────
        with torch.no_grad():
            out = self.model(inp_t).cpu().numpy()[0, 0]     # (128, 128)

        # ── Post-process ────────────────────────────────────────────────────
        thr = 0.5
        bin_mask = (out > thr).astype(np.uint8) * 255

        # Resize back to original resolution
        bin_mask_orig = cv2.resize(
            bin_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )

        # ── Output paths (unique prefix avoids collisions) ──────────────────
        uid = uuid.uuid4().hex[:8]
        mask_filename   = f"{uid}_{base_name}_mask.png"
        overlay_filename = f"{uid}_{base_name}_overlay.png"
        mask_path    = os.path.join(output_dir, mask_filename)
        overlay_path = os.path.join(output_dir, overlay_filename)

        # Save binary mask
        cv2.imwrite(mask_path, bin_mask_orig)

        # Create colour overlay (red highlights on greyscale original)
        orig_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        color_mask = np.zeros_like(orig_color)
        color_mask[bin_mask_orig == 255] = [0, 0, 255]   # red in BGR

        alpha = 0.4
        overlay = cv2.addWeighted(orig_color, 1 - alpha, color_mask, alpha, 0)
        cv2.imwrite(overlay_path, overlay)

        # ── Confidence: fraction of pixels predicted positive (lesion load) ─
        lesion_fraction = float(np.mean(bin_mask_orig > 0))
        conf_score = round(lesion_fraction * 100, 2)      # 0–100 %
        detected   = conf_score > 0.1                      # >0.1 % lesion load

        return {
            'mask_filename':    mask_filename,
            'overlay_filename': overlay_filename,
            'confidence':       f"{conf_score:.2f}%",
            'detected':         detected,
        }


# ── Module-level singleton ───────────────────────────────────────────────────
_service: InferenceService | None = None


def get_inference_service() -> InferenceService:
    global _service
    if _service is None:
        _service = InferenceService()
    return _service

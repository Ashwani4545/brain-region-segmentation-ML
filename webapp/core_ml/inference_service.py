import os
import torch
import numpy as np
import cv2
from PIL import Image

# Add root to python path or import from existing module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from model import UNet2p5D
except ImportError:
    pass

class InferenceService:
    def __init__(self, checkpoint_path=None, device='cpu'):
        self.device = device
        self.model = UNet2p5D(in_channels=5, out_channels=1)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"Loaded model weights from {checkpoint_path}")
        else:
            print("WARNING: No valid checkpoint found. Initializing with RANDOM weights. This is for demonstration purposes only.")
            
        self.model.to(self.device).eval()

    def process_image(self, input_path, output_dir):
        """
        Process a single 2D image (JPG/PNG/DICOM).
        DICOM involves `pydicom` reading, but cv2 can sometimes struggle.
        For simplicity, we'll try cv2 first, then fallback or assume the user converts it, Or implement basic pydicom.
        """
        filename = os.path.basename(input_path)
        base_name, _ = os.path.splitext(filename)
        
        # Load image (Grayscale since medical)
        if input_path.lower().endswith('.dcm'):
            import pydicom
            ds = pydicom.dcmread(input_path)
            img = ds.pixel_array
            # Normalize to 0-255 uint8 for processing
            img = ((img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8) * 255).astype(np.uint8)
        else:
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            
        if img is None:
            raise ValueError("Failed to load image.")
            
        # Resize to (128, 128) expected by model as per configs
        orig_h, orig_w = img.shape
        img_resized = cv2.resize(img, (128, 128))
        
        # Model expects [1, 5, H, W]
        img_norm = img_resized.astype('float32') / 255.0
        # Stack into 5 channels (pseudo-2.5D)
        inp = np.stack([img_norm]*5, axis=0)[None, ...]
        inp_t = torch.from_numpy(inp).to(self.device)
        
        with torch.no_grad():
            out = self.model(inp_t).cpu().numpy()[0, 0] # [H, W]
            
        # Thresholding
        thr = 0.5
        bin_mask = (out > thr).astype(np.uint8) * 255
        
        # Resize back to original
        bin_mask_orig = cv2.resize(bin_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        # Connected area check (optional) handled here or simple threshold is ok
        
        # Create output paths
        mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
        overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
        
        # Save Mask
        cv2.imwrite(mask_path, bin_mask_orig)
        
        # Create Overlay (Original image as RGB + colored mask)
        orig_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Create a red and blue highlight map (Hypodense regions mapped as red)
        color_mask = np.zeros_like(orig_color)
        color_mask[bin_mask_orig == 255] = [0, 0, 255] # Red channel (BGR in OpenCV)
        
        alpha = 0.4
        overlay = cv2.addWeighted(orig_color, 1 - alpha, color_mask, alpha, 0)
        
        cv2.imwrite(overlay_path, overlay)
        
        # Calculate confidence score (average prob of positive pixels)
        pos_probs = out[out > thr]
        conf_score = float(np.mean(pos_probs)) if len(pos_probs) > 0 else 0.0
        
        return {
            'mask_filename': f"{base_name}_mask.png",
            'overlay_filename': f"{base_name}_overlay.png",
            'confidence': f"{conf_score * 100:.2f}%",
            'detected': conf_score > 0
        }

# Global singleton to keep model in memory
_service = None

def get_inference_service():
    global _service
    if _service is None:
        _service = InferenceService()
    return _service

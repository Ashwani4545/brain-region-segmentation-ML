
# inference.py - run segmentation inference on nifti files in a folder
import os, argparse
import torch
import nibabel as nib
import numpy as np
from model import UNet2p5D
from skimage import measure

def load_model(checkpoint, num_slices=5, device='cpu'):
    model = UNet2p5D(in_channels=num_slices)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()
    return model

def run_inference(model, nifti_path, out_path, num_slices=5, thr=0.5):
    img = nib.load(nifti_path).get_fdata()  # z,y,x
    z,y,x = img.shape
    half = num_slices//2
    pred_vol = np.zeros((z,y,x), dtype=np.float32)
    for i in range(z):
        idxs = np.clip(np.arange(i-half, i+half+1), 0, z-1)
        slices = img[idxs,:,:]
        inp = np.stack(slices, axis=0)[None,...]  # 1,C,H,W
        inp_t = torch.from_numpy(inp.astype('float32'))
        with torch.no_grad():
            out = model(inp_t).numpy()[0,0]
        pred_vol[i,:,:] = out
    bin_vol = (pred_vol > thr).astype('uint8')
    # connected component filtering: remove small islands < 100 voxels
    lab = measure.label(bin_vol)
    props = measure.regionprops(lab)
    final = np.zeros_like(bin_vol)
    for p in props:
        if p.area >= 100:
            final[lab==p.label] = 1
    # save as nifti
    nib.Nifti1Image(final.astype('uint8'), affine=np.eye(4)).to_filename(out_path)
    print("Saved:", out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_slices', type=int, default=5)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.model, num_slices=args.num_slices, device=device)
    for f in os.listdir(args.input):
        if f.endswith('.nii') or f.endswith('.nii.gz'):
            run_inference(model, os.path.join(args.input,f), os.path.join(args.output_dir, f.replace('.nii.gz','.mask.nii.gz')), num_slices=args.num_slices)

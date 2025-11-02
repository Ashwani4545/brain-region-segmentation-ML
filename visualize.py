
# visualize.py - quick visualization of image, mask and overlay using matplotlib
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import argparse

def overlay(img, mask, slice_idx=None):
    z = img.shape[0]
    if slice_idx is None:
        slice_idx = z//2
    im = img[slice_idx,:,:]
    m = mask[slice_idx,:,:]
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(im, cmap='gray')
    plt.title('Original slice')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(im, cmap='gray')
    plt.imshow(m, cmap='Reds', alpha=0.4)
    plt.title('Overlay (mask)')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--mask', required=True)
    args = parser.parse_args()
    img = nib.load(args.image).get_fdata()
    mask = nib.load(args.mask).get_fdata()
    overlay(img, mask)

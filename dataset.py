
# dataset.py - PyTorch Dataset for 2.5D slice stacks with optional augmentations
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
import albumentations as A

class NCCT2p5DDataset(Dataset):
    def __init__(self, nifti_paths, mask_paths=None, num_slices=5, patch_size=(128,128), transforms=None, train=True):
        self.nifti_paths = nifti_paths
        self.mask_paths = mask_paths
        self.num_slices = num_slices
        self.patch_size = patch_size
        self.transforms = transforms
        self.train = train
        assert num_slices % 2 == 1, "num_slices should be odd"

    def __len__(self):
        return len(self.nifti_paths)

    def __getitem__(self, idx):
        img = nib.load(self.nifti_paths[idx]).get_fdata()  # z,y,x
        z, y, x = img.shape
        mid = z // 2
        # center crop for simplicity
        cz = mid
        half = self.num_slices // 2
        zs = np.clip(np.arange(cz-half, cz+half+1), 0, z-1)
        slices = img[zs,:,:]
        inp = np.stack([s for s in slices], axis=0).astype('float32')  # C,H,W
        if self.mask_paths:
            m = nib.load(self.mask_paths[idx]).get_fdata()
            mask = (m>0).astype('float32')
            mask_crop = mask[cz,:,:]
        else:
            mask_crop = np.zeros((y,x), dtype='float32')
        if self.transforms:
            a = np.transpose(inp, (1,2,0))
            augmented = self.transforms(image=a, mask=mask_crop)
            a_img = augmented['image']
            a_mask = augmented['mask']
            inp = np.transpose(a_img, (2,0,1))
            mask_crop = a_mask
        return torch.from_numpy(inp), torch.from_numpy(mask_crop).unsqueeze(0)

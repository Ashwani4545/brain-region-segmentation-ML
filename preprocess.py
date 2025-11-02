
# preprocess.py - DICOM/NIfTI loader, windowing, resampling, and simple brain cropping
import os, argparse
import numpy as np
import SimpleITK as sitk
import glob

def load_dicom_series(folder):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(folder)
    if not series_ids:
        raise ValueError("No DICOM series found in %s" % folder)
    series_file_names = reader.GetGDCMSeriesFileNames(folder, series_ids[0])
    reader.SetFileNames(series_file_names)
    image = reader.Execute()
    return image

def window_image(img_array, wl=40, ww=80, out_min=0, out_max=1.0):
    low = wl - ww/2.0
    high = wl + ww/2.0
    img_array = np.clip(img_array, low, high)
    img_array = (img_array - low) / (high - low)
    return img_array

def resample_image(sitk_img, new_spacing=(1.0,1.0,5.0)):
    original_spacing = sitk_img.GetSpacing()
    original_size = sitk_img.GetSize()
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetOutputOrigin(sitk_img.GetOrigin())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled = resampler.Execute(sitk_img)
    return resampled

def save_nifti(np_image, out_path, reference_sitk=None):
    img = sitk.GetImageFromArray(np_image)
    if reference_sitk is not None:
        img.SetSpacing(reference_sitk.GetSpacing())
        img.SetOrigin(reference_sitk.GetOrigin())
        img.SetDirection(reference_sitk.GetDirection())
    sitk.WriteImage(img, out_path)

def process_folder(folder, out_path, wl=40, ww=80, spacing=(1.0,1.0,5.0)):
    print("Processing:", folder)
    try:
        img = load_dicom_series(folder)
    except Exception as e:
        # try reading as nifti
        try:
            img = sitk.ReadImage(folder)
        except Exception as e2:
            print("Skipping", folder)
            return
    arr = sitk.GetArrayFromImage(img).astype('float32')  # z,y,x
    # Window
    arr = window_image(arr, wl=wl, ww=ww)
    # Resample
    res_img = resample_image(sitk.GetImageFromArray(arr), new_spacing=spacing)
    res_arr = sitk.GetArrayFromImage(res_img)
    save_nifti(res_arr, out_path, reference_sitk=res_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Folder with DICOM subfolders or NIfTI files')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--wl', type=float, default=40)
    parser.add_argument('--ww', type=float, default=80)
    parser.add_argument('--spacing', nargs=3, type=float, default=[1.0,1.0,5.0])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for root, dirs, files in os.walk(args.input_dir):
        # treat leaf folders with files as dicom folders
        if files:
            out_name = os.path.basename(root)
            out_path = os.path.join(args.output_dir, out_name + '.nii.gz')
            try:
                process_folder(root, out_path, wl=args.wl, ww=args.ww, spacing=tuple(args.spacing))
            except Exception as e:
                # try direct file read
                if root.lower().endswith(('.nii', '.nii.gz')):
                    try:
                        process_folder(root, out_path, wl=args.wl, ww=args.ww, spacing=tuple(args.spacing))
                    except:
                        pass
                pass


# report_generator.py - simple rule-based report template based on mask and classification
import argparse
import nibabel as nib
import numpy as np

def compute_volume(mask, spacing=(1.0,1.0,5.0)):
    # mask: z,y,x binary
    voxel_vol = spacing[0]*spacing[1]*spacing[2]  # mm^3
    vox_count = mask.sum()
    vol_ml = vox_count * voxel_vol / 1000.0  # convert to mL
    return vol_ml

def get_side_of_lesion(mask):
    # determine left or right by splitting x dimension into halves
    z,y,x = mask.shape
    mids = x//2
    left = mask[:,:, :mids].sum()
    right = mask[:,:, mids:].sum()
    if left==0 and right==0:
        return "None"
    return "Left" if left>right else "Right"

def generate_text_report(image_path, mask_path, classification=None, spacing=(1.0,1.0,5.0)):
    try:
        mask = nib.load(mask_path).get_fdata().astype(bool)
    except Exception as e:
        return "Error reading mask: " + str(e)
    vol_ml = compute_volume(mask, spacing=spacing)
    side = get_side_of_lesion(mask)
    lines = []
    lines.append("Automated NCCT segmentation report")
    lines.append("-------------------------------")
    if classification is not None:
        lines.append(f"Model classification: {'Abnormal' if classification==1 else 'Normal'}")
    if vol_ml<=0.01:
        lines.append("No significant hypodense lesion was detected.")
    else:
        lines.append(f"Hypodense lesion detected in the {side} hemisphere.")
        lines.append(f"Estimated lesion volume: {vol_ml:.2f} mL.")
        if vol_ml < 5.0:
            lines.append("Lesion size: small (<5 mL). Correlate clinically and with diffusion MRI where available.")
        elif vol_ml < 50.0:
            lines.append("Lesion size: moderate (5-50 mL). Consider urgent clinical correlation for ischemia.")
        else:
            lines.append("Lesion size: large (>50 mL). High clinical concern; urgent attention recommended.")
    lines.append("Note: This is an automated, template-based report for research only. Not for clinical use.")
    return "\n".join(lines)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=False)
    parser.add_argument('--mask', required=True)
    parser.add_argument('--classification', type=int, default=None)
    parser.add_argument('--spacing', nargs=3, type=float, default=[1.0,1.0,5.0])
    parser.add_argument('--out', required=False)
    args = parser.parse_args()
    report = generate_text_report(args.image, args.mask, classification=args.classification, spacing=tuple(args.spacing))
    if args.out:
        with open(args.out, 'w') as f:
            f.write(report)
        print('Saved report to', args.out)
    else:
        print(report)

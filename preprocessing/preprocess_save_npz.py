"""
Preprocess raw CCTA NIfTI volumes into the NPZ format consumed by CORA.

For each subject:
    1. Resample the CCTA volume to ISOTROPIC 0.5 x 0.5 x 0.5 mm^3 (linear).
    2. Resample the coronary artery mask to the same grid (nearest neighbor),
       after removing small connected components (< min_component_voxels).
    3. Save a compressed NPZ containing:
         - CTA_HU : raw CCTA volume in Hounsfield Units (D, H, W), float32
         - CA     : binary coronary artery mask (D, H, W), uint8

Output layout (matches pretrain/dataset.py):
    npz_root/<subject>/CTA_<subject>.npz

Spacing matches configs/cora_config.yaml (data.voxel_spacing = [0.5, 0.5, 0.5]).

The coronary artery mask is produced upstream by a segmentation model
(e.g., nnU-Net trained on ImageCAS); see README in this directory.
"""

import os
import sys
import argparse
import multiprocessing as mp

import numpy as np
import SimpleITK as sitk

NEW_SPACING = (0.5, 0.5, 0.5)        # (x, y, z) mm — isotropic
MIN_COMPONENT_VOXELS = 500


def resample(image, new_spacing, interpolator, default_value=0.0):
    """Resample a SimpleITK image to `new_spacing`."""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]
    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing(new_spacing)
    rs.SetSize(new_size)
    rs.SetOutputOrigin(image.GetOrigin())
    rs.SetOutputDirection(image.GetDirection())
    rs.SetTransform(sitk.Transform())
    rs.SetDefaultPixelValue(default_value)
    rs.SetInterpolator(interpolator)
    return rs.Execute(image)


def process_subject(subject_id, source_root, target_root):
    src = os.path.join(source_root, subject_id)
    cta_path = os.path.join(src, f"{subject_id}_CTA.nii.gz")
    mask_path = os.path.join(src, f"CA_{subject_id}_CTA.nii.gz")
    if not (os.path.exists(cta_path) and os.path.exists(mask_path)):
        print(f"[{subject_id}] missing CTA or mask, skipping.")
        return

    dst = os.path.join(target_root, subject_id)
    os.makedirs(dst, exist_ok=True)

    # CCTA -> resample, keep raw HU.
    cta = sitk.ReadImage(cta_path, sitk.sitkFloat32)
    cta = resample(cta, NEW_SPACING, sitk.sitkLinear, default_value=-1000.0)
    cta_hu = sitk.GetArrayFromImage(cta).astype(np.float32)  # (D, H, W)

    # Mask -> binarize, drop small components, resample nearest.
    mask = sitk.ReadImage(mask_path)
    mask = sitk.BinaryThreshold(mask, lowerThreshold=1, upperThreshold=65535,
                                insideValue=1, outsideValue=0)
    comp = sitk.ConnectedComponent(mask)
    comp = sitk.RelabelComponent(comp, minimumObjectSize=MIN_COMPONENT_VOXELS)
    mask = sitk.BinaryThreshold(comp, lowerThreshold=1, insideValue=1, outsideValue=0)
    mask = resample(mask, NEW_SPACING, sitk.sitkNearestNeighbor, default_value=0)
    ca = sitk.GetArrayFromImage(mask).astype(np.uint8)

    out_path = os.path.join(dst, f"CTA_{subject_id}.npz")
    np.savez_compressed(out_path, CTA_HU=cta_hu, CA=ca)
    print(f"[{subject_id}] saved {out_path}  CTA_HU{cta_hu.shape}  CA sum={int(ca.sum())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_root", required=True,
                    help="Directory of <subject>/ folders with raw NIfTI files.")
    ap.add_argument("--target_root", required=True,
                    help="Output directory for <subject>/CTA_<subject>.npz files.")
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    if not os.path.isdir(args.source_root):
        sys.exit(f"Source directory not found: {args.source_root}")
    os.makedirs(args.target_root, exist_ok=True)

    subjects = [d for d in os.listdir(args.source_root)
                if os.path.isdir(os.path.join(args.source_root, d))]
    print(f"Found {len(subjects)} subjects. Resampling to {NEW_SPACING} mm.")

    tasks = [(s, args.source_root, args.target_root) for s in subjects]
    with mp.Pool(processes=args.num_workers) as pool:
        pool.starmap(process_subject, tasks)
    print("Done.")


if __name__ == "__main__":
    main()

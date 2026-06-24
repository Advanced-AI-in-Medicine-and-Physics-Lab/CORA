# Preprocessing

CORA consumes per-subject NPZ files. Each NPZ contains:

| Key | Description |
|-----|-------------|
| `CTA_HU` | Raw CCTA volume in Hounsfield Units, `(D, H, W)`, `float32`. |
| `CA` | Binary coronary artery mask, `(D, H, W)`, `uint8`. |

All volumes are resampled to an **isotropic `0.5 × 0.5 × 0.5 mm³`** grid
(`configs/cora_config.yaml → data.voxel_spacing`).

## Pipeline

1. **Coronary artery mask.** Segment the coronary arteries from each CCTA volume
   with a pretrained segmentation model (we used nnU-Net trained on
   [ImageCAS](https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation)).
   Save as `CA_<subject>_CTA.nii.gz` next to the CCTA volume.
2. **Resample + package.** Run:

   ```bash
   python preprocess_save_npz.py \
       --source_root /path/to/raw_nifti \
       --target_root /path/to/preprocessed/npz \
       --num_workers 8
   ```

   Expected input layout:
   ```
   source_root/
   ├── <subject>/
   │   ├── <subject>_CTA.nii.gz
   │   └── CA_<subject>_CTA.nii.gz
   ```

   Output layout (consumed by `pretrain/dataset.py` and the downstream loaders):
   ```
   target_root/
   ├── <subject>/
   │   └── CTA_<subject>.npz   # keys: CTA_HU, CA
   ```

A patient index file (Excel) listing the subject identifiers is required by the
training scripts. **The pretraining index must contain only pretraining
patients — all internal and external test patients are excluded.**

# Self-supervised pretraining baselines

The self-supervised pretraining baselines compared against CORA under matched
conditions (same 10,138-volume cohort, same multi-window input, same backbone
and compute budget) are:

- **MAE** (Masked Autoencoding)
- **VolumeFusion**
- **VoCo** (Volume Contrast)

All three were trained using the implementations in the **nnssl** framework:

> https://github.com/MIC-DKFZ/nnssl/tree/openneuro

We do not re-distribute these baselines here; please refer to the nnssl
repository for their exact implementations. To reproduce our comparison, train
each baseline with the nnssl framework on the same preprocessed CCTA cohort
(see [`../preprocessing/`](../preprocessing/README.md)) and the same 4-channel
multi-window input and 3D Residual U-Net backbone described in
[`../configs/cora_config.yaml`](../configs/cora_config.yaml), then fine-tune on
the downstream tasks with the scripts in [`../downstream/`](../downstream/).

## Training-from-scratch baseline

The fully-supervised "from scratch" baseline uses the **same 3D Residual U-Net**
(random initialization, no pretraining). Reproduce it by running any downstream
`train.py` **without** the `--pretrained` argument (the encoder is then randomly
initialized).

"""
CORA backbone and downstream heads.

A 3D Residual U-Net (4-stage encoder-decoder, manuscript Table 2) is used for
synthesis-driven self-supervised pretraining. The pretrained encoder is reused
by every downstream task:

    - CORAPretrainModel    : pretraining (lesion abnormality segmentation)
    - CORASegmentationModel: stenosis detection / coronary artery segmentation
    - CORAClassifier       : plaque characterization (volume-level multi-label)
    - CORAMultimodalMACE   : multimodal MACE risk stratification

Architecture constants mirror configs/cora_config.yaml (model: ...).
"""

import torch
from torch import nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from transformers import AutoModel, AutoTokenizer


# =============================================================================
# Core Architecture
# =============================================================================

def build_unet(
    num_input_channels: int = 4,
    num_output_channels: int = 1,
    deep_supervision: bool = False,
) -> nn.Module:
    """
    Build a 3D Residual U-Net (4-stage encoder-decoder).

    Args:
        num_input_channels: Number of input channels (4 for multi-window CCTA).
        num_output_channels: Number of segmentation output channels.
        deep_supervision: Whether to enable deep supervision.

    Returns:
        A ResidualEncoderUNet instance.
    """
    n_stages = 4
    model = ResidualEncoderUNet(
        input_channels=num_input_channels,
        n_stages=n_stages,
        features_per_stage=[32, 64, 128, 256],
        conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3]] * n_stages,
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        n_blocks_per_stage=[1, 3, 4, 4],
        num_classes=num_output_channels,
        n_conv_per_stage_decoder=[1, 1, 1],
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=deep_supervision,
    )
    return model


def load_pretrained_encoder(encoder: nn.Module, checkpoint_path: str, device="cpu"):
    """
    Load CORA-pretrained encoder weights into a downstream encoder.

    Accepts either a raw state_dict, a `{"model_state_dict": ...}` checkpoint,
    or a full CORAPretrainModel state_dict (keys prefixed with `unet.encoder.`).
    Returns the (possibly partially) loaded encoder.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    # Strip the `unet.encoder.` prefix if present (full pretrain checkpoint).
    encoder_weights = {
        k.replace("unet.encoder.", ""): v
        for k, v in state.items()
        if k.startswith("unet.encoder.")
    }
    if not encoder_weights:
        encoder_weights = state  # already an encoder-only state_dict

    missing, unexpected = encoder.load_state_dict(encoder_weights, strict=False)
    print(f"[load_pretrained_encoder] loaded from {checkpoint_path} "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    return encoder


# =============================================================================
# Pooling Layers
# =============================================================================

class AdaptiveConcatPool3d(nn.Module):
    """Concatenation of adaptive average and max pooling (doubles feature dim)."""

    def __init__(self):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool3d(1)
        self.mp = nn.AdaptiveMaxPool3d(1)

    def forward(self, x):
        return torch.cat([self.ap(x), self.mp(x)], dim=1)


# =============================================================================
# Pretraining Model: Abnormality Detection (Segmentation)
# =============================================================================

class CORAPretrainModel(nn.Module):
    """
    CORA pretraining model for synthesis-driven self-supervised learning.

    Performs abnormality segmentation using the full U-Net encoder-decoder.
    The encoder learns pathology-centric representations; the decoder produces
    voxel-level abnormality response maps. An auxiliary classification head
    operates on the bottleneck features.
    """

    def __init__(self, num_input_channels=4, num_output_channels=1):
        super().__init__()
        self.unet = build_unet(num_input_channels, num_output_channels)
        self.bottleneck_channels = 256
        self.adaptive_pool = AdaptiveConcatPool3d()
        self.cls_head = nn.Sequential(
            nn.Linear(self.bottleneck_channels * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        # Encoder: extract multi-scale features
        skips = self.unet.encoder(x)
        bottleneck = skips[-1]

        # Classification branch (auxiliary)
        global_feat = self.adaptive_pool(bottleneck).view(x.size(0), -1)
        logits_cls = self.cls_head(global_feat)

        # Segmentation branch: decode with skip connections
        logits_seg = self.unet.decoder(skips)

        return logits_cls, logits_seg


# =============================================================================
# Downstream: Dense Segmentation (Stenosis / Coronary Artery)
# =============================================================================

class CORASegmentationModel(nn.Module):
    """
    Downstream dense-segmentation model: pretrained CORA encoder + decoder.

    Used for:
      - Stenosis detection (segmentation formulation, lesion-level evaluation).
      - Coronary artery segmentation (ImageCAS).

    The encoder is initialized from CORA-pretrained weights; the decoder is
    randomly initialized and fine-tuned (with a Dice / DiceCE / lesion loss).

    Args:
        num_input_channels: Number of input image channels.
        num_classes: Number of output segmentation channels.
        pretrained_path: Optional path to CORA-pretrained encoder weights.
    """

    def __init__(self, num_input_channels=4, num_classes=1, pretrained_path=None):
        super().__init__()
        self.unet = build_unet(num_input_channels, num_classes)
        if pretrained_path is not None:
            load_pretrained_encoder(self.unet.encoder, pretrained_path)

    def forward(self, x):
        skips = self.unet.encoder(x)
        return self.unet.decoder(skips)


# =============================================================================
# Downstream: Classification (Plaque Characterization)
# =============================================================================

class CORAClassifier(nn.Module):
    """
    Downstream classifier using the pretrained CORA encoder + global pooling +
    linear head, for volume-level multi-label plaque characterization.

    Produces:
    - Head 1: single logit (e.g., binary outcome).
    - Head 2: two logits (calcified / non-calcified plaque).
    """

    def __init__(self, num_input_channels=4, pretrained_path=None):
        super().__init__()

        # Extract encoder from full U-Net
        full_unet = build_unet(num_input_channels, num_output_channels=1)
        self.encoder = full_unet.encoder
        del full_unet.decoder, full_unet

        if pretrained_path is not None:
            load_pretrained_encoder(self.encoder, pretrained_path)

        self.bottleneck_channels = 256
        self.adaptive_pool = AdaptiveConcatPool3d()

        self.cls_head1 = nn.Linear(self.bottleneck_channels * 2, 1)
        self.cls_head2 = nn.Linear(self.bottleneck_channels * 2, 2)

    def forward(self, x):
        skips = self.encoder(x)
        bottleneck = skips[-1]

        global_feat = self.adaptive_pool(bottleneck).view(x.size(0), -1)
        logits1 = self.cls_head1(global_feat)
        logits2 = self.cls_head2(global_feat)

        return logits1, logits2


# =============================================================================
# Downstream: Multimodal MACE Prediction (Image + frozen LLM Text Encoder)
# =============================================================================

class CORAMultimodalMACE(nn.Module):
    """
    Multimodal MACE risk stratification model.

    Fuses CORA image-encoder features with clinical text embeddings from a
    frozen Qwen language model and projected clinical variables.

    Args:
        num_input_channels: Number of input image channels.
        qwen_model_path: Path or HuggingFace ID of the (frozen) Qwen model.
        num_clinical_features: Dimension of the structured clinical feature vector.
        pretrained_path: Optional path to CORA-pretrained encoder weights.
    """

    def __init__(
        self,
        num_input_channels=4,
        qwen_model_path="Qwen/Qwen2.5-7B",
        num_clinical_features=21,
        pretrained_path=None,
    ):
        super().__init__()

        # Image encoder
        full_unet = build_unet(num_input_channels, num_output_channels=1)
        self.encoder = full_unet.encoder
        del full_unet.decoder, full_unet
        if pretrained_path is not None:
            load_pretrained_encoder(self.encoder, pretrained_path)

        self.bottleneck_channels = 256
        self.adaptive_pool = AdaptiveConcatPool3d()

        # Frozen Qwen text encoder
        self.tokenizer = AutoTokenizer.from_pretrained(
            qwen_model_path, trust_remote_code=True
        )
        self.text_model = AutoModel.from_pretrained(
            qwen_model_path, trust_remote_code=True
        )
        for param in self.text_model.parameters():
            param.requires_grad = False
        self.text_model.eval()
        self.text_embed_dim = self.text_model.config.hidden_size

        # Text projection: map LLM hidden dim -> 256
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # Clinical feature projection -> 64
        self.clinical_projection = nn.Sequential(
            nn.Linear(num_clinical_features, 64),
            nn.ReLU(inplace=True),
        )

        # Fusion classifier: image (512) + text (256) + clinical (64) -> risk
        self.combined_dim = (self.bottleneck_channels * 2) + 256 + 64
        self.cls_head = nn.Sequential(
            nn.Linear(self.combined_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def encode_text(self, raw_texts, device):
        inputs = self.tokenizer(
            raw_texts, return_tensors="pt",
            padding=True, truncation=True, max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            return outputs.last_hidden_state.mean(dim=1)

    def forward(self, x_img, raw_texts, clinical_features):
        """
        Args:
            x_img: Image tensor [B, C, D, H, W].
            raw_texts: List of clinical text strings, length B.
            clinical_features: Structured clinical feature tensor [B, num_clinical_features].

        Returns:
            logits: MACE log-risk logits of shape [B, 1].
        """
        skips = self.encoder(x_img)
        img_feat = self.adaptive_pool(skips[-1]).view(x_img.size(0), -1)

        text_embeds = self.encode_text(raw_texts, x_img.device)
        text_feat = self.text_projection(text_embeds)

        clinical_feat = self.clinical_projection(clinical_features.to(x_img.device))

        combined = torch.cat([img_feat, text_feat, clinical_feat], dim=1)
        return self.cls_head(combined)

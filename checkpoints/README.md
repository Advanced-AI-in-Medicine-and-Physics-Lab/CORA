# Checkpoints

Pretrained weights and fine-tuned task checkpoints are released as external
assets (they are too large to track in git).

| Checkpoint | Description | Link |
|------------|-------------|------|
| `cora_pretrained_best.pth` | CORA self-supervised encoder-decoder (best by validation loss) |[Google Drive](https://drive.google.com/file/d/1ATK9GbN5wt89HkSZRUI2z9WoFGNFfwbb/view?usp=sharing) |
| Downstream task checkpoints | Fine-tuned plaque / stenosis / coronary-seg / MACE models | _placeholder — to be added_ |

Once released, place `cora_pretrained_best.pth` in this directory (or pass its
path via `--pretrained` to any downstream `train.py`).

## Minimal load-and-run example

```python
import torch
from models.model import CORAClassifier

# Load the pretrained encoder into a downstream classifier.
clf = CORAClassifier(num_input_channels=4,
                     pretrained_path="checkpoints/cora_pretrained_best.pth")
clf.eval()

# Run on a single 4-channel patch (B, C, D, H, W).
x = torch.randn(1, 4, 96, 96, 96)
with torch.no_grad():
    logit_binary, logits_plaque = clf(x)
print(logit_binary.shape, logits_plaque.shape)
```

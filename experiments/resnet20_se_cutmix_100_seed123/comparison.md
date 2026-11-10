| Model | Test Accuracy | Macro F1 | Cat Recall | Dog Recall | Best Val Accuracy | Training Time |
|---|---:|---:|---:|---:|---:|---:|
| Original CNN + Crop/Flip | 85.95% | 85.94% | 67.40% | 84.90% | 86.76% | 9.98 min |
| ResNet20 + SE (30 Epoch) | 86.26% | 86.20% | 66.40% | 83.50% | 86.84% | 18.91 min |
| ResNet20 + SE + Cosine (100 Epoch) | 88.49% | 88.48% | 77.00% | 82.50% | 89.74% | 65.78 min |
| ResNet20 + SE + CutMix (100 Epoch) | 89.20% | 89.17% | 76.10% | 82.40% | 90.48% | 64.34 min |

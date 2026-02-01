from src.utils import get_best_device, get_model
import pyterrier as pt
import torch
from src.splade import CrossEncoderTeacher, distillation_loss

device = get_best_device()
cross_encoder = CrossEncoderTeacher("cross-encoder/ms-marco-MiniLM-L6-v2")
cross_encoder = cross_encoder.to(device)
print(f"Cross-encoder loaded on {device}")

# Test
test_scores = cross_encoder(
    ["How to install packages?"],
    ["Use pip install to install Python packages."],
)
print(f"Cross-encoder score: {test_scores.item():.4f}")
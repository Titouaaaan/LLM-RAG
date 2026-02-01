from src.utils import get_best_device, get_model
import pyterrier as pt
import torch
from src.splade import compute_splade_representation, SPLADEEncoder, visualize_splade_terms, contrastive_loss, train_splade
from pathlib import Path
import json

splade_baseline_model = "distilbert-base-uncased"
device = get_best_device()
mlm_model, mlm_tokenizer = get_model(splade_baseline_model, device, is_splade=True)

# for sefty put model on eval mode
mlm_model.eval()

print(f"MLM model loaded: {splade_baseline_model} on {device}")
print(f"Vocabulary size: {mlm_tokenizer.vocab_size}")

# Test SPLADE representation
test_text = "How do I install Python packages using pip?"
sparse_rep = compute_splade_representation(test_text, mlm_model, mlm_tokenizer, device)

print(f"Dimension: {sparse_rep.shape}")
print(f"Non-zero terms: {(sparse_rep > 0).sum().item()}")
print(f"Sparsity: {100 * (sparse_rep == 0).sum().item() / len(sparse_rep):.1f}%")

# Display most activated terms
top_indices = torch.topk(sparse_rep, k=15).indices
top_tokens = mlm_tokenizer.convert_ids_to_tokens(top_indices.tolist())
top_values = sparse_rep[top_indices].tolist()

print("\nMost activated terms:")
for token, value in zip(top_tokens, top_values):
    print(f"  {token}: {value:.3f}")

# Test the encoder
splade_encoder = SPLADEEncoder(splade_baseline_model)
splade_encoder = splade_encoder.to(device)
print("\nTesting SPLADE encoder")
print(f"SPLADE encoder loaded on {device}")

test_queries = ["How to install Python packages?"]
test_docs = ["Use pip install to install Python packages from PyPI."]

with torch.no_grad():
    q_rep, d_rep, flops = splade_encoder(test_queries, test_docs)

print(f"Query rep shape: {q_rep.shape}")
print(f"Doc rep shape: {d_rep.shape}")
print(f"FLOPS loss: {flops.item():.4f}")

# Similarity score (dot product)
score = (q_rep * d_rep).sum().item()
print(f"Query-doc score: {score:.4f}")

print("=== Query ===")
visualize_splade_terms(q_rep[0], splade_encoder.tokenizer)
print("\n=== Document ===")
visualize_splade_terms(d_rep[0], splade_encoder.tokenizer)

# Test the loss
batch_q = splade_encoder.encode(["Question 1", "Question 2"])
batch_d = splade_encoder.encode(["Answer 1", "Answer 2"])

loss = contrastive_loss(batch_q, batch_d)
print(f"Contrastive loss: {loss.item():.4f}")

# Train SPLADE with hard negatives (reduced version for the practical)
num_epochs = 20

triplets_path = Path("./outputs/practical-04/splade_training_triplets.json")

with open(triplets_path, "r", encoding="utf-8") as f:
    training_triplets = json.load(f)

print(f"Loaded {len(training_triplets)} training triplets")

splade_output_dir = Path("./outputs/practical-04/splade_model")

model_exists = (splade_output_dir / "config.json").exists()

if model_exists:
    print("Found existing SPLADE model — loading from disk")

    splade_encoder.model.from_pretrained(splade_output_dir)
    splade_encoder.tokenizer.from_pretrained(splade_output_dir)

else:
    print("No trained SPLADE model found — training from scratch")

    train_splade(
        splade_encoder,
        training_triplets,
        num_epochs=num_epochs,
        batch_size=8,
        use_hard_negatives=True,
    )

    print("Saving trained SPLADE model...")
    splade_output_dir.mkdir(parents=True, exist_ok=True)
    splade_encoder.model.save_pretrained(splade_output_dir)
    splade_encoder.tokenizer.save_pretrained(splade_output_dir)
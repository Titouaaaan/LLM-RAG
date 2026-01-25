from rouge_score import rouge_scorer
import pyterrier as pt
import evaluate
from pathlib import Path
from src.data import generate_queries_for_document, get_doc_text
from src.utils import get_best_device, get_model
from src.score import validate_query_generation

# Initialize evaluation metrics
rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
bertscore = evaluate.load(
    "bertscore",
    model_type="distilbert-base-uncased",  # MUCH smaller
)
dataset = pt.get_dataset("irds:lotte/technology/dev/search")
irds = dataset.irds_ref()
index_path = Path("./outputs/practical-04/index_lotte").absolute()
index_ref = str(index_path)
index = pt.IndexFactory.of(index_ref, memory={"meta": True})
meta_index = index.getMetaIndex()
print(f"Index has {index.getCollectionStatistics().getNumberOfDocuments()} documents")

device = get_best_device()
model, tokenizer = get_model(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    device
)

qrels_df = dataset.get_qrels()
queries_df = dataset.get_topics()
sample_doc_id = qrels_df["docno"].iloc[0]
sample_doc = get_doc_text(meta_index, sample_doc_id)

# Validate query generation quality
print("Evaluating query generation quality...")
print("(Generating queries for documents with known real queries)\n")

num_validation_samples = 10


validation_results = validate_query_generation(
    meta_index,
    model,
    tokenizer,
    device,
    rouge,
    bertscore,
    queries_df,
    qrels_df,
    num_samples=num_validation_samples,
    queries_per_doc=8,
)

if "error" not in validation_results:
    print("=== Query Generation Validation ===")
    print(f"Max ROUGE-1 (avg): {validation_results['avg_max_rouge1']:.3f}")
    print(f"Max ROUGE-L (avg): {validation_results['avg_max_rougeL']:.3f}")
    print(f"BERTScore (best): {validation_results['avg_bertscore']:.3f}")

    # Show examples
    examples = validation_results["examples"]
    examples_sorted = sorted(examples, key=lambda x: x["max_rouge1"], reverse=True)

    print("\n--- Best Examples (highest max ROUGE-1) ---")
    for ex in examples_sorted[:2]:
        print(f"  Real query: {ex['real_query']}")
        print(f"  Best generated: {ex['best_generated']}")
        print(f"  Max ROUGE-1: {ex['max_rouge1']:.3f}")
        print(f"  All generated: {ex['generated_queries']}")
        print()

    print("--- Challenging Examples (lower ROUGE) ---")
    for ex in examples_sorted[-2:]:
        print(f"  Real query: {ex['real_query']}")
        print(f"  Best generated: {ex['best_generated']}")
        print(f"  Max ROUGE-1: {ex['max_rouge1']:.3f}")
        print()
else:
    print(f"Validation skipped: {validation_results['error']}")
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from src.data import ir_get_text_from_index
from src.utils import get_best_device, get_model
from src.rag import RAGPipeline
from src.splade import SPLADEEncoder, retriever_to_results
from pathlib import Path
from typing import List, Tuple
import pyterrier as pt
from src.info_retrieval import to_ir_measures_qrels, to_ir_measures_run
import ir_measures


device = get_best_device()

# Load generation model
gen_model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)

# Detect if model is pre-quantized (AWQ/GPTQ) by name
is_quantized = "AWQ" in gen_model_name or "GPTQ" in gen_model_name

if is_quantized:
    gen_model = AutoModelForCausalLM.from_pretrained(
        gen_model_name,
        device_map="auto",
    )
    print(f"Generation model loaded: {gen_model_name} (pre-quantized)")
else:
    gen_model = AutoModelForCausalLM.from_pretrained(
        gen_model_name,
        dtype=torch.float16,
    )
    gen_model = gen_model.to(device)
    print(f"Generation model loaded: {gen_model_name} on {device}")

gen_model.eval()

#data
dataset = pt.get_dataset("irds:lotte/technology/dev/search")
# Configuration
num_docs = 5000  # Limit documents for practical
num_queries = 100  # Limit queries
queries_df = dataset.get_topics().head(num_queries)
qrels_df = dataset.get_qrels()

# Filter qrels for our queries first
query_ids = set(queries_df["qid"].tolist())
qrels_df = qrels_df[qrels_df["qid"].isin(query_ids)].copy()

# Get the set of relevant document IDs we need
relevant_doc_ids = set(qrels_df["docno"].tolist())
doc_id_list = list(relevant_doc_ids)
index_path = Path("./outputs/practical-04/index_lotte").absolute()
index_ref = str(index_path)
index = pt.IndexFactory.of(index_ref, memory={"meta": True})

# SPLADE encoder
splade_baseline_model = "distilbert-base-uncased"
#splade_encoder = SPLADEEncoder(splade_baseline_model)
splade_encoder = SPLADEEncoder("naver/splade-cocondenser-ensembledistil")
splade_encoder = splade_encoder.to(device)

splade_output_dir = Path("./outputs/practical-04/splade_model")

model_exists = (splade_output_dir / "config.json").exists()

""" if model_exists:
    print("Found existing SPLADE model — loading from disk")

    splade_encoder.model.from_pretrained(splade_output_dir)
    splade_encoder.tokenizer.from_pretrained(splade_output_dir)

else:
    print("No custom trained SPLADE model found — using default model")
    # and proceeding with the default initialized model """

rag = RAGPipeline(
    retriever=splade_encoder,
    device=device,
    generator=gen_model,
    generator_tokenizer=gen_tokenizer,
    pt_index=index,
    doc_ids=doc_id_list,
    top_k=3,
)

# Test the RAG pipeline
test_query = "what is ram in computers?"
result = rag(test_query)

print(f"Question: {result['query']}")
print(f"\n{'=' * 80}")
print("Retrieved documents:")

# Loop over retrieved documents and print score + snippet
for doc in result["retrieved_docs"]:
    doc_id = doc["doc_id"]
    score = doc["score"]
    text_snippet = doc["text"].replace("\n", " ")  # truncate & remove newlines
    print(f"  [{score:.2f}] Doc ID: {doc_id}")
    print(f"      Text snippet: {text_snippet}...\n")

print(f"\n{'=' * 80}")
print(f"Generated answer:\n{result['generated_answer']}")

def splade_retriever(query: str) -> List[Tuple[str, float]]:
    return rag.retrieve(query, top_k=20)

bm25 = pt.BatchRetrieve(
    index,
    wmodel="BM25",
    controls={"ql.type": "literal"}  # literal mode disables operator parsing
)
metrics = [ir_measures.RR, ir_measures.Recall@1, ir_measures.Recall@5, ir_measures.Recall@10]
qrels_ir = to_ir_measures_qrels(qrels_df)


test_queries = queries_df.tail(20)  # Use last queries as test
splade_results = retriever_to_results(splade_retriever, test_queries)
splade_metrics = ir_measures.calc_aggregate(
    metrics, qrels_ir, to_ir_measures_run(splade_results)
)

print("\n=== SPLADE Results ===")
for metric, value in splade_metrics.items():
    print(f"  {metric}: {value:.4f}")

# Compare with BM25
bm25_results = bm25.transform(test_queries[["qid", "query"]])
bm25_metrics = ir_measures.calc_aggregate(
    metrics, qrels_ir, to_ir_measures_run(bm25_results)
)

print("\n=== BM25 Results ===")
for metric, value in bm25_metrics.items():
    print(f"  {metric}: {value:.4f}")

# Comparison table
print("\n=== Comparison ===")
print(f"{'Metric':<12} {'BM25':>10} {'SPLADE':>10} {'Diff':>10}")
print("-" * 44)
for metric in splade_metrics:
    bm25_val = bm25_metrics[metric]
    splade_val = splade_metrics[metric]
    diff = splade_val - bm25_val
    diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
    print(f"{metric!s:<12} {bm25_val:>10.4f} {splade_val:>10.4f} {diff_str:>10}")
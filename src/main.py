from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from src.data import ir_get_text_from_index
from src.utils import get_best_device, get_model
from src.rag import RAGPipeline
from src.splade import SPLADEEncoder
from pathlib import Path
import pyterrier as pt

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
    # and proceeding with the default initialized model
 """
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

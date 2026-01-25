import pyterrier as pt
from pathlib import Path
from src.utils import get_best_device, get_model
from src.data import ir_get_doc_text
from src.info_retrieval import to_ir_measures_qrels, to_ir_measures_run
import ir_measures
from ir_measures import RR, Recall

dataset = pt.get_dataset("irds:lotte/technology/dev/search")
irds = dataset.irds_ref()
index_path = Path("./outputs/practical-04/index_lotte").absolute()
index_ref = str(index_path)
index = pt.terrier.TerrierIndex(str(index_path))
meta_index = index.meta_index()
print(f"Index has {index.collection_statistics().getNumberOfDocuments()} documents")

device = get_best_device()
model, tokenizer = get_model(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    device
)

qrels_df = dataset.get_qrels()
queries_df = dataset.get_topics()

# Filter qrels for our queries first
query_ids = set(queries_df["qid"].tolist())
qrels_df = qrels_df[qrels_df["qid"].isin(query_ids)].copy()

# Get the set of relevant document IDs we need
relevant_doc_ids = set(qrels_df["docno"].tolist())

print("Queries and Qrels loaded:")
print(f"  - Queries: {len(queries_df)}")
print(f"  - Qrels: {len(qrels_df)}")

bm25 = pt.BatchRetrieve(
    index,
    wmodel="BM25",
    controls={"ql.type": "literal"}  # literal mode disables operator parsing
)

# Test search
test_query = "How do I install Python packages?"
results = bm25.search(test_query)
print(f"Top 3 results for: '{test_query}'")
for i, row in results.head(3).iterrows():
    doc_text = ir_get_doc_text(index, row["docno"])
    print(f"\n[{i + 1}] Score: {row['score']:.2f}")
    print(f"    Doc: {doc_text[:100]}...")

qrels_ir = to_ir_measures_qrels(qrels_df)

queries_df_clean = queries_df.copy()
queries_df_clean["query"] = queries_df_clean["query"].str.replace(r'[^\w\s?]', ' ', regex=True)

results = bm25.transform(queries_df_clean[["qid", "query"]])

# Compute metrics using ir-measures
metrics = [RR, Recall @ 1, Recall @ 5, Recall @ 10]
eval_results = ir_measures.calc_aggregate(
    metrics, qrels_ir, to_ir_measures_run(results)
)

print("BM25 Baseline:")
for metric, value in eval_results.items():
    print(f"  {metric}: {value:.4f}") 
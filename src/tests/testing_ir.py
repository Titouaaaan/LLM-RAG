import pyterrier as pt
from pathlib import Path
from src.utils import get_best_device, get_model
from src.data import ir_get_doc_text, ir_get_text_from_index, ir_doc_exists_in_index
from src.info_retrieval import create_training_triplets, to_ir_measures_qrels, to_ir_measures_run, find_hard_negatives_bm25
import ir_measures
from ir_measures import RR, Recall
import json
from tqdm import tqdm

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

# Test hard negative mining
test_qid = qrels_df["qid"].iloc[0]
test_query_row = queries_df[queries_df["qid"] == test_qid].iloc[0]
test_query_text = test_query_row["query"]
test_positives = set(qrels_df[qrels_df["qid"] == test_qid]["docno"].tolist())

hard_negs = find_hard_negatives_bm25(
    test_query_text, test_positives, bm25, num_negatives=3
)

print(f"Query: {test_query_text}")
print(f"\nPositive documents ({len(test_positives)}):")
pos_texts = ir_get_text_from_index(index, list(test_positives)[:2])
for pos_id, pos_text in pos_texts.items():
    print(f"  {pos_text[:100]}...")
print(f"\nHard negatives ({len(hard_negs)}):")
neg_texts = ir_get_text_from_index(index, hard_negs)
for neg_id in hard_negs:
    print(f"  {neg_texts.get(neg_id, '')[:100]}...")

print("\nLoading synthetic training data created in testing_data.py...")

synthetic_training_path = Path("./outputs/practical-04/training_data_for_splade.json")
synthetic_pairs = []

if synthetic_training_path.exists():
    with open(synthetic_training_path, "r", encoding="utf-8") as f:
        synthetic_data = json.load(f)

    # Convert to query-positive pairs (filter to docs in our index)
    for item in synthetic_data:
        # Check if doc exists in index
        if ir_doc_exists_in_index(index, item["doc_id"]):
            synthetic_pairs.append(
                {
                    "query": item["query"],
                    "doc_id": item["doc_id"],
                    "source": item.get("source", "synthetic"),
                }
            )

    print(f"Loaded {len(synthetic_pairs)} synthetic training pairs from Practical 04")
else:
    print(f"No synthetic training data found at {synthetic_training_path}")
    print(
        "Run Practical 04 first to generate synthetic queries, or continue with qrels only."
    )

# Create training data with hard negatives
num_train_queries = 50


# Create triplets from qrels (original data)
training_triplets = create_training_triplets(
    queries_df,
    qrels_df,
    index,
    bm25,
    num_hard_negatives=3,
    max_queries=num_train_queries,
)

print(f"\nTraining triplets from qrels: {len(training_triplets)}")

# Add triplets from synthetic pairs (if available)
if synthetic_pairs:
    print(f"Adding triplets from {len(synthetic_pairs)} synthetic pairs...")

    for pair in tqdm(
        synthetic_pairs[:num_train_queries], desc="Processing synthetic pairs"
    ):
        query_text = pair["query"]
        positive_id = pair["doc_id"]
        positive_text = ir_get_doc_text(index, positive_id)

        if not positive_text:
            continue

        # Find hard negatives for synthetic queries too
        hard_neg_ids = find_hard_negatives_bm25(
            query_text, {positive_id}, bm25, num_negatives=3
        )
        hard_neg_texts = ir_get_text_from_index(index, hard_neg_ids)
        hard_neg_list = [
            hard_neg_texts[nid] for nid in hard_neg_ids if nid in hard_neg_texts
        ]

        if hard_neg_list:
            training_triplets.append(
                {
                    "query": query_text,
                    "positive": positive_text,
                    "negatives": hard_neg_list,
                }
            )

    print(f"Total training triplets (qrels + synthetic): {len(training_triplets)}")
if training_triplets:
    print("Example triplet:")
    print(f"  Query: {training_triplets[0]['query'][:60]}...")
    print(f"  Positive: {training_triplets[0]['positive'][:60]}...")
    print(f"  Num negatives: {len(training_triplets[0]['negatives'])}")

triplets_path = Path("./outputs/practical-04/splade_training_triplets.json")

with open(triplets_path, "w", encoding="utf-8") as f:
    json.dump(training_triplets, f, indent=2, ensure_ascii=False)

print(f"Saved {len(training_triplets)} training triplets to {triplets_path}")
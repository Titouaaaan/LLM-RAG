import pyterrier as pt
import json
from pathlib import Path
import shutil
from src.data import create_combined_training_data, get_doc_text, get_text_from_index
from src.utils import get_best_device, get_model
from src.data import generate_queries_for_document, generate_training_pairs, filter_synthetic_pairs

# Load the LoTTE technology dataset via PyTerrier's ir-datasets integration
dataset = pt.get_dataset("irds:lotte/technology/dev/search")
print("Dataset loaded: lotte/technology/dev/search")

# Explore the structure using PyTerrier's dataset methods
print("\nDataset info:")
# Get counts from the underlying ir-datasets object
irds = dataset.irds_ref()
print(f"  - Documents: {irds.docs_count()}")
print(f"  - Queries: {irds.queries_count()}")
print(f"  - Relevance judgments (qrels): {irds.qrels_count()}")

# Display some documents using PyTerrier's corpus iterator
print("\nExample documents (StackExchange answers):")
for i, doc in enumerate(dataset.get_corpus_iter()):
    if i >= 3:
        break
    print(f"\n[{i+1}] ID: {doc['docno']}")
    print(f"    Text: {doc['text'][:200]}...")

# Display some queries using PyTerrier's topics method
topics_df = dataset.get_topics()
print("\nExample queries:")
print(topics_df.head())

# Display some relevance judgments using PyTerrier's qrels method
qrels_df_all = dataset.get_qrels()
print("\nExample qrels (query-document relevance):")
print(qrels_df_all.head())

# Get queries and qrels
queries_df = dataset.get_topics()
qrels_df = dataset.get_qrels()

print("Loaded:")
print(f"  - Queries: {len(queries_df)}")
print(f"  - Qrels: {len(qrels_df)}")


# Index the entire corpus with PyTerrier (or load existing index)
# We store document text in metadata for retrieval during RAG
index_path = Path("./outputs/practical-04/index_lotte").absolute()

# Check if index already exists
if (index_path / "data.properties").exists():
    print(f"Loading existing index from {index_path}")
    index_ref = str(index_path)
else:
    print(f"Creating new index at {index_path}")
    if index_path.is_dir():
        shutil.rmtree(index_path)
    index_path.mkdir(parents=True, exist_ok=True)

    indexer = pt.IterDictIndexer(
        str(index_path),
        overwrite=True,
        meta={"docno": 50, "text": 4096},  # Store text in metadata
        meta_reverse=["docno"],
    )

    # Index the corpus - PyTerrier handles iteration efficiently
    print("Indexing corpus...")
    index_ref = indexer.index(dataset.get_corpus_iter())

# Get index statistics
index = pt.IndexFactory.of(index_ref, memory={"meta": True})
meta_index = index.getMetaIndex()
print(f"Index has {index.getCollectionStatistics().getNumberOfDocuments()} documents")

# Configuration for query generation
num_docs_to_augment = 50  # Number of documents to generate queries for
queries_per_doc = 3  # Number of queries to generate per document

device = get_best_device()
model, tokenizer = get_model(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    device
)

sample_doc_id = qrels_df["docno"].iloc[0]
sample_doc = get_doc_text(meta_index, sample_doc_id)

print(f"Document ({sample_doc_id}):")
print(f"{sample_doc}")
print()

generated = generate_queries_for_document(
    document=sample_doc,
    num_queries=queries_per_doc,
    model=model,
    tokenizer=tokenizer,
    device=device,
    max_new_tokens=50,
    temperature=0.7,
)

print("Generated queries:")
for i, q in enumerate(generated, 1):
    print(f"  {i}. {q}")

# Generate synthetic training pairs
print(f"Generating queries for {num_docs_to_augment} documents...")
synthetic_pairs = generate_training_pairs(
    index_ref,
    qrels_df,
    model,
    tokenizer,
    device,
    num_docs=num_docs_to_augment,
    queries_per_doc=queries_per_doc,
)

print(f"\nGenerated {len(synthetic_pairs)} synthetic query-document pairs")

# Show examples
print("\nExample synthetic pairs:")
for pair in synthetic_pairs[:3]:
    print(f"  Query: {pair['query']}")
    print(f"  Doc: {pair['document'][:100]}...")
    print()

filtered_pairs = filter_synthetic_pairs(synthetic_pairs)

print(f"Filtered: {len(synthetic_pairs)} -> {len(filtered_pairs)} pairs")
print(f"Kept: {100 * len(filtered_pairs) / max(1, len(synthetic_pairs)):.1f}%")

# Create combined training data
combined_training = create_combined_training_data(
    index_ref, meta_index, qrels_df, queries_df, filtered_pairs
)

print(f"Combined training data: {len(combined_training)} pairs")
print(f"  - From qrels: {sum(1 for p in combined_training if p['source'] == 'qrels')}")
print(
    f"  - Synthetic: {sum(1 for p in combined_training if p['source'] == 'synthetic')}"
)

# Prepare data for saving (don't include full document text to save space)
training_export = []
for pair in combined_training:
    training_export.append(
        {
            "query": pair["query"],
            "doc_id": pair["doc_id"],
            "source": pair["source"],
        }
    )

output_dir = Path("./outputs/practical-04")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "training_data_for_splade.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(training_export, f, indent=2)

print(f"Saved {len(training_export)} training pairs to {output_path}")
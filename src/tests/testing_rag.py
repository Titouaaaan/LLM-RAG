import pyterrier as pt
from pathlib import Path
from src.data import get_doc_text
from src.rag import simple_rag, answer_without_retrieval
from src.utils import get_best_device, get_model

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

# Create BM25 retriever using the index we created earlier
bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25")
print("BM25 retriever ready")

# Test RAG with BM25
test_questions_rag = [
    "How do I install Python packages?",
    "What is the difference between a list and a tuple?",
]

print("=== RAG with BM25 ===\n")
for question in test_questions_rag[:1]:  # Just one for the practical
    result = simple_rag(question, bm25, index_ref, model, tokenizer, device, meta_index, top_k=3)

    print(f"Question: {result['question']}")
    print(f"\nRetrieved documents ({len(result['retrieved_docs'])}):")
    for doc_info in result["retrieved_docs"]:
        print(f"  - {doc_info['doc_id'][:20]}... (score: {doc_info['score']:.2f})")
    print(f"\nGenerated answer:\n{result['answer'][:500]}")
    print("\n" + "=" * 80)

# Compare approaches
comparison_question = "How do I debug a Python program?"

print(f"Question: {comparison_question}\n")

# Without retrieval
print("=== WITHOUT RETRIEVAL ===")
direct_answer = answer_without_retrieval(comparison_question, model, tokenizer, device)
print(direct_answer[:400])

print("\n" + "=" * 80)

# With RAG
print("\n=== WITH RAG (BM25) ===")
rag_result = simple_rag(comparison_question, bm25, index_ref, model, tokenizer, device, meta_index, top_k=3)
print(rag_result["answer"][:400])
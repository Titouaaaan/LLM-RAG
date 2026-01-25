from src.prompting import build_prompt, generate_text
from typing import List, Dict
import pyterrier as pt
import random
from tqdm import tqdm

# Helper functions to get document text from index (uses cached meta_index)
def get_doc_text(meta_index, doc_id: str) -> str:
    """Retrieve document text from PyTerrier index metadata."""
    try:
        docid = meta_index.getDocument("docno", doc_id)
        if docid >= 0:
            return meta_index.getItem("text", docid)
    except Exception:
        pass
    return ""


def get_text_from_index(meta_index, doc_ids: list[str]) -> dict[str, str]:
    """Retrieve text for multiple documents from the index metadata."""
    result = {}
    for doc_id in doc_ids:
        try:
            docid = meta_index.getDocument("docno", doc_id)
            if docid >= 0:
                result[doc_id] = meta_index.getItem("text", docid)
        except Exception:
            pass
    return result

def safe_corpus_iter(corpus_iter):
    for doc in corpus_iter:
        try:
            doc['text'] = doc['text'].encode('utf-8', errors='ignore').decode('utf-8')
            yield doc
        except Exception:
            continue
    
def generate_queries_for_document(
    document: str,
    num_queries: int = 2,
    max_doc_length: int = 500,
    *,
    model,
    tokenizer,
    device,
    temperature: float = 0.7,
    max_new_tokens: int = 50,
) -> List[str]:
    """
    Generate search queries that the document would answer.

    Args:
        document: The document text
        num_queries: Number of queries to generate
        max_doc_length: Maximum document length to use
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        device: torch device
        temperature: Sampling temperature
        max_new_tokens: Max tokens to generate

    Returns:
        List of generated queries
    """
    if not document.strip():
        return []

    # Truncate document
    doc_excerpt = document[:max_doc_length]

    # Clear, instruction-following prompt
    user_prompt = (
        f"Given the following document, generate {num_queries} concise search queries "
        f"that a user might type to find this document.\n"
        f"Each query must be on its own line.\n"
        f"Do not number the queries.\n\n"
        f"Document:\n{doc_excerpt}\n\n"
        f"Queries:"
    )

    # Build chat-style prompt (important for instruct models)
    prompt = build_prompt(
        tokenizer=tokenizer,
        user=user_prompt,
    )

    # Generate text
    outputs = generate_text(
        tokenizer=tokenizer,
        model=model,
        device=device,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        num_return_sequences=1,
    )[0]

    generated_queries = [q.strip() for q in outputs.split('\n') if q.strip()]

    # Return only the requested number of queries
    return generated_queries[:num_queries]

def generate_training_pairs(
    index_ref,
    qrels_df,
    model,
    tokenizer,
    device,
    num_docs: int = 50,
    queries_per_doc: int = 2,
    max_doc_length: int = 500,
) -> List[Dict]:
    """
    Generate query-document training pairs.

    Args:
        index_ref: PyTerrier index reference
        qrels_df: DataFrame with qrels (to get document IDs)
        num_docs: Number of documents to process
        queries_per_doc: Queries to generate per document
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        device: torch device
        max_doc_length: Max document length for prompting

    Returns:
        List of {"query": str, "doc_id": str, "document": str, "source": str}
    """

    # Load index + metadata
    index = pt.IndexFactory.of(index_ref, memory={"meta": True})
    meta_index = index.getMetaIndex()

    # Sample document IDs from qrels (guaranteed indexed)
    doc_ids = qrels_df["docno"].unique().tolist()
    sampled_ids = random.sample(doc_ids, min(num_docs, len(doc_ids)))

    training_pairs = []

    for doc_id in tqdm(sampled_ids, desc="Generating synthetic queries"):
        # Resolve internal docid
        internal_id = meta_index.getDocument("docno", doc_id)
        if internal_id < 0:
            continue

        doc_text = meta_index.getItem("text", internal_id)
        if not doc_text or not doc_text.strip():
            continue

        # Generate queries
        queries = generate_queries_for_document(
            document=doc_text,
            num_queries=queries_per_doc,
            max_doc_length=max_doc_length,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

        for query in queries:
            training_pairs.append(
                {
                    "query": query,
                    "doc_id": doc_id,
                    "document": doc_text,
                    "source": "synthetic",
                }
            )

    return training_pairs

def filter_synthetic_pairs(
    pairs: List[Dict],
    min_query_words: int = 3,
    max_query_words: int = 20,
) -> List[Dict]:
    """
    Filter synthetic query-document pairs for quality.

    Args:
        pairs: List of training pairs
        min_query_words: Minimum query length
        max_query_words: Maximum query length

    Returns:
        Filtered list of pairs
    """
    filtered = []
    seen_queries = set()

    for pair in pairs:
        query = pair["query"]

        # Length filter
        query_words = len(query.split())
        if query_words < min_query_words or query_words > max_query_words:
            continue

        # Duplicate filter
        query_lower = query.lower()
        if query_lower in seen_queries:
            continue
        seen_queries.add(query_lower)

        filtered.append(pair)

    return filtered
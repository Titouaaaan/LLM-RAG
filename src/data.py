from src.prompting import build_prompt, generate_text
from typing import List, Dict
import pyterrier as pt
import random
from tqdm import tqdm
import pandas as pd

# Helper functions to get document text from index (uses cached meta_index)
def get_doc_text(pt_index, doc_id: str) -> str:
    """Retrieve document text from PyTerrier index metadata."""
    try:
        # PyTerrier metadata retrieval - docid is the internal ID
        metadata = pt_index.getMetaIndex()
        docid = metadata.getDocument("docno", doc_id)
        
        if docid >= 0:
            text = metadata.getItem("text", docid)
            return text if text else ""
    except Exception as e:
        print(f"Warning: Could not retrieve text for doc_id {doc_id}: {e}")
    
    return ""


def get_text_from_index(pt_index, doc_ids: list[str]) -> dict[str, str]:
    """Retrieve text for multiple documents from the index metadata."""
    result = {}
    try:
        metadata = pt_index.getMetaIndex()
        
        for doc_id in doc_ids:
            try:
                docid = metadata.getDocument("docno", doc_id)
                if docid >= 0:
                    text = metadata.getItem("text", docid)
                    if text:
                        result[doc_id] = text
            except Exception:
                pass
    except Exception as e:
        print(f"Error accessing metadata index: {e}")
    
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
    user_prompt = f"""
        You are generating search engine queries.

        Generate {num_queries} search queries that could retrieve the document below.

        Rules:
        - Each query must be 3 to 8 words
        - Use keyword-style phrases (not full sentences)
        - Do NOT use question words (how, what, why, when, where)
        - Do NOT use punctuation
        - Do NOT number the queries
        - One query per line
        - Prefer noun phrases over verbs

        Document:
        {doc_excerpt}

        Queries:
        """.strip()

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

def create_combined_training_data(
    index_ref,
    meta_index,
    qrels_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    synthetic_pairs: List[Dict],
) -> List[Dict]:
    """
    Combine real qrels with synthetic pairs.

    Args:
        index_ref: PyTerrier index reference
        qrels_df: Original relevance judgments
        queries_df: Original queries
        synthetic_pairs: Synthetically generated pairs

    Returns:
        Combined list of training pairs
    """
    training_data = []

    # Add real pairs from qrels
    for _, row in qrels_df.iterrows():
        qid = row["qid"]
        doc_id = row["docno"]

        query_rows = queries_df[queries_df["qid"] == qid]["query"].values
        if len(query_rows) == 0:
            continue

        doc_text = get_doc_text(meta_index, doc_id)
        if not doc_text:
            continue

        training_data.append(
            {
                "query": query_rows[0],
                "doc_id": doc_id,
                "document": doc_text,
                "source": "qrels",  # From original dataset
            }
        )

    # Add synthetic pairs
    training_data.extend(synthetic_pairs)

    return training_data

def ir_get_text_from_index(
    index: pt.terrier.TerrierIndex, doc_ids: List[str]
) -> Dict[str, str]:
    """
    Retrieve document text from the index metadata.

    Args:
        index: PyTerrier index reference
        doc_ids: List of document IDs

    Returns:
        Dict mapping doc_id -> text
    """
    meta_index = index.meta_index()

    result = {}
    for doc_id in doc_ids:
        try:
            # Get internal docid from docno
            docid = meta_index.getDocument("docno", doc_id)
            if docid >= 0:
                text = meta_index.getItem("text", docid)
                result[doc_id] = text
        except Exception:
            pass  # Document not found

    return result


# Helper for single document lookup
def ir_get_doc_text(index, doc_id: str) -> str:
    """Get text for a single document from the index."""
    texts = ir_get_text_from_index(index, [doc_id])
    return texts.get(doc_id, "")


def ir_doc_exists_in_index(index, doc_id: str) -> bool:
    """Check if a document exists in the index (without fetching text)."""
    meta_index = index.meta_index()
    try:
        return meta_index.getDocument("docno", doc_id) >= 0
    except Exception:
        return False
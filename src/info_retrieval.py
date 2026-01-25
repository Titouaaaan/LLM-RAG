from pathlib import Path
from typing import List, Dict
from src.data import ir_get_doc_text, ir_get_text_from_index
import pandas as pd
from tqdm import tqdm

# Convert PyTerrier dataframes to ir-measures format
def to_ir_measures_qrels(qrels_df):
    """Convert PyTerrier qrels to ir-measures format."""
    return qrels_df.rename(
        columns={"qid": "query_id", "docno": "doc_id", "label": "relevance"}
    )


def to_ir_measures_run(run_df):
    """Convert PyTerrier run to ir-measures format."""
    return run_df.rename(columns={"qid": "query_id", "docno": "doc_id"})

def find_hard_negatives_bm25(
    query: str,
    positive_doc_ids: set,
    retriever,
    num_negatives: int = 5,
) -> List[str]:
    """
    Find hard negatives using BM25.

    Hard negatives are documents that are lexically similar to the query
    but are NOT the ground truth positive document.

    Args:
        query: The query text
        positive_doc_ids: Set of positive document IDs (to exclude)
        retriever: BM25 retriever
        num_negatives: Number of hard negatives to return

    Returns:
        List of document IDs for hard negatives
    """
    # Implement hard negative mining with BM25

    # 1. Retrieve top documents with BM25
    results = retriever.search(query)
    
    # 2. Filter out the positive documents
    hard_negatives = []
    for _, row in results.iterrows():
        docno = row["docno"]
        # Skip positive documents
        if docno not in positive_doc_ids:
            hard_negatives.append(docno)
        # Stop once we have enough hard negatives
        if len(hard_negatives) >= num_negatives:
            break
    
    # 3. Return the top-k remaining as hard negatives
    return hard_negatives

def create_training_triplets(
    queries_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    index,
    retriever,
    num_hard_negatives: int = 3,
    max_queries: int = None,
) -> List[Dict]:
    """
    Create training triplets with hard negatives.

    Each triplet contains:
    - query: The question
    - positive: A relevant document
    - negatives: List of hard negative documents

    Args:
        queries_df: DataFrame with queries
        qrels_df: DataFrame with relevance judgments
        index: PyTerrier index reference (for text retrieval)
        retriever: BM25 retriever for hard negative mining
        num_hard_negatives: Number of hard negatives per query
        max_queries: Maximum number of queries to process

    Returns:
        List of training triplets
    """
    triplets = []

    queries_to_process = queries_df.head(max_queries) if max_queries else queries_df

    for _, query_row in tqdm(
        queries_to_process.iterrows(),
        desc="Creating training triplets",
        total=len(queries_to_process),
    ):
        qid = query_row["qid"]
        query_text = query_row["query"]

        # Get positive documents for this query
        positive_doc_ids = set(qrels_df[qrels_df["qid"] == qid]["docno"].tolist())

        if not positive_doc_ids:
            continue

        # Get the first positive document
        positive_id = list(positive_doc_ids)[0]
        positive_text = ir_get_doc_text(index, positive_id)

        if not positive_text:
            continue

        # Find hard negatives
        hard_neg_ids = find_hard_negatives_bm25(
            query_text, positive_doc_ids, retriever, num_hard_negatives
        )
        hard_neg_texts = ir_get_text_from_index(index, hard_neg_ids)
        hard_neg_list = [
            hard_neg_texts[nid] for nid in hard_neg_ids if nid in hard_neg_texts
        ]

        if hard_neg_list:
            triplets.append(
                {
                    "query": query_text,
                    "positive": positive_text,
                    "negatives": hard_neg_list,
                }
            )

    return triplets
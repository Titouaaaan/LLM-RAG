import re
import unicodedata
from typing import List, Dict
from rouge_score import rouge_scorer
from tqdm import tqdm
import pandas as pd
from src.data import get_doc_text, generate_queries_for_document
from src.utils import get_best_device, get_model
import random
import pyterrier as pt

STOP_PREFIXES = [
    "find", "show me", "give me", "search for", "what is", "what are"
]

def normalize_query(q: str) -> str:
    q = q.lower()
    q = unicodedata.normalize("NFKD", q)
    q = re.sub(r"[^\w\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()

    for p in STOP_PREFIXES:
        if q.startswith(p + " "):
            q = q[len(p):].strip()

    return q

def compute_rouge(prediction: str, references, rouge) -> dict:
    """
    prediction: str
    references: str or List[str]
    """
    prediction = normalize_query(prediction)

    if isinstance(references, str):
        references = [references]

    best_scores = {"rouge1": 0.0, "rougeL": 0.0}

    for ref in references:
        ref = normalize_query(ref)
        scores = rouge.score(ref, prediction)

        best_scores["rouge1"] = max(
            best_scores["rouge1"], scores["rouge1"].fmeasure
        )
        best_scores["rougeL"] = max(
            best_scores["rougeL"], scores["rougeL"].fmeasure
        )

    return best_scores



def compute_bertscore(predictions: List[str], references: List[str], bertscore, device) -> dict:
    results = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
        device=device,
    )
    return {
        "precision": sum(results["precision"]) / len(results["precision"]),
        "recall": sum(results["recall"]) / len(results["recall"]),
        "f1": sum(results["f1"]) / len(results["f1"]),
    }


def validate_query_generation(
    meta_index,
    model,
    tokenizer,
    device,
    rouge,
    bertscore,
    queries_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    num_samples: int = 10,
    queries_per_doc: int = 3,
) -> Dict:
    """
    Validate query generation by comparing to real queries for the same document.

    For documents with known queries, generate synthetic queries and compare
    to the ground truth. Takes the max score over multiple generations.

    Args:
        index_ref: PyTerrier index reference
        queries_df: DataFrame with real queries
        qrels_df: DataFrame with relevance judgments
        num_samples: Number of documents to evaluate
        queries_per_doc: Number of queries to generate per document

    Returns:
        Dictionary with validation metrics and examples
    """
    # Implement query validation

    doc_to_queries = {}  # Map doc_id -> list of real queries
    for _, row in qrels_df.iterrows():
        doc_id = row["docno"]
        qid = row["qid"]
        query = queries_df[queries_df["qid"] == qid]["query"].iloc[0]

        if doc_id not in doc_to_queries:
            doc_to_queries[doc_id] = []
        doc_to_queries[doc_id].append(query)

    # Filter for documents that have actual queries in the qrels
    valid_doc_ids = [doc_id for doc_id, qs in doc_to_queries.items() if len(qs) > 0]

    if len(valid_doc_ids) == 0:
        return {"error": "No valid documents with queries found in qrels."}

    # Sample documents for evaluation
    sampled_doc_ids = random.sample(valid_doc_ids, min(num_samples, len(valid_doc_ids)))

    all_max_rouge1 = []
    all_max_rougeL = []
    all_bertscores_f1 = []
    evaluation_examples = []

    for doc_id in tqdm(sampled_doc_ids, desc="Validating query generation"):
        doc_text = get_doc_text(meta_index, doc_id)
        if not doc_text:
            continue

        real_queries = doc_to_queries[doc_id]

        # Generate synthetic queries for the document
        generated_queries = generate_queries_for_document(
            document=doc_text,
            num_queries=queries_per_doc,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=50,
            temperature=0.5,
        )

        if not generated_queries:
            continue

        # Evaluate each generated query against all real queries for the document
        max_rouge1_for_doc = 0.0
        max_rougeL_for_doc = 0.0
        best_generated_query = ""
        best_bertscore_f1 = 0.0

        # Use the first real query as a single reference for examples and if only one is preferred for BERTScore
        reference_query = real_queries[0]

        for gen_q in generated_queries:
            # ROUGE: Compare generated query against all real queries for this document
            rouge_scores = compute_rouge(gen_q, real_queries, rouge)
            if rouge_scores["rouge1"] > max_rouge1_for_doc:
                max_rouge1_for_doc = rouge_scores["rouge1"]
                max_rougeL_for_doc = rouge_scores["rougeL"]
                best_generated_query = gen_q

        # BERTScore for the best generated query: Compare against all real queries
        if best_generated_query:
            max_bertscore_f1_for_doc = 0.0
            # Compute BERTScore ONCE, batched
            bertscore_result = bertscore.compute(
                predictions=[best_generated_query] * len(real_queries),
                references=real_queries,
                lang="en",
                device=device,
                batch_size=8,
            )

            best_bertscore_f1 = max(bertscore_result["f1"])

        all_max_rouge1.append(max_rouge1_for_doc)
        all_max_rougeL.append(max_rougeL_for_doc)
        all_bertscores_f1.append(best_bertscore_f1)


        evaluation_examples.append({
            "doc_id": doc_id,
            "real_query": reference_query, # Use the single reference for displaying
            "generated_queries": generated_queries,
            "best_generated": best_generated_query,
            "max_rouge1": max_rouge1_for_doc,
            "max_rougeL": max_rougeL_for_doc,
            "bertscore_f1": best_bertscore_f1
        })

    return {
        "avg_max_rouge1": sum(all_max_rouge1) / len(all_max_rouge1) if all_max_rouge1 else 0.0,
        "avg_max_rougeL": sum(all_max_rougeL) / len(all_max_rougeL) if all_max_rougeL else 0.0,
        "avg_bertscore": sum(all_bertscores_f1) / len(all_bertscores_f1) if all_bertscores_f1 else 0.0,
        "examples": evaluation_examples
    }
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
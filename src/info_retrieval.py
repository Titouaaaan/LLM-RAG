# Convert PyTerrier dataframes to ir-measures format
def to_ir_measures_qrels(qrels_df):
    """Convert PyTerrier qrels to ir-measures format."""
    return qrels_df.rename(
        columns={"qid": "query_id", "docno": "doc_id", "label": "relevance"}
    )


def to_ir_measures_run(run_df):
    """Convert PyTerrier run to ir-measures format."""
    return run_df.rename(columns={"qid": "query_id", "docno": "doc_id"})

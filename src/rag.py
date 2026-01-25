from src.data import get_text_from_index
from src.prompting import generate_text
import torch

def simple_rag(
    question: str,
    retriever,
    index_ref,
    model,
    tokenizer,
    device,
    meta_index,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int = 3,
) -> dict:
    """
    Simple RAG pipeline with BM25.

    Args:
        question: User question
        retriever: PyTerrier retriever (BM25)
        index_ref: PyTerrier index reference (for text retrieval)
        model: Generation model
        tokenizer: Model tokenizer
        top_k: Number of documents to retrieve
        meta_index: PyTerrier meta index (for retrieving document text)
    Returns:
        Dict with retrieved docs and generated answer
    """
    # Implement simple RAG with BM25

    results = retriever.search(question)
    top_docs = results.head(top_k)
    doc_ids = top_docs["docno"].tolist()
    doc_texts = get_text_from_index(meta_index, doc_ids)

    context_parts = []
    retrieved_docs_info = []
    for _, row in top_docs.iterrows():
        doc_id = row["docno"]
        doc_text = doc_texts.get(doc_id, "")
        context_parts.append(doc_text[:300]) # Use only first 300 chars for context
        retrieved_docs_info.append({
            "doc_id": doc_id,
            "score": row["score"],
            "text": doc_text
        })

    context = "\n\n".join(context_parts)
    rag_prompt = f"Answer the following question based only on the provided context.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

    generated_answer = generate_text(
        tokenizer=tokenizer,
        model=model,
        device=device,
        prompt=rag_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        num_return_sequences=1,
    )[0]

    return {
        "question": question,
        "retrieved_docs": retrieved_docs_info,
        "answer": generated_answer,
    }

def answer_without_retrieval(question: str, model, tokenizer, device) -> str:
    """Generate an answer without retrieval (direct generation)."""
    messages = [
        {"role": "system", "content": "You are a helpful technical assistant."},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "assistant" in answer.lower():
        answer = answer.split("assistant")[-1].strip()
    return answer
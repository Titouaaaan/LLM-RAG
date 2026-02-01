from src.data import get_text_from_index
from src.prompting import generate_text
from src.splade import SPLADEEncoder
from src.data import get_doc_text, ir_get_doc_text
from typing import List, Tuple, Dict
from tqdm import tqdm
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

class RAGPipeline:
    """
    Complete RAG pipeline: Retrieval + Generation.

    Uses PyTerrier index for document storage (scalable to large collections).
    """

    def __init__(
        self,
        retriever: SPLADEEncoder,
        device,
        generator,
        generator_tokenizer,
        pt_index,
        doc_ids: List[str],
        top_k: int = 3,
    ):
        self.retriever = retriever
        self.generator = generator
        self.gen_tokenizer = generator_tokenizer
        self.pt_index = pt_index
        self.doc_ids = doc_ids
        self.top_k = top_k
        self.device = device
        # Pre-compute document representations
        self._index_documents()

    # MIGHT NEED TO CHANGE THIS TO THE IR VERSIONS OF GET DOC TEXT AND TEXT FROM INDEX
    def _get_doc_text(self, doc_id: str) -> str:
        """Get document text from PyTerrier index."""
        return get_doc_text(self.pt_index, doc_id)

    def _get_texts(self, doc_ids: List[str]) -> Dict[str, str]:
        """Get multiple document texts from PyTerrier index."""
        return get_text_from_index(self.pt_index, doc_ids)

    def _index_documents(self, batch_size: int = 16):
        """Index all documents with SPLADE."""
        print("Indexing documents with SPLADE...")

        # Get texts from PyTerrier index in batches
        all_reps = []

        with torch.no_grad():
            for i in tqdm(range(0, len(self.doc_ids), batch_size)):
                batch_ids = self.doc_ids[i : i + batch_size]
                batch_texts = self._get_texts(batch_ids)
                # Truncate texts and maintain order
                texts = [batch_texts.get(doc_id, "")[:1000] for doc_id in batch_ids]
                reps = self.retriever.encode(texts)
                all_reps.append(reps.cpu())

        self.doc_reps = torch.cat(all_reps, dim=0)
        print(f"Documents indexed: {len(self.doc_ids)}")

    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        """
        Retrieve most relevant documents.

        Args:
            query: User question
            top_k: Number of documents to retrieve

        Returns:
            List of (doc_id, score)
        """
        if top_k is None:
            top_k = self.top_k

        # Encode query
        with torch.no_grad():
            query_rep = self.retriever.encode([query]).cpu()

        # Compute scores
        scores = torch.matmul(query_rep, self.doc_reps.T).squeeze(0)

        # Top-k
        top_indices = torch.topk(scores, k=min(top_k, len(scores))).indices

        results = []
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            score = scores[idx].item()
            results.append((doc_id, score))

        return results

    def generate(
        self,
        query: str,
        context: str,
        max_new_tokens: int = 200,
    ) -> str:
        """Generate a response based on context."""
        system_msg = "You are a helpful assistant that answers technical questions using the provided context."
        user_msg = f"Context:\n{context}\n\nQuestion: {query}"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        prompt = self.gen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.gen_tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.gen_tokenizer.eos_token_id,
            )

        response = self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the response
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()

        return response

    def __call__(self, query: str) -> Dict:
        """
        Complete pipeline: retrieval + generation.

        Args:
            query: User question

        Returns:
            Dict with retrieved_docs and generated_answer
        """
        # Retrieve top-k documents
        retrieved = self.retrieve(query)

        # Fetch the text of the retrieved documents
        doc_texts = self._get_texts([doc_id for doc_id, _ in retrieved])

        # Build a context string for the generator
        context = "\n\n".join([doc_texts.get(doc_id, "") for doc_id, _ in retrieved])

        # Generate the answer using the generator
        answer = self.generate(query, context)

        # Return full info
        return {
            "query": query,
            "retrieved_docs": [{"doc_id": doc_id, "score": score, "text": doc_texts.get(doc_id, "")}
                            for doc_id, score in retrieved],
            "context": context,
            "generated_answer": answer
        }

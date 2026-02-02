import pyterrier as pt
import torch
from src.rag import RAGPipeline
from src.utils import get_best_device, get_model
from src.splade import SPLADEEncoder
from pathlib import Path
from typing import Dict, List
import re
'''
The idea is to have something like this
ENHANCED RAG PIPELINE:
User question
   ↓
LLM rewrites question (CoT-style decomposition)
   ↓
Retriever (SPLADE)
   ↓
Generator produces answer using retrieved context
   ↓
LLM checks answer quality with reasoning
   ↓
(if bad) → generate alternative answer and compare
'''


class ChainOfThoughtRAG:
    """
    Enhanced CoT RAG pipeline with improved consistency and clarity.
    """
    
    def __init__(self, rag_pipeline: RAGPipeline, model, tokenizer, device):
        self.rag = rag_pipeline
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def rewrite_query(self, query: str, num_rewrites: int = 2) -> List[str]:
        """
        query using CoT decomposition to find better query formulations.
        """
        system_msg = """You are an expert at reformulating search queries to improve retrieval.
                        Given a question, generate alternative phrasings that might retrieve more relevant documents.
                        Think step-by-step about:
                        1. What are the key concepts in this question?
                        2. What synonyms or related terms exist?
                        3. How can I rephrase this to be more specific or more general?

                        Provide ONLY the reformulated queries, one per line, without numbering or explanations."""
        
        user_msg = f"Original question: {query}\n\nGenerate {num_rewrites} alternative formulations:"
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        
        rewrites = [q.strip() for q in response.split('\n') if q.strip()]
        return [query] + rewrites[:num_rewrites]
    
    def think_about_retrieval(self, query: str, retrieved_docs: List[Dict]) -> str:
        """
        analyze retrieved documents with chain-of-thought reasoning.
        """
        system_msg = """You are analyzing retrieved documents for a question.
                        Provide chain-of-thought reasoning about:
                        1. How relevant are these documents to the query?
                        2. What information do they contain that's useful?
                        3. Are there any gaps in the retrieved context?
                        4. What aspects of the question might NOT be well covered?

                        Be concise but thorough in your reasoning."""
        
        doc_summary = ""
        for i, doc in enumerate(retrieved_docs, 1):
            doc_summary += f"\nDoc {i} (score: {doc['score']:.2f}):\n{doc['text'][:300]}...\n"
        
        user_msg = f"Query: {query}\n\nRetrieved documents:{doc_summary}\n\nAnalyze this retrieval:"
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        
        return response
    
    def check_answer_quality(self, query: str, answer: str, context: str) -> Dict:
        """
        evaluate answer quality with a score (1-5).
        Returns only the score, not a rewritten version of the answer.
        """
        system_msg = """You are a critical evaluator of AI-generated answers.
                        Analyze the provided answer for:
                        1. Correctness: Does it accurately address the question?
                        2. Completeness: Does it cover all aspects of the question?
                        3. Relevance: Is all the content relevant to the query?
                        4. Clarity: Is it well-structured and understandable?

                        Provide:
                        - A quality score (1-5)
                        - Reasoning about strengths and weaknesses
                        - Specific suggestions for improvement if score < 4
                        
                        IMPORTANT: Do NOT rewrite or paraphrase the answer. Only evaluate it."""
        
        user_msg = f"""Question: {query}
                    Generated Answer: {answer}
                    Supporting Context: {context[:500]}...
                    Evaluate this answer (remember, do NOT rewrite it, only evaluate):"""
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=250,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        
        quality_score = self._extract_quality_score(response)
        
        return {
            "quality_score": quality_score,
            "reasoning": response,
            "needs_improvement": quality_score < 5
        }
    
    def generate_alternative_answer(self, query: str, context: str, 
                                   original_answer: str) -> str:
        """
        generate a different answer for diversity and comparison.
        """
        system_msg = """You are a helpful assistant that answers technical questions using provided context.
                        Generate a DIFFERENT answer than the one shown, perhaps:
                        - Starting with a different aspect
                        - Using different phrasing
                        - Emphasizing different examples
                        - Providing more or less detail in certain areas

                        Focus on correctness and using the provided context."""
        
        user_msg = f"""Context:
                    {context}
                    Question: {query}
                    Previous answer (for reference, don't copy):
                    {original_answer}
                    Now generate a different answer to the same question:"""
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        
        return response
    
    def compare_answers(self, query: str, answer1: str, answer2: str, 
                       quality1: Dict, quality2: Dict) -> Dict:
        """
        compare answers only according to their given score
        """
        system_msg = """You are comparing two answers to the same question.
                        Provide:
                        1. Brief analysis of each answer
                        2. Which one aligns better with the quality metrics
                        3. Explanation of key differences
                        
                        Note: Selection will be based on quality scores (1-5), not your preference.
                        Consider correctness, completeness, clarity, and relevance."""
        
        user_msg = f"""Question: {query}
                    Answer 1 (Quality Score: {quality1['quality_score']}/5):
                    {answer1}
                    Answer 2 (Quality Score: {quality2['quality_score']}/5):
                    {answer2}
                    Compare these answers. Which one has the higher quality score and why might that be?"""
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        
        if quality1["quality_score"] >= quality2["quality_score"]:
            selected_answer = answer1
            selected_quality = quality1["quality_score"]
            selected_idx = 1
        else:
            selected_answer = answer2
            selected_quality = quality2["quality_score"]
            selected_idx = 2

        return {
            "selected_answer": selected_answer,
            "selected_idx": selected_idx,
            "quality_score": selected_quality,
            "comparison_reasoning": response,
            "answer1_quality": quality1["quality_score"],
            "answer2_quality": quality2["quality_score"],
        }
    
    def _extract_quality_score(self, response: str) -> int:
        """
        try and find the score given by the llm and try different patterns
        we do different patterns because sometimes the llm is not very consistent
        was a pain to get it right but gpt helped a lot :)
        """
        
        # Pattern 1: "QUALITY SCORE: X" (as requested in system prompt)
        match = re.search(r'QUALITY\s+SCORE\s*[:=]\s*(\d)', response, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 5:
                return score
        
        # Pattern 2: "Score: X/5" or "Score: X out of 5"
        match = re.search(r'score\s*[:=]\s*(\d)\s*(?:/5|out\s+of\s+5)', response, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 5:
                return score
        
        # Pattern 3: "Quality score: X" (various formats)
        match = re.search(r'quality\s+score\s*[:=]\s*(\d)', response, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 5:
                return score
        
        # Pattern 4: "Rating: X" or "Rate: X"
        match = re.search(r'(?:rating|rate)\s*[:=]\s*(\d)', response, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 5:
                return score
        
        # Pattern 5: Look for any isolated number between 1-5 after "score" or "quality"
        match = re.search(r'(?:quality|score).*?(\d)', response, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 5:
                return score
        
        # Pattern 6: Just look for any number 1-5 on a line with "score"
        for line in response.split('\n'):
            if 'score' in line.lower():
                match = re.search(r'\d', line)
                if match:
                    score = int(match.group())
                    if 1 <= score <= 5:
                        return score
        
        print(f"WARNING: Could not extract quality score from response. Defaulting to 2.")
        print(f"Response preview: {response[:200]}...")
        return 2
    
    def process_query(self, query: str, use_rewrites: bool = True, 
                 use_alternatives: bool = True, verbose: bool = True) -> Dict:
        """
        full pipeline to go from query to answer with cot
        """
        result = {
            "original_query": query,
            "chain_of_thought": {},
            "final_answer": None,
            "final_quality": None,
        }
        
        # step 1: query rewriting
        if use_rewrites:
            if verbose:
                print("\n" + "="*80)
                print("STEP 1: Query Rewriting (CoT Decomposition)")
                print("="*80)
            
            query_variants = self.rewrite_query(query, num_rewrites=2)
            result["chain_of_thought"]["query_variants"] = query_variants
            
            if verbose:
                print(f"Original: {query}")
                for i, variant in enumerate(query_variants[1:], 1):
                    print(f"Rewrite {i}: {variant}")
        else:
            query_variants = [query]
        
        # step 2: retrieve documents for Aall query variants
        if verbose:
            print("\n" + "="*80)
            print("STEP 2: Document Retrieval")
            print("="*80)
        
        all_variant_retrievals = {}
        best_retrieved = None
        best_score_sum = -float('inf')
        
        for variant in query_variants:
            retrieved = self.rag.retrieve(variant, top_k=self.rag.top_k)
            score_sum = sum(score for _, score in retrieved)
            
            all_variant_retrievals[variant] = {
                "retrieved": retrieved,
                "score_sum": score_sum,
            }
            
            if verbose:
                print(f"\nQuery variant: '{variant}'")
                print(f"  Score sum: {score_sum:.2f}")
                for doc_id, score in retrieved:
                    print(f"    - {doc_id}: {score:.2f}")
            
            if score_sum > best_score_sum:
                best_score_sum = score_sum
                best_retrieved = retrieved
        
        # get document texts from best retrieval for context
        doc_texts = self.rag._get_texts([doc_id for doc_id, _ in best_retrieved])
        retrieved_docs = [
            {"doc_id": doc_id, "score": score, "text": doc_texts.get(doc_id, "")}
            for doc_id, score in best_retrieved
        ]
        
        context = "\n\n".join([doc["text"] for doc in retrieved_docs])
        result["retrieved_docs"] = retrieved_docs
        result["context"] = context
        
        # step 3: analyze retrieval with cot (reuse stored retrievals)
        if verbose:
            print("\n" + "="*80)
            print("STEP 3: Chain-of-Thought Analysis of Retrieved Documents")
            print("="*80)
        
        retrieval_analyses = {}
        for variant in query_variants:
            variant_retrieved = all_variant_retrievals[variant]["retrieved"]
            doc_texts_variant = self.rag._get_texts([doc_id for doc_id, _ in variant_retrieved])
            variant_docs = [
                {"doc_id": doc_id, "score": score, "text": doc_texts_variant.get(doc_id, "")}
                for doc_id, score in variant_retrieved
            ]
            
            variant_reasoning = self.think_about_retrieval(variant, variant_docs)
            retrieval_analyses[variant] = variant_reasoning

            if verbose:
                print(f"\nVariant: '{variant}'")
                print(variant_reasoning)

        result["chain_of_thought"]["retrieval_analysis"] = retrieval_analyses
        
        # step 4: generate initial answer
        if verbose:
            print("\n" + "="*80)
            print("STEP 4: Generate Initial Answer")
            print("="*80)
        
        initial_answer = self.rag.generate(query, context)
        result["initial_answer"] = initial_answer
        
        if verbose:
            print(initial_answer)
        
        # step 5: check answer quality
        if verbose:
            print("\n" + "="*80)
            print("STEP 5: Quality Evaluation (with reasoning)")
            print("="*80)
        
        quality_check = self.check_answer_quality(query, initial_answer, context)
        result["chain_of_thought"]["initial_quality"] = quality_check
        
        if verbose:
            print(f"Quality Score: {quality_check['quality_score']}/5")
            print(f"Needs Improvement: {quality_check['needs_improvement']}")
            print(f"\nReasoning:\n{quality_check['reasoning']}")
        
        # step 6-8: generate alternatives, evaluate, and compare if needed
        if use_alternatives and quality_check["needs_improvement"]:
            if verbose:
                print("\n" + "="*80)
                print("STEP 6: Generating Alternative Answer")
                print("="*80)
            
            alternative_answer = self.generate_alternative_answer(
                query, context, initial_answer
            )
            
            if verbose:
                print(alternative_answer)
            
            # step 7: check alternative quality
            if verbose:
                print("\n" + "="*80)
                print("STEP 7: Evaluating Alternative Answer")
                print("="*80)
            
            alt_quality = self.check_answer_quality(query, alternative_answer, context)
            result["chain_of_thought"]["alternative_quality"] = alt_quality
            
            if verbose:
                print(f"Quality Score: {alt_quality['quality_score']}/5")
                print(f"\nReasoning:\n{alt_quality['reasoning']}")
            
            # step 8: compare and select best (based on quality scores)
            if verbose:
                print("\n" + "="*80)
                print("STEP 8: Comparing Answers and Selecting Best")
                print("="*80)
            
            comparison = self.compare_answers(
                query, initial_answer, alternative_answer, 
                quality_check, alt_quality
            )
            result["chain_of_thought"]["comparison"] = comparison
            result["final_answer"] = comparison["selected_answer"]
            result["final_quality"] = comparison["quality_score"]
            
            if verbose:
                print(f"\nComparison Analysis:")
                print(comparison["comparison_reasoning"])
                print(f"\n--- Selection Based on Quality Scores ---")
                print(f"Answer 1 Quality: {comparison['answer1_quality']}/5")
                print(f"Answer 2 Quality: {comparison['answer2_quality']}/5")
                print(f"Selected: Answer {comparison['selected_idx']} (Score: {comparison['quality_score']}/5)")
        else:
            result["final_answer"] = initial_answer
            result["final_quality"] = quality_check["quality_score"]
        
        return result
    
    def interactive_loop(self):
        print("\n" + "="*80)
        print("ENHANCED RAG PIPELINE - Interactive Mode")
        print("="*80)
        print("Type 'quit' to exit, 'help' for options\n")
        
        verbose = True
        use_rewrites = True
        use_alternatives = True
        
        while True:
            try:
                user_input = input("\n>>> Your question: ").strip()
                
                if user_input.lower() == 'quit':
                    print("Exiting...")
                    break
                elif user_input.lower() == 'help':
                    print("""Commands:
                            quit           - Exit the program
                            help           - Show this help message
                            else ask a quesiton and get an answer!""")
                    continue
                elif not user_input:
                    continue
                
                result = self.process_query(
                    user_input,
                    use_rewrites=use_rewrites,
                    use_alternatives=use_alternatives,
                    verbose=verbose
                )
                
                print("\n" + "="*80)
                print("FINAL ANSWER:")
                print("="*80)
                print(result["final_answer"])
                print(f"\nQuality Score: {result['final_quality']}/5")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Exiting...")
                break
            except Exception as e:
                print(f"Error processing query: {e}")
                import traceback
                traceback.print_exc()
                continue


if __name__ == "__main__":
    # Load dataset and initialize models
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
    query_ids = set(queries_df["qid"].tolist())
    qrels_df = qrels_df[qrels_df["qid"].isin(query_ids)].copy()

    relevant_doc_ids = set(qrels_df["docno"].tolist())
    doc_id_list = list(relevant_doc_ids)

    splade_encoder = SPLADEEncoder("naver/splade-cocondenser-ensembledistil")
    splade_encoder = splade_encoder.to(device)

    rag = RAGPipeline(
        retriever=splade_encoder,
        device=device,
        generator=model,
        generator_tokenizer=tokenizer,
        pt_index=meta_index,
        doc_ids=doc_id_list,
        top_k=3,
    )

    cot_rag = ChainOfThoughtRAG(rag, model, tokenizer, device)

    # run a forced exmaple here, will probably remove this later
    # but useful when running code just to test things
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Query with Full Chain-of-Thought")
    print("="*80)
    
    test_query = "what is an SSD?"
    result = cot_rag.process_query(
        test_query,
        use_rewrites=True,
        use_alternatives=True,
        verbose=True
    )
    
    print("\n" + "="*80)
    print("FINAL RESULT SUMMARY")
    print("="*80)
    print(f"Original Question: {result['original_query']}")
    print(f"\nFinal Answer:\n{result['final_answer']}")
    print(f"\nQuality Score: {result['final_quality']}/5")
    
    # the main interactive loop for the user
    print("\n\n" + "="*80)
    print("Starting Interactive Loop")
    print("="*80)
    
    cot_rag.interactive_loop()
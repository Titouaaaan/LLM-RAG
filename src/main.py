import pyterrier as pt
import torch
from src.rag import RAGPipeline
from src.utils import get_best_device, get_model
from src.splade import SPLADEEncoder
from pathlib import Path
from typing import Dict, List

'''
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
    much better cot rag
    rewrote many functions because the ones from the practical are way too simple
    and dont really fit with more complex reasoning/questions
    """
    
    def __init__(self, rag_pipeline: RAGPipeline, model, tokenizer, device):
        self.rag = rag_pipeline # this is the og rag class we made in tests
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def rewrite_query(self, query: str, num_rewrites: int = 2) -> List[str]:
        """
        rewrite the query using cot decomposition.
        so we rewrite the questions into possibly better formulations 
        in order to retrieve docs that may be relevant (even tho the original query may not be great)
        """
        # this prompt seems to work well because it forces the model to find adjacent terms
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
        
        # we keep hyper params similar to practicals
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # extract the response part
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        
        # parse rewrites from response
        rewrites = [q.strip() for q in response.split('\n') if q.strip()]
        
        # return og query and the new ones
        return [query] + rewrites[:num_rewrites]
    
    def think_about_retrieval(self, query: str, retrieved_docs: List[Dict]) -> str:
        """
        cot reasoning about the retrieved documents
        time for the llm to actually analyse what SPLADE returned
        """
        system_msg = """You are analyzing retrieved documents for a question.
                        Provide chain-of-thought reasoning about:
                        1. How relevant are these documents to the query?
                        2. What information do they contain that's useful?
                        3. Are there any gaps in the retrieved context?
                        4. What aspects of the question might NOT be well covered?

                        Be concise but thorough in your reasoning."""
        
        # summary of retrieved docs
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
        verify the quality of the generated answer with reasoning by giving a score
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
                        - Specific suggestions for improvement if score < 4"""
        # we still add a limiter to the context here to avoid using too many tokens
        user_msg = f"""Question: {query}

                        Generated Answer: {answer}

                        Supporting Context: {context[:500]}...

                        Evaluate this answer:"""
        
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
        
        # try and fetch the score
        # normally the model should return it correctly after doing many tests
        quality_score = None
        for line in response.split('\n'):
            if 'score' in line.lower() and any(c.isdigit() for c in line):
                # Try to find a number 1-5
                for word in line.split():
                    if word.isdigit() and 1 <= int(word) <= 5:
                        quality_score = int(word)
                        break
            if quality_score:
                break
        
        if quality_score is None:
            quality_score = 2  # give it a sub neutral score if theres an issue (if no score assume not great just in case)
        
        return {
            "quality_score": quality_score,
            "reasoning": response,
            "needs_improvement": quality_score < 4
        }
    
    def generate_alternative_answer(self, query: str, context: str, 
                                   original_answer: str) -> str:
        """
        generate a new answer that is different from the original one to have more diversity
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
                temperature=0.9,  # tweaked the temp to see if it helps diversity (but honestly 0.7 or 0.9 actually doesnt matter i think)
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
        Compare answers in order to select the best one with reasoning
        """
        system_msg = """You are comparing two answers to the same question.
                        Provide:
                        1. Brief analysis of each answer
                        2. Which is better and why
                        3. A combined/improved answer if possible

                        Consider correctness, completeness, clarity, and relevance."""
        
        user_msg = f"""Question: {query}

                    Answer 1:
                    {answer1}

                    Answer 2:
                    {answer2}

                    Compare these answers and recommend the best one:"""
        
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
        
        # Select answer based on quality scores
        selected_answer = answer1 if quality1["quality_score"] >= quality2["quality_score"] else answer2
        selected_quality = quality1 if quality1["quality_score"] >= quality2["quality_score"] else quality2
        
        return {
            "selected_answer": selected_answer,
            "quality_score": selected_quality["quality_score"],
            "comparison_reasoning": response,
            "answer1_quality": quality1["quality_score"],
            "answer2_quality": quality2["quality_score"],
        }
    
    def process_query(self, query: str, use_rewrites: bool = True, 
                     use_alternatives: bool = True, verbose: bool = True) -> Dict:
        """
        full pipeline to process a query into cot reasoning
        """
        result = {
            "original_query": query,
            "chain_of_thought": {},
            "final_answer": None,
            "final_quality": None,
        }
        
        # Step 1: Query rewriting
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
        
        # Step 2: Retrieve with best query variant
        if verbose:
            print("\n" + "="*80)
            print("STEP 2: Document Retrieval")
            print("="*80)
        
        best_retrieved = None
        best_score_sum = -float('inf')
        
        for variant in query_variants:
            retrieved = self.rag.retrieve(variant, top_k=self.rag.top_k)
            score_sum = sum(score for _, score in retrieved)
            
            if verbose:
                print(f"\nQuery variant: '{variant}'")
                print(f"  Score sum: {score_sum:.2f}")
                for doc_id, score in retrieved:
                    print(f"    - {doc_id}: {score:.2f}")
            
            if score_sum > best_score_sum:
                best_score_sum = score_sum
                best_retrieved = retrieved
        
        # get document texts
        doc_texts = self.rag._get_texts([doc_id for doc_id, _ in best_retrieved])
        retrieved_docs = [
            {"doc_id": doc_id, "score": score, "text": doc_texts.get(doc_id, "")}
            for doc_id, score in best_retrieved
        ]
        
        context = "\n\n".join([doc["text"] for doc in retrieved_docs])
        result["retrieved_docs"] = retrieved_docs
        result["context"] = context
        
        # Step 3: Analyze retrieval with CoT
        if verbose:
            print("\n" + "="*80)
            print("STEP 3: Chain-of-Thought Analysis of Retrieved Documents")
            print("="*80)
        
        retrieval_reasoning = self.think_about_retrieval(query, retrieved_docs)
        result["chain_of_thought"]["retrieval_analysis"] = retrieval_reasoning
        
        if verbose:
            print(retrieval_reasoning)
        
        # Step 4: Generate initial answer
        if verbose:
            print("\n" + "="*80)
            print("STEP 4: Generate Initial Answer")
            print("="*80)
        
        initial_answer = self.rag.generate(query, context)
        result["initial_answer"] = initial_answer
        
        if verbose:
            print(initial_answer)
        
        # Step 5: Check answer quality
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
        
        # Step 6: Generate alternatives if needed
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
            
            # Step 7: Check alternative quality
            if verbose:
                print("\n" + "="*80)
                print("STEP 7: Evaluating Alternative Answer")
                print("="*80)
            
            alt_quality = self.check_answer_quality(query, alternative_answer, context)
            result["chain_of_thought"]["alternative_quality"] = alt_quality
            
            if verbose:
                print(f"Quality Score: {alt_quality['quality_score']}/5")
                print(f"\nReasoning:\n{alt_quality['reasoning']}")
            
            # Step 8: Compare and select best
            if verbose:
                print("\n" + "="*80)
                print("STEP 8: Comparing Answers")
                print("="*80)
            
            comparison = self.compare_answers(
                query, initial_answer, alternative_answer, 
                quality_check, alt_quality
            )
            result["chain_of_thought"]["comparison"] = comparison
            result["final_answer"] = comparison["selected_answer"]
            result["final_quality"] = comparison["quality_score"]
            
            if verbose:
                print(comparison["comparison_reasoning"])
        else:
            result["final_answer"] = initial_answer
            result["final_quality"] = quality_check["quality_score"]
        
        return result
    
    def interactive_loop(self):
        """
        this time for the user to actually talk to the llm rag pipeline
        """
        print("\n" + "="*80)
        print("ENHANCED RAG PIPELINE - Interactive Mode")
        print("="*80)
        print("Type 'quit' to exit, 'help' for options\n")
        
        verbose = True
        use_rewrites = True
        use_alternatives = True
        # wanted to add some options to use rewrites or alternatives but ig not useful at the moment ...
        # so everything set to true atm
        while True:
            try:
                user_input = input("\n>>> Your question: ").strip()
                
                if user_input.lower() == 'quit':
                    print("Exiting...")
                    break
                elif user_input.lower() == 'help':
                    print("""
                        Commands:
                        quit           - Exit the program
                        Or just type a question to get an answer!
                                            """)
                    continue
                elif not user_input:
                    continue
                
                # send query into pipeline
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
                continue

if __name__ == "__main__":
    # index and splade models are already available normally 
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

    # Get the set of relevant document IDs
    relevant_doc_ids = set(qrels_df["docno"].tolist())
    doc_id_list = list(relevant_doc_ids)

    # SPLADE encoder
    splade_encoder = SPLADEEncoder("naver/splade-cocondenser-ensembledistil")
    splade_encoder = splade_encoder.to(device)

    # Initialize RAG pipeline
    rag = RAGPipeline(
        retriever=splade_encoder,
        device=device,
        generator=model,
        generator_tokenizer=tokenizer,
        pt_index=meta_index,
        doc_ids=doc_id_list,
        top_k=3,
    )

    # Initialize enhanced CoT pipeline
    cot_rag = ChainOfThoughtRAG(rag, model, tokenizer, device)

    # Example 1: Single query with full chain-of-thought
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Query with Full Chain-of-Thought")
    print("="*80)
    
    test_query = "what is ram in computers?"
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
    
    # Example 2: Interactive mode
    print("\n\n" + "="*80)
    print("EXAMPLE 2: Starting Interactive Loop")
    print("="*80)
    
    cot_rag.interactive_loop()
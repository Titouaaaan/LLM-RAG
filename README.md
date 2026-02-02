# Enhanced RAG Pipeline with SPLADE Retrieval and Chain-of-Thought Reasoning

## Overview

This project implements an enhanced **Retrieval-Augmented Generation (RAG) pipeline** for answering technical questions. The main goals were to investigate:

1. **Using SPLADE for document retrieval**  
2. **Use an LLM to rewrite user queries before retrieval** (CoT-style decomposition)
3. **Applying Chain-of-Thought (CoT) reasoning and validation** on answer generation  

In addition to SPLADE, the project also supports traditional **BM25 retrieval**. We included a script to train SPLADE models — it works, but improvements over BM25 are limited on the current dataset. We believe performance would improve with larger and more diverse data, as the current model may overfit to what it has seen.

The project builds on many concepts from practicals, including:

- Document indexing and retrieval  
- Data handling and preprocessing  
- Prompting strategies for LLMs  
- Chain-of-Thought reasoning for both question decomposition and answer validation  

---

## Project Structure

- **`src/`**: Core classes and functions

- **`tests/`**: Tests for the various components  

But most importantly:

- **`main.py`**: Demonstrates the full pipeline with CoT reasoning and SPLADE retrieval. This is the main entry point for experiments.

---

## Features

- **Document Retrieval**
  - BM25 baseline  
  - SPLADE  
  - SPLADE training script included  

- **Chain-of-Thought Reasoning**
  - Query rewriting / decomposition  
  - Analysis of retrieved documents with reasoning  
  - Answer generation with self-consistency  
  - Answer quality evaluation and optional alternative answer generation  

- **RAG Pipeline**
  - Modular integration of retrieval and LLM generation  
  - Provides detailed reasoning and vote counts for self-consistency  

---

## Data and Index

- **Dataset**: IRDS Lotte Technology development set  
- **Document Index**: Precomputed index for retrieval experiments  

**Download the index** (if not already cached):

[Download index_lotte](https://master-mind.isir.upmc.fr/llm/data/index_lotte.tar.gz)  

and put it in the folder `./outputs/practical-04/index_lotte`.

> ⚠️ Note: This path will most likely be changed bc it uses the old path from the practicals

## Usage

### Requirements and venv

Make sure you have an active virtual environment with the proper dependencies installed (everything is in the `requirements.txt` file).

### Run Single Query with Full CoT

```bash
python -m src.main
```

This will:

- Rewrite the query using CoT decomposition,

- Retrieve relevant documents with SPLADE,

- Generate an answer using retrieved context,

- Evaluate answer quality and optionally generate an alternative answer,

- Compare answers and output the final selection with reasoning.

At first an exmaple runs just to see how it works, it will probably be removed because it was mostly for debugging.
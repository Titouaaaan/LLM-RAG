from typing import List, Tuple
from collections import Counter
from src.prompting import generate_text, build_prompt

def build_cot_prompt(
        examples: List[Tuple[str, str, str]],
        query: str,
        task_description: str = "Solve the following problem step by step.",
    ) -> str:
    """
    Build a few-shot CoT prompt with reasoning examples.
    """
    prompt_parts = [task_description, ""]

    for i, (question, reasoning, answer) in enumerate(examples, 1):
        prompt_parts.append(f"Example {i}:")
        prompt_parts.append(f"Question: {question}")
        prompt_parts.append(f"Reasoning: {reasoning}")
        prompt_parts.append(f"Answer: {answer}")
        prompt_parts.append("")

    # Final query (no answer)
    prompt_parts.append("Now solve this problem:")
    prompt_parts.append(f"Question: {query}")
    prompt_parts.append("Reasoning:")

    return "\n".join(prompt_parts)

def extract_answer_from_cot(response: str) -> str:
    """
    Extract the final numerical answer from a CoT response.

    Args:
        response: The model's response with reasoning

    Returns:
        The extracted answer (number as string)
    """
    import re

    # Try to find "Answer: X" pattern
    answer_match = re.search(r"Answer:\s*(\d+)", response, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1)

    # Try to find "= X" at the end of reasoning
    equals_match = re.search(r"=\s*(\d+)", response)
    if equals_match:
        return equals_match.group(1)

    # Fall back to last number in response
    numbers = re.findall(r"\d+", response)
    if numbers:
        return numbers[-1]

    return ""

def self_consistency_classify(
        prompt: str,
        tokenizer,
        model,
        device,
        num_samples: int = 5,
        temperature: float = 0.7,
        valid_labels: List[str] | None = None,
    ) -> Tuple[str, dict]:
    """
    Classification with self-consistency.

    Args:
        prompt: The classification prompt (already chat-formatted)
        tokenizer: HF tokenizer
        model: HF model
        device: torch device
        num_samples: Number of samples to generate
        temperature: Temperature for sampling
        valid_labels: Optional list of valid labels

    Returns:
        (majority_label, counts)
    """

    responses = generate_text(
        tokenizer=tokenizer,
        model=model,
        device=device,
        prompt=prompt,
        num_return_sequences=num_samples,
        temperature=temperature,
        do_sample=True,
        max_new_tokens=10,
    )

    labels: List[str] = []

    for resp in responses:
        resp_clean = resp.strip().lower()

        # Try to match against valid labels
        if valid_labels is not None:
            for v in valid_labels:
                if v in resp_clean:
                    labels.append(v)
                    break
        else:
            # Fallback: take first token
            labels.append(resp_clean.split()[0])

    counts = dict(Counter(labels))

    if not counts:
        return "", {}

    majority_label = max(counts, key=counts.get)
    return majority_label, counts

def self_consistency_cot(
        question: str,
        cot_examples: List[Tuple[str, str, str]],
        tokenizer,
        model,
        device,
        num_samples: int = 10,
        temperature: float = 0.7,
    ) -> Tuple[str, dict]:
    """
    Self-consistency with Chain-of-Thought for reasoning tasks.

    Args:
        question: The question to solve
        cot_examples: CoT examples (question, reasoning, answer)
        tokenizer: HF tokenizer
        model: HF model
        device: torch device
        num_samples: Number of reasoning paths to generate
        temperature: Temperature for sampling

    Returns:
        (majority_answer, vote_counts)
    """

    # Build CoT prompt (text-level)
    cot_text = build_cot_prompt(cot_examples, question)

    # Convert to chat format
    chat_prompt = build_prompt(
        tokenizer=tokenizer,
        user=cot_text,
        system="Solve the problem step by step and give the final answer.",
    )

    # Generate multiple reasoning paths
    responses = generate_text(
        tokenizer=tokenizer,
        model=model,
        device=device,
        prompt=chat_prompt,
        num_return_sequences=num_samples,
        temperature=temperature,
        do_sample=True,
        max_new_tokens=200,
    )

    # Extract final answers
    answers = []
    for r in responses:
        ans = extract_answer_from_cot(r)
        if ans:
            answers.append(ans)

    vote_counts = dict(Counter(answers))

    if not vote_counts:
        return "", {}

    majority_answer = max(vote_counts, key=vote_counts.get)
    return majority_answer, vote_counts

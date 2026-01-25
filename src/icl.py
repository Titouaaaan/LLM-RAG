from typing import List, Tuple

def build_few_shot_prompt(
    examples: List[Tuple[str, str]],
    query: str,
    task_description: str = "Classify the following text.",
    input_label: str = "Text",
    output_label: str = "Classification",
) -> str:
    """
    Build a few-shot prompt from examples.

    Args:
        examples: List of tuples (input, output) for examples
        query: The text to classify
        task_description: Description of the task
        input_label: Label for input (e.g., "Text", "Question")
        output_label: Label for output (e.g., "Classification", "Answer")

    Returns:
        Formatted prompt for the model
    """
    prompt_parts = [task_description, ""]

    for i, (inp, out) in enumerate(examples, 1):
        prompt_parts.append(f"Example {i}:")
        prompt_parts.append(f'{input_label}: "{inp}"')
        prompt_parts.append(f"{output_label}: {out}")
        prompt_parts.append("")

    prompt_parts.append("Now:")
    prompt_parts.append(f'{input_label}: "{query}"')
    prompt_parts.append(f"{output_label}:")

    return "\n".join(prompt_parts)
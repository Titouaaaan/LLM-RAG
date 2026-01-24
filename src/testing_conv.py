from utils import get_best_device, get_model
from prompting import build_prompt, generate_text
import torch
from transformers import AutoTokenizer


def test_chat_template(model_name, device):
    """Test the chat template application of the tokenizer."""
    print(f"Testing chat template for model: {model_name}")

    device = get_best_device()
    #model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    model, tokenizer = get_model(model_name, device)

    # The chat template is a Jinja2 template stored in the tokenizer
    print(f"Chat template for {model_name}:")
    print("-" * 60)
    print(tokenizer.chat_template)

    # Let's see how a simple conversation gets formatted
    simple_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]

    # Without add_generation_prompt
    formatted = tokenizer.apply_chat_template(simple_messages, tokenize=False)
    print("Formatted (without generation prompt):")
    print(repr(formatted))
    print()

    # With add_generation_prompt - adds the assistant's turn start
    # indicator to signal where generation should begin, important for inference
    formatted_gen = tokenizer.apply_chat_template(
        simple_messages, tokenize=False, add_generation_prompt=True
    )
    print("Formatted (with generation prompt):")
    print(repr(formatted_gen))

    # Multi-turn conversation example
    multi_turn = [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "What is 5 + 3?"},
        {"role": "assistant", "content": "5 + 3 = 8"},
        {"role": "user", "content": "And what is 8 × 2?"},
    ]

    formatted_multi = tokenizer.apply_chat_template(
        multi_turn, tokenize=False, add_generation_prompt=True
    )
    print("Multi-turn conversation:")
    print(formatted_multi)

# --- Thinking models
def test_thinking_mode(model_name, device):
    """Test the thinking mode of a model that supports it."""
    # Load a Qwen3 tokenizer to explore thinking mode
    # (Qwen3 is a "thinking" model with enable_thinking support)
    thinking_model_name = model_name #"Qwen/Qwen3-0.6B"
    thinking_tokenizer = AutoTokenizer.from_pretrained(thinking_model_name)

    print(f"Loaded tokenizer for: {thinking_model_name}")
    # Compare prompts with and without thinking enabled
    test_messages = [
        {"role": "user", "content": "What is the capital of France?"},
    ]

    # Without thinking (standard mode)
    prompt_no_think = thinking_tokenizer.apply_chat_template(
        test_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    print("=== Without thinking (enable_thinking=False) ===")
    print(prompt_no_think)

test_chat_template("HuggingFaceTB/SmolLM2-1.7B-Instruct", get_best_device())
test_thinking_mode("Qwen/Qwen3-0.6B", get_best_device())
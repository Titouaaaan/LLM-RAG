from src.utils import get_model, get_best_device
from src.prompting import build_prompt, generate_text
from src.icl import build_few_shot_prompt
from src.chain_of_thought import self_consistency_classify, extract_answer_from_cot, build_cot_prompt, self_consistency_cot

device = get_best_device()
model, tokenizer = get_model(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    device
)

# Example tech support questions (simplified)
# For the test we will use the first two
# Since it should originally get the first one wrong but correct itself with few-shot
test_questions = [
    "How do I fix the blue screen error on Windows?",
    "Python is such a terrible language, nothing works!",
    "Can someone explain how to set up a VPN?",
    "This software is garbage, crashes every 5 minutes.",
]

# Expected labels: question, complaint, question, complaint
expected_labels = ["question", "complaint", "question", "complaint"]

zero_shot_prompt = """Classify the following text as "question" or "complaint".
Reply with a single word only.

Text: {text}
Classification:"""

print("\n---- Without few-shot prompting ----\n")
for text, expected in zip(test_questions[:2], expected_labels[:2]):
    user_text = zero_shot_prompt.format(text=text)

    prompt = build_prompt(
        tokenizer=tokenizer,
        user=user_text,
        system="You are a text classification assistant. Reply with a single word."
    )

    response = generate_text(
        tokenizer=tokenizer,
        model=model,
        device=device,
        prompt=prompt,
        max_new_tokens=10,
        temperature=0.1,
        do_sample=False,
    )[0]

    print(f"Text: {text[:60]}...")
    print(f"Predicted: {response.strip()} | Expected: {expected}")
    print()

print("\n---- Now testing with few-shot prompting ----\n")

# few shot examples to add in the prompt
few_shot_examples = """Example 1:
Text: "Where can I find the settings menu in this app?"
Classification: question

Example 2:
Text: "This update broke everything and I lost all my data!"
Classification: complaint
"""

few_shot_prompt = (
    few_shot_examples
    + """
Now, classify this text:
Text: "{text}"
Classification:"""
)

# now test on the first two again with the new few-shot prompt
for text, expected in zip(test_questions[:2], expected_labels[:2]):
    user_text = few_shot_prompt.format(text=text)

    prompt = build_prompt(
        tokenizer=tokenizer,
        user=user_text,
        system="You are a text classification assistant. Reply with a single word."
    )

    response = generate_text(
        tokenizer=tokenizer,
        model=model,
        device=device,
        prompt=prompt,
        max_new_tokens=10,
        temperature=0.1,
        do_sample=False,
    )[0]

    print(f"Text: {text[:60]}...")
    print(f"Predicted: {response.strip()} | Expected: {expected}")
    print()

examples = [
    ("How do I update my graphics drivers?", "question"),
    ("The app is so slow it's unusable.", "complaint"),
    ("What's the difference between RAM and ROM?", "question"),
]

test_query = "This product is a complete waste of money."

icl_text = build_few_shot_prompt(
    examples,
    test_query,
    task_description="Classify the text as 'question' or 'complaint'.",
)

prompt = build_prompt(
    tokenizer=tokenizer,
    user=icl_text,
    system="You are a text classification assistant. Reply with a single word."
)

response = generate_text(
    tokenizer=tokenizer,
    model=model,
    device=device,
    prompt=prompt,
    max_new_tokens=10,
    temperature=0.1,
    do_sample=False,
)[0]

print("Generated prompt:")
print(icl_text)
print("\nModel output:")
print(response)

print("\n---- Now lets test with chain-of-thought prompting ----\n")

# A task that benefits from reasoning
reasoning_questions = [
    # Query 1
    "A store has 45 apples. "
    "They sell 12 in the morning and receive 20 more. "
    "How many apples do they have?",
    # Query 2
    "If a train travels at 60 km/h for 2.5 hours, how far does it travel?",
    # Query 3
    "Marie has 3 times as many books as Paul. "
    "Paul has 8 books. How many books does Marie have?",
]

expected_answers = ["53", "150", "24"]

# Direct prompting (no CoT)
print("\n---- Direct prompting (no CoT) ----\n")
direct_prompt = """Answer the following question with just the number.

Question: {question}
Answer:"""

for question, expected in zip(reasoning_questions[:2], expected_answers[:2]):
    chat_prompt = build_prompt(
        tokenizer=tokenizer,
        user=direct_prompt.format(question=question),
    )
    response = generate_text(
        tokenizer=tokenizer,
        model=model,
        device=device,
        prompt=chat_prompt,
        max_new_tokens=20,
        temperature=0.1,
        do_sample=False,
    )[0]

    print("=" * 80)
    print(f"Q: {question}")
    print(f"A: {response.strip()} (expected: {expected})")
    print()

# Zero-shot CoT
print("\n---- Zero-shot CoT prompting ----\n")
cot_prompt = """Answer the following question. Let's think step by step.

Question: {question}

Step-by-step reasoning:"""

for question, expected in zip(reasoning_questions[:2], expected_answers[:2]):
    chat_prompt = build_prompt(
        tokenizer=tokenizer,
        user=cot_prompt.format(question=question),
    )
    response = generate_text(
        tokenizer=tokenizer,
        model=model,
        device=device,
        prompt=chat_prompt,
        max_new_tokens=150,
        temperature=0.1,
        do_sample=False,
    )[0]

    print("=" * 80)
    print(f"Q: {question}")
    print(f"Reasoning: {response.strip()}")
    print(f"(Expected answer: {expected})")
    print()

# Few-shot CoT examples with reasoning
print("\n---- Few-shot CoT prompting ----\n")
cot_examples = [
    (
        # Query
        "A baker makes 24 cookies. He gives away 8 and bakes 15 more. "
        "How many cookies does he have?",
        # Think
        "Start with 24 cookies. Give away 8: 24 - 8 = 16. Bake 15 more: 16 + 15 = 31.",
        # Answer
        "31",
    ),
    (
        "A car travels at 80 km/h for 3 hours. How far does it go?",
        "Distance = speed × time. Distance = 80 × 3 = 240 km.",
        "240",
    ),
]

for question, expected in zip(reasoning_questions, expected_answers):
    cot_text = build_cot_prompt(cot_examples, question)

    chat_prompt = build_prompt(
        tokenizer=tokenizer,
        user=cot_text,
    )

    response = generate_text(
        tokenizer=tokenizer,
        model=model,
        device=device,
        prompt=chat_prompt,
        max_new_tokens=100,
        temperature=0.1,
        do_sample=False,
    )[0]

    print("=" * 80)
    print(f"Q: {question}")
    print(f"Response: {response.strip()}")
    print(f"(Expected: {expected})")
    print()

# Test answer extraction
test_responses = [
    "Start with 45. Subtract 12: 45 - 12 = 33. Add 20: 33 + 20 = 53. Answer: 53",
    "Speed is 60 km/h, time is 2.5 hours. Distance = 60 × 2.5 = 150 km.",
    "Paul has 8 books. Marie has 3 times more: 3 × 8 = 24 books.",
]

print("Testing answer extraction:")
for resp in test_responses:
    answer = extract_answer_from_cot(resp)
    print(f"  Response: {resp[:50]}...")
    print(f"  Extracted: {answer}")
    print()

print("\n---- Now testing self-consistency classification ----\n")
for text, expected in zip(test_questions, expected_labels):
    chat_prompt = build_prompt(
        tokenizer=tokenizer,
        user=few_shot_prompt.format(text=text),
    )

    label, counts = self_consistency_classify(
        chat_prompt,
        tokenizer=tokenizer,
        model=model,
        device=device,
        num_samples=5,
        valid_labels=["question", "complaint"],
    )

    print(f"Text: {text[:50]}...")
    print(f"Votes: {counts}")
    print(f"Result: {label} | Expected: {expected}")
    print()

print('\n---- Comparing single-shot and self-consistency ----\n')

# Create a test dataset
test_data = [
    ("How do I install Python on Mac?", "question"),
    ("This keyboard is absolutely terrible.", "complaint"),
    ("What programming language should I learn first?", "question"),
    ("The customer service is useless.", "complaint"),
    ("Can you explain how databases work?", "question"),
    ("I wasted hours on this broken software.", "complaint"),
]
# Configuration
num_consistency_samples = 5

# Compare the two approaches
single_correct = 0
consistency_correct = 0
for text, expected in test_data:
    single_prompt = build_prompt(
        tokenizer=tokenizer,
        user=few_shot_prompt.format(text=text),
        system=(
            "You are a text classification assistant.\n"
            "Reply with exactly ONE word.\n"
            "Valid labels: question, complaint."
        ),
    )

    single_response = generate_text(
        tokenizer=tokenizer,
        model=model,
        device=device,
        prompt=single_prompt,
        max_new_tokens=10,
        temperature=0.1,
        do_sample=False,
    )[0].strip().lower()

    if expected in single_response:
        single_correct += 1

    label, counts = self_consistency_classify(
        prompt=single_prompt,
        tokenizer=tokenizer,
        model=model,
        device=device,
        num_samples=num_consistency_samples,
        temperature=0.7,
        valid_labels=["question", "complaint"],
    )

    if label == expected:
        consistency_correct += 1

    print("-" * 80)
    print(f"Text: {text}")
    print(f"Expected: {expected}")
    print(f"Single-shot: {single_response}")
    print(f"Self-consistency votes: {counts}")
    print(f"Self-consistency result: {label}")

print("\n==== Summary ====")
print(f"Single-shot accuracy: {single_correct}/{len(test_data)}")
print(f"Self-consistency accuracy: {consistency_correct}/{len(test_data)}")

print("\n---- Now testing self-consistency with chain-of-thought ----\n")

# Test CoT + Self-Consistency

for question, expected in zip(reasoning_questions, expected_answers):
    answer, votes = self_consistency_cot(
        question=question,
        cot_examples=cot_examples,
        tokenizer=tokenizer,
        model=model,
        device=device,
        num_samples=num_consistency_samples,
    )

    print("=" * 80)
    print(f"Q: {question}")
    print(f"Votes: {votes}")
    print(f"Answer: {answer} (expected: {expected})")

    correct = "✓" if answer == expected else "✗"
    print(f"Result: {correct}")
    print()
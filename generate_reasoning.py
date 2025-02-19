import json
import signal
import sys

import ollama

def signal_handler(signum, frame):
    print("\nSaving progress and exiting...")
    sys.exit(0)

def load_examples(file_path: str) -> list:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_examples(examples: list, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=4)

def generate_batch(batch_size: int = 20) -> list:
    prompt = f"""You are in strict JSON mode. Output an array of reasoning problems. Each problem must follow this format:
[
    {{
        "text": "Your reasoning problem here",
        "labels": "reasoning_llm"
    }}
]
Generate {batch_size} new, unique, and complex problems now (start with [ and only output JSON array):"""

    response = ollama.generate(
        model="huihui_ai/qwen2.5-abliterate:14b",
        prompt=prompt,
        options={
            "temperature": 1.3,  # Slightly higher for more variety
            "top_p": 0.95,
            "num_ctx": 2048,    # Increased context for larger batches
            "stop": ["]"]
        }
    )
    
    text = response.response.strip()
    if not text.startswith("["): text = text[text.find("["):]
    if not text.endswith("]"): text += "]"
    
    try:
        batch = json.loads(text)
        return [ex for ex in batch if 'text' in ex and 'labels' in ex 
                and ex['labels'] == "reasoning_llm"]
    except:
        return []

def main(batch_size: int = 20):
    signal.signal(signal.SIGINT, signal_handler)
    examples = load_examples('reasoning_examples.txt')
    seen = {ex['text'] for ex in examples}
    total_generated = len(examples)
    
    print(f"Starting with {total_generated} examples")
    print(f"Generating in batches of {batch_size}")
    print("Press Ctrl+C to save and exit")
    
    try:
        while True:
            new_batch = generate_batch(batch_size)
            unique_new = [ex for ex in new_batch if ex['text'] not in seen]
            
            if unique_new:
                examples.extend(unique_new)
                seen.update(ex['text'] for ex in unique_new)
                total_generated += len(unique_new)
                save_examples(examples, 'reasoning_examples.txt')
                print(f"\rTotal examples: {total_generated} (+{len(unique_new)} new)", end="", flush=True)
    
    except KeyboardInterrupt:
        print(f"\nFinal count: {total_generated} examples")
        print("Examples saved to reasoning_examples.txt")
        sys.exit(0)

if __name__ == "__main__":
    main(batch_size=5)  # Increased default batch size
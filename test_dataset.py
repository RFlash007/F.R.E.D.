from datasets import load_dataset

print("Testing dataset access...")
try:
    dataset = load_dataset('tweet_eval', 'emotion')
    print(f'Dataset loaded successfully. Splits: {dataset.keys()}')
    print(f'Training examples: {len(dataset["train"])}')
    print(f'First example: {dataset["train"][0]}')
    print(f'Labels: {dataset["train"].features["label"].names}')
except Exception as e:
    print(f"Error loading dataset: {str(e)}") 
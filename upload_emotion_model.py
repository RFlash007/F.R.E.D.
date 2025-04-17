"""
upload_emotion_model.py - Upload F.R.E.D.'s Emotion Detection Model to Hugging Face Hub

This script uploads the fine-tuned emotion detection model to the Hugging Face Hub
for easy sharing and deployment.
"""

import os
import argparse
from huggingface_hub import HfFolder, Repository, create_repo
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Import our emotion detector config to ensure consistency
from Miscallaneous.EmotionDetector import OUTPUT_DIR, label_mapping, MODEL_NAME

def upload_to_huggingface(
    model_path=OUTPUT_DIR,
    repo_name="emotion-detector",
    description="Emotion detection model fine-tuned on the tweet_eval/emotion dataset. Detects four emotions: anger, joy, optimism, and sadness.",
    hf_token=None
):
    """
    Upload the emotion detection model to Hugging Face Hub.
    
    Args:
        model_path (str): Path to the fine-tuned model
        repo_name (str): Name for the Hugging Face repository
        description (str): Description of the model
        hf_token (str): Hugging Face API token
    
    Returns:
        str: URL of the uploaded model
    """
    # Verify model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Set Hugging Face token
    if hf_token:
        HfFolder.save_token(hf_token)
    elif os.environ.get("HF_TOKEN"):
        hf_token = os.environ.get("HF_TOKEN")
        HfFolder.save_token(hf_token)
    else:
        raise ValueError("No Hugging Face token provided. Please provide a token via the --token argument or set the HF_TOKEN environment variable.")

    # Validate that we have a token
    if not HfFolder.get_token():
        raise ValueError("Failed to set Hugging Face token. Please check your token and try again.")

    print(f"Using model from: {model_path}")
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Prepare model card content with fixed template strings
    model_card = f"""
# F.R.E.D. Emotion Detection Model

{description}

## Model Details

- **Base Model:** {MODEL_NAME}
- **Task:** Emotion Detection
- **Training Dataset:** tweet_eval/emotion
- **Emotions:** {', '.join(label_mapping.values())}
- **Framework:** PyTorch
- **Language:** English

## Usage

```python
from transformers import pipeline

# Load emotion classification pipeline
classifier = pipeline("text-classification", model="{repo_name}")

# Classify text
example_result = classifier("I'm so happy today!")
print(f"Emotion: {{example_result[0]['label']}}, Confidence: {{example_result[0]['score']:.4f}}")
```

## Emotion Label Mapping

```
{label_mapping}
```

## Performance

This model was fine-tuned on the tweet_eval/emotion dataset and achieves approximately 81.5% accuracy.

## Limitations

The model is specifically trained for detecting emotions in short text segments and may not perform as well on longer or more complex texts.
"""

    # Create or clone repo
    repo_url = f"https://huggingface.co/{repo_name}"
    
    try:
        # Create a temporary directory for the repo
        repo_dir = os.path.join(os.getcwd(), "temp_repo")
        os.makedirs(repo_dir, exist_ok=True)
        
        # Try to create the repository first (this will fail if it already exists)
        try:
            print(f"Creating repository {repo_name}...")
            create_repo(
                repo_id=repo_name,
                token=hf_token,
                private=False,
                exist_ok=True,
            )
            print(f"Repository created or already exists at {repo_url}")
        except Exception as e:
            print(f"Note: {str(e)} (This may be okay if the repository already exists)")
        
        # Clone the repository (will work whether it existed before or not)
        try:
            print(f"Cloning repository {repo_name}...")
            repo = Repository(
                local_dir=repo_dir,
                clone_from=repo_name,
                use_auth_token=hf_token,
                git_user="F.R.E.D.",
                git_email="fred.assistant@example.com"
            )
        except Exception as e:
            print(f"Could not clone existing repository: {str(e)}")
            print("Creating new repository...")
            # If cloning failed, create a new repository from scratch
            repo = Repository(
                local_dir=repo_dir,
                repo_type="model",
                use_auth_token=hf_token,
                git_user="F.R.E.D.",
                git_email="fred.assistant@example.com"
            )
        
        # Save model and tokenizer to the repo directory
        print("Saving model and tokenizer to repository...")
        model.save_pretrained(repo_dir)
        tokenizer.save_pretrained(repo_dir)
        
        # Create README.md
        with open(os.path.join(repo_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(model_card)
        
        # Push to hub
        print("Pushing model to Hugging Face Hub...")
        repo.push_to_hub(commit_message="Upload F.R.E.D. emotion detection model")
        
        print(f"Model successfully uploaded to {repo_url}")
        return repo_url
    
    except Exception as e:
        print(f"Error uploading model: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload F.R.E.D.'s emotion detection model to Hugging Face Hub")
    parser.add_argument("--model_path", type=str, default=OUTPUT_DIR, help="Path to the fine-tuned model")
    parser.add_argument("--repo_name", type=str, default="RFlash/emotion-detector", help="Name for the Hugging Face repository")
    parser.add_argument("--description", type=str, default="Emotion detection model fine-tuned on the tweet_eval/emotion dataset. Detects four emotions: anger, joy, optimism, and sadness.", help="Description of the model")
    parser.add_argument("--token", type=str, help="Hugging Face API token")
    
    args = parser.parse_args()
    
    upload_to_huggingface(
        model_path=args.model_path,
        repo_name=args.repo_name,
        description=args.description,
        hf_token=args.token
    ) 
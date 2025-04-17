# Uploading F.R.E.D.'s Emotion Model to Hugging Face

This guide provides step-by-step instructions for uploading your emotion detection model to the Hugging Face Hub, making it easily accessible and shareable.

## Prerequisites

- Hugging Face account
- Hugging Face API token
- Python 3.8+
- Required packages (already installed in your environment):
  - transformers
  - huggingface_hub (updated to latest version)

## Step 1: Create a Hugging Face Account

If you don't already have a Hugging Face account:

1. Go to [huggingface.co](https://huggingface.co)
2. Click "Sign Up" to create an account
3. Complete the registration process

## Step 2: Generate a Hugging Face API Token

1. Log in to your Hugging Face account
2. Go to your profile settings (click on your profile picture → Settings)
3. Navigate to the "Access Tokens" section
4. Click "New token"
5. Give it a meaningful name (e.g., "FRED-Emotion-Upload")
6. Set the appropriate permission level (at least "Write" access)
7. Click "Generate token"
8. **IMPORTANT**: Copy the token immediately, as it won't be shown again

## Step 3: Upload Your Model

You have two options to upload your model:

### Option 1: Use Environment Variable (Recommended)

1. Open a PowerShell terminal
2. Set your Hugging Face token as an environment variable:
   ```powershell
   $env:HF_TOKEN = "your_hugging_face_token"
   ```
3. Run the upload script:
   ```powershell
   python upload_emotion_model.py --repo_name "your-username/FRED-emotion-detector"
   ```

### Option 2: Provide Token Directly

Run the script with the token provided as an argument:

```powershell
python upload_emotion_model.py --repo_name "your-username/FRED-emotion-detector" --token "your_hugging_face_token"
```

## Parameters

The script accepts the following parameters:

- `--model_path`: Path to your fine-tuned model (default: `./models/emotion_detector`)
- `--repo_name`: Name for the Hugging Face repository (default: `FRED-emotion-detector`)
  - **Recommendation**: Use your username as a namespace, e.g., `your-username/FRED-emotion-detector`
- `--description`: Brief description of the model (a default is provided)
- `--token`: Your Hugging Face API token (optional if set as environment variable)

## Example Usage

```powershell
python upload_emotion_model.py --repo_name "your-username/FRED-emotion-detector" --description "F.R.E.D.'s emotion detection model fine-tuned for anger, joy, optimism, and sadness detection."
```

## After Uploading

1. The script will create or update a repository on Hugging Face Hub
2. It will upload your model files, tokenizer, and a README
3. Upon successful completion, it will print the URL to your model on Hugging Face

## Using Your Model From Hugging Face

After uploading, you can use your model from anywhere with:

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="your-username/FRED-emotion-detector")
result = classifier("I'm excited about this new feature!")
print(f"Emotion: {result[0]['label']}, Confidence: {result[0]['score']:.4f}")
```

## Troubleshooting

- **Authentication Error**: Ensure your token has the correct permissions
- **Repository Already Exists**: Use a different repository name or ensure you have write access
- **File Size Issues**: Large models may take some time to upload

## Notes

- The upload process may take several minutes depending on your internet connection speed
- The model will be publicly accessible by default
- You can manage access settings from the Hugging Face website 
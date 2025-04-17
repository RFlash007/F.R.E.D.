# F.R.E.D. Emotion Detection System

## Overview
This module enhances F.R.E.D. with emotion detection capabilities, allowing for more empathetic and context-aware responses. It uses a fine-tuned RoBERTa model trained on the tweet_eval/emotion dataset from Hugging Face, which can identify four basic emotions: anger, joy, optimism, and sadness.

## Components

### 1. EmotionDetector.py
The core module that handles:
- Fine-tuning a RoBERTa model on the emotions dataset
- Loading and saving the model
- Processing text for emotion detection
- Making predictions

### 2. EmotionIntegration.py
Integrates the emotion detection with F.R.E.D.'s chat system:
- Provides functions to modify prompts with emotion context
- Implements response templates based on detected emotions
- Offers direct patching of F.R.E.D.'s process_message function

## Requirements
All dependencies are listed in `emotion_requirements.txt`. Major requirements:
- torch
- transformers
- datasets
- numpy
- pandas
- scikit-learn

## Getting Started

### 1. Training the Model
```python
python EmotionDetector.py
# Select option 1 to train a new model
```

This will:
- Download the tweet_eval/emotion dataset from Hugging Face
- Fine-tune a RoBERTa model for emotion classification
- Save the model to ./models/emotion_detector/

### 2. Testing the Model
```python
python EmotionDetector.py
# Select option 2 to test the model with example texts
```

### 3. Integrating with F.R.E.D.
```python
python EmotionIntegration.py
# Select option 2 to apply direct integration with F.R.E.D.
```

## How It Works

### Training Process
1. The emotion dataset is downloaded and preprocessed
2. A RoBERTa model is fine-tuned for classification
3. The model is trained for 3 epochs with evaluation after each epoch
4. The best model is saved for later use

### Integration Process
There are two ways to integrate with F.R.E.D.:

#### Method 1: Direct Patching
The `direct_patch_integration()` function in EmotionIntegration.py:
- Loads the trained model
- Replaces F.R.E.D.'s process_message function with an emotion-aware version
- Adds emotion information to user inputs

#### Method 2: Manual Integration
Alternatively, you can modify Chat.py directly:

```python
# In Chat.py:
from EmotionIntegration import integrate_with_chat_module

# Replace the original process_message with the emotion-aware version
process_message = integrate_with_chat_module()
```

## Emotion Response Templates
The system includes customizable response templates for each emotion, allowing F.R.E.D. to acknowledge and respond appropriately to the user's emotional state:

- **Anger**: Acknowledges frustration and offers methodical problem-solving
- **Joy**: Recognizes positive emotion and builds on enthusiasm
- **Optimism**: Responds to hopeful outlooks with forward-thinking perspectives
- **Sadness**: Provides empathetic support and solutions

## Resume Project Highlights
This project demonstrates:
- Fine-tuning transformer models for emotion classification
- Integrating emotion awareness into conversational AI
- Implementing empathetic response generation
- Using Hugging Face datasets and transformers

## Emotion Dataset
The tweet_eval/emotion dataset contains:
- ~3,250 training examples
- ~370 validation examples
- ~1,420 test examples
- Four emotion classes: anger, joy, optimism, and sadness

## Future Enhancements
- Add more granular emotion detection (beyond the four basic emotions)
- Implement response generation that's conditioned on detected emotions
- Develop emotion tracking over conversation history for trend analysis
- Create a personalized emotion profile for individual users 
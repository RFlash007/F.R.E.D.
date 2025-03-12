"""
EmotionDetector.py - Emotion Detection System for F.R.E.D.

This module implements an emotion detection system based on the tweet_eval/emotion dataset
from Hugging Face. It uses a RoBERTa model fine-tuned for emotion classification.

The emotions detected are: anger, joy, optimism, and sadness.
"""

import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from datasets import load_dataset

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "roberta-base"  # Base model to fine-tune
EMOTIONS_DATASET = "tweet_eval"  # Dataset on HuggingFace
EMOTIONS_DATASET_CONFIG = "emotion"  # Dataset configuration
OUTPUT_DIR = "./models/emotion_detector"  # Directory to save the model
NUM_LABELS = 4  # Number of emotion classes (anger, joy, optimism, sadness)
BATCH_SIZE = 16  # Batch size for training
LEARNING_RATE = 2e-5  # Learning rate for fine-tuning
NUM_EPOCHS = 3  # Number of epochs for training
MAX_SEQ_LENGTH = 128  # Maximum sequence length
SAVE_STEPS = 500  # Save model every N steps

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mapping of emotion labels
label_mapping = {
    0: "anger",
    1: "joy",
    2: "optimism",
    3: "sadness"
}

# Reverse mapping for prediction
reverse_mapping = {v: k for k, v in label_mapping.items()}

class EmotionDetector:
    """
    Emotion detection model class that provides methods for loading,
    fine-tuning and making predictions.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the emotion detector.
        
        Args:
            model_path (str, optional): Path to a fine-tuned model. If None, 
                                        will use the base model. Default is None.
        """
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading fine-tuned model from {model_path}")
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            logger.info(f"Loading base model {MODEL_NAME}")
            self.model = None
            self.tokenizer = None
        
        self.pipeline = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def download_dataset(self):
        """
        Download the emotions dataset from HuggingFace.
        
        Returns:
            dataset: The loaded dataset
        """
        logger.info(f"Downloading dataset: {EMOTIONS_DATASET}/{EMOTIONS_DATASET_CONFIG}")
        return load_dataset(EMOTIONS_DATASET, EMOTIONS_DATASET_CONFIG)
    
    def preprocess_data(self, dataset):
        """
        Preprocess the dataset for training.
        
        Args:
            dataset: The loaded dataset
            
        Returns:
            preprocessed_dataset: The preprocessed dataset ready for training
        """
        logger.info("Preprocessing dataset")
        
        # Initialize tokenizer if not already done
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=MAX_SEQ_LENGTH
            )
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Set the format for PyTorch
        tokenized_dataset = tokenized_dataset.remove_columns(["text"])
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        tokenized_dataset.set_format("torch")
        
        return tokenized_dataset
    
    def fine_tune(self, dataset=None, output_dir=OUTPUT_DIR):
        """
        Fine-tune the model on the emotions dataset.
        
        Args:
            dataset: The dataset to fine-tune on. If None, will download it.
            output_dir (str): Directory to save the fine-tuned model
            
        Returns:
            trainer: The trained Trainer object
        """
        # Download dataset if not provided
        if dataset is None:
            dataset = self.download_dataset()
        
        # Preprocess the dataset
        tokenized_dataset = self.preprocess_data(dataset)
        
        # Initialize the model if not already done
        if self.model is None:
            logger.info(f"Initializing model for fine-tuning: {MODEL_NAME}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=NUM_LABELS
            )
        
        # Define training arguments with matching evaluation and save strategies
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=NUM_EPOCHS,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",     # Changed from "steps" to "epoch" to match evaluation_strategy
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=False,
        )
        
        # Define compute metrics function
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            acc = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average='weighted')
            return {"accuracy": acc, "f1": f1}
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],  # Using validation instead of test
            compute_metrics=compute_metrics,
        )
        
        # Train the model
        logger.info("Starting fine-tuning...")
        start_time = datetime.now()
        trainer.train()
        end_time = datetime.now()
        logger.info(f"Fine-tuning completed in {end_time - start_time}")
        
        # Save the final model
        logger.info(f"Saving fine-tuned model to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Set up the pipeline for prediction
        self.setup_pipeline()
        
        return trainer
    
    def setup_pipeline(self):
        """
        Set up the prediction pipeline with the current model.
        """
        logger.info("Setting up prediction pipeline")
        self.pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
    
    def predict(self, text):
        """
        Predict the emotion for the given text.
        
        Args:
            text (str): The input text to analyze
            
        Returns:
            str: The predicted emotion
        """
        if self.pipeline is None:
            if self.model is not None and self.tokenizer is not None:
                self.setup_pipeline()
            else:
                raise ValueError("Model is not loaded. Please load or fine-tune a model first.")
        
        # Get the prediction
        result = self.pipeline(text)[0]
        
        # Extract the predicted label
        predicted_label = result['label']
        confidence = result['score']
        
        # Convert label ID to emotion name
        emotion = label_mapping.get(int(predicted_label.split('_')[-1]), "unknown")
        
        logger.debug(f"Predicted emotion: {emotion} (confidence: {confidence:.4f})")
        return emotion
    
    def evaluate(self, dataset=None):
        """
        Evaluate the model on the test dataset.
        
        Args:
            dataset: The dataset to evaluate on. If None, will download it.
            
        Returns:
            dict: A dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Please load or fine-tune a model first.")
        
        # Download dataset if not provided
        if dataset is None:
            dataset = self.download_dataset()
        
        # Preprocess the dataset
        tokenized_dataset = self.preprocess_data(dataset)
        
        # Define compute metrics function
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            
            # Calculate accuracy and F1 score
            acc = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average='weighted')
            
            # Calculate confusion matrix
            cm = confusion_matrix(labels, predictions)
            
            # Map numeric labels to emotion names
            cm_labeled = pd.DataFrame(
                cm,
                index=[label_mapping[i] for i in range(NUM_LABELS)],
                columns=[label_mapping[i] for i in range(NUM_LABELS)]
            )
            
            return {
                "accuracy": acc,
                "f1": f1,
                "confusion_matrix": cm_labeled
            }
        
        # Initialize trainer with matching strategies
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir="./eval_results",
                per_device_eval_batch_size=BATCH_SIZE,
                evaluation_strategy="epoch",
                save_strategy="epoch",
            ),
            eval_dataset=tokenized_dataset["test"],
            compute_metrics=compute_metrics,
        )
        
        # Evaluate the model
        logger.info("Evaluating model...")
        results = trainer.evaluate()
        
        for key, value in results.items():
            if key != "confusion_matrix":
                logger.info(f"{key}: {value}")
        
        return results


def train_model():
    """
    Convenience function to train the model from scratch.
    """
    detector = EmotionDetector()
    detector.fine_tune()
    return detector


def load_model(model_path=OUTPUT_DIR):
    """
    Load a pre-trained emotion detection model.
    
    Args:
        model_path (str): Path to the pre-trained model
        
    Returns:
        EmotionDetector: Initialized emotion detector with loaded model
    """
    return EmotionDetector(model_path=model_path)


def example_usage():
    """
    Example of how to use the emotion detector.
    """
    # Load the model
    detector = load_model()
    
    # Example texts
    examples = [
        "I'm really excited about this new project!",
        "I'm feeling down today, nothing seems to be going right.",
        "That movie made me so angry, I can't believe they ended it that way!",
        "I'm hopeful that things will improve soon.",
        "I can't believe they would do something so horrible!",
        "Wow! I didn't expect that plot twist at all!"
    ]
    
    # Predict emotions
    for text in examples:
        emotion = detector.predict(text)
        print(f"Text: {text}")
        print(f"Emotion: {emotion}\n")


def integrate_with_fred():
    """
    Example of how to integrate the emotion detector with F.R.E.D.
    """
    # Code to integrate with F.R.E.D.
    # This is a template that can be modified based on your specific needs
    
    # First, import the necessary modules
    from Chat import process_message
    
    # Load the emotion detector
    detector = load_model()
    
    # Original process_message function for reference
    original_process_message = process_message
    
    # Define the new process_message function that incorporates emotion detection
    def emotion_aware_process_message(user_input, ui_instance=None):
        # Detect emotion in the user input
        emotion = detector.predict(user_input)
        
        # Log the detected emotion
        logger.info(f"Detected emotion: {emotion}")
        
        # Add emotion context to the user input
        emotion_context = f"The user's message has been analyzed and shows signs of {emotion}. "
        emotion_context += f"Consider responding with appropriate tone and empathy."
        
        # Modify user_input to include emotion context
        enhanced_input = f"{user_input}\n\n[EMOTION: {emotion}]"
        
        # Process the enhanced input
        response = original_process_message(enhanced_input, ui_instance)
        
        return response
    
    # Replace the original process_message function with the emotion-aware one
    # Note: This is a conceptual example and would need to be implemented
    # in the actual Chat.py file
    
    return emotion_aware_process_message


if __name__ == "__main__":
    print("Emotion Detector for F.R.E.D.")
    print("Choose an option:")
    print("1. Train a new model")
    print("2. Test the pre-trained model with examples")
    print("3. Evaluate model performance")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == "1":
        print("Training new model...")
        detector = train_model()
        print("Training complete! Model saved to", OUTPUT_DIR)
    
    elif choice == "2":
        if os.path.exists(OUTPUT_DIR):
            print("Running example predictions...")
            example_usage()
        else:
            print(f"No trained model found at {OUTPUT_DIR}. Please train a model first.")
    
    elif choice == "3":
        if os.path.exists(OUTPUT_DIR):
            print("Evaluating model performance...")
            detector = load_model()
            results = detector.evaluate()
            print("Evaluation complete!")
        else:
            print(f"No trained model found at {OUTPUT_DIR}. Please train a model first.")
    
    else:
        print("Invalid choice.") 
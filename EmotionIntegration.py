"""
EmotionIntegration.py - Simplified Emotion Detection for F.R.E.D.

This module provides a simple interface to detect emotions in text using the Hugging Face hosted model.
"""

import os
import logging
from transformers import pipeline
from Miscallaneous.EmotionDetector import OUTPUT_DIR, label_mapping

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
emotion_detector_pipeline = None

# Constants
HF_MODEL_NAME = "RFlash/emotion-detector"  # Hugging Face model path
USE_HF_MODEL = True  # Set to Trueto use the Hugging Face model, False to use local model

# Initialize emotion detection on module import
def initialize_emotion_detection():
    """
    Initialize the emotion detection system, either from Hugging Face or local model.
    
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    global emotion_detector_pipeline
    
    try:
        if USE_HF_MODEL:
            # Use the Hugging Face hosted model
            logger.info(f"Loading emotion detection model from Hugging Face: {HF_MODEL_NAME}")
            try:
                # First try with CUDA if available
                emotion_detector_pipeline = pipeline(
                    "text-classification", 
                    model=HF_MODEL_NAME,
                    device=0  # Will use CPU if CUDA is not available
                )
                logger.info("Successfully initialized Hugging Face emotion detection model with GPU acceleration")
            except Exception as gpu_e:
                logger.warning(f"Could not initialize with GPU, falling back to CPU: {str(gpu_e)}")
                # Fall back to CPU
                emotion_detector_pipeline = pipeline(
                    "text-classification", 
                    model=HF_MODEL_NAME,
                    device=-1  # Force CPU
                )
                logger.info("Successfully initialized Hugging Face emotion detection model on CPU")
        else:
            # Use local model (legacy approach)
            from Miscallaneous.EmotionDetector import load_model, EmotionDetector
            
            if os.path.exists(OUTPUT_DIR):
                logger.info(f"Loading local emotion detection model from {OUTPUT_DIR}")
                local_detector = load_model(OUTPUT_DIR)
                
                # Ensure the pipeline is set up
                if local_detector.pipeline is None:
                    logger.info("Setting up prediction pipeline")
                    local_detector.setup_pipeline()
                
                emotion_detector_pipeline = local_detector.pipeline
            else:
                logger.warning(f"No pre-trained model found at {OUTPUT_DIR}. Falling back to Hugging Face model.")
                emotion_detector_pipeline = pipeline(
                    "text-classification", 
                    model=HF_MODEL_NAME,
                    device=-1
                )
        
        return emotion_detector_pipeline is not None
    except Exception as e:
        logger.error(f"Failed to initialize emotion detection: {str(e)}")
        return False


def get_emotion(text):
    """
    Detect the emotion in the given text.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        tuple: (emotion, confidence) or (None, 0.0) if detection failed
    """
    global emotion_detector_pipeline
    
    if emotion_detector_pipeline is None:
        if not initialize_emotion_detection():
            logger.error("Emotion detector is not initialized and initialization failed.")
            return None, 0.0
    
    try:
        # Get the prediction from the pipeline
        result = emotion_detector_pipeline(text)[0]
        
        # Extract the predicted label and confidence
        predicted_label = result['label']
        confidence = result['score']
        
        # Convert label ID to emotion name
        emotion_id = int(predicted_label.split('_')[-1])
        emotion = label_mapping.get(emotion_id, "unknown")
        
        logger.info(f"Detected emotion: {emotion} (confidence: {confidence:.4f})")
        return emotion, confidence
    except Exception as e:
        logger.error(f"Error detecting emotion: {str(e)}")
        return None, 0.0


# Auto-initialize on module import
initialize_emotion_detection()


if __name__ == "__main__":
    print("F.R.E.D. Emotion Detection")
    print("=========================")
    
    # Initialize emotion detection
    if emotion_detector_pipeline is not None:
        print("Emotion detection initialized successfully.")
        if USE_HF_MODEL:
            print(f"Using Hugging Face model: {HF_MODEL_NAME}")
        else:
            print("Using local model")
        
        # Simple loop to test emotion detection
        print("\nEnter text to detect emotion (or 'exit' to quit):")
        while True:
            text = input("\nText: ")
            if text.lower() == 'exit':
                break
                
            emotion, confidence = get_emotion(text)
            print(f"Detected emotion: {emotion}")
            print(f"Confidence: {confidence:.4f}")
    else:
        print("Failed to initialize emotion detection.")
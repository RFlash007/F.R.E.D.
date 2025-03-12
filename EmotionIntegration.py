"""
EmotionIntegration.py - Simplified Emotion Detection for F.R.E.D.

This module provides a simple interface to detect emotions in text.
"""

import os
import logging
from EmotionDetector import load_model, EmotionDetector, OUTPUT_DIR, label_mapping

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
emotion_detector = None

def initialize_emotion_detection():
    """
    Initialize the emotion detection system.
    
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    global emotion_detector
    
    try:
        # Check if the model exists
        if os.path.exists(OUTPUT_DIR):
            logger.info(f"Loading emotion detection model from {OUTPUT_DIR}")
            emotion_detector = load_model(OUTPUT_DIR)
        else:
            logger.warning(f"No pre-trained model found at {OUTPUT_DIR}. Creating new detector instance.")
            emotion_detector = EmotionDetector()
        
        return True
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
    global emotion_detector
    
    if emotion_detector is None:
        if not initialize_emotion_detection():
            logger.error("Emotion detector is not initialized and initialization failed.")
            return None, 0.0
    
    try:
        # Get the prediction from the pipeline
        result = emotion_detector.pipeline(text)[0]
        
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


if __name__ == "__main__":
    print("F.R.E.D. Emotion Detection")
    print("=========================")
    
    # Initialize emotion detection
    if initialize_emotion_detection():
        print("Emotion detection initialized successfully.")
        
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
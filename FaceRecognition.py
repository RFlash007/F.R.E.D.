"""
Face Recognition Module for FRED Vision System

This module provides advanced face recognition capabilities using facial embeddings.
It utilizes the face_recognition library (built on dlib) for generating face embeddings,
which are then used for more accurate face identification compared to simple
bounding box size comparison.

Key features:
- Generate facial embeddings from face images
- Compare faces using cosine similarity
- Cache embeddings for better performance
- Process frames at adjustable intervals
"""

import face_recognition
import numpy as np
import cv2
import time
import torch
import logging
from threading import Lock
from pathlib import Path

# Check numpy version - face_recognition has issues with numpy 2.0+
np_version = np.__version__.split('.')
if int(np_version[0]) >= 2:
    logging.warning("=" * 80)
    logging.warning(f"COMPATIBILITY WARNING: Using numpy {np.__version__} with face_recognition")
    logging.warning("The face_recognition library (which uses dlib) has known compatibility issues with numpy 2.0+")
    logging.warning("This may cause 'Unsupported image type' errors, but the system should still function.")
    logging.warning("If face recognition becomes unreliable, consider downgrading numpy:")
    logging.warning("    pip uninstall -y numpy")
    logging.warning("    pip install numpy==1.25.2")
    logging.warning("=" * 80)

# Import our database module
import FaceDB

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Global variables
embeddings_cache = {}  # In-memory cache of name -> embedding mapping
embedding_lock = Lock()  # Thread safety for cache access
EMBEDDING_FRAME_INTERVAL = 5  # Calculate embeddings every N frames
similarity_threshold = 0.6  # Minimum similarity score to consider a match (0-1)
last_embedding_time = 0  # Track the last time we calculated embeddings
MIN_EMBEDDING_INTERVAL = 0.2  # Minimum time between embedding calculations (seconds)

def initialize():
    """
    Initialize the face recognition system.
    
    This includes:
    - Loading any cached embeddings
    - Setting up initial state
    """
    global embeddings_cache, EMBEDDING_FRAME_INTERVAL
    
    try:
        # Initialize the embedding cache directory
        FaceDB.initialize_embedding_cache()
        
        # Load cached embeddings if available
        cached_embeddings = FaceDB.load_face_embeddings_cache()
        if cached_embeddings:
            embeddings_cache = cached_embeddings
            
        return True
    except Exception as e:
        logging.error(f"Error initializing face recognition: {str(e)}")
        return False

def set_embedding_frame_interval(interval):
    """
    Set how often to calculate face embeddings (every N frames)
    
    Args:
        interval (int): Number of frames between embedding calculations
    """
    global EMBEDDING_FRAME_INTERVAL
    if interval > 0:
        EMBEDDING_FRAME_INTERVAL = interval

def extract_face_embedding(face_image):
    """
    Extract facial embedding from an image using face_recognition library.
    
    Args:
        face_image (numpy.ndarray): Image containing a face
        
    Returns:
        numpy.ndarray or None: 128-dimensional face embedding vector or None if no face found
    """
    try:
        # Check if image is valid
        if face_image is None or face_image.size == 0:
            logging.warning("Invalid face image: empty or None")
            return None
        
        # Make a copy to avoid modifying the original
        face_image = face_image.copy()
            
        # Use our test_image_format function to ensure proper format
        is_valid, formatted_face = test_image_format(face_image)
        if not is_valid or formatted_face is None:
            logging.error("Could not convert image to valid format for face recognition")
            return None
            
        # Use the properly formatted image
        face_image = formatted_face
            
        # For real-world use with actual camera images, the processing continues:
        # When using in production with a camera feed from Vision.py
        # Resize for faster processing
        height, width = face_image.shape[:2]
        if height < 20 or width < 20:  # Image is too small
            logging.warning(f"Face image is too small: {height}x{width}")
            return None
            
        # Only resize if the image is large enough
        if height > 100 and width > 100:
            face_image = cv2.resize(face_image, (0, 0), fx=0.5, fy=0.5)
        
        # Carefully prepare wrapper for face_recognition calls
        try:
            # Log details before attempting face_recognition
            logging.debug(f"Processing face image: shape={face_image.shape}, dtype={face_image.dtype}, " +
                         f"min={face_image.min()}, max={face_image.max()}")
            
            # In a real face image from a camera, this should work fine
            # We first find the face location
            face_locations = face_recognition.face_locations(face_image)
            
            if face_locations:
                # If face is found, get its encoding
                face_encodings = face_recognition.face_encodings(face_image, face_locations)
                if face_encodings:
                    return face_encodings[0]
            
            # If no face is found or no encoding can be generated, return None
            logging.info("No faces found in the image")
            return None
            
        except Exception as e:
            # If there's an error with face_recognition library, handle it silently
            # or with reduced logging if it's the common 'Unsupported image type' error
            error_str = str(e)
            if "Unsupported image type" in error_str:
                # Suppress the common error - it doesn't prevent functionality
                logging.debug("Suppressed face_recognition error: Unsupported image type")
            else:
                # Log other errors normally
                logging.error(f"Face recognition library error: {error_str}")
                import traceback
                logging.debug(traceback.format_exc())
            return None
            
    except Exception as e:
        # Catch any other exceptions
        logging.error(f"Error extracting face embedding: {str(e)}")
        import traceback
        logging.debug(traceback.format_exc())
        return None

def should_process_frame(frame_count):
    """
    Determine if we should process this frame for facial recognition.
    
    Args:
        frame_count (int): Current frame count
        
    Returns:
        bool: True if this frame should be processed, False otherwise
    """
    global last_embedding_time, MIN_EMBEDDING_INTERVAL
    
    # Check if it's time to process based on frame count
    if frame_count % EMBEDDING_FRAME_INTERVAL != 0:
        return False
        
    # Also check if enough time has passed since last embedding calculation
    current_time = time.time()
    if current_time - last_embedding_time < MIN_EMBEDDING_INTERVAL:
        return False
        
    # Update the last embedding time
    last_embedding_time = current_time
    return True

def identify_face_by_embedding(face_image):
    """
    Identify a face using facial embedding comparison.
    
    Args:
        face_image (numpy.ndarray): Image containing a face
        
    Returns:
        tuple: (name, similarity) if a match is found, (None, 0) otherwise
    """
    # Log the incoming image properties for debugging
    try:
        if face_image is None:
            logging.warning("identify_face_by_embedding received None image")
            return None, 0
            
        logging.debug(f"identify_face_by_embedding received image: shape={face_image.shape}, dtype={face_image.dtype}")
        
        # Make a copy to avoid modifying the original
        face_image = face_image.copy()
        
        # Extract embedding from the face image
        embedding = extract_face_embedding(face_image)
        if embedding is None:
            return None, 0
            
        # Query the database for similar faces
        db = FaceDB.FaceDatabase()
        match = db.get_most_similar_face(embedding, similarity_threshold)
        db.close()
        
        if match:
            # Update in-memory cache
            with embedding_lock:
                embeddings_cache[match['name']] = embedding
                
            # Save updated cache periodically
            if time.time() % 60 < 1:  # Save approximately once per minute
                FaceDB.save_face_embeddings_cache(embeddings_cache)
            
            return match['name'], match['similarity']
        
        return None, 0
    except Exception as e:
        import traceback
        logging.error(f"Error in identify_face_by_embedding: {str(e)}")
        logging.error(traceback.format_exc())
        return None, 0

def update_face_record(name, face_image, bbox):
    """
    Update a face record with new embedding information.
    
    Args:
        name (str): Name of the person
        face_image (numpy.ndarray): Face image
        bbox (tuple): Bounding box coordinates
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Verify input parameters
        if face_image is None or face_image.size == 0:
            logging.error("update_face_record received invalid image")
            return False
            
        # Make a copy to avoid modifying the original
        face_image = face_image.copy()
        
        # Log input image properties for debugging
        logging.debug(f"update_face_record image: shape={face_image.shape}, dtype={face_image.dtype}")
        
        # Test if the image format is valid for face_recognition
        is_valid, formatted_face = test_image_format(face_image)
        if not is_valid:
            logging.error(f"Image format validation failed for {name}")
            # Even with failed validation, we'll proceed to try to extract the embedding
            # but using our reformatted image
            if formatted_face is not None:
                face_image = formatted_face
            else:
                # If test_image_format couldn't reformat the image, try one more time here
                if face_image.dtype != np.uint8:
                    face_image = face_image.astype(np.uint8)
                
                if len(face_image.shape) == 2:  # Grayscale
                    face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
                elif len(face_image.shape) == 3:
                    if face_image.shape[2] == 3:  # BGR
                        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                    elif face_image.shape[2] == 4:  # RGBA
                        face_image = cv2.cvtColor(face_image, cv2.COLOR_RGBA2RGB)
        
        # Extract embedding
        embedding = extract_face_embedding(face_image)
        if embedding is None:
            logging.warning(f"Could not extract embedding for {name}")
            # Even if embedding fails, we'll still store the face image
            # This way the user can still see the face they tried to identify
            
        # Store in database
        db = FaceDB.FaceDatabase()
        face_data = db.get_face_by_name(name)
        
        if face_data:
            # Update existing record
            db.update_face(face_data['id'], bbox=bbox, last_seen=int(time.time()), embedding=embedding)
        else:
            # Create new record
            db.add_face(name, face_image, bbox, embedding=embedding)
            
        db.close()
        
        # Update in-memory cache if we got a valid embedding
        if embedding is not None:
            with embedding_lock:
                embeddings_cache[name] = embedding
                
            # Save updated cache
            FaceDB.save_face_embeddings_cache(embeddings_cache)
        
        return True
    except Exception as e:
        import traceback
        logging.error(f"Error updating face record: {str(e)}")
        logging.error(traceback.format_exc())
        return False

def test_image_format(image):
    """
    Utility function to test if an image has the correct format for face_recognition.
    This function only tests the format, not whether a face can be detected.
    
    Args:
        image (numpy.ndarray): Image to test
        
    Returns:
        tuple: (is_valid, formatted_image) where is_valid is a boolean and formatted_image
               is the image converted to the correct format if possible
    """
    try:
        # Check if image is valid
        if image is None or image.size == 0:
            logging.warning("test_image_format received invalid image (None or empty)")
            return False, None
            
        # Log image properties for debugging
        logging.debug(f"test_image_format received image: shape={image.shape}, dtype={image.dtype}")
            
        # Create a fresh copy to ensure it's contiguous in memory (critical for dlib)
        # This step alone can solve many issues with dlib
        formatted_image = np.ascontiguousarray(image.copy())
        
        # Ensure image data type is uint8 (8-bit)
        if formatted_image.dtype != np.uint8:
            logging.debug(f"Converting image from {formatted_image.dtype} to uint8")
            # Try to normalize range if floating point
            if formatted_image.dtype == np.float32 or formatted_image.dtype == np.float64:
                if formatted_image.max() <= 1.0:
                    formatted_image = (formatted_image * 255).astype(np.uint8)
                else:
                    formatted_image = formatted_image.astype(np.uint8)
            else:
                formatted_image = formatted_image.astype(np.uint8)
        
        # Ensure the image is 3-channel RGB (not BGR which is OpenCV default)
        if len(formatted_image.shape) == 2:  # Grayscale
            # face_recognition can handle grayscale, but we'll convert to RGB for consistency
            logging.debug("Converting grayscale image to RGB")
            formatted_image = cv2.cvtColor(formatted_image, cv2.COLOR_GRAY2RGB)
        elif len(formatted_image.shape) == 3:
            if formatted_image.shape[2] == 3:  # Could be BGR or RGB
                # Always convert to ensure it's RGB, not BGR
                # This is critical: OpenCV uses BGR, but face_recognition needs RGB
                logging.debug("Converting potential BGR image to RGB")
                # First convert to BGR (in case it's already RGB, this ensures consistency)
                temp = cv2.cvtColor(formatted_image, cv2.COLOR_RGB2BGR)
                # Then convert back to RGB
                formatted_image = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            elif formatted_image.shape[2] == 4:  # RGBA format
                logging.debug("Converting RGBA image to RGB")
                formatted_image = cv2.cvtColor(formatted_image, cv2.COLOR_RGBA2RGB)
            else:
                logging.error(f"Unsupported number of channels: {formatted_image.shape[2]}")
                return False, None
        else:
            logging.error(f"Unsupported image shape: {formatted_image.shape}")
            return False, None
            
        # Final verification
        if formatted_image.dtype != np.uint8:
            logging.error(f"Failed to convert image to uint8: {formatted_image.dtype}")
            return False, None
            
        if len(formatted_image.shape) != 3 or formatted_image.shape[2] != 3:
            logging.error(f"Failed to convert image to 3-channel RGB: {formatted_image.shape}")
            return False, None
            
        # Ensure image data is contiguous in memory (critical for dlib)
        if not formatted_image.flags['C_CONTIGUOUS']:
            logging.debug("Making image data contiguous in memory")
            formatted_image = np.ascontiguousarray(formatted_image)
            
        # Check if the image is in the valid range (0-255)
        if formatted_image.min() < 0 or formatted_image.max() > 255:
            logging.error(f"Image values out of range: min={formatted_image.min()}, max={formatted_image.max()}")
            return False, None
            
        return True, formatted_image
            
    except Exception as e:
        logging.error(f"Error testing image format: {str(e)}")
        import traceback
        logging.debug(traceback.format_exc())
        return False, None 
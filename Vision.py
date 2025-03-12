#!/usr/bin/env python3
"""
FRED Vision System - Object Detection and Face Recognition.

This module provides vision capabilities for the FRED AI Assistant:
- YOLOv8 object detection 
- Face recognition with facial embeddings
- Face identification with database storage
"""

import os
import cv2
import time
import math
import queue
import logging
import threading
import numpy as np
import traceback
from pathlib import Path

# For timestamp generation

# Set up proper logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Try to import the YOLO model
try:
    from ultralytics import YOLO
except ImportError:
    logging.error("Error importing YOLO - please install with 'pip install ultralytics'")
    YOLO = None

# Import FRED-specific modules
try:
    import FaceDB                  # For face database operations
    import FaceRecognition         # For facial embedding recognition
except ImportError as e:
    logging.error(f"Error importing FRED modules: {str(e)}")

# Global variables for vision system
vision_active = False
vision_thread = None
vision_system = None  # Global variable to track the VisionSystem instance
detected_objects = []
current_frame = None
frame_lock = threading.Lock()
frame_count = 0  # Count frames for embedding calculation

# Face recognition thread variables
face_recognition_thread = None
face_queue = queue.Queue(maxsize=5)  # Limit queue size to avoid memory issues
face_recognition_active = False

# Configuration parameters
max_detected_objects = 4  # Maximum number of objects to detect

# Threshold parameters
confidence_threshold = 0.5  # Minimum confidence to display a detection
detection_threshold = 0.3   # Lower threshold for detection

# Base directory for models
MODEL_DIR = Path("./models/vision")

# Runtime cache for known faces
known_faces = {}

class VisionSystem:
    def __init__(self):
        """Initialize the FRED Vision System with YOLOv8s object detection"""
        self.cap = None
        self.model = None
        self.classes = []
        self.colors = []
        
        # Vision system state
        self.is_running = False
        self.show_window = False
        
        # JARVIS UI Colors - Darker purple tones as requested
        self.jarvis_colors = {
            'stark_blue': '#5a0096',     # Darker purple
            'stark_glow': '#8a30c5',     # Darker arc reactor glow
            'accent': '#6a30a8',         # Darker primary accent
            'accent_bright': '#9950cc',  # Darker bright highlights
            'accent_dim': '#321456',     # Deeper dark purple
            'hologram': '#9972cc',       # Darker hologram lavender
            'success': '#42ffac',        # Success green 
            'warning': '#ffac42',        # Warning amber
            'error': '#ff4278',          # Error red
        }
        
        # Initialize detection model
        self._load_model()
    
    def _load_model(self):
        """Load the YOLOv8s object detection model with Open Images V7 dataset (600+ classes)"""
        try:
            # Create models directory if it doesn't exist
            os.makedirs(MODEL_DIR, exist_ok=True)
            
            # Suppress warnings and logging
            import warnings
            warnings.filterwarnings("ignore")
            os.environ["PYTHONWARNINGS"] = "ignore"
            
            # Load YOLO model - upgraded to YOLOv8s (small) from YOLOv8n (nano)
            # The small variant provides better accuracy while maintaining reasonable performance
            model_filename = "yolov8s-oiv7.pt"
            local_model_path = MODEL_DIR / model_filename
            
            # If model doesn't exist locally, we'll let YOLO download it
            if not local_model_path.exists():
                logging.info(f"YOLOv8s model not found at {local_model_path}")
                logging.info(f"Downloading YOLOv8s model...")
                # The YOLO constructor will download the model if not found
                self.model = YOLO(model_filename, verbose=False)
                # Save the model to our models directory
                logging.info(f"Saving model to {local_model_path}")
                if not local_model_path.parent.exists():
                    local_model_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(local_model_path)
            else:
                logging.info(f"Loading YOLOv8s model from {local_model_path}")
                self.model = YOLO(local_model_path, verbose=False)
            # For Open Images V7 dataset (600+ classes), we need to adjust the color generation
            # to match the larger number of possible classes
            self.colors = np.random.uniform(0, 255, size=(600, 3))
                
            logging.info("YOLOv8s model loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading vision model: {str(e)}")
            logging.error(traceback.format_exc())
            raise
    
    def start(self, camera_index=0, show_window=False):
        """Start the vision system using the specified camera"""
        global vision_active
        
        # Check if vision system is already active
        if vision_active:
            return True
        
        # Initialize video capture with specified camera index
        self.cap = cv2.VideoCapture(camera_index)
        
        # Verify camera was successfully opened
        if not self.cap.isOpened():
            return False
        
        # Set system running state to True
        self.is_running = True
        
        # Store whether to show the video window
        self.show_window = show_window
        
        # Update global vision active state
        vision_active = True
        
        # Return success status
        return True
    
    def stop(self):
        """Stop the vision system"""
        global vision_active
        
        self.is_running = False
        vision_active = False
        
        if self.cap is not None:
            self.cap.release()
        
        if self.show_window:
            cv2.destroyAllWindows()
    
    def detect_objects(self, frame):
        """
        Detect objects in a video frame.
        
        This method processes a video frame through the YOLOv8 model to detect objects
        and handles the recognition of previously identified faces.
        
        Args:
            frame (numpy.ndarray): Video frame to process
            
        Returns:
            list: Detected objects with their metadata
        """
        # Basic validation
        if frame is None or self.model is None:
            return []
        
        # Run YOLOv8 inference with the detection threshold
        results = self.model(frame, conf=detection_threshold, verbose=False)
        
        # Process detection results
        all_detections = []
        
        # Process each detection from the model
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            
            for i in range(len(boxes)):
                box = boxes[i]
                # Get bounding box coordinates (x1, y1, x2, y2 format)
                x1, y1, x2, y2 = box.xyxy[0]
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Convert coordinates to integers for drawing and processing
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Skip boxes that are too small
                if (x2 - x1) < 10 or (y2 - y1) < 10:
                    continue
                
                # Skip low confidence detections
                if confidence < confidence_threshold:
                    continue
                
                # Convert to center format with width and height for easier processing
                x = int((x1 + x2) / 2)  # center x
                y = int((y1 + y2) / 2)  # center y
                w = int(x2 - x1)        # width
                h = int(y2 - y1)        # height
                
                # Get class name from the model's class list
                label = result.names[class_id]
                
                # If this is a face detection, check if we've already identified this person
                if label == "Human face":
                    # Format coordinates for face comparison
                    face_box = (x1, y1, x2, y2)
                    
                    # Try to match with a known face using our face verification helper
                    if self._is_known_face(frame, face_box):
                        # Update label to reflect identified person
                        face_image = frame[y1:y2, x1:x2].copy()
                        success, identified_name, _ = self._verify_face_with_embedding(face_image)
                        if success and identified_name:
                            label = identified_name
                    
                    # Add to background processing queue if the face is large enough
                    # and the queue isn't full, regardless of whether we found a match
                    if (x2 - x1) >= 20 and (y2 - y1) >= 20 and face_recognition_active:
                        try:
                            # Extract face image for processing
                            face_image = frame[y1:y2, x1:x2].copy()
                            # Add to queue for background processing
                            if not face_queue.full():
                                face_queue.put_nowait((face_image, face_box, label))
                        except queue.Full:
                            # Queue is full, skip this frame
                            pass
                        except Exception as e:
                            logging.error(f"Error adding face to queue: {str(e)}")
                
                # Choose visualization color based on class
                jarvis_color_keys = list(self.jarvis_colors.keys())
                jarvis_color_hex = self.jarvis_colors[jarvis_color_keys[class_id % len(jarvis_color_keys)]]
                
                # Convert hex color to BGR for OpenCV
                r = int(jarvis_color_hex[1:3], 16)
                g = int(jarvis_color_hex[3:5], 16)
                b = int(jarvis_color_hex[5:7], 16)
                jarvis_color = (b, g, r)  # OpenCV uses BGR color order
                
                # Get status color for the default "DETECTED" status
                status_color_hex = self.jarvis_colors['stark_glow']
                r = int(status_color_hex[1:3], 16)
                g = int(status_color_hex[3:5], 16)
                b = int(status_color_hex[5:7], 16)
                status_color = (b, g, r)  # OpenCV uses BGR color order
                
                # Create detection object with all metadata
                detection = {
                    'label': label,
                    'confidence': confidence,
                    'box': (x, y, w, h),
                    'xyxy': (x1, y1, x2, y2),
                    'color': jarvis_color,
                    'status': "DETECTED", # Status for new detections
                    'status_color': status_color  # Status color as BGR tuple
                }
                
                # If this is an identified face, update the status
                if label != "Human face" and "Human face" in label:
                    detection['status'] = "IDENTIFIED"
                    # Convert success color to BGR
                    success_hex = self.jarvis_colors['success']
                    r = int(success_hex[1:3], 16)
                    g = int(success_hex[3:5], 16)
                    b = int(success_hex[5:7], 16)
                    detection['status_color'] = (b, g, r)  # BGR tuple
                
                # Add to all detections list
                all_detections.append(detection)
        
        # Sort detections by confidence (highest first)
        all_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Get the top detections based on confidence
        all_objects = all_detections[:max_detected_objects]
        
        # Draw objects on frame if window is shown
        if self.show_window:
            self._draw_objects(frame, all_objects)
        
        return all_objects
    
    def _draw_objects(self, frame, objects):
        """Draw all objects on the frame with circular Stark-inspired aesthetic"""
        frame_h, frame_w = frame.shape[:2]
        
        for obj in objects:
            # Skip objects with negative or zero-size boxes
            x1, y1, x2, y2 = obj['xyxy']
            if x2 <= x1 or y2 <= y1:
                continue
                
            x, y, w, h = obj['box']
            color = obj['color']
            
            # Get the label for this object
            original_label = obj['label']
            
            # For debug: log any identified faces
            if original_label != "Human face" and "Human face" in original_label:
                logging.debug(f"Drawing identified face: {original_label}")
                
            # Clean up the label - extract just the person's name if this is an identified face
            # We need to show just the person name, not "Person Human face"
            display_label = original_label
            if original_label != "Human face" and "Human face" in original_label:
                # Remove "Human face" text to get the clean name
                display_label = original_label.replace(" Human face", "").replace("Human face ", "").strip()
                if not display_label:  # Fallback if somehow removing "Human face" leaves nothing
                    display_label = original_label
                logging.debug(f"Cleaned label for display: '{display_label}'")
                
            # Create overlay for transparency
            overlay = frame.copy()
            
            # Status text (DETECTED, IDENTIFIED, etc.)
            status = obj.get('status', "DETECTED")
            status_color_hex = obj.get('status_color', self.jarvis_colors['stark_glow'])
            
            # Convert hex color to BGR for OpenCV
            if isinstance(status_color_hex, str) and status_color_hex.startswith('#'):
                r = int(status_color_hex[1:3], 16)
                g = int(status_color_hex[3:5], 16)
                b = int(status_color_hex[5:7], 16)
                status_color = (b, g, r)  # OpenCV uses BGR color order
            else:
                # If already a tuple, use it directly
                status_color = status_color_hex
                
            # Convert stark blue for main circle
            stark_blue_hex = self.jarvis_colors['stark_blue']
            r_blue = int(stark_blue_hex[1:3], 16)
            g_blue = int(stark_blue_hex[3:5], 16)
            b_blue = int(stark_blue_hex[5:7], 16)
            stark_blue = (b_blue, g_blue, r_blue)
            
            # Use glow color for outer elements
            glow_color = status_color
            
            # For identified faces, use success color
            is_identified = status == "IDENTIFIED" or (original_label != "Human face" and "Human face" in original_label)
            if is_identified:
                success_hex = self.jarvis_colors['success']
                r_success = int(success_hex[1:3], 16)
                g_success = int(success_hex[3:5], 16)
                b_success = int(success_hex[5:7], 16)
                glow_color = (b_success, g_success, r_success)
            
            # Draw the main circle around object center
            radius = int(max(w, h) / 2)
            cv2.circle(overlay, (x, y), radius, stark_blue, 2)
            
            # Draw outer dashed segments using ellipse for smoother arcs
            outer_radius = radius + 10
            segments = 16
            for i in range(segments):
                if i % 2 == 0:  # Skip every other segment to create dashes
                    angle_start = i * (360 / segments)
                    angle_end = (i + 1) * (360 / segments)
                    
                    # Draw arc using ellipse for smoother curves
                    cv2.ellipse(overlay, (x, y), (outer_radius, outer_radius), 
                               0, angle_start, angle_end, glow_color, 2)
            
            # Calculate text size for position
            confidence = obj.get('confidence', 0)
            confidence_text = f"{confidence:.2f}" if confidence > 0 else ""
            
            # Use different font sizes based on the box size
            box_area = w * h
            font_scale = min(1.0, max(0.4, math.sqrt(box_area) / 400))
            
            # Calculate data panel position
            panel_width = 150
            panel_height = 70
            
            # Position panel based on available space
            space_left = x1
            space_right = frame_w - x2
            space_top = y1
            space_bottom = frame_h - y2
            spaces = [space_right, space_bottom, space_left, space_top]
            max_space_idx = spaces.index(max(spaces))
            
            if max_space_idx == 0:  # Right has most space
                panel_x = x2 + 10
                panel_y = max(y - 35, 30)
                # Ensure panel doesn't go off right edge
                panel_x = min(panel_x, frame_w - panel_width - 10)
            elif max_space_idx == 1:  # Bottom has most space
                panel_x = max(x - 75, 10)
                panel_y = y2 + 10
                # Ensure panel doesn't go off bottom edge
                panel_y = min(panel_y, frame_h - panel_height - 10)
            elif max_space_idx == 2:  # Left has most space
                panel_x = max(x1 - 160, 10)
                panel_y = max(y - 35, 30)
            else:  # Top has most space
                panel_x = max(x - 75, 10)
                panel_y = max(y1 - 90, 30)
            
            # Create data panel background
            cv2.rectangle(overlay, 
                        (panel_x, panel_y), 
                        (panel_x + panel_width, panel_y + panel_height), 
                        (0, 0, 0), -1)
            
            # Add technical lines around the panel
            corner_len = 20
            # Top-left corner
            cv2.line(overlay, (panel_x, panel_y), (panel_x + corner_len, panel_y), stark_blue, 2)
            cv2.line(overlay, (panel_x, panel_y), (panel_x, panel_y + corner_len), stark_blue, 2)
            
            # Top-right corner
            cv2.line(overlay, (panel_x + panel_width, panel_y), (panel_x + panel_width - corner_len, panel_y), stark_blue, 2)
            cv2.line(overlay, (panel_x + panel_width, panel_y), (panel_x + panel_width, panel_y + corner_len), stark_blue, 2)
            
            # Bottom-left corner
            cv2.line(overlay, (panel_x, panel_y + panel_height), (panel_x + corner_len, panel_y + panel_height), stark_blue, 2)
            cv2.line(overlay, (panel_x, panel_y + panel_height), (panel_x, panel_y + panel_height - corner_len), stark_blue, 2)
            
            # Bottom-right corner
            cv2.line(overlay, (panel_x + panel_width, panel_y + panel_height), (panel_x + panel_width - corner_len, panel_y + panel_height), stark_blue, 2)
            cv2.line(overlay, (panel_x + panel_width, panel_y + panel_height), (panel_x + panel_width, panel_y + panel_height - corner_len), stark_blue, 2)
            
            # Add text data to panel in JARVIS style
            # Target label - use the cleaned display label
            text_label = display_label.upper()  # Use uppercase for display
            cv2.putText(overlay, f"TARGET: {text_label}", 
                       (panel_x + 5, panel_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, glow_color, 1)
            
            # Confidence
            line_y = panel_y + 40
            cv2.putText(overlay, f"CONFIDENCE: {obj['confidence']:.2f}", 
                       (panel_x + 5, line_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, glow_color, 1)
            line_y += 20
            
            # Status with dynamic color based on status
            cv2.putText(overlay, f"STATUS: {status}", 
                       (panel_x + 5, line_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
            
            # Calculate connecting line endpoints
            panel_center_x = panel_x + panel_width // 2
            panel_center_y = panel_y + panel_height // 2
            
            # Determine line start point (on panel edge)
            if max_space_idx == 0:  # Panel on right
                line_start = (panel_x, panel_center_y)
            elif max_space_idx == 1:  # Panel below
                line_start = (panel_center_x, panel_y)
            elif max_space_idx == 2:  # Panel on left
                line_start = (panel_x + panel_width, panel_center_y)
            else:  # Panel above
                line_start = (panel_center_x, panel_y + panel_height)
            
            # Calculate line end point (on object edge)
            # Find intersection of line from panel center to object center with object circle
            dx = x - panel_center_x
            dy = y - panel_center_y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Normalize direction vector
            if distance > 0:
                dx /= distance
                dy /= distance
            
            # Calculate end point on object edge
            line_end = (int(x - dx * radius), int(y - dy * radius))
            
            # Draw connecting line with dot at end
            cv2.line(overlay, line_start, line_end, stark_blue, 1)
            cv2.circle(overlay, line_end, 3, glow_color, -1)
            
            # Apply overlay with transparency
            alpha = 0.7  # Transparency factor
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    def process_frame(self):
        """Process a single frame from the camera"""
        global current_frame, detected_objects, frame_count
        
        if not self.is_running or self.cap is None:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            return
        
        # Increment frame counter for embedding calculations
        frame_count += 1
        
        # Detect objects in the frame
        detections = self.detect_objects(frame)
        
        # Ensure identifications persist between frames
        # This is critical for keeping names displayed consistently
        preserved_detections = []
        for detection in detections:
            # Check if this detection might match any existing one with a name
            # This helps ensure names persist between frames even without tracking
            matches_existing = False
            for existing in detected_objects:
                # Only consider existing objects that have custom names (not "Human face")
                if existing.get('label') != "Human face" and "Human face" in existing.get('label', ''):
                    # Get bounding boxes
                    if 'xyxy' in detection and 'xyxy' in existing:
                        d_x1, d_y1, d_x2, d_y2 = detection['xyxy']
                        e_x1, e_y1, e_x2, e_y2 = existing['xyxy']
                        
                        # Calculate centers
                        d_center_x = (d_x1 + d_x2) / 2
                        d_center_y = (d_y1 + d_y2) / 2
                        e_center_x = (e_x1 + e_x2) / 2
                        e_center_y = (e_y1 + e_y2) / 2
                        
                        # Calculate distance between centers
                        distance = math.sqrt(
                            (d_center_x - e_center_x) ** 2 + 
                            (d_center_y - e_center_y) ** 2
                        )
                        
                        # If the detection is close to an existing named object, preserve the name
                        if distance < 50 and detection.get('label') == "Human face":
                            # Clone the detection and update with the existing name
                            updated_detection = detection.copy()
                            updated_detection['label'] = existing['label']
                            updated_detection['status'] = "IDENTIFIED"
                            
                            # Convert success color to BGR
                            success_hex = self.jarvis_colors['success']
                            r = int(success_hex[1:3], 16)
                            g = int(success_hex[3:5], 16)
                            b = int(success_hex[5:7], 16)
                            updated_detection['status_color'] = (b, g, r)
                            
                            # Add the updated detection
                            preserved_detections.append(updated_detection)
                            matches_existing = True
                            logging.debug(f"Preserved identification: {existing['label']}")
                            break
            
            # If no match found, add the original detection
            if not matches_existing:
                preserved_detections.append(detection)
        
        # Update global variables with thread safety
        with frame_lock:
            current_frame = frame.copy()
            # Use the preserved detections to ensure names persist
            detected_objects = preserved_detections
        
        # Display frame if window is enabled
        if self.show_window:
            # Draw on a copy to avoid modifying the original
            display_frame = frame.copy()
            self._draw_objects(display_frame, detected_objects)
            cv2.imshow("FRED Vision System", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
    
    def _verify_face_with_embedding(self, face_image, name=None):
        """
        Helper method to verify a face using facial embeddings
        
        Args:
            face_image (numpy.ndarray): The face image to verify
            name (str, optional): The name to verify against. If None, just tries to extract embedding.
            
        Returns:
            tuple: (success, name, similarity) where:
                - success (bool): Whether the operation succeeded
                - name (str or None): The matched name or None if no match
                - similarity (float): Similarity score (0-1) or 0 if no match
        """
        global known_faces
        
        try:
            # Ensure valid image
            if face_image is None or face_image.size == 0:
                return False, None, 0
                
            # Make a copy to avoid modifying the original
            face_image = face_image.copy()
            
            # Format the image for face recognition
            is_valid, formatted_face = FaceRecognition.test_image_format(face_image)
            if not is_valid or formatted_face is None:
                return False, None, 0
                
            # Extract embedding
            embedding = FaceRecognition.extract_face_embedding(formatted_face)
            if embedding is None:
                return False, None, 0
            
            # Convert embedding to tensor
            current_embedding = torch.tensor(embedding, dtype=torch.float32)
                
            # If name is provided, verify against that specific person
            if name:
                # First check our in-memory cache for the most recent embedding
                if name in known_faces and 'embedding' in known_faces[name]:
                    # Use the cached embedding for faster verification
                    stored_embedding = torch.tensor(known_faces[name]['embedding'], dtype=torch.float32)
                    
                    # Calculate similarity
                    similarity = torch.cosine_similarity(
                        current_embedding.unsqueeze(0),
                        stored_embedding.unsqueeze(0),
                        dim=1
                    ).item()
                    
                    # If similarity is good, return right away without database lookup
                    if similarity >= FaceRecognition.similarity_threshold:
                        return True, name, similarity
                
                # If not in cache or similarity isn't high enough, check the database
                db = FaceDB.FaceDatabase()
                face_data = db.get_face_by_name(name)
                db.close()
                
                if face_data and face_data.get('embedding') is not None:
                    # Compare embeddings
                    stored_embedding = torch.tensor(face_data['embedding'], dtype=torch.float32)
                    
                    # Calculate similarity
                    similarity = torch.cosine_similarity(
                        current_embedding.unsqueeze(0),
                        stored_embedding.unsqueeze(0),
                        dim=1
                    ).item()
                    
                    # Update in-memory cache with this embedding for faster future matching
                    if name not in known_faces:
                        known_faces[name] = {}
                    known_faces[name]['embedding'] = face_data['embedding']
                    known_faces[name]['last_seen'] = time.time()
                    
                    return True, name, similarity
                    
                # If no embedding found, return success but low similarity
                return True, name, 0.0
            else:
                # No specific name provided, try to match against all known faces
                
                # First check our in-memory cache for faster matching
                best_match = None
                best_similarity = 0.0
                
                # Check against known faces in the cache first (much faster than DB lookup)
                for cache_name, cache_data in known_faces.items():
                    if 'embedding' in cache_data:
                        stored_embedding = torch.tensor(cache_data['embedding'], dtype=torch.float32)
                        
                        # Calculate similarity
                        similarity = torch.cosine_similarity(
                            current_embedding.unsqueeze(0),
                            stored_embedding.unsqueeze(0),
                            dim=1
                        ).item()
                        
                        # If we have a good match, update our best match info
                        if similarity > best_similarity and similarity >= FaceRecognition.similarity_threshold:
                            best_match = cache_name
                            best_similarity = similarity
                
                # If we found a good match in the cache, return it
                if best_match is not None:
                    # Update the last_seen timestamp
                    known_faces[best_match]['last_seen'] = time.time()
                    return True, best_match, best_similarity
                
                # If no match in cache, try the database
                identified_name, similarity = FaceRecognition.identify_face_by_embedding(formatted_face)
                
                # If we find a match in the DB, update our cache
                if identified_name and similarity >= FaceRecognition.similarity_threshold:
                    # Get the face data from the database
                    db = FaceDB.FaceDatabase()
                    face_data = db.get_face_by_name(identified_name)
                    db.close()
                    
                    # Update in-memory cache
                    if face_data and 'embedding' in face_data:
                        if identified_name not in known_faces:
                            known_faces[identified_name] = {}
                        known_faces[identified_name]['embedding'] = face_data['embedding']
                        known_faces[identified_name]['last_seen'] = time.time()
                
                return True, identified_name, similarity
                
            return False, None, 0
        except Exception as e:
            # Suppress common dlib errors
            error_str = str(e)
            if "Unsupported image type" in error_str:
                logging.debug("Suppressed face_recognition error in _verify_face_with_embedding")
            else:
                logging.error(f"Error in face verification: {str(e)}")
            return False, None, 0

    def _is_known_face(self, frame, bbox):
        """
        Quick check if a face box might match a known face
        
        Args:
            frame (numpy.ndarray): Video frame
            bbox (tuple): Bounding box (x1, y1, x2, y2)
            
        Returns:
            bool: True if this might be a known face, False otherwise
        """
        global known_faces
        
        # If no known faces, we can exit early
        if not known_faces:
            return False
            
        # Extract coordinates
        x1, y1, x2, y2 = bbox
        box_width = x2 - x1
        box_height = y2 - y1
        
        # Calculate center of current face
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # If box is too small, skip
        if box_width < 20 or box_height < 20:
            return False
            
        # For each known face, check if it might be the same person
        for name, face_data in known_faces.items():
            if 'bbox' not in face_data:
                continue
                
            # Get stored box
            stored_x1, stored_y1, stored_x2, stored_y2 = face_data['bbox']
            stored_width = stored_x2 - stored_x1
            stored_height = stored_y2 - stored_y1
            
            # Calculate center of stored face
            stored_center_x = (stored_x1 + stored_x2) / 2
            stored_center_y = (stored_y1 + stored_y2) / 2
            
            # Calculate distance between centers
            center_distance = math.sqrt(
                (center_x - stored_center_x) ** 2 + 
                (center_y - stored_center_y) ** 2
            )
            
            # Compare box sizes - similarity check
            width_ratio = box_width / max(1, stored_width)  # Avoid division by zero
            height_ratio = box_height / max(1, stored_height)
            
            # Size similarity (between 0.5x and 2x)
            size_similar = 0.5 <= width_ratio <= 2.0 and 0.5 <= height_ratio <= 2.0
            
            # Check if either the centers are close OR the sizes are similar
            if center_distance < 100 or size_similar:
                # If this face was seen very recently, it's more likely to be the same person
                last_seen = face_data.get('last_seen', 0)
                time_since_last_seen = time.time() - last_seen
                
                # If seen in the last 5 seconds, very likely the same person
                if time_since_last_seen < 5:
                    return True
                
                # If center is very close, likely the same person
                if center_distance < 50:
                    return True
                    
                # If both center is moderately close AND sizes are similar, likely the same person
                if center_distance < 100 and size_similar:
                    return True
                
        return False

def vision_loop(vision_system):
    """Main vision processing loop"""
    while vision_system.is_running:
        vision_system.process_frame()
        time.sleep(0.03)  # ~30fps

def start_vision_system(camera_index=0, show_window=False):
    """
    Start the vision system.
    
    This function initializes the vision system, starts camera capture,
    and loads previously identified faces from the database.
    
    Args:
        camera_index (int): Index of the camera to use (default: 0)
        show_window (bool): Whether to display the camera feed window (default: False)
        
    Returns:
        bool: True if the vision system started successfully, False otherwise
    """
    global vision_thread, vision_active, vision_system
    
    if vision_active:
        return False
    
    try:
        # First, load known faces from database for persistent recognition
        load_known_faces()
        
        # Initialize the vision system
        vision_system = VisionSystem()
        if not vision_system.start(camera_index, show_window):
            return False
        
        # Start the face recognition thread
        start_face_recognition_thread()
        
        # Start vision processing in a separate thread
        vision_thread = threading.Thread(target=vision_loop, args=(vision_system,))
        vision_thread.daemon = True
        vision_thread.start()
        
        vision_active = True
        return True
    
    except Exception as e:
        logging.error(f"Error starting vision system: {str(e)}")
        logging.error(traceback.format_exc())
        return False

def stop_vision_system():
    """Stop the vision system"""
    global vision_active, vision_thread, vision_system
    
    if not vision_active:
        return
    
    vision_active = False
    
    # Stop the face recognition thread
    stop_face_recognition_thread()
    
    # Properly release the camera if it exists
    if vision_system and vision_system.cap and vision_system.cap.isOpened():
        vision_system.cap.release()
    
    if vision_thread:
        vision_thread.join(timeout=1.0)
        vision_thread = None
    
    vision_system = None

def get_current_detections():
    """Get the current object detections"""
    global detected_objects
    
    with frame_lock:
        return detected_objects.copy()

def get_current_frame():
    """Get the current processed frame"""
    global current_frame
    
    with frame_lock:
        if current_frame is None:
            return None
        return current_frame.copy()

def get_objects_by_class(class_name):
    """Get detected objects of a specific class"""
    with frame_lock:
        return [obj for obj in detected_objects if obj['label'].lower() == class_name.lower()]

def is_vision_active():
    """Check if the vision system is active"""
    global vision_active
    return vision_active

def initialize_vision(show_window=False):
    """
    Initialize the vision system.
    
    This function initializes the vision system, which includes:
    1. Setting up the camera
    2. Loading the YOLO model
    3. Starting the vision processing thread
    
    Args:
        show_window (bool): Whether to display the camera feed window
        
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    global vision_active
    
    if vision_active:
        logging.info("Vision system is already active")
        return True
    
    try:
        # Initialize face recognition system
        FaceRecognition.initialize()
        
        # Adjust face recognition parameters for better reliability
        # Process embeddings more frequently (every 3 frames instead of 5)
        FaceRecognition.set_embedding_frame_interval(3)
        
        # Lower the similarity threshold slightly for more lenient matching (0.6 -> 0.55)
        # This is safe to do because we're using multiple verification steps
        FaceRecognition.similarity_threshold = 0.55
        
        # Start the vision system
        success = start_vision_system(camera_index=0, show_window=show_window)
        
        if success:
            logging.info("Vision system initialized successfully")
        else:
            logging.error("Failed to initialize vision system")
            
        return success
    except Exception as e:
        logging.error(f"Error in vision initialization: {str(e)}")
        logging.error(traceback.format_exc())
        return False

def identify_face(name):
    """
    Identify a face in the current frame and associate it with a name.
    
    This function takes the current frame from the camera, identifies the most prominent
    face in the frame, and associates it with the provided name. The face information
    is stored in a SQLite database for future recognition using facial embeddings.
    
    Multiple identifications of the same person DO improve accuracy because:
    1. Each new identification updates the face embedding in the database
    2. More samples provide better recognition across different lighting and angles
    3. The system uses the most recent identification for matching
    4. The database keeps high-quality embeddings that help with recognition
    
    Threading note: This function creates its own database connection to avoid SQLite
    threading issues, as SQLite connections can only be used in the thread they were created in.
    
    Args:
        name (str): Name to assign to the detected face
        
    Returns:
        str: Status message indicating success or failure
    """
    global current_frame, detected_objects, vision_system, known_faces
    
    # Basic input validation
    name = name.strip()
    if not name:
        return "Please provide a valid name for identification."
    
    # Check if vision system is active
    if not vision_active or current_frame is None:
        return "Vision system is not active. Please enable vision first."
    
    # Safely get a copy of the current frame and find faces
    # Using frame_lock to prevent concurrent access issues
    with frame_lock:
        if current_frame is None:
            return "No camera feed available."
        frame_copy = current_frame.copy()
        # Filter for objects labeled as "Human face" or already identified faces
        face_objects = [
            obj for obj in detected_objects 
            if obj['label'] == 'Human face' or "Human face" in obj['label']
        ]
    
    # Check if any faces were detected
    if not face_objects:
        return "No faces detected in the current frame."
    
    try:
        # Find the largest face in the frame (assumed to be the most prominent/closest)
        largest_face = max(face_objects, key=lambda x: (x['xyxy'][2] - x['xyxy'][0]) * (x['xyxy'][3] - x['xyxy'][1]))
        
        # Store coordinates for later comparison
        largest_face_coords = largest_face['xyxy']
        
        # Extract bounding box coordinates and ensure they are integers
        x1, y1, x2, y2 = map(int, largest_face_coords)
        
        # Validate coordinates to prevent errors
        height, width = frame_copy.shape[:2]
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(x1+1, min(x2, width))
        y2 = max(y1+1, min(y2, height))
        
        # Check if the box is valid and large enough
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            return "Face detection is too small for reliable recognition."
        
        # Crop the face from the frame
        face_image = frame_copy[y1:y2, x1:x2].copy()
        
        # Verify the face image is valid
        if face_image.size == 0:
            return "Error: Extracted face image is empty."
        
        # Flag to track if embedding was successfully generated
        embedding_success = False
        
        # Update face record with embedding
        if vision_system:
            # Use the vision system's face verification helper
            success, _, _ = vision_system._verify_face_with_embedding(face_image)
            embedding_success = success
            
            if not success:
                # This is not a critical error - we'll still try to save the face image
                logging.info(f"Could not generate facial embedding for {name}, but will try to save face image.")
        
        # Update face record with embedding
        success = FaceRecognition.update_face_record(name, face_image, (x1, y1, x2, y2))
        
        if success:
            embedding_success = True
        elif not embedding_success:
            # This is not a critical error - the system still saved the face image
            logging.info(f"Could not generate facial embedding for {name}, but face image was saved.")
            # We'll still consider this a success since the image was saved
            success = True
        
        # Update in-memory cache for fast recognition
        known_faces[name] = {
            'last_seen': time.time(),
            'bbox': (x1, y1, x2, y2)  # Include the bbox in the known_faces entry
        }
        
        # Get the face embedding from the database to store in our cache
        try:
            db = FaceDB.FaceDatabase()
            face_data = db.get_face_by_name(name)
            db.close()
            
            if face_data and 'embedding' in face_data and face_data['embedding'] is not None:
                # Store in our cache
                known_faces[name]['embedding'] = face_data['embedding']
                embedding_success = True
        except Exception as e:
            logging.error(f"Error retrieving face embedding: {str(e)}")
        
        # Update the current detections with the new name
        with frame_lock:
            updated_any = False
            for i, obj in enumerate(detected_objects):
                # Compare the bounding box coordinates with a small tolerance instead of direct object comparison
                obj_coords = obj.get('xyxy', (0, 0, 0, 0))
                
                # Check if the bounding boxes are very close or identical
                coords_match = (
                    abs(obj_coords[0] - largest_face_coords[0]) < 5 and
                    abs(obj_coords[1] - largest_face_coords[1]) < 5 and
                    abs(obj_coords[2] - largest_face_coords[2]) < 5 and
                    abs(obj_coords[3] - largest_face_coords[3]) < 5
                )
                
                if coords_match:
                    # Update this specific detection with the new name
                    logging.info(f"Updating detection at index {i} with name: {name}")
                    detected_objects[i]['label'] = name
                    detected_objects[i]['status'] = "IDENTIFIED"
                    if vision_system:
                        # Convert success color to BGR
                        success_hex = vision_system.jarvis_colors['success']
                        r = int(success_hex[1:3], 16)
                        g = int(success_hex[3:5], 16)
                        b = int(success_hex[5:7], 16)
                        detected_objects[i]['status_color'] = (b, g, r)
                    updated_any = True
                    break
            
            # Log whether we updated any detection
            if updated_any:
                logging.info(f"Successfully updated detection with name: {name}")
            else:
                logging.warning(f"Failed to find matching detection to update for name: {name}")
                # As a fallback, force an update to the largest face object
                for i, obj in enumerate(detected_objects):
                    if obj.get('label') == "Human face":
                        detected_objects[i]['label'] = name
                        detected_objects[i]['status'] = "IDENTIFIED"
                        if vision_system:
                            # Convert success color to BGR
                            success_hex = vision_system.jarvis_colors['success']
                            r = int(success_hex[1:3], 16)
                            g = int(success_hex[3:5], 16)
                            b = int(success_hex[5:7], 16)
                            detected_objects[i]['status_color'] = (b, g, r)
                        logging.info(f"Applied fallback update to a face detection with name: {name}")
                        break
        
        # Generate success message
        if embedding_success:
            return f"Successfully identified {name} with facial embedding."
        else:
            return f"Identified {name}, but could not generate facial embedding."
    
    except Exception as e:
        logging.error(f"Error in face identification: {str(e)}")
        logging.error(traceback.format_exc())
        return f"Error in face identification: {str(e)}"

def list_known_faces():
    """
    List all faces that have been identified and stored in the database.
    
    This function retrieves all stored face records from the SQLite database,
    groups them by name, and returns a formatted string with each person's name
    and when they were last seen. It creates its own database connection to avoid 
    threading issues with SQLite.
    
    Returns:
        str: Formatted list of known faces or an error message
    """
    try:
        # Create a new thread-safe database connection
        temp_db = FaceDB.FaceDatabase()
        
        # Retrieve all face records
        all_faces = temp_db.get_all_faces()
        
        # Close the connection as soon as we have the data
        temp_db.close()
        
        # Handle the case where no faces have been identified
        if not all_faces:
            return "No faces have been identified yet."
        
        # Group faces by person's name (a person may have multiple face records)
        faces_by_name = {}
        for face in all_faces:
            name = face['name']
            if name not in faces_by_name:
                faces_by_name[name] = []
            faces_by_name[name].append(face)
        
        # Format the response message
        response = "Known faces:"
        
        # Add each person with their last seen timestamp
        for name, faces in faces_by_name.items():
            # Get the most recent face record for this person
            most_recent = max(faces, key=lambda x: x['last_seen'])
            
            # Format the timestamp nicely
            last_seen_time = datetime.fromtimestamp(most_recent['last_seen']).strftime('%Y-%m-%d %H:%M:%S')
            
            # Add to the response
            response += f"\n- {name} (last seen: {last_seen_time})"
        
        return response
    except Exception as e:
        return f"Error retrieving known faces: {str(e)}"

def load_known_faces():
    """
    Load known faces from the database into memory.
    
    This function retrieves all face records from the database and
    stores them in memory for faster recognition.
    
    Returns:
        dict: Dictionary of known faces by name
    """
    global known_faces
    
    # Clear existing cache
    known_faces = {}
    
    try:
        logging.info("Loading known faces from database...")
        # Create database connection
        db = FaceDB.FaceDatabase()
        
        # Get all faces from the database
        faces = db.get_all_faces()
        
        # Close database connection
        db.close()
        
        # Group faces by name (keep only the most recent for each person)
        seen_names = set()
        for face in faces:
            name = face['name']
            if name not in seen_names:
                seen_names.add(name)
                # Ensure bbox exists in the returned data
                if 'bbox' not in face:
                    logging.warning(f"Face record for {name} is missing bbox - skipping")
                    continue
                    
                known_faces[name] = {
                    'id': face['id'],
                    'bbox': face['bbox'],
                    'last_seen': face['last_seen']
                }
                # Add embedding if available
                if 'embedding' in face and face['embedding'] is not None:
                    known_faces[name]['embedding'] = face['embedding']
        
        logging.info(f"Loaded {len(known_faces)} known faces from database")
        return known_faces
    
    except Exception as e:
        logging.error(f"Error loading known faces: {str(e)}")
        logging.error(traceback.format_exc())
        return {}

def _face_recognition_worker():
    """Background worker thread for face recognition processing"""
    global face_recognition_active, detected_objects
    
    logging.info("Face recognition worker thread started")
    
    while face_recognition_active:
        try:
            # Wait for a face to process, with a timeout to allow checking if we should exit
            try:
                # Get item from queue with a 0.5 second timeout
                item = face_queue.get(timeout=0.5)
            except queue.Empty:
                # No item in the queue, just continue the loop
                continue
                
            # Unpack the item
            face_image, face_bbox, current_label = item
            
            # Skip if face is invalid
            if face_image is None or face_image.size == 0:
                face_queue.task_done()
                continue
                
            # Process the face in the background thread
            # Format the image for face recognition
            is_valid, formatted_face = FaceRecognition.test_image_format(face_image)
            if not is_valid or formatted_face is None:
                face_queue.task_done()
                continue
                
            # Try to identify the face
            identified_name, similarity = FaceRecognition.identify_face_by_embedding(formatted_face)
            
            # If identification was successful and passed the threshold
            if identified_name and similarity >= FaceRecognition.similarity_threshold:
                # Update current detections with the new identification
                with frame_lock:
                    # Only update if we have a valid bbox
                    if face_bbox:
                        x1, y1, x2, y2 = face_bbox
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # Update any matching faces in the current detections
                        for i, obj in enumerate(detected_objects):
                            if "Human face" in obj.get('label', ''):
                                # Get detection box
                                obj_x1, obj_y1, obj_x2, obj_y2 = obj.get('xyxy', (0, 0, 0, 0))
                                obj_center_x = (obj_x1 + obj_x2) / 2
                                obj_center_y = (obj_y1 + obj_y2) / 2
                                
                                # Calculate distance between centers
                                distance = math.sqrt(
                                    (center_x - obj_center_x) ** 2 + 
                                    (center_y - obj_center_y) ** 2
                                )
                                
                                # If centers are close, update this detection
                                if distance < 50:  # 50 pixel threshold
                                    detected_objects[i]['label'] = identified_name
                                    detected_objects[i]['status'] = "IDENTIFIED"
                                    if vision_system:
                                        # Convert success color to BGR
                                        success_hex = vision_system.jarvis_colors['success']
                                        r = int(success_hex[1:3], 16)
                                        g = int(success_hex[3:5], 16)
                                        b = int(success_hex[5:7], 16)
                                        detected_objects[i]['status_color'] = (b, g, r)
                                        logging.info(f"Background recognition identified {identified_name}")
            
            # Mark task as done
            face_queue.task_done()
            
        except Exception as e:
            logging.error(f"Error in face recognition worker: {str(e)}")
            traceback.print_exc()
            # Continue the loop even if there was an error
            continue
    
    logging.info("Face recognition worker thread stopped")

def start_face_recognition_thread():
    """Start the background face recognition thread"""
    global face_recognition_thread, face_recognition_active
    
    if face_recognition_thread is not None and face_recognition_thread.is_alive():
        return  # Thread already running
        
    face_recognition_active = True
    face_recognition_thread = threading.Thread(target=_face_recognition_worker)
    face_recognition_thread.daemon = True
    face_recognition_thread.start()
    
def stop_face_recognition_thread():
    """Stop the background face recognition thread"""
    global face_recognition_active
    
    face_recognition_active = False
    
    # Wait for the thread to finish
    if face_recognition_thread is not None:
        face_recognition_thread.join(timeout=2.0)

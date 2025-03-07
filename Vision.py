"""
FRED Vision System - Object Detection and Face Recognition

This module provides a computer vision system for FRED using YOLOv8 for object detection,
OpenCV for tracking, and a custom face recognition system.

Key features:
- Real-time object detection using YOLOv8
- Object tracking between frames
- Face detection and recognition
- Face identification with custom naming
- Persistent storage of identified faces

Face Recognition System:
The face recognition component allows identifying people in the camera view.
When a human face is detected, you can:
1. Identify it with a name using the identify_face() function or "/identify [name]" command
2. View all identified faces using list_known_faces() function or "/faces" command
3. Faces are stored in a SQLite database and automatically recognized in future frames

Threading Note:
This system runs in multiple threads. The main vision processing runs in its own thread,
while commands from Chat.py run in the main thread. SQLite connections in Python cannot
be shared between threads, so temporary connections are created as needed.
"""

import cv2
import numpy as np
import threading
import time
from pathlib import Path
import sys
import os
import torch
from ultralytics import YOLO
import math
import uuid  # For generating unique tracker IDs
import FaceDB
from datetime import datetime

# Global variables for vision system
vision_active = False
vision_thread = None
vision_system = None  # Global variable to track the VisionSystem instance
detected_objects = []
confidence_threshold = 0.5  # Display threshold
detection_threshold = 0.3  # Tracking initialization threshold
current_frame = None
frame_lock = threading.Lock()
max_detected_objects = 3  # Maximum number of objects to detect and track

# Base directory for models
MODEL_DIR = Path("./models/vision")

# Runtime cache for known faces
known_faces = {}  # Cache for known faces during runtime

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
        
        # Tracking parameters
        self.active_trackers = {}
        self.max_tracker_age = 90  # Maximum frames to track without re-detection
        self.max_objects = max_detected_objects  # Use the global max_detected_objects value
        
        # Initialize detection model
        self._load_model()
    
    def _load_model(self):
        """Load the YOLOv8s object detection model with COCO dataset (80 classes)"""
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
                print(f"YOLOv8s model not found at {local_model_path}")
                print(f"Downloading YOLOv8s model...")
                # The YOLO constructor will download the model if not found
                self.model = YOLO(model_filename, verbose=False)
                # Save the model to our models directory
                print(f"Saving model to {local_model_path}")
                if not local_model_path.parent.exists():
                    local_model_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(local_model_path)
            else:
                print(f"Loading YOLOv8s model from {local_model_path}")
                self.model = YOLO(local_model_path, verbose=False)
            
            # Generate colors for class visualization (80 classes for COCO dataset)
            self.colors = np.random.uniform(0, 255, size=(80, 3))
            
            print("YOLOv8s model loaded successfully")
            
        except Exception as e:
            print(f"Error loading vision model: {str(e)}")
            raise
    
    def start(self, camera_index=0, show_window=False):
        """Start the vision system using the specified camera"""
        global vision_active
        
        if vision_active:
            return
        
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            return False
        
        self.is_running = True
        self.show_window = show_window
        vision_active = True
        
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
        Detect objects in the given frame using YOLOv8 and update trackers.
        
        This method processes a video frame through the YOLOv8 model to detect objects,
        maintains tracking of objects between frames, and handles the recognition
        of previously identified faces.
        
        Args:
            frame (numpy.ndarray): Video frame to process
            
        Returns:
            list: Detected and tracked objects with their metadata
        """
        # Basic validation
        if frame is None or self.model is None:
            return []
        
        # Create a copy for tracker updates to avoid modifying the original
        frame_for_trackers = frame.copy()
        
        # First update existing tracked objects from previous frames
        tracked_objects = self._update_trackers(frame_for_trackers)
        
        # Run YOLOv8 inference with the detection threshold
        # (lower than display threshold to allow tracking borderline detections)
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
                
                # Convert to center format with width and height for easier processing
                x = int((x1 + x2) / 2)  # center x
                y = int((y1 + y2) / 2)  # center y
                w = int(x2 - x1)        # width
                h = int(y2 - y1)        # height
                
                # Get class name from the model's class list
                label = result.names[class_id]
                
                # ===== FACE RECOGNITION LOGIC =====
                # If a human face is detected, check if it matches a known person
                if label == "Human face":
                    # Iterate through our cache of known faces to find a match
                    for name, face_data in known_faces.items():
                        # Only consider recently seen faces (within the last hour)
                        if time.time() - face_data['last_seen'] < 3600:
                            # Get the stored bounding box of the known face
                            old_x1, old_y1, old_x2, old_y2 = face_data['bbox']
                            old_w = old_x2 - old_x1
                            old_h = old_y2 - old_y1
                            
                            # Simple size-based comparison
                            # In a production system, this would use face embeddings for more accurate recognition
                            size_diff = abs((w * h) - (old_w * old_h)) / (old_w * old_h)
                            
                            # If the size is roughly similar, consider it the same person
                            # This is a simplified approach - real face recognition would be more sophisticated
                            if size_diff < 0.5:
                                # Replace the generic "Human face" label with the person's name
                                label = name
                                # Update the last seen timestamp
                                face_data['last_seen'] = time.time()
                                break
                # ===== END FACE RECOGNITION LOGIC =====
                
                # Choose visualization color based on class
                jarvis_color_keys = list(self.jarvis_colors.keys())
                jarvis_color_hex = self.jarvis_colors[jarvis_color_keys[class_id % len(jarvis_color_keys)]]
                
                # Convert hex color to BGR for OpenCV
                r = int(jarvis_color_hex[1:3], 16)
                g = int(jarvis_color_hex[3:5], 16)
                b = int(jarvis_color_hex[5:7], 16)
                jarvis_color = (b, g, r)  # OpenCV uses BGR color order
                
                # Create detection object with all metadata
                detection = {
                    'label': label,
                    'confidence': confidence,
                    'box': (x, y, w, h),
                    'xyxy': (x1, y1, x2, y2),
                    'color': jarvis_color,
                    'is_tracked': False,  # This is a fresh detection, not from a tracker
                    'tracker_id': None,   # Will be assigned if this gets tracked
                    'status': "DETECTED", # Default status for new detections
                    'status_color': self.jarvis_colors['stark_glow']  # Default status color
                }
                
                # Add to all detections list
                all_detections.append(detection)
        
        # Sort detections by confidence (highest first)
        all_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Get the top detections based on confidence
        top_detections = all_detections[:max_detected_objects]
        
        # Process each detection to determine if it should be tracked
        detections = []
        for detection in top_detections:
            x1, y1, x2, y2 = detection['xyxy']
            confidence = detection['confidence']
            
            # Check if this detection overlaps with any existing tracker
            tracker_match = self._find_matching_tracker((x1, y1, x2, y2))
            
            # Handle the detection based on confidence and tracker status
            if confidence < confidence_threshold and tracker_match is None:
                # For low confidence detections with no existing tracker, create a new tracker
                tracker_id = self._create_tracker(frame, (x1, y1, x2, y2), detection['label'], detection['color'])
                detection['tracker_id'] = tracker_id
                # Don't add low-confidence detections to the output, they'll be tracked instead
                continue
            elif tracker_match is not None:
                # If there's a matching tracker, update it with the new detection
                self._update_tracker_data(tracker_match, detection)
                # Skip adding this detection since we'll use the tracker's data
                continue
            
            # For high confidence detections, add them to the output list
            if confidence >= confidence_threshold:
                detections.append(detection)
        
        # Limit tracked objects to the maximum allowed
        tracked_objects = tracked_objects[:max(0, max_detected_objects - len(detections))]
        
        # Combine detected objects with tracked objects (limited to max_detected_objects)
        all_objects = detections + tracked_objects
        if len(all_objects) > max_detected_objects:
            all_objects = all_objects[:max_detected_objects]
        
        # Draw objects on frame if window is shown
        if self.show_window:
            self._draw_objects(frame, all_objects)
        
        return all_objects
    
    def _create_tracker(self, frame, bbox, label, color):
        """Initialize a new object tracker for the given bounding box"""
        x1, y1, x2, y2 = bbox
        
        # Generate a unique ID for this tracker
        tracker_id = str(uuid.uuid4())
        
        # Initialize an OpenCV tracker (CSRT offers good accuracy with reasonable speed)
        tracker = cv2.TrackerCSRT_create()
        bbox_for_tracker = (x1, y1, x2 - x1, y2 - y1)  # Convert to (x, y, w, h) format
        tracker.init(frame, bbox_for_tracker)
        
        # Store the tracker with metadata
        self.active_trackers[tracker_id] = {
            'tracker': tracker,
            'label': label,
            'color': color,
            'bbox': bbox,
            'age': 0,
            'quality': 1.0,  # Initial tracking quality
            'last_position': bbox,  # Store the last position to detect stagnation
            'stagnant_count': 0,    # Counter for frames where position hasn't changed
            'last_update_time': time.time(),  # Timestamp of last successful update
            'positions': [],  # Store recent positions with timestamps for velocity calculation
            'velocity': None  # Store calculated velocity in pixels per second
        }
        
        return tracker_id
    
    def _update_trackers(self, frame):
        """Update all active trackers and return currently tracked objects"""
        tracked_objects = []
        trackers_to_remove = []
        
        current_time = time.time()
        
        # Process each active tracker
        for tracker_id, tracker_data in self.active_trackers.items():
            tracker = tracker_data['tracker']
            tracker_data['age'] += 1
            
            # Update tracker with new frame
            success, bbox = tracker.update(frame)
            
            # Get the last known position for stagnation check
            last_x1, last_y1, last_x2, last_y2 = tracker_data['last_position']
            
            # If tracking was successful and hasn't exceeded max age
            if success and tracker_data['age'] < self.max_tracker_age:
                x, y, w, h = [int(v) for v in bbox]
                x1, y1, x2, y2 = x, y, x + w, y + h
                
                # Check for invalid bounding box (sometimes trackers report success with invalid boxes)
                if w <= 0 or h <= 0:
                    trackers_to_remove.append(tracker_id)
                    continue
                
                # Check if object is still within frame boundaries
                frame_h, frame_w = frame.shape[:2]
                if x1 < 0 or y1 < 0 or x2 >= frame_w or y2 >= frame_h:
                    # Object is at least partially outside frame - give it a few frames to return
                    tracker_data['boundary_violations'] = tracker_data.get('boundary_violations', 0) + 1
                    if tracker_data['boundary_violations'] > 5:  # Remove after 5 consecutive boundary violations
                        trackers_to_remove.append(tracker_id)
                        continue
                else:
                    # Reset boundary violations counter
                    tracker_data['boundary_violations'] = 0
                
                # Check for stagnation - if the object hasn't moved significantly for several frames
                position_change = abs(x1 - last_x1) + abs(y1 - last_y1) + abs(x2 - last_x2) + abs(y2 - last_y2)
                if position_change < 4:  # Less than 1 pixel change per corner
                    tracker_data['stagnant_count'] += 1
                    if tracker_data['stagnant_count'] > 15:  # If stagnant for half a second (15 frames at 30fps)
                        # Try to verify the tracker is still valid using template matching
                        if not self._verify_tracker(frame, (x1, y1, x2, y2), tracker_data):
                            trackers_to_remove.append(tracker_id)
                            continue
                else:
                    # Reset stagnation counter
                    tracker_data['stagnant_count'] = 0
                
                # Check time since last successful quality update
                time_since_update = current_time - tracker_data.get('last_update_time', 0)
                if time_since_update > 2.0:  # If more than 2 seconds since last good update
                    # Try to verify the tracker is still valid
                    if not self._verify_tracker(frame, (x1, y1, x2, y2), tracker_data):
                        trackers_to_remove.append(tracker_id)
                        continue
                
                # Update the tracker's data
                tracker_data['bbox'] = (x1, y1, x2, y2)
                tracker_data['last_position'] = (x1, y1, x2, y2)
                tracker_data['last_update_time'] = current_time
                
                # Store position with timestamp for velocity calculation
                # Calculate center point of the object
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                tracker_data['positions'].append({
                    'pos': (center_x, center_y),
                    'time': current_time
                })
                
                # Keep only the last 10 positions
                if len(tracker_data['positions']) > 10:
                    tracker_data['positions'] = tracker_data['positions'][-10:]
                
                # Calculate velocity if we have enough data
                if len(tracker_data['positions']) >= 3:
                    # Calculate velocity based on recent positions
                    recent_pos = tracker_data['positions'][-3:]
                    dx = recent_pos[-1]['pos'][0] - recent_pos[0]['pos'][0]  # Change in x
                    dy = recent_pos[-1]['pos'][1] - recent_pos[0]['pos'][1]  # Change in y
                    dt = recent_pos[-1]['time'] - recent_pos[0]['time']      # Change in time
                    
                    if dt > 0:
                        # Calculate velocity magnitude in pixels per second
                        velocity_magnitude = math.sqrt(dx**2 + dy**2) / dt
                        tracker_data['velocity'] = velocity_magnitude
                
                # Create a tracked object entry
                tracked_obj = {
                    'label': tracker_data['label'],
                    'confidence': tracker_data.get('confidence', 0.0),  # Use stored confidence or default to 0
                    'box': (x + w//2, y + h//2, w, h),
                    'xyxy': (x1, y1, x2, y2),
                    'color': tracker_data['color'],
                    'is_tracked': True,
                    'tracker_id': tracker_id,
                    'velocity': tracker_data.get('velocity', None),  # Add velocity data
                    'positions': tracker_data.get('positions', [])   # Add position history
                }
                
                # Add movement status
                if tracker_data.get('velocity') is not None:
                    velocity = tracker_data['velocity']
                    if velocity < 5:
                        tracked_obj['status'] = "STATIONARY"
                        tracked_obj['status_color'] = self.jarvis_colors['success']  # Green for stationary
                    elif velocity < 50:
                        tracked_obj['status'] = "MOVING"
                        tracked_obj['status_color'] = self.jarvis_colors['accent_bright']  # Purple for normal movement
                    else:
                        tracked_obj['status'] = "FAST"
                        tracked_obj['status_color'] = self.jarvis_colors['warning']  # Amber for fast movement
                else:
                    tracked_obj['status'] = "TRACKING"
                    tracked_obj['status_color'] = self.jarvis_colors['stark_blue']  # Default color
                
                tracked_objects.append(tracked_obj)
            else:
                # If tracker update failed, remove it
                trackers_to_remove.append(tracker_id)
        
        # Remove trackers that are no longer valid
        for tracker_id in trackers_to_remove:
            self.active_trackers.pop(tracker_id, None)
        
        # Sort tracked objects by confidence (highest first)
        tracked_objects.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Limit to max_detected_objects
        return tracked_objects[:max_detected_objects]
    
    def _find_matching_tracker(self, bbox):
        """Find a tracker that overlaps with the given bounding box"""
        x1, y1, x2, y2 = bbox
        area1 = (x2 - x1) * (y2 - y1)
        
        for tracker_id, tracker_data in self.active_trackers.items():
            tx1, ty1, tx2, ty2 = tracker_data['bbox']
            
            # Calculate intersection
            ix1 = max(x1, tx1)
            iy1 = max(y1, ty1)
            ix2 = min(x2, tx2)
            iy2 = min(y2, ty2)
            
            if ix1 < ix2 and iy1 < iy2:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                area2 = (tx2 - tx1) * (ty2 - ty1)
                overlap = intersection / min(area1, area2)
                
                # If overlap is significant, consider it a match
                if overlap > 0.5:
                    return tracker_id
        
        return None
    
    def _update_tracker_data(self, tracker_id, detection):
        """Update the data for an existing tracker with new detection information"""
        if tracker_id in self.active_trackers:
            # Update the last update time
            self.active_trackers[tracker_id]['last_update_time'] = time.time()
            
            # Store the confidence value from the detection
            self.active_trackers[tracker_id]['confidence'] = detection['confidence']
            
            # Reset tracking quality checks since we have a new detection
            self.active_trackers[tracker_id]['stagnant_count'] = 0
            self.active_trackers[tracker_id]['boundary_violations'] = 0
    
    def _draw_objects(self, frame, objects):
        """Draw all objects (detected and tracked) on the frame"""
        for obj in objects:
            x1, y1, x2, y2 = obj['xyxy']
            x, y, w, h = obj['box']
            color = obj['color']
            is_tracked = obj.get('is_tracked', False)
            label = obj['label']
            
            # Create overlay for transparency
            overlay = frame.copy()
            
            # Get colors for drawing
            if isinstance(color, tuple):
                stark_blue = color
                glow_color = color  # Simplify for tracked objects
            else:
                # Convert hex colors if needed
                stark_blue_hex = self.jarvis_colors['stark_blue']
                r_blue = int(stark_blue_hex[1:3], 16)
                g_blue = int(stark_blue_hex[3:5], 16)
                b_blue = int(stark_blue_hex[5:7], 16)
                stark_blue = (b_blue, g_blue, r_blue)
                
                glow_hex = self.jarvis_colors['stark_glow']
                r_glow = int(glow_hex[1:3], 16)
                g_glow = int(glow_hex[3:5], 16)
                b_glow = int(glow_hex[5:7], 16)
                glow_color = (b_glow, g_glow, r_glow)
            
            # Draw the main circle - thicker for detected objects, thinner for tracked ones
            circle_thickness = 1 if is_tracked else 2
            radius = int(max(w, h) / 2)
            cv2.circle(overlay, (x, y), radius, stark_blue, circle_thickness)
            
            # Draw outer dashed segments for detected objects
            if not is_tracked:
                outer_radius = radius + 10
                segments = 16
                for i in range(segments):
                    if i % 2 == 0:  # Skip every other segment to create dashes
                        angle_start = i * (2 * np.pi / segments)
                        angle_end = (i + 1) * (2 * np.pi / segments)
                        
                        # Calculate start and end points of arc
                        start_x = int(x + outer_radius * np.cos(angle_start))
                        start_y = int(y + outer_radius * np.sin(angle_start))
                        end_x = int(x + outer_radius * np.cos(angle_end))
                        end_y = int(y + outer_radius * np.sin(angle_end))
                        
                        cv2.line(overlay, (start_x, start_y), (end_x, end_y), glow_color, 2)
            
            # Note: Position tracking is still maintained for velocity calculations,
            # but we no longer draw the visual trail behind tracked objects
            
            # Calculate smart data panel position
            frame_h, frame_w = frame.shape[:2]
            space_left = x1
            space_right = frame_w - x2
            space_top = y1
            space_bottom = frame_h - y2
            spaces = [space_right, space_bottom, space_left, space_top]
            max_space_idx = spaces.index(max(spaces))
            
            # Position panel based on available space
            if max_space_idx == 0:  # Right has most space
                panel_x = x2 + 10
                panel_y = max(y - 35, 30)
            elif max_space_idx == 1:  # Bottom has most space
                panel_x = max(x - 75, 10)
                panel_y = y2 + 10
            elif max_space_idx == 2:  # Left has most space
                panel_x = max(x1 - 160, 10)
                panel_y = max(y - 35, 30)
            else:  # Top has most space
                panel_x = max(x - 75, 10)
                panel_y = max(y1 - 90, 30)
            
            # Panel height depends on whether this is a tracked or detected object
            # Increase panel height to accommodate more information
            panel_height = 90 if is_tracked else 70
            
            # Create data panel background
            cv2.rectangle(overlay, 
                        (panel_x, panel_y), 
                        (panel_x + 150, panel_y + panel_height), 
                        (0, 0, 0), -1)
            
            # Add technical lines around the panel
            corner_len = 20
            # Top-left corner
            cv2.line(overlay, (panel_x, panel_y), (panel_x + corner_len, panel_y), stark_blue, 2)
            cv2.line(overlay, (panel_x, panel_y), (panel_x, panel_y + corner_len), stark_blue, 2)
            
            # Top-right corner
            cv2.line(overlay, (panel_x + 150, panel_y), (panel_x + 150 - corner_len, panel_y), stark_blue, 2)
            cv2.line(overlay, (panel_x + 150, panel_y), (panel_x + 150, panel_y + corner_len), stark_blue, 2)
            
            # Bottom-left corner
            cv2.line(overlay, (panel_x, panel_y + panel_height), (panel_x + corner_len, panel_y + panel_height), stark_blue, 2)
            cv2.line(overlay, (panel_x, panel_y + panel_height), (panel_x, panel_y + panel_height - corner_len), stark_blue, 2)
            
            # Bottom-right corner
            cv2.line(overlay, (panel_x + 150, panel_y + panel_height), (panel_x + 150 - corner_len, panel_y + panel_height), stark_blue, 2)
            cv2.line(overlay, (panel_x + 150, panel_y + panel_height), (panel_x + 150, panel_y + panel_height - corner_len), stark_blue, 2)
            
            # Add text data to panel in JARVIS style
            # Get status text and color
            if is_tracked and 'status' in obj:
                status_text = obj['status']
                status_color_hex = obj['status_color']
                r_status = int(status_color_hex[1:3], 16)
                g_status = int(status_color_hex[3:5], 16)
                b_status = int(status_color_hex[5:7], 16)
                status_color = (b_status, g_status, r_status)
            else:
                status_text = "DETECTED"
                status_color = glow_color
            
            # Target label
            cv2.putText(overlay, f"TARGET: {label.upper()}", 
                       (panel_x + 5, panel_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, glow_color, 1)
            
            # Confidence for detected objects
            line_y = panel_y + 40
            if not is_tracked:
                cv2.putText(overlay, f"CONFIDENCE: {obj['confidence']:.2f}", 
                           (panel_x + 5, line_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, glow_color, 1)
                line_y += 20
            
            # Status with dynamic color based on movement
            cv2.putText(overlay, f"STATUS: {status_text}", 
                       (panel_x + 5, line_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
            line_y += 20
            
            # Add velocity for tracked objects
            if is_tracked and 'velocity' in obj and obj['velocity'] is not None:
                velocity = obj['velocity']
                cv2.putText(overlay, f"VELOCITY: {velocity:.1f} px/s", 
                           (panel_x + 5, line_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, glow_color, 1)
            
            # Add connecting line from panel to object
            if max_space_idx == 0:  # Panel on right
                line_start = (panel_x, panel_y + panel_height//2)
                line_end = (x2, y)
            elif max_space_idx == 1:  # Panel below
                line_start = (panel_x + 75, panel_y)
                line_end = (x, y2)
            elif max_space_idx == 2:  # Panel on left
                line_start = (panel_x + 150, panel_y + panel_height//2)
                line_end = (x1, y)
            else:  # Panel above
                line_start = (panel_x + 75, panel_y + panel_height)
                line_end = (x, y1)
            
            # Draw connecting line with dot at end
            cv2.line(overlay, line_start, line_end, stark_blue, 1)
            cv2.circle(overlay, line_end, 3, glow_color, -1)
            
            # Combine overlay with original frame
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    def process_frame(self):
        """Process a single frame from the camera"""
        global current_frame, detected_objects
        
        if not self.is_running or self.cap is None:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            return
        
        # Detect objects in the frame
        detections = self.detect_objects(frame)
        
        # Update global variables with thread safety
        with frame_lock:
            current_frame = frame.copy()
            detected_objects = detections
        
        # Display frame if window is enabled
        if self.show_window:
            cv2.imshow("FRED Vision System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
    
    def _verify_tracker(self, frame, bbox, tracker_data):
        """Verifies if a tracker is still tracking a valid object using template matching"""
        x1, y1, x2, y2 = bbox
        
        # Skip verification if box is too small
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return False
            
        # Skip if the box is too large (likely an error)
        if (x2 - x1) > frame.shape[1] * 0.9 or (y2 - y1) > frame.shape[0] * 0.9:
            return False
            
        try:
            # Convert to grayscale for template matching
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Extract the ROI
            roi = frame_gray[y1:y2, x1:x2]
            
            # Check if ROI is valid
            if roi.size == 0:
                return False
                
            # Calculate histogram of the region
            hist = cv2.calcHist([roi], [0], None, [32], [0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            
            # Compare with previous histogram if it exists
            if 'hist' in tracker_data:
                prev_hist = tracker_data['hist']
                similarity = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL)
                
                # If the histograms are very different, the tracker may have drifted
                if similarity < 0.5:
                    return False
            
            # Store the new histogram
            tracker_data['hist'] = hist
            return True
            
        except Exception:
            # If any error occurs during verification, assume failure
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
        
        # Start vision processing in a separate thread
        vision_thread = threading.Thread(target=vision_loop, args=(vision_system,))
        vision_thread.daemon = True
        vision_thread.start()
        
        vision_active = True
        return True
    
    except Exception as e:
        print(f"Error starting vision system: {str(e)}")
        return False

def stop_vision_system():
    """Stop the vision system"""
    global vision_active, vision_thread, vision_system
    
    if not vision_active:
        return
    
    vision_active = False
    
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
    Initialize the FRED Vision System - call this from Chat.py
    
    This is the main entry point for starting the vision system from Chat.py.
    It handles camera initialization, loads known faces from the database,
    and starts the vision processing thread.
    
    Args:
        show_window (bool): Whether to display the camera feed window (default: False)
        
    Returns:
        bool: True if vision system initialized successfully, False otherwise
    """
    try:
        # Start the vision system
        success = start_vision_system(camera_index=0, show_window=show_window)
        
        if success:
            print("Vision system initialized successfully")
        else:
            print("Failed to initialize vision system")
            
        return success
    except Exception as e:
        print(f"Error in vision initialization: {str(e)}")
        return False

def identify_face(name):
    """
    Identify a face in the current frame and associate it with a name.
    
    This function takes the current frame from the camera, identifies the most prominent
    face in the frame, and associates it with the provided name. The face information
    is stored in a SQLite database for future recognition and the label is updated
    in real-time on the display.
    
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
        # Filter for objects labeled as "Human face"
        face_objects = [obj for obj in detected_objects if obj['label'] == 'Human face']
    
    # Check if any faces were detected
    if not face_objects:
        return "No faces detected in the current frame."
    
    try:
        # Find the largest face in the frame (assumed to be the most prominent/closest)
        largest_face = max(face_objects, key=lambda x: (x['xyxy'][2] - x['xyxy'][0]) * (x['xyxy'][3] - x['xyxy'][1]))
        
        # Extract bounding box coordinates
        x1, y1, x2, y2 = largest_face['xyxy']
        
        # Validate coordinates to prevent errors
        if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > frame_copy.shape[1] or y2 > frame_copy.shape[0]:
            return "Invalid face detection coordinates."
        
        # Crop the face from the frame
        face_image = frame_copy[y1:y2, x1:x2]
        
        # Create a new database connection for this call (thread-safe approach)
        temp_db = FaceDB.FaceDatabase()
        
        # Save face to database
        face_id = temp_db.add_face(name, face_image, (x1, y1, x2, y2))
        
        # Always close the database connection when done
        temp_db.close()
        
        # Update in-memory cache for fast recognition
        known_faces[name] = {
            'id': face_id,
            'bbox': (x1, y1, x2, y2),
            'last_seen': time.time()
        }
        
        # If the face has an active tracker, update its label immediately
        tracker_id = largest_face.get('tracker_id')
        if tracker_id and vision_system and vision_system.active_trackers and tracker_id in vision_system.active_trackers:
            vision_system.active_trackers[tracker_id]['label'] = name
        
        return f"Successfully identified face as {name}."
    except Exception as e:
        return f"Error identifying face: {str(e)}"

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
    Load all known faces from the database into the runtime cache.
    
    This function is called when the vision system starts to ensure that
    previously identified faces can be recognized without requiring
    re-identification.
    
    Returns:
        int: Number of faces loaded from the database
    """
    global known_faces
    
    try:
        # Create a new thread-safe database connection
        temp_db = FaceDB.FaceDatabase()
        
        # Retrieve all face records
        all_faces = temp_db.get_all_faces()
        
        # Close the connection as soon as we have the data
        temp_db.close()
        
        # If no faces exist in the database
        if not all_faces:
            return 0
        
        # Group faces by person and keep only the most recent one for each person
        faces_by_name = {}
        for face in all_faces:
            name = face['name']
            if name not in faces_by_name or face['last_seen'] > faces_by_name[name]['last_seen']:
                faces_by_name[name] = face
        
        # Add each face to the runtime cache
        count = 0
        for name, face in faces_by_name.items():
            known_faces[name] = {
                'id': face['id'],
                'bbox': face['bbox'],
                'last_seen': face['last_seen']
            }
            count += 1
        
        print(f"Loaded {count} known faces from database")
        return count
    except Exception as e:
        print(f"Error loading known faces: {str(e)}")
        return 0

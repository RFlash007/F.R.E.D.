# F.R.E.D. - Frankly Rude Educated Droid

## Overview
F.R.E.D. is an advanced AI assistant combining conversational intelligence with computer vision capabilities. Built with Python, it integrates multiple AI models and systems to provide comprehensive assistance.

## Core Features

### Vision System
- **Object Detection**: YOLOv8s model identifies 600+ objects
- **Face Recognition**: Custom facial embedding system with SQLite storage
- **Object Tracking**: Persistent tracking with unique IDs
- **Real-time Processing**: Continuous camera feed analysis
- **Two-stage Architecture**:
  1. Objective Description Layer (neutral descriptions)
  2. Personality Layer (FRED's characteristic tone)

### Memory Systems
- **Semantic Memory**: Stores facts and information
- **Episodic Memory**: Remembers conversations and interactions
- **Dreaming**: Develops insights about user preferences
- **Face Database**: Persistent storage of facial embeddings and metadata

### Daily Operations
- Automatic morning briefing
- Task management with auto-cleanup
- News summarization
- System monitoring

## Technical Architecture

### Core Modules

1. **Vision System (Vision.py)**
   - YOLOv8 integration for object detection
   - Face detection and cropping
   - Object tracking with history
   - Camera feed processing
   - Detection threshold management
   - Tracker cleanup and maintenance

2. **Face Recognition (FaceRecognition.py)**
   - Facial embedding extraction
   - Face matching with cosine similarity
   - Embedding cache system
   - Face record updates
   - Similarity threshold management

3. **Database Management (FaceDB.py)**
   - SQLite database operations
   - Face record storage and retrieval
   - Embedding serialization
   - Bounding box storage
   - Timestamp management
   - Image file storage

4. **Chat Interface (Chat.py)**
   - Command processing
   - Face identification commands
   - Response generation
   - Error handling
   - User interaction management

5. **Tool Integration (Tools.py)**
   - Vision system access
   - Object detection summaries
   - Detection result formatting
   - Error handling and fallbacks
   - Prominent object identification

### Key Data Structures

- **Face Records**
  - name: String
  - bbox: Tuple (x1, y1, x2, y2)
  - last_seen: Timestamp
  - embedding: Numpy array
  - image_path: String

- **Object Detections**
  - label: String
  - confidence: Float
  - bbox: Tuple (x1, y1, x2, y2)
  - tracker_id: UUID

- **Trackers**
  - label: String
  - bbox: Tuple (x1, y1, x2, y2)
  - last_seen: Timestamp
  - history: Deque of positions
  - frames_since_detection: Int

### Threading Model

- **Main Thread**
  - Handles user interaction
  - Processes commands
  - Manages UI updates

- **Vision Thread**
  - Continuous camera feed processing
  - Object detection and tracking
  - Face recognition
  - Frame analysis

- **Database Access**
  - Thread-safe connections
  - Short-lived connections for operations
  - Connection pooling for efficiency

### Configuration Parameters

- **Detection Thresholds**
  - confidence_threshold: 0.5
  - detection_threshold: 0.3
  - similarity_threshold: 0.6

- **Tracking Parameters**
  - max_tracker_age: 60 frames
  - history_length: 30 frames
  - max_detected_objects: 3

- **Embedding Parameters**
  - embedding_frame_interval: 5
  - min_embedding_interval: 0.2 seconds

## Usage Examples

### Vision Commands
- Identify face: "/identify [name]"
- List known faces: "/faces"
- Object detection: "What do you see?"

### Memory Management
- Query memories: "What do you remember about..."
- Forget conversations: "Forget our discussion about..."
- End sessions: "Goodbye"

### Task Management
- Add task: "Add task [description]"
- List tasks: "Show my tasks"
- Delete task: "Delete task [id]"

## Future Development Roadmap

### Vision System Enhancements
- Multi-camera support
- Advanced motion analysis
- Gesture recognition
- Enhanced text recognition
- Document scanning capabilities

### Memory System Improvements
- Contextual memory linking
- Memory prioritization
- Automatic memory organization
- Memory visualization tools

### Integration Features
- Smart home device control
- Calendar integration
- Email management
- Document processing

### Performance Optimizations
- GPU acceleration
- Model quantization
- Asynchronous processing
- Memory-efficient caching

## Technical Specifications

### Dependencies
- Python 3.8+
- PyTorch
- OpenCV
- SQLite3
- NumPy
- face_recognition library

### System Requirements
- Minimum 4GB RAM
- Webcam support
- Python environment
- CUDA support (optional for GPU acceleration)

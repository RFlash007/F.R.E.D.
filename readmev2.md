@README.md

# F.R.E.D. - Frankly Rude Educated Droid (Technical Reference and Implementation)

This document builds on `README.md` and provides an in-depth exploration of F.R.E.D.'s architecture, core modules, underlying technologies, and integration points. It is intended as context for an AI research assistant to guide enhancements and optimizations.

## Table of Contents

1. Introduction
2. System Architecture
3. Core Modules
   - Chat Engine (`Chat.py`)
   - User Interface (`ChatUI.py`)
   - Memory Systems
     - Semantic Memory (`Semantic.py`)
     - Episodic Memory (`Episodic.py`)
     - Dreaming Module (`Dreaming.py`)
   - Memory Consolidation Script (`update_memory.py`)
   - Vision System (`Vision.py`)
   - Voice Synthesis (`Voice.py`)
4. Integration and APIs
5. Data Storage and Formats
6. Threading and Concurrency
7. Configuration and Parameters
8. External Dependencies and Requirements
9. Extension Points and Roadmap

---

## 1. Introduction

F.R.E.D. is a Python-based AI assistant combining conversational intelligence with computer vision and memory systems. Powered by advanced LLMs (via Ollama), it supports real-time camera processing, face recognition, and multi-layered memory (semantic, episodic, and dreaming). This document details the internal implementation, highlighting how modules interact, what third-party libraries and APIs are employed, and where to extend or optimize.

## 2. System Architecture

F.R.E.D. follows a modular design with three main subsystems:

- **Chat Engine**: Handles user input, routes through LLMs, integrates memory/tool calls, and produces responses.
- **User Interface**: A rich desktop UI built with Tkinter, offering holographic visualizations, arc-reactor animations, and memory management panels.
- **Perception & Memory**: Vision pipeline for object/face detection and layered memory modules for knowledge retention and insight generation.

These subsystems communicate via Python function calls, shared queues, and JSON/SQLite persistence.

## 3. Core Modules

### 3.1 Chat Engine (Chat.py)

Responsibilities:
- **User Message Processing**: Normalizes commands, routes vision or identification triggers, or forwards to LLM.
- **Memory Recall**: Calls `Episodic.recall_episodic`, `Semantic.recall_semantic`, `Dreaming.recall_dreams` to retrieve relevant memories.
- **Emotion Detection**: Integrates with `EmotionIntegration` to tag user sentiment.
- **Model Invocation**: Uses Ollama Python API to chat with `FRED_14b` (Qwen2.5 backbone) for generation, streaming, and tool calls.
- **Tool Schema Definitions**: Exposes functions such as `get_system_status`, `search_web_information`, and memory/vision tools to augment LLM responses.
- **Conversation Management**: Maintains a short conversation history, summarizes on `goodbye`, and triggers `save_conversation` and `shutdown_app`.
- **Threading**: Orchestrates voice queue, UI callbacks, and vision initialization on startup.

Key integrations:
- `ollama.chat` and `ollama.embeddings` for LLM operations.
- `Voice.piper_speak` for TTS output.
- `Vision.initialize_vision` and `Vision.identify_face` for camera-based operations.

### 3.2 User Interface (ChatUI.py)

Built with Python Tkinter, this module delivers a clean, modern UI:

- **Arc Reactor Visualization**: Multi-layered pulsing circles and data points animated via Canvas.
- **Conversation Panel**: Scrollable text with custom tags, live typing animation, code block styling.
- **Holographic Memory Tabs**: Notebook tabs for Semantic, Episodic, and Dreams memories, each with search, filter, edit, delete, import/export controls.
- **Command Input**: Floating entry field styled with neon glow and Process button.
- **System Metrics Panel**: Displays OS, hostname, Python version, memory, etc., styled as tech dashboard.
- **Radial Menu**: Contextual buttons around arc reactor for quick tab access.

Notable patterns:
- **Thread-Safe UI Updates** via `msg_queue` and `root.after`.
- **Form Modals** for memory editing (`EditMemoryDialog`), dynamically generating fields based on memory type.
- **Resource Cleanup**: Unbinding events and destroying dialogs to prevent memory leaks.

### 3.3 Memory Systems

#### 3.3.1 Semantic Memory (Semantic.py)

- **Data Model**: `Fact` (Pydantic v2) with `category` and `content`.
- **Storage**: JSON Lines in `Semantic.json`.
- **Embeddings Cache**: Stored in `cache/semantic_embeddings.pt` via PyTorch tensor.
- **Functions**:
  - `create_semantic`: Prompts LLM to extract facts, appends to file.
  - `recall_semantic`: Retrieves top-k facts via cosine similarity embeddings.
  - `update_semantic`: Refreshes accessed facts with new context.
  - `remove_duplicate_semantic` & `consolidate_semantic`: Deduplication and merging similar facts beyond a threshold.

Integration:
- Ollama embeddings with `nomic-embed-text` model.
- Batch processing for efficient embedding generation.

#### 3.3.2 Episodic Memory (Episodic.py)

- **Data Model**: `Episode` (Pydantic v2) with timestamp, context tags, summary, what worked/avoid/learned.
- **Storage**: JSON Lines in `Episodic.json`.
- **Embeddings Cache**: `cache/episodic_embeddings.pt`.
- **Functions**:
  - `create_episodic`: Prompt LLM to produce structured episodes from conversation.
  - `recall_episodic`: Retrieve most relevant episodes via semantic similarity.
  - `update_episodic`, `remove_duplicate_episodic`, `consolidate_episodic`.

#### 3.3.3 Dreaming Module (Dreaming.py)

- **Data Model**: `Dream` with `insight_type`, `content`, `source`.
- **Storage**: JSON Lines in `Dreaming.json`, synthetic conversations in `synthetic_conversations/`.
- **Functions**:
  - `create_dream`: Extracts insights from both real and synthetic conversations.
  - `generate_synthetic_conversation`: Uses LLM to craft new dialogues for deeper pattern discovery.
  - `recall_dreams`: Fetch top insight for given query.
  - `update_dream`, `remove_duplicate_dreams`, `consolidate_dreams`.
- **Cache**: `cache/dreaming_embeddings.pt`.

### 3.4 Memory Consolidation Script (update_memory.py)

A lightweight CLI script:
```bash
python update_memory.py
```
Performs:
- Consolidate semantic, episodic, and dreams.
- Processes new conversations for dream extraction.

## 3.5 Vision System (Vision.py)

Provides camera-based perception:

- **YOLOv8s Integration** (Ultralytics) loaded from `./models/vision/yolov8s-oiv7.pt` with Open Images V7.
- **Face Recognition**: Uses `FaceRecognition` for embeddings, cosine matching, and `FaceDB` for SQLite storage.
- **VisionSystem Class**:
  - `start`/`stop`, `detect_objects`, `process_frame`, `vision_loop`.
  - Real-time object detection with bounding circles, dashed arcs, holographic panels drawn via OpenCV.
  - Circular piper laser UI aesthetic, dynamic data panels around targets.
- **Identification Flow**:
  - `/identify [name]` command triggers `identify_face` to crop face, generate/update embedding, save in DB.
  - Background worker thread (`face_recognition_thread`) consumes `face_queue` for asynchronous matching.
- **Thread Safety**: `frame_lock` ensures atomic access to `current_frame` and `detected_objects`.
- **Customization**: Color scheme overridden for darker purple tones.

## 3.6 Voice Synthesis (Voice.py)

- **TTS Engine**: `piper.exe` with ONNX model (`jarvis-high.onnx`).
- **Thread Safety**: `piper_lock` prevents concurrent calls.
- **Workflow**:
  1. Cleanup previous `test1.wav`.
  2. Run Piper via subprocess, pipe text input, timeout handling.
  3. Play audio with `winsound.PlaySound`.
  4. Async file cleanup.

---

## 4. Integration and APIs

- **Ollama**: Local LLM hosting for chat and embeddings.
- **Ultralytics YOLO**: Object detection.
- **OpenCV**: Frame capture, drawing, window management.
- **face_recognition**: dlib-based facial embedding extraction.
- **SQLite3**: Persistent face DB via `FaceDB.py`.
- **Pydantic v2**: Typed memory models.
- **Tkinter & PIL**: Desktop UI with advanced animations.
- **duckduckgo_search**, **transformers**: Supplemental search and emotion classification.

## 5. Data Storage and Formats

- Memory files (`Semantic.json`, `Episodic.json`, `Dreaming.json`) are JSONL (one JSON object per line).
- Embedding caches (`*.pt`) store tensors and last-modified timestamps.
- Synthetic conversations as plain `.txt` files in `synthetic_conversations/`.
- Face images and embeddings stored in SQLite via `FaceDB`.

## 6. Threading and Concurrency

- **Main Thread**: UI event loop, command dispatch.
- **Voice Thread**: Daemon thread processing `voice_queue`.
- **Vision Thread**: Continuously captures frames and processes detections.
- **Face Recognition Worker**: Background thread for embedding-based identification.
- Use of thread-safe queues and locks (`queue.Queue`, `threading.Lock`, `frame_lock`).

## 7. Configuration and Parameters

- **Detection Thresholds**: `confidence_threshold=0.5`, `detection_threshold=0.3`, `similarity_threshold=0.55` (tunable).
- **Tracking Params**: `max_tracker_age=60`, `history_length=30`, `max_detected_objects=4`.
- **Embedding Frames**: Interval reduced from 5→3 frames.
- **Arc Reactor Animation**: Pulse speeds, glow radii, point counts adjustable in UI.

## 8. External Dependencies and Requirements

- Python 3.8+
- PyTorch
- ultralytics (YOLOv8)
- OpenCV-Python
- face_recognition
- sqlite3 (standard library)
- numpy
- pandas (optional for data analysis)
- pydantic
- Pillow
- duckduckgo_search
- transformers
- ollama Python API
- Piper TTS (piper.exe, jarvis-high.onnx)

Install via:
```bash
pip install torch ultralytics opencv-python face_recognition numpy pydantic pillow duckduckgo_search transformers ollama
``` 

## 9. Extension Points and Roadmap

- **Plugin Architecture**: Introduce modular tool registry for adding new LLM tools.
- **Web Interface**: Migrate UI to web (React/Flask) for cross-platform.
- **Multi-Camera Support**: Extend VisionSystem to handle streams from multiple cameras.
- **GPU Acceleration**: Leverage CUDA for YOLO inference and PyTorch embeddings.
- **Hybrid Memory**: Graph database (Neo4j) for relational memory linking.
- **Voice Platforms**: Support cross-platform TTS and ASR engines.
- **Security & Privacy**: Encrypted local storage, opt-in face recognition.

---

*End of Technical Reference* 
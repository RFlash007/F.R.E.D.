"""
Face Database Module for FRED Vision System

This module handles storage and retrieval of face information using a SQLite database.
It provides functionality to:
- Store identified faces with metadata and facial embeddings
- Retrieve face information by name or by embedding similarity
- Track when faces were last seen

Threading Note:
SQLite connections cannot be shared between threads in Python.
Each function that requires database access should create its own connection,
use it, and promptly close it to avoid threading issues.
"""

# Import required libraries
import sqlite3      # For SQLite database operations
import os           # For file and directory operations
import numpy as np  # For numerical operations on arrays (used for face image processing)
import json         # For serializing/deserializing data structures (used for bbox storage)
import cv2          # OpenCV library for computer vision tasks
import time         # For timestamp generation
import pickle       # For serializing face embeddings
import torch        # For tensor operations and cosine similarity
from pathlib import Path  # For cross-platform file path handling
import logging      # For logging errors

# Define the database directory and file path using Path for cross-platform compatibility
DB_DIR = Path("./data")           # Directory to store database and related files
DB_PATH = DB_DIR / "faces.db"     # Full path to the SQLite database file
EMBEDDING_CACHE_PATH = DB_DIR / "face_embeddings_cache.pkl"  # Cache for face embeddings

class FaceDatabase:
    """
    Manages a database of identified faces with associated metadata and embeddings.
    
    This class provides an interface to store and retrieve face data including:
    - Person's name
    - Face image
    - Face embeddings for recognition
    - Bounding box coordinates
    - Timestamps for tracking
    
    Note: Create a new instance of this class for each operation to avoid threading issues.
    Do not share database connections between threads.
    """
    
    def __init__(self):
        """
        Initialize the face database and create tables if they don't exist.
        
        Creates a new SQLite connection and initializes the faces table
        with the required schema if it doesn't already exist.
        """
        # Create the data directory if it doesn't exist
        os.makedirs(DB_DIR, exist_ok=True)  # Creates directory only if it doesn't exist
        
        # Initialize SQLite database connection
        self.conn = sqlite3.connect(DB_PATH)  # Establish connection to the database
        self.cursor = self.conn.cursor()      # Create a cursor object to execute SQL commands
        
        # First check if the faces table exists
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='faces'")
        table_exists = self.cursor.fetchone() is not None
        
        if not table_exists:
            # Create faces table if it doesn't exist
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Unique identifier for each face record
                    name TEXT NOT NULL,                    -- Person's name (required field)
                    face_embedding BLOB,                   -- Binary blob of face embedding vector
                    bbox TEXT,                             -- Bounding box in JSON format (x1,y1,x2,y2)
                    last_seen TIMESTAMP,                   -- When the face was last seen (Unix timestamp)
                    image_path TEXT                        -- Path to stored face image on disk
                )
            ''')
            self.conn.commit()
        else:
            # Check if the face_embedding column exists
            self.cursor.execute("PRAGMA table_info(faces)")
            columns = [column[1] for column in self.cursor.fetchall()]
            
            if 'face_embedding' not in columns:
                # Add the face_embedding column if it doesn't exist
                print("Upgrading faces database to support facial embeddings...")
                try:
                    self.cursor.execute('ALTER TABLE faces ADD COLUMN face_embedding BLOB')
                    self.conn.commit()
                    print("Database upgrade completed successfully.")
                except sqlite3.Error as e:
                    print(f"Error upgrading database: {str(e)}")
        
        # Ensure the table exists regardless of the path taken
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                face_embedding BLOB,
                bbox TEXT,
                last_seen TIMESTAMP,
                image_path TEXT
            )
        ''')
        self.conn.commit()  # Commit the table creation to save changes to the database
    
    def add_face(self, name, face_image, bbox, embedding=None):
        """
        Add a new face to the database.
        
        Args:
            name (str): Name of the person
            face_image (numpy.ndarray): Face image cropped from the frame
            bbox (tuple): Bounding box of the face (x1, y1, x2, y2)
            embedding (numpy.ndarray, optional): Face embedding vector
        
        Returns:
            int: ID of the inserted face record
            
        Notes:
            - Saves the face image to disk in the data/face_images directory
            - Stores the bounding box as a JSON string for flexibility
            - Sets the last_seen timestamp to the current time
            - Stores the face embedding as a binary blob if provided
        """
        # Generate current timestamp for file naming and last_seen field
        timestamp = int(time.time())  # Current time as Unix timestamp (seconds since epoch)
        
        # Create face_images directory if it doesn't exist
        os.makedirs(DB_DIR / "face_images", exist_ok=True)
        
        # Define the image path with unique name based on person and timestamp
        image_path = f"face_images/{name}_{timestamp}.jpg"
        
        # Save the face image to disk using OpenCV
        cv2.imwrite(str(DB_DIR / image_path), face_image)
        
        # Convert bbox tuple to JSON string for storage in database
        bbox_json = json.dumps(bbox)  # Serializes the tuple to a JSON string
        
        # Serialize the embedding if provided
        embedding_blob = None
        if embedding is not None:
            embedding_blob = pickle.dumps(embedding)
        
        # Insert new face record into the database
        self.cursor.execute(
            "INSERT INTO faces (name, face_embedding, bbox, last_seen, image_path) VALUES (?, ?, ?, ?, ?)",
            (name, embedding_blob, bbox_json, timestamp, image_path)  # Parameters to be inserted
        )
        self.conn.commit()  # Commit the insertion to save changes
        
        # Return the ID of the newly inserted record
        return self.cursor.lastrowid  # SQLite's built-in way to get the last inserted row ID
    
    def update_face(self, face_id, bbox=None, last_seen=None, embedding=None):
        """
        Update information for an existing face.
        
        Args:
            face_id (int): ID of the face to update
            bbox (tuple, optional): New bounding box
            last_seen (int, optional): Timestamp when the face was last seen
            embedding (numpy.ndarray, optional): Updated face embedding
        """
        # Update the bounding box if provided
        if bbox is not None:
            bbox_json = json.dumps(bbox)  # Convert tuple to JSON string
            self.cursor.execute(
                "UPDATE faces SET bbox = ? WHERE id = ?",
                (bbox_json, face_id)  # Parameters for the SQL query
            )
        
        # Update the last_seen timestamp if provided
        if last_seen is not None:
            self.cursor.execute(
                "UPDATE faces SET last_seen = ? WHERE id = ?",
                (last_seen, face_id)  # Parameters for the SQL query
            )
        
        # Update the face embedding if provided
        if embedding is not None:
            embedding_blob = pickle.dumps(embedding)
            self.cursor.execute(
                "UPDATE faces SET face_embedding = ? WHERE id = ?",
                (embedding_blob, face_id)  # Parameters for the SQL query
            )
        
        # Commit changes to the database
        self.conn.commit()
    
    def get_face_by_name(self, name):
        """
        Get the most recent face record for a person by name.
        
        Args:
            name (str): Name of the person
        
        Returns:
            dict or None: Face data if found, None otherwise
            
        The returned dictionary contains:
        - id: Database ID
        - name: Person's name
        - bbox: Bounding box coordinates (x1, y1, x2, y2)
        - embedding: Face embedding vector (if available)
        - last_seen: Timestamp when last seen
        - image_path: Path to the saved face image
        """
        try:
            # Check if the face_embedding column exists
            self.cursor.execute("PRAGMA table_info(faces)")
            columns = [column[1] for column in self.cursor.fetchall()]
            
            # Query the database for the most recent face record with the given name
            if 'face_embedding' in columns:
                # If face_embedding column exists, include it in the query
                self.cursor.execute(
                    "SELECT id, name, face_embedding, bbox, last_seen, image_path FROM faces WHERE name = ? ORDER BY last_seen DESC LIMIT 1",
                    (name,)  # Parameter for the SQL query (note the comma to make it a tuple)
                )
            else:
                # For backward compatibility with older database schemas
                self.cursor.execute(
                    "SELECT id, name, bbox, last_seen, image_path FROM faces WHERE name = ? ORDER BY last_seen DESC LIMIT 1",
                    (name,)
                )
                
            face = self.cursor.fetchone()  # Get the first (and only) result row
            
            # If a face was found, convert it to a dictionary with parsed values
            if face:
                if 'face_embedding' in columns:
                    face_id, name, embedding_blob, bbox_json, last_seen, image_path = face  # Unpack the tuple
                    
                    # Deserialize the embedding if it exists
                    embedding = None
                    if embedding_blob:
                        try:
                            embedding = pickle.loads(embedding_blob)
                        except Exception:
                            # If there's an error deserializing, just return None for the embedding
                            pass
                else:
                    # For backward compatibility with older database schemas
                    face_id, name, bbox_json, last_seen, image_path = face
                    embedding = None
                
                return {
                    'id': face_id,
                    'name': name,
                    'embedding': embedding,
                    'bbox': json.loads(bbox_json),  # Parse JSON string back to Python object
                    'last_seen': last_seen,
                    'image_path': image_path
                }
            return None  # Return None if no face was found
        except Exception as e:
            logging.error(f"Error in get_face_by_name: {str(e)}")
            return None
    
    def get_all_faces(self):
        """
        Get all faces in the database.
        
        Returns:
            list: List of face dictionaries, each containing:
            - id: Database ID
            - name: Person's name
            - bbox: Bounding box coordinates (x1, y1, x2, y2)
            - embedding: Face embedding vector (if available)
            - last_seen: Timestamp when last seen
            - image_path: Path to the saved face image
        """
        try:
            # Check if the face_embedding column exists
            self.cursor.execute("PRAGMA table_info(faces)")
            columns = [column[1] for column in self.cursor.fetchall()]
            
            # Query the database for all faces, ordered by name and then by most recent
            if 'face_embedding' in columns:
                # If face_embedding column exists, include it in the query
                self.cursor.execute(
                    "SELECT id, name, face_embedding, bbox, last_seen, image_path FROM faces ORDER BY name, last_seen DESC"
                )
            else:
                # For backward compatibility with older database schemas
                self.cursor.execute(
                    "SELECT id, name, bbox, last_seen, image_path FROM faces ORDER BY name, last_seen DESC"
                )
                
            faces = self.cursor.fetchall()  # Get all result rows
            
            # Initialize an empty list to store the results
            result = []
            
            # Process each face record and add it to the result list
            for face in faces:
                if 'face_embedding' in columns:
                    face_id, name, embedding_blob, bbox_json, last_seen, image_path = face  # Unpack the tuple
                    
                    # Deserialize the embedding if it exists
                    embedding = None
                    if embedding_blob:
                        try:
                            embedding = pickle.loads(embedding_blob)
                        except Exception:
                            # If there's an error deserializing, just return None for the embedding
                            pass
                else:
                    # For backward compatibility with older database schemas
                    face_id, name, bbox_json, last_seen, image_path = face
                    embedding = None
                
                result.append({
                    'id': face_id,
                    'name': name,
                    'embedding': embedding,
                    'bbox': json.loads(bbox_json),  # Parse JSON string back to Python object
                    'last_seen': last_seen,
                    'image_path': image_path
                })
            
            # Return the list of face dictionaries
            return result
        except Exception as e:
            logging.error(f"Error in get_all_faces: {str(e)}")
            return []
    
    def get_most_similar_face(self, embedding, similarity_threshold=0.6):
        """
        Find the most similar face to the given embedding.
        
        Args:
            embedding (numpy.ndarray): Face embedding to compare against
            similarity_threshold (float): Minimum similarity score to consider a match (0-1)
            
        Returns:
            dict or None: The most similar face if above threshold, None otherwise
        """
        # Convert input embedding to tensor
        query_embedding = torch.tensor(embedding, dtype=torch.float32)
        
        # Get all faces with embeddings
        self.cursor.execute(
            "SELECT id, name, face_embedding, bbox, last_seen, image_path FROM faces WHERE face_embedding IS NOT NULL"
        )
        faces = self.cursor.fetchall()
        
        if not faces:
            return None
            
        # Compare with all faces
        max_similarity = -1
        best_match = None
        
        for face in faces:
            face_id, name, embedding_blob, bbox_json, last_seen, image_path = face
            
            try:
                # Deserialize the embedding
                face_embedding = pickle.loads(embedding_blob)
                face_embedding_tensor = torch.tensor(face_embedding, dtype=torch.float32)
                
                # Calculate cosine similarity
                similarity = torch.cosine_similarity(
                    query_embedding.unsqueeze(0),
                    face_embedding_tensor.unsqueeze(0),
                    dim=1
                ).item()
                
                # Update best match if this one is better
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = {
                        'id': face_id,
                        'name': name,
                        'embedding': face_embedding,
                        'bbox': json.loads(bbox_json),
                        'last_seen': last_seen,
                        'image_path': image_path,
                        'similarity': similarity
                    }
            except Exception:
                # Skip faces with invalid embeddings
                continue
        
        # Return the best match if it's above the threshold
        if best_match and best_match['similarity'] >= similarity_threshold:
            return best_match
        
        return None
    
    def close(self):
        """
        Close the database connection.
        
        Always call this method when done using the database to free resources
        and prevent connection leaks.
        """
        # Check if connection exists before trying to close it
        if self.conn:
            self.conn.close()  # Close the SQLite connection to release resources

# Functions for embedding cache management
def initialize_embedding_cache():
    """Initialize the embedding cache directory"""
    os.makedirs(DB_DIR, exist_ok=True)
    
def load_face_embeddings_cache():
    """Load the cached face embeddings if available"""
    if os.path.exists(EMBEDDING_CACHE_PATH):
        try:
            with open(EMBEDDING_CACHE_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}

def save_face_embeddings_cache(embeddings_dict):
    """Save the face embeddings cache to disk"""
    try:
        with open(EMBEDDING_CACHE_PATH, 'wb') as f:
            pickle.dump(embeddings_dict, f)
    except Exception:
        pass
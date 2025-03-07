"""
Face Database Module for FRED Vision System

This module handles storage and retrieval of face information using a SQLite database.
It provides functionality to:
- Store identified faces with metadata
- Retrieve face information 
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
from pathlib import Path  # For cross-platform file path handling

# Define the database directory and file path using Path for cross-platform compatibility
DB_DIR = Path("./data")           # Directory to store database and related files
DB_PATH = DB_DIR / "faces.db"     # Full path to the SQLite database file

class FaceDatabase:
    """
    Manages a database of identified faces with associated metadata.
    
    This class provides an interface to store and retrieve face data including:
    - Person's name
    - Face image
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
        
        # Create faces table if it doesn't exist using SQL DDL (Data Definition Language)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Unique identifier for each face record
                name TEXT NOT NULL,                    -- Person's name (required field)
                face_encoding TEXT,                    -- Reserved for future use with face embeddings
                bbox TEXT,                             -- Bounding box in JSON format (x1,y1,x2,y2)
                last_seen TIMESTAMP,                   -- When the face was last seen (Unix timestamp)
                image_path TEXT                        -- Path to stored face image on disk
            )
        ''')
        self.conn.commit()  # Commit the table creation to save changes to the database
    
    def add_face(self, name, face_image, bbox):
        """
        Add a new face to the database.
        
        Args:
            name (str): Name of the person
            face_image (numpy.ndarray): Face image cropped from the frame
            bbox (tuple): Bounding box of the face (x1, y1, x2, y2)
        
        Returns:
            int: ID of the inserted face record
            
        Notes:
            - Saves the face image to disk in the data/face_images directory
            - Stores the bounding box as a JSON string for flexibility
            - Sets the last_seen timestamp to the current time
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
        
        # Insert new face record into the database
        self.cursor.execute(
            "INSERT INTO faces (name, bbox, last_seen, image_path) VALUES (?, ?, ?, ?)",
            (name, bbox_json, timestamp, image_path)  # Parameters to be inserted
        )
        self.conn.commit()  # Commit the insertion to save changes
        
        # Return the ID of the newly inserted record
        return self.cursor.lastrowid  # SQLite's built-in way to get the last inserted row ID
    
    def update_face(self, face_id, bbox=None, last_seen=None):
        """
        Update information for an existing face.
        
        Args:
            face_id (int): ID of the face to update
            bbox (tuple, optional): New bounding box
            last_seen (int, optional): Timestamp when the face was last seen
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
        - last_seen: Timestamp when last seen
        - image_path: Path to the saved face image
        """
        # Query the database for the most recent face record with the given name
        self.cursor.execute(
            "SELECT id, name, bbox, last_seen, image_path FROM faces WHERE name = ? ORDER BY last_seen DESC LIMIT 1",
            (name,)  # Parameter for the SQL query (note the comma to make it a tuple)
        )
        face = self.cursor.fetchone()  # Get the first (and only) result row
        
        # If a face was found, convert it to a dictionary with parsed values
        if face:
            face_id, name, bbox_json, last_seen, image_path = face  # Unpack the tuple
            return {
                'id': face_id,
                'name': name,
                'bbox': json.loads(bbox_json),  # Parse JSON string back to Python object
                'last_seen': last_seen,
                'image_path': image_path
            }
        return None  # Return None if no face was found
    
    def get_all_faces(self):
        """
        Get all faces in the database.
        
        Returns:
            list: List of face dictionaries, each containing:
            - id: Database ID
            - name: Person's name
            - bbox: Bounding box coordinates (x1, y1, x2, y2)
            - last_seen: Timestamp when last seen
            - image_path: Path to the saved face image
        """
        # Query the database for all faces, ordered by name and then by most recent
        self.cursor.execute(
            "SELECT id, name, bbox, last_seen, image_path FROM faces ORDER BY name, last_seen DESC"
        )
        faces = self.cursor.fetchall()  # Get all result rows
        
        # Initialize an empty list to store the results
        result = []
        
        # Process each face record and add it to the result list
        for face in faces:
            face_id, name, bbox_json, last_seen, image_path = face  # Unpack the tuple
            result.append({
                'id': face_id,
                'name': name,
                'bbox': json.loads(bbox_json),  # Parse JSON string back to Python object
                'last_seen': last_seen,
                'image_path': image_path
            })
        
        # Return the list of face dictionaries
        return result
    
    def close(self):
        """
        Close the database connection.
        
        Always call this method when done using the database to free resources
        and prevent connection leaks.
        """
        # Check if connection exists before trying to close it
        if self.conn:
            self.conn.close()  # Close the SQLite connection to release resources
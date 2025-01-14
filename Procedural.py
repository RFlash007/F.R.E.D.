# Import required libraries
import json  # For handling JSON data structures and file operations
import ollama  # For AI model interactions

# Define constant file paths
PREFERENCES_PATH = 'Preferences.json'  # Path to the preferences JSON file
PROCEDURAL_FILE_PATH = 'Procedural.txt'  # Path to the procedural template file

def open_file(filepath):
    """
    Opens and reads a file, returning its contents as a string.
    
    Args:
        filepath (str): Path to the file to be opened
        
    Returns:
        str: Contents of the file
    """
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def load_preferences():
    """
    Loads preferences from a JSON file with comprehensive error handling.
    
    JSON (JavaScript Object Notation) is a lightweight data format that stores
    data as key-value pairs and arrays. In Python, JSON data is converted to 
    dictionaries and lists.
    
    The function attempts to:
    1. Open and read the JSON file
    2. Parse the JSON content into a Python dictionary
    3. Handle potential errors gracefully
    
    Returns:
        dict: A dictionary containing preference key-value pairs
              If file not found, returns default preferences
              If JSON decode error, returns empty dictionary
    
    Error Handling:
    - FileNotFoundError: Creates new file with default preferences
    - JSONDecodeError: Returns empty dict if JSON syntax is invalid
    """
    try:
        # Attempt to open and parse the JSON file
        with open(PREFERENCES_PATH, 'r', encoding='utf-8') as file:
            return json.load(file)  # Converts JSON to Python dictionary
    except FileNotFoundError:
        # If file doesn't exist, create default preferences
        print(f"Preferences file not found at {PREFERENCES_PATH}. Creating a new one with default preferences.")
        default_prefs = {
            "communication_style": {
                "formality": "casual",
                "humor_level": "moderate",
                "verbosity": "concise",
                "technical_depth": "advanced",
                "include_references": True,
                "primary_topics": ["python", "ai", "system_design", "Large Language Models"],
            }
        }
        save_preferences(default_prefs)
        return default_prefs
    except json.JSONDecodeError as e:
        # Handle invalid JSON format
        print(f"Error decoding JSON from {PREFERENCES_PATH}: {e}")
        return {}

def save_preferences(preferences):
    """
    Saves preferences dictionary to a JSON file.
    
    This function demonstrates how to write Python data structures to JSON format.
    The json.dump() method serializes Python objects to JSON format:
    - Python dict → JSON object
    - Python list → JSON array
    - Python str → JSON string
    - Python int/float → JSON number
    - Python True/False → JSON true/false
    - Python None → JSON null
    
    Args:
        preferences (dict): Dictionary containing preference settings
        
    The indent=4 parameter in json.dump() creates pretty-printed JSON with
    proper indentation for better readability.
    """
    try:
        with open(PREFERENCES_PATH, 'w', encoding='utf-8') as file:
            json.dump(preferences, file, indent=4)
    except Exception as e:
        print(f"Error saving preferences: {e}")

def get_prompt():
    """
    Retrieves the procedural prompt template and injects current preferences.
    
    This function demonstrates string formatting with JSON data:
    1. Reads the prompt template
    2. Loads current preferences
    3. Converts preferences dict back to formatted JSON string
    4. Injects the JSON string into the template
    
    Returns:
        str: Formatted prompt with current preferences
    """
    prompt_template = open_file(PROCEDURAL_FILE_PATH)
    #preferences = load_preferences()
    # Convert Python dict back to formatted JSON string
    #formatted_preferences = json.dumps(preferences, indent=4)
    #return prompt_template.format(preferences=formatted_preferences)
    return prompt_template

def update_preferences(new_preferences):
    """
    Updates existing preferences with new values while validating keys.
    
    This function demonstrates how to safely update JSON-based data:
    1. Load existing preferences
    2. Validate new keys against existing structure
    3. Update only valid keys
    4. Save back to JSON file
    
    Args:
        new_preferences (dict): Dictionary containing new preference values
    """
    preferences = load_preferences()
    valid_keys = preferences.keys()
    for key in new_preferences:
        if key in valid_keys:
            preferences[key] = new_preferences[key]
        else:
            print(f"Invalid preference key: {key}. Skipping.")
    save_preferences(preferences)
    print("Preferences have been updated successfully.")

def set_preference(key, value):
    """
    Sets a single preference value.
    
    A convenience function demonstrating how to update a single JSON value
    by converting it to a dictionary update operation.
    
    Args:
        key (str): The preference key to update
        value: The new value for the preference
    """
    update_preferences({key: value})

def test_update_preferences():
    """
    Tests the preference update functionality.
    
    This function demonstrates JSON operations in a testing context:
    1. Loading initial state
    2. Performing updates
    3. Verifying changes
    4. Validating data integrity
    """
    print("Testing preference updates...")
    original_prefs = load_preferences()
    print("Original Preferences:", original_prefs)
    
    # Update preferences with test data
    test_prefs = {
        "theme": "dark",
        "language": "es",
        "invalid_key": "test"  # This key should be rejected
    }
    update_preferences(test_prefs)
    
    # Verify updates
    updated_prefs = load_preferences()
    print("Updated Preferences:", updated_prefs)
    
    # Validate changes
    assert updated_prefs["theme"] == "dark"
    assert updated_prefs["language"] == "es"
    assert "invalid_key" not in updated_prefs
    print("All tests passed.")

def main():
    """
    Main function demonstrating complete JSON workflow:
    1. Reading and displaying current state
    2. Performing bulk updates
    3. Displaying modified state
    """
    # Display current prompt with preferences
    print("Current Prompt:")
    print(get_prompt())
    
    # Demonstrate bulk preference updates
    new_prefs = {
        "theme": "dark",
        "notifications": False,
        "language": "fr",
        "default_task_priority": "high"
    }
    update_preferences(new_prefs)
    
    # Display updated prompt
    print("\nUpdated Prompt:")
    print(get_prompt())

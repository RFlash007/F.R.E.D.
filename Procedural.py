# Import required libraries
import json  # For handling JSON data structures and file operations
import ollama  # For AI model interactions

# Define constant file paths
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
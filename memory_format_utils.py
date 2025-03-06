import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MEMORY_FILES = {
    "semantic": "Semantic.json",
    "episodic": "Episodic.json", 
    "dreaming": "Dreaming.json"  # updated from assumptions
}

def ensure_jsonl_format(file_path, silent=False):
    """
    Ensure the memory file is in JSONL format (one JSON object per line).
    
    Args:
        file_path (str): Path to the memory file
        silent (bool): If True, suppress log messages
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not silent:
        logger.info(f"Checking format of {file_path}...")
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            if not silent:
                logger.warning(f"File {file_path} not found. Will be created when needed.")
            return True
            
        # Skip empty files
        if os.path.getsize(file_path) == 0:
            return True
            
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            return True
            
        # First check if the file is already in JSONL format
        # by trying to parse each line as JSON
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                
            # Try parsing each line as JSON
            for line in lines:
                json.loads(line)
                
            # If we got here, the file is already in JSONL format
            return True
                
        except json.JSONDecodeError:
            # Not in JSONL format, check if it's a JSON array
            pass
        
        # Try parsing as a JSON array
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing {file_path}: {e}")
            logger.error(f"The file is neither valid JSON nor JSONL format.")
            return False
        
        # Ensure it's a list
        if not isinstance(data, list):
            logger.error(f"Expected a JSON array in {file_path}, but found {type(data).__name__}.")
            return False
        
        # Create backup
        backup_file = f"{file_path}.bak"
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
            
        if not silent:
            logger.info(f"Created backup at {backup_file}")
        
        # Convert to JSONL format
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        
        if not silent:
            logger.info(f"Converted {file_path} to JSONL format ({len(data)} items).")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return False

def ensure_all_memory_files():
    """
    Ensure all memory files are in JSONL format.
    
    Returns:
        bool: True if all conversions successful, False otherwise
    """
    success = True
    for memory_type, file_path in MEMORY_FILES.items():
        if not ensure_jsonl_format(file_path):
            success = False
    return success

if __name__ == "__main__":
    # Run as a standalone script
    logger.info("Checking memory file formats...")
    success = ensure_all_memory_files()
    if success:
        logger.info("All memory files are now in the correct format.")
    else:
        logger.error("Failed to process one or more memory files.") 
import os
import ollama
import os
import torch
import logging
import json
from pathlib import Path
import numpy as np
def get_project_info(user_input) -> tuple[str, str]:
    """
    Open a project file

    Args:
        user_input (str): The user's input message containing project name

    Returns:
        tuple[str, str]: A tuple containing (project_name, content) or (error_message, "")
    """
    # Get the project name from the user input
    prompt = f"""Lets think about this step by step
                1. Get the project name from this message
                2. **ONLY RETURN THE PROJECT NAME NO OTHER TEXT OR DIALOGUE**
                For example if the user says "Open a project called 'My Project'" return "My Project"
                or if the user says "Edit the project called 'My Project'" return "My Project"
                Message: {user_input}"""
    messages = [{"role": "user", "content": prompt}]
    response = ollama.chat(model="llama3.2:3b", messages=messages)
    project_name = response["message"]["content"]

    try:
        # Use proper path joining and add .txt extension
        project_file = os.path.join('Projects', f"{project_name.lower()}.txt")
        
        # Check if file exists
        if not os.path.exists(project_file):
            return f"Project {project_name} does not exist", ""
            
        # Read the file content
        with open(project_file, 'r') as file:
            content = file.read()
            
        return project_name, content
    except Exception as e:
        return f"Failed to open project: {e}", ""
    

# Cache setup for storing embeddings and processed facts
CACHE_DIR = Path("cache")
EMBEDDINGS_CACHE = CACHE_DIR / "semantic_embeddings.pt"  # PyTorch tensor cache
FACTS_CACHE = CACHE_DIR / "facts.json"  # Processed facts cache

def initialize_cache():
    """Initialize cache directory and files
    
    Creates:
    - cache/ directory if it doesn't exist
    - semantic_embeddings.pt: stores tensor embeddings and last modified time
    - facts.json: stores processed fact data
    """
    CACHE_DIR.mkdir(exist_ok=True)
    if not EMBEDDINGS_CACHE.exists():
        torch.save({"embeddings": [], "last_modified": 0}, EMBEDDINGS_CACHE)
    if not FACTS_CACHE.exists():
        with open(FACTS_CACHE, 'w', encoding='utf-8') as f:
            json.dump([], f)

def load_cached_embeddings():
    """Load embeddings from cache if they're up to date
    
    Returns:
        torch.Tensor: Cached embeddings if valid
        None: If cache is invalid or missing
    
    Cache is considered valid if the cache's last_modified timestamp
    is newer than or equal to Semantic.txt's last modified time
    """
    semantic_modified = os.path.getmtime("Semantic.txt")
    try:
        cache = torch.load(EMBEDDINGS_CACHE, weights_only=True, map_location='cpu')
        if cache["last_modified"] >= semantic_modified:
            return torch.tensor(cache["embeddings"]) if isinstance(cache["embeddings"], list) else cache["embeddings"]
    except Exception as e:
        logging.warning(f"Error loading semantic cache: {e}")
    return None

def save_embeddings_cache(embeddings):
    """Save embeddings to cache
    
    Args:
        embeddings (torch.Tensor): Embedding vectors to cache
        
    Saves a dictionary containing:
        - embeddings: The actual embedding vectors
        - last_modified: Current timestamp of Semantic.txt
    """
    try:
        cache_data = {
            "embeddings": embeddings,
            "last_modified": torch.tensor(os.path.getmtime("Semantic.txt"))
        }
        torch.save(cache_data, EMBEDDINGS_CACHE)
    except Exception as e:
        logging.error(f"Error saving semantic cache: {e}")

def batch_embed_texts(texts, batch_size=5):
    """Embed multiple texts in batches for efficiency
    
    Args:
        texts (list): List of text chunks to embed
        batch_size (int): Number of texts to process at once
        
    Returns:
        torch.Tensor: Matrix of embeddings, one per text
    """
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = [ollama.embeddings(model='nomic-embed-text', prompt=text)["embedding"] 
                     for text in batch]
        all_embeddings.extend(embeddings)
    return torch.tensor(all_embeddings)

def open_file(filepath):
    """Helper function to read file content with robust encoding handling
    
    Args:
        filepath (str): Path to file to read
        
    Returns:
        str: Content of file
        
    Tries UTF-8 first, falls back to other encodings if needed
    """
    encodings = ['utf-8', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as infile:
                return infile.read()
        except UnicodeDecodeError:
            continue
    
    # If all encodings fail, try binary read and decode
    try:
        with open(filepath, 'rb') as infile:
            content = infile.read()
            return content.decode('utf-8', errors='ignore')
    except Exception as e:
        logging.error(f"Failed to read file {filepath} with any encoding: {e}")
        return ""

def create_semantic(memory: str) -> None:
    """Extracts factual information from conversations
    
    Args:
        memory (str): Conversation to extract facts from
        
    Process:
        1. Extracts facts using LLM
        2. Saves to file
        3. Invalidates cache
    """
    #Extracts factual information from conversations and stores it in a simple format
    fact_extraction_prompt = f"""Extract factual information from this conversation.

    CONVERSATION:
    {memory}

    INSTRUCTIONS:
    - Extract only verifiable facts and knowledge
    - Ignore conversation flow, timestamps, or contextual details
    - Each fact should be self-contained and complete
    - Format: "• [CATEGORY] fact"
    
    Example format:
    [PERSONAL] John is allergic to peanuts
    [PREFERENCE] John prefers tea over coffee
    [TECHNICAL] Python was created by Guido van Rossum
    [LOCATION] John lives in Seattle
    """

    response = ollama.chat(
        model="huihui_ai/qwen2.5-abliterate:14b", 
        messages=[{"role": "user", "content": fact_extraction_prompt}]
    )

    if response["message"]["content"]:
        with open("Semantic.txt", 'a', encoding='utf-8') as file:
            file.write(f"\n{response['message']['content']}")
        
        # Invalidate cache by updating last_modified time
        if EMBEDDINGS_CACHE.exists():
            save_embeddings_cache([])

def update_semantic(conversation: str) -> None:
    """Updates semantic memory by consolidating facts
    
    Args:
        conversation (str): New conversation to integrate
        
    Process:
        1. Gets LLM to update accessed facts
        2. Preserves non-accessed facts
        3. Saves updated facts
        4. Invalidates cache
    """
    #Updates semantic memory by consolidating and deduplicating facts.
    fact_update_prompt = f"""Review and update these facts based on new information.

    NEW CONVERSATION:
    {conversation}

    EXISTING FACTS:
    {accessed_memories}

    INSTRUCTIONS:
    1. Compare new facts with existing ones
    2. Remove duplicates
    3. Resolve conflicts (keep most recent/accurate)
    4. Combine related facts when possible
    5. Use format: "[CATEGORY] fact"
    6. Return ONLY the final, consolidated list of facts.
    
    Return ONLY the final, consolidated list of facts."""

    try:
        # Get updated facts from LLM
        response = ollama.chat(
            model="huihui_ai/qwen2.5-abliterate:14b",
            messages=[{"role": "user", "content": fact_update_prompt}]
        )
        
        if response["message"]["content"]:
            updated_facts = [response["message"]["content"]]
            
            # Update file with consolidated facts
            with open("Semantic.txt", 'r+', encoding='utf-8') as file:
                content = file.read()
                existing_facts = [fact.strip() for fact in content.split("\n") if fact.strip()]
                preserved_facts = [fact for fact in existing_facts if fact not in accessed_memories]
                
                # Write updated content
                file.seek(0)
                file.write("\n".join(preserved_facts + updated_facts))
                file.truncate()
            consolidate_memories()
            # Invalidate cache
            if EMBEDDINGS_CACHE.exists():
                save_embeddings_cache([])

                
    except Exception as e:
        logging.error(f"Error updating semantic memory: {str(e)}")
        raise

def recall_semantic(query: str, top_k: int = 2) -> list:
    """Retrieves relevant facts based on a query using semantic search"""
    if not os.path.exists("Semantic.txt"):
        return []

    try:
        initialize_cache()
        
        # Load and preprocess facts
        content = open_file("Semantic.txt")
        facts = [fact.strip() for fact in content.split("\n") if fact.strip()]
        
        # Return early if no facts exist
        if not facts:
            return []

        # Try to load cached embeddings
        fact_embeddings_tensor = load_cached_embeddings()
        
        if fact_embeddings_tensor is None or fact_embeddings_tensor.nelement() == 0:
            # Generate new embeddings in batches
            fact_embeddings_tensor = batch_embed_texts(facts)
            save_embeddings_cache(fact_embeddings_tensor)
        
        # Verify we have valid embeddings
        if fact_embeddings_tensor.nelement() == 0:
            logging.error("No valid embeddings found")
            return []
            
        # Get query embedding
        query_response = ollama.embeddings(model='nomic-embed-text', prompt=query)
        query_embedding = torch.tensor(query_response["embedding"])

        # Calculate similarities using optimized tensor operations
        similarities = torch.cosine_similarity(
            query_embedding.unsqueeze(0), 
            fact_embeddings_tensor
        )
        
        # Get top-k most relevant facts
        top_k = min(top_k, len(similarities))
        if top_k == 0:
            return []
            
        top_k_indices = torch.topk(similarities, top_k).indices
        relevant_facts = [facts[idx] for idx in top_k_indices]
        
        # Track accessed facts
        accessed_memories.extend(relevant_facts)
        
        return relevant_facts

    except Exception as e:
        logging.error(f"Error in recall_semantic: {str(e)}")
        return []
        
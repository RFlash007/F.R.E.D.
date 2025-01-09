import logging
import os
import torch
import ollama
import time
import json
from pathlib import Path

# ANSI escape sequences for coloring the output
CYAN = "\033[96m"
RESET_COLOR = "\033[0m"

# Track which memories have been accessed during the current session
accessed_memories = []

# Cache setup
CACHE_DIR = Path("cache")
EPISODIC_EMBEDDINGS_CACHE = CACHE_DIR / "episodic_embeddings.pt"
MEMORIES_CACHE = CACHE_DIR / "episodic_memories.json"

def initialize_cache():
    """Initialize cache directory and files
    
    Creates:
    - cache/ directory if it doesn't exist
    - episodic_embeddings.pt: stores tensor embeddings and last modified time
    - episodic_memories.json: stores processed memory data
    """
    CACHE_DIR.mkdir(exist_ok=True)
    if not EPISODIC_EMBEDDINGS_CACHE.exists():
        torch.save({"embeddings": [], "last_modified": 0}, EPISODIC_EMBEDDINGS_CACHE)
    if not MEMORIES_CACHE.exists():
        with open(MEMORIES_CACHE, 'w') as f:
            json.dump([], f)

def load_cached_embeddings():
    """Load embeddings from cache if they're up to date
    
    Returns:
    - torch.Tensor: Cached embeddings if valid
    - None: If cache is invalid or missing
    
    Cache is considered valid if:
    1. Cache file exists
    2. Cache's last_modified timestamp >= Episodic.txt's last modified time
    """
    episodic_modified = os.path.getmtime("Episodic.txt")  # Get source file timestamp
    try:
        cache = torch.load(EPISODIC_EMBEDDINGS_CACHE, weights_only=True, map_location='cpu')
        if cache["last_modified"] >= episodic_modified:  # Check if cache is fresh
            # Convert to tensor if stored as list
            return torch.tensor(cache["embeddings"]) if isinstance(cache["embeddings"], list) else cache["embeddings"]
    except Exception as e:
        logging.warning(f"Error loading episodic cache: {e}")
    return None

def save_embeddings_cache(embeddings):
    """Save embeddings to cache
    
    Args:
        embeddings: torch.Tensor of embeddings to cache
        
    Saves:
    - embeddings: The actual embedding vectors
    - last_modified: Current timestamp of Episodic.txt
    """
    try:
        cache_data = {
            "embeddings": embeddings,
            "last_modified": torch.tensor(os.path.getmtime("Episodic.txt"))
        }
        torch.save(cache_data, EPISODIC_EMBEDDINGS_CACHE)
    except Exception as e:
        logging.error(f"Error saving episodic cache: {e}")

def batch_embed_texts(texts, batch_size=5):
    """Embed multiple texts in batches for efficiency"""
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

def get_relevant_context(user_input, vault_embeddings, vault_content, top_k=2):
    if not isinstance(vault_embeddings, torch.Tensor) or vault_embeddings.nelement() == 0:  # Check type and emptiness
        return []

    # Get embedding for user input
    response = ollama.embeddings(model='nomic-embed-text', prompt=user_input)
    input_embedding = torch.tensor(response["embedding"])

    # Compute cosine similarity using optimized tensor operations
    cos_scores = torch.cosine_similarity(input_embedding.unsqueeze(0), vault_embeddings)
    
    # Use efficient topk operation
    top_k = min(top_k, len(cos_scores))
    if top_k == 0:
        return []
    top_indices = torch.topk(cos_scores, k=top_k).indices.tolist()

    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

def recall_episodic(input):
    """Retrieve relevant episodic memories with caching"""
    vault_path = "Episodic.txt"
    if not os.path.exists(vault_path):
        print("No Episodic.txt found.")
        return []

    try:
        initialize_cache()
        
        # Load vault content
        content = open_file(vault_path)
        vault_content = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]

        # Try to load cached embeddings
        vault_embeddings_tensor = load_cached_embeddings()
        
        if vault_embeddings_tensor is None:
            # Generate new embeddings in batches
            vault_embeddings_tensor = batch_embed_texts(vault_content)
            save_embeddings_cache(vault_embeddings_tensor)

        # Retrieve best matching context
        relevant_context = get_relevant_context(input, vault_embeddings_tensor, vault_content, top_k=1)
        
        # Store accessed memories
        if relevant_context:
            accessed_memories.extend(relevant_context)
            
        return relevant_context

    except Exception as e:
        logging.error(f"Error in recall_episodic: {e}")
        return []

def create_memory(conversation: str) -> str:
    """Creates a memory entry and invalidates cache
    
    Cache invalidation happens when:
    1. New memories are created
    2. Existing memories are updated
    3. Memories are pruned or consolidated
    """
    try:
        current_time = time.time()
        conversation_date = time.strftime("%d %B %Y", time.localtime(current_time))

        reflection_prompt_template = """You are creating a memory from the perspective of the Assistant Fred in this conversation summary. The conversation occurred on """ + conversation_date + """. If you do not have enough information for a field, use "N/A". Write one concise sentence per field. Focus on information that will be useful in future interactions. Include context_tags that are specific and reusable. Provide a memory_timestamp.
        
        Output valid JSON in this exact format and nothing else **WRITE NO OTHER TEXT OR DIALOGUE**:

            [
                "timestamp": "YYYY-MM-DD HH:MM",
                "tags": ["tag1", "tag2"],
                "summary": "Brief conversation summary",
                "insights": {
                    "positive": "What worked well",
                    "negative": "What to improve",
                    "learned": "Key learnings"
                }
            ]

        Conversation Summary:
        """ + conversation + """"
        """

        response = ollama.chat(
            model="huihui_ai/qwen2.5-abliterate:14b", 
            messages=[{"role": "user", "content": reflection_prompt_template}]
        )

        if not response.get("message", {}).get("content"):
            print("Error: No content in memory creation response")
            return ""

        episodic_content = response["message"]["content"]
        
        with open("Episodic.txt", 'a', encoding='utf-8') as file:
            file.write(f"\n\n{episodic_content}")
        
        # Invalidate cache by saving empty embeddings
        if EPISODIC_EMBEDDINGS_CACHE.exists():
            save_embeddings_cache([])
            
        return episodic_content
        
    except Exception as e:
        print(f"Error creating memory: {e}")
        return ""

def update_episodic(conversation_summary):
    """Updates Episodic memories and invalidates cache"""
    updated_memories = []
    memory_update_prompt = """Review and update these memory entries based on the new conversation.

    NEW CONVERSATION:
    """ + conversation_summary + """

    EXISTING MEMORIES:
    """ + str(accessed_memories) + """

    INSTRUCTIONS:
    1. Compare new conversation with existing memories
    2. Update only if new information conflicts or adds value
    3. Merge overlapping memories
    4. Keep JSON format
    5. **WRITE NO OTHER TEXT OR DIALOGUE**
    [
        "timestamp": "YYYY-MM-DD HH:MM",
        "tags": ["tag1", "tag2"],
        "summary": "Brief conversation summary",
        "insights": {
            "positive": "What worked well",
            "negative": "What to improve",
            "learned": "Key learnings"
        }
    ]
    """

    try:
        # Get updated memories from LLM
        messages = [{"role": "user", "content": memory_update_prompt}]
        response = ollama.chat(
            model="huihui_ai/qwen2.5-abliterate:14b", 
            messages=messages
        )

        if not response["message"]["content"]:
            logging.warning("No content returned from LLM")
            return

        updated_memories = [response["message"]["content"]]

        # Update file with consolidated memories
        with open("Episodic.txt", 'r+', encoding='utf-8') as file:
            # Read and filter existing content
            content = file.read()
            chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
            preserved_chunks = [chunk for chunk in chunks if chunk not in accessed_memories]
            
            # Write updated content
            file.seek(0)
            file.write("\n\n".join(preserved_chunks + updated_memories))
            file.truncate()
        consolidate_memories()
        # Invalidate cache after update
        if EPISODIC_EMBEDDINGS_CACHE.exists():
            save_embeddings_cache([])
    except Exception as e:
        logging.error(f"Error updating episodic memory: {str(e)}")
        raise

def prune_old_memories(max_memories=1000):
    """Remove oldest memories when exceeding threshold"""
    with open("Episodic.txt", 'r') as f:
        memories = f.read().split("\n\n")
    if len(memories) > max_memories:
        # Keep most recent memories
        memories = memories[-max_memories:]
        with open("Episodic.txt", 'w') as f:
            f.write("\n\n".join(memories))
        # Invalidate cache
        save_embeddings_cache([])

def consolidate_memories(similarity_threshold=0.95):
    """Merge similar facts to reduce redundancy
    
    Args:
        similarity_threshold (float): Threshold for considering facts similar
        
    Process:
        1. First removes exact duplicates
        2. Then embeds remaining facts
        3. Calculates similarity matrix
        4. Merges similar facts
        5. Updates file and cache
    """
    try:
        content = open_file("Episodic.txt")
        if not content:
            return
            
        # First remove exact duplicates using a set
        memories = []
        seen_facts = set()
        
        for memory in content.split("\n\n"):
            memory = memory.strip()
            if memory and memory not in seen_facts:
                memories.append(memory)
                seen_facts.add(memory)
        
        if not memories:
            return
            
        # Then check for semantic similarity
        embeddings = batch_embed_texts(memories)
        
        # Calculate similarity matrix
        similarity_matrix = torch.cosine_similarity(embeddings.unsqueeze(1), 
                                                 embeddings.unsqueeze(0))
        
        # Find and merge similar memories
        consolidated = []
        seen_indices = set()
        
        for i in range(len(memories)):
            if i in seen_indices:
                continue
                
            # Find similar memories
            similar_indices = torch.where(similarity_matrix[i] > similarity_threshold)[0].tolist()
            
            # Add to seen indices
            seen_indices.update(similar_indices)
            
            # Keep the first occurrence
            consolidated.append(memories[i])
        
        # Update file with consolidated memories
        with open("Episodic.txt", 'w', encoding='utf-8') as f:
            f.write("\n\n".join(consolidated))
        
        # Invalidate cache
        save_embeddings_cache([])
        
        logging.info(f"Consolidated {len(content.split('\n\n'))} memories into {len(consolidated)} unique memories")
        
    except Exception as e:
        logging.error(f"Error consolidating memories: {e}")

import logging
import os
import json
from typing import List, Optional
from pathlib import Path
from datetime import datetime
import re
import torch
import ollama
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.ERROR)

# Track which dreams have been accessed during the current session
accessed_dreams = []

# Cache setup for storing embeddings and processed dreams
CACHE_DIR = Path("cache")
EMBEDDINGS_CACHE = CACHE_DIR / "dreaming_embeddings.pt"  # PyTorch tensor cache
DREAMS_FILE = "Dreaming.json"  # Processed dreams storage
SYNTHETIC_CONVERSATIONS_DIR = Path("synthetic_conversations")  # Directory for synthetic conversations

class Dream(BaseModel):
    """
    Represents a single dream entry with insight type and content.
    Dreams represent insights, patterns, and creative connections derived from conversations.
    """
    insight_type: str
    content: str
    source: str = "unknown"  # Can be "real", "synthetic", or "unknown"
    
    def to_json(self) -> str:
        """
        Convert dream to JSON format.
        - Replaced .dict() with .model_dump() for Pydantic v2.
        """
        return json.dumps(self.model_dump())
    
    @classmethod
    def from_json(cls, json_str: str) -> "Dream":
        """
        Create a Dream from a JSON string.
        - Replaced parse_raw() with model_validate_json() for Pydantic v2.
        """
        try:
            return cls.model_validate_json(json_str)
        except Exception:
            # Fallback for compatibility with older versions
            return cls(**json.loads(json_str))

def initialize_cache():
    """Initialize cache directory and files."""
    CACHE_DIR.mkdir(exist_ok=True)
    if not EMBEDDINGS_CACHE.exists():
        torch.save({"embeddings": [], "last_modified": 0}, EMBEDDINGS_CACHE)
    if not os.path.exists(DREAMS_FILE):
        with open(DREAMS_FILE, 'w', encoding='utf-8') as f:
            pass
    SYNTHETIC_CONVERSATIONS_DIR.mkdir(exist_ok=True)
    
    # Direct cache population
    try:
        if os.path.getsize(DREAMS_FILE) > 0:
            dreams = [Dream.from_json(line) for line in open(DREAMS_FILE) if line.strip()]
            dream_texts = [f"[{d.insight_type}] {d.content}" for d in dreams]
            embeddings = batch_embed_texts(dream_texts)
            save_embeddings_cache(embeddings)
    except Exception as e:
        logging.error(f"Cache initialization failed: {e}")

def load_cached_embeddings():
    """Load embeddings from cache file if it exists and is up to date."""
    try:
        if not EMBEDDINGS_CACHE.exists():
            return None
            
        # Load cached data
        cache_data = torch.load(EMBEDDINGS_CACHE, map_location='cpu', weights_only=True)
        
        # Check if dreams file has been modified
        dream_file_mtime = os.path.getmtime(DREAMS_FILE) if os.path.exists(DREAMS_FILE) else 0
        
        if dream_file_mtime > cache_data.get("last_modified", 0):
            logging.info("Dreams file modified since last embedding, cache invalid")
            return None
            
        # Return embeddings if available
        embeddings = cache_data.get("embeddings", None)
        if embeddings is None or len(embeddings) == 0:
            return None
            
        # Ensure we return a proper tensor
        if isinstance(embeddings, list):
            return torch.tensor(embeddings)
        return embeddings
    except Exception as e:
        logging.error(f"Error loading cached embeddings: {e}")
        return None

def save_embeddings_cache(embeddings):
    """Save embeddings to cache file with current timestamp."""
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        
        # Get current mtime of dreams file
        dream_file_mtime = os.path.getmtime(DREAMS_FILE) if os.path.exists(DREAMS_FILE) else datetime.now().timestamp()
        
        # Save embeddings and timestamp
        torch.save({
            "embeddings": embeddings,
            "last_modified": dream_file_mtime
        }, EMBEDDINGS_CACHE)
    except Exception as e:
        logging.error(f"Error saving embeddings cache: {e}")

def batch_embed_texts(texts, batch_size=5):
    """Process text embeddings in batches to avoid memory issues."""
    if not texts:
        return torch.tensor([])
        
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            # Process each text individually to handle errors gracefully
            for text in batch:
                try:
                    response = ollama.embeddings(model='nomic-embed-text', prompt=text)
                    all_embeddings.append(response["embedding"])
                except Exception as e:
                    logging.error(f"Error embedding text: {e}")
                    # Add a zero vector as placeholder for failed embedding
                    all_embeddings.append([0.0] * 1024)  # Typical dimension for embeddings
        except Exception as e:
            logging.error(f"Error processing batch: {e}")
    
    if not all_embeddings:
        return torch.tensor([])
        
    return torch.tensor(all_embeddings)

def clean_ollama_response(text: str) -> str:
    """
    Clean up the response from Ollama by removing markdown code blocks and other formatting.
    """
    # Remove markdown code blocks
    text = re.sub(r'```(?:json)?\n', '', text)
    text = re.sub(r'```', '', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def generate_synthetic_conversation(conversation: str) -> Optional[str]:
    """
    Generate a synthetic conversation based on a real one and save it to file.
    """
    logging.debug("Generating synthetic conversation")
    
    # If conversation is too long, truncate it to avoid context limits
    max_length = 12000
    if len(conversation) > max_length:
        logging.warning(f"Truncating conversation from {len(conversation)} to {max_length} chars")
        conversation = conversation[:max_length]
    
    prompt = f"""Based on this conversation, create a NEW, DIFFERENT conversation that reveals deeper patterns, connections, and insights that might not be directly stated. Focus on themes, underlying motivations, and creative associations. Make it natural and reflective, as if exploring subconscious connections. Format as plain text dialogue.

Original conversation for context: {conversation}

Generate new conversation ***RETURN ONLY THE CONVERSATION NO EXTRA TEXT***:"""

    try:
        response = ollama.chat(
            model="huihui_ai/qwen2.5-abliterate:14b", 
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 2048}
        )
        
        synthetic_conversation = clean_ollama_response(response["message"]["content"])
        
        # Save the synthetic conversation with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        synthetic_filename = os.path.join(SYNTHETIC_CONVERSATIONS_DIR, f"synthetic_{timestamp}.txt")
        
        with open(synthetic_filename, 'w', encoding='utf-8') as f:
            f.write(synthetic_conversation)
            
        logging.info(f"Saved synthetic conversation to {synthetic_filename}")
        
        return synthetic_conversation
    except Exception as e:
        logging.error(f"Error generating synthetic conversation: {e}")
        return None

def read_synthetic_conversations(limit: int = 2) -> List[str]:
    """
    Read synthetic conversations from the synthetic_conversations directory.
    """
    try:
        if not os.path.exists(SYNTHETIC_CONVERSATIONS_DIR):
            logging.warning(f"Synthetic conversations directory does not exist: {SYNTHETIC_CONVERSATIONS_DIR}")
            return []
        
        # Get list of synthetic conversation files, sorted by creation time (newest first)
        files = [(os.path.getmtime(os.path.join(SYNTHETIC_CONVERSATIONS_DIR, f)), f) 
                for f in os.listdir(SYNTHETIC_CONVERSATIONS_DIR) 
                if f.startswith('synthetic_') and f.endswith('.txt')]
        
        # Sort files by modification time (newest first)
        files.sort(reverse=True)
        
        # Limit the number of files to process
        files = files[:limit]
        
        conversations = []
        for _, filename in files:
            try:
                with open(os.path.join(SYNTHETIC_CONVERSATIONS_DIR, filename), 'r', encoding='utf-8') as f:
                    content = f.read()
                    conversations.append(content)
            except Exception as e:
                logging.error(f"Error reading synthetic conversation {filename}: {e}")
        
        return conversations
    except Exception as e:
        logging.error(f"Error reading synthetic conversations: {e}")
        return []

def create_dream(conversation: str) -> str:
    """
    Extract dreams/insights from a conversation and store them in Dreaming.json.
    Also generates a synthetic conversation and extracts dreams from that.
    """
    result_messages = []
    dreams_saved = 0
    
    try:
        # 1. Extract dreams from the real conversation
        dreams_from_real = extract_dreams_from_conversation(conversation, is_synthetic=False)
        if dreams_from_real > 0:
            result_messages.append(f"Extracted {dreams_from_real} dreams from real conversation")
            dreams_saved += dreams_from_real
        
        # 2. Generate a synthetic conversation
        synthetic_conversation = generate_synthetic_conversation(conversation)
        if synthetic_conversation:
            # 3. Extract dreams from the synthetic conversation
            dreams_from_synthetic = extract_dreams_from_conversation(synthetic_conversation, is_synthetic=True)
            if dreams_from_synthetic > 0:
                result_messages.append(f"Extracted {dreams_from_synthetic} dreams from synthetic conversation")
                dreams_saved += dreams_from_synthetic
        else:
            result_messages.append("Failed to generate synthetic conversation")
        
        # 4. Clean up: remove duplicates and consolidate
        remove_duplicate_dreams()
        consolidate_dreams()
        
        # 5. Invalidate cache so next recall triggers re-embedding
        if EMBEDDINGS_CACHE.exists():
            save_embeddings_cache([])
            
        if dreams_saved > 0:
            return f"Created {dreams_saved} dreams"
        else:
            return "No dreams were created"
            
    except Exception as e:
        logging.error(f"Error in create_dream: {e}")
        return f"Error creating dreams: {str(e)}"

def extract_dreams_from_conversation(conversation: str, is_synthetic: bool = False) -> int:
    """
    Extract dreams/insights from a conversation and save them to the dreams file.
    Returns the number of dreams saved.
    """
    source_type = "synthetic" if is_synthetic else "real"
    dreams_saved = 0
    
    try:
        dream_extraction_prompt = f"""
        Analyze this conversation and extract 3-5 insights, patterns, or creative connections.
        
        For each insight, return JSON with:
        - "insight_type": one of "Pattern", "Connection", "Implication", "Question", "Hypothesis", "Metaphor", "Insight"
        - "content": detailed insight description (1-2 sentences)
        
        Focus on deeper meanings, creative connections, and novel observations. These should be interesting and varied.
        
        CONVERSATION:
        {conversation}
        
        FORMAT: Return a JSON array with this exact structure (and nothing else):
        [
            {{"insight_type": "Type", "content": "Description"}}
        ]
        """
        
        # Use format="json"
        response = ollama.chat(
            model="huihui_ai/qwen2.5-abliterate:14b",
            messages=[{"role": "user", "content": dream_extraction_prompt}],
            format="json",
            options={"temperature": 0.7}
        )
        
        if not response.get("message", {}).get("content"):
            logging.warning("No content in AI response")
            return 0
            
        content = response["message"]["content"]
        
        # Parse JSON response
        try:
            dreams_data = json.loads(content)
            
            # Handle empty or invalid responses
            if not dreams_data:
                logging.warning("AI returned empty dreams data")
                return 0
                
            if not isinstance(dreams_data, list):
                dreams_data = [dreams_data]
                
            # Process and save dream objects
            for dream_dict in dreams_data:
                # Validate presence of keys
                if not isinstance(dream_dict, dict) or "insight_type" not in dream_dict or "content" not in dream_dict:
                    logging.warning(f"Dream missing required fields: {dream_dict}")
                    continue
                
                # Create Dream object with source type
                dream = Dream(
                    insight_type=dream_dict["insight_type"],
                    content=dream_dict["content"],
                    source=source_type
                )
                
                # Save to file
                with open(DREAMS_FILE, 'a', encoding='utf-8') as f:
                    f.write(dream.to_json() + '\n')
                dreams_saved += 1
                
            return dreams_saved
                
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {e}")
            return 0
            
    except Exception as e:
        logging.error(f"Error extracting dreams: {e}")
        return 0

def update_dream(conversation: str) -> str:
    """
    Update the dreaming system by analyzing a new conversation and 
    extracting dreams from both the conversation and a synthetic version.
    """
    if not accessed_dreams:
        logging.info("No dreams to update")
        create_result = create_dream(conversation)
        return f"No dreams accessed for update. {create_result}"
    
    # First extract new dreams from the conversation
    create_result = create_dream(conversation)
    
    result_messages = [create_result]
    
    # Then update existing dreams that were accessed
    for dream_json in accessed_dreams:
        try:
            dream = Dream.from_json(dream_json)
            dream_update_prompt = f"""Review and update this specific dream insight based on the new conversation.
Focus only on this insight and integrate any relevant new connections or patterns.

NEW CONVERSATION:
{conversation}

CURRENT INSIGHT [{dream.insight_type}]:
{dream.content}

INSTRUCTIONS:
1. Update the insight if the new conversation reveals additional depth or connections
2. Return the updated insight as a JSON object with 'insight_type' and 'content' fields
3. Only change if there's a meaningful improvement; otherwise return the original
"""

            # Use format="json"
            response = ollama.chat(
                model="huihui_ai/qwen2.5-abliterate:14b",
                messages=[{"role": "user", "content": dream_update_prompt}],
                format="json",
                options={"temperature": 0.2}
            )
            
            if response.get("message", {}).get("content"):
                try:
                    updated_data = json.loads(response["message"]["content"])
                    if "insight_type" in updated_data and "content" in updated_data:
                        # Check if it actually changed
                        if (dream.insight_type != updated_data["insight_type"] or 
                            dream.content != updated_data["content"]):
                            
                            # Update dream data while preserving source
                            updated_dream = Dream(
                                insight_type=updated_data["insight_type"],
                                content=updated_data["content"],
                                source=dream.source
                            )
                            
                            # Load all dreams
                            with open(DREAMS_FILE, 'r', encoding='utf-8') as file:
                                lines = [line.strip() for line in file if line.strip()]
                            
                            # Replace the updated dream
                            found = False
                            with open(DREAMS_FILE, 'w', encoding='utf-8') as file:
                                for line in lines:
                                    try:
                                        current = Dream.from_json(line)
                                        # If this is the one we just updated, write the new version
                                        if (current.insight_type == dream.insight_type and 
                                            current.content == dream.content and
                                            current.source == dream.source and
                                            not found):
                                            file.write(updated_dream.to_json() + '\n')
                                            found = True
                                        else:
                                            file.write(line + '\n')
                                    except Exception:
                                        # Keep any lines that can't be parsed
                                        file.write(line + '\n')
                                        
                            if found:
                                result_messages.append(f"Updated dream of type {dream.insight_type}")
                except Exception as e:
                    logging.error(f"Error updating dream: {e}")
                    
        except Exception as e:
            logging.error(f"Error processing dream update: {e}")
    
    # Clean up
    remove_result = remove_duplicate_dreams()
    result_messages.append(remove_result)
    
    consolidate_result = consolidate_dreams()
    result_messages.append(consolidate_result)
    
    # Clear accessed memories after update
    accessed_dreams.clear()
    
    return "\n".join(result_messages)

def recall_dreams(query: str, top_k: int = 1, source_filter: str = None) -> list:
    """
    Load and return the single most relevant dream based on the query.
    Always returns just the top 1 match for simplicity.
    
    Args:
        query (str): The search query to find relevant dream
        top_k (int): Always returns 1 regardless of this value (kept for backwards compatibility)
        source_filter (str, optional): Filter by dream source ("real", "synthetic", or None for any)
        
    Returns:
        list: A list containing the single most relevant dream as a JSON string, or empty list if none found
    """
    global accessed_dreams
    
    if not os.path.exists(DREAMS_FILE):
        return []
        
    try:
        # Ensure the file is in JSONL format
        try:
            import memory_format_utils
            memory_format_utils.ensure_jsonl_format(DREAMS_FILE, silent=True)
        except ImportError:
            # If memory_format_utils isn't available, continue with current format
            pass
            
        # Load and preprocess dreams
        with open(DREAMS_FILE, 'r', encoding='utf-8') as f:
            dreams = [Dream.from_json(line) for line in f if line.strip()]
            
        if not dreams:
            return []
        
        # Filter by source if requested
        if source_filter:
            dreams = [d for d in dreams if d.source == source_filter]
            if not dreams:  # If no dreams match the filter, return empty
                return []
            
        # Try to load cached embeddings
        dream_embeddings_tensor = load_cached_embeddings()
        if dream_embeddings_tensor is None or dream_embeddings_tensor.nelement() == 0:
            # Generate new embeddings in batches
            dream_embeddings_tensor = batch_embed_texts(
                [f"[{dream.insight_type}] {dream.content}" for dream in dreams]
            )
            save_embeddings_cache(dream_embeddings_tensor)

        # Get query embedding
        query_response = ollama.embeddings(model='nomic-embed-text', prompt=query)
        query_embedding = torch.tensor(query_response["embedding"], dtype=torch.float32)

        # Calculate similarities - ensure tensors are on the same device and have the same dtype
        dream_embeddings_tensor = dream_embeddings_tensor.to(dtype=torch.float32)
        similarities = torch.cosine_similarity(
            query_embedding.unsqueeze(0),
            dream_embeddings_tensor,
            dim=1
        )
        
        # Get top-1 index only
        if len(similarities) == 0:
            return []
        
        # Get the single most relevant dream
        top_index = similarities.argmax().item()
        
        # Track accessed dream for potential updates
        accessed_dreams.append(dreams[top_index].to_json())
        
        # Remove duplicates from accessed_dreams
        accessed_dreams = list(set(accessed_dreams))
        
        # Return as a list for backward compatibility
        return [dreams[top_index].to_json()]
        
    except Exception as e:
        logging.error(f"Error in recall_dreams: {e}")
        return []

def remove_duplicate_dreams() -> str:
    """
    Remove exact duplicates from Dreaming.json.
    Duplicates are lines with identical insight_type & content.
    """
    try:
        if not os.path.exists(DREAMS_FILE):
            return "No dreams file exists"

        with open(DREAMS_FILE, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file if line.strip()]

        if not lines:
            return "No dreams to process"

        # Convert JSON lines to Dream objects, track unique ones
        unique_dreams = []
        seen = set()
        for line in lines:
            try:
                dream_obj = Dream.from_json(line)
                # A simple uniqueness check uses the (insight_type, content) tuple
                dream_tuple = (dream_obj.insight_type, dream_obj.content)
                if dream_tuple not in seen:
                    seen.add(dream_tuple)
                    unique_dreams.append(dream_obj)
            except Exception as e:
                logging.error(f"Error parsing dream: {e}")
                # Skip invalid lines

        # Rewrite file with only unique dreams
        with open(DREAMS_FILE, 'w', encoding='utf-8') as file:
            for dream in unique_dreams:
                file.write(dream.to_json() + "\n")
            file.truncate()

        return f"Removed duplicates. Kept {len(unique_dreams)} unique dreams from {len(lines)} total."

    except Exception as e:
        logging.error(f"Error removing duplicates: {e}")
        return f"Error: {str(e)}"

def consolidate_dreams(similarity_threshold: float = 0.85) -> str:
    """
    Consolidate similar dreams based on a similarity threshold.
    1. Remove exact duplicates first.
    2. Embed remaining dreams.
    3. Merge any pairs above the threshold, rewriting the file.
    """
    try:
        # First, remove duplicates
        remove_msg = remove_duplicate_dreams()
        logging.info(remove_msg)
        
        # Load remaining dreams after deduplication
        with open(DREAMS_FILE, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file if line.strip()]

        # Parse dreams from JSONL
        dreams = []
        for line in lines:
            try:
                dreams.append(Dream.from_json(line))
            except Exception as e:
                logging.error(f"Error parsing dream: {e}")
                # Skip invalid lines

        n_dreams = len(dreams)
        if n_dreams <= 1:
            return "Not enough dreams to consolidate"

        # Create dream texts for embedding
        dream_texts = [f"[{d.insight_type}] {d.content}" for d in dreams]

        # Convert to embeddings in batches
        embeddings = batch_embed_texts(dream_texts)

        # Track which dreams to merge
        merged_indices = set()

        # Calculate pairwise similarities
        for i in range(n_dreams):
            if i in merged_indices:
                continue

            for j in range(i + 1, n_dreams):
                if j in merged_indices:
                    continue

                # Calculate similarity between dreams
                similarity = torch.cosine_similarity(
                    embeddings[i].unsqueeze(0),
                    embeddings[j].unsqueeze(0),
                    dim=1
                ).item()

                if similarity >= similarity_threshold:
                    merged_indices.add(j)

        # Rebuild the file with merges applied
        final_dreams = [dreams[idx] for idx in range(n_dreams) if idx not in merged_indices]

        # Open file in write mode and truncate afterward
        with open(DREAMS_FILE, 'w+', encoding='utf-8') as file:
            for dream in final_dreams:
                file.write(dream.to_json() + '\n')
            file.truncate()  # Truncate leftover data if any

        # Invalidate cache since we've modified the file
        if EMBEDDINGS_CACHE.exists():
            save_embeddings_cache([])

        return f"Consolidation complete. Original: {n_dreams} dreams, merged down to {len(final_dreams)}."

    except Exception as e:
        logging.error(f"Error consolidating dreams: {e}")
        return f"Error: {str(e)}"

def process_new_conversation(conversation_path: str) -> int:
    """
    Process a newly saved conversation to extract dreams.
    This function should be called after a conversation is saved.
    """
    try:
        # Read the new conversation
        with open(conversation_path, 'r', encoding='utf-8') as f:
            conversation_data = json.load(f)
            
        # Convert the conversation data to text format
        conversation_text = ""
        for message in conversation_data:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            conversation_text += f"{role}: {content}\n\n"
        
        # Create dreams from the conversation
        create_dream(conversation_text)
        
        # Count the total number of dreams in the dreams file
        try:
            with open(DREAMS_FILE, 'r', encoding='utf-8') as f:
                dream_count = sum(1 for line in f if line.strip())
            return dream_count
        except Exception as e:
            logging.error(f"Error counting dreams: {e}")
            return 0
        
    except Exception as e:
        logging.error(f"Error processing new conversation: {e}")
        return 0
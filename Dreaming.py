import logging
import os
import json
from typing import List, Optional
from pathlib import Path
from datetime import datetime
import re

import ollama
from pydantic import BaseModel
import torch

logging.basicConfig(level=logging.ERROR)

CACHE_DIR = Path("cache")
EMBEDDINGS_CACHE = CACHE_DIR / "dreaming_embeddings.pt"  # PyTorch tensor cache
DREAMS_FILE = "Dreaming.json"  # Processed dreams storage
accessed_dreams = []

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
        return cls.model_validate_json(json_str)

def initialize_cache():
    """Initialize cache directory and files."""
    CACHE_DIR.mkdir(exist_ok=True)
    if not EMBEDDINGS_CACHE.exists():
        torch.save({"embeddings": [], "last_modified": 0}, EMBEDDINGS_CACHE)
    if not os.path.exists(DREAMS_FILE):
        with open(DREAMS_FILE, 'w', encoding='utf-8') as f:
            pass
    
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
    """
    Load embeddings from cache if they're up to date.
    - Use weights_only=True to address future security warnings in torch.load().
    """
    dreaming_modified = os.path.getmtime(DREAMS_FILE)
    try:
        cache = torch.load(EMBEDDINGS_CACHE, map_location='cpu', weights_only=True)
        if cache["last_modified"] >= dreaming_modified:
            if isinstance(cache["embeddings"], list):
                return torch.tensor(cache["embeddings"])
            return cache["embeddings"]
    except Exception as e:
        logging.warning(f"Error loading dreaming cache: {e}")
    return None

def save_embeddings_cache(embeddings):
    """Save embeddings to cache."""
    try:
        cache_data = {
            "embeddings": embeddings,
            "last_modified": os.path.getmtime(DREAMS_FILE)
        }
        torch.save(cache_data, EMBEDDINGS_CACHE)
    except Exception as e:
        logging.error(f"Error saving dreaming cache: {e}")

def batch_embed_texts(texts, batch_size=5):
    """Embed multiple texts in batches for efficiency."""
    all_embeddings = []

    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []

            for text in batch:
                response = ollama.embeddings(model='nomic-embed-text', prompt=text)
                embedding = response["embedding"]
                batch_embeddings.append(embedding)

            all_embeddings.extend(batch_embeddings)

        return torch.tensor(all_embeddings)
    except Exception as e:
        logging.error(f"Error in batch_embed_texts: {e}")
        return torch.tensor([])  # Return empty tensor on error

def read_conversations(directory: str = 'conversation_history') -> List[str]:
    """Read and filter conversation history files, sorted by most recent first."""
    logging.debug(f"Reading conversations from {directory}")
    conversations = []
    if not os.path.exists(directory):
        logging.warning(f"Directory {directory} does not exist")
        return conversations
    
    # Get all json files sorted by modification time (newest first)
    files = sorted(
        Path(directory).glob('*.json'),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    for filename in files:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                filtered = [msg for msg in data if msg.get('content')]
                if filtered:
                    conversations.append(json.dumps(filtered))
                    logging.debug(f"Read conversation from {filename}")
        except Exception as e:
            logging.error(f"Error reading {filename}: {e}")
    return conversations

def generate_synthetic_conversation(conversation: str) -> Optional[str]:
    """
    Generate a synthetic conversation based on a real one.
    
    This function takes an existing conversation and creates a new, synthetically 
    generated conversation that explores deeper themes and connections.
    
    Args:
        conversation: Text of the original conversation
        
    Returns:
        A new synthetic conversation as text, or None if generation failed
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
            stream=False,
            options={"temperature": 1.2}
        )
        result = response['message']['content']
        logging.debug(f"Generated synthetic conversation: {result[:100]}...")
        
        # Check if we got a valid result
        if result and len(result) > 50:  # Arbitrary but reasonable minimum size
            return result
        else:
            logging.warning("Generated conversation too short or empty")
            return None
    except Exception as e:
        logging.error(f"Error generating conversation: {e}")
        return None

def clean_ollama_response(text: str) -> str:
    """
    Clean up the response from Ollama by removing markdown code blocks and other formatting.
    
    Args:
        text: The text response from Ollama
        
    Returns:
        Cleaned text
    """
    # Remove markdown code blocks
    text = re.sub(r'```(?:json)?\n', '', text)
    text = re.sub(r'```', '', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def extract_dreams(conversation: str, is_synthetic: bool = False) -> str:
    """
    Extract and validate dreams from conversation.
    
    Args:
        conversation: The conversation text to analyze
        is_synthetic: Whether this is a synthetic conversation (affects prompt)
    
    Returns:
        String containing newline-separated JSON objects representing dreams
    """
    logging.debug("Extracting dreams")
    
    # Determine the appropriate system prompt based on conversation type
    system_prompt = "You are DreamWeaver, an AI that identifies insights, patterns, and creative connections."
    
    if is_synthetic:
        # For synthetic conversations, extract the deeper themes
        extraction_prompt = f"""
{system_prompt}

Extract 2-4 key insights from this synthetic conversation. Each insight should reveal patterns, connections, 
or perspectives that weren't explicitly stated. Format each as JSON with fields:
- insight_type: "pattern", "connection", "perspective", or "question"
- content: detailed description of the insight (1-3 sentences)

Return only valid JSON objects, one per line, nothing else.

CONVERSATION:
{conversation}

INSIGHTS:"""
    else:
        # For real conversations, focus on practical insights from actual exchanges
        extraction_prompt = f"""
{system_prompt}

Extract 1-3 key insights from this conversation. Each insight should reveal something valuable 
about the exchange that wasn't explicitly stated. Format each as JSON with fields:
- insight_type: "pattern", "connection", "perspective", or "question"
- content: detailed description of the insight (1-3 sentences)

Return only valid JSON objects, one per line, nothing else.

CONVERSATION:
{conversation}

INSIGHTS:"""

    # Get the response from the model
    try:
        response = ollama.chat(
            model="huihui_ai/qwen2.5-abliterate:14b",
            messages=[{"role": "user", "content": extraction_prompt}],
            stream=False,
            options={"temperature": 0.7}
        )
        result = response['message']['content']
        logging.debug(f"Extracted potential dreams (raw): {result[:100]}...")
    except Exception as e:
        logging.error(f"Error extracting dreams: {e}")
        return ""

    # Clean up the response to remove any markdown code blocks
    result = clean_ollama_response(result)
    
    # Validate each dream and set the source property based on is_synthetic
    validated_dreams = []
    
    # Split by lines and look for JSON objects
    for line in result.split('\n'):
        line = line.strip()
        if not line or line.startswith('```') or line.endswith('```'):
            continue
            
        try:
            # Try to parse the line as JSON
            dream_data = json.loads(line)
            
            # Validate required fields
            if 'insight_type' not in dream_data or 'content' not in dream_data:
                logging.warning(f"Skipping invalid dream - missing required fields: {line}")
                continue
                
            # Set the source based on the is_synthetic parameter
            dream_data['source'] = "synthetic" if is_synthetic else "real"
            
            # Create and validate Dream object
            dream = Dream(**dream_data)
            validated_dreams.append(dream.to_json())
            
        except json.JSONDecodeError:
            logging.warning(f"Failed to parse JSON from line: {line}")
        except Exception as e:
            logging.warning(f"Error validating dream: {e} - {line}")
    
    if not validated_dreams:
        logging.warning("No valid dreams extracted")
        return ""
        
    return "\n".join(validated_dreams)

def create_dream() -> None:
    """
    Read recent real conversations, generate synthetic ones, 
    extract dreams from both, and save them to file.
    """
    # Create timestamped file for saving generated conversations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dream_file = f"dreams_{timestamp}.txt"
    
    # Track total number of dreams created
    dreams_created = 0
    
    # Get recent conversations to analyze
    recent_conversations = read_conversations()
    
    # Process each conversation
    for i, conversation in enumerate(recent_conversations):
        try:
            logging.info(f"Processing conversation {i+1}/{len(recent_conversations)}")
            
            # 1. Extract dreams directly from the real conversation
            logging.info("Extracting dreams from real conversation")
            real_dreams = extract_dreams(conversation, is_synthetic=False)
            if real_dreams:
                dreams_saved = 0
                # Save dreams extracted from real conversation
                for dream_line in real_dreams.strip().split('\n'):
                    if dream_line.strip():
                        try:
                            dream_obj = json.loads(dream_line)
                            with open(DREAMS_FILE, 'a', encoding='utf-8') as f:
                                f.write(json.dumps(dream_obj) + '\n')
                            dreams_saved += 1
                        except json.JSONDecodeError as e:
                            logging.error(f"Error parsing dream from real conversation: {e}")
                logging.info(f"Saved {dreams_saved} dreams from real conversation")
                dreams_created += dreams_saved
            
            # 2. Generate a synthetic conversation
            logging.info("Generating synthetic conversation")
            synthetic = generate_synthetic_conversation(conversation)
            if not synthetic:
                logging.warning("Failed to generate synthetic conversation, continuing to next")
                continue
                
            # Save synthetic conversation for reference
            with open(dream_file, 'a', encoding='utf-8') as f:
                f.write(f"ORIGINAL: {conversation[:100]}...\n")
                f.write(f"SYNTHETIC: {synthetic}\n\n")
            
            # 3. Extract dreams from the synthetic conversation
            logging.info("Extracting dreams from synthetic conversation")
            synthetic_dreams = extract_dreams(synthetic, is_synthetic=True)
            if synthetic_dreams:
                dreams_saved = 0
                # Save dreams extracted from synthetic conversation
                for dream_line in synthetic_dreams.strip().split('\n'):
                    if dream_line.strip():
                        try:
                            dream_obj = json.loads(dream_line)
                            with open(DREAMS_FILE, 'a', encoding='utf-8') as f:
                                f.write(json.dumps(dream_obj) + '\n')
                            dreams_saved += 1
                        except json.JSONDecodeError as e:
                            logging.error(f"Error parsing dream from synthetic conversation: {e}")
                logging.info(f"Saved {dreams_saved} dreams from synthetic conversation")
                dreams_created += dreams_saved
                    
        except Exception as e:
            logging.error(f"Error processing conversation {i+1}: {e}")
            # Continue with other conversations even if one fails
    
    # After creating dreams, remove duplicates
    logging.info(f"Created {dreams_created} dreams in total, now removing duplicates")
    remove_duplicate_dreams()
    logging.info("Dream creation completed successfully")

def recall_dreams(query: str, top_k: int = 2, source_filter: str = None) -> list:
    """
    Load and return relevant dreams based on the query.
    
    Args:
        query: The text query to match against dreams
        top_k: Maximum number of dreams to return
        source_filter: Filter dreams by source ("real", "synthetic", or None for all)
    
    Returns:
        list: List of dreams as JSON strings
    """
    if not os.path.exists(DREAMS_FILE):
        return []
    try:    
        # Before loading, ensure the file is in JSONL format
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
        query_embedding = torch.tensor(query_response["embedding"])

        # Calculate similarities
        similarities = torch.cosine_similarity(
            query_embedding.unsqueeze(0),
            dream_embeddings_tensor,
            dim=1
        )
        
        # Get top-k indices
        if len(similarities) <= top_k:
            top_k_indices = list(range(len(similarities)))
        else:
            top_k_indices = similarities.argsort(descending=True)[:top_k].tolist()
        
        # Ensure we don't have an index out of range error    
        top_k_indices = [idx for idx in top_k_indices if idx < len(dreams)]
        
        relevant_dreams = [dreams[idx].to_json() for idx in top_k_indices]
        return relevant_dreams
        
    except Exception as e:
        logging.error(f"Error in recall_dreams: {e}")
        return []
    
def update_dreaming():
    """Update the dreaming system by consolidating dreams."""
    consolidate_dreams()

def remove_duplicate_dreams():
    """
    Remove exact duplicates from Dreaming.json.
    Duplicates are lines with identical insight_type & content.
    """
    try:
        if not os.path.exists(DREAMS_FILE):
            return "No dreams file exists"

        # Before processing, ensure the file is in JSONL format
        try:
            import memory_format_utils
            memory_format_utils.ensure_jsonl_format(DREAMS_FILE, silent=True)
        except ImportError:
            # If memory_format_utils isn't available, continue with current format
            pass

        with open(DREAMS_FILE, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file if line.strip()]

        if not lines:
            return "No dreams to process"

        # Create backup
        backup_file = f"{DREAMS_FILE}.bak"
        with open(backup_file, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')
        
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

        # Rewrite file with only unique dreams in JSONL format
        with open(DREAMS_FILE, 'w', encoding='utf-8') as file:
            for dream in unique_dreams:
                file.write(dream.to_json() + '\n')

        logging.info(f"Removed {len(lines) - len(unique_dreams)} duplicate dreams.")
        return f"Removed {len(lines) - len(unique_dreams)} duplicate dreams."

    except Exception as e:
        logging.error(f"Error removing duplicates: {e}")
        return f"Error: {str(e)}"

def consolidate_dreams(similarity_threshold: float = 0.80) -> str:
    """
    Combine similar dreams using a similarity threshold.
    This helps reduce redundancy while preserving unique insights.
    """
    try:
        if not os.path.exists(DREAMS_FILE):
            return "No dreams file exists"

        # Before processing, ensure the file is in JSONL format
        try:
            import memory_format_utils
            memory_format_utils.ensure_jsonl_format(DREAMS_FILE, silent=True)
        except ImportError:
            # If memory_format_utils isn't available, continue with current format
            pass

        with open(DREAMS_FILE, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file if line.strip()]

        if not lines:
            return "No dreams to process"

        # Create backup
        backup_file = f"{DREAMS_FILE}.bak"
        with open(backup_file, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')

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

        # Step 1: Calculate pairwise similarities
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

        # Step 3: Rebuild the file with merges applied
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

def recall_dreams_hybrid(query: str, top_k: int = 3, real_ratio: float = 0.5) -> list:
    """
    Load and return relevant dreams with a blend of real and synthetic origins.
    
    Args:
        query: The text query to match against dreams
        top_k: Maximum number of dreams to return
        real_ratio: Ratio of real dreams to include (0.0-1.0)
            0.0 = all synthetic, 1.0 = all real, 0.5 = equal mix
    
    Returns:
        list: List of dreams as JSON strings, blended from both sources
    """
    if not os.path.exists(DREAMS_FILE):
        logging.warning(f"Dreams file {DREAMS_FILE} not found")
        return []
        
    # Ensure real_ratio is within bounds
    real_ratio = max(0.0, min(1.0, real_ratio))
        
    try:
        # Calculate how many of each type to retrieve
        real_count = max(1, int(top_k * real_ratio))
        synthetic_count = max(1, top_k - real_count)
        
        # Get dreams from both sources
        try:
            real_dreams = recall_dreams(query, top_k=real_count, source_filter="real")
        except Exception as e:
            logging.error(f"Error recalling real dreams: {e}")
            real_dreams = []
            
        try:
            synthetic_dreams = recall_dreams(query, top_k=synthetic_count, source_filter="synthetic")
        except Exception as e:
            logging.error(f"Error recalling synthetic dreams: {e}")
            synthetic_dreams = []
        
        # If we got no dreams of either type, try getting any dreams regardless of source
        if not real_dreams and not synthetic_dreams:
            logging.warning("No dreams found with specified sources, trying without source filter")
            return recall_dreams(query, top_k=top_k)
        
        # Combine and return
        combined_dreams = []
        
        # Interleave for better mixing (if we have both types)
        if real_dreams and synthetic_dreams:
            # Start with real for more grounded first response
            for i in range(max(len(real_dreams), len(synthetic_dreams))):
                if i < len(real_dreams):
                    combined_dreams.append(real_dreams[i])
                if i < len(synthetic_dreams):
                    combined_dreams.append(synthetic_dreams[i])
        else:
            # Just append whatever we have
            combined_dreams = real_dreams + synthetic_dreams
            
        return combined_dreams[:top_k]  # Ensure we don't exceed requested count
        
    except Exception as e:
        logging.error(f"Error in recall_dreams_hybrid: {e}")
        # Fallback to regular recall if hybrid fails
        try:
            return recall_dreams(query, top_k=top_k)
        except Exception as e2:
            logging.error(f"Fallback also failed: {e2}")
            return []

if __name__ == "__main__":
    update_dreaming()

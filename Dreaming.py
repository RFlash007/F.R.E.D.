import logging
import os
import json
from typing import List, Optional
from pathlib import Path
from datetime import datetime

import ollama
from pydantic import BaseModel
import torch

logging.basicConfig(level=logging.ERROR)

CACHE_DIR = Path("cache")
EMBEDDINGS_CACHE = CACHE_DIR / "assumptions_embeddings.pt"  # PyTorch tensor cache
FACTS_FILE = "Assumptions.json"  # Processed facts storage
accessed_memories = []

class Fact(BaseModel):
    """
    Represents a single factual entry with a category and content.
    """
    category: str
    content: str
    
    def to_json(self) -> str:
        """
        Convert fact to JSON format.
        - Replaced .dict() with .model_dump() for Pydantic v2.
        """
        return json.dumps(self.model_dump())
    
    @classmethod
    def from_json(cls, json_str: str) -> "Fact":
        """
        Create a Fact from a JSON string.
        - Replaced parse_raw() with model_validate_json() for Pydantic v2.
        """
        return cls.model_validate_json(json_str)

def initialize_cache():
    """Initialize cache directory and files."""
    CACHE_DIR.mkdir(exist_ok=True)
    if not EMBEDDINGS_CACHE.exists():
        torch.save({"embeddings": [], "last_modified": 0}, EMBEDDINGS_CACHE)
    if not os.path.exists(FACTS_FILE):
        with open(FACTS_FILE, 'w', encoding='utf-8') as f:
            pass
    
    # Direct cache population
    try:
        if os.path.getsize(FACTS_FILE) > 0:
            facts = [Fact.from_json(line) for line in open(FACTS_FILE) if line.strip()]
            fact_texts = [f"[{f.category}] {f.content}" for f in facts]
            embeddings = batch_embed_texts(fact_texts)
            save_embeddings_cache(embeddings)
    except Exception as e:
        logging.error(f"Cache initialization failed: {e}")

def load_cached_embeddings():
    """
    Load embeddings from cache if they're up to date.
    - Use weights_only=True to address future security warnings in torch.load().
    """
    assumptions_modified = os.path.getmtime(FACTS_FILE)
    try:
        cache = torch.load(EMBEDDINGS_CACHE, map_location='cpu', weights_only=True)
        if cache["last_modified"] >= assumptions_modified:
            if isinstance(cache["embeddings"], list):
                return torch.tensor(cache["embeddings"])
            return cache["embeddings"]
    except Exception as e:
        logging.warning(f"Error loading assumptions cache: {e}")
    return None

def save_embeddings_cache(embeddings):
    """Save embeddings to cache."""
    try:
        cache_data = {
            "embeddings": embeddings,
            "last_modified": os.path.getmtime(FACTS_FILE)
        }
        torch.save(cache_data, EMBEDDINGS_CACHE)
    except Exception as e:
        logging.error(f"Error saving assumptions cache: {e}")

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
    """Generate synthetic conversation with personality insights."""
    logging.debug("Generating synthetic conversation")
    prompt = f"""Based on this conversation, create a NEW, DIFFERENT conversation between the same people that reveals their personality traits, preferences, and habits. Make it natural and engaging, focusing on different topics than the original. Format as plain text dialogue.

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
        return result
    except Exception as e:
        logging.error(f"Error generating conversation: {e}")
        return None

def extract_facts(conversation: str) -> str:
    """Extract and validate facts from conversation."""
    logging.debug("Extracting facts")
    prompt = f"""Extract key personality traits, preferences, and behavioral patterns as a JSON array. ##ONLY WRITE THE FACTS ABOUT IAN, NOT THE ASSISTANT##.
Each fact should have:
- category: Personality|Preference|Habit|Interest|Motivation
- content: Clear, specific insight

Example: [{{"category": "Personality", "content": "Shows enthusiasm for technical challenges"}}, {{"category": "Interest", "content": "Passionate about cryptocurrency and financial technology"}}]

From this conversation: {conversation}

***RETURN ONLY THE JSON ARRAY NO EXTRA TEXT***:"""

    try:
        response = ollama.chat(
            model="huihui_ai/qwen2.5-abliterate:14b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.5}
        )
        
        content = response['message']['content'].strip()
        # Parse JSON and validate with Pydantic
        facts = json.loads(content)
        validated_facts = [Fact(**fact) for fact in facts]
        result = '\n'.join(fact.model_dump_json() for fact in validated_facts)
        
        logging.debug(f"Extracted {len(validated_facts)} facts")
        return result
    except Exception as e:
        logging.error(f"Error extracting facts: {e}")
        return ""

def create_dream() -> None:
    """Generate and save synthetic conversations and facts."""
    logging.info("Starting dream creation")
    dreams_dir = Path('dreams')
    dreams_dir.mkdir(exist_ok=True)
    
    conversations = read_conversations()
    logging.info(f"Found {len(conversations)} conversations")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dream_file = dreams_dir / f"dream_{timestamp}.txt"
    facts_file = "Assumptions.json"
    
    for conversation in conversations:
        try:
            synthetic = generate_synthetic_conversation(conversation)
            if not synthetic:
                continue
                
            # Save synthetic conversation
            with open(dream_file, 'a', encoding='utf-8') as f:
                f.write(f"{synthetic}\n")
            
            # Extract and save facts
            facts = extract_facts(synthetic)
            if facts:
                with open(facts_file, 'a', encoding='utf-8') as f:
                    f.write(f"{facts}\n")
        except Exception as e:
            logging.error(f"Error processing conversation: {e}")

def recall_assumptions(query: str, top_k: int = 2) -> list:
    """
    Load and return relevant assumptions.
    """
    facts_file = "Assumptions.json"
    if not os.path.exists(facts_file):
        return []
    try:    
        #Load and preprocess facts
        with open(facts_file, 'r', encoding='utf-8') as f:
            facts = [Fact.from_json(line) for line in f if line.strip()]

        #Try to load cached embeddings
        fact_embeddings_tensor = load_cached_embeddings()
        if fact_embeddings_tensor is None or fact_embeddings_tensor.nelement() == 0:
            #Generate new embeddings in batches
            fact_embeddings_tensor = batch_embed_texts(
                [f"[{fact.category}] {fact.content}" for fact in facts]
            )
            save_embeddings_cache(fact_embeddings_tensor)

        #Get query embedding
        query_response = ollama.embeddings(model='nomic-embed-text', prompt=query)
        query_embedding = torch.tensor(query_response["embedding"])

        #Calculate similarities
        similarities = torch.cosine_similarity(
            query_embedding.unsqueeze(0),
            fact_embeddings_tensor
        )

        #Get top-k most relevant facts
        top_k = min(top_k, len(similarities))
        top_k_indices = torch.topk(similarities, top_k).indices
        relevant_facts = [facts[idx].to_json() for idx in top_k_indices]

        #Track accessed facts checking for duplicates
        for fact in relevant_facts:
            if fact not in accessed_memories:
                accessed_memories.append(fact)

        #Return top k most similar facts
        return relevant_facts
    except Exception as e:
        logging.error(f"Error reading assumptions: {e}")
        return []
    
def update_assumptions():
    consolidate_assumptions()

def remove_duplicate_assumptions():
    """
    Remove exact duplicates from Semantic.json.
    Duplicates are lines with identical category & content.
    """
    try:
        if not os.path.exists(FACTS_FILE):
            return "No facts file exists"

        with open(FACTS_FILE, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file if line.strip()]

        if not lines:
            return "No facts to process"

        # Convert JSON lines to Fact objects, track unique ones
        unique_facts = []
        seen = set()
        for line in lines:
            fact_obj = Fact.from_json(line)
            # A simple uniqueness check uses the (category, content) tuple
            fact_tuple = (fact_obj.category, fact_obj.content)
            if fact_tuple not in seen:
                seen.add(fact_tuple)
                unique_facts.append(fact_obj)

        # Rewrite file with only unique facts
        with open(FACTS_FILE, 'w', encoding='utf-8') as file:
            for fact in unique_facts:
                file.write(fact.to_json() + "\n")
            file.truncate()

        return f"Removed duplicates. Kept {len(unique_facts)} unique facts."

    except Exception as e:
        logging.error(f"Error removing duplicates: {e}")
        return f"Error: {str(e)}"


def consolidate_assumptions(similarity_threshold: float = 0.80) -> str:
    """
    Consolidate similar semantic facts based on a similarity threshold.
    1. Remove exact duplicates first.
    2. Embed remaining facts.
    3. Merge any pairs above the threshold, rewriting the file.
    """
    try:
        # First, remove duplicates
        remove_msg = remove_duplicate_assumptions()
        logging.info(remove_msg)

        # Load remaining facts
        with open(FACTS_FILE, 'r', encoding='utf-8') as file:
            facts = [Fact.from_json(line) for line in file if line.strip()]

        if not facts:
            return "No facts to consolidate."

        # Step 1: Get embeddings for all facts
        # Reuse 'batch_embed_texts' from your existing code
        fact_texts = [f"[{fact.category}] {fact.content}" for fact in facts]
        embeddings_tensor = batch_embed_texts(fact_texts, batch_size=5)
        if embeddings_tensor.nelement() == 0:
            return "No valid embeddings. Aborting."

        # We'll keep track of which facts are 'merged'
        merged_indices = set()  # if an index is merged, we won't rewrite it individually

        # Step 2: Compare each pair (i, j) only once, i < j
        n_facts = len(facts)
        for i in range(n_facts):
            if i in merged_indices:
                continue
            for j in range(i + 1, n_facts):
                if j in merged_indices:
                    continue

                sim = float(torch.cosine_similarity(
                    embeddings_tensor[i].unsqueeze(0),
                    embeddings_tensor[j].unsqueeze(0)
                ))

                # If they exceed the threshold, we merge them
                if sim >= similarity_threshold:
                    logging.info(f"Merging facts {i} and {j} with sim={sim:.3f}")
                    # Example merge policy:
                    merged_cat = facts[i].category  # or choose whichever you like
                    merged_content = (
                        f"{facts[i].content}"
                    )
                    # Overwrite fact i
                    facts[i] = Fact(category=merged_cat, content=merged_content)
                    # Mark fact j as merged
                    merged_indices.add(j)

        # Step 3: Rebuild the file with merges applied
        # Filter out merged indices
        final_facts = [facts[idx] for idx in range(n_facts) if idx not in merged_indices]

        # Open file in write mode and truncate afterward
        with open(FACTS_FILE, 'w+', encoding='utf-8') as file:
            for fact in final_facts:
                file.write(fact.to_json() + "\n")
            file.truncate()  # Truncate leftover data if any


        # Invalidate cache since we've modified the file
        if EMBEDDINGS_CACHE.exists():
            save_embeddings_cache([])

        return f"Consolidation complete. Original: {n_facts} facts, merged down to {len(final_facts)}."

    except Exception as e:
        logging.error(f"Error consolidating semantic memories: {e}")
        return f"Error: {str(e)}"

    

if __name__ == "__main__":
    update_assumptions()
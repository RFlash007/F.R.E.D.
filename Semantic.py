import torch
import ollama
import logging
import json
import os
from pathlib import Path
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.ERROR)

# Track which facts have been accessed during the current session
accessed_memories = []

# Cache setup for storing embeddings and processed facts
CACHE_DIR = Path("cache")
EMBEDDINGS_CACHE = CACHE_DIR / "semantic_embeddings.pt"  # PyTorch tensor cache
FACTS_FILE = "Semantic.json"  # Processed facts storage

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
            pass  # Create empty file if it doesn't exist

def load_cached_embeddings():
    """
    Load embeddings from cache if they're up to date.
    - Use weights_only=True to address future security warnings in torch.load().
    """
    semantic_modified = os.path.getmtime(FACTS_FILE)
    try:
        cache = torch.load(EMBEDDINGS_CACHE, map_location='cpu', weights_only=True)
        if cache["last_modified"] >= semantic_modified:
            if isinstance(cache["embeddings"], list):
                return torch.tensor(cache["embeddings"])
            return cache["embeddings"]
    except Exception as e:
        logging.warning(f"Error loading semantic cache: {e}")
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
        logging.error(f"Error saving semantic cache: {e}")

def create_semantic(memory: str) -> str:
    """
    Extract verifiable facts from a conversation and store them in Semantic.json.
    """
    try:
        fact_extraction_prompt = f"""
        Extract only verifiable facts and knowledge from the conversation.
        Return an array of JSON objects where each object has:
        - "category": short label, e.g. "Education", "Hardware"
        - "content": the fact

        DO NOT wrap them in any other JSON keys (like "facts"). Just return the array.

        CONVERSATION:
        {memory}
        """

        # Use format="json" instead of Fact.schema_json()
        response = ollama.chat(
            model="huihui_ai/qwen2.5-abliterate:14b",
            messages=[{"role": "user", "content": fact_extraction_prompt}],
            format="json",
            options={"temperature": 0}
        )

        if not response.get("message", {}).get("content"):
            logging.warning("No content in AI response")
            return ""

        content = response["message"]["content"]

        try:
            # Parse the JSON response directly
            facts_data = json.loads(content)
            if not isinstance(facts_data, list):
                facts_data = [facts_data]

            with open(FACTS_FILE, 'a', encoding='utf-8') as file:
                for data in facts_data:
                    # Validate presence of keys
                    if "category" not in data or "content" not in data:
                        logging.error(f"AI output missing 'category' or 'content': {data}")
                        continue

                    fact = Fact(category=data["category"], content=data["content"])
                    file.write(fact.to_json() + "\n")

        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON: {e}")
            return ""
        except Exception as e:
            logging.error(f"Error processing fact: {e}")
            return ""

        # Invalidate cache so next recall triggers re-embedding
        if EMBEDDINGS_CACHE.exists():
            save_embeddings_cache([])

        return content

    except Exception as e:
        logging.error(f"Error in create_semantic: {e}")
        return ""

def update_semantic(conversation: str) -> str:
    """
    Update previously accessed facts based on new conversation input.
    """
    try:
        if not accessed_memories:
            logging.info("No facts to update")
            return "No facts accessed for update"

        updated_facts = []

        # Load existing facts
        with open(FACTS_FILE, 'r', encoding='utf-8') as file:
            existing_facts = [Fact.from_json(line) for line in file if line.strip()]

        # Process each accessed fact individually
        for fact_data in accessed_memories:
            fact = Fact.from_json(fact_data)
            fact_update_prompt = f"""Review and update this specific fact based on the new conversation json summary.
Focus only on this fact and integrate any relevant new information.

NEW CONVERSATION:
{conversation}

CURRENT FACT:
[{fact.category}] {fact.content}

INSTRUCTIONS:
1. Update the fact if new information adds value
2. Remove redundant information, summarizing where it's possible.
2. Return the updated fact as a JSON object with 'category' and 'content' fields
"""

            # Use format="json"
            response = ollama.chat(
                model="huihui_ai/qwen2.5-abliterate:14b",
                messages=[{"role": "user", "content": fact_update_prompt}],
                format="json",
                options={"temperature": 0}
            )

            if response.get("message", {}).get("content"):
                try:
                    new_data = json.loads(response["message"]["content"])
                    if "category" not in new_data or "content" not in new_data:
                        logging.warning(f"AI output missing keys. Keeping original: {new_data}")
                        updated_facts.append(fact)
                    else:
                        updated_fact = Fact(
                            category=new_data["category"],
                            content=new_data["content"]
                        )
                        updated_facts.append(updated_fact)
                except json.JSONDecodeError as e:
                    logging.error(f"JSON decode error for updated fact: {e}")
                    updated_facts.append(fact)
            else:
                logging.warning(f"Failed to update fact: {fact}")
                updated_facts.append(fact)

        # Update file with preserved (unchanged) + updated facts
        preserved_facts = [
            fact for fact in existing_facts
            if fact.to_json() not in accessed_memories
        ]
        with open(FACTS_FILE, 'w', encoding='utf-8') as file:
            for fact in preserved_facts + updated_facts:
                file.write(fact.to_json() + "\n")

        try:
            consolidate_semantic()
        except Exception as e:
            logging.error(f"Error consolidating semantic data: {e}")

        # Invalidate cache
        if EMBEDDINGS_CACHE.exists():
            save_embeddings_cache([])

        accessed_memories.clear()  # Clear accessed memories after update
        return f"Successfully updated {len(updated_facts)} facts"

    except Exception as e:
        logging.error(f"Error updating semantic memory: {e}")
        return f"Error: {str(e)}"

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
        logging.error(f"Error in batch_embed_texts: {str(e)}")
        return torch.tensor([])  # Return empty tensor on error

def recall_semantic(query: str, top_k: int = 2) -> list:
    """
    Retrieves relevant facts based on a query using semantic search.
    """
    if not os.path.exists(FACTS_FILE):
        return []

    try:
        initialize_cache()

        # Load and preprocess facts
        with open(FACTS_FILE, 'r', encoding='utf-8') as file:
            facts = [Fact.from_json(line) for line in file if line.strip()]

        if not facts:
            return []

        # Try to load cached embeddings
        fact_embeddings_tensor = load_cached_embeddings()

        if fact_embeddings_tensor is None or fact_embeddings_tensor.nelement() == 0:
            # Generate new embeddings in batches
            fact_embeddings_tensor = batch_embed_texts(
                [f"[{fact.category}] {fact.content}" for fact in facts]
            )
            save_embeddings_cache(fact_embeddings_tensor)

        # Get query embedding
        query_response = ollama.embeddings(model='nomic-embed-text', prompt=query)
        query_embedding = torch.tensor(query_response["embedding"])

        # Calculate similarities
        similarities = torch.cosine_similarity(
            query_embedding.unsqueeze(0),
            fact_embeddings_tensor
        )

        # Get top-k most relevant facts
        top_k = min(top_k, len(similarities))
        top_k_indices = torch.topk(similarities, top_k).indices
        relevant_facts = [facts[idx].to_json() for idx in top_k_indices]

        # Track accessed facts checking for duplicates
        for fact in relevant_facts:
            if fact not in accessed_memories:
                accessed_memories.append(fact)

        return relevant_facts

    except Exception as e:
        logging.error(f"Error in recall_semantic: {str(e)}")
        return []
    


def remove_duplicate_semantic():
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


def consolidate_semantic(similarity_threshold: float = 0.96) -> str:
    """
    Consolidate similar semantic facts based on a similarity threshold.
    1. Remove exact duplicates first.
    2. Embed remaining facts.
    3. Merge any pairs above the threshold, rewriting the file.
    """
    try:
        # First, remove duplicates
        remove_msg = remove_duplicate_semantic()
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
            for fact in final_facts:  # or unique_facts
                file.write(fact.to_json() + "\n")
            file.truncate()  # Truncate leftover data if any


        # Invalidate cache since we've modified the file
        if EMBEDDINGS_CACHE.exists():
            save_embeddings_cache([])

        return f"Consolidation complete. Original: {n_facts} facts, merged down to {len(final_facts)}."

    except Exception as e:
        logging.error(f"Error consolidating semantic memories: {e}")
        return f"Error: {str(e)}"
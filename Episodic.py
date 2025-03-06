import torch
import ollama
import logging
import json
import os
from pathlib import Path
from pydantic import BaseModel
import Tools

# Configure logging
logging.basicConfig(level=logging.ERROR)

# Track which episodes have been accessed during the current session
accessed_episodes = []

# Cache setup for storing embeddings and processed episodes
CACHE_DIR = Path("cache")
EMBEDDINGS_CACHE = CACHE_DIR / "episodic_embeddings.pt"  # PyTorch tensor cache
EPISODES_FILE = "Episodic.json"  # Processed episodes storage

class Episode(BaseModel):
    """
    Represents an episodic memory entry in JSON with the new structure:
      - memory_timestamp
      - context_tags
      - conversation_summary
      - what_worked
      - what_to_avoid
      - what_you_learned
    """
    memory_timestamp: str
    context_tags: list[str]
    conversation_summary: str
    what_worked: str
    what_to_avoid: str
    what_you_learned: str

    def to_json(self) -> str:
        """
        Convert episode to JSON format.
        Use .model_dump() instead of .dict() for Pydantic v2.
        """
        return json.dumps(self.model_dump())
    
    @classmethod
    def from_json(cls, json_str: str) -> "Episode":
        """
        Create an Episode from a JSON string.
        Use .model_validate_json() instead of parse_raw().
        """
        return cls.model_validate_json(json_str)


def initialize_cache():
    """Initialize cache directory and files for episodic data."""
    CACHE_DIR.mkdir(exist_ok=True)
    if not EMBEDDINGS_CACHE.exists():
        torch.save({"embeddings": [], "last_modified": 0}, EMBEDDINGS_CACHE)
    if not os.path.exists(EPISODES_FILE):
        with open(EPISODES_FILE, 'w', encoding='utf-8') as f:
            pass
    
    # Direct cache population without dummy query
    try:
        if os.path.getsize(EPISODES_FILE) > 0:
            episodes = [Episode.from_json(line) for line in open(EPISODES_FILE) if line.strip()]
            episode_texts = [
                f"{ep.memory_timestamp}|{ep.context_tags}|{ep.conversation_summary}|"
                f"{ep.what_worked}|{ep.what_to_avoid}|{ep.what_you_learned}"
                for ep in episodes
            ]
            embeddings = batch_embed_texts(episode_texts)
            save_embeddings_cache(embeddings)
    except Exception as e:
        logging.error(f"Cache initialization failed: {e}")

def load_cached_embeddings():
    """
    Load episodic embeddings from cache if they're up to date.
    Use weights_only=True to avoid future security warnings.
    """
    episodic_modified = os.path.getmtime(EPISODES_FILE)
    try:
        cache = torch.load(EMBEDDINGS_CACHE, map_location='cpu', weights_only=True)
        if cache["last_modified"] >= episodic_modified:
            if isinstance(cache["embeddings"], list):
                return torch.tensor(cache["embeddings"])
            return cache["embeddings"]
    except Exception as e:
        logging.warning(f"Error loading episodic cache: {e}")
    return None


def save_embeddings_cache(embeddings):
    """Save episodic embeddings to cache."""
    try:
        cache_data = {
            "embeddings": embeddings,
            "last_modified": os.path.getmtime(EPISODES_FILE)
        }
        torch.save(cache_data, EMBEDDINGS_CACHE)
    except Exception as e:
        logging.error(f"Error saving episodic cache: {e}")


def create_episodic(memory: str) -> str:
    """
    Extract relevant episodic information from the conversation
    and store it in EPISODES_FILE using the new JSON structure.
    """
    try:
        time = Tools.get_time()
        # Prompt instructing the model to produce the new structure
        episode_extraction_prompt = f"""Extract a single or multiple episodes from this conversation.
TIME: {time}
CONVERSATION:
{memory}

INSTRUCTIONS:
- Return each episode as a JSON object with the following keys:
  1. "memory_timestamp"
  2. "context_tags" (array of strings)
  3. "conversation_summary"
  4. "what_worked"
  5. "what_to_avoid"
  6. "what_you_learned"
- If something doesn't apply, use "N/A" or an empty string, but do NOT omit the field.
- Return an array of these JSON objects if more than one.
"""

        response = ollama.chat(
            model="huihui_ai/qwen2.5-abliterate:14b",
            messages=[{"role": "user", "content": episode_extraction_prompt}],
            format="json",
            options={"temperature": 0}
        )

        if not response.get("message", {}).get("content"):
            logging.warning("No content in AI response")
            return ""

        content = response["message"]["content"]

        try:
            # Parse the JSON response directly
            episodes_data = json.loads(content)
            if not isinstance(episodes_data, list):
                episodes_data = [episodes_data]

            required_fields = {
                "memory_timestamp", "context_tags", "conversation_summary",
                "what_worked", "what_to_avoid", "what_you_learned"
            }

            with open(EPISODES_FILE, 'a', encoding='utf-8') as file:
                for data in episodes_data:
                    # Check that all required fields are present
                    if not required_fields.issubset(data.keys()):
                        logging.error(
                            f"Missing one or more fields from: {required_fields}\n"
                            f"Output was: {data}"
                        )
                        continue

                    # Convert any list fields to single string
                    for key in ["what_worked", "what_to_avoid", "what_you_learned"]:
                        if isinstance(data.get(key), list):
                            data[key] = "\n".join(data[key])

                    # Only create Episode if everything is present (post-conversion)
                    episode = Episode(
                        memory_timestamp=data["memory_timestamp"],
                        context_tags=data["context_tags"],  # still a list
                        conversation_summary=data["conversation_summary"],
                        what_worked=data["what_worked"],
                        what_to_avoid=data["what_to_avoid"],
                        what_you_learned=data["what_you_learned"]
                    )
                    file.write(episode.to_json() + "\n")

        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON: {e}")
            return ""
        except Exception as e:
            logging.error(f"Error processing episode: {e}")
            return ""

        # Invalidate cache
        if EMBEDDINGS_CACHE.exists():
            save_embeddings_cache([])

        return content

    except Exception as e:
        logging.error(f"Error in create_episodic: {e}")
        return ""


def update_episodic(conversation: str) -> str:
    """
    Update the episodic data in EPISODES_FILE based on a new conversation,
    only for episodes that were previously accessed in `accessed_episodes`.
    """
    try:
        if not accessed_episodes:
            logging.info("No episodes to update")
            return "No episodes accessed for update"

        updated_episodes = []

        # Load existing episodes
        with open(EPISODES_FILE, 'r', encoding='utf-8') as file:
            existing_episodes = [Episode.from_json(line) for line in file if line.strip()]

        # Process each accessed episode individually
        for episode_data in accessed_episodes:
            episode = Episode.from_json(episode_data)

            # Summarize the current fields to help the model see them
            episode_update_prompt = f"""Review and update this specific episode based on the new conversation json summary.
Focus only on this one episode; integrate any relevant new information.

NEW CONVERSATION:
{conversation}

CURRENT EPISODE:
memory_timestamp: {episode.memory_timestamp}
context_tags: {episode.context_tags}
conversation_summary: {episode.conversation_summary}
what_worked: {episode.what_worked}
what_to_avoid: {episode.what_to_avoid}
what_you_learned: {episode.what_you_learned}

INSTRUCTIONS:
1. Update the episode if new information adds value
2. Remove redundant information, summarizing where it's possible.
3. Return the updated episode as a JSON object with the same keys:
   - memory_timestamp
   - context_tags (list)
   - conversation_summary
   - what_worked
   - what_to_avoid
   - what_you_learned
"""

            response = ollama.chat(
                model="huihui_ai/qwen2.5-abliterate:14b",
                messages=[{"role": "user", "content": episode_update_prompt}],
                format="json",  # Avoid schema_json()
                options={"temperature": 0}
            )

            if response.get("message", {}).get("content"):
                try:
                    new_data = json.loads(response["message"]["content"])
                    # Make sure the updated data has all the required fields
                    required_fields = {
                        "memory_timestamp", "context_tags", "conversation_summary",
                        "what_worked", "what_to_avoid", "what_you_learned"
                    }
                    if not required_fields.issubset(new_data.keys()):
                        logging.warning(
                            f"Failed to update episode due to missing fields in AI output: {new_data}"
                        )
                        # Keep old version if update is invalid
                        updated_episodes.append(episode)
                    else:
                        # Convert lists to string if needed
                        for key in ["what_worked", "what_to_avoid", "what_you_learned"]:
                            if isinstance(new_data.get(key), list):
                                new_data[key] = "\n".join(new_data[key])

                        updated_episode = Episode(
                            memory_timestamp=new_data["memory_timestamp"],
                            context_tags=new_data["context_tags"],
                            conversation_summary=new_data["conversation_summary"],
                            what_worked=new_data["what_worked"],
                            what_to_avoid=new_data["what_to_avoid"],
                            what_you_learned=new_data["what_you_learned"]
                        )
                        updated_episodes.append(updated_episode)
                except json.JSONDecodeError as e:
                    logging.error(f"JSON decode error for updated episode: {e}")
                    updated_episodes.append(episode)
            else:
                logging.warning(f"Failed to update episode: {episode}")
                updated_episodes.append(episode)

        # Rebuild the entire file with preserved + updated episodes
        preserved_episodes = [
            ep for ep in existing_episodes
            if ep.to_json() not in accessed_episodes
        ]
        with open(EPISODES_FILE, 'w', encoding='utf-8') as file:
            for ep in preserved_episodes + updated_episodes:
                file.write(ep.to_json() + "\n")

        try:
            consolidate_episodic()
        except Exception as e:
            logging.error(f"Error consolidating episodic data: {e}")

        # Invalidate cache
        if EMBEDDINGS_CACHE.exists():
            save_embeddings_cache([])

        accessed_episodes.clear()  # Clear accessed episodes after update
        return f"Successfully updated {len(updated_episodes)} episodes"

    except Exception as e:
        logging.error(f"Error updating episodic data: {e}")
        return f"Error: {str(e)}"


def batch_embed_texts(texts, batch_size=5):
    """Embed multiple texts in batches for efficiency (episodic version)."""
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


def recall_episodic(query: str, top_k: int = 1) -> list:
    """
    Retrieves relevant episodes based on a query using semantic search.
    """
    if not os.path.exists(EPISODES_FILE):
        return []

    try:

        # Load and preprocess episodes
        with open(EPISODES_FILE, 'r', encoding='utf-8') as file:
            episodes = [Episode.from_json(line) for line in file if line.strip()]

        if not episodes:
            return []

        # Try to load cached embeddings
        episode_embeddings_tensor = load_cached_embeddings()

        if episode_embeddings_tensor is None or episode_embeddings_tensor.nelement() == 0:
            # Generate new embeddings in batches
            episode_embeddings_tensor = batch_embed_texts(
                [
                    # Convert the new structure to a text prompt
                    f"{ep.memory_timestamp} | {ep.context_tags} | {ep.conversation_summary} | "
                    f"What worked: {ep.what_worked} | Avoid: {ep.what_to_avoid} | Learned: {ep.what_you_learned}"
                    for ep in episodes
                ]
            )
            save_embeddings_cache(episode_embeddings_tensor)

        # Get query embedding
        query_response = ollama.embeddings(model='nomic-embed-text', prompt=query)
        query_embedding = torch.tensor(query_response["embedding"], dtype=torch.float32)

        # Calculate similarities - ensure tensors are on the same device and have the same dtype
        episode_embeddings_tensor = episode_embeddings_tensor.to(dtype=torch.float32)
        similarities = torch.cosine_similarity(
            query_embedding.unsqueeze(0),
            episode_embeddings_tensor,
            dim=1
        )

        # Get top-k most relevant episodes
        top_k = min(top_k, len(similarities))
        top_k_indices = torch.topk(similarities, top_k).indices
        relevant_episodes = [episodes[idx].to_json() for idx in top_k_indices]

        # Track accessed episodes
        for episode in relevant_episodes:
            if episode not in accessed_episodes:
                accessed_episodes.append(episode)

        return relevant_episodes

    except Exception as e:
        logging.error(f"Error in recall_episodic: {str(e)}")
        return []
    


def remove_duplicate_episodic():
    """
    Remove exact duplicates from Episodic.json.
    Duplicates = lines with identical 6 fields in an Episode.
    """
    try:
        if not os.path.exists(EPISODES_FILE):
            return "No episodic file exists."

        with open(EPISODES_FILE, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file if line.strip()]

        if not lines:
            return "No episodes to process."

        unique_episodes = []
        seen = set()

        for line in lines:
            episode_obj = Episode.from_json(line)

            # Identify uniqueness by the 6 fields of an Episode
            episode_tuple = (
                episode_obj.memory_timestamp,
                tuple(episode_obj.context_tags),  # must be a tuple for set() usage
                episode_obj.conversation_summary,
                episode_obj.what_worked,
                episode_obj.what_to_avoid,
                episode_obj.what_you_learned
            )

            if episode_tuple not in seen:
                seen.add(episode_tuple)
                unique_episodes.append(episode_obj)

        # Rewrite file with only unique episodes
        with open(EPISODES_FILE, 'w+', encoding='utf-8') as file:
            for ep in unique_episodes:
                file.write(ep.to_json() + "\n")
            file.truncate()

        return f"Removed duplicates. Kept {len(unique_episodes)} unique episodes."

    except Exception as e:
        logging.error(f"Error removing duplicates: {e}")
        return f"Error: {str(e)}"



def consolidate_episodic(similarity_threshold: float = 0.99) -> str:
    """
    Consolidate similar episodic memories based on a similarity threshold.
    1. Remove exact duplicates first.
    2. Embed remaining episodes.
    3. Merge any pairs above the threshold, rewriting the file.
    """

    try:
        # First, remove duplicates
        remove_msg = remove_duplicate_episodic()
        logging.info(remove_msg)

        # Load remaining episodes
        with open(EPISODES_FILE, 'r', encoding='utf-8') as file:
            episodes = [Episode.from_json(line) for line in file if line.strip()]

        if not episodes:
            return "No episodes to consolidate."

        # Step 1: Embed all episodes
        # Convert each Episode into a text prompt for embeddings
        episode_prompts = [
            (
                f"{ep.memory_timestamp} | {ep.context_tags} | {ep.conversation_summary} | "
                f"What worked: {ep.what_worked} | Avoid: {ep.what_to_avoid} | Learned: {ep.what_you_learned}"
            )
            for ep in episodes
        ]
        embeddings_tensor = batch_embed_texts(episode_prompts, batch_size=5)
        if embeddings_tensor.nelement() == 0:
            return "No valid embeddings. Aborting."

        merged_indices = set()  # track which episodes get merged out
        n_episodes = len(episodes)

        # Step 2: Compare each pair (i, j) only once, i < j
        for i in range(n_episodes):
            if i in merged_indices:
                continue
            for j in range(i + 1, n_episodes):
                if j in merged_indices:
                    continue

                sim = float(torch.cosine_similarity(
                    embeddings_tensor[i].unsqueeze(0),
                    embeddings_tensor[j].unsqueeze(0)
                ))

                if sim >= similarity_threshold:
                    logging.info(f"Merging episodes {i} and {j} with sim={sim:.3f}")

                    # Example merge policy:
                    # 1) Keep earliest memory_timestamp (e.g., i).
                    # 2) Combine context_tags from both.
                    # 3) Combine the textual fields with line breaks.
                    new_timestamp = episodes[i].memory_timestamp
                    combined_tags = list(set(episodes[i].context_tags + episodes[j].context_tags))
                    new_conversation_summary = (
                        f"{episodes[i].conversation_summary}"
                    )
                    new_what_worked = (
                        f"{episodes[i].what_worked}"
                    )
                    new_what_to_avoid = (
                        f"{episodes[i].what_to_avoid}"
                    )
                    new_what_you_learned = (
                        f"{episodes[i].what_you_learned}"
                    )

                    # Overwrite i
                    episodes[i] = Episode(
                        memory_timestamp=new_timestamp,
                        context_tags=combined_tags,
                        conversation_summary=new_conversation_summary,
                        what_worked=new_what_worked,
                        what_to_avoid=new_what_to_avoid,
                        what_you_learned=new_what_you_learned
                    )
                    # Mark j as merged
                    merged_indices.add(j)

        # Step 3: Rebuild the file with merges applied
        final_episodes = [episodes[idx] for idx in range(n_episodes) if idx not in merged_indices]

        with open(EPISODES_FILE, 'w+', encoding='utf-8') as file:
            for ep in final_episodes:
                file.write(ep.to_json() + "\n")
            file.truncate()

        # Invalidate cache
        if EMBEDDINGS_CACHE.exists():
            save_embeddings_cache([])

        return (f"Consolidation complete. Original: {n_episodes} episodes, "
                f"merged down to {len(final_episodes)}.")

    except Exception as e:
        logging.error(f"Error consolidating episodic memories: {e}")
        return f"Error: {str(e)}"
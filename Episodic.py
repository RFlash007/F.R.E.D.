import logging
import os
import torch
import ollama
import time
import Semantic

# ANSI escape sequences for coloring the output
CYAN = "\033[96m"
RESET_COLOR = "\033[0m"

accessed_memories = []

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# function takes input and gets k most relevant chunks from vault
def get_relevant_context(user_input, vault_embeddings, vault_content, top_k=2):
    if vault_embeddings.nelement() == 0:  # No embeddings available
        return []

    # Get embedding for user input
    response = ollama.embeddings(model='nomic-embed-text', prompt=user_input)
    input_embedding = response["embedding"]
    input_embedding_tensor = torch.tensor(input_embedding).unsqueeze(0)

    # Compute cosine similarity
    cos_scores = torch.cosine_similarity(input_embedding_tensor, vault_embeddings)
    top_k = min(top_k, len(cos_scores))
    if top_k == 0:
        return []
    top_indices = torch.topk(cos_scores, k=top_k).indices.tolist()

    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

def recall_episodic(input):
    # Load vault content and split by double newlines
    vault_path = "Episodic.txt"
    if not os.path.exists(vault_path):
        print("No Episodic.txt found.")
        return []

    content = open_file(vault_path)
    vault_content = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]

    # Generate embeddings for each chunk
    vault_embeddings = []
    for chunk in vault_content:
        response = ollama.embeddings(model='nomic-embed-text', prompt=chunk)
        vault_embeddings.append(response["embedding"])

    # Convert embeddings to a tensor
    vault_embeddings_tensor = torch.tensor(vault_embeddings)

    # Retrieve best matching context
    try:
        relevant_context = get_relevant_context(input, vault_embeddings_tensor, vault_content, top_k=1)
        #store accessed memories
        if relevant_context is not None:
            accessed_memories.append(relevant_context)
    except Exception as e:
        print("Error retrieving relevant context: ", e)
        relevant_context = []

    return relevant_context

def create_memory(conversation: str) -> str:
    """
    Creates a memory entry from a conversation.
    
    Args:
        conversation: The conversation text to analyze
        
    Returns:
        str: The created memory entry in JSON format
    """
    try:
        current_time = time.time()
        conversation_date = time.strftime("%d %B %Y", time.localtime(current_time))

    # Note: Using f-string instead of concatenation and fixing the curly brace issue
        reflection_prompt_template = f"""
        You are creating a memory from the perspective of the Assistant Fred in this conversation summary. The conversation occurred on {conversation_date}. If you do not have enough information for a field, use "N/A". Write one concise sentence per field. Focus on information that will be useful in future interactions. Include context_tags that are specific and reusable. Provide a memory_timestamp.
        
        Output valid JSON in this exact format and nothing else **WRITE NO OTHER TEXT OR DIALOGUE**:

            [
                "timestamp": "YYYY-MM-DD HH:MM",
                "tags": ["tag1", "tag2"],
                "summary": "Brief conversation summary",
                "insights": {{
                    "positive": "What worked well",
                    "negative": "What to improve",
                    "learned": "Key learnings"
                }}
            ]

        Conversation Summary:
        {conversation}"""

        response = ollama.chat(
            model="huihui_ai/qwen2.5-abliterate:14b", 
            messages=[{"role": "user", "content": reflection_prompt_template}]
        )

        if not response.get("message", {}).get("content"):
            print("Error: No content in memory creation response")
            return ""

        episodic_content = response["message"]["content"]
        Semantic.create_semantic(episodic_content)
        
        with open("Episodic.txt", 'a', encoding='utf-8') as file:
            file.write(f"\n\n{episodic_content}")
            
        return episodic_content
        
    except Exception as e:
        print(f"Error creating memory: {e}")
        return ""


def update_episodic(conversation_summary):
    """
    Updates Episodic memories based on a conversation summary.

    :param conversation: A string containing the conversation summary.
    :return: None. Updates the file 'Semantic.txt' in-place.
    """
    #array to hold altered memories
    updated_memories = []
    memory_update_prompt = f"""Review and update these memory entries based on the new conversation.

    NEW CONVERSATION:
    """ + conversation_summary + """

    EXISTING MEMORIES:
    """ + "\n\n".join(str(mem) for mem in accessed_memories) + """

    INSTRUCTIONS:
    1. Compare new conversation with existing memories
    2. Update only if new information conflicts or adds value
    3. Merge overlapping memories
    4. Keep JSON format:
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

    except Exception as e:
        logging.error(f"Error updating episodic memory: {str(e)}")
        raise
import ollama
import Episodic
import os
import torch
import logging

accessed_memories = []
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def create_semantic(memory: str) -> None:
    """
    Extracts factual information from conversations and stores it in a simple format.
    """
    fact_extraction_prompt = f"""Extract factual information from this conversation.

    CONVERSATION:
    {memory}

    INSTRUCTIONS:
    - Extract only verifiable facts and knowledge
    - Ignore conversation flow, timestamps, or contextual details
    - Each fact should be self-contained and complete
    - Format: "• [CATEGORY] fact"
    
    Example format:
    • [PERSONAL] John is allergic to peanuts
    • [PREFERENCE] John prefers tea over coffee
    • [TECHNICAL] Python was created by Guido van Rossum
    • [LOCATION] John lives in Seattle
    """

    response = ollama.chat(
        model="huihui_ai/qwen2.5-abliterate:14b", 
        messages=[{"role": "user", "content": fact_extraction_prompt}]
    )

    if response["message"]["content"]:
        with open("Semantic.txt", 'a', encoding='utf-8') as file:
            file.write(f"\n{response['message']['content']}")

def update_semantic(conversation: str) -> None:
    """
    Updates semantic memory by consolidating and deduplicating facts.
    """
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
    5. Use format: "• [CATEGORY] fact"
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
                
    except Exception as e:
        logging.error(f"Error updating semantic memory: {str(e)}")
        raise

def recall_semantic(query: str, top_k: int = 2) -> list:
    """
    Retrieves relevant facts based on a query using semantic search.
    
    Args:
        query (str): The search query
        top_k (int): Number of facts to return
    
    Returns:
        list: Most relevant facts
    """
    if not os.path.exists("Semantic.txt"):
        return []

    try:
        # Load and preprocess facts
        content = open_file("Semantic.txt")
        facts = [fact.strip() for fact in content.split("\n") if fact.strip()]

        # Generate embeddings
        fact_embeddings = []
        for fact in facts:
            response = ollama.embeddings(model='nomic-embed-text', prompt=fact)
            fact_embeddings.append(response["embedding"])

        # Convert to tensor for efficient computation
        fact_embeddings_tensor = torch.tensor(fact_embeddings)
        
        # Get query embedding
        query_response = ollama.embeddings(model='nomic-embed-text', prompt=query)
        query_embedding = torch.tensor(query_response["embedding"])

        # Calculate similarities
        similarities = torch.cosine_similarity(query_embedding.unsqueeze(0), fact_embeddings_tensor)
        
        # Get top-k most relevant facts
        top_k_indices = torch.argsort(similarities, descending=True)[:top_k]
        relevant_facts = [facts[idx] for idx in top_k_indices]
        
        # Add to accessed memories
        accessed_memories.extend(relevant_facts)
        
        return relevant_facts

    except Exception as e:
        logging.error(f"Error in recall_semantic: {str(e)}")
        return []

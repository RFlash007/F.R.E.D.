import ollama
import Episodic
import os
import torch

accessed_memories = []
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def create_semantic(memory):
    semantic_memory_prompt = f"""
    You are an expert in fact learning from conversations. I need you to review this conversation summary and provide a list of the facts that are learned from the conversation.\n 
    The facts written should just be knowledge facts, not a summary of the conversation.\n
    You do not need to write any reference of time or date unless it's relevant to the facts, for example: fact: Donald Trump was re-elected in **2024**.\n
    here is the format I want the list of facts to be in:\n
    fact: Users name is Ian\n
    fact: assistant's name is FRED\n
    fact: Mars is a planet\n
    Conversation Summary:\n
    {memory}
    """
    blankConvo = [{"role": "user", "content": semantic_memory_prompt}]

    # Initiate streaming chat response
    response = ollama.chat(model="huihui_ai/qwen2.5-abliterate:14b", messages=blankConvo)

    # Store the result
    if "message" in response and "content" in response["message"]:
        semantic_memory = response["message"]["content"]
        with open("Semantic.txt", 'a', encoding='utf-8') as file:
            file.write(f"\n{semantic_memory}")


def recall_semantic(user_input):
    # Load vault content and split by double newlines
    vault_path = "Semantic.txt"
    if not os.path.exists(vault_path):
        print("No Episodic.txt found.")
        return []

    content = open_file(vault_path)
    vault_content = [chunk.strip() for chunk in content.split("\n") if chunk.strip()]

    # Generate embeddings for each chunk
    vault_embeddings = []
    for chunk in vault_content:
        response = ollama.embeddings(model='nomic-embed-text', prompt=chunk)
        vault_embeddings.append(response["embedding"])

    # Convert embeddings to a tensor
    vault_embeddings_tensor = torch.tensor(vault_embeddings)

    # Retrieve best matching context
    try:
        relevant_memories = Episodic.get_relevant_context(user_input, vault_embeddings_tensor, vault_content, top_k=2)
        for memory in relevant_memories:
            accessed_memories.append(memory)
    except Exception as e:
        print("Error retrieving context:", str(e))
        relevant_memories = []
    return relevant_memories


def update_semantic(conversation):
    """
    Updates Semantic memories based on a conversation summary.

    :param conversation: A string containing the conversation summary.
    :return: None. Updates the file 'Semantic.txt' in-place.
    """
    # Step 1: Prompt the model with the conversation summary + existing facts
    memory_update_prompt = f"""Using the following conversation summary:

    {conversation}
    
    Carefully review the existing facts below and **only make changes** if the new information from the summary **requires** it.
    Do not remove valid information unless it is contradicted by the summary.
    If additional details should be added, integrate them in a way that preserves the format.
    Consolidate the memories where you can, if two or more facts can be combined into one, do so.
    Return only what is requested nothing more. If there is any extra dialogue outside of the requested format remove it.
    for example: fact: Users name is Ian\n
    fact: assistant's name is FRED\n
    fact: Mars is a planet\n
    
    **Existing Memory**:
    {accessed_memories}
    """

    messages = [{"role": "user", "content": memory_update_prompt}]
    response = ollama.chat(model="huihui_ai/qwen2.5-abliterate:14b", messages=messages)

    # Step 2: Store the single updated memory from the LLM
    updated_memories = [response["message"]["content"]]

    # Step 3: Read the contents of 'Semantic.txt'
    with open("Semantic.txt", 'r+', encoding='utf-8') as file:
        content = file.read()
        # Split into lines, ignoring any empty lines
        chunks = [chunk.strip() for chunk in content.split("\n") if chunk.strip()]

        # Step 4: Iterate over each chunk and skip if it exists in accessed_memories
        updated_content = []

        for chunk in chunks:
            if chunk not in accessed_memories:
                updated_content.append(chunk)

        # Now append the updated memory from the LLM
        updated_content.extend(updated_memories)

        # Step 5: Overwrite 'Semantic.txt' with the updated content
        file.seek(0)
        file.write("\n".join(updated_content))
        file.truncate()

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

def create_memory(conversation):
    # Get time for timestamp in prompt
    current_time = time.time()  # Get current UNIX timestamp
    conversation_date = time.strftime("%d %B %Y", time.localtime(current_time))

    # Use double braces to ensure Python doesn't interpret as format specifiers
    reflection_prompt_template = f"""
    You are creating a memory from the perspective of the Assistant Fred in this conversation summary. The conversation occurred on {conversation_date}. If you do not have enough information for a field, use "N/A". Write one concise sentence per field. Focus on information that will be useful in future interactions. Include context_tags that are specific and reusable. Provide a memory_timestamp.
    
    Output valid JSON in this exact format (no extra text):

    {{
      "memory_timestamp": "string",
      "context_tags": [
        "string",
        "..."
      ],
      "conversation_summary": "string",
      "what_worked": "string",
      "what_to_avoid": "string",
      "what_you_learned": "string"
    }}
    Conversation Summary:
    {conversation}
    """

    blankConvo = [{"role": "user", "content": reflection_prompt_template}]

    # Initiate streaming chat response
    response = ollama.chat(model="huihui_ai/qwen2.5-abliterate:14b", messages=blankConvo)

    # Store the result
    if "message" in response and "content" in response["message"]:
        episodic_content = response["message"]["content"]
        Semantic.create_semantic(episodic_content)
        with open("Episodic.txt", 'a', encoding='utf-8') as file:
            file.write(f"\n\n{episodic_content}")
        return episodic_content
    else:
        print("No content returned from the model.")


def update_episodic(conversation):
    """
    Updates Episodic memories based on a conversation summary.

    :param conversation: A string containing the conversation summary.
    :return: None. Updates the file 'Semantic.txt' in-place.
    """
    #array to hold altered memories
    updated_memories = []
    memory_update_prompt = f"""Using the following conversation summary:
    \n
    {conversation}
    \n
    You are given multiple memory objects. Carefully review them **all at once** and only make changes if the new information from the summary requires it.
    Do not remove valid information unless it is contradicted by the summary.
    If additional details should be added, integrate them in a way that preserves the JSON structure and property names.
    If you find that two or more memories contain overlapping or redundant information, **consolidate** them into a single memory, retaining all relevant details. 
    If there is any extra dialogue outside of the requested format remove it.

    **Existing Memories**:
    {accessed_memories}
    """

    #get updated episodic memories
    messages = [{"role": "user", "content": memory_update_prompt}]
    response = ollama.chat(model="huihui_ai/qwen2.5-abliterate:14b", messages=messages)

    updated_memories.append(response["message"]["content"])

    # Step 3: Read the contents of 'Semantic.txt'
    with open("Episodic.txt", 'r+', encoding='utf-8') as file:
        content = file.read()
        # Split into lines, ignoring any empty lines
        chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]

        # Step 4: Iterate over each chunk and skip if it exists in accessed_memories
        updated_content = []

        for chunk in chunks:
            if chunk not in accessed_memories:
                updated_content.append(chunk)

        # Now append the updated memory from the LLM
        updated_content.extend(updated_memories)

        # Step 5: Overwrite 'Semantic.txt' with the updated content
        file.seek(0)
        file.write("\n\n".join(updated_content))
        file.truncate()
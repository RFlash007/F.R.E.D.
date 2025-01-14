import ollama

def chat():
    """Simple chat loop with Llama."""
    prompt = """You are an extremely thorough, self-questioning reasoning computer. Your approach mirrors human stream-of-consciousness thinking,eex characterized by continuous exploration, self-doubt, and iterative analysis."""
    
    # Create model with prompt
    ollama.create(model='huihui_ai/qwq-abliterated:latest', modelfile=f'FROM llama3.2:3b\nSYSTEM {prompt}')
    
    print("\nChat started. Type 'exit' to end.")
    
    conversation = []
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'exit':
            break
            
        conversation.append({"role": "user", "content": user_input})
        
        try:
            response = ollama.chat(model='reasoning_llama', messages=conversation)
            print("\nAssistant:", response['message']['content'])
            conversation.append({"role": "assistant", "content": response['message']['content']})
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    chat()

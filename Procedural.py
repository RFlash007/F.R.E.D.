import ollama

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def get_prompt():
    return open_file('Procedural.txt')

def prompt_update(memory):
    try:
        with open("Procedural.txt", 'r+') as file:
            current_prompt = file.read()
            messages = [{
                "role": "user",
                "content": (
                    f"You are a prompt optimization expert. Review this conversation summary and the current AI assistant prompt.\n"
                    f"Rules:\n"
                    f"1. Only output the improved prompt. No explanations.\n"
                    f"2. Keep the prompt on a single line\n"
                    f"3. If the current prompt works well, return it unchanged\n"
                    f"4. Focus on maintaining the assistant's core personality while improving clarity\n\n"
                    f"Current Prompt:\n{current_prompt}\n\n"
                    f"Conversation Summary:\n{memory}"
                )
            }]
            
            response = ollama.chat(
                model="huihui_ai/qwen2.5-abliterate:14b", 
                messages=messages
            )
            
            if not response.get('message', {}).get('content'):
                print("Error: No content in prompt update")
                return current_prompt
                
            new_prompt = response['message']['content']
            print(new_prompt)
            
            file.seek(0)
            file.write(new_prompt)
            file.truncate()
            return new_prompt
            
    except Exception as e:
        print(f"Error updating prompt: {e}")
        return current_prompt

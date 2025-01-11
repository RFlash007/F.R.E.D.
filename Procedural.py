import ollama

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def get_prompt():
    return open_file('Procedural.txt')

def prompt_update(memory):
    try:
        with open("Procedural.txt", 'r+', encoding='utf-8') as file:
            current_prompt = file.read()

            # We explicitly instruct the model to return ONE line, no explanations,
            # and to maintain or minimally improve the prompt's personality.
            messages = [{
                "role": "user",
                "content": (
                    "You are a 'prompt optimization expert.' "
                    "Review the conversation summary and the current system prompt.\n\n"
                    "RULES:\n"
                    "1. Return the improved prompt on ONE SINGLE LINE (no extra line breaks).\n"
                    "2. If the prompt is already good, leave it unchanged.\n"
                    "3. Focus on clarity while preserving the assistant's personality.\n"
                    "4. Output ONLY the final prompt text, no explanations, no markdown.\n\n"
                    f"CURRENT PROMPT:\n{current_prompt}\n\n"
                    f"CONVERSATION SUMMARY:\n{memory}"
                )
            }]

            response = ollama.chat(
                model="huihui_ai/qwen2.5-abliterate:14b",
                messages=messages,
                # We set format="" for plain text output, ensuring no JSON formatting
                # We also lower temperature for minimal randomness.
                options={"temperature": 0}
            )
            
            if not response.get('message', {}).get('content'):
                print("Error: No content in prompt update")
                return current_prompt

            new_prompt = response['message']['content']
            # Ensure we strip excessive newlines/spaces:
            new_prompt = " ".join(new_prompt.split())

            # Print for debugging
            print(new_prompt)

            # Overwrite file with the new prompt
            file.seek(0)
            file.write(new_prompt)
            file.truncate()

            return new_prompt

    except Exception as e:
        print(f"Error updating prompt: {e}")
        return current_prompt

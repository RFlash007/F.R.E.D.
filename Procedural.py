import ollama

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def get_prompt():
    return open_file('Procedural.txt')

def prompt_update(memory):
    current_prompt = open_file("Procedural.txt")
    blankConvo = []
    blankConvo = [{"role": "user", "content": f"Analyze the following summary of a conversation between a user and his AI assistant.\n"
                                              f"Slightly alter the current prompt so it more effectively meets the user’s needs **IF NECESSARY**.\n"
                                              f"If no change is needed, **dont change it**. Provide only the revised prompt, with no additional explanations or any other text for example \"You are FRED\".\n"
                                              f"write the prompt all on one line, dont use any newlines: Current Prompt:\n{current_prompt}\n Conversation:\n{memory}"}]
    # Initiate streaming chat response
    response = ollama.chat(model="huihui_ai/qwen2.5-abliterate:14b", messages=blankConvo)
    print(response['message']['content'])
    with open('Procedural.txt', "w") as file:
        file.write(response['message']['content'])

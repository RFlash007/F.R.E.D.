import ollama
from duckduckgo_search import DDGS
import Tools
import Voice
import Semantic
import Procedural
import Episodic
from ChatUI import ChatUI
import threading
from queue import Queue
import time
import sys

# Move voice_queue to a new file called shared_resources.py
from shared_resources import voice_queue

conversation = []
#available tools
available_functions = {
    'quick_learn': Tools.quick_learn,
    'verify_memories': Tools.verify_memories,
    'system_status': Tools.get_system_status    
}
MAX_CONVERSATION_LENGTH = 10  # or whatever number makes sense

def shutdown_app(current_ui):
    """Cleanly shutdown the application"""
    if current_ui:
        current_ui.root.destroy()
    sys.exit(0)

def process_message(user_input, ui_instance=None):
    """Process a single message and return the response"""
    episodic_memories = Episodic.recall_episodic(user_input)
    semantic_memories = Semantic.recall_semantic(user_input)
    
    if user_input.lower() == "goodbye":
        #summarize the conversation
        summary = summarize(conversation)
        #update the episodic memory and procedural memory
        Procedural.prompt_update(Episodic.create_memory(summary))
        Semantic.create_semantic(summary)
        #update the episodic memory
        Episodic.update_episodic(summary)
        #update the semantic memory
        Semantic.update_semantic(summary)
        if ui_instance:
            ui_instance.display_message("F.R.E.D.: Goodbye for now.", "assistant")
            Voice.piper_speak("Goodbye for now.")
            ui_instance.root.after(2000, lambda: shutdown_app(ui_instance))
        return None

    #create user prompt with no tools
    user_prompt = (
        f"{user_input}\n\n"
        "(END OF USER INPUT)\n\n"
        f"The current time and date is: {Tools.get_time()}\n"
        "These are your memories that may help answer the user's question. "
        "reference them only if they are directly helpful:\n"
        f"{episodic_memories}\n"
        "Here are facts from your memory reference them only if they are directly helpful:\n"
        f"{semantic_memories}"
    )

    conversation.append({"role": "user", "content": user_prompt})
    response = ollama.chat(model="Fred", messages=conversation, tools=[Tools.quick_learn, Tools.verify_memories, Tools.get_system_status, Tools.create_project, Tools.news, Tools.open_project], stream=False)
    #if tools are needed, handle them
    try:
        tool_answer = Tools.handle_tool_calls(response, user_input)

        if tool_answer is not None:
            user_prompt = (
                f"{user_input}\n\n"
                "(END OF USER INPUT)\n\n"
                f"The current time and date is: {Tools.get_time()}\n"
                f"If the relevant information is in your database it will appear here: {tool_answer}\n"
                "These are your memories that may help answer the user's question. "
                "reference them only if they are directly helpful:\n"
                f"{episodic_memories}\n"
                "Here are facts from your memory reference them only if they are directly helpful:\n"
                f"{semantic_memories}"
            )
            #remove the initial prompt and add the new user prompt accounting for tool call
            conversation.pop()
            conversation.append({"role": "user", "content": user_prompt})
            response = ollama.chat(model="Fred", messages=conversation, stream=False)

    except Exception as e:
        return f"An error occurred while communicating with Ollama: {e}"
    print(user_prompt)
    response_content = response['message']['content']
    conversation.append({"role": "assistant", "content": response_content})
    
    # Trim conversation if it gets too long
    if len(conversation) > MAX_CONVERSATION_LENGTH:
        conversation.pop(0)  # Remove oldest message
    
    #remove asterisks from the response
    response_content = response_content.replace('*', '')

    #if voice is enabled, add the response to the voice queue
    
    voice_queue.put(response_content)
    threading.Thread(target=process_voice_queue, daemon=True).start()
    
    return response_content

def chat_loop():
    from Transcribe import initialize_voice_system
    # Initialize UI first
    ui = ChatUI(lambda msg: process_message(msg, ui))  # Pass UI instance through lambda
    
    # Initialize voice system with process_message callback that includes UI
    voice_system = initialize_voice_system(lambda msg: process_message(msg, ui))
    voice_system.set_ui(ui)
    
    try:
        # Start voice processing in a separate thread
        voice_thread = threading.Thread(target=voice_system.process_audio, daemon=True)
        voice_thread.start()
        
        # Run UI in the main thread
        ui.run()  # This blocks until window is closed
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        voice_system.stop()
        print("Voice system stopped.")
        sys.exit(0)  # Ensure complete shutdown

def summarize(input):
    user_input = f'''
    You are given a conversation transcript between multiple participants. Please produce a concise but thorough summary with the following structure:

    1. **Key Facts and Information**: Summarize the most important points, including any background details learned about each participant (e.g., their preferences, needs, or constraints).
    
    2. **Effective Approaches**: Identify which solutions or actions were particularly helpful or successful, and explain briefly why they worked.
    
    3. **Ineffective or Problematic Approaches**: Note any solutions or actions that caused confusion or did not work well, and explain why they were less effective.
    
    4. **Insights and Lessons Learned**: Highlight any new knowledge, lessons, or observations that arose from the discussion.
    
    5. **Potential Next Steps**: Suggest possible improvements or follow-up actions based on what was discussed or discovered during the conversation.
    
    Make sure to include:
    - All significant facts about the participants’ preferences, goals, or constraints.
    - Any points of agreement or disagreement.
    - Observations about the conversation flow, tone, or style that may be relevant later.
    
    Remember, the goal is to capture every important detail in a concise narrative. 
    
    **Conversation Transcript**:
    {input}
    '''
    blankConvo = []
    blankConvo.append({"role": "user", "content": user_input})
    response = ollama.chat(model="huihui_ai/qwen2.5-abliterate:14b", messages=blankConvo)
    print(response['message']['content'])
    return response['message']['content']

def process_voice_queue():
    """Process voice responses in queue"""
    while True:
        try:
            message = voice_queue.get()
            if message:
                success = Voice.piper_speak(message)
                if not success:
                    print("Failed to process speech synthesis")
            voice_queue.task_done()
        except Exception as e:
            print(f"Error processing voice queue: {str(e)}")
            voice_queue.task_done()  # Make sure to mark task as done even on error

if __name__ == "__main__":
    #create the model with new prompt
    prompt = Procedural.get_prompt()
    modelfile = f'''
    FROM huihui_ai/qwen2.5-abliterate:14b
    SYSTEM {prompt}
    '''
    ollama.create(model='FRED', modelfile=modelfile)
    chat_loop()


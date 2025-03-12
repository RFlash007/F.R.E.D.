import ollama
from duckduckgo_search import DDGS
import Dreaming
import MorningReport
import Tools
import Voice
import Semantic
import Procedural
import Episodic
import Vision
from ChatUI import ChatUI
import threading
from queue import Queue
import time
import sys
import logging
import json
from datetime import datetime
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# Move voice_queue to a new file called shared_resources.py
from shared_resources import voice_queue


def route_query(query: str) -> str:
    """
    Always returns FRED_14b as the model to use.
    Model routing has been disabled to exclusively use 14b.
    """
    return "FRED_14b"

conversation = []
MAX_CONVERSATION_LENGTH = 5  # Adjust as needed

def shutdown_app(current_ui):
    """Cleanly shut down the application."""
    if current_ui:
        current_ui.root.destroy()
    sys.exit(0)

def save_conversation(conversation_history):
    """Save conversation history to JSON files."""
    try:
        history_dir = "conversation_history"    
        os.makedirs(history_dir, exist_ok=True)
        current_session = os.path.join(history_dir, "current_session.json")
        archive_file = os.path.join(history_dir, f"archive_{datetime.now().strftime('%Y%m%d')}.json")
        with open(current_session, 'w', encoding='utf-8') as f:
            json.dump(conversation_history, f, indent=2)
        with open(archive_file, 'w', encoding='utf-8') as f:
            json.dump(conversation_history, f, indent=2)
            
        # Process the newly saved conversation to extract dreams
        dreams_count = Dreaming.process_new_conversation(archive_file)
        logging.info(f"Processed conversation and saved {dreams_count} dreams")
        
        return True
    except Exception as e:
        print(f"Error saving conversation history: {str(e)}")
        return False

def toggle_vision(command, ui_instance=None):
    """
    Toggle the vision system on/off based on the command.
    Commands:
        on - Activate vision system
        off/disable - Deactivate vision system
        status - Check vision system status
        toggle - Toggle current vision system state
    Returns a response message.
    """
    import Vision
    
    command = command.lower().strip()
    
    if command == "on":
        if Vision.is_vision_active():
            return "Vision system is already active, sir."
        
        try:
            vision_initialized = Vision.initialize_vision(show_window=True)
            if vision_initialized:
                return "Vision system online, sir. I can now see through your camera."
            else:
                return "Warning: I was unable to initialize the vision system."
        except Exception as e:
            return f"Error initializing vision system: {str(e)}"
            
    elif command == "off" or command == "disable":
        if not Vision.is_vision_active():
            return "Vision system is already offline, sir."
        
        try:
            Vision.stop_vision_system()
            return "Vision system has been disabled, sir. Your privacy is restored."
        except Exception as e:
            return f"Error disabling vision system: {str(e)}"
            
    elif command == "status":
        if Vision.is_vision_active():
            return "Vision system is currently active and operational, sir."
        else:
            return "Vision system is currently offline, sir."
    
    elif command == "toggle":
        if Vision.is_vision_active():
            Vision.stop_vision_system()
            return "Vision system has been disabled, sir."
        else:
            vision_initialized = Vision.initialize_vision(show_window=True)
            if vision_initialized:
                return "Vision system online, sir."
            else:
                return "Warning: I was unable to initialize the vision system."
    
    return "Invalid vision command. Available commands: on, off, disable, status, toggle"

def identify_person(command, ui_instance=None):
    """
    Identify a person in the camera view with a name.
    
    Args:
        command (str): The command containing the person's name
        ui_instance (ChatUI, optional): The UI instance for displaying messages
        
    Returns:
        str: Response message
    """
    # Extract the person's name from the command
    name = command.strip()
    
    if not name:
        return "Please provide a valid name for the person."
    
    # Call the vision system to identify the face
    response = Vision.identify_face(name)
    return response

def process_message(user_input, ui_instance=None):
    """
    Process a single message and return the response from the model.
    If the user says "goodbye", summarize the conversation, update memories, and shut down.
    
    Special commands:
    - /vision [on|off|disable|status|toggle] - Control the vision system
    - /identify [name] - Identify a person in the camera view
    - /faces - List all known faces
    """
    # Check for vision system commands - process silently without UI messages
    if user_input.lower().startswith("/vision "):
        command = user_input.lower().replace("/vision ", "").strip()
        response = toggle_vision(command, ui_instance)
        
        # Only speak the response, don't display in UI
        Voice.piper_speak(response)
        return
    
    # Check for identify person command
    if user_input.lower().startswith("/identify "):
        response = identify_person(user_input.lower().replace("/identify ", "").strip(), ui_instance)
        
        # Display and speak concise response
        concise_response = response.split(".")[0] + "." if "." in response else response
        Voice.piper_speak(concise_response)
        return
    
    # Check for list faces command
    if user_input.lower() == "/faces" or user_input.lower() == "list faces":
        response = Vision.list_known_faces()
        
        # Display and speak the response
        if ui_instance:
            ui_instance.display_message(f"F.R.E.D.: {response}", "assistant")
        Voice.piper_speak(response.replace("\n-", ". ").replace(":\n", ": "))
        return
    
    # Check for natural language identification patterns when vision is active
    if Vision.is_vision_active():
        input_lower = user_input.lower()
        name = None
        
        # Pattern: "That's [Name]" or "This is [Name]"
        if input_lower.startswith("that's ") or input_lower.startswith("that is "):
            name = user_input[input_lower.find(" ") + 1:].strip()
        elif input_lower.startswith("this is "):
            name = user_input[8:].strip()
        
        # If we found a name pattern and vision is active, try to identify the person
        if name:
            response = Vision.identify_face(name)
            
            # Display and speak concise response
            concise_response = response.split(".")[0] + "." if "." in response else response
            if ui_instance:
                ui_instance.display_message(f"F.R.E.D.: {concise_response}", "assistant")
            Voice.piper_speak(concise_response)
            return
        
    # 1. Recall episodic and semantic memories.
    start_time = datetime.now()
    episodic_memories = Episodic.recall_episodic(user_input)
    semantic_memories = Semantic.recall_semantic(user_input)
    # Use the standard recall method for dreams - always returning the best single match
    dreams = Dreaming.recall_dreams(user_input, top_k=1)
    
    elapsed_time = datetime.now() - start_time
    # Removed print statement for memory recall time
    
    # 2. If the user says "goodbye", finalize conversation.
    if user_input.lower() == "goodbye":
        # Convert conversation list to string format for summarization
        conversation_text = ""
        for msg in conversation:
            if 'content' in msg:
                role = msg.get('role', 'unknown')
                conversation_text += f"{role}: {msg['content']}\n\n"
        
        summary = summarize(conversation_text)
        Episodic.create_episodic(summary)
        Semantic.create_semantic(summary)
        Dreaming.create_dream(summary)
        Episodic.update_episodic(summary)
        Semantic.update_semantic(summary)
        Dreaming.update_dream(summary)
        save_conversation(conversation)
        if ui_instance:
            ui_instance.display_message("F.R.E.D.: Goodbye for now.", "assistant")
            Voice.piper_speak("Goodbye for now.")
            ui_instance.root.after(2000, lambda: shutdown_app(ui_instance))
        return None

    # 3. Build the user prompt.
    user_prompt = (
        f"{user_input}\n\n(END OF USER INPUT)\n\n"
        f"The current time and date is: {Tools.get_time()}\n"
        f"These are your memories that may help answer the user's question. Reference them only if directly helpful:\n{episodic_memories}\n\n"
        f"Here are facts from your memory. Reference them only if directly helpful:\n{semantic_memories}\n"
        f"These are your dreams and insights. Reference them only if directly helpful:\n{dreams}\n"
    )
    
    user_prompt += "(END OF FRED DATABASE)"
    
    conversation.append({"role": "user", "content": user_prompt})

    # 4. Define tool schemas (unchanged from your original script)
    tools_schema = [
        {
            'type': 'function',
            'function': {
                'name': 'search_web_information',
                'description': (
                    "Search for EXTERNAL information from the web - NOT from F.R.E.D.'s internal memory. "
                    "Provides two modes: 'educational' for learning about topics, and 'news' for current events. "
                    "Use 'educational' mode for general knowledge and explanations. "
                    "Use 'news' mode for recent events and developments. "
                    "Provide 'topics' as comma-separated search topics."
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'topics': {
                            'type': 'string',
                            'description': 'Comma-separated list of topics to search for.'
                        },
                        'mode': {
                            'type': 'string',
                            'enum': ['educational', 'news'],
                            'description': 'The mode of summarization.'
                        }
                    },
                    'required': ['topics']
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'get_system_status',
                'description': (
                    "Get system status information: CPU usage, Memory usage, Disk usage, "
                    "and GPU usage (if available). Returns a formatted report with current system metrics."
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {},
                    'required': []
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'create_note',
                'description': (
                    "Create a new note with the given title and content."
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'note_title': {
                            'type': 'string',
                            'description': 'Title of the note to create'
                        },
                        'note_content': {
                            'type': 'string',
                            'description': 'Content to write in the note'
                        }
                    },
                    'required': ['note_title', 'note_content']
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'update_note',
                'description': (
                    "Update an existing note with the given title and content."
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'note_title': {
                            'type': 'string',
                            'description': 'Title of the note to update'
                        },
                        'note_content': {
                            'type': 'string',
                            'description': 'New content for the note'
                        }
                    },
                    'required': ['note_title', 'note_content']
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'read_note',
                'description': (
                    "Read the content of an existing note."
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'note_title': {
                            'type': 'string',
                            'description': 'Title of the note to read'
                        }
                    },
                    'required': ['note_title']
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'delete_note',
                'description': (
                    "Delete an existing note."
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'note_title': {
                            'type': 'string',
                            'description': 'Title of the note to delete'
                        }
                    },
                    'required': ['note_title']
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'list_notes',
                'description': (
                    "List all available notes with their creation dates."
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {},
                    'required': []
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'add_task',
                'description': (
                    "Add a new task to the task list with an optional due date."
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'task_title': {
                            'type': 'string',
                            'description': 'Title of the task to add'
                        },
                        'task_content': {
                            'type': 'string',
                            'description': 'Content/description of the task'
                        },
                        'due_date': {
                            'type': 'string',
                            'description': 'Due date for the task in YYYY-MM-DD HH:MM:SS format. Will be set to None if not provided.'
                        }
                    },
                    'required': ['task_title', 'task_content']
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'list_tasks',
                'description': (
                    "List all tasks from the task list."
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {},
                    'required': []
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'delete_task',
                'description': (
                    "Delete a task from the task list."
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'task_title': {
                            'type': 'string',
                            'description': 'Title of the task to delete'
                        }
                    },
                    'required': ['task_title']
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'access_memory_database',
                'description': (
                    "Access F.R.E.D.'s internal memory databases (episodic, semantic, and dreams) to recall historical information. "
                    "Use this for retrieving past conversations, stored facts, or AI insights - NOT for web searches. "
                    "ALWAYS use this tool when the user asks about previous discussions or what F.R.E.D. remembers."
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {
                            'type': 'string',
                            'description': 'The search query to find relevant memories.'
                        },
                        'memory_type': {
                            'type': 'string',
                            'enum': ['episodic', 'semantic', 'dreams', 'all'],
                            'description': 'The type of memory to search.'
                        },
                        'top_k': {
                            'type': 'integer',
                            'description': 'Number of results to return for each memory type.'
                        }
                    },
                    'required': ['query']
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'get_sight',
                'description': (
                    "Get information about what F.R.E.D. can currently see through the vision system. "
                    "Use this tool when the user asks about what you can see, what objects are visible, "
                    "or when visual identification is requested. This accesses the webcam feed with real-time "
                    "object detection."
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                    },
                    'required': []
                }
            }
        },
    ]

    # 5. Determine which model to use.
    model_to_use = route_query(user_input)
    print(f"Using model {model_to_use} for query")

    # 6. Call the chosen model via Ollama.
    response = ollama.chat(
        model=model_to_use,
        messages=conversation,
        tools=tools_schema,
        stream=False
    )

    # 7. Handle potential tool calls.
    try:
        tool_answer = Tools.handle_tool_calls(response, user_input)
        if tool_answer is not None:
            user_prompt = (
                f"{user_input}\n\n(END OF USER INPUT)\n\n"
                f"The current time and date is: {Tools.get_time()}\n"
                f"Relevant info from the tool:\n{tool_answer}\n\n"
                f"These are your memories that may help answer the user's question. Reference them only if directly helpful:\n{episodic_memories}\n\n"
                f"Here are facts from your memory. Reference them only if directly helpful:\n{semantic_memories}\n"
                f"These are your dreams and insights. Reference them only if directly helpful:\n{dreams}\n"
                "(END OF FRED DATABASE)"
            )
            conversation.pop()
            conversation.append({"role": "user", "content": user_prompt})
            response = ollama.chat(
                model=model_to_use,
                messages=conversation,
                stream=False
            )
    except Exception as e:
        return f"An error occurred while communicating with Ollama: {e}"

    # 8. Extract the final text from the response.
    response_content = response['message']['content']
    conversation.append({"role": "assistant", "content": response_content})
    print(user_prompt)
    response_content = response_content.replace('*', '')

    # 9. Send the response to voice output.
    voice_queue.put(response_content)

    # Check if the input is a JSON string containing messages
    # Convert conversation list to a string to check if it's a JSON format
    conversation_str = json.dumps(conversation)
    if conversation_str.strip().startswith('[') and any('role' in msg and 'content' in msg for msg in conversation):
        # We already have the structured conversation in the 'conversation' variable
        # Extract only the text content from each message to create a text representation
        text_content = ""
        for msg in conversation:
            if 'content' in msg:
                role = msg.get('role', 'unknown')
                text_content += f"{role}: {msg['content']}\n\n"
        conversation_text = text_content.strip()
        # Note: We're not reassigning 'conversation' as it should remain a list

    return response_content

def chat_loop():
    """
    Main loop handling voice and UI. Runs as __main__.
    """
    from Transcribe import initialize_voice_system

    # Initialize the chat UI.
    ui = ChatUI(lambda msg: process_message(msg, ui))
    # Initialize the voice system.
    voice_system = initialize_voice_system(lambda msg: process_message(msg, ui))
    voice_system.set_ui(ui)
    
    # Initialize the vision system
    try:
        vision_initialized = Vision.initialize_vision(show_window=True)
        if vision_initialized:
            voice_queue.put("Vision system online, sir. I can now see through your camera.")
        else:
            voice_queue.put("Warning: I was unable to initialize the vision system.")
    except Exception as e:
        # Empty exception handler - silently ignore errors
        pass
    
    # Generate morning report when AI starts
    try:
        morning_report = MorningReport.generate_morning_report()
        # Remove UI display of morning report
        ui.display_message(morning_report, "assistant")
        voice_queue.put(f"Good morning, sir. I've prepared your daily briefing.")
        conversation.append({"role": "assistant", "content": morning_report})
    except Exception as e:
        print(f"Error generating morning report: {str(e)}")
    # Start the voice processing thread.
    try:
        # Start the voice processing thread.
        voice_thread = threading.Thread(target=process_voice_queue, daemon=True)
        voice_thread.start()
        ui.run()  # This call blocks until the window is closed.
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Stop the vision system
        if Vision.is_vision_active():
            Vision.stop_vision_system()
            print("Vision system stopped.")
        voice_system.stop()
        print("Voice system stopped.")
        sys.exit(0)

def summarize(input_data: str) -> str:
    current_time = Tools.get_time()
    user_prompt = f"""
Return ONE valid JSON object with these keys:
  "high_level_summary"  : string
  "possible_facts"      : array of objects, each with keys "category" and "content"
  "episode_candidates"  : array of objects, each with:
     "memory_timestamp"
     "context_tags" (array of strings)
     "conversation_summary"
     "what_worked"
     "what_to_avoid"
     "what_you_learned"

Use empty arrays/strings if any section is not applicable.
No extra text—JSON only.

TIME: {current_time}
CONVERSATION:
{input_data}
"""
    response = ollama.chat(
        model="huihui_ai/qwen2.5-abliterate:14b",
        messages=[{"role": "user", "content": user_prompt}],
        format="json",
        options={"temperature": 0}
    )
    return response["message"]["content"]

def process_voice_queue():
    """
    Process voice responses in the queue. Runs as a daemon thread.
    """
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
            voice_queue.task_done()

if __name__ == "__main__":
    # Initialize final_prompt before using it.
    final_prompt = ""
    prompt = Procedural.get_prompt()
    final_prompt = " ".join(prompt.splitlines())

    time_start = datetime.now()
    Semantic.initialize_cache()  # Initialize semantic cache.
    Episodic.initialize_cache()   # Initialize episodic cache.
    Dreaming.initialize_cache()   # Initialize dreams cache.
    
    # Perform a "fake" query to ensure the cache is properly loaded for each memory system
    dummy_query = "test query for cache initialization"
    Semantic.recall_semantic(dummy_query, top_k=1)
    Episodic.recall_episodic(dummy_query, top_k=1)
    Dreaming.recall_dreams(dummy_query, top_k=1)
    
    print(f"Initialization took: {datetime.now() - time_start}")

    # Define modelfile.
    modelfile_14b = f'''
FROM huihui_ai/qwen2.5-abliterate:14b
SYSTEM {final_prompt}
PARAMETER num_ctx 8192
'''

    # Load vision system prompt from file
    vision_prompt_file = "vision_prompt.txt"
    try:
        with open(vision_prompt_file, 'r') as f:
            vision_prompt = f.read()
            vision_prompt = vision_prompt.replace('\n', ' ')  # Format for modelfile
    except Exception as e:
        print(f"Error loading vision prompt file: {str(e)}")
        vision_prompt = "Describe the image with neutral, detailed precision."

    # Define vision modelfile
    modelfile_vision = f'''
FROM llama3.2-vision:latest
SYSTEM {vision_prompt}
PARAMETER num_ctx 4096
'''

    try:
        # Create the FRED model with modelfile
        ollama.create(model='FRED_14b', modelfile=modelfile_14b)
        
        # Create the vision model with modelfile
        print("Creating vision model...")
        ollama.create(model='FRED_vision', modelfile=modelfile_vision)
        print("Vision model created successfully")
        
        # Start the main chat loop.
        chat_loop()
    except Exception as e:
        print(f"Error initializing FRED: {str(e)}")
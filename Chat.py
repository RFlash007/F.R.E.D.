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
MAX_CONVERSATION_LENGTH = 10  # or whatever number makes sense


def shutdown_app(current_ui):
    """Cleanly shut down the application."""
    if current_ui:
        current_ui.root.destroy()
    sys.exit(0)



def process_message(user_input, ui_instance=None):
    """
    Process a single message and return the response from the model.
    If user says 'goodbye', we summarize conversation and update memory.
    """
    # 1. Attempt to recall episodic & semantic memories relevant to the user_input
    episodic_memories = Episodic.recall_episodic(user_input)
    semantic_memories = Semantic.recall_semantic(user_input)

    # 2. If user wants to end the conversation
    if user_input.lower() == "goodbye":
        summary = summarize(conversation)
        # Update memories
        Procedural.prompt_update(Episodic.create_episodic(summary))
        Semantic.create_semantic(summary)
        Episodic.update_episodic(summary)
        Semantic.update_semantic(summary)

        if ui_instance:
            ui_instance.display_message("F.R.E.D.: Goodbye for now.", "assistant")
            Voice.piper_speak("Goodbye for now.")
            ui_instance.root.after(2000, lambda: shutdown_app(ui_instance))
        return None

    # 3. Create user prompt with no tools
    user_prompt = (
        f"{user_input}\n\n"
        "(END OF USER INPUT)\n\n"
        f"The current time and date is: {Tools.get_time()}\n"
        "These are your memories that may help answer the user's question. "
        "Reference them only if they are directly helpful:\n"
        f"{episodic_memories}\n\n"
        "Here are facts from your memory. Reference them only if they are directly helpful:\n"
        f"{semantic_memories}"
    )

    conversation.append({"role": "user", "content": user_prompt})

    # ----
    # 4. Provide tool definitions with JSON schemas
    #     So Ollama knows how to pass arguments for quick_learn, news, etc.
    # ----
    tools_schema = [
        {
            'type': 'function',
            'function': {
                'name': 'quick_learn',
                'description': (
                    "Perform a DuckDuckGo-based search for informational learning. "
                    "Provide 'topics' as comma-separated search topics."
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'topics': {
                            'type': 'string',
                            'description': 'Comma-separated topics to search for'
                        }
                    },
                    'required': ['topics']
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'news',
                'description': (
                    "Perform a DuckDuckGo-based search for news topics. "
                    "Provide 'topics' as comma-separated search topics."
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'topics': {
                            'type': 'string',
                            'description': 'Comma-separated news topics'
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
                    "Get system status information: CPU usage, Memory usage, Disk usage, and GPU usage."
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
                'name': 'create_project',
                'description': (
                    "Create a new project directory."
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'project_name': {
                            'type': 'string',
                            'description': 'Name of the project to create'
                        }
                    },
                    'required': ['project_name']
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'edit_file_in_project',
                'description': (
                    "Edit a specific file within a project."
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'project_name': {
                            'type': 'string',
                            'description': 'Name of the project'
                        },
                        'file_name': {
                            'type': 'string',
                            'description': 'Name of the file to edit'
                        },
                        'file_content': {
                            'type': 'string',
                            'description': 'A concise description of what the Python code should do, including any essential features or constraints. Keep it clear and direct.'
                        }
                    },
                    'required': ['project_name', 'file_name', 'file_content']
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'read_file_in_project',
                'description': (
                    "Read a specific file within a project."
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'project_name': {
                            'type': 'string',
                            'description': 'Name of the project'
                        },
                        'file_name': {
                            'type': 'string',
                            'description': 'Name of the file to read'
                        }
                    },
                    'required': ['project_name', 'file_name']
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'delete_project',
                'description': (
                    "Delete an entire project directory."
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'project_name': {
                            'type': 'string',
                            'description': 'Name of the project to delete'
                        }
                    },
                    'required': ['project_name']
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'delete_file_in_project',
                'description': (
                    "Delete a specific file within a project."
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'project_name': {
                            'type': 'string',
                            'description': 'Name of the project'
                        },
                        'file_name': {
                            'type': 'string',
                            'description': 'Name of the file to delete'
                        }
                    },
                    'required': ['project_name', 'file_name']
                }
            }
        }
    ]

    # 5. Chat with Ollama, passing the JSON schema for each tool
    response = ollama.chat(
        model="FRED",
        messages=conversation,
        tools=tools_schema,  # <--- This is where Ollama sees the argument definitions
        stream=False
    )

    # 6. If the model wants to call a tool, handle it
    try:
        tool_answer = Tools.handle_tool_calls(response, user_input)
        if tool_answer is not None:
            # The model used a tool. Let's incorporate the result into the next prompt.
            user_prompt = (
                f"{user_input}\n\n"
                "(END OF USER INPUT)\n\n"
                f"The current time and date is: {Tools.get_time()}\n"
                "Relevant info from the tool:\n"
                f"{tool_answer}\n\n"
                "These are your memories that may help answer the user's question. "
                "Reference them only if they are directly helpful:\n"
                f"{episodic_memories}\n\n"
                "Here are facts from your memory. Reference them only if they are directly helpful:\n"
                f"{semantic_memories}"
            )
            # Replace the last user prompt with the updated info
            conversation.pop()
            conversation.append({"role": "user", "content": user_prompt})
            response = ollama.chat(
                model="FRED",
                messages=conversation,
                stream=False
            )

    except Exception as e:
        return f"An error occurred while communicating with Ollama: {e}"

    # 7. Extract final text from the model's response
    response_content = response['message']['content']
    conversation.append({"role": "assistant", "content": response_content})
    print(user_prompt)
    # 8. Trim conversation if it gets too long
    if len(conversation) > MAX_CONVERSATION_LENGTH:
        conversation.pop(0)
        "Integrate working conversation history summary here"

    # 9. Remove asterisks or extraneous characters
    response_content = response_content.replace('*', '')

    # 10. Handle voice output
    #voice_queue.put(response_content)
    #threading.Thread(target=process_voice_queue, daemon=True).start()

    return response_content


def chat_loop():
    """
    Main loop handling voice and UI. This is run as __main__.
    """
    from Transcribe import initialize_voice_system

    # Initialize UI first
    ui = ChatUI(lambda msg: process_message(msg, ui))  # Pass UI instance to process_message

    # Initialize voice system with callback that includes UI
    voice_system = initialize_voice_system(lambda msg: process_message(msg, ui))
    voice_system.set_ui(ui)

    try:
        # Start voice processing in a separate thread
        voice_thread = threading.Thread(target=voice_system.process_audio, daemon=True)
        voice_thread.start()

        # Run UI in the main thread
        ui.run()  # Blocks until the window is closed

    except KeyboardInterrupt:
        print("\nShutting down...")

    finally:
        voice_system.stop()
        print("Voice system stopped.")
        sys.exit(0)  # Ensure complete shutdown


def summarize(input_data: str) -> str:
    time = Tools.get_time()
    """
    Returns a single JSON object summarizing the conversation, containing:
      1. "high_level_summary": string
      2. "possible_facts": list of { "category": ..., "content": ... }
      3. "episode_candidates": list of objects for episodic storage

    Usage:
      summary_json_str = summarize(some_conversation)
      # Then pass summary_json_str to create_semantic(...) and create_episodic(...)
    """

    user_prompt = f"""
    Return ONE valid JSON object with these keys:
      "high_level_summary"  : string
      "possible_facts"      : array of objects, each {{ "category": ..., "content": ... }}
      "episode_candidates"  : array of objects, each with:
         "memory_timestamp"
         "context_tags" (array of strings)
         "conversation_summary"
         "what_worked"
         "what_to_avoid"
         "what_you_learned"

    Use empty arrays/strings if any section is not applicable.
    No extra text—JSON only.

    TIME: {time}
    CONVERSATION:
    {input_data}
    """

    response = ollama.chat(
        model="huihui_ai/qwen2.5-abliterate:14b",
        messages=[{"role": "user", "content": user_prompt}],
        format="json",        # Encourage JSON-only response
        options={"temperature": 0}  # Low temperature for consistency
    )

    # Return raw JSON string (the calling code can do json.loads(...) if needed)
    return response["message"]["content"]


def process_voice_queue():
    """
    Process voice responses in queue. Called as a separate daemon thread.
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
            voice_queue.task_done()  # Mark task as done on error as well


if __name__ == "__main__":
    prompt = Procedural.get_prompt()
    modelfile = f'''
    FROM huihui_ai/qwen2.5-abliterate:14b
    SYSTEM {prompt} If you wish to call a tool, ***REMEMBER TO RETURN THE VALID TOOL CALL JSON***
    '''
    # Create the custom model named 'FRED' with the system prompt
    ollama.create(model='FRED', modelfile=modelfile)
    # Start the main loop
    chat_loop()

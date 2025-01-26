import ollama
from duckduckgo_search import DDGS
import Dreaming
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
import logging
import json
from datetime import datetime
import os

# Move voice_queue to a new file called shared_resources.py
from shared_resources import voice_queue

conversation = []
MAX_CONVERSATION_LENGTH = 5  # or whatever number makes sense


def shutdown_app(current_ui):
    """Cleanly shut down the application."""
    if current_ui:
        current_ui.root.destroy()
    sys.exit(0)


def save_conversation(conversation_history):
    """Save conversation history to JSON files"""
    try:
        # Create conversation_history directory if it doesn't exist
        history_dir = "conversation_history"
        os.makedirs(history_dir, exist_ok=True)
        
        # Save to both current session and archive
        current_session = os.path.join(history_dir, "current_session.json")
        archive_file = os.path.join(
            history_dir,
            f"archive_{datetime.now().strftime('%Y%m%d')}.json"
        )
        
        # Save to both files
        with open(current_session, 'w', encoding='utf-8') as f:
            json.dump(conversation_history, f, indent=2)
        with open(archive_file, 'w', encoding='utf-8') as f:
            json.dump(conversation_history, f, indent=2)
            
        return True
    except Exception as e:
        print(f"Error saving conversation history: {str(e)}")
        return False


def process_message(user_input, ui_instance=None):
    """
    Process a single message and return the response from the model.
    If user says 'goodbye', we summarize conversation and update memory.
    """
    # 1. Attempt to recall episodic & semantic memories relevant to the user_input
    episodic_memories = Episodic.recall_episodic(user_input)
    semantic_memories = Semantic.recall_semantic(user_input)
    assumptions = Dreaming.recall_assumptions(user_input)
    # 2. If user wants to end the conversation
    if user_input.lower() == "goodbye":
        summary = summarize(conversation)
        # Update memories
        Episodic.create_episodic(summary)
        Semantic.create_semantic(summary)
        Episodic.update_episodic(summary)
        Semantic.update_semantic(summary)
        Dreaming.update_assumptions()
        # Save conversation history
        save_conversation(conversation)

        if ui_instance:
            ui_instance.display_message("F.R.E.D.: Goodbye for now.", "assistant")
            Voice.piper_speak("Goodbye for now.")
            ui_instance.root.after(2000, lambda: shutdown_app(ui_instance))
        return None

    # 3. Create user prompt with no tools
    user_prompt = (
        f"{user_input}\n\n(END OF USER INPUT)\n\n"
        f"The current time and date is: {Tools.get_time()}\n"
        f"These are your memories that may help answer the user's question. Reference them only if they are directly helpful:\n{episodic_memories}\n\n"
        f"Here are facts from your memory. Reference them only if they are directly helpful:\n{semantic_memories}"
        f"These are your assumptions. Reference them only if they are directly helpful:\n{assumptions}\n"
        "(END OF FRED DATABASE)"
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
                            'description': 'A simple description of what the Python code should do. Do not include any code.'
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
        },
        {
            'type': 'function',
            'function': {
                'name': 'add_task',
                'description': (
                    "Add a new task to the task list."
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
                        }
                    },
                    'required': ['task_title', 'task_content']
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'read_task',
                'description': (
                    "Read all tasks from the task list."
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
                f"{user_input}\n\n(END OF USER INPUT)\n\n"
                f"The current time and date is: {Tools.get_time()}\n"
                f"Relevant info from the tool:\n{tool_answer}\n\n"
                f"These are your memories that may help answer the user's question. Reference them only if they are directly helpful:\n{episodic_memories}\n\n"
                f"Here are facts from your memory. Reference them only if they are directly helpful:\n{semantic_memories}"
                f"These are your assumptions. Reference them only if they are directly helpful:\n{assumptions}\n"
                "(END OF FRED DATABASE)"
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
    # 8. Trim conversation if it gets too long and summarize perspectives
    MAX_MESSAGES_BEFORE_SUMMARY = 10  # Adjust based on your needs
    MAX_TOKENS_PER_MESSAGE = 1000     # Approximate token limit per message
    
    def should_summarize(conv):
        """Determine if conversation needs summarization based on multiple factors"""
        message_count = len(conv)
        total_length = sum(len(msg["content"]) for msg in conv)
        avg_message_length = total_length / message_count if message_count > 0 else 0
        
        return (
            message_count > MAX_MESSAGES_BEFORE_SUMMARY or
            total_length > MAX_MESSAGES_BEFORE_SUMMARY * MAX_TOKENS_PER_MESSAGE or
            avg_message_length > MAX_TOKENS_PER_MESSAGE * 1.5  # Allow some messages to be longer
        )
    
    if should_summarize(conversation):
        # Get summaries from perspective_summary function
        user_summary, assistant_summary = perspective_summary(str(conversation))
        
        # Keep the last few messages for immediate context
        recent_messages = conversation[-3:]  # Keep last 3 messages
        
        # Clear existing conversation and replace with summaries
        conversation.clear()
        
        # Add both summaries as context messages
        conversation.extend([
            {"role": "user", "content": user_summary},
            {"role": "assistant", "content": assistant_summary},
        ])
        
        # Add back recent messages for immediate context
        conversation.extend(recent_messages)
        
        print("\nConversation summarized while maintaining recent context.")
    
    # 9. Remove asterisks or extraneous characters
    response_content = response_content.replace('*', '')

    # 10. Handle voice output
    voice_queue.put(response_content)

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
        # Initialize a single dedicated thread for voice processing at startup
        voice_thread = threading.Thread(target=process_voice_queue, daemon=True)
        voice_thread.start()

        # Run UI in the main thread
        ui.run()  # Blocks until the window is closed

    except KeyboardInterrupt:
        print("\nShutting down...")

    finally:
        voice_system.stop()
        print("Voice system stopped.")
        sys.exit(0)  # Ensure complete shutdown

def perspective_summary(input_data: str) -> tuple[str, str]:
    """
    Enhanced conversation summarization that maintains distinct perspectives while preserving
    key context and semantic relationships.
    
    Args:
        input_data (str): The conversation history to summarize
        
    Returns:
        tuple[str, str]: (user_summary, assistant_summary)
    """
    # Parse conversation data
    if isinstance(input_data, str):
        try:
            conv_list = eval(input_data)
        except:
            conv_list = input_data.split('\n')
    else:
        conv_list = input_data

    # Group messages by role with context
    user_context = []
    assistant_context = []
    
    for i, msg in enumerate(conv_list):
        if msg["role"] == "user":
            # Get surrounding context
            prev_msg = conv_list[i-1] if i > 0 else None
            next_msg = conv_list[i+1] if i < len(conv_list)-1 else None
            
            context = {
                "message": msg["content"],
                "prev_context": prev_msg["content"] if prev_msg and prev_msg["role"] == "assistant" else None,
                "next_context": next_msg["content"] if next_msg and next_msg["role"] == "assistant" else None
            }
            user_context.append(context)
            
        elif msg["role"] == "assistant":
            context = {
                "message": msg["content"],
                "prev_context": conv_list[i-1]["content"] if i > 0 else None,
                "query": conv_list[i-1]["content"] if i > 0 and conv_list[i-1]["role"] == "user" else None
            }
            assistant_context.append(context)

    # Create enhanced prompts that preserve context
    user_prompt = """
    ###INSTRUCTIONS###
    Combine all my messages into one natural message, as if I'm asking everything at once.
    Keep my original tone and style.
    
    ###FORMAT EXAMPLE### 
    "Hey, I'm working on X and need help with a few things. Could you explain how Y works? Also wondering about Z..."
    
    ###MESSAGES###
    {context}
    
    ###RETURN ONLY THE MESSAGE###
    """.format(context=json.dumps(user_context, indent=2))

    assistant_prompt = """
    ###INSTRUCTIONS###
    Combine all your responses into one natural reply that addresses everything.
    Keep your original helpful tone.
    
    ###FORMAT EXAMPLE###
    "Here's what you need to know about X. For Y, the process works like this... Regarding Z..."
    
    ###MESSAGES###
    {context}
    
    ###RETURN ONLY THE REPLY###
    """.format(context=json.dumps(assistant_context, indent=2))

    try:
        # Get user perspective with enhanced context
        user_response = ollama.chat(
            model="llama3.2:3b",
            messages=[{"role": "user", "content": user_prompt}],
            stream=False
        )
        user_summary = user_response['message']['content']
        
        # Get assistant perspective with enhanced context
        assistant_response = ollama.chat(
            model="llama3.2:3b",
            messages=[{"role": "user", "content": assistant_prompt}],
            stream=False
        )
        assistant_summary = assistant_response['message']['content']

        # Add metadata to help with future context
        user_summary = f"\n{user_summary}"
        assistant_summary = f"\n{assistant_summary}"

        return user_summary, assistant_summary

    except Exception as e:
        logging.error(f"Error in perspective summary: {e}")
        return ("Error creating user summary.", 
                "Error creating assistant summary.")

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
    # Initialize final_prompt before using it
    final_prompt = ""
    
    # Get the system prompt
    prompt = Procedural.get_prompt()
    final_prompt = " ".join(prompt.splitlines())

    Semantic.initialize_cache()  # Initialize semantic cache
    Episodic.initialize_cache()   # Initialize episodic cache
    Dreaming.initialize_cache()   # Initialize dreams cache

    # Define the modelfile with system prompt and increased context
    modelfile = f'''
    FROM huihui_ai/qwen2.5-abliterate:14b
    SYSTEM {final_prompt}
    PARAMETER num_ctx 4096
    '''
    
    try:
        # Create the custom model named 'FRED' with the system prompt and increased context
        ollama.create(model='FRED', modelfile=modelfile)
        # Start the main loop
        chat_loop()
    except Exception as e:
        print(f"Error initializing FRED: {str(e)}")

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
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# Move voice_queue to a new file called shared_resources.py
from shared_resources import voice_queue

# --- ROUTING MODEL SETUP ---
# We use the fine-tuned routing model "philschmid/modernbert-llm-router"
# which is based on BERT-base and outputs multiple classes

#Routing is broken due to 7b model not being able to handle the tool calls.
def route_query(query: str) -> str:
    """
    Routes the query to the appropriate model based on query complexity.
    Uses a simple heuristic approach:
    - Longer queries (>100 chars) or those with complex terms go to FRED_14b
    - Shorter, simpler queries go to FRED_7b
    """
    try:
        # List of terms that suggest complex queries
        complex_terms = [
            'explain', 'analyze', 'compare', 'design', 'implement',
            'optimize', 'debug', 'architecture', 'system', 'framework'
        ]
        
        # Check query complexity
        is_complex = (
            len(query) > 100 or
            any(term in query.lower() for term in complex_terms)
        )
        
        if is_complex:
            print(f"Routing complex query to FRED_14b")
            return "FRED_14b"
        else:
            print(f"Routing simple query to FRED_7b")
            return "FRED_14b"
            
    except Exception as e:
        print(f"Error in route_query: {str(e)}")
        print(f"Error type: {type(e)}")
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
        return True
    except Exception as e:
        print(f"Error saving conversation history: {str(e)}")
        return False

def process_message(user_input, ui_instance=None):
    """
    Process a single message and return the response from the model.
    If the user says "goodbye", summarize the conversation, update memories, and shut down.
    """
    # 1. Recall episodic and semantic memories.
    start_time = datetime.now()
    episodic_memories = Episodic.recall_episodic(user_input)
    semantic_memories = Semantic.recall_semantic(user_input)
    
    # Use the hybrid recall mode for dreams - balancing real and synthetic dreams
    # Adjust real_ratio based on query type for optimal results
    if any(word in user_input.lower() for word in ['fact', 'specific', 'what did', 'history', 'remember', 'recall']):
        # For factual/specific queries, prioritize real dreams
        dreams = Dreaming.recall_dreams_hybrid(user_input, top_k=3, real_ratio=0.7)
    elif any(word in user_input.lower() for word in ['imagine', 'creative', 'idea', 'possibility', 'future']):
        # For creative/imaginative queries, prioritize synthetic dreams
        dreams = Dreaming.recall_dreams_hybrid(user_input, top_k=3, real_ratio=0.3)
    else:
        # For balanced queries, use an even mix
        dreams = Dreaming.recall_dreams_hybrid(user_input, top_k=3, real_ratio=0.5)
    
    elapsed_time = datetime.now() - start_time
    print(f"Memory recall took: {elapsed_time.total_seconds():.2f} seconds")
    
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
        Dreaming.create_dream()
        Episodic.update_episodic(summary)
        Semantic.update_semantic(summary)
        Dreaming.update_dreaming()
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
        "(END OF FRED DATABASE)"
    )
    conversation.append({"role": "user", "content": user_prompt})

    # 4. Define tool schemas (unchanged from your original script)
    tools_schema = [
        {
            'type': 'function',
            'function': {
                'name': 'search_and_summarize',
                'description': (
                    "Perform a DuckDuckGo-based search for information and summarize the results. "
                    "Provide 'topics' as comma-separated search topics and 'mode' as either 'educational' or 'news'."
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'topics': {
                            'type': 'string',
                            'description': 'Comma-separated topics to search for'
                        },
                        'mode': {
                            'type': 'string',
                            'description': 'Search mode: "educational" (default) or "news"',
                            'enum': ['educational', 'news']
                        }
                    },
                    'required': ['topics']
                }
            }
        },
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

    # 5. Determine which model to use.
    model_to_use = route_query(user_input)
    print(f"Routing decision: using model {model_to_use} for input: {user_input}")

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

    try:
        # Start the voice processing thread.
        voice_thread = threading.Thread(target=process_voice_queue, daemon=True)
        voice_thread.start()
        ui.run()  # This call blocks until the window is closed.
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
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

    # Define separate modelfiles.
    modelfile_14b = f'''
FROM huihui_ai/qwen2.5-abliterate:14b
SYSTEM {final_prompt}
PARAMETER num_ctx 8192
'''
    modelfile_7b = f'''
FROM huihui_ai/qwen2.5-abliterate:7b
SYSTEM {final_prompt}
PARAMETER num_ctx 8192
'''

    try:
        # Create the two models with distinct modelfiles.
        ollama.create(model='FRED_14b', modelfile=modelfile_14b)
        ollama.create(model='FRED_7b', modelfile=modelfile_7b)
        # Start the main chat loop.
        chat_loop()
    except Exception as e:
        print(f"Error initializing FRED: {str(e)}")
import os
import time
import logging
import Task
import MorningReport
import subprocess
import platform

import psutil
import GPUtil
from duckduckgo_search import DDGS
import Semantic
import Episodic
import Dreaming
import Vision
import ollama
import base64
import cv2

logging.basicConfig(level=logging.ERROR)

def get_time() -> str:
    """
    Get the current time in a human-readable format.
    """
    current_time = time.time()
    return time.strftime("%I:%M:%S %p, %d %B %Y", time.localtime(current_time))


def search_web_information(topics: str, mode: str = "educational") -> str:
    """
    Searches for EXTERNAL information from the web using DuckDuckGo.
    This tool retrieves current information from the internet - NOT from F.R.E.D.'s memory.
    
    1. Perform text & news searches on each topic (comma-separated).
    2. Summarize the combined results in the specified style.
    3. If summarization fails, return raw search results.
    
    Args:
        topics (str): The user's desired topics, comma-separated.
        mode (str): The summarization mode - "educational" or "news".
        
    Returns:
        str: A summary or raw results if summarization fails.
    """
    ddgs = DDGS()
    
    # Split topics by commas
    topics_list = [t.strip() for t in topics.split(",") if t.strip()]
    
    region = "us-en"
    safesearch = "off"
    
    # Gather results
    text_results = []
    news_results = []
    
    # Configure search parameters based on mode
    text_max_results = 2
    news_max_results = 1 if mode == "educational" else 2
    
    # For each topic, do text + news
    for topic in topics_list:
        text_results.extend(ddgs.text(
            keywords=topic,
            region=region,
            safesearch=safesearch,
            max_results=text_max_results
        ))
        news_results.extend(ddgs.news(
            keywords=topic,
            region=region,
            safesearch=safesearch,
            max_results=news_max_results
        ))
    
    # Configure prompt based on mode
    if mode == "educational":
        combined_prompt = (
            "You are an educational summarizer. Summarize both the search results and news "
            "in a concise but thorough manner, focusing on learning and clarity. "
            "If not enough info is provided, do your best to fill in context. "
            "Structure your response with 'Text Summary:' and 'News Summary:' sections. "
            "Return ONLY the summary.\n\n"
            f"Search Results:\n{text_results}\n\n"
            f"News Results:\n{news_results}"
        )
    else:  # news mode
        combined_prompt = (
            "You are a news summarizer. Summarize both the text results and news "
            "in a journalistic style, highlighting recent or important events. "
            "Structure your response with 'Text Summary:' and 'News Summary:' sections. "
            "Return ONLY the summary:\n\n"
            f"Text Results:\n{text_results}\n\n"
            f"News Results:\n{news_results}"
        )
    
    # Fallback if summarization fails
    bare_info = (
        f"[Text Results]\n{text_results}\n\n"
        f"[News Results]\n{news_results}"
    )
    
    try:
        # Single summarization call
        response = ddgs.chat(
            keywords=combined_prompt,
            model="gpt-4o-mini",
            timeout=60
        )
        return response
    except Exception as e:
        logging.error(f"Summarization error: {e}")
        return bare_info


def get_system_status() -> str:
    """
    Get system status information: CPU usage, Memory usage, Disk usage,
    and GPU usage (if available).

    Returns:
        str: A formatted system status report.
    """
    cpu_usage = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    try:
        gpus = GPUtil.getGPUs()
        gpu_info = [f"GPU ({gpu.name}): {gpu.load*100:.1f}%" for gpu in gpus]
    except Exception as e:
        logging.error(f"GPU Info error: {e}")
        gpu_info = []

    report = [
        "💻 System Status:",
        f"CPU Usage: {cpu_usage}%",
        f"Memory Usage: {memory.percent}%",
        f"Disk Usage: {disk.percent}%"
    ]
    # Add GPU info if present
    report.extend(gpu_info)

    return "\n".join(report)


def access_memory_database(query: str, memory_type: str = "all", top_k: int = 2) -> str:
    """
    Access F.R.E.D.'s internal memory databases based on the provided query.
    This tool searches through episodic memories, semantic knowledge, and dream insights
    stored within the system - NOT external web information.
    
    Args:
        query (str): The search query to find relevant memories
        memory_type (str): Type of memory to search - "episodic", "semantic", "dreams", or "all"
        top_k (int): Number of results to return for each memory type
        
    Returns:
        str: Formatted results of the memory search
    """
    results = []
    
    if memory_type.lower() in ["episodic", "all"]:
        try:
            episodic_results = Episodic.recall_episodic(query, top_k=top_k)
            if episodic_results:
                results.append("## Episodic Memories")
                for i, memory in enumerate(episodic_results, 1):
                    results.append(f"{i}. {memory['content']}")
                results.append("")
        except Exception as e:
            results.append(f"Error searching episodic memory: {str(e)}")
    
    if memory_type.lower() in ["semantic", "all"]:
        try:
            semantic_results = Semantic.recall_semantic(query, top_k=top_k)
            if semantic_results:
                results.append("## Semantic Knowledge")
                for i, fact in enumerate(semantic_results, 1):
                    results.append(f"{i}. {fact['content']}")
                results.append("")
        except Exception as e:
            results.append(f"Error searching semantic memory: {str(e)}")
    
    if memory_type.lower() in ["dreams", "all"]:
        try:
            # Use standard recall which now always returns the top match
            dream_results = Dreaming.recall_dreams(query, top_k=1)
                
            if dream_results:
                results.append("## Dream Insights")
                for i, dream in enumerate(dream_results, 1):
                    results.append(f"{i}. {dream['content']}")
                results.append("")
        except Exception as e:
            results.append(f"Error searching dream memory: {str(e)}")
    
    if not results:
        return f"No relevant memories found for query: '{query}'"
    
    if all(r.startswith("Error") for r in results if r):
        return f"Errors occurred while searching memory: \n" + "\n".join([r for r in results if r])
    
    return "\n".join(results)


def get_sight() -> str:
    """
    Retrieve information about what FRED can currently see through the vision system.
    Uses Llama3.2-Vision via Ollama to provide an intelligent description of the camera feed.
    
    Returns:
        str: AI-powered description of what the camera sees
    """
    if not Vision.is_vision_active():
        return "Vision system is not active. Unable to provide visual information."
    
    # Get the current frame from the vision system
    frame = Vision.get_current_frame()
    if frame is None:
        return "Unable to capture a frame from the vision system."
    
    try:
        # Save the frame to a temporary file
        import tempfile
        import os
        
        # Create a temporary file with .jpg extension
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        temp_filepath = temp_file.name
        temp_file.close()
        
        # Save the frame to the temporary file
        cv2.imwrite(temp_filepath, frame)
        
        try:
            # Call Llama3.2-Vision via Ollama with correct format
            response = ollama.chat(
                model="FRED_vision",
                messages=[
                    {
                        'role': 'user',
                        'content': 'Describe this image in detail, focusing on objects, their positions, and any interactions visible.',
                        'images': [temp_filepath]
                    }
                ],
                stream=False
            )
            
            vision_description = response["message"]["content"]
            logging.info(f"Vision description successfully generated: {vision_description[:50]}...")
            
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_filepath)
            except Exception as e:
                logging.error(f"Error removing temporary file: {str(e)}")
        
        # Also include traditional object detection results
        detections = Vision.get_current_detections()
        if detections:
            object_counts = {}
            for obj in detections:
                label = obj['label']
                if label in object_counts:
                    object_counts[label] += 1
                else:
                    object_counts[label] = 1
            
            detection_result = "\n\nObject detection system reports: "
            items = []
            for label, count in object_counts.items():
                if count > 1:
                    items.append(f"{count} {label}s")
                else:
                    items.append(f"1 {label}")
            
            detection_result += ", ".join(items)
            
            return f"{vision_description}{detection_result}"
        
        return vision_description
        
    except Exception as e:
        logging.error(f"Error processing vision with Llama3.2-Vision: {str(e)}")
        
        # Fall back to regular object detection if AI vision fails
        detections = Vision.get_current_detections()
        
        if not detections:
            return "I don't see any recognizable objects at the moment."
        
        # Format the detection results
        result = "I can currently see the following objects:\n"
        
        # Group similar objects for cleaner reporting
        object_counts = {}
        for obj in detections:
            label = obj['label']
            if label in object_counts:
                object_counts[label] += 1
            else:
                object_counts[label] = 1
        
        # Format the grouped results
        items = []
        for label, count in object_counts.items():
            if count > 1:
                items.append(f"{count} {label}s")
            else:
                items.append(f"1 {label}")
        
        result += ", ".join(items)
        
        # Add a summary of the most prominent objects
        result += "\n\nThe most prominent objects in view are: "
        sorted_objects = sorted(
            [(label, count, max([obj['confidence'] for obj in detections if obj['label'] == label])) 
             for label, count in object_counts.items()],
            key=lambda x: x[2],  # Sort by confidence
            reverse=True
        )
        
        if sorted_objects:
            prominent = [f"{label} ({confidence*100:.1f}% confidence)" 
                        for label, _, confidence in sorted_objects[:3]]
            result += ", ".join(prominent)
        
        # Add information about detected persons
        persons = [obj for obj in detections if obj['label'].lower() == 'person']
        if persons:
            result += f"\n\nI can see {len(persons)} {'person' if len(persons) == 1 else 'people'} in the frame."

        # Add information about electronics
        electronics = [obj['label'] for obj in detections 
                      if obj['label'].lower() in ['laptop', 'tv', 'cell phone', 'remote', 'keyboard', 'mouse', 'monitor']]
        if electronics:
            result += f"\n\nI can see the following electronics: {', '.join(set(electronics))}."
            
        # Add information about potential text
        documents = [obj['label'] for obj in detections 
                    if obj['label'].lower() in ['book', 'cell phone', 'laptop', 'tv', 'remote']]
        if documents:
            result += f"\n\nThere may be text visible on the following objects: {', '.join(set(documents))}. "
            result += "You can ask me to attempt text recognition if needed."
        
        return result + "\n\n(Note: AI vision description failed, using fallback object detection)"



available_functions = {
    'search_web_information': search_web_information,
    'get_system_status': get_system_status,
    'create_note': Task.create_note,
    'update_note': Task.update_note,
    'delete_note': Task.delete_note,
    'read_note': Task.read_note,
    'list_notes': Task.list_notes,
    'add_task': Task.add_task,
    'delete_task': Task.delete_task,
    'list_tasks': Task.list_tasks,
    'access_memory_database': access_memory_database,
    'get_sight': get_sight,
}


def extract_and_execute(function_name, function, tool_args, param_config=None):
    """
    Generic function to extract parameters and execute functions with proper error handling
    
    Args:
        function_name (str): The name of the function being called
        function (callable): The function to execute
        tool_args (dict): Arguments passed to the function
        param_config (dict, optional): Configuration for parameter extraction
            - required (list): List of required parameter names
            - transforms (dict): Mapping of parameter names to transform functions
    
    Returns:
        tuple: (success, result_or_error_message)
    """
    try:
        # Default configuration if none provided
        if param_config is None:
            param_config = {
                'required': [],
                'transforms': {}
            }
        
        # Extract parameters with optional transformation
        params = {}
        for param_name in param_config.get('required', []):
            # Get the parameter value
            param_value = tool_args.get(param_name)
            
            # Apply transform function if specified
            transform = param_config.get('transforms', {}).get(param_name)
            if transform and param_value is not None:
                param_value = transform(param_value)
                
            # Check if required parameter is missing
            if param_value is None:
                raise ValueError(f"Missing required parameter: {param_name}")
                
            params[param_name] = param_value
            
        # Call the function with extracted parameters
        if params:
            outcome = function(**params)
        else:
            outcome = function()
            
        return True, outcome
        
    except Exception as e:
        err_msg = f"Failed calling {function_name}: {e}"
        print(err_msg)
        return False, err_msg


def handle_tool_calls(response, user_input):
    """
    Handle tool calls returned from the Ollama chat response.
    Collect multiple tool calls, run each, and append the results.
    """
    tool_calls = getattr(response.message, 'tool_calls', []) or []
    if not tool_calls:
        print("No tool calls found in the response.")
        return None

    results = []

    # Function parameter configurations
    param_configs = {
        'search_web_information': {
            'required': ['topics'],
            'transforms': {}
        },
        'get_system_status': {
            'required': [],
            'transforms': {}
        },
        'create_note': {
            'required': ['note_title', 'note_content'],
            'transforms': {'note_title': str.lower}
        },
        'update_note': {
            'required': ['note_title', 'note_content'],
            'transforms': {'note_title': str.lower}
        },
        'delete_note': {
            'required': ['note_title'],
            'transforms': {'note_title': str.lower}
        },
        'read_note': {
            'required': ['note_title'],
            'transforms': {'note_title': str.lower}
        },
        'add_task': {
            'required': ['task_title', 'task_content'],
            'transforms': {}
        },
        'delete_task': {
            'required': ['task_title'],
            'transforms': {}
        },
        'list_tasks': {
            'required': [],
            'transforms': {}
        },
        'access_memory_database': {
            'required': ['query'],
            'transforms': {}
        },
    }

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        tool_args = tool_call.function.arguments or {}
        function = available_functions.get(function_name)

        if not function:
            msg = f"Tool function '{function_name}' not found."
            print(msg)
            results.append(msg)
            continue
            
        # Special case for search functions
        if function_name in ('search_web_information', 'get_system_status'):
            # Default to user input if topics not provided
            if 'topics' not in tool_args:
                tool_args['topics'] = user_input
                
        # Use the generic extraction and execution function
        config = param_configs.get(function_name, {'required': [], 'transforms': {}})
        success, outcome = extract_and_execute(function_name, function, tool_args, config)
        results.append(outcome)

    # Return combined output if multiple calls
    if len(results) == 1:
        return results[0]
    elif len(results) > 1:
        return "\n\n".join(results)
    else:
        return None

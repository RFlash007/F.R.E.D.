import os
import time
import logging
import Task
import MorningReport

import psutil
import GPUtil
from duckduckgo_search import DDGS

import Projects

logging.basicConfig(level=logging.ERROR)

def get_time() -> str:
    """
    Get the current time in a human-readable format.
    """
    current_time = time.time()
    return time.strftime("%I:%M:%S %p, %d %B %Y", time.localtime(current_time))


def search_and_summarize(topics: str, mode: str = "educational") -> str:
    """
    Unified function to perform DuckDuckGo-based search and summarization.
    
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


available_functions = {
    'search_and_summarize': search_and_summarize,
    'get_system_status': get_system_status,
    'create_note': Task.create_note,
    'update_note': Task.update_note,
    'delete_note': Task.delete_note,
    'read_note': Task.read_note,
    'create_project': Projects.create_project,
    'delete_project': Projects.delete_project,
    'delete_file_in_project': Projects.delete_file_in_project,
    'read_file_in_project': Projects.read_file_in_project,
    'edit_file_in_project': Projects.edit_file_in_project,
    'add_task': Task.add_task,
    'delete_task': Task.delete_task,
    'list_tasks': Task.list_tasks,
    'check_expired_tasks': Task.check_expired_tasks,
    'morning_report': MorningReport.generate_morning_report,
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
        'search_and_summarize': {
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
        'create_project': {
            'required': ['project_name'],
            'transforms': {'project_name': str.lower}
        },
        'delete_project': {
            'required': ['project_name'],
            'transforms': {'project_name': str.lower}
        },
        'delete_file_in_project': {
            'required': ['project_name', 'file_name'],
            'transforms': {'project_name': str.lower, 'file_name': str.lower}
        },
        'read_file_in_project': {
            'required': ['project_name', 'file_name'],
            'transforms': {'project_name': str.lower, 'file_name': str.lower}
        },
        'edit_file_in_project': {
            'required': ['project_name', 'file_name', 'file_content'],
            'transforms': {'project_name': str.lower, 'file_name': str.lower}
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
        'check_expired_tasks': {
            'required': [],
            'transforms': {}
        },
        'morning_report': {
            'required': [],
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
        if function_name in ('search_and_summarize', 'get_system_status'):
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

import os
import time
import logging
import Notes
import Task

import psutil
import GPUtil
from duckduckgo_search import DDGS

import Projects

# NEW: Import DeepResearch
import DeepResearch

logging.basicConfig(level=logging.ERROR)

def get_time() -> str:
    """
    Get the current time in a human-readable format.
    """
    current_time = time.time()
    return time.strftime("%I:%M:%S %p, %d %B %Y", time.localtime(current_time))


def quick_learn(topics: str) -> str:
    """
    Perform a DuckDuckGo-based search for informational learning.

    1. Perform text & news searches on each topic (comma-separated).
    2. Summarize the combined results in an "educational" style.
    3. If summarization fails, return raw search results.

    Args:
        topics (str): The user's desired learning topics, comma-separated.

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

    # For each topic, do text + news
    for topic in topics_list:
        text_results.extend(ddgs.text(
            keywords=topic,
            region=region,
            safesearch=safesearch,
            max_results=2
        ))
        news_results.extend(ddgs.news(
            keywords=topic,
            region=region,
            safesearch=safesearch,
            max_results=1
        ))

    # Combine results for single summarization
    combined_prompt = (
        "You are an educational summarizer. Summarize both the search results and news "
        "in a concise but thorough manner, focusing on learning and clarity. "
        "If not enough info is provided, do your best to fill in context. "
        "Structure your response with 'Text Summary:' and 'News Summary:' sections. "
        "Return ONLY the summary.\n\n"
        f"Search Results:\n{text_results}\n\n"
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


def news(topics: str) -> str:
    """
    Perform a DuckDuckGo-based search for news topics.

    1. Perform text & news searches on each topic (comma-separated).
    2. Summarize the combined results with a news-oriented style.
    3. If summarization fails, return raw search results.

    Args:
        topics (str): The user's desired news topics, comma-separated.

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

    for topic in topics_list:
        text_results.extend(ddgs.text(
            keywords=topic,
            region=region,
            safesearch=safesearch,
            max_results=2
        ))
        news_results.extend(ddgs.news(
            keywords=topic,
            region=region,
            safesearch=safesearch,
            max_results=2
        ))

    # Combined prompt for single summarization
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
    'quick_learn': quick_learn,
    'news': news,
    'get_system_status': get_system_status,
    'create_note': Notes.create_note,
    'update_note': Notes.update_note,
    'delete_note': Notes.delete_note,
    'read_note': Notes.read_note,
    'create_project': Projects.create_project,
    'delete_project': Projects.delete_project,
    'delete_file_in_project': Projects.delete_file_in_project,
    'read_file_in_project': Projects.read_file_in_project,
    'edit_file_in_project': Projects.edit_file_in_project,
    'add_task': Task.add_task,
    'read_task': Task.read_task,
    'delete_task': Task.delete_task,
    # NEW: Add deep_research tool function
    'deep_research': DeepResearch.perform_research
}


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

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        tool_args = tool_call.function.arguments or {}
        function = available_functions.get(function_name)

        if not function:
            msg = f"Tool function '{function_name}' not found."
            print(msg)
            results.append(msg)
            continue

        # Attempt to call the function with the provided arguments
        if function_name in ('quick_learn', 'news'):
            # e.g. "topics" is expected
            topics = tool_args.get('topics', user_input)
            try:
                outcome = function(topics)
                results.append(outcome)
            except Exception as e:
                err_msg = f"Failed calling {function_name}: {e}"
                print(err_msg)
                results.append(err_msg)

        elif function_name == 'get_system_status':
            try:
                system_report = function()
                results.append(system_report)
            except Exception as e:
                err_msg = f"Failed calling get_system_status: {e}"
                print(err_msg)
                results.append(err_msg)
        elif function_name in ('create_note', 'update_note'):
            try:
                note_title = tool_args.get('note_title').lower()
                note_content = tool_args.get('note_content')
                if note_title is None or note_content is None:
                    raise ValueError("Missing required note_title or note_content")
                outcome = function(note_title, note_content)
                results.append(outcome)
            except Exception as e:
                err_msg = f"Failed calling {function_name}: {e}"
                print(err_msg)
                results.append(err_msg)
        elif function_name in ('delete_note', 'read_note'):
            try:
                note_title = tool_args.get('note_title').lower()
                if note_title is None:
                    raise ValueError("Missing required note_title")
                outcome = function(note_title)
                results.append(outcome)
            except Exception as e:
                err_msg = f"Failed calling {function_name}: {e}"
                print(err_msg)
                results.append(err_msg)
        elif function_name in ('create_project', 'delete_project'):
            try:
                project_name = tool_args.get('project_name').lower()
                if project_name is None:
                    raise ValueError("Missing required project_name")
                outcome = function(project_name)
                results.append(outcome)
            except Exception as e:
                err_msg = f"Failed calling {function_name}: {e}"
                print(err_msg)
                results.append(err_msg)
        elif function_name in ('delete_file_in_project', 'read_file_in_project'):
            try:
                project_name = tool_args.get('project_name').lower()
                file_name = tool_args.get('file_name').lower()
                outcome = function(project_name, file_name)
                results.append(outcome)
            except Exception as e:
                err_msg = f"Failed calling {function_name}: {e}"
                print(err_msg)
                results.append(err_msg)
        elif function_name in ('edit_file_in_project'):
            try:
                project_name = tool_args.get('project_name').lower()
                file_name = tool_args.get('file_name').lower()
                file_content = tool_args.get('file_content')
                outcome = function(project_name, file_name, file_content)
                results.append(outcome)
            except Exception as e:
                err_msg = f"Failed calling {function_name}: {e}"
                print(err_msg)
                results.append(err_msg)
        elif function_name == 'add_task':
            try:
                task_title = tool_args.get('task_title')
                task_content = tool_args.get('task_content')
                if task_title is None or task_content is None:
                    raise ValueError("Missing required task_title or task_content")
                outcome = function(task_title, task_content)
                results.append(outcome)
            except Exception as e:
                err_msg = f"Failed calling add_task: {e}"
                print(err_msg)
                results.append(err_msg)
        elif function_name == 'read_task':
            try:
                outcome = function()
                results.append(outcome)
            except Exception as e:
                err_msg = f"Failed calling read_task: {e}"
                print(err_msg)
                results.append(err_msg)
        elif function_name == 'delete_task':
            try:
                task_title = tool_args.get('task_title')
                if task_title is None:
                    raise ValueError("Missing required task_title")
                outcome = function(task_title)
                results.append(outcome)
            except Exception as e:
                err_msg = f"Failed calling delete_task: {e}"
                print(err_msg)
                results.append(err_msg)
        elif function_name == 'deep_research':
            try:
                research_query = tool_args.get('research_query')
                if research_query is None:
                    raise ValueError("Missing required research_query")
                outcome = function(research_query)
                results.append(outcome)
            except Exception as e:
                err_msg = f"Failed calling deep_research: {e}"
                print(err_msg)
                results.append(err_msg)
        else:
            msg = f"Unhandled function: {function_name}"
            print(msg)
            results.append(msg)

    # Return combined output if multiple calls
    if len(results) == 1:
        return results[0]
    elif len(results) > 1:
        return "\n\n".join(results)
    else:
        return None

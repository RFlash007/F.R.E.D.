import os
import time
import logging

import psutil
import GPUtil
from duckduckgo_search import DDGS

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

    # Attempt summarization
    text_summary_prompt = (
        "You are an educational summarizer. Summarize these search results "
        "in a concise but thorough manner, focusing on learning and clarity. "
        "If not enough info is provided, do your best to fill in context. "
        "Return ONLY the summary:\n"
        f"{text_results}"
    )
    news_summary_prompt = (
        "You are an educational summarizer. Summarize these news results "
        "in a concise but thorough manner, focusing on learning and clarity. "
        "Return ONLY the summary:\n"
        f"{news_results}"
    )

    # Fallback if summarization fails
    bare_info = (
        f"[Text Results]\n{text_results}\n\n"
        f"[News Results]\n{news_results}"
    )

    try:
        # Summarize text results
        text_response = ddgs.chat(
            keywords=text_summary_prompt,
            model="gpt-4o-mini",
            timeout=60
        )
        # Summarize news results
        news_response = ddgs.chat(
            keywords=news_summary_prompt,
            model="gpt-4o-mini",
            timeout=60
        )
    except Exception as e:
        logging.error(f"Summarization error: {e}")
        return bare_info

    return f"Text Summary:\n{text_response}\n\nNews Summary:\n{news_response}"


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

    # Attempt summarization
    text_summary_prompt = (
        "You are a news summarizer. Summarize these text results in a "
        "journalistic style, highlighting recent or important events. "
        "Return ONLY the summary:\n"
        f"{text_results}"
    )
    news_summary_prompt = (
        "You are a news summarizer. Summarize these news results with a "
        "journalistic approach. Return ONLY the summary:\n"
        f"{news_results}"
    )

    # Fallback if summarization fails
    bare_info = (
        f"[Text Results]\n{text_results}\n\n"
        f"[News Results]\n{news_results}"
    )

    try:
        text_response = ddgs.chat(
            keywords=text_summary_prompt,
            model="gpt-4o-mini",
            timeout=60
        )
        news_response = ddgs.chat(
            keywords=news_summary_prompt,
            model="gpt-4o-mini",
            timeout=60
        )
    except Exception as e:
        logging.error(f"Summarization error: {e}")
        return bare_info

    return f"Text Summary:\n{text_response}\n\nNews Summary:\n{news_response}"


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

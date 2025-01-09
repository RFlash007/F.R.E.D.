import os
import ollama
#from typing import List
import time
from duckduckgo_search import DDGS
import psutil
import GPUtil

def get_time()->str:
    """
    Get the time in human-readable format
    """
    current_time = time.time()
    return time.strftime("%I:%M:%S %p, %d %B %Y", time.localtime(current_time))

def quick_learn(topics) -> str:
    """
    Learn about anything from DuckDuckGo with search.

    Args:
        topics (str): The topics to learn about

    Returns:
        str: Curated information about the topics from multiple sources
    """
    #If the user provides a topic use it if not use default topics
    ddgs = DDGS()
    if topics:
        try:
            topics = ddgs.chat(
                keywords=f"""Based on this message: ({topics}), write exactly three phrases to search google for In order to find what the user is asking for. If the user ask for todays news use stuff like: Technology, Science, Presidents, and war. Write just the keywords or phrases, no other text. Write commas in between every keyword or phrase
                for example: OpenAI, War in Ukraine, etc. ***ONLY RETURN THREE PHRASES OR KEYWORDS***""",
                model="gpt-4o-mini",  # or "claude-3-haiku", "llama-3.1-70b", etc.
                timeout=60           # optional, defaults to 30 seconds
            )
        except Exception as e:
            print(f"Error generating keywords: {e}")
            return "Error generating keywords"

    #remove commas from the topics and put into array for duckduckgo search
    topics = topics.split(",")
    topics = [topic.strip() for topic in topics]
    print(topics)
    #set settings for duckduckgo search
    region = "us-en"
    safesearch = "off"

    #News and Text Search for all topics
    search_results = []
    news_results = []
    for topic in topics:
        search_results.extend(ddgs.text(
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

    #Summarize the text
    text_summary_prompt = f"You are a information summarizer, include every detail of the text results but make it as concise as possible. Return just the summary, no other text. Here is the information to summarize: {search_results}"
    #Summarize the news
    news_summary_prompt = f"You are a information summarizer, include every detail of the news results but make it as concise as possible. Return just the summary, no other text. Here is the information to summarize: {news_results}"

   
    #Storing non Summary of info in case AI fails to generate summary
    bare_info = (f"\nText Results: {search_results}\nNews Results: {news_results}")

    #Summarize the text
    try:
        text_response = ddgs.chat(
            keywords=text_summary_prompt,
            model="gpt-4o-mini",  # or "claude-3-haiku", "llama-3.1-70b", etc.
            timeout=60           # optional, defaults to 30 seconds
        )
    except Exception as e:
        return bare_info
    try:
        news_response = ddgs.chat(
            keywords=news_summary_prompt,
            model="gpt-4o-mini",  # or "claude-3-haiku", "llama-3.1-70b", etc.
            timeout=60           # optional, defaults to 30 seconds
        )
    except Exception as e:
        return bare_info
    
    return f"\nText Summary: {text_response}\nNews Summary: {news_response}"

def news(user_input) -> str:
    """
    Get the news from the user input
    """
    return quick_learn(user_input)

def verify_memories() -> str:
    """
    Goes through Models memory database and verifies that the formatting of the data is correct.
    Verifies the memories.

    Returns:
        A String stating whether the verification was successful or not.
    """
    print("Verifying Memories...")
    # 1) Verify SEMANTIC.TXT
    with open("Semantic.txt", 'r+', encoding='utf-8') as file:
        content = file.read()

        semantic_verification_prompt = f"""You are a database verification bot. Your task is to verify the formatting of data in a database.
        Ensure that the following information is in the required format. Re-write the data in the required format if it is not correct.
        If there are no errors, simply return all the original data in the required format. **Include NO extra dialogue or text** — only return the data in the correct format.
        Here is the required format:
        [PERSONAL] John is allergic to peanuts
        [PREFERENCE] John prefers tea over coffee
        [TECHNICAL] Python was created by Guido van Rossum
        [LOCATION] John lives in Seattle

        Here is the data:
        {content}
        """

        messages = [{"role": "user", "content": semantic_verification_prompt}]
        response = ollama.chat(model="huihui_ai/qwen2.5-abliterate:14b", messages=messages)

        file.seek(0)
        # If LLM returns a string:
        file.write(response["message"]["content"])
        file.truncate()
    print("Semantic Verification Complete")
    # 2) Verify EPISODIC.TXT
    with open("Episodic.txt", 'r+', encoding='utf-8') as file:
        content = file.read()

        episodic_verification_prompt = """You are a database verification bot. Verify and reformat the following memory entries.
        Required format for each memory.
        1. **IF THERE IS ANY EXTRA TEXT OR DIALOGUE, REMOVE IT**
        2. **RETURN ONLY THE RESULT**
        3. **WRITE NO EXTRA TEXT OR DIALOGUE**

        REQUIRED FORMAT:       
        [
            "timestamp": "YYYY-MM-DD HH:MM",
            "tags": ["tag1", "tag2"],
            "summary": "Brief conversation summary",
            "insights": {
                "positive": "What worked well",
                "negative": "What to improve",
                "learned": "Key learnings"
            }
        ]\n\n(Another memory)\n\n(Another memory) etc...
        
        Here is the data to verify:
        """ + content

        messages = [{"role": "user", "content": episodic_verification_prompt}]
        response = ollama.chat(model="huihui_ai/qwen2.5-abliterate:14b", messages=messages)

        file.seek(0)
        # If LLM returns a string:
        file.write(response["message"]["content"])
        file.truncate()
    # (e.g., try to parse the JSON, check for certain keys).
    print("Episodic Verification Complete")
    return "Memory Verification was a Success"

def get_system_status() -> str:
    """
    Get system status information
    
    Returns:
        str: Formatted system status report
    """
    cpu_usage = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
        
    try:
        gpus = GPUtil.getGPUs()
        gpu_info = [{'name': gpu.name, 'load': gpu.load*100} for gpu in gpus]
    except:
        gpu_info = []
        
    try:
        gpus = GPUtil.getGPUs()
        gpu_info = [{'name': gpu.name, 'load': gpu.load*100} for gpu in gpus]
    except:
        gpu_info = []
            
    
    status = {'cpu': cpu_usage,
    'memory': memory.percent,
    'disk': disk.percent,
    'gpu': gpu_info}
    
    report = ""
    report = "💻 System Status:\n\n"
    report += f"CPU Usage: {status['cpu']}%\n"
    report += f"Memory Usage: {status['memory']}%\n"
    report += f"Disk Usage: {status['disk']}%\n"
    
    if status['gpu']:
        for gpu in status['gpu']:
            report += f"GPU ({gpu['name']}): {gpu['load']:.1f}%\n"
            
    return report

def create_project(user_input) -> str:
    """
    Create a new project

    Args:
        project_name (str): The name of the project to create

    Returns:
       (str): A message stating whether the project was created successfully or not.
    """
    #Get the project name from the user input
    prompt = f"""1. Get the project name from this messasge
                 2. **ONLY RETURN THE PROJECT NAME NO OTHER TEXT OR DIALOGUE**
                 For example if the user says "Create a project called 'My Project'" return "My Project"
                Message: {user_input}"""
    messages = [{"role": "user", "content": prompt}]
    response = ollama.chat(model="llama3.2:3b", messages=messages)
    project_name = response["message"]["content"]

    try:
        os.chdir('Projects')
        os.mkdir(project_name.lower())
        os.chdir('..')
        return f"Project {project_name.lower()} has beencreated"
    except Exception as e:
        return f"Failed to create project: {e}"

def open_project(user_input) -> str:
    """
    Open a project

    Args:
        project_name (str): The name of the project to open

    Returns:
       (str): A message stating whether the project was opened successfully or not.
    """
    #Get the project name from the user input
    prompt = f"""1. Get the project name from this messasge
                 2. **ONLY RETURN THE PROJECT NAME NO OTHER TEXT OR DIALOGUE**
                 For example if the user says "Open a project called 'My Project'" return "My Project"
                Message: {user_input}"""
    messages = [{"role": "user", "content": prompt}]
    response = ollama.chat(model="llama3.2:3b", messages=messages)
    project_name = response["message"]["content"]

    try:
        os.chdir('Projects')
        with open(project_name.lower(), 'r') as file:
            content = file.read()
        return f"Project content {content} has been opened"
    except Exception as e:
        return f"Failed to open project: {e}"
    
# Map function names to actual functions
available_functions = {
    'quick_learn': quick_learn,
    'verify_memories': verify_memories,
    'get_system_status': get_system_status,
    'create_project': create_project,
    'news': news,
    'open_project': open_project
}

def handle_tool_calls(response, user_input):
    """
    Handle tool calls returned from the Ollama chat response.

    Args:
        response: The response object from Ollama chat.
    """

    #Get the tool calls from the response
    tool_calls = getattr(response.message, 'tool_calls', []) or []
    if not tool_calls:
        print("No tool calls found in the response.")
        return

    for tool in tool_calls:
        function_name = tool.function.name
        function_to_call = available_functions.get(function_name)
        #determine which function to call
        if not function_to_call:
            print(f"Function not found: {function_name}")
            continue
        elif function_name == 'quick_learn':
            try:
                learned_data = function_to_call(user_input)
                return learned_data
            except Exception as e:
                print("Failed to learn")
        elif function_name == 'verify_memories':
            return verify_memories()
        elif function_name == 'get_system_status':
            return get_system_status()
        elif function_name == 'create_project':
            return create_project(user_input)
        elif function_name == 'open_project':
            return open_project(user_input)
        elif function_name == 'news':
            try:
                learned_data = function_to_call(user_input)
                return learned_data
            except Exception as e:
                print("Failed to learn")
        else:
            print(f"Unhandled function: {function_name}")

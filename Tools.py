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
    Learn about anything from DuckDuckGo.

    Args:
        topic (str): The topic to learn about.

    Returns:
        str: Various information about the topic from DuckDuckGo
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
    max_results = 1

    #News and Text Search for all topics
    search_results = []
    news_results = []
    for topic in topics:
        search_results.extend(ddgs.text(
            keywords=topic,
            region=region,
            safesearch=safesearch,
            max_results=max_results
        ))
        news_results.extend(ddgs.news(
            keywords=topic,
            region=region,
            safesearch=safesearch,
            max_results=max_results
        ))

    #Prompt for AI to summarize the DuckDuckGo search results
    summary_query = (f"You are a web search information summarizer, your job is to get as much information in a very concise format\n"
                     f"Here are the results: Text Results:\n{search_results}\n\nNews Results:\n{news_results}")

    #Storing non Summary of info in case AI fails to generate summary
    bare_info = (f"Text Results:\n{search_results}\n\nNews Results:\n{news_results}")

    # Use AI to summarize all info
    try:
        response = ddgs.chat(
            keywords=summary_query,
            model="gpt-4o-mini",  # or "claude-3-haiku", "llama-3.1-70b", etc.
            timeout=60           # optional, defaults to 30 seconds
        )
    except Exception as e:
        return bare_info
    return response

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
        fact: information
        fact: information
        fact: information
        
        Here is the data:
        {content}
        """

        messages = [{"role": "user", "content": semantic_verification_prompt}]
        response = ollama.chat(model="huihui_ai/qwen2.5-abliterate:14b", messages=messages)

        file.seek(0)
        # If LLM returns a string:
        file.write(response["message"]["content"])
        file.truncate()

    # 2) Verify EPISODIC.TXT
    with open("Episodic.txt", 'r+', encoding='utf-8') as file:
        content = file.read()

        episodic_verification_prompt = f"""You are a database verification bot. Your task is to verify the formatting of data in a database.
        Ensure that the following information is in the required format. Re-write the data in the required format if it is not correct.
        If there are no errors, simply return all the original data in the required format. **Include NO extra dialogue or text** — only return the data in the correct format.
        
        Here is the required JSON format (example):
        
        {{
          "memory_timestamp": "03 January 2025",
          "context_tags": [
            "raspberry-pi-discussion",
            "project-guidance",
            "hardware-acquisition"
          ],
          "conversation_summary": "User recently acquired a Raspberry Pi...",
          "what_worked": "Discussing multiple project setup options kept the conversation productive.",
          "what_to_avoid": "Clarify ambiguous inputs like 'q' to avoid confusion.",
          "what_you_learned": "User prefers hands-on learning and is interested in beginner-friendly projects."
        }}
        
        {{
          "memory_timestamp": "04 January 2025",
          "context_tags": [
            "RaspberryPiProjects",
            "HandsOnLearning"
          ],
          "conversation_summary": "User discussed setting up projects...",
          ...
        }}
        
        Here is the data:
        {content}
        """

        messages = [{"role": "user", "content": episodic_verification_prompt}]
        response = ollama.chat(model="huihui_ai/qwen2.5-abliterate:14b", messages=messages)

        file.seek(0)
        # If LLM returns a string:
        file.write(response["message"]["content"])
        file.truncate()
    # (e.g., try to parse the JSON, check for certain keys).
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

# Map function names to actual functions
available_functions = {
    'quick_learn': quick_learn,
    'verify_memories': verify_memories,
    'get_system_status': get_system_status
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
        else:
            print(f"Unhandled function: {function_name}")

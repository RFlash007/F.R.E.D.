import logging
import os
import json
from datetime import datetime

import ollama

import Chat
from Episodic import Episode
import Episodic
import Tools

# Read conversation history
def read_conversation_history(directory='conversation_history'):
    """Read all conversation history files in the specified directory."""
    conversations = []
    
    if not os.path.exists(directory):
        print(f"Directory {directory} not found")
        return conversations
    
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    conversations.append(f.read())
            except Exception as e:
                print(f"Error reading {filename}: {str(e)}")
    
    return conversations

def summarize_augmented_conversation(conversation: str) -> str:
    """
    Analyze the augmented conversation to extract facts and inferences about both user and assistant.
    Focus on personality traits, preferences, behaviors, and patterns.
    """
    summary_prompt = f"""Analyze this conversation and extract any possible facts, implications, or educated guesses about both the user and assistant. Focus on:

1. User traits:
   - Personality characteristics
   - Technical skills and knowledge
   - Preferences and interests
   - Work or study habits
   - Environmental clues
   - Communication style

2. Assistant traits:
   - Personality development
   - Knowledge specialties
   - Interaction patterns
   - Decision-making approach
   - Learning adaptations
   - Communication preferences

Return a detailed analysis that includes both explicit facts and reasoned inferences.

Conversation:
{conversation}"""

    response = ollama.chat(
        model="huihui_ai/qwen2.5-abliterate:14b",
        messages=[{"role": "user", "content": summary_prompt}],
        stream=False,
        format="json"
    )
    
    return response['message']['content']

#create "dreams"
def create_dream():
    """Create dreams based on conversation history and save them"""
    if not os.path.exists('dreams'):
        os.makedirs('dreams')

    conversations = read_conversation_history()
    
    for conversation in conversations:
        # Create a more detailed prompt for better dream generation
        dream_prompt = f"""Continue and expand this conversation naturally, focusing on revealing more about both participants. Your continuation should:

1. User Insights:
   - Make educated guesses about their background, skills, and environment
   - Infer their preferences, habits, and working style
   - Expand on any mentioned interests or expertise
   - Add realistic personal details that fit their profile

2. Assistant Evolution:
   - Develop deeper personality traits and quirks
   - Show growth and learning from the interaction
   - Demonstrate specialized knowledge areas
   - Express unique perspectives and approaches

3. Interaction Dynamics:
   - Maintain the existing conversation style
   - Add natural back-and-forth that reveals more about both parties
   - Include subtle details that hint at ongoing relationship development

Continue the conversation naturally while incorporating these elements:
{conversation}"""
        
        response = ollama.chat(
            model="huihui_ai/qwen2.5-abliterate:14b",
            messages=[{"role": "user", "content": dream_prompt}],
            stream=False,
            format="json",
            options={"temperature": 1.5}
        )
        
        dream_continuation = response['message']['content']
        augmented_conversation = f"{conversation}\n{dream_continuation}"
        print(augmented_conversation)
        # Save augmented conversation with simple newline separation
        dreams_file = os.path.join('dreams', 'dreams.txt')
        with open(dreams_file, 'a', encoding='utf-8') as f:
            f.write(f"{augmented_conversation}\n\n")
        
        # Create summary focusing on user and assistant traits
        conversation_summary = summarize_augmented_conversation(augmented_conversation)
        create_episodic_dream(conversation_summary)

def create_episodic_dream(memory: str) -> str:
    """
    Extract relevant episodic information from the conversation
    and store it in dreams/dream.txt using the Episode class.
    """
    try:
        time = Tools.get_time()
        episode_extraction_prompt = f"""Extract key insights and create an episodic memory from this conversation summary.
TIME: {time}
SUMMARY:
{memory}

INSTRUCTIONS:
Create a JSON object that captures the key learnings and insights with these fields:
1. "memory_timestamp": Current time
2. "context_tags": Array of relevant topics, traits, or themes discovered
3. "conversation_summary": Brief overview of key interaction points
4. "what_worked": Successful interaction patterns and approaches identified
5. "what_to_avoid": Potential friction points or areas for improvement
6. "what_you_learned": New insights about:
   - User preferences and traits
   - Assistant capabilities and personality
   - Interaction dynamics and patterns
   - Environmental or contextual details

Focus on extracting both explicit facts and reasoned inferences about both parties.
"""

        response = ollama.chat(
            model="huihui_ai/qwen2.5-abliterate:14b",
            messages=[{"role": "user", "content": episode_extraction_prompt}],
            format="json",
            options={"temperature": 0}
        )

        if not response.get("message", {}).get("content"):
            logging.warning("No content in AI response")
            return ""

        content = response["message"]["content"]

        try:
            episodes_data = json.loads(content)
            if not isinstance(episodes_data, list):
                episodes_data = [episodes_data]

            required_fields = {
                "memory_timestamp", "context_tags", "conversation_summary",
                "what_worked", "what_to_avoid", "what_you_learned"
            }

            with open('dreams/dream.txt', 'a', encoding='utf-8') as file:
                for data in episodes_data:
                    if not required_fields.issubset(data.keys()):
                        logging.error(f"Missing fields from: {required_fields}")
                        continue

                    for key in ["what_worked", "what_to_avoid", "what_you_learned"]:
                        if isinstance(data.get(key), list):
                            data[key] = "\n".join(data[key])

                    episode = Episodic.Episode(
                        memory_timestamp=data["memory_timestamp"],
                        context_tags=data["context_tags"],
                        conversation_summary=data["conversation_summary"],
                        what_worked=data["what_worked"],
                        what_to_avoid=data["what_to_avoid"],
                        what_you_learned=data["what_you_learned"]
                    )
                    file.write(episode.to_json() + "\n")

        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON: {e}")
            return ""
        except Exception as e:
            logging.error(f"Error processing episode: {e}")
            return ""

        return content

    except Exception as e:
        logging.error(f"Error in create_episodic_dream: {e}")
        return ""

if __name__ == "__main__":
    create_dream()

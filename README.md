# F.R.E.D. - Frankly Rude Educated Droid

A conversational AI assistant that remembers your interactions and occasionally delivers information with attitude.

## What is F.R.E.D.?

F.R.E.D. is an AI assistant that learns from your conversations, maintains context between sessions, and provides personalized assistance with a touch of sass. Unlike standard assistants, F.R.E.D. remembers what you've discussed before and becomes more helpful (though not necessarily more polite) over time.

## Daily Briefing

F.R.E.D. provides an automatic morning report each time you start the application, including:

- **Current Weather**: Temperature, conditions, and forecast for your location
- **Task Management**: F.R.E.D. automatically cleans up your task list on startup by removing any task more than 3 days past its due date, then shows your current active tasks
- **News Summaries**: Curated news in several categories:
  - World News
  - Technology News
  - AI Industry News
  - General Headlines

This ensures you're always up-to-date with essential information without having to ask.

## Vision System

F.R.E.D. is equipped with a computer vision system using the state-of-the-art YOLOv8s model that provides visual awareness through your webcam:

- **Object Detection**: Identifies over 600 types of objects using the Open Images V7 dataset with improved accuracy (upgraded from YOLOv8n)
- **Object Tracking**: Maintains persistent tracking of objects between frames with unique identifiers
- **Person Detection & Recognition**: Identifies people with high accuracy and remembers faces with custom naming
- **Face Database**: Stores identified faces in a SQLite database for automatic recognition in future sessions
- **Motion Analysis**: Tracks object movements and calculates simple velocity data for moving objects
- **Real-time Processing**: Analyzes the visual feed continuously while active
- **Comprehensive Summaries**: Provides object lists with confidence scores, counts, and contextual information

### Vision Architecture

The vision system employs a two-stage architecture for comprehensive scene understanding:
1. **Objective Description Layer**: Uses a dedicated LLM (FRED_vision model) to produce neutral, precise descriptions focusing on spatial relationships and object properties
2. **FRED Personality Layer**: Interprets the neutral descriptions through FRED's characteristic British humor and sarcasm

This separation ensures reliable object detection while maintaining FRED's distinctive tone in responses.

### Upcoming Vision Features

The vision system is continuously evolving with planned enhancements including:
- Voice-activated visual searches and object queries
- Enhanced information display with smart annotations for objects
- Advanced text recognition with multi-language translation (planned for future release)
- Document scanning and analysis capabilities (planned for future release)

To use the vision system, simply ask questions like:
- "What can you see right now?"
- "What objects are in front of you?"
- "Describe what you're looking at."
- "Do you recognize anyone in the camera?"

The vision system automatically initializes when F.R.E.D. starts up.

## Memory Systems

F.R.E.D. uses three types of memory:

### Semantic Memory
Stores facts and information
```
"Python is a high-level programming language, but you probably knew that already."
```

### Episodic Memory
Remembers your conversations and interactions
```
"Last time we talked about website design. You seemed to prefer minimalist designs, though your taste is questionable."
```

### Dreaming
Develops insights about your preferences and patterns, providing the single most relevant insight for any query
```
"I've noticed you tend to break down problems systematically. Not a bad approach, for a human."
```

## How to Use F.R.E.D.

### Basic Commands

Simply talk to F.R.E.D. as you would any assistant. Some examples:

**General Questions**
```
You: Tell me about quantum computing.
F.R.E.D.: *explains quantum computing with a hint of condescension*
```

**Web Searches**
```
You: Search for the latest news about AI.
F.R.E.D.: *finds and summarizes news, possibly with commentary*
```

**Vision Queries**
```
You: What objects can you see?
F.R.E.D.: *describes objects detected in the camera view with characteristic dry wit*
```

**Managing Tasks**
```
You: Add a task to call my dentist tomorrow with a due date next Friday.
F.R.E.D.: Added "Call dentist" to your tasks with a due date of Friday. Try not to forget this time.
```

### Available Tools

F.R.E.D. can help you with:

- **Web Information**: Finding and summarizing current information from the internet
- **Memory Access**: Retrieving past conversations, facts, and insights from F.R.E.D.'s memory databases
- **Notes**: Creating and managing notes
- **Tasks**: Tracking your to-do list with due dates (old tasks are automatically cleaned up)
- **System Status**: Monitoring your computer's performance
- **Vision**: Identifying and describing objects through the webcam

### Memory Management

F.R.E.D. automatically manages memories, but you can:

- Ask what F.R.E.D. remembers about specific topics
- Request to forget certain conversations
- Say "goodbye" at the end of sessions to ensure memories are processed
- Access memory databases with specific queries (e.g., "What do you remember about our discussions on programming?")

## Tips for Getting the Most Out of F.R.E.D.

1. **Be specific** with your questions
2. **End conversations** with "goodbye" so F.R.E.D. can process memories
3. **Don't take it personally** when F.R.E.D. is a bit rude - it's in the name
4. **Use the tools** for organizing information and tasks
5. **Trust the automatic task management** - tasks older than 3 days past their due date are automatically cleaned up on startup
6. **Provide feedback** when F.R.E.D. misunderstands you
7. **Ask about your surroundings** to leverage the vision system

## Common Commands

| What You Want | What to Say |
|---------------|-------------|
| Search the web | "Search for..." or "Find information about..." |
| Access F.R.E.D.'s memories | "What do you remember about..." or "Recall our discussions on..." |
| Create a note | "Create a note titled..." |
| Add a task | "Add a task..." |
| Add a task with due date | "Add a task... with due date..." |
| List your tasks | "List my tasks" or "Show my tasks" |
| Delete a task | "Delete task..." |
| Check system status | "How's my system doing?" |
| Check what F.R.E.D. can see | "What do you see?" or "What objects are in view?" |
| End a session | "Goodbye" |

---

F.R.E.D. gets better the more you use it. Despite the attitude, it's here to help. 
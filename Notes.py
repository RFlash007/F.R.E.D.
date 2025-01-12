import os
from contextlib import chdir


def create_note(note_title: str, note_content: str) -> str:
    """
    Create a new note with the given title and content.

    Args:
        note_title (str): The title of the note.
        note_content (str): The content of the note.

    Returns:
        str: A summary or raw results if summarization fails.
    """
    with chdir("Notes"):
        with open(f"{note_title}.txt", "w") as file:
            file.write(note_content)
        return f"Note '{note_title}' created successfully."


def update_note(note_title: str, note_content: str) -> str:
    """
    Update an existing note with the given title and content.

    Args:
        note_title (str): The title of the note to update.
        note_content (str): The new content for the note.

    Returns:
        str: A message indicating the success of the update.
    """
    with chdir("Notes"):
        with open(f"{note_title}.txt", "w") as file:
            file.write(note_content)
        return f"Note '{note_title}' updated successfully."


def read_note(note_title: str) -> str:
    """
    Read the content of an existing note.
    """
    with chdir("Notes"):
        with open(f"{note_title}.txt", "r") as file:
            return file.read()


def delete_note(note_title: str) -> str:
    """
    Delete an existing note.

    Args:
        note_title (str): The title of the note to delete.

    Returns:
        str: A message indicating the success of the deletion.
    """
    with chdir("Notes"):
        os.remove(f"{note_title}.txt")
        return f"Note '{note_title}' deleted successfully."


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
    }
]

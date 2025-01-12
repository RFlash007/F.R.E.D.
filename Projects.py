import os
import mimetypes
from pathlib import Path
from contextlib import chdir

import ollama

def create_project(project_name: str) -> str:
    """
    Create a new project directory.

    Args:
        project_name (str): The name of the project.

    Returns:
        str: A message indicating the success of the creation.
    """
    os.makedirs(f"Projects/{project_name}", exist_ok=True)
    return f"Project '{project_name}' created successfully."

def edit_file_in_project(project_name: str, file_name: str, file_content: str) -> str:
    """
    Edit a specific file within a project.

    Args:
        project_name (str): The name of the project.
        file_name (str): The name of the file to edit.
        file_content (str): directions for what to write to the file.

    Returns:
        str: A message indicating the success of the edit.
    """
    #get python code from model
    code_prompt = f"""
You are tasked with writing Python code according to the following requirements.

### Instructions:
- Write the code to be clear, well-structured, and easy to follow.
- Use helpful comments to explain each significant part of the code.
- Think step by step; ensure the code is robust, efficient, and follows best practices.
- Return **only the final code**—no additional text or explanations.

### Requirements:
{file_content}

Take the time to produce accurate, well-documented Python code that executes as intended.
    """
    print(file_content)

    response = ollama.chat(
        model="huihui_ai/qwen2.5-coder-abliterate:32b",
        messages=[{"role": "user", "content": code_prompt}],
    )
    file_content = response['message']['content']
    with chdir(f"Projects/{project_name}"):
        file_name = Path(file_name).with_suffix('.py')
        with open(file_name, "w") as file:
            file.write(file_content)
        return f"File '{file_name}' in project '{project_name}' updated successfully."

def read_file_in_project(project_name: str, file_name: str) -> str:
    """
    Read a specific file within a project.

    Args:
        project_name (str): The name of the project.
        file_name (str): The name of the file to read.

    Returns:
        str: The content of the file.
    """
    with chdir(f"Projects/{project_name}"):
        file_name = Path(file_name).with_suffix('.py')
        with open(file_name, "r") as file:
            return file.read()

def delete_project(project_name: str) -> str:
    """
    Delete an entire project directory.

    Args:
        project_name (str): The name of the project to delete.

    Returns:
        str: A message indicating the success of the deletion.
    """
    os.rmdir(f"Projects/{project_name}")
    return f"Project '{project_name}' deleted successfully."

def delete_file_in_project(project_name: str, file_name: str) -> str:
    """
    Delete a specific file within a project.

    Args:
        project_name (str): The name of the project.
        file_name (str): The name of the file to delete.

    Returns:
        str: A message indicating the success of the deletion.
    """
    with chdir(f"Projects/{project_name}"):
        file_name = Path(file_name).with_suffix('.py')
        os.remove(file_name)
        return f"File '{file_name}' in project '{project_name}' deleted successfully."


tools_schema = [
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
                        'description': 'New content for the file'
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
    }
]


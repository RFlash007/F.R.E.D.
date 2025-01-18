import os
from contextlib import chdir


def add_task(task_title: str, task_content: str) -> str:
    """
    Add a task to the task list.

    Args:
        task_title (str): The title of the task.
        task_content (str): The content of the task.

    Returns:
        str: A string describing if the task was added successfully.
    """
    with chdir("Tasks"):
        # Create Tasks directory if it doesn't exist
        os.makedirs("Tasks", exist_ok=True)
        
        if not os.path.exists("task.txt"):
            with open("task.txt", "w") as _:
                pass
                
        with open("task.txt", "a") as file:
            file.write(f"\n\ntask: {task_title}\n{task_content}")
            return f"Task '{task_title}' created successfully."


def read_task() -> str:
    """
    Read all tasks from the task list.

    Returns:
        str: All tasks if any exist, or an error message if no tasks exist.
    """
    with chdir("Tasks"):
        if not os.path.exists("task.txt"):
            return "No tasks found."
            
        with open("task.txt", "r") as file:
            tasks = file.read()
            return tasks.strip() if tasks else "No tasks found."


def delete_task(task_title: str) -> str:
    """
    Delete an existing task.

    Args:
        task_title (str): The title of the task to delete.

    Returns:
        str: A message indicating the success of the deletion.
    """
    with chdir("Tasks"):
        if not os.path.exists("task.txt"):
            return "No tasks found."
            
        new_tasks = []
        with open("task.txt", "r") as file:
            tasks = file.read()
            if not tasks.strip():
                return "No tasks found."
                
            tasks = tasks.split("\n\ntask:")
            task_found = False
            for task in tasks:
                if task_title not in task:
                    if task.strip():  # Only add non-empty tasks
                        new_tasks.append(task)
                else:
                    task_found = True
                    
        if not task_found:
            return f"Task '{task_title}' not found."
            
        with open("task.txt", "w") as file:
            file.write("\n\ntask:".join(new_tasks))
        return f"Task '{task_title}' deleted successfully."




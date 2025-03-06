from datetime import datetime, timedelta
import os
from contextlib import chdir


def add_task(task_title: str, task_content: str, due_date: str = None) -> str:
    """
    Add a task to the task list.

    Args:
        task_title (str): The title of the task.
        task_content (str): The content of the task.
        due_date (str, optional): The due date for the task in YYYY-MM-DD HH:MM:SS format.
                                 If not provided, no due date will be set.

    Returns:
        str: A string describing if the task was added successfully.
    """
    creation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Validate due_date format if provided
    if due_date:
        try:
            # Try to parse the due date to validate format
            datetime.strptime(due_date, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return "Error: Due date must be in the format YYYY-MM-DD HH:MM:SS"
    
    # Create Tasks directory if it doesn't exist
    os.makedirs("Tasks", exist_ok=True)
    
    with chdir("Tasks"):
        if not os.path.exists("task.txt"):
            with open("task.txt", "w") as _:
                pass
                
        with open("task.txt", "a") as file:
            due_date_line = f"Due: {due_date}" if due_date else "Due: None"
            file.write(f"\n\ntask: {task_title}\n{task_content}\n{due_date_line}\nCreated: {creation_date}")
            return f"Task '{task_title}' created successfully with {'due date ' + due_date if due_date else 'no due date'}."


def read_task() -> str:
    """
    Read all tasks from the task list.

    Returns:
        str: All tasks if any exist, or an error message if no tasks exist.
    """
    # Create Tasks directory if it doesn't exist
    os.makedirs("Tasks", exist_ok=True)
    
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
    # Create Tasks directory if it doesn't exist
    os.makedirs("Tasks", exist_ok=True)
    
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
                # Extract the task title from the first line
                task_lines = task.strip().split('\n')
                current_task_title = task_lines[0].strip() if task_lines else ""
                
                # Compare with exact task title (not substring)
                if task_title != current_task_title:
                    if task.strip():  # Only add non-empty tasks
                        new_tasks.append(task)
                else:
                    task_found = True
                    
        if not task_found:
            return f"Task '{task_title}' not found."
            
        with open("task.txt", "w") as file:
            file.write("\n\ntask:".join(new_tasks))
        return f"Task '{task_title}' deleted successfully."


def list_tasks() -> str:
    """
    List all tasks from the task list and delete tasks that are 3 days past their due date.

    Returns:
        str: All current tasks if any exist, or an error message if no tasks exist.
               Also includes information about deleted expired tasks.
    """
    # Create Tasks directory if it doesn't exist
    os.makedirs("Tasks", exist_ok=True)
    
    with chdir("Tasks"):
        if not os.path.exists("task.txt"):
            return "No tasks found."
            
        current_time = datetime.now()
        new_tasks = []
        tasks_with_dates = []
        deleted_count = 0
        
        with open("task.txt", "r") as file:
            tasks = file.read()
            if not tasks.strip():
                return "No tasks found."
                
            tasks = tasks.split("\n\ntask:")
            for task in tasks:
                if task.strip():  # Only process non-empty tasks
                    # Extract information from the task
                    task_lines = task.strip().split('\n')
                    task_title = task_lines[0] if len(task_lines) > 0 else "Unknown"
                    
                    # Extract due date if available
                    due_date_str = None
                    creation_date_str = None
                    task_content_lines = []
                    
                    for line in task_lines[1:]:
                        if line.startswith("Due: "):
                            due_date_str = line[5:] if line[5:] != "None" else None
                        elif line.startswith("Created: "):
                            creation_date_str = line[9:]
                        else:
                            task_content_lines.append(line)
                    
                    # Default task content if nothing extracted
                    task_content = "\n".join(task_content_lines) if task_content_lines else ""
                    
                    # Check if task is past due date by more than 3 days
                    should_delete = False
                    if due_date_str:
                        try:
                            due_date = datetime.strptime(due_date_str, "%Y-%m-%d %H:%M:%S")
                            # Check if task is more than 3 days past its due date
                            if current_time - due_date > timedelta(days=3):
                                should_delete = True
                                deleted_count += 1
                        except ValueError:
                            # If due date parsing fails, keep the task
                            pass
                    
                    if not should_delete:
                        # Keep task if it's not expired
                        new_tasks.append(task)
                        
                        # Format the task with dates for display
                        formatted_task = f"task: {task_title}\n{task_content}"
                        if due_date_str:
                            formatted_task += f"\nDue: {due_date_str}"
                        if creation_date_str:
                            formatted_task += f"\nCreated: {creation_date_str}"
                            
                        tasks_with_dates.append(formatted_task)
        
        # Write back only the non-expired tasks
        with open("task.txt", "w") as file:
            file.write("\n\ntask:".join(new_tasks))
            
        if not tasks_with_dates:
            return "No tasks found."
        
        result = "\n\n".join(tasks_with_dates).strip()
        
        # Add information about deleted tasks if any were removed
        if deleted_count > 0:
            result += f"\n\n{deleted_count} expired task(s) have been automatically deleted (3+ days past due date)."
            
        return result

# Note functionality (moved from Notes.py)

def create_note(note_title: str, note_content: str) -> str:
    """
    Create a new note with the given title and content.

    Args:
        note_title (str): The title of the note.
        note_content (str): The content of the note.

    Returns:
        str: A message indicating the success of the creation.
    """
    # Create Notes directory if it doesn't exist
    os.makedirs("Notes", exist_ok=True)
    
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
    # Create Notes directory if it doesn't exist
    os.makedirs("Notes", exist_ok=True)
    
    with chdir("Notes"):
        with open(f"{note_title}.txt", "w") as file:
            file.write(note_content)
        return f"Note '{note_title}' updated successfully."


def read_note(note_title: str) -> str:
    """
    Read the content of an existing note.
    
    Args:
        note_title (str): The title of the note to read.
        
    Returns:
        str: The content of the note.
    """
    # Create Notes directory if it doesn't exist
    os.makedirs("Notes", exist_ok=True)
    
    with chdir("Notes"):
        try:
            with open(f"{note_title}.txt", "r") as file:
                return file.read()
        except FileNotFoundError:
            return f"Note '{note_title}' not found."


def delete_note(note_title: str) -> str:
    """
    Delete a note file.

    Args:
        note_title (str): The title of the note to delete.

    Returns:
        str: A message indicating whether the note was deleted successfully.
    """
    try:
        os.makedirs("Notes", exist_ok=True)
        note_path = os.path.join("Notes", f"{note_title}.txt")
        
        if os.path.exists(note_path):
            os.remove(note_path)
            return f"Note '{note_title}' deleted successfully."
        else:
            return f"Note '{note_title}' not found."
    except Exception as e:
        return f"Error deleting note: {str(e)}"


def list_notes() -> str:
    """
    List all available notes.

    Returns:
        str: A formatted string containing all note titles and their creation dates.
    """
    try:
        os.makedirs("Notes", exist_ok=True)
        notes_dir = "Notes"
        
        notes = []
        if os.path.exists(notes_dir):
            for filename in os.listdir(notes_dir):
                if filename.endswith(".txt"):
                    note_title = filename[:-4]  # Remove .txt extension
                    note_path = os.path.join(notes_dir, filename)
                    
                    # Get the creation time of the note file
                    creation_time = datetime.fromtimestamp(os.path.getctime(note_path))
                    creation_date = creation_time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    notes.append(f"📝 {note_title} (Created: {creation_date})")
        
        if notes:
            return "Available Notes:\n" + "\n".join(notes)
        else:
            return "No notes found."
    except Exception as e:
        return f"Error listing notes: {str(e)}"


def check_expired_tasks() -> str:
    """
    Check for and delete tasks that are more than 3 days past their due date.
    This function runs automatically on startup before morning report generation.
    It helps keep the task list clean by removing outdated tasks.

    Returns:
        str: A message indicating how many tasks were deleted, if any.
    """
    # Create Tasks directory if it doesn't exist
    os.makedirs("Tasks", exist_ok=True)
    
    with chdir("Tasks"):
        if not os.path.exists("task.txt"):
            return "No tasks found."
            
        current_time = datetime.now()
        new_tasks = []
        deleted_count = 0
        
        with open("task.txt", "r") as file:
            tasks = file.read()
            if not tasks.strip():
                return "No tasks found."
                
            tasks = tasks.split("\n\ntask:")
            for task in tasks:
                if task.strip():  # Only process non-empty tasks
                    # Extract information from the task
                    task_lines = task.strip().split('\n')
                    task_title = task_lines[0] if len(task_lines) > 0 else "Unknown"
                    
                    # Extract due date if available
                    due_date_str = None
                    
                    for line in task_lines[1:]:
                        if line.startswith("Due: "):
                            due_date_str = line[5:] if line[5:] != "None" else None
                            break
                    
                    # Check if task is past due date by more than 3 days
                    should_delete = False
                    if due_date_str:
                        try:
                            due_date = datetime.strptime(due_date_str, "%Y-%m-%d %H:%M:%S")
                            # Check if task is more than 3 days past its due date
                            if current_time - due_date > timedelta(days=3):
                                should_delete = True
                                deleted_count += 1
                        except ValueError:
                            # If due date parsing fails, keep the task
                            pass
                    
                    if not should_delete:
                        # Keep task if it's not expired
                        new_tasks.append(task)
        
        # Write back only the non-expired tasks
        with open("task.txt", "w") as file:
            file.write("\n\ntask:".join(new_tasks))
            
        if deleted_count > 0:
            return f"{deleted_count} expired task(s) have been automatically deleted (3+ days past due date)."
        else:
            return "No expired tasks found."



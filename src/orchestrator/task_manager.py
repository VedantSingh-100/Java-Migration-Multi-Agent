"""
Task Management for Migration Orchestrator

This module handles:
- VISIBLE_TASKS.md file creation and updates
- Task extraction from TODO.md
- Task completion marking
- Progress tracking

The key design principle is that agents only see a LIMITED view of tasks
(next 3 unchecked tasks) via VISIBLE_TASKS.md to prevent cherry-picking
tasks from later phases.
"""

import os
from typing import Optional

from src.utils.logging_config import log_agent, log_summary


class TaskManager:
    """
    Manages task visibility and completion for migration agents.

    Key Features:
    1. Limited task visibility - agents only see next 3 tasks
    2. Auto-sync - marks tasks complete after successful commits
    3. Progress tracking - completed/total counts

    The VISIBLE_TASKS.md file is the agent's ONLY view into TODO.md.
    This prevents agents from skipping ahead or cherry-picking tasks.
    """

    def __init__(self, project_path: str = None):
        """
        Args:
            project_path: Path to project directory containing TODO.md
        """
        self.project_path = project_path

    def set_project_path(self, project_path: str):
        """Update the project path"""
        self.project_path = project_path

    def get_visible_tasks(self, todo_content: str, max_visible: int = 3) -> dict:
        """
        Extract only next N unchecked tasks from TODO content.

        This limits what the agent can see, preventing it from cherry-picking
        tasks from later phases. Agent only sees immediate next tasks.

        Args:
            todo_content: Full TODO.md file content
            max_visible: Maximum number of unchecked tasks to show (default 3)

        Returns:
            {
                'current': "First unchecked task description",
                'upcoming': ["Next task", "Task after that"],
                'completed_count': 5,
                'total_count': 54,
                'remaining_count': 49,
                'all_done': False
            }
        """
        if not todo_content:
            # IMPORTANT: Missing/empty TODO.md is NOT "all done" - it's an error state
            # all_done: False ensures system doesn't falsely claim completion
            # file_missing: True allows callers to handle this case appropriately
            log_agent("[TASK_MANAGER] WARNING: TODO.md is empty or missing - this is an error state, not completion")
            return {
                'current': None,
                'upcoming': [],
                'completed_count': 0,
                'total_count': 0,
                'remaining_count': 0,
                'all_done': False,  # Changed from True - missing file != completion
                'file_missing': True  # New flag to indicate error state
            }

        lines = todo_content.split('\n')
        unchecked_tasks = []
        checked_count = 0

        # Parse TODO.md to extract checked and unchecked tasks
        for line in lines:
            if '- [x]' in line.lower() or '- [X]' in line:
                checked_count += 1
            elif '- [ ]' in line:
                # Extract task description (remove checkbox marker)
                task_desc = line.replace('- [ ]', '').strip()
                if task_desc:  # Ignore empty lines
                    unchecked_tasks.append(task_desc)

        total_count = checked_count + len(unchecked_tasks)
        all_done = len(unchecked_tasks) == 0

        return {
            'current': unchecked_tasks[0] if unchecked_tasks else None,
            'upcoming': unchecked_tasks[1:max_visible] if len(unchecked_tasks) > 1 else [],
            'completed_count': checked_count,
            'total_count': total_count,
            'remaining_count': len(unchecked_tasks),
            'all_done': all_done
        }

    def extract_current_task(self, visible_content: str) -> Optional[str]:
        """
        Extract the current task description from VISIBLE_TASKS.md content.

        Args:
            visible_content: Content of VISIBLE_TASKS.md file

        Returns:
            Task description string, or None if not found
        """
        try:
            if 'CURRENT TASK' not in visible_content:
                return None

            # Extract between "CURRENT TASK" and "UPCOMING" or end
            current_section = visible_content.split('CURRENT TASK')[1]
            if 'UPCOMING' in current_section:
                current_section = current_section.split('UPCOMING')[0]

            # Find the task line (starts with - [ ] or - [x])
            for line in current_section.split('\n'):
                line = line.strip()
                if line.startswith('- [') and ']' in line:
                    # Extract just the task description (remove - [ ] or - [x])
                    task = line.split(']', 1)[1].strip()
                    return task

            return None
        except Exception as e:
            log_agent(f"[TASK_MANAGER] Error extracting current task: {str(e)}", "ERROR")
            return None

    def mark_task_complete(self, task_description: str) -> bool:
        """
        Deterministically mark a task as complete in TODO.md.

        Args:
            task_description: The exact task description to mark complete

        Returns:
            True if task was found and marked, False otherwise
        """
        if not self.project_path or not task_description:
            return False

        todo_path = os.path.join(self.project_path, "TODO.md")
        if not os.path.exists(todo_path):
            return False

        try:
            with open(todo_path, 'r') as f:
                content = f.read()

            # Find the exact task line and mark it
            original = f"- [ ] {task_description}"
            completed = f"- [x] {task_description}"

            if original in content:
                updated = content.replace(original, completed, 1)  # Only replace first occurrence

                with open(todo_path, 'w') as f:
                    f.write(updated)

                log_agent(f"[TASK_MANAGER] Marked task complete in TODO.md: {task_description[:60]}...")
                log_summary(f"TASK_MANAGER: Task marked complete in TODO.md")
                return True
            else:
                log_agent(f"[TASK_MANAGER] Task not found in TODO.md: {task_description[:60]}...", "WARNING")
                return False

        except Exception as e:
            log_agent(f"[TASK_MANAGER] Error marking task in TODO.md: {str(e)}", "ERROR")
            return False

    def create_visible_tasks_file(self, visible_tasks: dict) -> bool:
        """
        Create VISIBLE_TASKS.md file with only next 3 unchecked tasks.

        This prevents agent from bypassing read_file filter using run_command.
        Agent can 'cat VISIBLE_TASKS.md' all day - it will only see 3 tasks.

        Args:
            visible_tasks: Dictionary from get_visible_tasks()

        Returns:
            True if file created successfully, False otherwise
        """
        if not self.project_path:
            return False

        visible_tasks_path = os.path.join(self.project_path, "VISIBLE_TASKS.md")

        try:
            # Handle file_missing state - this is an ERROR, not completion
            if visible_tasks.get('file_missing'):
                content = """# Visible Tasks

⚠️ ERROR: TODO.md FILE MISSING OR EMPTY ⚠️

The TODO.md file could not be found or is empty.
This is an error state - migration cannot proceed.

POSSIBLE CAUSES:
- State files were lost during branch switch
- Git stash removed working directory files
- Files were not properly initialized

RECOVERY:
- Check if files are in git stash: `git stash list`
- If stashed, pop them: `git stash pop`
- Otherwise, analysis phase may need to be re-run"""
                log_agent("[TASK_MANAGER] ERROR: Creating error-state VISIBLE_TASKS.md due to missing TODO.md")
            elif visible_tasks['all_done']:
                content = """# Visible Tasks

ALL TASKS COMPLETE!

All migration tasks have been marked as complete in TODO.md.
The migration is finished."""
            elif visible_tasks['current']:
                content = f"""# Visible Tasks

This file shows only your next 3 tasks. Complete these before moving forward.

## CURRENT TASK (DO THIS NOW):

- [ ] {visible_tasks['current']}

Complete the CURRENT TASK above before attempting upcoming tasks.
"""
                if visible_tasks['upcoming']:
                    content += "\n## UPCOMING TASKS (for reference - complete current first):\n\n"
                    for task in visible_tasks['upcoming']:
                        content += f"- [ ] {task}\n"

                hidden_count = visible_tasks['remaining_count'] - len(visible_tasks['upcoming']) - 1
                if hidden_count > 0:
                    content += f"\n{hidden_count} additional tasks are hidden and will be shown after you complete current tasks.\n"

                content += f"\nProgress: {visible_tasks['completed_count']}/{visible_tasks['total_count']} complete\n"
            else:
                content = "# Visible Tasks\n\n(No tasks defined yet)"

            with open(visible_tasks_path, 'w') as f:
                f.write(content)

            log_agent(f"[TASK_MANAGER] Created {visible_tasks_path} with next {min(3, visible_tasks['remaining_count'])} tasks")
            return True
        except Exception as e:
            log_agent(f"[TASK_MANAGER] Error creating file: {str(e)}", "ERROR")
            return False

    def update_visible_tasks_file(self, state_file_manager, mark_current_complete: bool = False) -> bool:
        """
        Update VISIBLE_TASKS.md with fresh next 3 tasks after TODO.md changes.

        Args:
            state_file_manager: StateFileManager instance for reading files
            mark_current_complete: If True, mark the current VISIBLE task as complete in TODO.md first

        Returns:
            True if update successful, False otherwise
        """
        if not self.project_path:
            return False

        try:
            # If requested, mark current task complete in TODO.md
            if mark_current_complete:
                visible_tasks_path = os.path.join(self.project_path, "VISIBLE_TASKS.md")
                if os.path.exists(visible_tasks_path):
                    with open(visible_tasks_path, 'r') as f:
                        current_visible = f.read()

                    # Extract current task
                    current_task = self.extract_current_task(current_visible)
                    if current_task:
                        log_agent(f"[AUTO_SYNC] Marking task complete after commit: {current_task[:60]}...")
                        self.mark_task_complete(current_task)

            # Read TODO.md and regenerate VISIBLE_TASKS.md
            todo_content = state_file_manager.read_file("TODO.md", keep_beginning=True)
            if todo_content:
                visible_tasks = self.get_visible_tasks(todo_content, max_visible=3)
                return self.create_visible_tasks_file(visible_tasks)
            return False
        except Exception as e:
            log_agent(f"[TASK_MANAGER] Error updating file: {str(e)}", "ERROR")
            return False

    def get_progress(self, state_file_manager) -> dict:
        """
        Get current task progress.

        Args:
            state_file_manager: StateFileManager instance for reading files

        Returns:
            Dictionary with progress info
        """
        todo_content = state_file_manager.read_file("TODO.md", keep_beginning=True)
        return self.get_visible_tasks(todo_content)


def sync_tasks_after_commit(task_manager, state_file_manager) -> bool:
    """
    Sync tasks after a successful commit.

    This is called after git_commit or commit_changes tools execute.
    It marks the current task complete and updates VISIBLE_TASKS.md.

    Args:
        task_manager: TaskManager instance
        state_file_manager: StateFileManager instance

    Returns:
        True if sync successful
    """
    log_agent("[AUTO_SYNC] Triggered after commit")
    return task_manager.update_visible_tasks_file(
        state_file_manager,
        mark_current_complete=True
    )

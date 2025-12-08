from .file_operations import file_tools
from .command_executor import command_tools
from .maven_api import maven_tools
from .openrewrite_client import openrewrite_tools
from .web_search_tools import web_search_toolkit
from .git_operations import git_tools
from .state_management import (
    check_migration_state,
    can_call_analysis_expert,
    can_call_execution_expert,
    can_call_error_expert
)
from .guarded_handoff import (
    guarded_analysis_handoff,
    guarded_execution_handoff,
    guarded_error_handoff
)

# State management and guarded handoff tools
state_tools = [
    check_migration_state,
    can_call_analysis_expert,
    can_call_execution_expert,
    can_call_error_expert
]

guarded_handoff_tools = [
    guarded_analysis_handoff,
    guarded_execution_handoff,
    guarded_error_handoff
]

# Combine all tools
all_tools = file_tools + command_tools + maven_tools + openrewrite_tools + web_search_toolkit + git_tools + state_tools + guarded_handoff_tools

# Flatten all_tools into a single list of all tool objects (if not already flat)
all_tools_flat = []
for tool_group in all_tools:
    if isinstance(tool_group, list):
        all_tools_flat.extend(tool_group)
    else:
        all_tools_flat.append(tool_group)

__all__ = [
    "all_tools",
    "all_tools_flat",
    "file_tools",
    "command_tools",
    "maven_tools",
    "openrewrite_tools",
    "web_search_toolkit",
    "git_tools",
    "state_tools",
    "guarded_handoff_tools",
    "completion_tools",
    "check_migration_state",
    "can_call_analysis_expert",
    "can_call_execution_expert",
    "can_call_error_expert",
    "guarded_analysis_handoff",
    "guarded_execution_handoff",
    "guarded_error_handoff",
    "mark_execution_complete",
    "mark_analysis_complete"
]
"""
Migration Orchestrator Modules

This package contains the refactored orchestrator components:

- constants.py: All configuration constants
- state.py: State class, phase management, state file operations
- message_manager.py: Message pruning, external memory, prompt building
- task_manager.py: VISIBLE_TASKS, task extraction, TODO operations
- tool_registry.py: Tool sets for each agent, tool wrapping
- action_logger.py: Action logging, result formatting
- error_handler.py: Error detection, tracking, history
- agent_wrappers.py: Node wrappers for analysis, execution, error agents

The main supervisor_orchestrator.py imports from these modules.
"""

# Constants
from .constants import (
    MAX_CONTEXT_TOKENS,
    SUMMARISE_TO_TOKENS,
    MAX_SUMMARY_LENGTH,
    MAX_LLM_CALLS,
    MAX_EXECUTION_LOOPS_PER_PHASE,
    MAX_LOOPS_WITHOUT_PROGRESS,
    EXECUTION_WINDOW_SIZE,
    ERROR_WINDOW_SIZE,
    MAX_HISTORY_MESSAGES,
    ERROR_SIGNATURE_LENGTH,
    MAX_ERROR_ATTEMPTS,
    TRACKED_TOOLS,
    COMMIT_TOOLS,
    PHASES,
    PHASE_DESCRIPTIONS,
    STATE_FILES,
    PROTECTED_STATE_FILES,
    ERROR_AGENT_BLOCKED_FILES,
)

# State management
from .state import (
    State,
    StateFileManager,
    calculate_todo_progress,
    determine_phase_from_progress,
    is_valid_phase_transition,
)

# Message management
from .message_manager import (
    MessagePruner,
    ExternalMemoryBuilder,
    PromptBuilder,
    create_clean_execution_context,
    compile_execution_context,
    extract_last_tool_result,
    summarize_completed_tasks,
)

# Task management
from .task_manager import (
    TaskManager,
    sync_tasks_after_commit,
)

# Tool registry
from .tool_registry import (
    ANALYSIS_TOOL_NAMES,
    EXECUTION_TOOL_NAMES,
    ERROR_TOOL_NAMES,
    SUPERVISOR_TOOL_NAMES,
    ANALYSIS_ALLOWED_FILES,
    ToolWrapper,
    get_tools_for_agent,
)

# Action logging
from .action_logger import (
    ActionLogger,
    initialize_completed_actions_file,
)

# Error handling
from .error_handler import (
    # Unified error classification (Single Source of Truth)
    MavenErrorType,
    UnifiedErrorClassifier,
    unified_classifier,
    # Legacy error handler (for error history, stuck detection)
    ErrorHandler,
    StuckDetector,
    initialize_error_history_file,
    # Signature-based loop detection helpers
    ToolResultCategory,
    categorize_tool_result,
    hash_tool_args,
)

# Agent wrappers
from .agent_wrappers import (
    AnalysisNodeWrapper,
    ExecutionNodeWrapper,
    ErrorNodeWrapper,
)

__all__ = [
    # Constants
    'MAX_CONTEXT_TOKENS',
    'SUMMARISE_TO_TOKENS',
    'MAX_SUMMARY_LENGTH',
    'MAX_LLM_CALLS',
    'MAX_EXECUTION_LOOPS_PER_PHASE',
    'MAX_LOOPS_WITHOUT_PROGRESS',
    'EXECUTION_WINDOW_SIZE',
    'ERROR_WINDOW_SIZE',
    'MAX_HISTORY_MESSAGES',
    'ERROR_SIGNATURE_LENGTH',
    'MAX_ERROR_ATTEMPTS',
    'TRACKED_TOOLS',
    'COMMIT_TOOLS',
    'PHASES',
    'PHASE_DESCRIPTIONS',
    'STATE_FILES',
    'PROTECTED_STATE_FILES',
    'ERROR_AGENT_BLOCKED_FILES',

    # State
    'State',
    'StateFileManager',
    'calculate_todo_progress',
    'determine_phase_from_progress',
    'is_valid_phase_transition',

    # Messages
    'MessagePruner',
    'ExternalMemoryBuilder',
    'PromptBuilder',
    'create_clean_execution_context',
    'compile_execution_context',
    'extract_last_tool_result',
    'summarize_completed_tasks',

    # Tasks
    'TaskManager',
    'sync_tasks_after_commit',

    # Tools
    'ANALYSIS_TOOL_NAMES',
    'EXECUTION_TOOL_NAMES',
    'ERROR_TOOL_NAMES',
    'SUPERVISOR_TOOL_NAMES',
    'ANALYSIS_ALLOWED_FILES',
    'ToolWrapper',
    'get_tools_for_agent',

    # Actions
    'ActionLogger',
    'initialize_completed_actions_file',

    # Errors (Unified Classification)
    'MavenErrorType',
    'UnifiedErrorClassifier',
    'unified_classifier',
    'ErrorHandler',
    'StuckDetector',
    'initialize_error_history_file',
    # Signature-based loop detection
    'ToolResultCategory',
    'categorize_tool_result',
    'hash_tool_args',

    # Wrappers
    'AnalysisNodeWrapper',
    'ExecutionNodeWrapper',
    'ErrorNodeWrapper',
]

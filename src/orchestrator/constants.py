"""
Constants for the Migration Orchestrator

All configuration values in one place for easy tuning and visibility.
This helps prevent inconsistencies where the same value is defined differently
in multiple places.
"""

# =============================================================================
# CONTEXT MANAGEMENT
# =============================================================================

# Maximum tokens allowed in LLM context
MAX_CONTEXT_TOKENS = 140_000

# Target token count when summarizing/trimming context
SUMMARISE_TO_TOKENS = 30_000

# Maximum characters from state files (TODO.md, COMPLETED_ACTIONS.md, etc.)
# Increased from 2000 to handle larger TODO.md files (~7000 chars typical)
MAX_SUMMARY_LENGTH = 8000


# =============================================================================
# CIRCUIT BREAKER / LIMITS
# =============================================================================

# Maximum LLM calls allowed per migration (prevents runaway costs)
MAX_LLM_CALLS = 160

# Maximum execution loops per phase before hard stop
MAX_EXECUTION_LOOPS_PER_PHASE = 30

# Alert threshold for loops without progress
MAX_LOOPS_WITHOUT_PROGRESS = 5


# =============================================================================
# MESSAGE WINDOW SIZES
# =============================================================================

# Messages to keep for execution agent (affects context trimming)
EXECUTION_WINDOW_SIZE = 5

# Messages to keep for error expert (minimal context needed)
ERROR_WINDOW_SIZE = 3

# Maximum messages to keep in history (for prompt trimming)
MAX_HISTORY_MESSAGES = 30


# =============================================================================
# ERROR HANDLING
# =============================================================================

# Characters to identify duplicate errors (for deduplication)
ERROR_SIGNATURE_LENGTH = 100

# Maximum error resolution attempts per error
MAX_ERROR_ATTEMPTS = 3


# =============================================================================
# TOOL TRACKING
# =============================================================================

# Tools that are tracked for action logging and progress detection
# Maps tool_name -> human-readable description
TRACKED_TOOLS = {
    # Migration execution tools
    'add_openrewrite_plugin': 'OpenRewrite plugin',
    'configure_openrewrite_recipes': 'Configured OpenRewrite recipes',
    'update_java_version': 'Updated Java version',
    'create_branch': 'Created branch',
    'mvn_rewrite_run': 'Executed OpenRewrite migration',
    'mvn_rewrite_run_recipe': 'Executed OpenRewrite recipe',
    'git_commit': 'COMMIT',
    'commit_changes': 'COMMIT',  # Both commit tools trigger AUTO_SYNC

    # Verification tools (to see if agent is checking work)
    'mvn_compile': 'Compile check',
    'mvn_test': 'Test run',

    # File modification tools (to see what agent is changing)
    'write_file': 'File write',
    'find_replace': 'File update',

    # State management tools
    'git_add_all': 'Stage changes',
    'run_command': 'Shell command'
}

# Tools that trigger task completion (AUTO_SYNC)
COMMIT_TOOLS = {'git_commit', 'commit_changes'}


# =============================================================================
# PHASE DEFINITIONS
# =============================================================================

# Migration phases in order
PHASES = [
    "INIT",
    "ANALYSIS",
    "ANALYSIS_COMPLETE",
    "EXECUTION",
    "ERROR_RESOLUTION",
    "EXECUTION_COMPLETE",
    "SUCCESS",
    "FAILED"
]

# Phase descriptions for progress reporting
PHASE_DESCRIPTIONS = {
    "INIT": "Initializing migration",
    "ANALYSIS": "Analyzing project",
    "ANALYSIS_COMPLETE": "Analysis complete, ready for execution",
    "EXECUTION": "Executing migration tasks",
    "ERROR_RESOLUTION": "Resolving build/test errors",
    "EXECUTION_COMPLETE": "Execution complete, validating",
    "SUCCESS": "Migration successful",
    "FAILED": "Migration failed"
}


# =============================================================================
# FILE NAMES
# =============================================================================

# State files created during migration
STATE_FILES = {
    'TODO': 'TODO.md',
    'CURRENT_STATE': 'CURRENT_STATE.md',
    'COMPLETED_ACTIONS': 'COMPLETED_ACTIONS.md',
    'VISIBLE_TASKS': 'VISIBLE_TASKS.md',
    'ERROR_HISTORY': 'ERROR_HISTORY.md',
    'ANALYSIS': 'analysis.md'
}

# Files that execution agent cannot access directly
PROTECTED_STATE_FILES = {'TODO.md', 'CURRENT_STATE.md', 'COMPLETED_ACTIONS.md'}

# Files that error agent cannot read
ERROR_AGENT_BLOCKED_FILES = {'TODO.md', 'CURRENT_STATE.md', 'VISIBLE_TASKS.md', 'COMPLETED_ACTIONS.md'}

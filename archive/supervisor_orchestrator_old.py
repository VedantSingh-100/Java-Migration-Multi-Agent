"""
Supervisor-Based Migration Orchestrator
Uses LangGraph's create_supervisor for intelligent agent management
"""
import os
import sys
from datetime import datetime
from typing import Dict, Any, TypedDict, List
import tiktoken
import json
from src.utils.LLMLogger import LLMLogger
from src.utils.logging_config import log_agent, log_summary, log_console, log_llm
from langchain_anthropic import ChatAnthropic
from pydantic import PrivateAttr


class LLMCallLimitExceeded(Exception):
    """Exception raised when the LLM call limit is exceeded during migration"""
    pass


class CircuitBreakerChatAnthropic(ChatAnthropic):
    """Wrapper around ChatAnthropic that enforces LLM call limits by checking BEFORE each call"""

    # Declare private attributes using Pydantic's PrivateAttr
    _token_counter: Any = PrivateAttr()
    _max_calls: int = PrivateAttr()

    def __init__(self, token_counter, max_calls, **kwargs):
        # Initialize parent first (Pydantic model)
        super().__init__(**kwargs)
        # Then set private attributes (after Pydantic initialization)
        self._token_counter = token_counter
        self._max_calls = max_calls

    def _generate(self, *args, **kwargs):
        """Override _generate to check limit before making the call"""
        if self._token_counter.llm_calls >= self._max_calls:
            log_agent(f"Circuit breaker preventing LLM call: {self._token_counter.llm_calls}/{self._max_calls}", "ERROR")
            raise LLMCallLimitExceeded(
                f"LLM call limit of {self._max_calls} exceeded ({self._token_counter.llm_calls} calls already made)"
            )
        return super()._generate(*args, **kwargs)

    async def _agenerate(self, *args, **kwargs):
        """Override async _generate to check limit before making the call"""
        if self._token_counter.llm_calls >= self._max_calls:
            log_agent(f"Circuit breaker preventing LLM call (async): {self._token_counter.llm_calls}/{self._max_calls}", "ERROR")
            raise LLMCallLimitExceeded(
                f"LLM call limit of {self._max_calls} exceeded ({self._token_counter.llm_calls} calls already made)"
            )
        return await super()._agenerate(*args, **kwargs)


from dotenv import load_dotenv

from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from langchain_core.globals import set_verbose
from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import (
    trim_messages,
    count_tokens_approximately,
)
from langchain_core.prompts import ChatPromptTemplate

# DISABLED: SummarizationNode import - not using due to incompatibility with string prompt parameter
# from langmem.short_term import SummarizationNode

# sys.path manipulation removed - not needed when running from project root

from src.tools import all_tools_flat
from src.tools.command_executor import mvn_compile, mvn_test, run_command
from src.tools.state_management import check_migration_state, set_state_tracker
from src.tools.guarded_handoff import (
    guarded_analysis_handoff,
    guarded_execution_handoff,
    guarded_error_handoff,
)
from src.tools.completion_tools import mark_execution_complete, mark_analysis_complete
from prompts.prompt_loader import (
    get_supervisor_prompt,
    get_migration_request,
    get_analysis_expert_prompt,
    get_execution_expert_prompt,
    get_error_expert_prompt,
)
from src.utils.TokenCounter import TokenCounter
from src.utils.migration_state_tracker import MigrationStateTracker
from src.utils.completion_detector import detect_analysis_complete, detect_execution_complete
# REMOVED: get_todo_checked_count - now using _calculate_todo_progress() method

load_dotenv()
set_verbose(True)

# Anthropic API key from environment variable
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is required")

tc = TokenCounter()
ENC = tiktoken.get_encoding("cl100k_base")
MAX_CONTEXT_TOKENS = 140_000
SUMMARISE_TO_TOKENS = 30_000
MAX_LLM_CALLS = 80  # Circuit breaker: Maximum LLM calls allowed per migration

# Stuck detection thresholds (industry best practice)
MAX_LOOPS_WITHOUT_PROGRESS = 5  # Alert if no progress for 5 loops
MAX_EXECUTION_LOOPS_PER_PHASE = 30  # Hard limit per execution phase

# External memory thresholds (context management)
EXECUTION_WINDOW_SIZE = 5  # Keep last N messages for execution loops > 1
ERROR_WINDOW_SIZE = 3  # Keep last N messages for error expert
MAX_SUMMARY_LENGTH = 2000  # Max chars from summary files
ERROR_SIGNATURE_LENGTH = 100  # Chars to identify duplicate errors

# Tracked tools for action logging and deduplication
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

SYSTEM_PROMPT = get_supervisor_prompt()


class State(AgentState):
    # NOTE: we're adding this key to keep track of previous summary information
    # to make sure we're not summarizing on every LLM call
    context: dict[str, Any]
    
    # Explicit state tracking for deterministic routing
    current_phase: str = "INIT"
    analysis_done: bool = False
    execution_done: bool = False
    project_path: str = ""
    
    # Stuck detection tracking (industry best practice)
    last_todo_count: int = 0
    loops_without_progress: int = 0
    total_execution_loops: int = 0
    stuck_intervention_active: bool = False
    
    # Error detection and routing
    has_build_error: bool = False
    error_count: int = 0
    last_error_message: str = ""


model = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    api_key=ANTHROPIC_API_KEY,
    callbacks=[LLMLogger(), tc],
    max_tokens=8192,
    max_retries=5,
)

# DISABLED: Summarization model and prompt - not needed with state_modifier approach
# # Summarization model should only generate text summaries, NO tool calls
# summarization_model = model.bind(max_tokens=150000)
# summarization_model = summarization_model.with_retry(stop_after_attempt=5)
#
# initial_summarization_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", """You are an expert at summarizing technical conversations about Java migration projects.
#
# Your task is to create a concise summary of the conversation history while preserving:
# 1. Key migration decisions and actions taken
# 2. Important technical details about the Java project (like versions, frameworks, tools)
# 3. Full errors details
# 4. Current state of the migration process
# 5. Next steps or pending tasks
#
# Focus on actionable information and maintain context for ongoing migration work.
# Be concise but comprehensive - aim for clarity over brevity when technical details are important.
#
# Conversation to summarize:"""),
#         ("placeholder", "{messages}"),
#         ("user", "Create a summary of the conversation above:"),
#     ]
# )

# DISABLED: SummarizationNode does not work with string prompt parameter in create_react_agent
# See: SUMMARIZATIONNODE_BROKEN_ANALYSIS.md for details
# summarization_node = SummarizationNode(
#     token_counter=count_tokens_approximately,
#     model=summarization_model,
#     max_tokens=MAX_CONTEXT_TOKENS,
#     max_summary_tokens=SUMMARISE_TO_TOKENS,
#     output_messages_key="llm_input_messages",
#     initial_summary_prompt = initial_summarization_prompt
# )


class SupervisorMigrationOrchestrator:
    """Supervisor that manages specialized migration agents"""
    
    def __init__(self):
        log_agent("Initializing SupervisorMigrationOrchestrator")
        log_summary("COMPONENT: SupervisorMigrationOrchestrator initialized")
        log_agent(f"Circuit breaker initialized with limit: {MAX_LLM_CALLS}")
        
        # Initialize state tracker (will be set per migration)
        self.state_tracker = None
        self.project_path = None  # Set per migration
        log_agent("State management system initialized")
        
        # Loop detection: Track recent actions to detect stuck patterns
        self.recent_actions = []  # List of dicts: {tool_name, todo_item, completed, timestamp}
        self.action_window_size = 5  # Check last 5 actions for patterns
        log_agent("Loop detection system initialized")
        
        # Verification tracking: Track recent tracked tool calls for TODO.md marking verification
        self.recent_tracked_tools = []  # List of dicts: {tool_name, timestamp}
        self.verification_window_seconds = 60  # Allow TODO marking within 60 seconds of tracked tool call
        log_agent("Verification system initialized (60 second window for TODO marking)")
        
        # Phase transition flag: Lock TODO.md after analysis phase completes
        self._analysis_complete_flag = False
        log_agent("Phase transition tracking initialized")
        
        # Action logging: Sequential counter for all tool executions
        self.action_counter = 0  # Increments with each logged action
        log_agent("Action logging system initialized with sequential numbering")
        
        # Initialize context manager for smart message trimming
        from src.utils.context_manager import ContextManager
        self.context_manager = ContextManager(max_recent_messages=10, max_tool_output_lines=20)
        log_agent("Context manager initialized for state_modifier")
        
        # Create specialized migration agents as workers
        self.migration_workers = self._create_migration_workers()
        
        # Create supervisor workflow
        self.supervisor_workflow = self._create_supervisor()
        
        # Compile the workflow
        self.app = self.supervisor_workflow.compile()
        
        # print("Supervisor orchestrator initialized with workers:", [agent.name for agent in self.migration_workers])
        log_agent(f"Created {len(self.migration_workers)} migration workers")
        log_agent(f"Workers: {[agent.name for agent in self.migration_workers]}")
    
    def _check_llm_call_limit(self):
        """Check if LLM call limit has been exceeded. Raises LLMCallLimitExceeded if limit reached."""
        if tc.llm_calls >= MAX_LLM_CALLS:
            log_agent(f"LLM call limit reached: {tc.llm_calls}/{MAX_LLM_CALLS}", "ERROR")
            log_summary(f"CIRCUIT BREAKER: LLM call limit of {MAX_LLM_CALLS} exceeded ({tc.llm_calls} calls made)")
            log_console(f"⚠ LLM call limit reached ({tc.llm_calls}/{MAX_LLM_CALLS}) - stopping migration", "WARNING")
            raise LLMCallLimitExceeded(f"Migration stopped: LLM call limit of {MAX_LLM_CALLS} exceeded ({tc.llm_calls} calls made)")
    
    def _create_prompt_with_trimming(self, system_prompt: str, agent_name: str, max_messages: int = 30, inject_external_memory: bool = False):
        """Create callable prompt function that trims messages before sending to model
        
        This replaces the broken pre_model_hook/SummarizationNode approach.
        Uses callable prompt parameter which works with all LangGraph versions.
        
        Args:
            system_prompt: System prompt string for the agent
            agent_name: Name of agent (for logging)
            max_messages: Maximum messages to keep (token-aware trimming)
            inject_external_memory: If True, reads and injects COMPLETED_ACTIONS, TODO, CURRENT_STATE at fixed position
        
        Returns:
            Function that takes state dict and returns list of messages with system prompt
        """
        from langchain_core.messages import SystemMessage
        
        def prompt_builder(state: dict) -> list:
            """Build prompt with trimmed messages"""
            messages = state.get("messages", [])
            
            if not messages:
                log_agent(f"[PROMPT_TRIM] {agent_name}: No messages, returning system prompt only", "WARNING")
                result = [SystemMessage(content=system_prompt)]
                
                # Inject external memory even if no messages
                if inject_external_memory and self.project_path:
                    memory_block = self._build_external_memory_block()
                    if memory_block:
                        result.append(SystemMessage(content=memory_block))
                        log_agent(f"[EXTERNAL_MEMORY] Injected into empty prompt for {agent_name}")
                
                return result
            
            log_agent(f"[PROMPT_TRIM] {agent_name}: Input has {len(messages)} messages")
            
            # Anthropic-style tool result clearing: Keep tool_use/tool_result pairs but truncate old results
            try:
                from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
                
                if len(messages) <= max_messages:
                    log_agent(f"[PROMPT_TRIM] {agent_name}: Keeping all {len(messages)} messages")
                    
                    # Build final prompt with external memory even if no trimming needed
                    result = [SystemMessage(content=system_prompt)]
                    
                    if inject_external_memory and self.project_path:
                        memory_block = self._build_external_memory_block()
                        if memory_block:
                            result.append(SystemMessage(content=memory_block))
                            log_agent(f"[EXTERNAL_MEMORY] Injected at position [1] for {agent_name} (no trimming needed)")
                    
                    result.extend(messages)
                    log_agent(f"[PROMPT_STRUCTURE] Final prompt: {len(result)} messages (system + memory + {len(messages)} history)")
                    return result
                
                log_agent(f"[PROMPT_TRIM] {agent_name}: Applying tool result clearing (Anthropic method)")
                
                # Log BEFORE state
                log_agent(f"[PRUNE_DETAIL] BEFORE: {len(messages)} total messages")
                removed_msgs = messages[:-max_messages] if len(messages) > max_messages else []
                if removed_msgs:
                    log_agent(f"[PRUNE_DETAIL] Will remove {len(removed_msgs)} old messages:")
                    for idx, msg in enumerate(removed_msgs[:3]):  # Show first 3
                        msg_type = getattr(msg, 'type', type(msg).__name__)
                        tool_name = getattr(msg, 'name', 'unknown') if isinstance(msg, ToolMessage) else 'N/A'
                        content_preview = str(msg.content)[:80] if hasattr(msg, 'content') else 'no content'
                        log_agent(f"[PRUNE_DETAIL]   Msg {idx+1}: {msg_type} | tool={tool_name} | preview={content_preview}...")
                    if len(removed_msgs) > 3:
                        log_agent(f"[PRUNE_DETAIL]   ... and {len(removed_msgs) - 3} more messages")
                
                # Step 1: Take last N messages
                trimmed = messages[-max_messages:]
                
                # Step 2: Remove any orphaned ToolMessages at the start
                orphaned_count = 0
                while trimmed and isinstance(trimmed[0], ToolMessage):
                    orphaned_msg = trimmed[0]
                    tool_name = getattr(orphaned_msg, 'name', 'unknown')
                    log_agent(f"[PRUNE_DETAIL] Removing orphaned ToolMessage: {tool_name}")
                    trimmed = trimmed[1:]
                    orphaned_count += 1
                
                if orphaned_count > 0:
                    log_agent(f"[PROMPT_TRIM] {agent_name}: Removed {orphaned_count} orphaned ToolMessages")
                
                # Step 3: Clear/truncate old tool results (Anthropic's "tool result clearing")
                # Keep the AIMessage with tool_calls, but truncate the ToolMessage content
                # Keep recent messages (last 5) fully intact
                keep_full_count = 5
                cleared_count = 0
                cleared_details = []
                
                for i, msg in enumerate(trimmed[:-keep_full_count]):
                    if isinstance(msg, ToolMessage):
                        original_length = len(msg.content) if msg.content else 0
                        if original_length > 200:  # Only truncate if substantial
                            tool_name = getattr(msg, 'name', 'unknown')
                            cleared_details.append(f"{tool_name} ({original_length}→200 chars)")
                            # Create truncated version
                            truncated_content = msg.content[:200] + f"\n... [CLEARED: {original_length - 200} chars removed to save context]"
                            # Replace in-place
                            trimmed[i] = ToolMessage(
                                content=truncated_content,
                                tool_call_id=msg.tool_call_id,
                                name=msg.name if hasattr(msg, 'name') else None
                            )
                            cleared_count += 1
                
                if cleared_count > 0:
                    log_agent(f"[PROMPT_TRIM] {agent_name}: Cleared {cleared_count} old tool results")
                    log_agent(f"[PRUNE_DETAIL] Cleared tools: {', '.join(cleared_details[:5])}")
                    if len(cleared_details) > 5:
                        log_agent(f"[PRUNE_DETAIL]   ... and {len(cleared_details) - 5} more")
                
                # Safety check
                if not trimmed:
                    log_agent(f"[PROMPT_TRIM] {agent_name}: WARNING - Trimming removed everything, keeping original", "WARNING")
                    trimmed = messages[-max_messages:]
                
                # Log AFTER state
                log_agent(f"[PROMPT_TRIM] {agent_name}: Kept {len(trimmed)}/{len(messages)} messages with {cleared_count} results cleared")
                log_agent(f"[PROMPT_TRIM] {agent_name}: First: {getattr(trimmed[0], 'type', type(trimmed[0]).__name__)} | Last: {getattr(trimmed[-1], 'type', type(trimmed[-1]).__name__)}")
                log_agent(f"[PRUNE_DETAIL] AFTER: {len(trimmed)} messages, last {keep_full_count} kept fully intact")
                
                # Build final prompt: [system_prompt, external_memory (if enabled), ...trimmed_messages]
                result = [SystemMessage(content=system_prompt)]
                
                # Inject external memory at position [1], immune to trimming
                if inject_external_memory and self.project_path:
                    memory_block = self._build_external_memory_block()
                    if memory_block:
                        result.append(SystemMessage(content=memory_block))
                        log_agent(f"[EXTERNAL_MEMORY] Injected at position [1] for {agent_name} (immune to trimming)")
                    else:
                        log_agent(f"[EXTERNAL_MEMORY] No external memory files found for {agent_name}")
                
                # Add trimmed conversation history
                result.extend(trimmed)
                
                log_agent(f"[PROMPT_STRUCTURE] Final prompt: {len(result)} messages (system + memory + {len(trimmed)} history)")
                return result
            
            except Exception as e:
                log_agent(f"[PROMPT_TRIM] {agent_name}: Error: {e}", "ERROR")
                # Ultimate fallback: return system prompt + all messages
                log_agent(f"[PROMPT_TRIM] {agent_name}: Returning all {len(messages)} messages as fallback")
                result = [SystemMessage(content=system_prompt)]
                
                # Try to inject external memory even in error case
                if inject_external_memory and self.project_path:
                    try:
                        memory_block = self._build_external_memory_block()
                        if memory_block:
                            result.append(SystemMessage(content=memory_block))
                    except Exception as mem_err:
                        log_agent(f"[EXTERNAL_MEMORY] Failed to inject: {mem_err}", "ERROR")
                
                result.extend(messages)
                return result
        
        return prompt_builder
    
    def _create_migration_workers(self):
        """Create specialized worker agents for migration tasks"""
        
        common_agent_kwargs = dict(
            model=CircuitBreakerChatAnthropic(
                token_counter=tc,
                max_calls=MAX_LLM_CALLS,
                model="claude-sonnet-4-20250514",
                api_key=ANTHROPIC_API_KEY,
                callbacks=[LLMLogger(), tc],
                max_tokens=8192,
                max_retries=5,
            ),
            debug=False,
            checkpointer=InMemorySaver()
        )
        
        log_agent("[AGENTS] Using callable prompt instead of pre_model_hook (SummarizationNode broken)")
        
        # Analysis Worker - analyzes projects and recommends changes
        # NO TRIMMING for analysis - let it accumulate, phase transition will handle it
        analysis_worker = create_react_agent(
            tools=self._get_analysis_tools(),
            prompt=get_analysis_expert_prompt(),  # Direct prompt, no trimming
            name="analysis_expert",
            state_schema=State,
            **common_agent_kwargs,
        )
        log_agent("[AGENTS] ✓ analysis_expert created WITHOUT trimming (phase transition handles it)")
        
        # Execution Worker - executes plan
        execution_worker = create_react_agent(
            tools=self._get_execution_tools(),
            prompt=self._create_prompt_with_trimming(
                get_execution_expert_prompt(),
                "execution_expert",
                max_messages=30,
                inject_external_memory=True  # Always inject COMPLETED_ACTIONS, TODO, CURRENT_STATE
            ),
            name="execution_expert",
            state_schema=State,
            **common_agent_kwargs
        )
        log_agent("[AGENTS] ✓ execution_expert created with callable prompt + EXTERNAL MEMORY injection")
        
        # Error Fixing Worker - fixes compilation and build and test errors
        error_worker = create_react_agent(
            tools=self._get_error_tools(),
            prompt=self._create_prompt_with_trimming(get_error_expert_prompt(), "error_expert", max_messages=20),
            name="error_expert",
            state_schema=State,
            **common_agent_kwargs,
        )
        log_agent("[AGENTS] ✓ error_expert created with callable prompt")
        
        log_agent("[AGENTS] All agents created successfully with callable prompt approach")
        
        return [analysis_worker, execution_worker, error_worker]
    
    def _get_supervisor_tools(self):
        """Get tools for supervisor agent"""
        # Supervisor only has read-only tools - specialized agents manage state files
        supervisor_tools = [tool for tool in all_tools_flat if tool.name in [
            'read_file',
            'file_exists',
            'list_java_files',
            'read_pom',
        ]]
        
        # Add state management tools for checking migration progress
        supervisor_tools.append(check_migration_state)
        
        # Add guarded handoff tools that block duplicate agent calls
        supervisor_tools.extend([
            guarded_analysis_handoff,
            guarded_execution_handoff,
            guarded_error_handoff
        ])
        
        log_agent(f"Supervisor has {len(supervisor_tools)} tools available")
        return supervisor_tools
    
    def _wrap_analysis_tool(self, tool):
        """Wrap analysis agent tools with restrictions
        
        Analysis agent should ONLY write to state files:
        - TODO.md (migration plan)
        - CURRENT_STATE.md (project status)
        - analysis.md (analysis notes)
        
        Should NOT modify project files (pom.xml, .java files, etc.)
        """
        tool_name = tool.name
        
        # Only wrap write_file - other tools are safe
        if tool_name != 'write_file':
            return tool
        
        original_func = tool.func
        
        def restricted_write_file(*args, **kwargs):
            file_path = kwargs.get('file_path', args[0] if args else '')
            
            # List of allowed files for analysis agent
            allowed_files = ['TODO.md', 'CURRENT_STATE.md', 'analysis.md']
            
            # Check if writing to allowed file
            is_allowed = any(file_path.endswith(allowed_file) for allowed_file in allowed_files)
            
            if not is_allowed:
                log_agent(f"[ANALYSIS_BLOCK] 🚫 Analysis agent tried to write to '{file_path}' - BLOCKED")
                log_agent(f"[ANALYSIS_BLOCK] Analysis agent can only write to: {', '.join(allowed_files)}")
                return f"""ERROR: Analysis agent cannot modify project files.

🚫 FORBIDDEN: File modification blocked 🚫

You attempted to write to: {file_path}

Analysis agent can ONLY create these state files:
- TODO.md (migration plan with sequential tasks)
- CURRENT_STATE.md (project status and dependencies)
- analysis.md (analysis notes and recommendations)

You CANNOT modify project files such as:
- pom.xml (execution agent will handle this)
- .java files (execution agent will handle this)
- Any other project files

Your role is to ANALYZE and PLAN, not to execute changes.
The execution agent will handle all project modifications."""
            
            # Log allowed write
            log_agent(f"[ANALYSIS_WRITE] ✓ Analysis agent writing to allowed file: {file_path}")
            return original_func(*args, **kwargs)
        
        from langchain_core.tools import StructuredTool
        return StructuredTool(
            name=tool.name,
            description=tool.description,
            func=restricted_write_file,
            args_schema=tool.args_schema
        )
    
    def _get_analysis_tools(self):
        """Get tools for analysis agent (read + write state files only)"""
        analysis_tool_names = {
            # Read tools for inspection
            'read_file',
            'read_pom',
            'find_all_poms',
            'list_java_files',
            'search_files',
            'file_exists',
            'get_java_version',
            'list_dependencies',
            'get_available_recipes',
            'suggest_recipes_for_java_version',
            'web_search_tool',
            'call_openrewrite_agent',
            'get_status',
            'get_log',
            'list_branches',
            
            # State management
            'write_file',
            'check_migration_state',
            
            # Completion marking
            'mark_analysis_complete',
        }
        
        analysis_tools = [
            tool for tool in all_tools_flat
            if tool.name in analysis_tool_names
        ]
        
        # Wrap analysis tools with restrictions
        wrapped_tools = [self._wrap_analysis_tool(tool) for tool in analysis_tools]
        
        log_agent(f"[TOOLS] Analysis agent has {len(wrapped_tools)} tools (read + write state files)")
        log_agent(f"[TOOLS] Analysis tools: {sorted([t.name for t in wrapped_tools])}")
        log_agent(f"[TOOLS] write_file restricted to: TODO.md, CURRENT_STATE.md, analysis.md only")
        return wrapped_tools
    
    def _calculate_todo_progress(self) -> dict:
        """Calculate TODO progress by reading TODO.md file
        
        Uses same logic as _get_visible_tasks() to ensure consistency
        """
        if not hasattr(self, 'project_path') or not self.project_path:
            return {'completed': 0, 'total': 0, 'percent': 0, 'next_unchecked': None}
        
        todo_path = os.path.join(self.project_path, "TODO.md")
        if not os.path.exists(todo_path):
            return {'completed': 0, 'total': 0, 'percent': 0, 'next_unchecked': None}
        
        try:
            with open(todo_path, 'r') as f:
                content = f.read()
            
            # Use SAME parsing logic as _get_visible_tasks to avoid inconsistencies
            lines = content.split('\n')
            completed_tasks = 0
            unchecked_tasks = []
            
            for line in lines:
                # Match both [x] and [X] (case insensitive)
                if '- [x]' in line.lower() or '- [X]' in line:
                    completed_tasks += 1
                elif '- [ ]' in line:
                    task_desc = line.replace('- [ ]', '').strip()
                    if task_desc:  # Ignore empty lines
                        unchecked_tasks.append(task_desc)
            
            total_tasks = completed_tasks + len(unchecked_tasks)
            percent = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
            
            # Get first unchecked task (matches _get_visible_tasks behavior)
            next_unchecked = unchecked_tasks[0] if unchecked_tasks else None
            
            return {
                'completed': completed_tasks,
                'total': total_tasks,
                'percent': percent,
                'next_unchecked': next_unchecked
            }
        except Exception as e:
            log_agent(f"[STATE] Error calculating TODO progress: {e}")
            return {'completed': 0, 'total': 0, 'percent': 0, 'next_unchecked': None}
    
    def _determine_current_phase(self, todo_stats: dict) -> str:
        """Determine current migration phase based on TODO progress"""
        percent = todo_stats.get('percent', 0)
        
        if percent == 0:
            return "Initial Phase"
        elif percent < 20:
            return "Setup Phase"
        elif percent < 40:
            return "Java Migration Phase"
        elif percent < 60:
            return "Spring Boot Migration Phase"
        elif percent < 80:
            return "Jakarta EE Migration Phase"
        elif percent < 100:
            return "Final Verification Phase"
        else:
            return "Migration Complete"
    
    def _get_next_action_number(self) -> int:
        """Get next sequential action number and increment counter"""
        self.action_counter += 1
        return self.action_counter
    
    def _format_args(self, kwargs: dict) -> str:
        """Format tool arguments for logging (sanitized)"""
        if not kwargs:
            return "None"
        
        # Keep only important args, truncate long values
        important_keys = {'branch_name', 'version', 'recipe', 'recipe_name', 'message',
                         'file_path', 'group', 'artifact'}
        
        formatted = []
        for key in important_keys:
            if key in kwargs:
                value = str(kwargs[key])
                if len(value) > 80:
                    value = value[:77] + "..."
                formatted.append(f"{key}={value}")
        
        return ", ".join(formatted) if formatted else "args omitted"
    
    def _format_result(self, result: Any, tool_name: str) -> str:
        """Format tool result for logging based on tool type"""
        result_str = str(result)
        
        if tool_name == 'create_branch':
            if 'Created branch' in result_str:
                return "Created and checked out new branch"
            elif 'Checked out' in result_str:
                return "Checked out existing branch"
            return "Branch operation completed"
        
        elif tool_name == 'git_commit':
            if '] ' in result_str:
                # Extract commit hash and message
                parts = result_str.split('] ', 1)
                if len(parts) > 1:
                    return parts[1].split('\n')[0][:100]
            return "Commit successful"
        
        elif tool_name in ['mvn_rewrite_run_recipe', 'mvn_rewrite_run']:
            # Try to count changes
            change_count = result_str.count('Changes have been made')
            if change_count > 0:
                return f"Recipe executed, {change_count} files changed"
            return "Recipe executed"
        
        elif tool_name == 'add_openrewrite_plugin':
            return "OpenRewrite plugin added to pom.xml"
        
        elif tool_name == 'configure_openrewrite_recipes':
            # Try to extract recipe count
            if 'with' in result_str and 'recipe' in result_str:
                import re
                match = re.search(r'with (\d+)', result_str)
                if match:
                    return f"Configured {match.group(1)} recipes"
            return "Recipes configured"
        
        elif tool_name in ['mvn_compile', 'mvn_test']:
            if 'BUILD SUCCESS' in result_str:
                return "Build successful"
            elif 'Return code: 0' in result_str:
                return "Completed successfully"
            return "Maven command executed"
        
        elif tool_name == 'write_file':
            # Extract filename
            if 'Successfully wrote to' in result_str:
                path = result_str.replace('Successfully wrote to', '').strip()
                filename = path.split('/')[-1] if '/' in path else path
                return f"Wrote {filename}"
            return "File written"
        
        elif tool_name == 'find_replace':
            import re
            match = re.search(r'(\d+) replacement', result_str)
            if match:
                count = match.group(1)
                return f"{count} replacement(s) made"
            return "Text replaced"
        
        elif tool_name == 'git_add_all':
            return "Staged all changes"
        
        elif tool_name == 'run_command':
            if len(result_str) > 100:
                return result_str[:97] + "..."
            return "Command executed"
        
        # Default: truncate to 100 chars
        if len(result_str) > 100:
            return result_str[:97] + "..."
        return result_str
    
    def _extract_actionable_error(self, tool_name: str, result: Any, error: Any) -> str:
        """Extract actionable error messages instead of generic failures"""
        result_str = str(result) if result else ""
        error_str = str(error) if error else ""
        combined = result_str + " " + error_str
        
        # git_commit specific errors
        if tool_name == 'git_commit':
            if 'nothing to commit' in combined.lower():
                return "Nothing to commit (working directory clean). No changes were staged."
            elif 'not a git repository' in combined.lower():
                return "Not a git repository. Run git init first."
            elif 'no changes added' in combined.lower():
                return "No changes added to commit. Use git add to stage files first."
            elif 'please tell me who you are' in combined.lower():
                return "Git user.name/user.email not configured."
        
        # Maven compilation errors
        elif tool_name == 'mvn_compile':
            if 'compilation error' in combined.lower() or 'cannot find symbol' in combined.lower():
                # Try to extract the actual error
                lines = combined.split('\n')
                for i, line in enumerate(lines):
                    if 'error:' in line.lower():
                        # Get 2 lines of context
                        context = '\n'.join(lines[max(0,i):min(len(lines),i+2)])
                        return f"Compilation failed: {context[:200]}"
                return "Compilation error - check imports and dependencies"
            elif 'build failure' in combined.lower():
                return "Maven build failed - check pom.xml syntax"
        
        # Maven test errors
        elif tool_name == 'mvn_test':
            if 'test failures' in combined.lower() or 'tests run:' in combined.lower():
                # Extract failure count
                import re
                match = re.search(r'Failures: (\d+)', combined)
                if match:
                    return f"{match.group(1)} test(s) failed. Check test output for details."
                return "Tests failed - check test output"
        
        # OpenRewrite errors
        elif tool_name in ['mvn_rewrite_run', 'mvn_rewrite_run_recipe']:
            if 'recipe not found' in combined.lower():
                return "OpenRewrite recipe not found. Check recipe name in pom.xml."
            elif 'no recipe specified' in combined.lower():
                return "No recipe specified. Use configure_openrewrite_recipes first."
            elif 'build failure' in combined.lower():
                return "OpenRewrite execution failed. Check pom.xml OpenRewrite configuration."
        
        # File operation errors
        elif tool_name == 'write_file':
            if 'permission denied' in combined.lower():
                return "Permission denied writing file. Check file permissions."
            elif 'no such file or directory' in combined.lower():
                return "Directory does not exist. Create parent directories first."
        
        elif tool_name == 'find_replace':
            if 'not found' in combined.lower():
                return "Text to replace not found in file. Check exact string match."
        
        elif tool_name == 'git_add_all':
            if 'not a git repository' in combined.lower():
                return "Not a git repository. Initialize git first."
        
        elif tool_name == 'run_command':
            if 'command not found' in combined.lower():
                return "Command not found. Check if tool is installed."
            elif 'return code:' in combined.lower():
                import re
                match = re.search(r'return code: (\d+)', combined.lower())
                if match:
                    return f"Command failed with exit code {match.group(1)}"
        
        # Generic fallback - but try to extract useful info
        if error_str:
            return error_str[:300]
        elif 'unsuccessful' in combined.lower():
            return f"{tool_name} returned unsuccessful result. Check tool output for details."
        else:
            return combined[:300] if combined else "Tool execution failed"
    
    def _update_state_header(self):
        """Update COMPLETED_ACTIONS header with current migration state"""
        if not hasattr(self, 'project_path') or not self.project_path:
            return
        
        completed_actions_path = os.path.join(self.project_path, "COMPLETED_ACTIONS.md")
        
        # Read existing content
        existing_actions = ""
        existing_task_completions = ""
        if os.path.exists(completed_actions_path):
            with open(completed_actions_path, 'r') as f:
                content = f.read()

            # Extract TASK COMPLETIONS section (preserve it!)
            if '=== TASK COMPLETIONS ===' in content and '=== ACTION LOG ===' in content:
                task_start = content.find('=== TASK COMPLETIONS ===') + len('=== TASK COMPLETIONS ===')
                task_end = content.find('=== ACTION LOG ===')
                existing_task_completions = content[task_start:task_end].strip()

            # Extract action log (everything after ACTION LOG header)
            if '=== ACTION LOG ===' in content:
                existing_actions = content.split('=== ACTION LOG ===', 1)[1].strip()
            else:
                # Old format - keep everything
                existing_actions = content.strip()
        
        # Count actions from existing log
        success_count = existing_actions.count('| SUCCESS |')
        failed_count = existing_actions.count('| FAILED |')
        total_actions = success_count + failed_count
        
        # Extract last action timestamp
        last_action = "N/A"
        lines = existing_actions.strip().split('\n')
        for line in reversed(lines):
            if line.startswith('[') and ']' in line:
                # Extract timestamp from [001] 16:01:06 format
                parts = line.split('|')
                if len(parts) > 0:
                    timestamp_part = parts[0].split(']', 1)
                    if len(timestamp_part) > 1:
                        last_action = timestamp_part[1].strip().split('|')[0].strip()
                        break
        
        # Get TODO progress
        todo_stats = self._calculate_todo_progress()
        phase = self._determine_current_phase(todo_stats)
        
        # Log task visibility after this update
        if todo_stats.get('next_unchecked'):
            log_agent(f"[STATE_UPDATE] Next visible task will be: {todo_stats['next_unchecked'][:80]}...")
        
        # Get current task for display
        current_task = todo_stats.get('next_unchecked', 'None - check VISIBLE_TASKS.md')
        if current_task and len(current_task) > 60:
            current_task = current_task[:57] + "..."

        # Build header
        header = f"""=== MIGRATION STATE ===
STATUS: IN_PROGRESS
PROGRESS: {todo_stats['completed']}/{todo_stats['total']} ({todo_stats['percent']:.0f}%)
PHASE: {phase}
LAST_TOOL: {last_action}
ACTIONS_LOGGED: {total_actions} ({success_count} success, {failed_count} failed)

=== CURRENT TASK ===
{current_task}

=== IMPORTANT REMINDERS ===
⚠ DO NOT repeat tasks shown in TASK COMPLETIONS below
⚠ System tracks progress automatically after each commit
⚠ Focus ONLY on the current task in VISIBLE_TASKS.md

=== TASK COMPLETIONS ===

=== ACTION LOG ===
"""
        
        # Write updated file (preserving task completions and action log)
        try:
            with open(completed_actions_path, 'w') as f:
                # Build content: header -> task completions -> action log
                full_content = header

                # Insert existing task completions before ACTION LOG
                if existing_task_completions:
                    # Header already has "=== TASK COMPLETIONS ===" and "=== ACTION LOG ==="
                    # We need to insert completions BETWEEN them
                    parts = full_content.split('=== ACTION LOG ===')
                    full_content = parts[0] + existing_task_completions + "\n\n=== ACTION LOG ===" + (parts[1] if len(parts) > 1 else "")

                # Append existing action log
                if existing_actions:
                    full_content = full_content.rstrip() + "\n\n" + existing_actions

                f.write(full_content)
        except Exception as e:
            log_agent(f"[STATE] Error updating header: {e}")
    
    def _has_recent_tracked_tool_call(self) -> tuple[bool, str]:
        """Check if a tracked tool was successfully called recently (within verification window)
        
        Returns:
            (bool, str): (has_recent_call, tool_name_or_reason)
        """
        if not self.recent_tracked_tools:
            return False, "No tracked tools have been called yet"
        
        from datetime import datetime, timedelta
        now = datetime.now()
        cutoff_time = now - timedelta(seconds=self.verification_window_seconds)
        
        # Filter to recent calls within the window
        recent_calls = [
            tool_call for tool_call in self.recent_tracked_tools
            if tool_call['timestamp'] > cutoff_time
        ]
        
        if not recent_calls:
            oldest = self.recent_tracked_tools[-1]
            seconds_ago = (now - oldest['timestamp']).total_seconds()
            return False, f"Last tracked tool was {oldest['tool_name']} called {seconds_ago:.0f}s ago (outside {self.verification_window_seconds}s window)"
        
        # Return the most recent one
        most_recent = recent_calls[-1]
        seconds_ago = (now - most_recent['timestamp']).total_seconds()
        return True, f"{most_recent['tool_name']} called {seconds_ago:.0f}s ago"
    
    def _wrap_tool_with_tracking(self, tool):
        """Wrap a tool to log completed actions in real-time AND prevent duplicate execution"""
        original_func = tool.func
        tool_name = tool.name
        
        # Use class-level TRACKED_TOOLS constant
        tracked_tools = TRACKED_TOOLS
        
        # NOTE: Smart git_commit logic moved INSIDE wrapped_func to preserve tracking
        
        # Special handling for write_file and find_replace - protect system files
        if tool_name in ['write_file', 'find_replace']:
            def file_modify_protected(*args, **kwargs):
                file_path = kwargs.get('file_path', '')
                
                # PROTECTION 1: Block writes to COMPLETED_ACTIONS.md (system-managed)
                if 'COMPLETED_ACTIONS.md' in file_path:
                    log_agent(f"[PROTECT] 🚫 Blocked {tool_name} to COMPLETED_ACTIONS.md - file is append-only and managed by system")
                    log_agent(f"[PROTECT] Agent attempted to modify tracking file (this is expected behavior - agent wanted to 'organize' it)")
                    return "COMPLETED_ACTIONS.md is a system-managed, append-only file for tracking completed actions. You cannot modify it directly. \
The system automatically logs actions as they complete. To see what actions are completed, use read_file to view it."
                
                # PROTECTION 2: COMPLETELY block TODO.md access (execution agent cannot touch it)
                # System handles TODO.md deterministically after commits via AUTO_SYNC
                if 'TODO.md' in file_path and 'VISIBLE_TASKS.md' not in file_path:
                    log_agent(f"[PROTECT] 🚫 BLOCKED: {tool_name} to TODO.md - execution agent cannot access this file")
                    log_agent(f"[PROTECT] TODO.md is system-managed. Progress is tracked automatically after commits.")
                    return (
                        "BLOCKED: TODO.md is a system-managed file. You cannot access it directly.\n\n"
                        "YOUR WORKFLOW:\n"
                        "1. Read VISIBLE_TASKS.md to see your current task\n"
                        "2. Execute that task\n"
                        "3. Commit with commit_changes or git_commit\n"
                        "4. System AUTOMATICALLY marks task complete in TODO.md\n"
                        "5. System AUTOMATICALLY updates VISIBLE_TASKS.md with next task\n\n"
                        "You do NOT need to mark tasks - the system handles this deterministically."
                    )

                # PROTECTION 2b: COMPLETELY block VISIBLE_TASKS.md modification
                # Agent can READ it, but only SYSTEM can write/update it via _create_visible_tasks_file()
                if 'VISIBLE_TASKS.md' in file_path:
                    log_agent(f"[PROTECT] 🚫 BLOCKED: {tool_name} to VISIBLE_TASKS.md - agent cannot modify task list")
                    log_agent(f"[PROTECT] Agent tried to modify VISIBLE_TASKS.md directly - this breaks deterministic flow")
                    return (
                        "BLOCKED: VISIBLE_TASKS.md is READ-ONLY for you.\n\n"
                        "You CANNOT mark tasks complete or modify this file.\n\n"
                        "THE CORRECT WORKFLOW:\n"
                        "1. Read VISIBLE_TASKS.md to see your CURRENT task\n"
                        "2. Execute ONLY that task (don't skip ahead!)\n"
                        "3. Commit with commit_changes or git_commit\n"
                        "4. System AUTOMATICALLY updates VISIBLE_TASKS.md with next task\n\n"
                        "DO NOT try to mark tasks [x] yourself. DO NOT rewrite this file.\n"
                        "The system handles all progress tracking after your commits."
                    )

                # PROTECTION 3: Prevent analysis.md/CURRENT_STATE.md overwrites
                if tool_name == 'write_file' and (
                    'analysis.md' in file_path or 'CURRENT_STATE.md' in file_path
                ):
                    if os.path.exists(file_path):
                        try:
                            with open(file_path, 'r') as f:
                                existing_content = f.read()
                            if len(existing_content) > 50:  # Has ANY content (lowered from 200)
                                log_agent(f"[PROTECT] 🚫 Blocked write_file to existing {os.path.basename(file_path)}")
                                log_agent(f"[PROTECT] File has {len(existing_content)} chars, blocking overwrite")
                                return (
                                    f"{os.path.basename(file_path)} already exists. This file was created during initial analysis. "
                                    f"Do NOT recreate or overwrite it. The migration is already in progress. "
                                    f"Use read_file to view current state and continue from where you left off."
                                )
                        except Exception:
                            pass  # If can't read, allow write
                
                # NOTE: PROTECTION 4 (TODO.md verification) removed - now handled by complete block in PROTECTION 2

                # Allow all other file operations
                return original_func(*args, **kwargs)
            
            from langchain_core.tools import StructuredTool
            return StructuredTool(
                name=tool.name,
                description=tool.description,
                func=file_modify_protected,
                args_schema=tool.args_schema
            )
        
        # Special handling for read_file - redirect to VISIBLE_TASKS.md
        if tool_name == 'read_file':
            def read_file_with_task_redirect(*args, **kwargs):
                file_path = kwargs.get('file_path', args[0] if args else '')
                
                # REDIRECT: If reading TODO.md, read VISIBLE_TASKS.md instead
                if 'TODO.md' in file_path:
                    log_agent(f"[READ_REDIRECT] 📋 Agent tried to read TODO.md - redirecting to VISIBLE_TASKS.md")
                    visible_tasks_path = file_path.replace('TODO.md', 'VISIBLE_TASKS.md')
                    if 'file_path' in kwargs:
                        kwargs['file_path'] = visible_tasks_path
                    elif args:
                        args = (visible_tasks_path,) + args[1:]
                    return original_func(*args, **kwargs)
                
                # For all other files, return as-is
                return original_func(*args, **kwargs)
            
            from langchain_core.tools import StructuredTool
            return StructuredTool(
                name=tool.name,
                description=tool.description,
                func=read_file_with_task_redirect,
                args_schema=tool.args_schema
            )
        
        # Special handling for run_command - block TODO.md access
        if tool_name == 'run_command':
            def run_command_with_todo_block(*args, **kwargs):
                command = kwargs.get('command', args[0] if args else '')
                
                # BLOCK: Prevent TODO.md access via shell commands
                if 'TODO.md' in command:
                    log_agent(f"[CMD_BLOCK] 🚫 Execution agent tried to access TODO.md via run_command - BLOCKED")
                    log_agent(f"[CMD_BLOCK] Blocked command: {command[:100]}...")
                    return """ERROR: TODO.md is not accessible to execution agent.

🚫 FORBIDDEN ACCESS ATTEMPT 🚫

You attempted to access TODO.md directly, which is prohibited.
Use VISIBLE_TASKS.md instead - it contains your current task.

Your task tracking works like this:
1. Read VISIBLE_TASKS.md to see your CURRENT TASK
2. Complete that task
3. Commit your changes with git_commit
4. The system automatically marks the task complete
5. VISIBLE_TASKS.md is regenerated with the next task

VISIBLE_TASKS.md is the ONLY task file you can access.
TODO.md is managed by the analysis phase and is now locked."""
                
                # Allow other commands
                return original_func(*args, **kwargs)
            
            from langchain_core.tools import StructuredTool
            return StructuredTool(
                name=tool.name,
                description=tool.description,
                func=run_command_with_todo_block,
                args_schema=tool.args_schema
            )
        
        if tool_name not in tracked_tools:
            return tool  # Don't wrap, return as-is
        
        def wrapped_func(*args, **kwargs):
            # DEDUPLICATION: Check if this action is already completed
            if hasattr(self, 'project_path') and self.project_path:
                completed_actions_path = os.path.join(self.project_path, "COMPLETED_ACTIONS.md")
                
                # Read completed actions file
                try:
                    if os.path.exists(completed_actions_path):
                        with open(completed_actions_path, 'r') as f:
                            completed_content = f.read()
                        
                        # Check if this action is already done
                        action_pattern = tracked_tools[tool_name]
                        
                        # Special handling for create_branch - check specific branch name
                        if tool_name == 'create_branch':
                            branch_name = kwargs.get('branch_name', '')
                            if branch_name:
                                # Check if this exact branch was already created (both old and new format)
                                if (f"branch: {branch_name}" in completed_content or
                                    f"branch '{branch_name}'" in completed_content or
                                    f"branch_name={branch_name}" in completed_content):
                                    log_agent(f"[DEDUPE] ⚠ Branch '{branch_name}' already created (found in COMPLETED_ACTIONS) - skipping")
                                    return f"Branch '{branch_name}' was already created in a previous action. Skipping duplicate creation."
                        
                        # git_commit: Check git history + prevent retry loops
                        if tool_name == 'git_commit':
                            commit_msg = kwargs.get('message', '')
                            if commit_msg:
                                # Check if this exact commit message exists in recent git history
                                try:
                                    import subprocess
                                    result = subprocess.run(
                                        ['git', 'log', '--oneline', '-10', '--format=%s'],
                                        cwd=self.project_path,
                                        capture_output=True,
                                        text=True,
                                        timeout=5
                                    )
                                    if result.returncode == 0:
                                        recent_commits = result.stdout.strip().split('\n')
                                        if commit_msg in recent_commits:
                                            log_agent(f"[DEDUPE] ⚠ Commit '{commit_msg[:60]}...' already in git history")
                                            return f"Commit with message '{commit_msg}' was already made. Use 'git log' to verify."
                                except Exception as e:
                                    log_agent(f"[DEDUPE] Could not check git history: {e}")
                                
                                # Check for recent failed attempts (prevent retry loops)
                                import re
                                recent_failed = []
                                for line in completed_content.split('\n'):
                                    if 'git_commit | FAILED' in line and commit_msg[:30] in line:
                                        # Extract timestamp
                                        match = re.search(r'\[(\d+)\] (\d{2}:\d{2}:\d{2})', line)
                                        if match:
                                            recent_failed.append(match.group(2))
                                
                                if len(recent_failed) >= 2:
                                    log_agent(f"[DEDUPE] 🛑 Blocking git_commit - failed {len(recent_failed)} times recently")
                                    return (
                                        f"git_commit with message '{commit_msg}' failed {len(recent_failed)} times recently. "
                                        f"Last failures at: {', '.join(recent_failed[-2:])}. "
                                        f"Fix the underlying issue (nothing to commit? staging issue?) before retrying."
                                    )
                        
                        # For other tools, check if action type is already logged (CHECK ALL LINES)
                        elif action_pattern in completed_content:
                            log_agent(f"[DEDUPE] ⚠ Action '{action_pattern}' already completed (found in COMPLETED_ACTIONS) - skipping duplicate")
                            
                            # Return a message that looks like success but indicates skip
                            if tool_name == 'add_openrewrite_plugin':
                                return "OpenRewrite plugin is already configured in pom.xml (found in COMPLETED_ACTIONS). Skipping duplicate addition."
                            elif tool_name == 'configure_openrewrite_recipes':
                                return "OpenRewrite recipes are already configured (found in COMPLETED_ACTIONS). Skipping duplicate configuration."
                            elif tool_name == 'update_java_version':
                                return "Java version already updated (found in COMPLETED_ACTIONS). Skipping duplicate update."
                            elif tool_name in ['mvn_rewrite_run', 'mvn_rewrite_run_recipe']:
                                return "OpenRewrite migration already executed (found in COMPLETED_ACTIONS). Skipping duplicate execution."
                            else:
                                return f"Action already completed (found in COMPLETED_ACTIONS). Skipping duplicate."
                
                except Exception as e:
                    log_agent(f"[DEDUPE] ⚠ Could not check COMPLETED_ACTIONS for deduplication: {str(e)}")
                    # Continue with execution if we can't read the file
            
            # Execute original tool (not a duplicate) and track timing
            start_time = datetime.now()
            
            # SMART git_commit: Auto-stage files if nothing is staged (prevent repeated failures)
            if tool_name == 'git_commit':
                project_path = kwargs.get('project_path', self.project_path if hasattr(self, 'project_path') else None)
                if project_path:
                    try:
                        import subprocess
                        # Check if there are staged changes
                        status_result = subprocess.run(
                            ['git', 'diff', '--cached', '--name-only'],
                            cwd=project_path,
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        
                        if status_result.returncode == 0:
                            staged_files = status_result.stdout.strip()
                            if not staged_files:
                                # Nothing staged - check if there are changes to stage
                                changes_result = subprocess.run(
                                    ['git', 'status', '--porcelain'],
                                    cwd=project_path,
                                    capture_output=True,
                                    text=True,
                                    timeout=5
                                )
                                
                                if changes_result.returncode == 0 and changes_result.stdout.strip():
                                    log_agent(f"[SMART_COMMIT] No staged changes detected, auto-staging files before commit")
                                    # Auto-stage all changes
                                    add_result = subprocess.run(
                                        ['git', 'add', '-A'],
                                        cwd=project_path,
                                        capture_output=True,
                                        text=True,
                                        timeout=5
                                    )
                                    
                                    if add_result.returncode == 0:
                                        log_agent(f"[SMART_COMMIT] ✓ Auto-staged changes successfully")
                                    else:
                                        log_agent(f"[SMART_COMMIT] ⚠ Auto-stage failed: {add_result.stderr}")
                                else:
                                    log_agent(f"[SMART_COMMIT] No changes to stage, git_commit will likely fail")
                    except Exception as e:
                        log_agent(f"[SMART_COMMIT] Could not check/stage changes: {e}")
            
            try:
                result = original_func(*args, **kwargs)
                error = None
            except Exception as e:
                result = None
                error = str(e)
                log_agent(f"[TOOL_ERROR] {tool_name} raised exception: {e}")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            timestamp = start_time.strftime("%H:%M:%S")
            
            # Check for success
            is_success = False
            result_str = str(result) if result is not None else ""
            
            # Determine success based on tool type
            if tool_name == 'git_commit':
                is_success = 'Return code: 0' in result_str
            elif tool_name == 'create_branch':
                is_success = 'Created branch' in result_str or 'Checked out branch' in result_str
            elif tool_name == 'add_openrewrite_plugin':
                is_success = 'added' in result_str.lower() or 'success' in result_str.lower()
            elif tool_name == 'configure_openrewrite_recipes':
                is_success = 'Configured' in result_str or 'recipes' in result_str.lower()
            elif tool_name in ['mvn_rewrite_run', 'mvn_rewrite_run_recipe', 'mvn_rewrite_dry_run']:
                is_success = 'Return code: 0' in result_str and 'BUILD SUCCESS' in result_str
            else:
                # Generic success detection
                is_success = (
                    'success' in result_str.lower() or
                    'completed' in result_str.lower() or
                    'Return code: 0' in result_str
                )
            
            # Comprehensive result description
            result_desc = self._format_result(result, tool_name)
            error_desc = str(error)[:200] if error else None
            
            # Log action to COMPLETED_ACTIONS.md
            self._log_action_to_file(
                tool_name=tool_name,
                success=is_success,
                duration=duration,
                args=args,
                kwargs=kwargs,
                result=result_desc,
                error=error_desc
            )
            
            # DETERMINISTIC PROGRESS TRACKING: After successful commit (either tool), mark task complete
            COMMIT_TOOLS = {'git_commit', 'commit_changes'}
            if tool_name in COMMIT_TOOLS and is_success:
                log_agent(f"[AUTO_SYNC] 📝 Successful commit detected ({tool_name}) - marking current task complete")

                # Extract current task BEFORE marking complete (for task-level logging)
                current_task = None
                try:
                    visible_tasks_path = os.path.join(self.project_path, "VISIBLE_TASKS.md")
                    if os.path.exists(visible_tasks_path):
                        with open(visible_tasks_path, 'r') as f:
                            current_task = self._extract_current_task(f.read())
                except Exception as e:
                    log_agent(f"[AUTO_SYNC] Could not extract current task: {e}")

                # Mark task complete in TODO.md and update VISIBLE_TASKS.md
                self._update_visible_tasks_file(mark_current_complete=True)

                # Log TASK-LEVEL completion (useful external memory)
                if current_task:
                    # Try to extract commit hash from result
                    commit_hash = None
                    if result and isinstance(result, str) and 'commit' in result.lower():
                        import re
                        hash_match = re.search(r'[a-f0-9]{7,40}', result)
                        if hash_match:
                            commit_hash = hash_match.group(0)
                    self._log_task_completion(current_task, commit_hash)

                log_agent(f"[AUTO_SYNC] ✅ Task marked in TODO.md, VISIBLE_TASKS.md updated with next task")
            
            # Log ALL tracked tool executions (both success and failure) in structured format
            if hasattr(self, 'project_path') and self.project_path:
                completed_actions_path = os.path.join(self.project_path, "COMPLETED_ACTIONS.md")
                try:
                    # Get next action number
                    action_num = self._get_next_action_number()
                    
                    # Determine status
                    status = "SUCCESS" if is_success else "FAILED"
                    
                    # Build structured log entry (compact format - no extra blank lines)
                    entry_lines = [
                        f"[{action_num:03d}] {timestamp} | {tool_name} | {status} | duration={duration:.1f}s"
                    ]
                    
                    # Add arguments
                    args_str = self._format_args(kwargs)
                    entry_lines.append(f"        Args: {args_str}")
                    
                    # Add result or error
                    if is_success:
                        result_formatted = self._format_result(result, tool_name)
                        entry_lines.append(f"        Result: {result_formatted}")
                    else:
                        # Extract actionable error message
                        error_msg = self._extract_actionable_error(tool_name, result, error)
                        entry_lines.append(f"        Error: {error_msg}")
                    
                    # Append to file (add blank line before entry for readability)
                    with open(completed_actions_path, 'a') as f:
                        f.write("\n" + "\n".join(entry_lines) + "\n")
                    
                    log_agent(f"[ACTION_LOG] [{action_num:03d}] {tool_name} | {status} | {duration:.1f}s")
                    
                    # Update state header after logging
                    self._update_state_header()
                    
                    # Track action for loop detection
                    self._track_action(tool_name=tool_name, logged_to_completed=is_success)
                    
                    # Track successful tool call with timestamp for TODO.md verification
                    if is_success:
                        self.recent_tracked_tools.append({
                            'tool_name': tool_name,
                            'timestamp': datetime.now()
                        })
                        # Keep only last 10 calls to avoid memory bloat
                        if len(self.recent_tracked_tools) > 10:
                            self.recent_tracked_tools = self.recent_tracked_tools[-10:]
                        log_agent(f"[VERIFY] Recorded {tool_name} for TODO.md marking verification (window: {self.verification_window_seconds}s)")
                
                except Exception as e:
                    log_agent(f"[ACTION_LOG] ❌ Failed to log action: {str(e)}")
            else:
                # Track action even if not logged (for loop detection)
                self._track_action(tool_name=tool_name, logged_to_completed=is_success)
            
            return result
        
        # Create new tool with wrapped function
        from langchain_core.tools import StructuredTool
        wrapped_tool = StructuredTool(
            name=tool.name,
            description=tool.description,
            func=wrapped_func,
            args_schema=tool.args_schema
        )
        return wrapped_tool
    
    def _get_execution_tools(self):
        """Get WRITE/EXECUTE tools for execution agent (modify code, run recipes, commit changes)

        STREAMLINED TOOLSET - Analysis is already complete when execution starts.
        Agent should focus on VISIBLE_TASKS.md, not re-analyze the project.

        EXCLUDED (analysis tools - already done by analysis_expert):
        - find_all_poms, list_java_files, search_files (discovery)
        - get_java_version, list_dependencies (can use read_file instead)
        - check_migration_state (supervisor tool, not for execution)
        """
        # Execution phase: modify files, run OpenRewrite, commit changes, validate builds
        execution_tool_names = {
            # File operations (read for context, write for changes)
            'read_file', 'write_file', 'find_replace', 'file_exists',
            # Read pom for targeted checks before modification
            'read_pom',
            # Maven operations (execution - these MODIFY the project)
            'configure_openrewrite_recipes', 'update_java_version',
            'add_openrewrite_plugin', 'update_spring_boot_version',
            # OpenRewrite execution
            'mvn_rewrite_run', 'mvn_rewrite_run_recipe', 'mvn_rewrite_dry_run',
            # Git operations (write)
            'create_branch', 'checkout_branch', 'commit_changes',
            'git_add_all', 'git_commit', 'git_status', 'tag_checkpoint',
            # Build validation
            'mvn_compile', 'mvn_test',
            # Command execution (for edge cases)
            'run_command',
            # Completion marking
            'mark_execution_complete',
        }
        
        execution_tools = [
            tool for tool in all_tools_flat
            if tool.name in execution_tool_names
        ]
        
        # Wrap tools with tracking (logs to COMPLETED_ACTIONS.md)
        wrapped_tools = [self._wrap_tool_with_tracking(tool) for tool in execution_tools]
        
        log_agent(f"[TOOLS] Execution agent has {len(wrapped_tools)} tools (write + execute)")
        log_agent(f"[TOOLS] Execution tools: {sorted([t.name for t in wrapped_tools])}")
        return wrapped_tools
    
    def _wrap_error_read_file(self, tool):
        """Wrap read_file for error agent to block state files
        
        Error agent should ONLY see:
        - Error messages (injected in prompt)
        - Project code files (pom.xml, .java files)
        - Error history
        
        Error agent should NOT see:
        - TODO.md (it's not executing tasks)
        - CURRENT_STATE.md (irrelevant to error fixing)
        - VISIBLE_TASKS.md (irrelevant to error fixing)
        - COMPLETED_ACTIONS.md (irrelevant to error fixing)
        """
        original_func = tool.func
        
        def error_read_file_blocked(*args, **kwargs):
            file_path = kwargs.get('file_path', args[0] if args else '')
            
            # Block state files
            blocked_files = ['TODO.md', 'CURRENT_STATE.md', 'VISIBLE_TASKS.md', 'COMPLETED_ACTIONS.md']
            for blocked in blocked_files:
                if blocked in file_path:
                    log_agent(f"[ERROR_BLOCK] 🚫 Error agent tried to read {blocked} - BLOCKED")
                    return f"ERROR: {blocked} is not accessible to error agent. Focus on fixing the build error only."
            
            # Allow reading ERROR_HISTORY.md and project files
            return original_func(*args, **kwargs)
        
        from langchain_core.tools import StructuredTool
        return StructuredTool(
            name=tool.name,
            description=tool.description,
            func=error_read_file_blocked,
            args_schema=tool.args_schema
        )
    
    def _get_error_tools(self):
        """Get DIAGNOSTIC tools for error agent (diagnose errors, log to ERROR_HISTORY.md)"""
        # Error phase: diagnose build/test errors, log findings to ERROR_HISTORY.md
        # NO write_file - error agent should NOT modify TODO.md or project files!
        # NO read_file - error agent wrapped separately to block state files
        error_tool_names = {
            # Read tools - will be wrapped to block state files
            'read_file', 'file_exists',
            # Diagnostic tools
            'mvn_compile', 'mvn_test',
            # Git operations (diagnostic)
            'git_status', 'get_log', 'list_branches'
            # REMOVED: run_command (too dangerous - agent did manual sed/find edits)
            # REMOVED: check_migration_state (confuses agent about current phase)
        }
        
        error_tools = [
            tool for tool in all_tools_flat
            if tool.name in error_tool_names
        ]
        
        # Wrap read_file to block state files (TODO, CURRENT_STATE, VISIBLE_TASKS)
        wrapped_error_tools = []
        for tool in error_tools:
            if tool.name == 'read_file':
                wrapped_error_tools.append(self._wrap_error_read_file(tool))
            else:
                wrapped_error_tools.append(tool)
        
        log_agent(f"[TOOLS] Error agent has {len(wrapped_error_tools)} tools (diagnostic + fix)")
        log_agent(f"[TOOLS] Error agent CANNOT read: TODO.md, CURRENT_STATE.md, VISIBLE_TASKS.md")
        log_agent(f"[TOOLS] Error tools: {sorted([t.name for t in wrapped_error_tools])}")
        return wrapped_error_tools
    
    def _route_next_agent(self, state: State) -> str:
        """Deterministic router based on state flags (NO LLM)
        
        Returns:
            - 'analysis_expert': Need analysis
            - 'execution_expert': Execute tasks
            - 'error_expert': Fix build errors
            - 'END': Success (all tasks complete)
            - 'FAILED': Failure (errors, timeout, etc.)
        """
        current_phase = state.get('current_phase', 'INIT')
        has_build_error = state.get('has_build_error', False)
        error_count = state.get('error_count', 0)
        execution_done = state.get('execution_done', False)
        analysis_done = state.get('analysis_done', False)
        
        log_agent(f"[ROUTER] Current phase: {current_phase}")
        log_agent(f"[ROUTER] Analysis done: {analysis_done}")
        log_agent(f"[ROUTER] Execution done: {execution_done}")
        log_agent(f"[ROUTER] Build error: {has_build_error} (count: {error_count})")
        
        # Check for timeout condition
        if current_phase == "EXECUTION_TIMEOUT":
            log_agent("[ROUTER] → Execution timeout reached, ending migration", "WARNING")
            return "END"
        
        # REMOVED: Stuck loop detection was causing false "success" at 39% completion
        # Loop detection remains active for logging, but doesn't force END anymore
        
        # PRIORITY 1: If execution is complete (all TODOs done), END regardless of errors
        # The agent has finished all planned tasks - errors are acceptable
        if execution_done:
            if has_build_error:
                log_agent("[ROUTER] → Execution complete with build errors - ending migration (TODO tasks done)")
                log_agent("[ROUTER] → Build errors detected but all TODO items completed - migration finished")
            else:
                log_agent("[ROUTER] → Migration complete successfully, ending")
            return "END"
        
        # PRIORITY 2: Check for build errors ONLY if execution is NOT complete
        if has_build_error and error_count < 3:
            log_agent(f"[ROUTER] → Build error detected, routing to error_expert (attempt {error_count}/3)")
            return "error_expert"
        elif has_build_error and error_count >= 3:
            log_agent("[ROUTER] → Max error attempts reached, MIGRATION FAILED", "ERROR")
            log_summary("MIGRATION FAILED: Max error attempts (3) reached without resolution")
            return "FAILED"
        
        # PRIORITY 3: Route based on explicit state flags
        if not analysis_done:
            log_agent("[ROUTER] → Routing to analysis_expert")
            return "analysis_expert"
        else:
            log_agent("[ROUTER] → Routing to execution_expert")
            return "execution_expert"
    
    def _wrap_analysis_node(self, state: State):
        """Wrapper for analysis agent with automatic completion detection"""
        log_agent("[WRAPPER] Running analysis_expert")
        
        # Get analysis agent (first worker)
        analysis_agent = self.migration_workers[0]
        
        # Run the agent
        result = analysis_agent.invoke(state)
        
        # Auto-detect completion
        project_path = state.get("project_path", "")
        messages = result.get("messages", [])
        
        analysis_complete = detect_analysis_complete(project_path, messages)
        
        if analysis_complete:
            log_agent("[WRAPPER] 🎉 Analysis AUTO-DETECTED as complete")
            log_summary("ANALYSIS PHASE: AUTO-COMPLETED (files created)")
        
        # Update state with detection result
        return {
            "messages": messages,
            "analysis_done": analysis_complete,
            "current_phase": "ANALYSIS_COMPLETE" if analysis_complete else "ANALYSIS"
        }
    
    # ===== External Memory Helper Methods =====
    
    def _read_state_file(self, filename: str, keep_beginning: bool = False) -> str:
        """Read file from project dir, return empty if not exists

        Args:
            filename: Name of file in project directory
            keep_beginning: If True, truncate from END (keep beginning).
                           If False, truncate from START (keep end - for logs).
                           Default False for backwards compatibility.
        """
        if not self.project_path:
            return ""

        filepath = os.path.join(self.project_path, filename)
        if not os.path.exists(filepath):
            return ""

        try:
            with open(filepath, 'r') as f:
                content = f.read()
            # Truncate if too long
            if len(content) > MAX_SUMMARY_LENGTH:
                if keep_beginning:
                    # Keep first N chars (for TODO.md - need first tasks)
                    return content[:MAX_SUMMARY_LENGTH]
                else:
                    # Keep last N chars (for logs - need recent entries)
                    return content[-MAX_SUMMARY_LENGTH:]
            return content
        except Exception as e:
            log_agent(f"[MEMORY] Error reading {filename}: {str(e)}", "ERROR")
            return ""
    
    def _append_state_file(self, filename: str, content: str):
        """Append to file in project dir"""
        if not self.project_path:
            return

        filepath = os.path.join(self.project_path, filename)
        try:
            with open(filepath, 'a') as f:
                f.write(content + "\n")
        except Exception as e:
            log_agent(f"[MEMORY] Error writing to {filename}: {str(e)}", "ERROR")

    def _log_action_to_file(self, tool_name: str, success: bool, duration: float,
                            args: tuple = None, kwargs: dict = None,
                            result: str = None, error: str = None):
        """Log completed action to COMPLETED_ACTIONS.md for external memory tracking"""
        if not self.project_path:
            return

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "SUCCESS" if success else "FAILED"

        # Format the action entry
        entry = f"[{timestamp}] {tool_name}: {status}"
        if duration:
            entry += f" ({duration:.1f}s)"

        # Add brief result/error info
        if error:
            entry += f" - Error: {error[:100]}"
        elif result and len(result) < 100:
            entry += f" - {result}"

        self._append_state_file("COMPLETED_ACTIONS.md", entry)

    def _log_task_completion(self, task_description: str, commit_hash: str = None):
        """Log TASK-LEVEL completion to COMPLETED_ACTIONS.md

        This creates high-level task tracking that survives context trimming.
        Different from tool-level logging - this shows WHAT was accomplished.
        """
        if not self.project_path:
            return

        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Build task completion entry
        commit_info = f" (commit: {commit_hash[:7]})" if commit_hash else ""
        entry = f"[{timestamp}] ✅ TASK COMPLETED: {task_description}{commit_info}"

        # Append to special TASK COMPLETIONS section
        completed_actions_path = os.path.join(self.project_path, "COMPLETED_ACTIONS.md")
        try:
            # Read existing content
            content = ""
            if os.path.exists(completed_actions_path):
                with open(completed_actions_path, 'r') as f:
                    content = f.read()

            # Find or create TASK COMPLETIONS section
            if '=== TASK COMPLETIONS ===' not in content:
                # Insert section after header
                if '=== ACTION LOG ===' in content:
                    parts = content.split('=== ACTION LOG ===')
                    content = parts[0] + "=== TASK COMPLETIONS ===\n\n=== ACTION LOG ===" + parts[1]
                else:
                    content = "=== TASK COMPLETIONS ===\n\n" + content

            # Insert new completion entry
            if '=== TASK COMPLETIONS ===' in content and '=== ACTION LOG ===' in content:
                parts = content.split('=== ACTION LOG ===')
                task_section = parts[0]
                action_section = '=== ACTION LOG ===' + parts[1]

                # Add entry to task section
                new_content = task_section.rstrip() + "\n" + entry + "\n\n" + action_section
            else:
                new_content = content + "\n" + entry

            with open(completed_actions_path, 'w') as f:
                f.write(new_content)

            log_agent(f"[TASK_LOG] ✅ Logged task completion: {task_description[:60]}...")
        except Exception as e:
            log_agent(f"[TASK_LOG] Error logging task completion: {e}")

    def _get_visible_tasks(self, todo_content: str, max_visible: int = 3) -> dict:
        """Extract only next N unchecked tasks from TODO to show agent
        
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
            return {
                'current': None,
                'upcoming': [],
                'completed_count': 0,
                'total_count': 0,
                'remaining_count': 0,
                'all_done': True
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
    
    def _update_visible_tasks_file(self, mark_current_complete: bool = False):
        """Update VISIBLE_TASKS.md with fresh next 3 tasks after TODO.md changes
        
        Args:
            mark_current_complete: If True, mark the current VISIBLE task as complete in TODO.md first
        """
        if not self.project_path:
            return
        
        try:
            # If requested, mark current task complete in TODO.md
            if mark_current_complete:
                visible_tasks_path = os.path.join(self.project_path, "VISIBLE_TASKS.md")
                if os.path.exists(visible_tasks_path):
                    with open(visible_tasks_path, 'r') as f:
                        current_visible = f.read()
                    
                    # Extract current task
                    current_task = self._extract_current_task(current_visible)
                    if current_task:
                        log_agent(f"[AUTO_SYNC] Marking task complete after commit: {current_task[:60]}...")
                        self._mark_task_in_todo(current_task)
            
            # Read TODO.md and regenerate VISIBLE_TASKS.md
            todo_content = self._read_state_file("TODO.md", keep_beginning=True)
            if todo_content:
                visible_tasks = self._get_visible_tasks(todo_content, max_visible=3)
                self._create_visible_tasks_file(visible_tasks)
        except Exception as e:
            log_agent(f"[VISIBLE_TASKS] Error updating file: {str(e)}", "ERROR")
    
    def _extract_current_task(self, visible_content: str) -> str:
        """Extract the current task description from VISIBLE_TASKS.md content"""
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
            log_agent(f"[AUTO_SYNC] Error extracting current task: {str(e)}", "ERROR")
            return None
    
    def _mark_task_in_todo(self, task_description: str):
        """Deterministically mark a task as complete in TODO.md"""
        if not self.project_path or not task_description:
            return
        
        todo_path = os.path.join(self.project_path, "TODO.md")
        if not os.path.exists(todo_path):
            return
        
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
                
                log_agent(f"[AUTO_SYNC] ✅ Marked task complete in TODO.md: {task_description[:60]}...")
                log_summary(f"AUTO_SYNC: Task marked complete in TODO.md")
            else:
                log_agent(f"[AUTO_SYNC] ⚠ Task not found in TODO.md: {task_description[:60]}...", "WARNING")
        
        except Exception as e:
            log_agent(f"[AUTO_SYNC] Error marking task in TODO.md: {str(e)}", "ERROR")
    
    def _create_visible_tasks_file(self, visible_tasks: dict):
        """Create VISIBLE_TASKS.md file with only next 3 unchecked tasks
        
        This prevents agent from bypassing read_file filter using run_command.
        Agent can 'cat VISIBLE_TASKS.md' all day - it will only see 3 tasks.
        
        Args:
            visible_tasks: Dictionary from _get_visible_tasks()
        """
        if not self.project_path:
            return
        
        visible_tasks_path = os.path.join(self.project_path, "VISIBLE_TASKS.md")
        
        try:
            if visible_tasks['all_done']:
                content = """# Visible Tasks

🎉 ALL TASKS COMPLETE! 🎉

All migration tasks have been marked as complete in TODO.md.
The migration is finished."""
            elif visible_tasks['current']:
                content = f"""# Visible Tasks

⚠ This file shows only your next 3 tasks. Complete these before moving forward.

## ✅ CURRENT TASK (DO THIS NOW):

- [ ] {visible_tasks['current']}

⚠ Complete the CURRENT TASK above before attempting upcoming tasks.
"""
                if visible_tasks['upcoming']:
                    content += "\n## 📋 UPCOMING TASKS (for reference - complete current first):\n\n"
                    for task in visible_tasks['upcoming']:
                        content += f"- [ ] {task}\n"
                
                hidden_count = visible_tasks['remaining_count'] - len(visible_tasks['upcoming']) - 1
                if hidden_count > 0:
                    content += f"\n🔒 {hidden_count} additional tasks are hidden and will be shown after you complete current tasks.\n"
                
                content += f"\n📊 Progress: {visible_tasks['completed_count']}/{visible_tasks['total_count']} complete\n"
            else:
                content = "# Visible Tasks\n\n(No tasks defined yet)"
            
            with open(visible_tasks_path, 'w') as f:
                f.write(content)
            
            log_agent(f"[VISIBLE_TASKS] Created {visible_tasks_path} with next {min(3, visible_tasks['remaining_count'])} tasks")
            log_agent(f"[VISIBLE_TASKS] Agent can use run_command on this file - it only contains visible tasks")
        except Exception as e:
            log_agent(f"[VISIBLE_TASKS] Error creating file: {str(e)}", "ERROR")
    
    def _build_external_memory_block(self) -> str:
        """Build structured external memory block from state files
        
        This creates a clear, structured block containing:
        - COMPLETED_ACTIONS (what's already done - DO NOT REPEAT)
        - TODO (what remains - with next action highlighted)
        - CURRENT_STATE (current migration status)
        
        Returns:
            Formatted string for injection into system prompt
        """
        if not self.project_path:
            return ""
        
        from datetime import datetime
        
        # Read state files (with safe fallbacks)
        completed = self._read_state_file("COMPLETED_ACTIONS.md")
        todo = self._read_state_file("TODO.md", keep_beginning=True)
        current_state = self._read_state_file("CURRENT_STATE.md")
        
        # If no files exist yet, return empty
        if not completed and not todo and not current_state:
            return ""
        
        # Get only next 3 visible tasks (restrict agent view)
        visible_tasks = self._get_visible_tasks(todo, max_visible=3)
        
        # Log visibility information
        if visible_tasks['all_done']:
            log_agent(f"[TASK_VISIBILITY] ✅ ALL TASKS COMPLETE - {visible_tasks['completed_count']}/{visible_tasks['total_count']} done")
        elif visible_tasks['current']:
            log_agent(f"[TASK_VISIBILITY] 👁 Agent View Restriction Active:")
            log_agent(f"[TASK_VISIBILITY] ▪ VISIBLE - Current Task: {visible_tasks['current'][:80]}...")
            if visible_tasks['upcoming']:
                for idx, task in enumerate(visible_tasks['upcoming'], 1):
                    log_agent(f"[TASK_VISIBILITY] ▪ VISIBLE - Upcoming {idx}: {task[:80]}...")
            hidden_count = visible_tasks['remaining_count'] - len(visible_tasks['upcoming']) - 1
            if hidden_count > 0:
                log_agent(f"[TASK_VISIBILITY]   🔒 HIDDEN - {hidden_count} tasks not shown to agent")
            log_agent(f"[TASK_VISIBILITY] ▪ Progress: {visible_tasks['completed_count']}/{visible_tasks['total_count']} complete ({visible_tasks['remaining_count']} remaining)")
        else:
            log_agent(f"[TASK_VISIBILITY] ❌ No TODO file yet - analysis phase?")
        
        # Build structured memory block
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Format visible tasks section
        if visible_tasks['all_done']:
            tasks_section = """
✅ ALL TASKS COMPLETE! ✅

All TODO items are marked [x]. Migration work is done.
The system will automatically detect completion.
"""
        elif visible_tasks['current']:
            tasks_section = f"""
✔ CURRENT TASK (DO THIS NOW):
════════════════════════════════════════════════════════════════════════════════

{visible_tasks['current']}

════════════════════════════════════════════════════════════════════════════════
"""
            # Add upcoming tasks if they exist
            if visible_tasks['upcoming']:
                tasks_section += "\n▪ UPCOMING TASKS (for context only - DO NOT start these yet):\n"
                tasks_section += "────────────────────────────────────────────────────────────────────────────────\n\n"
                for idx, task in enumerate(visible_tasks['upcoming'], 1):
                    tasks_section += f"  {idx}. {task}\n"
                tasks_section += "\n⚠ Complete CURRENT TASK before attempting these!\n"
                tasks_section += "────────────────────────────────────────────────────────────────────────────────"
            
            # Add progress stats
            tasks_section += f"""

▪ PROGRESS: {visible_tasks['completed_count']}/{visible_tasks['total_count']} tasks complete ({visible_tasks['remaining_count']} remaining)
"""
        else:
            tasks_section = "(No TODO file yet - wait for analysis phase)"
        
        memory_block = f"""
════════════════════════════════════════════════════════════════════════════════
EXTERNAL MEMORY (Updated: {timestamp}) - READ THIS BEFORE EVERY ACTION
════════════════════════════════════════════════════════════════════════════════

⚠ CRITICAL RULES - READ BEFORE EVERY ACTION ⚠

✔ YOUR WORKFLOW:
1. Look at CURRENT TASK below (this is the ONLY task you should do now)
2. Execute that ONE task using appropriate tools
3. Mark it complete in TODO.md by changing [ ] to [x]
4. Commit your changes
5. The system will show you the NEXT task

🚫 NEVER:
- Work on UPCOMING TASKS (those are shown for context only)
- Repeat any action in COMPLETED ACTIONS below
- Try to modify COMPLETED_ACTIONS.md (it's system-managed)
- Skip the current task to do something else

▪ MIGRATION COMPLETE WHEN:
- All tasks show [x] in TODO.md
- System will automatically detect completion

════════════════════════════════════════════════════════════════════════════════
▌ COMPLETED ACTIONS (DO NOT REPEAT THESE)
════════════════════════════════════════════════════════════════════════════════

{completed if completed else "(No actions completed yet)"}

════════════════════════════════════════════════════════════════════════════════
▌ YOUR TASKS (Only showing next 3 - others hidden until these complete)
════════════════════════════════════════════════════════════════════════════════

{tasks_section}

════════════════════════════════════════════════════════════════════════════════
▌ CURRENT STATE (Migration Status)
════════════════════════════════════════════════════════════════════════════════

{current_state if current_state else "(No state file yet)"}

════════════════════════════════════════════════════════════════════════════════
""".strip()
        
        return memory_block
    
    def _detect_stuck_loop(self) -> tuple[bool, str]:
        """Detect if agent is stuck in a repetitive loop
        
        Analyzes recent actions to identify patterns:
        1. Same tool called repeatedly (3+ times in window)
        2. No completed actions logged in recent window
        3. Same TODO item attempted repeatedly without completion
        
        Returns:
            (is_stuck: bool, reason: str)
        """
        if len(self.recent_actions) < self.action_window_size:
            return (False, "Not enough action history yet")
        
        # Get last N actions
        last_n = self.recent_actions[-self.action_window_size:]
        
        # Pattern 1: Same tool called too many times
        from collections import Counter
        tool_counts = Counter(a.get('tool_name') for a in last_n if a.get('tool_name'))
        for tool, count in tool_counts.items():
            if count >= 3:
                log_agent(f"[LOOP_DETECT] 🔴 Tool '{tool}' called {count} times in last {self.action_window_size} actions", "WARNING")
                return (True, f"Tool '{tool}' called {count} times in last {self.action_window_size} actions")
        
        # Pattern 2: No completed actions logged
        completions_logged = sum(1 for a in last_n if a.get('logged_to_completed'))
        if completions_logged == 0:
            log_agent(f"[LOOP_DETECT] 🔴 No actions logged to COMPLETED_ACTIONS in last {self.action_window_size} calls", "WARNING")
            return (True, f"No progress: 0 completions in last {self.action_window_size} actions")
        
        # Pattern 3: Same TODO item attempted repeatedly
        todo_counts = Counter(a.get('todo_item') for a in last_n if a.get('todo_item'))
        for todo_item, count in todo_counts.items():
            if count >= 3:
                log_agent(f"[LOOP_DETECT] 🔴 TODO '{todo_item[:50]}...' attempted {count} times", "WARNING")
                return (True, f"TODO item attempted {count} times without completion")
        
        return (False, "Agent making progress")
    
    def _track_action(self, tool_name: str, todo_item: str = None, logged_to_completed: bool = False):
        """Track an action for loop detection
        
        Args:
            tool_name: Name of the tool that was called
            todo_item: Current TODO item being worked on (if known)
            logged_to_completed: Whether this action was logged to COMPLETED_ACTIONS.md
        """
        action = {
            'tool_name': tool_name,
            'todo_item': todo_item,
            'logged_to_completed': logged_to_completed,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        self.recent_actions.append(action)
        
        # Keep only last 20 actions in memory (2x window size for history)
        if len(self.recent_actions) > self.action_window_size * 4:
            self.recent_actions = self.recent_actions[-self.action_window_size * 4:]
    
    def _is_migration_complete(self) -> tuple[bool, str, dict]:
        """Check if migration is complete based on TODO.md status AND COMPLETED_ACTIONS.md verification
        
        Deterministic completion check:
        - Migration is complete when ALL TODO items are [x]
        - NO [ ] unchecked items remain
        - COMPLETED_ACTIONS.md shows real work was done (not just checkmarks)
        
        Returns:
            (is_complete: bool, reason: str, stats: dict)
        """
        if not self.project_path:
            return (False, "No project path set", {})
        
        todo_path = os.path.join(self.project_path, "TODO.md")
        if not os.path.exists(todo_path):
            return (False, "TODO.md not created yet", {})
        
        try:
            with open(todo_path, 'r') as f:
                todo_content = f.read()
        except Exception as e:
            log_agent(f"[COMPLETION] Failed to read TODO.md: {str(e)}", "ERROR")
            return (False, f"Failed to read TODO.md: {str(e)}", {})
        
        # Count checked vs unchecked items
        total_tasks = todo_content.count('- [ ]') + todo_content.count('- [x]')
        completed_tasks = todo_content.count('- [x]')
        unchecked_tasks = todo_content.count('- [ ]')
        
        # Read COMPLETED_ACTIONS.md to verify real work was done
        completed_actions_path = os.path.join(self.project_path, "COMPLETED_ACTIONS.md")
        action_count = 0
        has_real_work = False
        
        if os.path.exists(completed_actions_path):
            try:
                with open(completed_actions_path, 'r') as f:
                    actions_content = f.read()
                # Count logged actions (both old and new format)
                old_format_count = actions_content.count('✅')
                new_format_success = actions_content.count('| SUCCESS |')
                new_format_failed = actions_content.count('| FAILED |')
                action_count = max(old_format_count, new_format_success + new_format_failed)
                
                # Check for key migration actions (OpenRewrite, commits, etc.)
                has_real_work = any(keyword in actions_content for keyword in [
                    'OpenRewrite', 'COMMIT', 'recipe', 'branch', 'Java version',
                    'create_branch', 'git_commit', 'mvn_rewrite'
                ])
            except Exception as e:
                log_agent(f"[COMPLETION] Could not read COMPLETED_ACTIONS.md: {str(e)}")
        
        stats = {
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'unchecked_tasks': unchecked_tasks,
            'logged_actions': action_count,
            'has_real_work': has_real_work,
            'completion_percentage': round(completed_tasks / total_tasks * 100, 1) if total_tasks > 0 else 0
        }
        
        # DETERMINISTIC RULE: Complete when 0 unchecked tasks
        if total_tasks == 0:
            return (False, "No TODO tasks found in TODO.md", stats)
        
        if unchecked_tasks == 0:
            # Additional verification: Check if COMPLETED_ACTIONS shows real work
            if not has_real_work:
                log_agent(f"[COMPLETION] ⚠ All TODO tasks marked [x] but no real work logged in COMPLETED_ACTIONS.md", "WARNING")
                log_agent(f"[COMPLETION] TODO shows {completed_tasks} tasks done, but COMPLETED_ACTIONS shows only {action_count} logged actions")
                return (False, f"Tasks marked complete but verification failed: only {action_count} actions logged", stats)
            
            log_agent(f"[COMPLETION] ✅ Migration complete: {completed_tasks}/{total_tasks} tasks done, {action_count} actions logged")
            return (True, f"All {completed_tasks}/{total_tasks} tasks completed with {action_count} logged actions ({stats['completion_percentage']}%)", stats)
        
        reason = f"{completed_tasks}/{total_tasks} tasks done ({action_count} logged actions), {unchecked_tasks} remaining ({stats['completion_percentage']}% complete)"
        return (False, reason, stats)
    
    def _extract_latest_error(self, messages: List[BaseMessage]) -> str:
        """Extract most recent error from tool messages"""
        for msg in reversed(messages):
            msg_content = ""
            msg_name = ""
            
            if isinstance(msg, dict):
                msg_content = str(msg.get('content', ''))
                msg_name = msg.get('name', '')
            elif hasattr(msg, 'content'):
                msg_content = str(msg.content)
                msg_name = getattr(msg, 'name', '')
            
            # Check for Maven errors in tool output
            if (msg_name and 'mvn' in msg_name.lower()) or 'BUILD FAILURE' in msg_content or 'ERROR' in msg_content:
                # Extract error portion only (first 500 chars)
                if len(msg_content) > 500:
                    return msg_content[:500]
                return msg_content
        
        return "Unknown error"
    
    def _error_already_attempted(self, current_error: str, history: str) -> bool:
        """Check if this exact error signature already in history"""
        # Simple heuristic: first N chars of error
        if not current_error or not history:
            return False
        
        error_sig = current_error[:ERROR_SIGNATURE_LENGTH]
        return error_sig in history
    
    # DEPRECATED: Old logging system - commented out in favor of _wrap_tool_with_tracking()
    # This function scanned messages AFTER execution and logged in old format: "✅ COMMIT: message [17:52:40]"
    # New system logs DURING execution in structured format: "[001] 17:47:23 | tool_name | SUCCESS | duration=0.3s"
    # def _update_completed_actions(self, old_messages: List[BaseMessage], new_messages: List[BaseMessage]):
    #     """Detect and record completed actions IMMEDIATELY when they happen"""
    #     timestamp = datetime.now().strftime("%H:%M:%S")
    #     start_idx = len(old_messages)
    #
    #     # Debug: Log that tracking is being called
    #     new_msg_count = len(new_messages) - start_idx
    #     if new_msg_count > 0:
    #         log_agent(f"[TRACKING] Checking {new_msg_count} new messages for completed actions")
    #
    #     # Critical tools to track (prevent repetition) - NOW USING CLASS-LEVEL TRACKED_TOOLS CONSTANT
    #     tracked_tools = TRACKED_TOOLS
    #
    #     for msg in new_messages[start_idx:]:
    #         msg_name = ""
    #         msg_content = ""
    #
    #         if isinstance(msg, dict):
    #             msg_name = msg.get('name', '')
    #             msg_content = str(msg.get('content', ''))
    #         elif hasattr(msg, 'name'):
    #             msg_name = getattr(msg, 'name', '')
    #             msg_content = str(getattr(msg, 'content', ''))
    #
    #         # Check if this is a tracked tool with a successful result - DUPLICATE SUCCESS DETECTION
    #         is_success = False
    #
    #         if msg_name == 'git_commit':
    #             # git_commit uses run_command internally, check for Return code: 0
    #             is_success = 'Return code: 0' in msg_content
    #         elif msg_name == 'add_openrewrite_plugin':
    #             is_success = 'Successfully added' in msg_content or 'Successfully configured' in msg_content
    #         elif msg_name == 'configure_openrewrite_recipes':
    #             is_success = 'Successfully configured' in msg_content or 'Successfully added' in msg_content
    #         elif msg_name == 'update_java_version':
    #             # Only success if it says "Updated" AND not "No...found"
    #             is_success = 'Updated' in msg_content and 'No Java version properties found' not in msg_content
    #         elif msg_name == 'create_branch':
    #             is_success = 'Created branch' in msg_content
    #         elif msg_name == 'mvn_rewrite_run':
    #             is_success = 'Return code: 0' in msg_content or 'BUILD SUCCESS' in msg_content
    #         elif msg_name == 'mvn_rewrite_run_recipe':
    #             is_success = 'Return code: 0' in msg_content or 'BUILD SUCCESS' in msg_content
    #
    #         # Debug: Log what we're checking
    #         if msg_name in tracked_tools:
    #             log_agent(f"[TRACKING] Found tracked tool: {msg_name} | success: {is_success} | preview: {msg_content[:100]}...")
    #
    #         # Track critical tool calls immediately - OLD FORMAT LOGGING
    #         if msg_name in tracked_tools and is_success:
    #             if msg_name == 'git_commit':
    #                 # Extract commit message from git output format:
    #                 # "Return code: 0\nSTDOUT:\n[branch hash] Commit message\n 1 file changed..."
    #                 commit_msg = "Committed changes"
    #                 if '] ' in msg_content:
    #                     # Extract text after "] " (the commit message)
    #                     parts = msg_content.split('] ', 1)
    #                     if len(parts) > 1:
    #                         commit_msg = parts[1].split('\n')[0][:80]  # First line only
    #                 action_desc = f"✅ COMMIT: {commit_msg}"
    #             elif msg_name == 'configure_openrewrite_recipes':
    #                 # Extract recipe names if possible
    #                 action_desc = f"✅ {tracked_tools[msg_name]}"
    #                 if "recipes" in msg_content.lower():
    #                     # Try to extract recipe count or names
    #                     action_desc += f" (see pom.xml for details)"
    #             else:
    #                 action_desc = f"✅ {tracked_tools[msg_name]}"
    #
    #             self._append_state_file(
    #                 "COMPLETED_ACTIONS.md",
    #                 f"{action_desc} [{timestamp}]"
    #             )
    #             log_agent(f"[MEMORY] 📝 Logged to COMPLETED_ACTIONS: {action_desc}")
    
    def _append_error_attempt(self, error: str, attempt_num: int, was_successful: bool, attempted_fixes: list = None):
        """Record error resolution attempt to ERROR_HISTORY.md
        
        Args:
            error: Error message/description
            attempt_num: Which attempt number (1, 2, 3)
            was_successful: Whether the error was resolved
            attempted_fixes: List of fix descriptions (e.g., ["Updated pom.xml dependency", "Fixed import"])
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        error_snippet = error[:200] if error else "Unknown error"  # First 200 chars
        
        # Check if this is a new error or continuation
        error_history = self._read_state_file("ERROR_HISTORY.md")
        if f"## Error #{attempt_num}" not in error_history:
            # New error - create header
            self._append_state_file(
                "ERROR_HISTORY.md",
                f"\n## [x] Error #{attempt_num}: {error_snippet} [{timestamp}]"
            )
        
        # Record attempt result with details
        status = "✅ RESOLVED" if was_successful else "❌ FAILED"
        self._append_state_file(
            "ERROR_HISTORY.md",
            f"- [x] Attempt {attempt_num}: {status} [{timestamp}]"
        )
        
        # Log what fixes were attempted (if provided)
        if attempted_fixes:
            for fix in attempted_fixes:
                self._append_state_file(
                    "ERROR_HISTORY.md",
                    f"  - Tried: {fix}"
                )
        
        log_agent(f"[ERROR_HISTORY] Logged error attempt #{attempt_num}: {status}")
    
    def _detect_build_error(self, messages: List[BaseMessage]) -> tuple[bool, str]:
        """Check if messages contain build errors from Maven or compilation"""
        for msg in reversed(messages):
            msg_content = ""
            msg_name = ""
            
            if isinstance(msg, dict):
                msg_content = str(msg.get('content', ''))
                msg_name = msg.get('name', '')
            elif hasattr(msg, 'content'):
                msg_content = str(msg.content)
                msg_name = getattr(msg, 'name', '')
            
            # Check for Maven/build failures
            if msg_name and ('mvn' in msg_name.lower() or 'compile' in msg_name.lower()):
                if 'BUILD FAILURE' in msg_content or 'BUILD ERROR' in msg_content or '[ERROR]' in msg_content:
                    # Extract error summary
                    error_lines = [line for line in msg_content.split('\n') if 'ERROR' in line]
                    error_summary = '\n'.join(error_lines[:5]) if error_lines else msg_content[:500]
                    return True, error_summary
        
        return False, ""
    
    def _wrap_execution_node(self, state: State):
        """Wrapper for execution agent with automatic completion detection + stuck detection + external memory"""
        project_path = state.get("project_path", "")
        self.project_path = project_path  # Set for memory helpers
        total_loops = state.get("total_execution_loops", 0) + 1
        
        log_agent(f"[WRAPPER] Running execution_expert (loop #{total_loops})")
        
        # Check if we've hit max loops per phase
        if total_loops > MAX_EXECUTION_LOOPS_PER_PHASE:
            log_agent(f"[STUCK] Max execution loops ({MAX_EXECUTION_LOOPS_PER_PHASE}) exceeded - forcing completion", "WARNING")
            log_summary(f"WARNING: Execution phase exceeded {MAX_EXECUTION_LOOPS_PER_PHASE} loops - stopping")
            return {
                "messages": state.get("messages", []),
                "execution_done": False,  # Not truly done, but stopping
                "current_phase": "EXECUTION_TIMEOUT",
                "total_execution_loops": total_loops
            }
        
        # Check for stuck loop patterns (LOG ONLY - don't force stop)
        if total_loops >= 3:  # Start checking after a few loops
            is_stuck, stuck_reason = self._detect_stuck_loop()
            if is_stuck:
                log_agent(f"[STUCK] ⚠ Loop pattern detected: {stuck_reason}", "WARNING")
                log_summary(f"LOOP DETECTED: {stuck_reason} - agent may be stuck")
                log_console(f"⚠ Loop pattern: {stuck_reason}", "WARNING")
                # Continue execution - let the agent work through it or hit real completion criteria
        
        # Get execution agent (second worker)
        execution_agent = self.migration_workers[1]
        
        # Get current messages
        current_messages = state.get("messages", [])
        
        # FIRST EXECUTION (analysis → execution transition): Apply phase transition pruning
        if total_loops == 1:
            analysis_done = state.get("analysis_done", False)
            if analysis_done:
                log_agent("[PRUNE] ✂ FIRST EXECUTION: Applying phase transition message pruning")
                
                try:
                    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
                    
                    # Log BEFORE state - what's being removed
                    original_message_count = len(current_messages)
                    log_agent(f"[PRUNE_DETAIL] BEFORE Phase Transition: {original_message_count} messages from analysis")
                    
                    # Count message types
                    msg_types = {}
                    tool_calls = []
                    for msg in current_messages:
                        msg_type = getattr(msg, 'type', type(msg).__name__)
                        msg_types[msg_type] = msg_types.get(msg_type, 0) + 1
                        if isinstance(msg, ToolMessage):
                            tool_name = getattr(msg, 'name', 'unknown')
                            tool_calls.append(tool_name)
                    
                    log_agent(f"[PRUNE_DETAIL] Message types: {dict(msg_types)}")
                    if tool_calls:
                        from collections import Counter
                        tool_counts = Counter(tool_calls)
                        top_tools = tool_counts.most_common(5)
                        log_agent(f"[PRUNE_DETAIL] Top tools called: {dict(top_tools)}")
                    
                    # Read current state files
                    todo_content = self._read_state_file("TODO.md", keep_beginning=True)
                    current_state_content = self._read_state_file("CURRENT_STATE.md")
                    completed_actions = self._read_state_file("COMPLETED_ACTIONS.md")
                    
                    # Get restricted view of tasks (only next 3)
                    visible_tasks = self._get_visible_tasks(todo_content, max_visible=3)
                    
                    # CREATE PHYSICAL FILE: VISIBLE_TASKS.md (only next 3 tasks)
                    # This prevents agent from bypassing filter with run_command
                    self._create_visible_tasks_file(visible_tasks)
                    
                    # Build task description with restricted view
                    if visible_tasks['all_done']:
                        task_content = "🎉 ALL TASKS COMPLETE! All migration tasks are done."
                    elif visible_tasks['current']:
                        task_content = f"""✅ CURRENT TASK (DO THIS NOW):
════════════════════════════════════════════════════════════════════════════════

{visible_tasks['current']}

════════════════════════════════════════════════════════════════════════════════
📋 UPCOMING TASKS (for reference - complete current first):
"""
                        for idx, task in enumerate(visible_tasks['upcoming'], 1):
                            task_content += f"   {idx}. {task}\n"
                        
                        hidden_count = visible_tasks['remaining_count'] - len(visible_tasks['upcoming']) - 1
                        if hidden_count > 0:
                            task_content += f"\n🔒 {hidden_count} additional tasks hidden (will be shown after completing current tasks)"
                    else:
                        task_content = "(No TODO file yet)"
                    
                    log_agent(f"[PHASE_TRANSITION] Created VISIBLE_TASKS.md with next 3 tasks: {visible_tasks['remaining_count']} remaining")
                    log_agent(f"[PHASE_TRANSITION] Agent can only see tasks in VISIBLE_TASKS.md, not full TODO.md")
                    
                    # Set flag to lock TODO.md from further analysis agent modifications
                    self._analysis_complete_flag = True
                    log_agent("[PHASE_TRANSITION] Set analysis_complete_flag=True - TODO.md is now locked")
                    
                    # Build clean execution context - ONLY HumanMessage
                    # DO NOT add SystemMessage here - it will be added by _create_prompt_with_trimming
                    # Adding SystemMessage here causes "non-consecutive system messages" error
                    # because this message ends up in the MIDDLE of history after trimming
                    from langchain_core.messages import HumanMessage

                    current_messages = [
                        HumanMessage(content=f"""EXECUTION PHASE START - Project: {self.project_path}

Analysis is complete. Your task list is in VISIBLE_TASKS.md.

WORKFLOW:
1. Read VISIBLE_TASKS.md to see the current task
2. Execute that task
3. Commit with git_commit
4. Task list auto-updates after commit
5. Repeat

Start now: read VISIBLE_TASKS.md""")
                    ]

                    pruned_count = original_message_count - len(current_messages)
                    log_agent(f"[PRUNE] ✂ Pruned {original_message_count} → {len(current_messages)} messages (removed {pruned_count})")
                    log_agent(f"[PRUNE_DETAIL] Clean execution context: 1 HumanMessage (system prompt added by prompt_builder)")
                    log_agent(f"[PRUNE_DETAIL] Agent will read VISIBLE_TASKS.md for tasks (file-based state)")
                    log_summary(f"MESSAGE PRUNING: Removed {pruned_count} analysis messages, created clean 1-message execution context")
                    log_agent(f"[PRUNE] ✅ Fresh execution context created")
                
                except Exception as e:
                    log_agent(f"[PRUNE] ❌ Error during message pruning: {str(e)}", "ERROR")
                    log_agent(f"[PRUNE] Falling back to original messages with phase-awareness injection")
                    # Fallback - append a HumanMessage (not system) to avoid API issues
                    phase_transition_msg = HumanMessage(content="""EXECUTION PHASE - Analysis is complete.

Read VISIBLE_TASKS.md and execute the CURRENT TASK.
Do NOT repeat analysis. Just execute tasks and commit.""")
                    current_messages = current_messages + [phase_transition_msg]
        
        # SUBSEQUENT EXECUTIONS: External memory is now handled by prompt builder
        # The prompt builder automatically injects COMPLETED_ACTIONS.md after trimming
        # so we don't need to manually manage it here anymore
        elif total_loops > 1:
            log_agent(f"[MEMORY] 🔄 EXECUTION LOOP #{total_loops}: External memory handled by prompt builder")
            log_agent(f"[MEMORY] COMPLETED_ACTIONS.md will be automatically injected after message trimming")
            log_summary(f"MEMORY: Execution loop #{total_loops} - external memory active via prompt builder")
        
        # Check if stuck intervention is needed
        stuck_intervention = state.get("stuck_intervention_active", False)
        loops_without_progress = state.get("loops_without_progress", 0)
        
        if stuck_intervention:
            # Inject intervention as HumanMessage (not system) to avoid API issues
            from langchain_core.messages import HumanMessage
            intervention_msg = HumanMessage(content=f"""STUCK ALERT: {loops_without_progress} loops without progress.

Read VISIBLE_TASKS.md NOW. Execute ONE task. Commit. Repeat.
Do not read TODO.md directly - only VISIBLE_TASKS.md.""")

            # Add intervention message to existing messages
            current_messages = current_messages + [intervention_msg]
            
            log_agent("[STUCK] Injecting intervention message to agent")
            log_summary(f"STUCK INTERVENTION: Informing agent of {loops_without_progress} loops without progress")
        
        # Run agent with potentially modified messages
        if current_messages != state.get("messages", []):
            # Messages were modified (phase transition, stuck intervention, or external memory)
            state_with_messages = dict(state)
            state_with_messages["messages"] = current_messages
            old_message_count = len(state.get("messages", []))
            result = execution_agent.invoke(state_with_messages)
        else:
            # Normal execution with original state
            result = execution_agent.invoke(state)
        
        # OLD: ALWAYS update completed actions after agent runs (track critical tool calls)
        # REMOVED: self._update_completed_actions(state.get("messages", []), result.get("messages", []))
        # NOW: Actions logged in real-time by _wrap_tool_with_tracking() during tool execution
        
        # Get current TODO progress
        todo_progress = self._calculate_todo_progress()
        current_todo_count = todo_progress['completed']
        last_todo_count = state.get("last_todo_count", 0)
        
        # Detect if progress was made
        if current_todo_count > last_todo_count:
            # Progress made!
            new_loops_without_progress = 0
            log_agent(f"[PROGRESS] ✅ TODO count increased: {last_todo_count} → {current_todo_count}")
            log_summary(f"PROGRESS: Completed {current_todo_count - last_todo_count} new TODO items")
        else:
            # No progress
            new_loops_without_progress = loops_without_progress + 1
            log_agent(f"[PROGRESS] ⚠ No progress - loop #{new_loops_without_progress} without TODO updates")
            
            if new_loops_without_progress >= MAX_LOOPS_WITHOUT_PROGRESS:
                log_agent(f"[STUCK] Agent stuck for {new_loops_without_progress} loops - intervention needed!", "WARNING")
                log_summary(f"STUCK DETECTED: {new_loops_without_progress} loops without progress")
        
        # Auto-detect completion using deterministic TODO check
        messages = result.get("messages", [])
        execution_complete, completion_reason, completion_stats = self._is_migration_complete()
        
        if execution_complete:
            log_agent(f"[WRAPPER] ✅ Execution COMPLETE: {completion_reason}")
            log_summary(f"EXECUTION PHASE: COMPLETED - {completion_reason}")
            log_console(f"✅ Migration complete: {completion_stats.get('completion_percentage', 0)}% of tasks done", "SUCCESS")
        else:
            # Log progress even if not complete
            if completion_stats.get('completed_tasks', 0) > 0:
                log_agent(f"[PROGRESS] {completion_reason}")
        
        # Detect build errors
        has_error, error_msg = self._detect_build_error(messages)
        if has_error:
            log_agent(f"[WRAPPER] ⚠ Build error detected in execution output")
            log_summary(f"BUILD ERROR: {error_msg[:100]}...")
        
        # Determine if intervention needed for next loop
        needs_intervention = (new_loops_without_progress >= MAX_LOOPS_WITHOUT_PROGRESS and
                             not execution_complete)
        
        # Update state with detection result + progress tracking + error detection
        # CRITICAL: Preserve analysis_done flag to prevent re-routing to analysis!
        return {
            "messages": messages,
            "analysis_done": state.get("analysis_done", False),  # Preserve from previous state!
            "execution_done": execution_complete,
            "current_phase": "EXECUTION_COMPLETE" if execution_complete else "EXECUTION",
            "last_todo_count": current_todo_count,
            "loops_without_progress": new_loops_without_progress,
            "total_execution_loops": total_loops,
            "stuck_intervention_active": needs_intervention,
            "has_build_error": has_error,
            "error_count": state.get("error_count", 0) + (1 if has_error else 0),
            "last_error_message": error_msg if has_error else ""
        }
    
    def _wrap_error_node(self, state: State):
        """Wrapper for error agent with error resolution tracking + external memory"""
        error_count = state.get("error_count", 0)
        last_error_msg = state.get("last_error_message", "")
        project_path = state.get("project_path", "")
        self.project_path = project_path  # Set for memory helpers
        
        log_agent(f"[WRAPPER] Running error_expert (attempt {error_count}/3)")
        log_summary(f"ERROR RESOLUTION: error_expert attempting to fix build errors (attempt {error_count}/3)")
        
        # Get error agent (third worker)
        error_agent = self.migration_workers[2]
        
        # Extract current error
        current_messages = state.get("messages", [])
        current_error = self._extract_latest_error(current_messages)
        
        # Read error history
        error_history = self._read_state_file("ERROR_HISTORY.md")
        
        # Check if exact same error already tried
        if self._error_already_attempted(current_error, error_history):
            log_agent("[ERROR] Same error signature attempted before - escalating", "WARNING")
            log_summary("ERROR: Same error attempted before - max retries reached")
            return {
                "messages": current_messages,
                "analysis_done": state.get("analysis_done", False),
                "execution_done": state.get("execution_done", False),
                "has_build_error": True,
                "error_count": 3,  # Force max count to exit
                "last_error_message": "Duplicate error - cannot resolve"
            }
        
        # Apply aggressive pruning for error expert
        log_agent(f"[MEMORY] ↘ ERROR EXPERT: Applying external memory system")
        
        try:
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
            
            # Log BEFORE state
            original_message_count = len(current_messages)
            log_agent(f"[PRUNE_DETAIL] BEFORE Error Expert: {original_message_count} messages from execution")
            
            # Count message types
            msg_types = {}
            tool_calls = []
            for msg in current_messages:
                msg_type = getattr(msg, 'type', type(msg).__name__)
                msg_types[msg_type] = msg_types.get(msg_type, 0) + 1
                if isinstance(msg, ToolMessage):
                    tool_name = getattr(msg, 'name', 'unknown')
                    tool_calls.append(tool_name)
            
            log_agent(f"[PRUNE_DETAIL] Message types: {dict(msg_types)}")
            if tool_calls:
                from collections import Counter
                tool_counts = Counter(tool_calls)
                top_tools = tool_counts.most_common(5)
                log_agent(f"[PRUNE_DETAIL] Recent tools called: {dict(top_tools)}")
            
            # Build clean error context - ONLY HumanMessage
            # DO NOT add SystemMessage here - it will be added by _create_prompt_with_trimming
            # Adding SystemMessage here causes "non-consecutive system messages" error

            # Build clean 1-message context with error details in HumanMessage
            current_messages = [
                HumanMessage(content=f"""ERROR FIX REQUIRED - Project: {project_path}

## CURRENT ERROR:
{current_error}

## PREVIOUS ATTEMPTS:
{error_history if error_history else 'No previous attempts - this is your first try.'}

Do NOT repeat failed approaches. Try something different.

Analyze the error, then EXECUTE the fix using your tools.
Run mvn_compile to verify it works.""")
            ]

            pruned_count = original_message_count - len(current_messages)
            log_agent(f"[MEMORY] ✂ Error expert: {original_message_count} → {len(current_messages)} messages")
            log_agent(f"[PRUNE_DETAIL] Clean error context: 1 HumanMessage (system prompt added by prompt_builder)")
            log_summary(f"MEMORY: Error expert using clean 1-message context")
        
        except Exception as e:
            log_agent(f"[MEMORY] ❌ Error applying external memory: {str(e)}", "ERROR")
            # Fallback: use original messages (less ideal but won't crash)
            pass

        # Run error agent with clean context
        state_with_context = dict(state)
        state_with_context["messages"] = current_messages
        result = error_agent.invoke(state_with_context)
        
        # Check if errors are resolved
        messages = result.get("messages", [])
        still_has_error, error_msg = self._detect_build_error(messages)
        
        # Log error attempt to history
        self._append_error_attempt(
            error=current_error,
            attempt_num=error_count,
            was_successful=not still_has_error
        )
        
        if still_has_error:
            log_agent("[WRAPPER] ⚠ Build error still present after error_expert")
            log_summary(f"ERROR PERSISTS: {error_msg[:100]}...")
        else:
            log_agent("[WRAPPER] ✅ Build error RESOLVED by error_expert")
            log_summary("ERROR RESOLVED: Build errors fixed, returning to execution")
        
        # Update state - reset error count if resolved
        # CRITICAL: Preserve analysis_done and execution_done flags!
        return {
            "messages": messages,
            "analysis_done": state.get("analysis_done", False),  # Preserve!
            "execution_done": state.get("execution_done", False),  # Preserve!
            "has_build_error": still_has_error,
            "error_count": state.get("error_count", 0) if still_has_error else 0,  # Reset count on success
            "last_error_message": error_msg if still_has_error else ""
        }
    
    def _create_supervisor(self):
        """Create supervisor workflow with deterministic routing"""
        from langgraph.graph import StateGraph, END
        
        # Build workflow with explicit routing
        workflow = StateGraph(State)
        
        # Add agent wrapper nodes (with auto-detection + error handling)
        workflow.add_node("analysis_expert", self._wrap_analysis_node)
        workflow.add_node("execution_expert", self._wrap_execution_node)
        workflow.add_node("error_expert", self._wrap_error_node)
        
        # Add conditional routing based on state
        workflow.add_conditional_edges(
            "analysis_expert",
            self._route_next_agent,
            {
                "analysis_expert": "analysis_expert",  # Loop if not done
                "execution_expert": "execution_expert",
                "error_expert": "error_expert",  # Route to error expert if errors detected
                "END": END,
                "FAILED": END  # Failure also ends workflow
            }
        )
        
        workflow.add_conditional_edges(
            "execution_expert",
            self._route_next_agent,
            {
                "analysis_expert": "analysis_expert",
                "execution_expert": "execution_expert",  # Loop if not done
                "error_expert": "error_expert",  # Route to error expert if errors detected
                "END": END,
                "FAILED": END  # Failure also ends workflow
            }
        )
        
        workflow.add_conditional_edges(
            "error_expert",
            self._route_next_agent,
            {
                "execution_expert": "execution_expert",  # Back to execution after fixing
                "error_expert": "error_expert",  # Retry if error persists
                "END": END,  # Success completion
                "FAILED": END  # Max error attempts reached
            }
        )
        
        # Set entry point
        workflow.set_entry_point("analysis_expert")
        
        log_agent("[SUPERVISOR] Custom workflow created with deterministic routing + error handling")
        
        return workflow
    
    def migrate_project_stream(self, project_path: str):
        """Start supervised migration with streaming progress updates"""
        print("="*80)
        print(f"STARTING SUPERVISED MIGRATION: {project_path}")
        print("="*80)
        
        if not os.path.exists(project_path):
            yield {"type": "complete", "success": False, "error": f"Project path does not exist: {project_path}"}
            return
        
        # Create and register state tracker for this migration
        self.state_tracker = MigrationStateTracker(project_path)
        self.project_path = project_path  # Set for external memory system
        set_state_tracker(self.state_tracker)
        log_agent(f"State tracker initialized for {project_path}")
        log_summary(f"STATE_TRACKER: Initialized for migration")
        
        # AUTO-CLEANUP: Delete ALL stale state files from previous runs
        log_agent("[CLEANUP] 🧹 Cleaning stale migration state files...")
        log_console("🧹 Cleaning previous migration state...", "INFO")
        
        state_files = [
            "COMPLETED_ACTIONS.md",
            "TODO.md",
            "CURRENT_STATE.md",
            "VISIBLE_TASKS.md",
            "analysis.md",
            "ERROR_HISTORY.md"
        ]
        
        cleaned_count = 0
        for state_file in state_files:
            file_path = os.path.join(project_path, state_file)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    log_agent(f"[CLEANUP] ▪ Deleted: {state_file}")
                    cleaned_count += 1
                except Exception as e:
                    log_agent(f"[CLEANUP] ⚠ Could not delete {state_file}: {e}", "WARNING")
        
        if cleaned_count > 0:
            log_console(f"✅ Deleted {cleaned_count} stale state files", "SUCCESS")
            log_agent(f"[CLEANUP] Removed {cleaned_count} files - starting with clean state")
        else:
            log_agent("[CLEANUP] No stale files found - clean start")
        
        # Initialize fresh state files
        completed_actions_path = os.path.join(project_path, "COMPLETED_ACTIONS.md")
        error_history_path = os.path.join(project_path, "ERROR_HISTORY.md")
        
        # Initialize COMPLETED_ACTIONS.md
        if not os.path.exists(completed_actions_path):
            with open(completed_actions_path, 'w') as f:
                f.write("")  # Empty file - real-time tracking will append entries
            log_agent(f"[MEMORY] ▪ Initialized COMPLETED_ACTIONS.md (empty, append-only)")
        
        # Initialize ERROR_HISTORY.md if needed
        if not os.path.exists(error_history_path):
            with open(error_history_path, 'w') as f:
                f.write("# Error Resolution History\n\n")
                f.write("This file tracks all build/test errors encountered and resolution attempts.\n\n")
            log_agent(f"[MEMORY] ▪ Initialized ERROR_HISTORY.md")
        
        # Create migration request using external template
        migration_request = get_migration_request(project_path)
        
        try:
            start_time = datetime.now()
            
            print(f"\nSupervisor: Starting migration workflow...")
            print(f"Supervisor: Available workers: {[agent.name for agent in self.migration_workers]}")
            print("-" * 60)
            
            # Initial progress message
            yield {
                "type": "progress",
                "step": 0,
                "message": f"Starting migration for {project_path}",
                "agent": "supervisor"
            }
            
            # Stream the workflow execution with progress tracking
            step_count = 0
            last_agent = None
            circuit_breaker_triggered = False
            last_chunk = None  # Track the final state from streaming
            
            for chunk in self.app.stream({
                "messages": [{"role": "user", "content": migration_request}],
                "project_path": project_path,
                "current_phase": "INIT",
                "analysis_done": False,
                "execution_done": False,
                "last_todo_count": 0,
                "loops_without_progress": 0,
                "total_execution_loops": 0,
                "stuck_intervention_active": False,
                "has_build_error": False,
                "error_count": 0,
                "last_error_message": ""
            }, {"recursion_limit": 500}):
                step_count += 1
                last_chunk = chunk  # Track last chunk for final state
                
                # Circuit breaker: Check LLM call limit before processing each step
                try:
                    self._check_llm_call_limit()
                except LLMCallLimitExceeded as e:
                    log_agent(f"Circuit breaker triggered at step {step_count}")
                    log_summary(f"Migration stopped at step {step_count} due to LLM call limit")
                    circuit_breaker_triggered = True
                    yield {
                        "type": "progress",
                        "step": step_count,
                        "message": f"⚠ LLM call limit reached ({tc.llm_calls}/{MAX_LLM_CALLS})",
                        "agent": "supervisor",
                        "event_type": "limit_exceeded"
                    }
                    # Exit the loop gracefully
                    break
                
                self._log_workflow_step(step_count, chunk)
                
                # Extract detailed progress information (can be multiple events per chunk)
                progress_events = self._extract_progress_info(chunk, step_count)
                
                # Yield each progress event
                for progress_event in progress_events:
                    current_agent = progress_event.get("agent", "supervisor")
                    
                    # Track agent transitions
                    if current_agent != last_agent:
                        yield {
                            "type": "progress",
                            "step": step_count,
                            "message": f"Calling {current_agent.replace('_', ' ').title()}",
                            "agent": current_agent,
                            "event_type": "agent_transition"
                        }
                        last_agent = current_agent
                    
                    # Yield the detailed progress event
                    yield progress_event
            
            duration = datetime.now() - start_time
            
            print("\n" + "="*80)
            print(f"SUPERVISED MIGRATION COMPLETED in {duration.total_seconds():.2f} seconds")
            print("="*80)
            
            # Extract final state from the last chunk (NO RE-INVOCATION!)
            # The stream already contains all the information we need
            if last_chunk:
                # Extract the final state from the last chunk
                for node_name, node_state in last_chunk.items():
                    if isinstance(node_state, dict) and "messages" in node_state:
                        final_result = node_state
                        break
                else:
                    # Fallback if we can't find messages in chunk structure
                    final_result = {"messages": [{"role": "assistant", "content": "Migration completed"}]}
            else:
                # No chunks received (unlikely)
                final_result = {"messages": [{"role": "assistant", "content": "Migration halted"}]}
            
            # Extract final result
            messages = final_result.get("messages", [])
            final_message = messages[-1] if messages else {}
            
            # Handle different message types
            if hasattr(final_message, 'content'):
                final_content = final_message.content
            elif isinstance(final_message, dict):
                final_content = final_message.get("content", str(final_message))
            else:
                final_content = str(final_message)
            
            # Generate token usage report
            print("\n" + "="*80)
            print("GENERATING TOKEN USAGE REPORT")
            print("="*80)
            token_stats = tc.get_stats()
            
            # Log to summary file
            log_summary("\n" + "="*60)
            log_summary("TOKEN USAGE & COST REPORT (Supervisor + All Agents)")
            log_summary("="*60)
            log_summary(f"LLM Calls:       {token_stats['llm_calls']:,}")
            log_summary(f"Prompt tokens:   {token_stats['prompt_tokens']:,}")
            log_summary(f"Response tokens: {token_stats['response_tokens']:,}")
            log_summary(f"Total tokens:    {token_stats['total_tokens']:,}")
            log_summary("-"*60)
            log_summary(f"Prompt cost:     ${token_stats['prompt_cost_usd']:.4f}")
            log_summary(f"Response cost:   ${token_stats['response_cost_usd']:.4f}")
            log_summary(f"TOTAL COST:      ${token_stats['total_cost_usd']:.4f}")
            log_summary("="*60)
            
            # Also log to agent log for detailed tracking
            log_agent("TOKEN USAGE SUMMARY")
            log_agent(f"Token usage: {token_stats['prompt_tokens']:,} prompt + {token_stats['response_tokens']:,} response = {token_stats['total_tokens']:,} total")
            log_agent(f"LLM calls: {token_stats['llm_calls']:,}")
            log_agent(f"Total cost: ${token_stats['total_cost_usd']:.4f}")
            
            # Print to console for user visibility
            print(f"\nMigration Cost: ${token_stats['total_cost_usd']:.4f} ({token_stats['total_tokens']:,} tokens)")
            print(f"LLM Calls: {token_stats['llm_calls']:,}")
            
            # Check if migration was stopped due to LLM call limit
            limit_exceeded = tc.llm_calls >= MAX_LLM_CALLS
            
            yield {
                "type": "complete",
                "success": True if not limit_exceeded else False,
                "result": final_content,
                "duration": duration.total_seconds(),
                "messages": len(messages),
                "steps": step_count,
                "project_path": project_path,
                "token_stats": token_stats,
                "limit_exceeded": limit_exceeded,
                "warning": f"LLM call limit reached ({tc.llm_calls}/{MAX_LLM_CALLS})" if limit_exceeded else None
            }
        
        except LLMCallLimitExceeded as e:
            # Graceful shutdown when LLM call limit is exceeded
            duration = datetime.now() - start_time
            token_stats = tc.get_stats()
            
            log_summary(f"MIGRATION HALTED: {str(e)}")
            log_console(f"Migration stopped gracefully: {str(e)}", "WARNING")
            
            # Log token usage stats even when circuit breaker triggers
            log_summary("\n" + "="*60)
            log_summary("TOKEN USAGE & COST REPORT (Circuit Breaker Triggered)")
            log_summary("="*60)
            log_summary(f"LLM Calls:       {token_stats['llm_calls']:,}")
            log_summary(f"Prompt tokens:   {token_stats['prompt_tokens']:,}")
            log_summary(f"Response tokens: {token_stats['response_tokens']:,}")
            log_summary(f"Total tokens:    {token_stats['total_tokens']:,}")
            log_summary("-"*60)
            log_summary(f"Prompt cost:     ${token_stats['prompt_cost_usd']:.4f}")
            log_summary(f"Response cost:   ${token_stats['response_cost_usd']:.4f}")
            log_summary(f"TOTAL COST:      ${token_stats['total_cost_usd']:.4f}")
            log_summary("="*60)
            
            log_console(f"\nMigration Cost: ${token_stats['total_cost_usd']:.4f} ({token_stats['total_tokens']:,} tokens)")
            log_agent(f"Token usage: {token_stats['prompt_tokens']:,} prompt + {token_stats['response_tokens']:,} response = {token_stats['total_tokens']:,} total")
            log_agent(f"Total cost: ${token_stats['total_cost_usd']:.4f}")
            
            yield {
                "type": "complete",
                "success": False,
                "error": str(e),
                "project_path": project_path,
                "duration": duration.total_seconds(),
                "steps": step_count,
                "token_stats": token_stats,
                "limit_exceeded": True
            }
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\nSUPERVISOR ERROR: {str(e)}")
            yield {
                "type": "complete",
                "success": False,
                "error": str(e),
                "project_path": project_path
            }

    def _is_ai_message(self, msg) -> bool:
        """Check if message is from AI/LLM"""
        if isinstance(msg, dict):
            return msg.get("type") in ["ai", "assistant"] or "ai" in str(msg.get("type", "")).lower()
        return hasattr(msg, "type") and (msg.type in ["ai", "assistant"] or "ai" in str(msg.type).lower())
    
    def _is_tool_message(self, msg) -> bool:
        """Check if message is a tool result"""
        if isinstance(msg, dict):
            return msg.get("type") in ["tool", "function"]
        return hasattr(msg, "type") and msg.type in ["tool", "function"]
    
    def _get_message_content(self, msg):
        """Extract content from message"""
        if isinstance(msg, dict):
            return msg.get("content", "")
        return getattr(msg, "content", "")
    
    def _get_message_name(self, msg):
        """Extract name/tool name from message"""
        if isinstance(msg, dict):
            return msg.get("name", "")
        return getattr(msg, "name", "")
    
    def _get_tool_calls(self, msg):
        """Extract tool calls from message"""
        if isinstance(msg, dict):
            return msg.get("tool_calls", [])
        return getattr(msg, "tool_calls", [])
    
    def _get_tool_name(self, tool_call):
        """Extract tool name from tool call"""
        if isinstance(tool_call, dict):
            return tool_call.get("name", "unknown_tool")
        return getattr(tool_call, "name", "unknown_tool")
    
    def _get_tool_args(self, tool_call):
        """Extract tool arguments from tool call"""
        if isinstance(tool_call, dict):
            # Try multiple possible keys for arguments
            args = tool_call.get("args")
            if args is None:
                args = tool_call.get("arguments")
            if args is None:
                args = tool_call.get("parameters")
            return args or {}
        else:
            # For object-style tool calls, try multiple attributes
            for attr in ["args", "arguments", "parameters"]:
                args = getattr(tool_call, attr, None)
                if args is not None:
                    return args
            return {}
    
    def _extract_tool_info(self, msg, progress_info: Dict[str, Any]) -> bool:
        """Extract tool usage information from messages"""
        try:
            # Handle different message types
            tool_calls = None
            if isinstance(msg, dict):
                tool_calls = msg.get("tool_calls", [])
            elif hasattr(msg, "tool_calls"):
                tool_calls = msg.tool_calls
            
            if tool_calls:
                for tool_call in tool_calls:
                    tool_name = None
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get("name", "")
                    elif hasattr(tool_call, "name"):
                        tool_name = tool_call.name
                    
                    if tool_name:
                        progress_info["message"] = f"{progress_info['agent'].replace('_', ' ').title()} using {tool_name}"
                        progress_info["tool"] = tool_name
                        return True
        
        except Exception as e:
            print(f"Error extracting tool info: {e}")
        
        return False
    
    def _extract_progress_info(self, chunk: Dict[str, Any], step_count: int) -> List[Dict[str, Any]]:
        """Extract detailed progress information from workflow chunks"""
        if not chunk:
            return []
        
        progress_events = []
        
        # Check for active agent and extract detailed information
        for node_name, node_data in chunk.items():
            if node_name in ["analysis_expert", "execution_expert", "error_expert", "supervisor"]:
                
                # Look for messages in this node
                if isinstance(node_data, dict) and "messages" in node_data:
                    messages = node_data["messages"]
                    
                    # Process each message for detailed info
                    for i, msg in enumerate(messages):
                        # Extract LLM calls and responses
                        if self._is_ai_message(msg):
                            content = self._get_message_content(msg)
                            if content and content.strip():
                                progress_events.append({
                                    "type": "progress",
                                    "step": step_count,
                                    "agent": node_name,
                                    "event_type": "llm_response",
                                    "message": "LLM Response",
                                    "content": content,
                                    "details": {"message_index": i, "node": node_name}
                                })
                        
                        # Extract tool calls
                        tool_calls = self._get_tool_calls(msg)
                        if tool_calls:
                            for j, tool_call in enumerate(tool_calls):
                                tool_name = self._get_tool_name(tool_call)
                                tool_args = self._get_tool_args(tool_call)
                                
                                # Debug the actual tool_call structure
                                print(f"DEBUG TOOL CALL: {tool_call}")
                                print(f"DEBUG TOOL ARGS: {tool_args}")
                                
                                progress_events.append({
                                    "type": "progress",
                                    "step": step_count,
                                    "agent": node_name,
                                    "event_type": "tool_call",
                                    "message": f"Tool Call: {tool_name}",
                                    "content": f"**Tool:** {tool_name}\n**Parameters:** {json.dumps(tool_args, indent=2) if tool_args else 'No parameters'}",
                                    "details": {"tool": tool_name, "args": tool_args, "call_index": j}
                                })
                        
                        # Extract tool results
                        if self._is_tool_message(msg):
                            tool_name = self._get_message_name(msg)
                            tool_result = self._get_message_content(msg)
                            
                            progress_events.append({
                                "type": "progress",
                                "step": step_count,
                                "agent": node_name,
                                "event_type": "tool_result",
                                "message": f"Tool Result: {tool_name}",
                                "content": f"**Tool:** {tool_name}\n**Result:** {tool_result[:500]}{'...' if len(str(tool_result)) > 500 else ''}",
                                "details": {"tool": tool_name, "result_preview": str(tool_result)[:100]}
                            })
        
        # If no detailed messages, show basic agent activity
        if not progress_events:
            progress_events.append({
                "type": "progress",
                "step": step_count,
                "agent": node_name,
                "event_type": "agent_activity",
                "message": f"{node_name.replace('_', ' ').title()} activated",
                "content": f"Agent {node_name} is now processing...",
                "details": {"node": node_name}
            })
        
        return progress_events
    
    def migrate_project(self, project_path: str) -> Dict[str, Any]:
        # AUTO-CLEANUP: Delete ALL stale state files from previous runs
        log_agent("[CLEANUP] 🧹 Cleaning stale migration state files...")
        log_console("🧹 Cleaning previous migration state...", "INFO")
        
        state_files = [
            "COMPLETED_ACTIONS.md",
            "TODO.md",
            "CURRENT_STATE.md",
            "VISIBLE_TASKS.md",
            "analysis.md",
            "ERROR_HISTORY.md"
        ]
        
        cleaned_count = 0
        for state_file in state_files:
            file_path = os.path.join(project_path, state_file)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    log_agent(f"[CLEANUP] ▪ Deleted: {state_file}")
                    cleaned_count += 1
                except Exception as e:
                    log_agent(f"[CLEANUP] ⚠ Could not delete {state_file}: {e}", "WARNING")
        
        if cleaned_count > 0:
            log_console(f"✅ Deleted {cleaned_count} stale state files", "SUCCESS")
            log_agent(f"[CLEANUP] Removed {cleaned_count} files - starting with clean state")
        else:
            log_agent("[CLEANUP] No stale files found - clean start")
        
        # Create and register state tracker for this migration
        self.state_tracker = MigrationStateTracker(project_path)
        self.project_path = project_path  # Set for external memory system
        set_state_tracker(self.state_tracker)
        log_agent(f"State tracker initialized for {project_path}")
        log_summary(f"STATE_TRACKER: Initialized for migration")
        
        # Initialize COMPLETED_ACTIONS.md (empty, system will append to it)
        completed_actions_path = os.path.join(project_path, "COMPLETED_ACTIONS.md")
        if not os.path.exists(completed_actions_path):
            with open(completed_actions_path, 'w') as f:
                f.write("")  # Empty file - real-time tracking will append entries
            log_agent(f"[MEMORY] ▪ Initialized COMPLETED_ACTIONS.md (empty, append-only)")
        
        # Create migration request using external template
        migration_request = get_migration_request(project_path)
        
        try:
            start_time = datetime.now()
            
            log_agent("Invoking supervisor graph")
            log_agent(f"Available workers: {[agent.name for agent in self.migration_workers]}")
            log_summary("PHASE: Agent coordination started")
            log_summary(f"Available workers: {[agent.name for agent in self.migration_workers]}")
            
            # Stream the workflow execution with progress tracking
            step_count = 0
            circuit_breaker_triggered = False
            last_chunk = None  # Track the final state from streaming
            for chunk in self.app.stream({
                "messages": [{"role": "user", "content": migration_request}],
                "project_path": project_path,
                "current_phase": "INIT",
                "analysis_done": False,
                "execution_done": False,
                "last_todo_count": 0,
                "loops_without_progress": 0,
                "total_execution_loops": 0,
                "stuck_intervention_active": False,
                "has_build_error": False,
                "error_count": 0,
                "last_error_message": ""
            }, {"recursion_limit": 500}):
                step_count += 1
                last_chunk = chunk  # Track last chunk for final state
                
                # Circuit breaker: Check LLM call limit before processing each step
                try:
                    self._check_llm_call_limit()
                except LLMCallLimitExceeded as e:
                    log_agent(f"Circuit breaker triggered at step {step_count}")
                    log_summary(f"Migration stopped at step {step_count} due to LLM call limit")
                    circuit_breaker_triggered = True
                    # Exit the loop gracefully
                    break
                
                self._log_workflow_step(step_count, chunk)
            
            # Extract final state from the last chunk (NO RE-INVOCATION!)
            # The stream already contains all the information we need
            if last_chunk:
                # Extract the final state from the last chunk
                for node_name, node_state in last_chunk.items():
                    if isinstance(node_state, dict) and "messages" in node_state:
                        final_result = node_state
                        break
                else:
                    # Fallback if we can't find messages in chunk structure
                    final_result = {"messages": [{"role": "assistant", "content": "Migration completed"}]}
            else:
                # No chunks received (unlikely)
                final_result = {"messages": [{"role": "assistant", "content": "Migration halted"}]}
            
            duration = datetime.now() - start_time
            
            # Check if migration actually succeeded (all tasks complete) or failed
            execution_done = final_result.get("execution_done", False)
            has_build_error = final_result.get("has_build_error", False)
            error_count = final_result.get("error_count", 0)
            
            # Determine true success/failure
            migration_success = execution_done and not (has_build_error and error_count >= 3)
            
            if migration_success:
                log_agent("Migration process completed successfully - all tasks complete")
                log_summary(f"PHASE: Agent coordination completed in {duration.total_seconds():.2f} seconds - SUCCESS")
            else:
                if error_count >= 3:
                    log_agent("Migration process FAILED - max error attempts reached", "ERROR")
                    log_summary(f"PHASE: Agent coordination ended in {duration.total_seconds():.2f} seconds - FAILED (errors)")
                else:
                    log_agent("Migration process ended - not all tasks complete", "WARNING")
                    log_summary(f"PHASE: Agent coordination ended in {duration.total_seconds():.2f} seconds - INCOMPLETE")
            
            # Extract final result
            messages = final_result.get("messages", [])
            final_message = messages[-1] if messages else {}
            final_content = final_message.get("content", "No final message") if isinstance(final_message, dict) else str(final_message)
            
            # Generate comprehensive token usage report
            log_summary("\n" + "="*60)
            log_summary("TOKEN USAGE & COST REPORT (Supervisor + All Agents)")
            log_summary("="*60)
            token_stats = tc.get_stats()
            log_summary(f"LLM Calls:       {token_stats['llm_calls']:,}")
            log_summary(f"Prompt tokens:   {token_stats['prompt_tokens']:,}")
            log_summary(f"Response tokens: {token_stats['response_tokens']:,}")
            log_summary(f"Total tokens:    {token_stats['total_tokens']:,}")
            log_summary("-"*60)
            log_summary(f"Prompt cost:     ${token_stats['prompt_cost_usd']:.4f}")
            log_summary(f"Response cost:   ${token_stats['response_cost_usd']:.4f}")
            log_summary(f"TOTAL COST:      ${token_stats['total_cost_usd']:.4f}")
            log_summary("="*60)
            
            # Also log to console and agent log
            log_console(f"\nMigration Cost: ${token_stats['total_cost_usd']:.4f} ({token_stats['total_tokens']:,} tokens)")
            log_agent(f"Token usage: {token_stats['prompt_tokens']:,} prompt + {token_stats['response_tokens']:,} response = {token_stats['total_tokens']:,} total")
            log_agent(f"Total cost: ${token_stats['total_cost_usd']:.4f}")
            
            # Check if migration was stopped due to LLM call limit
            limit_exceeded = tc.llm_calls >= MAX_LLM_CALLS
            if limit_exceeded:
                log_summary(f"WARNING: Migration incomplete - LLM call limit reached ({tc.llm_calls}/{MAX_LLM_CALLS})")
                log_console(f"⚠ Migration stopped: LLM call limit reached ({tc.llm_calls}/{MAX_LLM_CALLS})", "WARNING")
            
            # Determine actual success (must complete all tasks AND not fail due to errors)
            actual_success = migration_success and not limit_exceeded
            
            return {
                "success": actual_success,
                "result": final_content,
                "duration": duration.total_seconds(),
                "messages": len(messages),
                "steps": step_count,
                "token_stats": token_stats,
                "limit_exceeded": limit_exceeded,
                "execution_done": execution_done,
                "error_count": error_count,
                "warning": f"LLM call limit reached ({tc.llm_calls}/{MAX_LLM_CALLS})" if limit_exceeded else None
            }
        
        except LLMCallLimitExceeded as e:
            # Graceful shutdown when LLM call limit is exceeded
            duration = datetime.now() - start_time
            token_stats = tc.get_stats()
            
            log_agent(f"Migration halted: {str(e)}")
            log_summary(f"MIGRATION HALTED: {str(e)}")
            log_console(f"Migration stopped gracefully: {str(e)}", "WARNING")
            
            # Log token usage stats even when circuit breaker triggers
            log_summary("\n" + "="*60)
            log_summary("TOKEN USAGE & COST REPORT (Circuit Breaker Triggered)")
            log_summary("="*60)
            log_summary(f"LLM Calls:       {token_stats['llm_calls']:,}")
            log_summary(f"Prompt tokens:   {token_stats['prompt_tokens']:,}")
            log_summary(f"Response tokens: {token_stats['response_tokens']:,}")
            log_summary(f"Total tokens:    {token_stats['total_tokens']:,}")
            log_summary("-"*60)
            log_summary(f"Prompt cost:     ${token_stats['prompt_cost_usd']:.4f}")
            log_summary(f"Response cost:   ${token_stats['response_cost_usd']:.4f}")
            log_summary(f"TOTAL COST:      ${token_stats['total_cost_usd']:.4f}")
            log_summary("="*60)
            
            log_console(f"\nMigration Cost: ${token_stats['total_cost_usd']:.4f} ({token_stats['total_tokens']:,} tokens)")
            log_agent(f"Token usage: {token_stats['prompt_tokens']:,} prompt + {token_stats['response_tokens']:,} response = {token_stats['total_tokens']:,} total")
            log_agent(f"Total cost: ${token_stats['total_cost_usd']:.4f}")
            
            return {
                "success": False,
                "error": str(e),
                "duration": duration.total_seconds(),
                "steps": step_count,
                "token_stats": token_stats,
                "limit_exceeded": True
            }
        
        except Exception as e:
            log_agent(f"Migration failed with exception: SUPERVISOR ERROR: {e}")
            log_summary(f"ERROR: SUPERVISOR ERROR: - {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _log_workflow_step(self, step_count: int, chunk: Dict[str, Any]):
        """Log detailed information about each workflow step"""
        # print(f"\n[STEP {step_count}] " + "="*50)
        log_agent(f"Workflow Step {step_count} Data:")
        
        if not chunk:
            # print("Empty chunk received")
            log_agent("Empty chunk received", "WARNING")
            return
        
        for node_name, node_data in chunk.items():
            # print(f"NODE: {node_name}")
            log_agent(f"Node: {node_name}")
            
            # Check if this is a worker agent being called
            if node_name in ["analysis_expert", "execution_expert", "error_expert"]:
                # print(f"  -> Calling {node_name.replace('_', ' ').title()}")
                log_summary(f"Calling {node_name.replace('_', ' ').title()}")
            
            # Show messages if available
            if isinstance(node_data, dict) and "messages" in node_data:
                messages = node_data["messages"]
                # print(f"  Messages: {len(messages)}")
                log_agent(f"Processing {len(messages)} messages")
                
                # Show all messages to capture LLM responses
                for msg in messages:
                    self._display_detailed_message(msg)
            
            # Show other data if not messages
            elif node_data and str(node_data) != "{}":
                data_preview = str(node_data)[:150] + ("..." if len(str(node_data)) > 150 else "")
                log_agent(f"Node data: {data_preview}", "DEBUG")
        
        log_agent(f"Workflow Step {step_count} END")
    
    def _display_detailed_message(self, msg):
        """Display detailed information about a message including full LLM responses"""
        
        # Extract message details
        if isinstance(msg, dict):
            msg_content = msg.get("content", "")
            msg_type = msg.get("type", "unknown")
            msg_name = msg.get("name", "")
            tool_calls = msg.get("tool_calls", [])
        else:
            msg_content = getattr(msg, "content", "")
            msg_type = getattr(msg, "type", "unknown")
            msg_name = getattr(msg, "name", "")
            tool_calls = getattr(msg, "tool_calls", [])
        
        # Log message header
        if msg_name:
            log_agent(f"Message [{msg_type.upper()}] from {msg_name}")
        else:
            log_agent(f"Message [{msg_type.upper()}]")
        
        # Log LLM response content - use llm log type for AI messages
        if msg_content and (msg_type in ["ai", "assistant"] or "ai" in str(msg_type).lower()):
            log_llm("LLM RESPONSE START")
            # Split content into lines for better logging
            content_lines = str(msg_content).split('\n')
            for line in content_lines:
                if line.strip():  # Only log non-empty lines
                    log_llm(line, "DEBUG")
            log_llm("LLM RESPONSE END")
            
            # Also log summary of LLM response for high-level tracking
            if len(str(msg_content)) > 100:
                log_summary(f"LLM Response ({len(str(msg_content))} chars): {str(msg_content)[:100]}...")
            else:
                log_summary(f"LLM Response: {str(msg_content)}")
        
        elif msg_content and msg_type not in ["tool", "function"]:
            log_agent(f"CONTENT ({msg_type}) START")
            content_lines = str(msg_content).split('\n')
            for line in content_lines[:20]:  # Show first 20 lines
                if line.strip():
                    log_agent(line, "DEBUG")
            if len(content_lines) > 20:
                log_agent(f"... ({len(content_lines) - 20} more lines)", "DEBUG")
            log_agent(f"CONTENT ({msg_type}) END")
        
        elif msg_content:
            # Abbreviated for tool messages
            if len(str(msg_content)) > 200:
                log_agent(f"Content: {str(msg_content)[:200]}...", "DEBUG")
            else:
                log_agent(f"Content: {str(msg_content)}", "DEBUG")
        
        # Log detailed tool calls to agent log, summary to summary log
        if tool_calls:
            log_agent(f"TOOL CALLS ({len(tool_calls)}) START")
            
            for i, tool_call in enumerate(tool_calls, 1):
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get("name", "unknown")
                    tool_args = tool_call.get("args", {})
                    tool_id = tool_call.get("id", "")
                else:
                    tool_name = getattr(tool_call, "name", "unknown")
                    tool_args = getattr(tool_call, "args", {})
                    tool_id = getattr(tool_call, "id", "")
                
                log_agent(f"Tool Call {i}: {tool_name}")
                log_summary(f"TOOL: {tool_name}")  # High-level tool usage tracking
                
                if tool_id:
                    log_agent(f"Tool ID: {tool_id}", "DEBUG")
                if tool_args:
                    log_agent(f"Tool Args: {tool_args}", "DEBUG")
            
            log_agent("TOOL CALLS END")


## Test code below - run directly for a specific repo
# if __name__ == "__main__":
#     project_path = "/Users/xfmk897/Library/CloudStorage/OneDrive-BNY/Desktop/deepwiki-experiment/migration/repositories/apache__sling-org-apache-sling-resourceresolver"
#     
#     # Ensure TODO.md and CURRENT_STATE.md exist in the project path
#     for fname in ["TODO.md", "CURRENT_STATE.md"]:
#         fpath = os.path.join(project_path, fname)
#         if not os.path.exists(fpath):
#             with open(fpath, "w") as f:
#                 f.write(f"# {fname}\n\n")
#     
#     # Checkout to a new branch migration-java-spring
#     try:
#         subprocess.run(
#             ["git", "-C", project_path, "checkout", "-B", "migration-java-spring"],
#             check=True,
#             capture_output=True,
#             text=True
#         )
#         print("Checked out to new branch: migration-java-spring")
#     except subprocess.CalledProcessError as e:
#         print(f"Failed to checkout branch: {e.stderr}")
#     
#     orchestrator = SupervisorMigrationOrchestrator()
#     result = orchestrator.migrate_project(project_path)
#     
#     print(f"\nSupervised migration completed.")
#     print(f"Success: {result.get('success')}")
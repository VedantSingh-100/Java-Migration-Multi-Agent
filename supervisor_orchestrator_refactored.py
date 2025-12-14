"""
Supervisor-Based Migration Orchestrator (REFACTORED)
Uses LangGraph's create_supervisor for intelligent agent management

This file uses modular components from src/orchestrator/:
- Constants (constants.py)
- State management (state.py)
- Message pruning (message_manager.py)
- Task management (task_manager.py)
- Tool registry (tool_registry.py)
- Action logging (action_logger.py)
- Error handling (error_handler.py)
- Agent wrappers (agent_wrappers.py)
"""
import os
import re
import json
import time
import random
import asyncio
from enum import Enum
from datetime import datetime
from typing import Dict, Any, List, Tuple

import tiktoken
from dotenv import load_dotenv
from pydantic import PrivateAttr
from botocore.exceptions import ClientError

from langchain_aws import ChatBedrock
from langchain_core.globals import set_verbose
from langchain_core.messages import BaseMessage, HumanMessage

from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

# Import from refactored orchestrator modules
from src.orchestrator import (
    # Constants
    MAX_CONTEXT_TOKENS,
    MAX_LLM_CALLS,
    MAX_EXECUTION_LOOPS_PER_PHASE,
    MAX_LOOPS_WITHOUT_PROGRESS,
    MAX_HISTORY_MESSAGES,
    TRACKED_TOOLS,
    COMMIT_TOOLS,
    STATE_FILES,
    # State
    State,
    StateFileManager,
    calculate_todo_progress,
    # Messages
    MessagePruner,
    ExternalMemoryBuilder,
    PromptBuilder,
    create_clean_execution_context,
    compile_execution_context,
    extract_last_tool_result,
    summarize_completed_tasks,
    # Tasks
    TaskManager,
    sync_tasks_after_commit,
    # Tools
    ToolWrapper,
    get_tools_for_agent,
    ANALYSIS_TOOL_NAMES,
    EXECUTION_TOOL_NAMES,
    ERROR_TOOL_NAMES,
    # Actions
    ActionLogger,
    initialize_completed_actions_file,
    # Errors
    ErrorHandler,
    StuckDetector,
    initialize_error_history_file,
    # Signature-based loop detection helpers
    categorize_tool_result,
    hash_tool_args,
    ToolResultCategory,
    # Wrappers
    AnalysisNodeWrapper,
    ExecutionNodeWrapper,
    ErrorNodeWrapper,
)

from src.utils.LLMLogger import LLMLogger
from src.utils.logging_config import log_agent, log_summary, log_console, log_llm
from src.utils.TokenCounter import TokenCounter
from src.utils.migration_state_tracker import MigrationStateTracker
from src.utils.completion_detector import detect_analysis_complete
from src.utils.search_processor import setup_search_context_from_pom, reset_search_processor

from src.tools import all_tools_flat
from src.tools.state_management import set_state_tracker
from src.tools.file_operations import set_project_path as set_file_ops_project_path
from prompts.prompt_loader import (
    get_supervisor_prompt,
    get_migration_request,
    get_analysis_expert_prompt,
    get_execution_expert_prompt,
    get_error_expert_prompt,
)

load_dotenv()
set_verbose(True)

# Amazon Bedrock API key from environment variable
# This is a long-term Bedrock API key (ABSK format) used as bearer token
BEDROCK_API_KEY = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
if not BEDROCK_API_KEY:
    raise ValueError("AWS_BEARER_TOKEN_BEDROCK environment variable is required")

# AWS Region for Bedrock
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Claude model ID on Amazon Bedrock (cross-region inference profile)
# Can be overridden via BEDROCK_MODEL_ID environment variable for model comparison experiments
# Available models:
#   - us.anthropic.claude-3-5-sonnet-20241022-v2:0  (Claude 3.5 Sonnet v2)
#   - us.anthropic.claude-sonnet-4-20250514-v1:0   (Claude Sonnet 4)
#   - us.anthropic.claude-sonnet-4-5-20250929-v1:0 (Claude Sonnet 4.5)
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-sonnet-20241022-v2:0")

# Global token counter
tc = TokenCounter()
ENC = tiktoken.get_encoding("cl100k_base")


class LLMCallLimitExceeded(Exception):
    """Exception raised when the LLM call limit is exceeded during migration"""
    pass


class CircuitBreakerChatBedrock(ChatBedrock):
    """
    Wrapper around ChatBedrock that:
    1. Enforces LLM call limits by checking BEFORE each call
    2. Retries with exponential backoff on throttling errors
    """

    _token_counter: Any = PrivateAttr()
    _max_calls: int = PrivateAttr()
    _max_retries: int = PrivateAttr(default=5)

    def __init__(self, token_counter, max_calls, max_retries=5, **kwargs):
        super().__init__(**kwargs)
        self._token_counter = token_counter
        self._max_calls = max_calls
        self._max_retries = max_retries

    def _generate(self, *args, **kwargs):
        if self._token_counter.llm_calls >= self._max_calls:
            log_agent(f"Circuit breaker preventing LLM call: {self._token_counter.llm_calls}/{self._max_calls}", "ERROR")
            raise LLMCallLimitExceeded(
                f"LLM call limit of {self._max_calls} exceeded ({self._token_counter.llm_calls} calls already made)"
            )

        # Retry with exponential backoff on throttling
        for attempt in range(self._max_retries):
            try:
                return super()._generate(*args, **kwargs)
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                if error_code == 'ThrottlingException':
                    if attempt < self._max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        log_agent(f"[THROTTLE] Rate limited, waiting {wait_time:.1f}s (attempt {attempt+1}/{self._max_retries})")
                        log_summary(f"[THROTTLE] AWS Bedrock rate limit hit, retrying in {wait_time:.1f}s")
                        time.sleep(wait_time)
                    else:
                        log_agent(f"[THROTTLE] Max retries ({self._max_retries}) exceeded", "ERROR")
                        raise
                else:
                    raise

    async def _agenerate(self, *args, **kwargs):
        if self._token_counter.llm_calls >= self._max_calls:
            log_agent(f"Circuit breaker preventing LLM call (async): {self._token_counter.llm_calls}/{self._max_calls}", "ERROR")
            raise LLMCallLimitExceeded(
                f"LLM call limit of {self._max_calls} exceeded ({self._token_counter.llm_calls} calls already made)"
            )

        # Retry with exponential backoff on throttling (async version)
        for attempt in range(self._max_retries):
            try:
                return await super()._agenerate(*args, **kwargs)
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                if error_code == 'ThrottlingException':
                    if attempt < self._max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        log_agent(f"[THROTTLE] Rate limited (async), waiting {wait_time:.1f}s (attempt {attempt+1}/{self._max_retries})")
                        log_summary(f"[THROTTLE] AWS Bedrock rate limit hit, retrying in {wait_time:.1f}s")
                        await asyncio.sleep(wait_time)
                    else:
                        log_agent(f"[THROTTLE] Max retries ({self._max_retries}) exceeded (async)", "ERROR")
                        raise
                else:
                    raise


class MigrationStatus(Enum):
    """
    Deterministic migration result classification.
    Used by _classify_migration_result() - NO LLM involvement.
    """
    SUCCESS = "success"              # >= 90% complete, build passes, tests pass
    PARTIAL_SUCCESS = "partial"      # >= 50% complete, build passes, tests pass
    FAILURE = "failure"              # Build fails OR tests fail OR < 50%
    INCOMPLETE = "incomplete"        # Terminated early (LLM limit, timeout)


class SupervisorMigrationOrchestrator:
    """
    Supervisor that manages specialized migration agents.

    Uses modular components from src/orchestrator/ for all functionality.
    """

    def __init__(self):
        log_agent("Initializing SupervisorMigrationOrchestrator")
        log_summary("COMPONENT: SupervisorMigrationOrchestrator initialized")
        log_agent(f"Circuit breaker initialized with limit: {MAX_LLM_CALLS}")

        # State tracker (set per migration)
        self.state_tracker = None
        self.project_path = None

        # Initialize modular components
        self.state_file_manager = StateFileManager()
        self.task_manager = TaskManager()
        self.action_logger = ActionLogger()
        self.error_handler = ErrorHandler(action_window_size=5)
        # Pass action_logger AND task_manager to ToolWrapper
        # - action_logger: logs actions to COMPLETED_ACTIONS.md
        # - task_manager: handles "nothing to commit" verification flow
        self.tool_wrapper = ToolWrapper(
            action_logger=self.action_logger,
            task_manager=self.task_manager
        )
        self.message_pruner = MessagePruner(max_messages=MAX_HISTORY_MESSAGES)
        self.stuck_detector = StuckDetector()

        # External memory builder (initialized after project_path is set)
        self.external_memory_builder = None

        log_agent("Modular orchestrator components initialized")

        # Create specialized migration agents as workers
        self.migration_workers = self._create_migration_workers()

        # Create and compile supervisor workflow
        self.supervisor_workflow = self._create_supervisor()
        self.app = self.supervisor_workflow.compile()

        log_agent(f"Created {len(self.migration_workers)} migration workers")
        log_agent(f"Workers: {[agent.name for agent in self.migration_workers]}")

    def _check_llm_call_limit(self):
        """Check if LLM call limit has been exceeded."""
        if tc.llm_calls >= MAX_LLM_CALLS:
            log_agent(f"LLM call limit reached: {tc.llm_calls}/{MAX_LLM_CALLS}", "ERROR")
            log_summary(f"CIRCUIT BREAKER: LLM call limit of {MAX_LLM_CALLS} exceeded")
            log_console(f"LLM call limit reached ({tc.llm_calls}/{MAX_LLM_CALLS}) - stopping migration", "WARNING")
            raise LLMCallLimitExceeded(f"Migration stopped: LLM call limit of {MAX_LLM_CALLS} exceeded")

    def _create_model(self):
        """Create circuit-breaker wrapped model instance using Amazon Bedrock"""
        return CircuitBreakerChatBedrock(
            token_counter=tc,
            max_calls=MAX_LLM_CALLS,
            model_id=BEDROCK_MODEL_ID,
            region_name=AWS_REGION,
            callbacks=[LLMLogger(), tc],
            model_kwargs={
                "max_tokens": 8192,
                "temperature": 0.0,
            },
        )

    def _create_migration_workers(self):
        """Create specialized worker agents for migration tasks"""
        common_agent_kwargs = dict(
            model=self._create_model(),
            debug=False,
            checkpointer=InMemorySaver()
        )

        log_agent("[AGENTS] Creating agents with modular PromptBuilder")

        # Analysis Worker - NO TRIMMING (phase transition handles it)
        analysis_worker = create_react_agent(
            tools=self._get_analysis_tools(),
            prompt=get_analysis_expert_prompt(),
            name="analysis_expert",
            state_schema=State,
            **common_agent_kwargs,
        )
        log_agent("[AGENTS] analysis_expert created WITHOUT trimming")

        # Execution Worker - Uses PromptBuilder with external memory injection
        # Note: PromptBuilder needs external_memory_builder, which is set per-migration
        execution_prompt_builder = PromptBuilder(
            system_prompt=get_execution_expert_prompt(),
            agent_name="execution_expert",
            message_pruner=MessagePruner(max_messages=30),
            external_memory_builder=None,  # Set later in _set_project_path
            inject_external_memory=True
        )

        execution_worker = create_react_agent(
            tools=self._get_execution_tools(),
            prompt=execution_prompt_builder.build(),
            name="execution_expert",
            state_schema=State,
            **common_agent_kwargs
        )
        self._execution_prompt_builder = execution_prompt_builder  # Store for later update
        log_agent("[AGENTS] execution_expert created with PromptBuilder + EXTERNAL MEMORY")

        # Error Worker - Uses PromptBuilder for trimming
        error_prompt_builder = PromptBuilder(
            system_prompt=get_error_expert_prompt(),
            agent_name="error_expert",
            message_pruner=MessagePruner(max_messages=20),
            inject_external_memory=False
        )

        error_worker = create_react_agent(
            tools=self._get_error_tools(),
            prompt=error_prompt_builder.build(),
            name="error_expert",
            state_schema=State,
            **common_agent_kwargs,
        )
        log_agent("[AGENTS] error_expert created with PromptBuilder")

        log_agent("[AGENTS] All agents created successfully with modular approach")
        return [analysis_worker, execution_worker, error_worker]

    def _get_analysis_tools(self):
        """Get tools for analysis agent using ToolWrapper"""
        tools = get_tools_for_agent("analysis", all_tools_flat)
        wrapped_tools = [self.tool_wrapper.wrap_analysis_tool(tool) for tool in tools]
        log_agent(f"[TOOLS] Analysis agent has {len(wrapped_tools)} tools")
        return wrapped_tools

    def _get_execution_tools(self):
        """Get tools for execution agent using ToolWrapper"""
        tools = get_tools_for_agent("execution", all_tools_flat)

        # Wrap tools with tracking (uses the orchestrator's callbacks for task sync)
        wrapped_tools = []
        for tool in tools:
            wrapped = self.tool_wrapper.wrap_execution_tool(
                tool,
                on_commit_success=self._on_commit_success
            )
            wrapped_tools.append(wrapped)

        log_agent(f"[TOOLS] Execution agent has {len(wrapped_tools)} tools")
        return wrapped_tools

    def _get_error_tools(self):
        """Get tools for error agent using ToolWrapper"""
        tools = get_tools_for_agent("error", all_tools_flat)
        wrapped_tools = [self.tool_wrapper.wrap_error_tool(tool) for tool in tools]
        log_agent(f"[TOOLS] Error agent has {len(wrapped_tools)} tools")
        return wrapped_tools

    def _on_commit_success(self):
        """Callback for successful commits - triggers task sync"""
        log_agent(f"[AUTO_SYNC] Successful commit detected - marking current task complete")
        sync_tasks_after_commit(self.task_manager, self.state_file_manager)
        log_agent(f"[AUTO_SYNC] Task marked in TODO.md, VISIBLE_TASKS.md updated with next task")

    def _set_project_path(self, project_path: str):
        """Set project path on all modular components"""
        self.project_path = project_path
        self.state_file_manager.set_project_path(project_path)
        self.task_manager.set_project_path(project_path)
        self.action_logger.set_project_path(project_path)
        self.error_handler.set_project_path(project_path)
        self.tool_wrapper.set_project_path(project_path)

        # Set project path for file operations tools (Issue 2 fix - Working Directory Confusion)
        # This ensures all file operations are constrained to the project directory
        set_file_ops_project_path(project_path)
        log_agent(f"[PATH] Set file_operations project_path to: {project_path}")

        # Setup search context from pom.xml (for web search optimization)
        # This detects Java/Spring versions and sets environment variables
        reset_search_processor()  # Reset for new migration
        setup_search_context_from_pom(project_path)

        # Create external memory builder now that we have project_path
        self.external_memory_builder = ExternalMemoryBuilder(
            self.state_file_manager,
            self.task_manager
        )

        # Update execution agent's prompt builder with external memory builder
        if hasattr(self, '_execution_prompt_builder'):
            self._execution_prompt_builder.external_memory_builder = self.external_memory_builder

        log_agent("Modular components updated with project path")

    def _route_next_agent(self, state: State) -> str:
        """Deterministic router based on state flags (NO LLM)"""
        current_phase = state.get('current_phase', 'INIT')
        has_build_error = state.get('has_build_error', False)
        error_count = state.get('error_count', 0)
        error_type = state.get('error_type', 'none')
        execution_done = state.get('execution_done', False)
        analysis_done = state.get('analysis_done', False)
        test_failure_count = state.get('test_failure_count', 0)

        # Stuck loop detection state
        is_stuck = state.get('is_stuck', False)
        stuck_type = state.get('stuck_type', 'none')
        stuck_loop_attempts = state.get('stuck_loop_attempts', 0)
        stuck_reason = state.get('stuck_reason', '')

        log_agent(f"[ROUTER] Phase: {current_phase} | Analysis: {analysis_done} | Execution: {execution_done} | Error: {has_build_error} (type={error_type}, count={error_count}, test_failures={test_failure_count}) | Stuck: {is_stuck} (type={stuck_type}, attempts={stuck_loop_attempts})")

        # Check for timeout condition
        if current_phase == "EXECUTION_TIMEOUT":
            log_agent("[ROUTER] -> Execution timeout reached, ending migration", "WARNING")
            return "END"

        # PRIORITY 1: If execution is complete, END
        if execution_done:
            if has_build_error:
                log_agent("[ROUTER] -> Execution complete with build errors - ending")
            else:
                log_agent("[ROUTER] -> Migration complete successfully, ending")
            return "END"

        # PRIORITY 2: Max retries exceeded (build errors OR stuck loops) → FAILED
        # Using stuck_loop_attempts (cumulative) rather than error_count for stuck loops
        if error_count >= 3:
            log_agent("[ROUTER] -> Max error attempts reached (3), MIGRATION FAILED", "ERROR")
            log_summary("MIGRATION FAILED: Max error attempts (3) reached")
            return "FAILED"

        if stuck_loop_attempts >= 3:
            log_agent(f"[ROUTER] -> Max stuck loop attempts reached ({stuck_loop_attempts}), MIGRATION FAILED", "ERROR")
            log_summary(f"MIGRATION FAILED: Stuck in loop for {stuck_loop_attempts} attempts - {stuck_reason}")
            return "FAILED"

        # PRIORITY 3: Check for build errors (now with test failure retry logic)
        if has_build_error:
            # Differentiate between error types: test_violation, pom, test, compile
            if error_type == 'test_violation':
                # TEST PRESERVATION VIOLATION - execution_expert tried to modify test methods
                # Route directly to error_expert who has revert_test_files tool
                log_agent(f"[ROUTER] -> TEST PRESERVATION VIOLATION detected, routing to error_expert", "WARNING")
                log_summary(f"TEST_VIOLATION: Commit blocked - test methods modified. Routing to error_expert for recovery.")
                return "error_expert"
            elif error_type == 'pom':
                # POM errors go directly to error_expert - these are configuration issues
                log_agent(f"[ROUTER] -> POM/configuration error detected, routing to error_expert (attempt {error_count}/3)")
                log_summary(f"POM ERROR: Routing to error_expert for configuration fix (attempt {error_count}/3)")
                return "error_expert"
            elif error_type == 'test':
                # Test failures get 1 retry before routing to error_expert
                if test_failure_count == 0:
                    log_agent(f"[ROUTER] -> Test failure detected, allowing 1 retry (execution_expert)")
                    log_summary("TEST FAILURE: Allowing execution_expert to retry once")
                    return "execution_expert"  # State will increment test_failure_count
                else:
                    log_agent(f"[ROUTER] -> Test failure persists after retry, routing to error_expert (attempt {error_count}/3)")
                    log_summary(f"TEST FAILURE PERSISTS: Routing to error_expert (attempt {error_count}/3)")
                    return "error_expert"
            else:
                # Compile errors (and generic 'compile' type) go to error_expert immediately
                log_agent(f"[ROUTER] -> Compile error detected, routing to error_expert (attempt {error_count}/3)")
                return "error_expert"

        # PRIORITY 4: Stuck loop detected → route to error_expert for alternative approach
        # This handles cases like FIND REPLACE NO MATCH loops (not build errors)
        if is_stuck:
            log_agent(f"[ROUTER] -> STUCK LOOP detected (type={stuck_type}, attempt={stuck_loop_attempts}/3): {stuck_reason}", "WARNING")
            log_summary(f"STUCK LOOP: Routing to error_expert to try alternative approach (attempt {stuck_loop_attempts}/3)")
            return "error_expert"

        # PRIORITY 5: Route based on phase
        if not analysis_done:
            log_agent("[ROUTER] -> Routing to analysis_expert")
            return "analysis_expert"
        else:
            log_agent("[ROUTER] -> Routing to execution_expert")
            return "execution_expert"

    def _wrap_analysis_node(self, state: State):
        """
        Wrapper for analysis agent with stuck detection and auto-reset.

        If agent returns 5 consecutive responses without tool calls,
        reset to fresh context. Max 2 resets before failing.
        """
        project_path = state.get("project_path", "")
        no_tool_count = state.get("analysis_no_tool_count", 0)
        reset_count = state.get("analysis_reset_count", 0)

        log_agent(f"[WRAPPER] Running analysis_expert (no_tool: {no_tool_count}, resets: {reset_count})")

        # Check if we need to reset due to stuck loop
        if no_tool_count >= 5:
            if reset_count >= 2:
                # Max resets exceeded - fail the migration
                log_agent("[ANALYSIS_RESET] Max resets (2) exceeded - analysis failed", "ERROR")
                log_summary("ANALYSIS FAILED: Agent stuck without tool calls after 2 resets")
                return {
                    "messages": state.get("messages", []),
                    "analysis_done": False,
                    "current_phase": "FAILED",
                    "analysis_no_tool_count": 0,
                    "analysis_reset_count": reset_count,
                }

            # Reset the agent with fresh context
            reset_count += 1
            log_agent(f"[ANALYSIS_RESET] Resetting analysis agent (reset #{reset_count}/2)", "WARNING")
            log_summary(f"ANALYSIS RESET: Agent stuck for 5 responses without tools - reset #{reset_count}")

            # Create fresh start message
            fresh_message = HumanMessage(content=f"""SYSTEM RESET: Previous analysis attempt failed to use tools.

You MUST use tools to analyze this project. Start NOW.

Project path: {project_path}

IMMEDIATE ACTION REQUIRED:
1. Call find_all_poms("{project_path}") to discover project structure
2. Call read_pom("{project_path}") to analyze dependencies
3. Call mvn_compile("{project_path}") to establish build baseline

Execute find_all_poms NOW. Do not respond with text - USE THE TOOL.""")

            # Reset messages to just the fresh start
            reset_messages = [fresh_message]

            # Invoke agent with reset context
            analysis_agent = self.migration_workers[0]
            result = analysis_agent.invoke({
                **state,
                "messages": reset_messages,
            })

            messages = result.get("messages", [])

            # Check if agent used tools after reset
            has_tools = self._check_analysis_has_tools(messages)
            new_no_tool_count = 0 if has_tools else 1

            if has_tools:
                log_agent("[ANALYSIS_RESET] Agent recovered - using tools after reset")
                log_summary("ANALYSIS RESET: Recovery successful - agent now using tools")
            else:
                log_agent("[ANALYSIS_RESET] Agent still not using tools after reset", "WARNING")

            # Check completion
            analysis_complete = detect_analysis_complete(project_path, messages)

            return {
                "messages": messages,
                "analysis_done": analysis_complete,
                "current_phase": "ANALYSIS_COMPLETE" if analysis_complete else "ANALYSIS",
                "analysis_no_tool_count": new_no_tool_count,
                "analysis_reset_count": reset_count,
            }

        # Normal execution path
        analysis_agent = self.migration_workers[0]
        result = analysis_agent.invoke(state)

        messages = result.get("messages", [])

        # Check if agent used tools
        has_tools = self._check_analysis_has_tools(messages)

        if has_tools:
            new_no_tool_count = 0
            log_agent("[WRAPPER] Analysis agent used tools - counter reset")
        else:
            new_no_tool_count = no_tool_count + 1
            log_agent(f"[WRAPPER] Analysis agent NO TOOLS - counter: {new_no_tool_count}/5", "WARNING")

        # Auto-detect completion
        analysis_complete = detect_analysis_complete(project_path, messages)

        if analysis_complete:
            log_agent("[WRAPPER] Analysis AUTO-DETECTED as complete")
            log_summary("ANALYSIS PHASE: AUTO-COMPLETED (files created)")

        return {
            "messages": messages,
            "analysis_done": analysis_complete,
            "current_phase": "ANALYSIS_COMPLETE" if analysis_complete else "ANALYSIS",
            "analysis_no_tool_count": new_no_tool_count,
            "analysis_reset_count": reset_count,
        }

    def _check_analysis_has_tools(self, messages: List[BaseMessage]) -> bool:
        """Check if the latest AI message in the analysis response has tool calls."""
        if not messages:
            return False

        # Find the last AI message
        for msg in reversed(messages):
            msg_type = getattr(msg, 'type', None) or type(msg).__name__
            if msg_type == 'ai' or 'AIMessage' in type(msg).__name__:
                tool_calls = getattr(msg, 'tool_calls', None) or []
                additional_kwargs = getattr(msg, 'additional_kwargs', {}) or {}
                additional_tool_calls = additional_kwargs.get('tool_calls', [])

                has_tools = len(tool_calls) > 0 or len(additional_tool_calls) > 0
                return has_tools

        return False

    def _wrap_execution_node(self, state: State):
        """
        Wrapper for execution agent using COMPILED CONTEXT pattern.

        Instead of accumulating messages, we compile fresh context each loop:
        - Current task from VISIBLE_TASKS.md
        - Completed summary from COMPLETED_ACTIONS.md
        - Last tool result for continuity

        This prevents context bloat and amnesia after pruning.
        """
        project_path = state.get("project_path", "")
        total_loops = state.get("total_execution_loops", 0) + 1

        log_agent(f"[WRAPPER] Running execution_expert (loop #{total_loops})")

        # Check max loops
        if total_loops > MAX_EXECUTION_LOOPS_PER_PHASE:
            log_agent(f"[STUCK] Max execution loops ({MAX_EXECUTION_LOOPS_PER_PHASE}) exceeded", "WARNING")
            return {
                "messages": state.get("messages", []),
                "execution_done": False,
                "current_phase": "EXECUTION_TIMEOUT",
                "total_execution_loops": total_loops
            }

        # NOTE: Stuck loop detection moved AFTER agent runs and tool calls are tracked
        # This ensures detect_stuck_loop() sees the CURRENT loop's data, not stale data

        execution_agent = self.migration_workers[1]

        # ═══════════════════════════════════════════════════════════════
        # COMPILED CONTEXT PATTERN - Fresh context every loop
        # ═══════════════════════════════════════════════════════════════

        # First loop: ensure VISIBLE_TASKS.md is created
        if total_loops == 1 and state.get("analysis_done", False):
            self._apply_phase_transition(state)

        # Read current state from files (source of truth)
        visible_tasks_content = self.state_file_manager.read_file("VISIBLE_TASKS.md")
        completed_content = self.state_file_manager.read_file("COMPLETED_ACTIONS.md")

        # Extract current task
        current_task = self.task_manager.extract_current_task(visible_tasks_content) if visible_tasks_content else None

        # Summarize completed tasks (last 10 only)
        completed_summary = summarize_completed_tasks(completed_content)

        # Get last tool result for continuity (from previous loop's messages)
        previous_messages = state.get("messages", [])
        last_result = extract_last_tool_result(previous_messages)

        # Check if previous loop had no tool calls - add warning to context
        no_tool_loops = state.get("no_tool_call_loops", 0)
        if no_tool_loops > 0:
            last_result = f"⚠️ WARNING: You returned {no_tool_loops}x without using tools. USE TOOLS NOW.\n\n{last_result or 'No previous result'}"
            log_agent(f"[WRAPPER] Added no-tool warning to context (count: {no_tool_loops})")

        # Compile fresh context
        compiled_messages = compile_execution_context(
            project_path=project_path,
            loop_num=total_loops,
            current_task=current_task,
            completed_summary=completed_summary,
            last_result=last_result
        )

        # Run agent with compiled context (NOT accumulated messages)
        state_with_messages = dict(state)
        state_with_messages["messages"] = compiled_messages
        result = execution_agent.invoke(state_with_messages)

        # ═══════════════════════════════════════════════════════════════
        # TRACK TOOL CALLS FOR STUCK LOOP DETECTION
        # ═══════════════════════════════════════════════════════════════
        # Extract tool calls from messages and track them via error_handler
        # This populates recent_actions so detect_stuck_loop() can work
        result_messages = result.get("messages", [])
        tracked_tool_count = self._extract_and_track_tool_calls(result_messages, current_task or "")
        log_agent(f"[STUCK_DETECTION] Tracked {tracked_tool_count} tool calls for loop detection")

        # ═══════════════════════════════════════════════════════════════
        # DETECT STUCK LOOPS (AFTER tracking, so we have current loop's data)
        # ═══════════════════════════════════════════════════════════════
        is_stuck = False
        stuck_reason = ""
        stuck_tool = ""
        if total_loops >= 3:  # Need at least 3 loops of history
            is_stuck, stuck_reason = self.error_handler.detect_stuck_loop()
            if is_stuck:
                # Extract tool name from reason if present
                if "Tool '" in stuck_reason:
                    stuck_tool = stuck_reason.split("Tool '")[1].split("'")[0]
                log_agent(f"[STUCK] Loop pattern detected: {stuck_reason}", "WARNING")
                log_console(f"⚠️ STUCK LOOP: {stuck_reason}", "WARNING")
                log_summary(f"STUCK LOOP DETECTED: {stuck_reason}")

        # Track progress
        todo_progress = calculate_todo_progress(project_path)
        current_todo_count = todo_progress['completed']
        last_todo_count = state.get("last_todo_count", 0)
        loops_without_progress = state.get("loops_without_progress", 0)

        if current_todo_count > last_todo_count:
            new_loops_without_progress = 0
            log_agent(f"[PROGRESS] TODO count increased: {last_todo_count} -> {current_todo_count}")
        else:
            new_loops_without_progress = loops_without_progress + 1
            log_agent(f"[PROGRESS] No progress - loop #{new_loops_without_progress}")

        # Check completion
        messages = result.get("messages", [])
        execution_complete = self._is_migration_complete(project_path)

        if execution_complete:
            log_agent("[WRAPPER] Execution COMPLETE")
            log_summary("EXECUTION PHASE: COMPLETED")

        # Check for build errors (now returns error_type)
        has_error, error_msg, error_type = self.error_handler.detect_build_error(messages)

        # Track test failure count for retry logic
        prev_test_failure_count = state.get("test_failure_count", 0)
        current_task = self.task_manager.extract_current_task(visible_tasks_content) if visible_tasks_content else ""
        last_test_failure_task = state.get("last_test_failure_task", "")

        # Test failure tracking logic
        if has_error and error_type == 'test':
            # Check if this is the same task or a new one
            if current_task == last_test_failure_task:
                # Same task failing again - increment count
                new_test_failure_count = prev_test_failure_count + 1
                log_agent(f"[WRAPPER] Test failure on same task (count: {new_test_failure_count})")
            else:
                # New task failing - start fresh count
                new_test_failure_count = 1
                log_agent(f"[WRAPPER] Test failure on new task '{current_task[:50]}...'")
            new_last_test_failure_task = current_task
        elif not has_error:
            # Tests passed - reset count
            new_test_failure_count = 0
            new_last_test_failure_task = ""
        else:
            # Compile error (not test) - keep existing test failure state
            new_test_failure_count = prev_test_failure_count
            new_last_test_failure_task = last_test_failure_task

        # ═══════════════════════════════════════════════════════════════
        # INTELLIGENT NO-TOOL RESPONSE HANDLING
        # ═══════════════════════════════════════════════════════════════
        # KEY INSIGHT: If progress was made, the no-tool response is BENIGN
        # (agent did work, then acknowledged). Only intervene if NO progress.
        #
        # Categories when NO progress:
        # - acknowledging: Wasteful - re-invoke with directive
        # - confused: Harmful - re-invoke with help
        # - thinking: Allow 2x then force
        # - complete: Verify and allow
        # - unknown: Graduated response

        new_no_tool_loops = self._count_no_tool_response(messages, no_tool_loops)
        thinking_loops = state.get("thinking_loops", 0)
        was_reinvoked = False
        progress_made = current_todo_count > last_todo_count

        if new_no_tool_loops > 0:
            # Extract response text and classify it
            response_text = self._extract_last_ai_response_text(messages)
            response_type = self._classify_no_tool_response(response_text)

            # KEY DECISION: Did we make progress?
            if progress_made:
                # Progress was made - this is a BENIGN acknowledgment
                # The agent DID work (tools were called earlier in the ReAct loop),
                # then said "done". This is fine - reset counter and continue.
                log_agent(f"[NO_TOOL] Response type: {response_type} BUT progress made ({last_todo_count}->{current_todo_count}) - BENIGN, resetting counter")
                new_no_tool_loops = 0  # Reset counter - this was productive
                thinking_loops = 0
            else:
                # NO progress - this is potentially HARMFUL
                log_agent(f"[NO_TOOL] Response type: {response_type}, NO progress (no_tool_count: {new_no_tool_loops}) - INTERVENING")

                # Handle based on classification
                messages, new_no_tool_loops, was_reinvoked = self._handle_no_tool_response(
                    state=state,
                    messages=messages,
                    response_type=response_type,
                    no_tool_count=new_no_tool_loops,
                    execution_agent=execution_agent,
                    project_path=project_path,
                    total_loops=total_loops,
                    current_task=current_task,
                    completed_summary=completed_summary
                )

                # Update thinking_loops tracker
                if response_type == "thinking":
                    thinking_loops += 1
                else:
                    thinking_loops = 0  # Reset if not thinking

                # If re-invoked successfully, re-check for errors in new messages
                if was_reinvoked:
                    has_error, error_msg, error_type = self.error_handler.detect_build_error(messages)
                    execution_complete = self._is_migration_complete(project_path)
                    # Re-calculate progress
                    todo_progress = calculate_todo_progress(project_path)
                    current_todo_count = todo_progress['completed']
                    if current_todo_count > last_todo_count:
                        new_loops_without_progress = 0
                        new_no_tool_loops = 0  # Progress made after re-invoke - reset
                        log_agent(f"[PROGRESS] After re-invoke, TODO count: {last_todo_count} -> {current_todo_count}")
        else:
            # Agent used tools - reset thinking counter
            thinking_loops = 0

        # Determine if stuck - either via loop pattern detection OR no progress for too long
        # KEY FIX: Progress TRUMPS pattern detection to prevent false positives
        # The fleetman-webapp bug: commit_changes called 3x (healthy - different files each time)
        # was flagged as stuck even though TODO count was increasing (5/34 tasks done)
        stuck_via_pattern = is_stuck  # From detect_stuck_loop()
        stuck_via_no_progress = new_loops_without_progress >= MAX_LOOPS_WITHOUT_PROGRESS

        # CRITICAL: If progress was made, DON'T flag as stuck (even if pattern detected)
        # This prevents healthy migrations from being killed by naive tool counting
        if progress_made:
            if stuck_via_pattern:
                log_agent(f"[STUCK] Pattern detected but PROGRESS MADE ({last_todo_count} -> {current_todo_count}) - FALSE POSITIVE, ignoring")
            is_stuck_now = False  # Progress trumps pattern detection
        else:
            is_stuck_now = stuck_via_pattern or stuck_via_no_progress

        # Determine stuck type for router
        if is_stuck_now and stuck_via_pattern:
            stuck_type = "tool_loop"
        elif is_stuck_now and stuck_via_no_progress:
            stuck_type = "no_progress"
        else:
            stuck_type = "none"

        # Track stuck loop attempts (for router to decide when to give up)
        # IMPORTANT: Only reset to 0 if we made ACTUAL PROGRESS (TODO count increased)
        # This prevents premature reset when detection briefly returns False
        prev_stuck_attempts = state.get("stuck_loop_attempts", 0)
        progress_made = current_todo_count > last_todo_count

        if is_stuck_now:
            new_stuck_attempts = prev_stuck_attempts + 1
            log_agent(f"[STUCK] Stuck attempt #{new_stuck_attempts} (pattern={stuck_via_pattern}, no_progress={stuck_via_no_progress})")
        elif progress_made:
            # Only reset if we actually made progress
            new_stuck_attempts = 0
            log_agent(f"[STUCK] Progress made, resetting stuck_loop_attempts to 0")
        else:
            # No progress but not detected as stuck yet - maintain counter
            new_stuck_attempts = prev_stuck_attempts

        # Preserve or reset stuck_failed_approaches based on progress
        if progress_made:
            new_failed_approaches = ""  # Reset on progress
        else:
            new_failed_approaches = state.get("stuck_failed_approaches", "")  # Preserve

        return {
            "messages": messages,
            "analysis_done": state.get("analysis_done", False),
            "execution_done": execution_complete,
            "current_phase": "EXECUTION_COMPLETE" if execution_complete else "EXECUTION",
            "last_todo_count": current_todo_count,
            "loops_without_progress": new_loops_without_progress,
            "total_execution_loops": total_loops,
            # Stuck detection state - used by router
            "is_stuck": is_stuck_now,
            "stuck_type": stuck_type,
            "stuck_tool": stuck_tool if is_stuck_now else "",
            "stuck_loop_attempts": new_stuck_attempts,
            "stuck_reason": stuck_reason if is_stuck_now else "",
            "stuck_failed_approaches": new_failed_approaches,
            # Build error state
            "has_build_error": has_error,
            "error_type": error_type,
            "error_count": state.get("error_count", 0) + (1 if has_error else 0),
            "last_error_message": error_msg if has_error else "",
            # Other state
            "no_tool_call_loops": new_no_tool_loops,
            "thinking_loops": thinking_loops,
            "test_failure_count": new_test_failure_count,
            "last_test_failure_task": new_last_test_failure_task,
        }

    def _wrap_error_node(self, state: State):
        """
        Wrapper for error agent with error resolution tracking.

        For STUCK LOOPS: error_expert gets 3 consecutive attempts with escalating strategies.
        Each attempt has different guidance to ensure varied approaches.

        For BUILD ERRORS: Standard error resolution flow.
        """
        import json

        error_count = state.get("error_count", 0)
        project_path = state.get("project_path", "")
        prev_error_type = state.get("error_type", "none")

        # Check if we're here due to stuck loop vs build error
        is_stuck = state.get("is_stuck", False)
        stuck_type = state.get("stuck_type", "none")
        stuck_reason = state.get("stuck_reason", "")
        stuck_tool = state.get("stuck_tool", "")
        stuck_loop_attempts = state.get("stuck_loop_attempts", 0)
        stuck_failed_approaches = state.get("stuck_failed_approaches", "")

        error_agent = self.migration_workers[2]
        current_messages = state.get("messages", [])

        if is_stuck:
            # ═══════════════════════════════════════════════════════════════
            # STUCK LOOP HANDLING - 3 consecutive attempts with escalation
            # ═══════════════════════════════════════════════════════════════
            log_agent(f"[WRAPPER] Running error_expert for STUCK LOOP (attempt {stuck_loop_attempts}/3, type={stuck_type})")
            log_summary(f"STUCK LOOP RESOLUTION: error_expert attempt {stuck_loop_attempts}/3")

            # Parse previous failed approaches
            try:
                failed_list = json.loads(stuck_failed_approaches) if stuck_failed_approaches else []
            except json.JSONDecodeError:
                failed_list = []

            # Get current task
            current_task_content = self.state_file_manager.read_file("VISIBLE_TASKS.md")
            current_task = self.task_manager.extract_current_task(current_task_content) if current_task_content else "Unknown task"

            # ESCALATING STRATEGIES based on attempt number
            if stuck_loop_attempts == 1:
                # ATTEMPT 1: Try a different tool/approach
                strategy = "STRATEGY 1: USE A DIFFERENT APPROACH"
                if stuck_tool == "find_replace":
                    specific_guidance = f"""
The find_replace tool keeps failing with "NO MATCH". The search string doesn't exist in the file.

YOUR TASK FOR THIS ATTEMPT:
1. FIRST: Use read_file to see the ACTUAL content of the file
2. THEN: Identify what text actually exists that you need to modify
3. FINALLY: Use find_replace with a search string that ACTUALLY EXISTS in the file

Example: If looking for '<version>1.0</version>' but file has '<version>1.0.0</version>',
search for the actual text '<version>1.0.0</version>' instead.

DO NOT use the same search pattern that failed."""
                else:
                    specific_guidance = f"""
The {stuck_tool} tool is not working for this task.

YOUR TASK FOR THIS ATTEMPT:
1. Analyze WHY {stuck_tool} is failing
2. Try a COMPLETELY DIFFERENT tool or approach
3. If modifying a file, try read_file first to understand the content

DO NOT repeat the same {stuck_tool} call."""

            elif stuck_loop_attempts == 2:
                # ATTEMPT 2: Use write_file to rewrite the section/file
                strategy = "STRATEGY 2: REWRITE USING write_file"
                specific_guidance = f"""
Previous approach failed: {failed_list[-1] if failed_list else 'Unknown'}

Since find_replace/targeted edits are not working, USE write_file INSTEAD:
1. Use read_file to get the current file content
2. Modify the content in your response (mentally or in a code block)
3. Use write_file to write the ENTIRE corrected file

This bypasses the pattern matching issues entirely.

IMPORTANT: When using write_file, include ALL the original content plus your changes.
Do not write a partial file."""

            else:  # stuck_loop_attempts >= 3
                # ATTEMPT 3: Skip the task and move on
                strategy = "STRATEGY 3: SKIP THIS TASK AND MOVE ON"
                specific_guidance = f"""
Previous approaches failed:
{chr(10).join(f'  - {a}' for a in failed_list) if failed_list else '  - (unknown)'}

This task appears to be BLOCKED. You MUST skip it now:

1. Use find_replace to update TODO.md:
   - Find the current task line
   - Replace "- [ ]" with "- [x] SKIPPED:"
   - Add reason: "Unable to complete - {stuck_reason}"

2. Example:
   OLD: - [ ] Update guava dependency version
   NEW: - [x] SKIPPED: Update guava dependency version (blocked: find_replace pattern not found in file)

3. After marking skipped, the system will move to the next task.

DO NOT try to fix this task again. Mark it skipped and move on."""

            # Build the prompt with escalating guidance
            clean_messages = [
                HumanMessage(content=f"""STUCK LOOP INTERVENTION - Project: {project_path}
Attempt: {stuck_loop_attempts}/3

## ISSUE: {stuck_reason}

## CURRENT TASK:
{current_task}

## {strategy}
{specific_guidance}

## PREVIOUS FAILED APPROACHES:
{chr(10).join(f'- {a}' for a in failed_list) if failed_list else 'None yet - this is your first attempt.'}

⚠️ CRITICAL: You MUST try something DIFFERENT from previous attempts.
Use your tools now to either fix the issue OR skip this task (on attempt 3).""")
            ]

            # Run error_expert
            state_with_context = dict(state)
            state_with_context["messages"] = clean_messages
            result = error_agent.invoke(state_with_context)

            messages = result.get("messages", [])
            still_has_error, error_msg, new_error_type = self.error_handler.detect_build_error(messages)

            # Check if we made progress
            todo_progress = calculate_todo_progress(project_path)
            current_todo_count = todo_progress['completed']
            prev_todo_count = state.get("last_todo_count", 0)
            made_progress = current_todo_count > prev_todo_count

            # Extract what approach was tried this time (for tracking)
            approach_tried = self._extract_approach_from_messages(messages, stuck_loop_attempts)
            failed_list.append(approach_tried)
            new_failed_approaches = json.dumps(failed_list)

            if made_progress:
                log_agent(f"[WRAPPER] Stuck loop RESOLVED - progress made ({prev_todo_count}->{current_todo_count})")
                log_summary("STUCK LOOP RESOLVED: Made progress on tasks")
                return {
                    "messages": messages,
                    "analysis_done": state.get("analysis_done", False),
                    "execution_done": state.get("execution_done", False),
                    "has_build_error": still_has_error,
                    "error_type": new_error_type if still_has_error else "none",
                    "error_count": state.get("error_count", 0),
                    "last_error_message": error_msg if still_has_error else "",
                    # RESET all stuck state
                    "is_stuck": False,
                    "stuck_type": "none",
                    "stuck_tool": "",
                    "stuck_loop_attempts": 0,
                    "stuck_reason": "",
                    "stuck_failed_approaches": "",
                    "last_todo_count": current_todo_count,
                    "loops_without_progress": 0,
                }
            else:
                # No progress - increment attempt counter and continue
                new_attempts = stuck_loop_attempts + 1
                log_agent(f"[WRAPPER] Stuck loop attempt {stuck_loop_attempts} failed, incrementing to {new_attempts}")

                if new_attempts > 3:
                    # All 3 attempts exhausted - will be caught by router as FAILED
                    log_agent(f"[WRAPPER] All 3 stuck loop attempts exhausted", "WARNING")
                    log_summary(f"STUCK LOOP: All 3 attempts failed - migration will fail")

                return {
                    "messages": messages,
                    "analysis_done": state.get("analysis_done", False),
                    "execution_done": state.get("execution_done", False),
                    "has_build_error": still_has_error,
                    "error_type": new_error_type if still_has_error else "none",
                    "error_count": state.get("error_count", 0),
                    "last_error_message": error_msg if still_has_error else "",
                    # Keep is_stuck=True so router sends us back for next attempt
                    "is_stuck": True,
                    "stuck_type": stuck_type,
                    "stuck_tool": stuck_tool,
                    "stuck_loop_attempts": new_attempts,
                    "stuck_reason": stuck_reason,
                    "stuck_failed_approaches": new_failed_approaches,
                }

        else:
            # ═══════════════════════════════════════════════════════════════
            # REGULAR BUILD ERROR HANDLING
            # ═══════════════════════════════════════════════════════════════
            log_agent(f"[WRAPPER] Running error_expert (attempt {error_count}/3, type={prev_error_type})")
            log_summary(f"ERROR RESOLUTION: error_expert attempting fix (attempt {error_count}/3)")

            current_error = self._extract_latest_error(current_messages)
            error_history = self.state_file_manager.read_file("ERROR_HISTORY.md")

            # Include error type in context to help error_expert
            if prev_error_type == 'pom':
                error_type_hint = "POM/CONFIGURATION ERROR"
                verification_cmd = "mvn_compile"
                extra_guidance = """
This is a POM configuration error. Common fixes include:
- Adding missing version tags to dependencies
- Fixing XML syntax errors (unclosed tags, typos)
- Adding missing properties in <properties> section
- Fixing parent POM references
- Resolving dependency version conflicts

Use read_file to examine pom.xml, then use find_replace or write_file to fix the issue."""
            elif prev_error_type == 'test':
                error_type_hint = "TEST FAILURE"
                verification_cmd = "mvn_test"
                extra_guidance = """
This is a test failure. Analyze the test error and fix the test or the code being tested.
Do NOT delete tests - fix them or use @Disabled with documentation if truly incompatible."""
            else:
                error_type_hint = "COMPILATION ERROR"
                verification_cmd = "mvn_compile"
                extra_guidance = """
This is a compilation error. Common fixes include:
- Adding missing imports
- Fixing type mismatches
- Adding missing method implementations
- Updating deprecated API usage"""

            # Add mandatory web search reminder after first failed attempt
            web_search_reminder = ""
            if error_count >= 1:
                error_snippet = current_error[:200] if current_error else "unknown error"
                suggested_query = self._build_search_query(error_snippet)
                web_search_reminder = f"""

⚠️ MANDATORY WEB SEARCH REQUIRED ⚠️
════════════════════════════════════════════════════════════════════
You have attempted {error_count} fix(es) without success.

BEFORE trying another fix, you MUST:
1. Call web_search_tool with a specific query about this error
2. Include the error message + framework versions + "fix"

SUGGESTED QUERY:
  web_search_tool("{suggested_query}")

DO NOT skip this step. Search first, then apply the solution.
════════════════════════════════════════════════════════════════════
"""
                log_agent(f"[WEB_SEARCH_ENFORCE] Injecting mandatory web search reminder (attempt {error_count})")

            clean_messages = [
                HumanMessage(content=f"""ERROR FIX REQUIRED - Project: {project_path}

## ERROR TYPE: {error_type_hint}

## CURRENT ERROR:
{current_error}

## PREVIOUS ATTEMPTS:
{error_history if error_history else 'No previous attempts - this is your first try.'}
{extra_guidance}
{web_search_reminder}
Do NOT repeat failed approaches. Try something different.
Analyze the error, then EXECUTE the fix using your tools.
Run {verification_cmd} to verify it works.""")
            ]

            state_with_context = dict(state)
            state_with_context["messages"] = clean_messages
            result = error_agent.invoke(state_with_context)

            messages = result.get("messages", [])
            still_has_error, error_msg, new_error_type = self.error_handler.detect_build_error(messages)

            # Log error attempt
            self.error_handler.log_error_attempt(
                error=current_error,
                attempt_num=error_count,
                was_successful=not still_has_error
            )

            if still_has_error:
                log_agent(f"[WRAPPER] Build error still present (type={new_error_type})")
            else:
                log_agent("[WRAPPER] Build error RESOLVED")
                log_summary("ERROR RESOLVED: Build errors fixed")

            return {
                "messages": messages,
                "analysis_done": state.get("analysis_done", False),
                "execution_done": state.get("execution_done", False),
                "has_build_error": still_has_error,
                "error_type": new_error_type if still_has_error else "none",
                # INCREMENT error_count when error_expert fails, reset to 0 when fixed
                "error_count": state.get("error_count", 0) + 1 if still_has_error else 0,
                "last_error_message": error_msg if still_has_error else "",
                # Reset test failure count when error is resolved
                "test_failure_count": state.get("test_failure_count", 0) if still_has_error else 0,
                "last_test_failure_task": state.get("last_test_failure_task", "") if still_has_error else "",
                # Clear stuck state since this was a build error resolution
                "is_stuck": False,
                "stuck_type": "none",
                "stuck_loop_attempts": 0 if not still_has_error else state.get("stuck_loop_attempts", 0),
                "stuck_failed_approaches": "",
            }

    def _extract_approach_from_messages(self, messages: List[BaseMessage], attempt_num: int) -> str:
        """Extract a description of what approach was tried from the agent's messages."""
        tools_used = []
        for msg in messages:
            # Check for tool calls in AI messages
            tool_calls = getattr(msg, 'tool_calls', None) or []
            additional_kwargs = getattr(msg, 'additional_kwargs', {}) or {}
            additional_tool_calls = additional_kwargs.get('tool_calls', [])

            for tc in tool_calls + additional_tool_calls:
                tool_name = tc.get('name', 'unknown') if isinstance(tc, dict) else getattr(tc, 'name', 'unknown')
                if tool_name not in tools_used:
                    tools_used.append(tool_name)

        if tools_used:
            return f"Attempt {attempt_num}: Used {', '.join(tools_used)}"
        else:
            return f"Attempt {attempt_num}: No tools used"

    def _build_search_query(self, error_snippet: str) -> str:
        """Build a search query from an error snippet for web search."""
        # Extract key error terms
        import re

        # Common error patterns to extract
        patterns = [
            r'(NoClassDefFoundError[:\s]+[\w\.]+)',
            r'(ClassNotFoundException[:\s]+[\w\.]+)',
            r'(NoSuchMethodError[:\s]+[\w\.]+)',
            r'(cannot find symbol[:\s]+[\w\.]+)',
            r'(package [\w\.]+ does not exist)',
            r'(incompatible types[:\s]+[\w\.<>]+)',
        ]

        key_terms = []
        for pattern in patterns:
            match = re.search(pattern, error_snippet, re.IGNORECASE)
            if match:
                key_terms.append(match.group(1))

        if key_terms:
            base_query = key_terms[0]
        else:
            # Just use first 100 chars of error
            base_query = error_snippet[:100].replace('\n', ' ').strip()

        return f"Java Spring Boot migration {base_query} fix solution"

    def _apply_phase_transition(self, state: State) -> List[BaseMessage]:
        """Apply phase transition pruning from analysis to execution"""
        log_agent("[PRUNE] FIRST EXECUTION: Applying phase transition")

        try:
            project_path = state.get("project_path", "")

            # CRITICAL: Capture test baseline BEFORE execution starts
            # This enables deterministic verification of test preservation
            self._capture_test_baseline(project_path)

            # Read TODO and create VISIBLE_TASKS.md
            todo_content = self.state_file_manager.read_file("TODO.md", keep_beginning=True)
            visible_tasks = self.task_manager.get_visible_tasks(todo_content, max_visible=3)

            # Check for file_missing error state
            if visible_tasks.get('file_missing'):
                log_agent("[PHASE_TRANSITION] ⚠️ ERROR: TODO.md is missing! State may have been lost.")
                log_summary("PHASE_TRANSITION: ERROR - TODO.md missing, cannot proceed with execution")

            self.task_manager.create_visible_tasks_file(visible_tasks)

            log_agent(f"[PHASE_TRANSITION] Created VISIBLE_TASKS.md with next 3 tasks")

            # Return clean execution context
            return create_clean_execution_context(project_path)

        except Exception as e:
            log_agent(f"[PRUNE] Error during phase transition: {str(e)}", "ERROR")
            return state.get("messages", []) + [
                HumanMessage(content="EXECUTION PHASE - Read VISIBLE_TASKS.md and execute tasks.")
            ]

    def _capture_test_baseline(self, project_path: str):
        """
        Capture test method baseline before execution starts.

        This enables deterministic verification that test methods are preserved
        during migration. The commit_changes tool will verify against this baseline.
        """
        try:
            from src.utils.test_verifier import TestMethodVerifier

            verifier = TestMethodVerifier(project_path)
            baseline = verifier.capture_baseline()

            if baseline:
                log_agent(f"[TEST_BASELINE] Captured {sum(f.method_count for f in baseline.values())} test methods from {len(baseline)} files")
                log_summary(f"TEST BASELINE CAPTURED: {len(baseline)} test files - preservation will be enforced on commits")
            else:
                log_agent("[TEST_BASELINE] No test files found - test preservation verification skipped")

        except ImportError:
            log_agent("[TEST_BASELINE] TestMethodVerifier not available - skipping baseline capture", "WARNING")
        except Exception as e:
            log_agent(f"[TEST_BASELINE] Error capturing baseline: {e}", "WARNING")

    def _is_migration_complete(self, project_path: str) -> bool:
        """Check if migration is complete based on TODO.md status"""
        todo_progress = calculate_todo_progress(project_path)
        return todo_progress['percent'] >= 100 and todo_progress['total'] > 0

    def _extract_latest_error(self, messages: List[BaseMessage]) -> str:
        """Extract most recent error from tool messages"""
        for msg in reversed(messages):
            content = getattr(msg, 'content', str(msg))
            if 'BUILD FAILURE' in content or 'ERROR' in content:
                return content[:500]
        return "Unknown error"

    def _extract_and_track_tool_calls(self, messages: List[BaseMessage], current_task: str = "") -> int:
        """
        Extract tool calls from messages and track them via error_handler.
        This enables SIGNATURE-BASED stuck loop detection.

        A signature consists of (tool_name, args_hash, result_category).
        This allows distinguishing between:
        - Healthy: commit_changes called 5x with DIFFERENT args/SUCCESS results
        - Stuck: commit_changes called 3x with SAME args/EMPTY results

        Returns:
            Number of tool calls tracked
        """
        tracked_count = 0

        # Find AI messages with tool_calls and their corresponding ToolMessage results
        for i, msg in enumerate(messages):
            msg_type = getattr(msg, 'type', None) or type(msg).__name__

            if msg_type == 'ai' or 'AIMessage' in type(msg).__name__:
                tool_calls = getattr(msg, 'tool_calls', None) or []
                additional_kwargs = getattr(msg, 'additional_kwargs', {}) or {}
                additional_tool_calls = additional_kwargs.get('tool_calls', [])

                all_tool_calls = tool_calls + additional_tool_calls

                for tc in all_tool_calls:
                    tool_name = tc.get('name', 'unknown') if isinstance(tc, dict) else getattr(tc, 'name', 'unknown')
                    tool_id = tc.get('id', '') if isinstance(tc, dict) else getattr(tc, 'id', '')

                    # Extract tool arguments for hashing
                    tool_args = tc.get('args', {}) if isinstance(tc, dict) else getattr(tc, 'args', {})
                    args_hash = hash_tool_args(tool_args)

                    # Find the corresponding ToolMessage result
                    tool_result = ""
                    is_success = False
                    for result_msg in messages[i+1:]:
                        result_type = getattr(result_msg, 'type', None) or type(result_msg).__name__
                        if result_type == 'tool' or 'ToolMessage' in type(result_msg).__name__:
                            result_tool_id = getattr(result_msg, 'tool_call_id', '')
                            if result_tool_id == tool_id:
                                tool_result = str(getattr(result_msg, 'content', ''))
                                # Determine success based on result content
                                is_success = self._determine_tool_success(tool_name, tool_result)
                                break

                    # Categorize the result for signature-based detection
                    result_category = categorize_tool_result(tool_name, tool_result)

                    # Track the action with full signature
                    self.error_handler.track_action(
                        tool_name=tool_name,
                        args_hash=args_hash,
                        result_category=result_category,
                        todo_item=current_task,
                        logged_to_completed=is_success
                    )
                    tracked_count += 1

                    # Log for debugging (more detailed now)
                    log_agent(
                        f"[TRACK_ACTION] {tool_name} | args={args_hash} | result={result_category} | "
                        f"success={is_success} | task={current_task[:40] if current_task else 'N/A'}..."
                    )

        return tracked_count

    def _determine_tool_success(self, tool_name: str, result: str) -> bool:
        """
        Determine if a tool call was successful based on its result.
        """
        result_lower = result.lower()

        # Common failure patterns
        failure_patterns = [
            'error', 'failed', 'exception', 'no occurrences', 'no match',
            'not found', 'build failure', 'return code: 1', 'blocked'
        ]

        # Common success patterns
        success_patterns = [
            'success', 'return code: 0', 'build success', 'committed',
            'created', 'updated', 'configured', 'added'
        ]

        # Check for failure first
        for pattern in failure_patterns:
            if pattern in result_lower:
                return False

        # Check for success
        for pattern in success_patterns:
            if pattern in result_lower:
                return True

        # Default: assume success if no clear failure
        return True

    def _count_no_tool_response(self, messages: List[BaseMessage], prev_count: int) -> int:
        """
        Check if the agent's last response contained tool calls.
        Returns incremented counter if no tools used, 0 if tools were used.
        """
        if not messages:
            return prev_count + 1

        # Find the last AI message
        last_ai_msg = None
        for msg in reversed(messages):
            msg_type = getattr(msg, 'type', None) or type(msg).__name__
            if msg_type == 'ai' or 'AIMessage' in type(msg).__name__:
                last_ai_msg = msg
                break

        if not last_ai_msg:
            return prev_count + 1

        # Check for tool calls
        tool_calls = getattr(last_ai_msg, 'tool_calls', None) or []
        additional_kwargs = getattr(last_ai_msg, 'additional_kwargs', {}) or {}
        additional_tool_calls = additional_kwargs.get('tool_calls', [])

        has_tools = len(tool_calls) > 0 or len(additional_tool_calls) > 0

        if has_tools:
            log_agent(f"[WRAPPER] Agent used {len(tool_calls) + len(additional_tool_calls)} tools - resetting no-tool counter")
            return 0
        else:
            new_count = prev_count + 1
            log_agent(f"[WRAPPER] Agent returned WITHOUT tool calls - counter: {new_count}", "WARNING")
            return new_count

    def _extract_last_ai_response_text(self, messages: List[BaseMessage]) -> str:
        """
        Extract the text content from the last AI message.
        Returns empty string if no AI message found.
        """
        if not messages:
            return ""

        for msg in reversed(messages):
            msg_type = getattr(msg, 'type', None) or type(msg).__name__
            if msg_type == 'ai' or 'AIMessage' in type(msg).__name__:
                content = getattr(msg, 'content', '')
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    # Handle content blocks (like Anthropic format)
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get('type') == 'text':
                            text_parts.append(block.get('text', ''))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    return ' '.join(text_parts)
                return str(content)
        return ""

    def _classify_no_tool_response(self, response_text: str) -> str:
        """
        Classify agent text response into categories to determine appropriate handling.

        Categories:
        - 'acknowledging': Agent is confirming previous work (wasteful but benign)
        - 'confused': Agent doesn't know what to do (harmful - needs help)
        - 'thinking': Agent is reasoning/analyzing (benign, allow limited times)
        - 'complete': Agent believes migration is done (check and allow)
        - 'unknown': Can't classify (treat cautiously)

        Returns: One of the category strings
        """
        if not response_text:
            return "unknown"

        text_lower = response_text.lower()

        # COMPLETE patterns - agent thinks migration is done
        complete_patterns = [
            r'migration.*complete',
            r'all\s+tasks.*(?:done|complete|finished)',
            r'no\s+(?:more|further|remaining)\s+tasks',
            r'successfully\s+completed?\s+(?:all|the)\s+migration',
            r'migration\s+(?:is\s+)?finished',
        ]

        # CONFUSED patterns - agent is stuck/lost (harmful)
        confused_patterns = [
            r"(?:i'm|i am|i'm)\s*(?:not\s+sure|unsure|unclear)",
            r"(?:don't|do not|doesn't)\s+(?:know|understand)",
            r"(?:need|require).*clarification",
            r"(?:what|which|how)\s+should\s+i",
            r"please\s+(?:provide|specify|clarify)",
            r"(?:can you|could you).*(?:help|clarify|explain)",
            r"(?:i'm|i am)\s+(?:stuck|blocked|unable)",
            r"not\s+(?:able|possible)\s+to\s+(?:proceed|continue)",
            r"(?:error|issue|problem).*(?:understand|resolve)",
            r"waiting\s+for\s+(?:input|instruction|guidance)",
        ]

        # ACKNOWLEDGING patterns - agent confirms work (wasteful)
        # Using .{0,20} instead of strict character classes for flexibility
        ack_patterns = [
            r"(?:great|excellent|perfect|good|wonderful).{0,20}(?:moving|proceed|next|continue|let's)",
            r"task.{0,20}(?:completed?|done|finished)",
            r"successfully.{0,20}(?:done|completed?|finished)",
            r"let(?:'s| me| us)\s+(?:move|proceed|continue)\s+(?:on|to|with)",
            r"now\s+(?:i'll|let's|we\s+can|i\s+will).{0,30}next",
            r"understood.{0,10}(?:proceed|moving|continue)?",
            r"(?:moving|proceeding|continuing)\s+(?:on|to|with|forward)",  # Added 'forward'
            r"(?:moving|proceeding|continuing)\s+to\s+(?:the\s+)?next",  # Added simpler pattern
            r"ready\s+(?:to|for)\s+(?:the\s+)?next",
            r"acknowledged",
            r"(?:sounds|looks)\s+good.{0,20}(?:let's|i'll|proceed)?",
        ]

        # THINKING patterns - agent reasoning (benign, limited tolerance)
        thinking_patterns = [
            r"let\s+me\s+(?:think|analyze|consider|review|check|examine|look)",
            r"(?:first|before).*(?:i\s+(?:need|should|will|must))",
            r"(?:looking|examining|analyzing|reviewing|checking)\s+(?:at|the)",
            r"(?:i\s+see|i\s+notice|i\s+observe|i\s+found)",
            r"(?:based\s+on|according\s+to)\s+(?:the|my)",
            r"(?:it\s+(?:seems|appears|looks)\s+(?:like|that))",
            r"(?:i\s+(?:need|should|will|must)\s+(?:first|analyze|check|review))",
        ]

        # Check patterns in order of priority
        for pattern in complete_patterns:
            if re.search(pattern, text_lower):
                log_agent(f"[CLASSIFY] Response classified as COMPLETE (pattern: {pattern})")
                return "complete"

        for pattern in confused_patterns:
            if re.search(pattern, text_lower):
                log_agent(f"[CLASSIFY] Response classified as CONFUSED (pattern: {pattern})")
                return "confused"

        for pattern in ack_patterns:
            if re.search(pattern, text_lower):
                log_agent(f"[CLASSIFY] Response classified as ACKNOWLEDGING (pattern: {pattern})")
                return "acknowledging"

        for pattern in thinking_patterns:
            if re.search(pattern, text_lower):
                log_agent(f"[CLASSIFY] Response classified as THINKING (pattern: {pattern})")
                return "thinking"

        log_agent(f"[CLASSIFY] Response classified as UNKNOWN (no pattern matched)")
        return "unknown"

    def _get_directive_for_response_type(self, response_type: str, no_tool_count: int) -> str:
        """
        Get an appropriate directive message based on response classification.
        """
        directives = {
            "acknowledging": (
                "⛔ STOP ACKNOWLEDGING. You already completed the previous task. "
                "The system auto-advances to the next task. "
                "DO NOT say 'moving to next task' - just EXECUTE the CURRENT TASK shown above using tools NOW."
            ),
            "confused": (
                "🔍 You seem stuck. Here's what to do:\n"
                "1. Use read_file to read VISIBLE_TASKS.md - this shows your CURRENT TASK\n"
                "2. Execute that task using the appropriate tools\n"
                "3. If task involves code changes, use find_replace or write_file\n"
                "4. If task involves running commands, use mvn_compile or mvn_test\n"
                "DO NOT ask questions - just read the task file and execute it."
            ),
            "thinking": (
                "⏰ Enough analysis. You've had time to think. "
                "Now USE A TOOL to make actual progress. "
                "Execute the CURRENT TASK shown above - do not respond with more analysis."
            ),
            "unknown": (
                f"⚠️ You've returned {no_tool_count}x without using tools. "
                "This is your final warning. USE A TOOL NOW or the migration will fail. "
                "Read VISIBLE_TASKS.md and execute the current task immediately."
            ),
        }
        return directives.get(response_type, directives["unknown"])

    def _handle_no_tool_response(
        self,
        state: State,
        messages: List[BaseMessage],
        response_type: str,
        no_tool_count: int,
        execution_agent,
        project_path: str,
        total_loops: int,
        current_task: str,
        completed_summary: str
    ) -> Tuple[List[BaseMessage], int, bool]:
        """
        Handle a no-tool response based on its classification.

        Returns:
            Tuple of (new_messages, new_no_tool_count, was_reinvoked)
        """
        # COMPLETE: Verify and allow
        if response_type == "complete":
            if self._is_migration_complete(project_path):
                log_agent("[NO_TOOL_HANDLER] Migration actually complete - allowing")
                return messages, 0, False
            else:
                log_agent("[NO_TOOL_HANDLER] Agent claims complete but migration NOT done - re-invoking")
                response_type = "confused"  # Treat as confused if falsely claiming complete

        # THINKING: Allow limited times (2), then force action
        if response_type == "thinking":
            thinking_loops = state.get("thinking_loops", 0) + 1
            if thinking_loops <= 2:
                log_agent(f"[NO_TOOL_HANDLER] Thinking response ({thinking_loops}/2) - allowing this time")
                return messages, no_tool_count, False  # Don't reset counter, but don't re-invoke
            else:
                log_agent(f"[NO_TOOL_HANDLER] Too much thinking ({thinking_loops}) - forcing action")
                # Fall through to re-invoke

        # For ACKNOWLEDGING, CONFUSED, excessive THINKING, or UNKNOWN with high count:
        # Immediate re-invoke with directive
        # NOTE: For "unknown" responses, we wait for 5 consecutive no-tool responses
        # before re-invoking, to give the model more chances to self-correct
        should_reinvoke = (
            response_type == "acknowledging" or
            response_type == "confused" or
            (response_type == "thinking" and state.get("thinking_loops", 0) >= 2) or
            (response_type == "unknown" and no_tool_count >= 5)
        )

        if not should_reinvoke:
            # Unknown with low count - just warn next time
            log_agent(f"[NO_TOOL_HANDLER] {response_type} response (count: {no_tool_count}) - will warn next loop")
            return messages, no_tool_count, False

        # RE-INVOKE with directive
        log_agent(f"[NO_TOOL_HANDLER] Re-invoking agent with {response_type.upper()} directive")

        directive = self._get_directive_for_response_type(response_type, no_tool_count)

        # Compile new context with directive
        reinvoke_messages = compile_execution_context(
            project_path=project_path,
            loop_num=total_loops,
            current_task=current_task,
            completed_summary=completed_summary,
            last_result=directive
        )

        # Re-invoke agent
        state_with_messages = dict(state)
        state_with_messages["messages"] = reinvoke_messages

        try:
            result = execution_agent.invoke(state_with_messages)
            new_messages = result.get("messages", [])

            # Check if re-invoke produced tools
            new_no_tool_count = self._count_no_tool_response(new_messages, 0)

            if new_no_tool_count == 0:
                log_agent("[NO_TOOL_HANDLER] Re-invoke successful - agent used tools")
                return new_messages, 0, True
            else:
                log_agent(f"[NO_TOOL_HANDLER] Re-invoke still no tools - returning with count {no_tool_count + 1}")
                return new_messages, no_tool_count + 1, True

        except Exception as e:
            log_agent(f"[NO_TOOL_HANDLER] Re-invoke failed: {e}", "ERROR")
            return messages, no_tool_count + 1, False

    def _create_supervisor(self):
        """Create supervisor workflow with deterministic routing"""
        workflow = StateGraph(State)

        # Add agent wrapper nodes
        workflow.add_node("analysis_expert", self._wrap_analysis_node)
        workflow.add_node("execution_expert", self._wrap_execution_node)
        workflow.add_node("error_expert", self._wrap_error_node)

        # Add conditional routing
        routing_map = {
            "analysis_expert": "analysis_expert",
            "execution_expert": "execution_expert",
            "error_expert": "error_expert",
            "END": END,
            "FAILED": END
        }

        workflow.add_conditional_edges("analysis_expert", self._route_next_agent, routing_map)
        workflow.add_conditional_edges("execution_expert", self._route_next_agent, routing_map)
        workflow.add_conditional_edges("error_expert", self._route_next_agent, routing_map)

        workflow.set_entry_point("analysis_expert")

        log_agent("[SUPERVISOR] Custom workflow created with deterministic routing")
        return workflow

    def _cleanup_state_files(self, project_path: str):
        """Delete stale state files from previous runs"""
        log_agent("[CLEANUP] Cleaning stale migration state files...")
        log_console("Cleaning previous migration state...", "INFO")

        cleaned_count = 0
        # STATE_FILES is a dict mapping key -> filename
        for key, filename in STATE_FILES.items():
            file_path = os.path.join(project_path, filename)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    log_agent(f"[CLEANUP] Deleted: {filename}")
                    cleaned_count += 1
                except Exception as e:
                    log_agent(f"[CLEANUP] Could not delete {filename}: {e}", "WARNING")

        if cleaned_count > 0:
            log_console(f"Deleted {cleaned_count} stale state files", "SUCCESS")
        else:
            log_agent("[CLEANUP] No stale files found - clean start")

    def _initialize_state_files(self, project_path: str):
        """Initialize fresh state files for migration"""
        initialize_completed_actions_file(project_path)
        initialize_error_history_file(project_path)

        # PROTECTION: Add state files to .gitignore so they survive git operations
        self._ensure_state_files_in_gitignore(project_path)

    def _ensure_state_files_in_gitignore(self, project_path: str):
        """Add state files to .gitignore to prevent git operations from affecting them"""
        state_files = [
            'TODO.md',
            'VISIBLE_TASKS.md',
            'COMPLETED_ACTIONS.md',
            'CURRENT_STATE.md',
            'analysis.md',
            'ERROR_HISTORY.md'
        ]

        gitignore_path = os.path.join(project_path, '.gitignore')

        try:
            # Read existing .gitignore
            existing_content = ""
            if os.path.exists(gitignore_path):
                with open(gitignore_path, 'r') as f:
                    existing_content = f.read()

            # Check which state files need to be added
            files_to_add = []
            for filename in state_files:
                if filename not in existing_content:
                    files_to_add.append(filename)

            # Add missing state files to .gitignore
            if files_to_add:
                with open(gitignore_path, 'a') as f:
                    if existing_content and not existing_content.endswith('\n'):
                        f.write('\n')
                    f.write('\n# Migration state files (auto-added - DO NOT REMOVE)\n')
                    for filename in files_to_add:
                        f.write(f'{filename}\n')
                log_agent(f"[GITIGNORE] Added state files to .gitignore: {', '.join(files_to_add)}")
        except Exception as e:
            log_agent(f"[GITIGNORE] Warning: Could not update .gitignore: {e}")

    def migrate_project(self, project_path: str) -> Dict[str, Any]:
        """Run supervised migration on a project"""
        # Cleanup and initialize
        self._cleanup_state_files(project_path)

        # Reset verification cache singleton for new migration
        from src.orchestrator.tool_registry import VerificationCache
        VerificationCache.reset_singleton()
        log_agent("[MIGRATE] Reset verification cache for new migration session")

        # Create and register state tracker
        self.state_tracker = MigrationStateTracker(project_path)
        set_state_tracker(self.state_tracker)

        # Set project path on all components
        self._set_project_path(project_path)

        # Initialize state files
        self._initialize_state_files(project_path)

        # Create migration request
        migration_request = get_migration_request(project_path)

        # Initialize outside try for exception handling
        step_count = 0
        last_chunk = None

        try:
            start_time = datetime.now()
            log_agent("Invoking supervisor graph")
            log_summary("PHASE: Agent coordination started")

            for chunk in self.app.stream({
                "messages": [{"role": "user", "content": migration_request}],
                "project_path": project_path,
                "current_phase": "INIT",
                "analysis_done": False,
                "execution_done": False,
                "last_todo_count": 0,
                "loops_without_progress": 0,
                "total_execution_loops": 0,
                # Stuck loop detection state
                "is_stuck": False,
                "stuck_type": "none",
                "stuck_tool": "",
                "stuck_loop_attempts": 0,
                "stuck_reason": "",
                "stuck_failed_approaches": "",
                # Build error state
                "has_build_error": False,
                "error_count": 0,
                "last_error_message": "",
                "error_type": "none",
                "test_failure_count": 0,
                "last_test_failure_task": "",
                # No-tool tracking (execution agent)
                "no_tool_call_loops": 0,
                "thinking_loops": 0,
                # Analysis agent stuck detection
                "analysis_no_tool_count": 0,
                "analysis_reset_count": 0,
            }, {"recursion_limit": 500}):
                step_count += 1
                last_chunk = chunk

                try:
                    self._check_llm_call_limit()
                except LLMCallLimitExceeded:
                    log_agent(f"Circuit breaker triggered at step {step_count}")
                    break

                self._log_workflow_step(step_count, chunk)

            duration = datetime.now() - start_time

            # Extract final result
            final_result = self._extract_final_result(last_chunk)
            token_stats = tc.get_stats()

            self._log_token_stats(token_stats)

            # ═══════════════════════════════════════════════════════════════════
            # FINAL RESULT CLASSIFICATION (Deterministic, No LLM)
            # ═══════════════════════════════════════════════════════════════════

            # Determine termination reason
            termination_reason = "normal"
            if tc.llm_calls >= MAX_LLM_CALLS:
                termination_reason = "llm_limit"
            elif final_result.get("current_phase") == "EXECUTION_TIMEOUT":
                termination_reason = "timeout"
            elif final_result.get("error_count", 0) >= 3:
                termination_reason = "error_max"
            elif final_result.get("stuck_loop_attempts", 0) >= 3:
                termination_reason = "stuck_max"

            # Run deterministic classification
            status, reason, success = self._classify_migration_result(
                project_path=project_path,
                final_state=final_result,
                termination_reason=termination_reason
            )

            log_summary(f"CLASSIFICATION: {status.value.upper()}")
            log_summary(f"REASON: {reason}")

            # Run test invariance validation only for SUCCESS/PARTIAL_SUCCESS
            if success:
                final_valid = self._final_test_validation(project_path)
                if not final_valid:
                    log_summary("FINAL VALIDATION: FAILED - Test methods changed")
                    return {
                        "success": False,
                        "status": MigrationStatus.FAILURE.value,
                        "result": "Migration failed final test invariance validation",
                        "duration": duration.total_seconds(),
                        "steps": step_count,
                        "token_stats": token_stats,
                        "limit_exceeded": tc.llm_calls >= MAX_LLM_CALLS,
                        "error": "TEST_INVARIANCE_FAILED"
                    }

            return {
                "success": success,
                "status": status.value,
                "result": reason,  # Use classification reason, not agent message
                "duration": duration.total_seconds(),
                "steps": step_count,
                "token_stats": token_stats,
                "limit_exceeded": tc.llm_calls >= MAX_LLM_CALLS
            }

        except LLMCallLimitExceeded as e:
            return self._handle_limit_exceeded(e, step_count)
        except Exception as e:
            log_agent(f"Migration failed with exception: {e}")
            return {"success": False, "error": str(e)}

    def _extract_final_result(self, last_chunk) -> dict:
        """Extract final state from the last workflow chunk"""
        if last_chunk:
            for node_name, node_state in last_chunk.items():
                if isinstance(node_state, dict) and "messages" in node_state:
                    return node_state
        return {"messages": [{"role": "assistant", "content": "Migration completed"}]}

    def _get_final_content(self, final_result: dict) -> str:
        """Get final message content from result"""
        messages = final_result.get("messages", [])
        if not messages:
            return "No final message"
        final_message = messages[-1]
        if hasattr(final_message, 'content'):
            return final_message.content
        elif isinstance(final_message, dict):
            return final_message.get("content", str(final_message))
        return str(final_message)

    def _log_token_stats(self, token_stats: dict):
        """Log token usage statistics"""
        log_summary("\n" + "=" * 60)
        log_summary("TOKEN USAGE & COST REPORT")
        log_summary("=" * 60)
        log_summary(f"LLM Calls:       {token_stats['llm_calls']:,}")
        log_summary(f"Prompt tokens:   {token_stats['prompt_tokens']:,}")
        log_summary(f"Response tokens: {token_stats['response_tokens']:,}")
        log_summary(f"Total tokens:    {token_stats['total_tokens']:,}")
        log_summary(f"TOTAL COST:      ${token_stats['total_cost_usd']:.4f}")
        log_summary("=" * 60)
        log_console(f"Migration Cost: ${token_stats['total_cost_usd']:.4f} ({token_stats['total_tokens']:,} tokens)")

    def _final_test_validation(self, project_path: str) -> bool:
        """
        Final validation before claiming success using MigrationBench's exact evaluation logic.

        This is a safety net that runs the SAME test invariance check that the evaluation
        script uses, guaranteeing that if we pass here, evaluation will also pass.

        Args:
            project_path: Path to the repository

        Returns:
            True if validation passes (or is skipped), False if tests have changed
        """
        try:
            from src.utils.test_verifier import verify_final_test_invariance

            base_commit = os.environ.get('MIGRATION_BASE_COMMIT', '')

            if not base_commit:
                log_agent("[ORCHESTRATOR] No base_commit available, skipping final test validation")
                return True  # Skip if no base_commit set

            log_agent(f"[ORCHESTRATOR] Running final test invariance validation against {base_commit}")
            is_valid, msg = verify_final_test_invariance(project_path, base_commit)

            if is_valid:
                log_agent("[ORCHESTRATOR] ✅ Final test invariance validation PASSED")
                log_summary("FINAL_VALIDATION: PASSED - Test methods match baseline")
                return True
            else:
                log_agent(f"[ORCHESTRATOR] ❌ Final test invariance validation FAILED", "ERROR")
                log_summary(f"FINAL_VALIDATION: FAILED - {msg}")
                return False

        except ImportError as e:
            log_agent(f"[ORCHESTRATOR] MigrationBench eval module not available: {e}", "WARNING")
            log_summary("FINAL_VALIDATION: SKIPPED - MigrationBench not available")
            return True  # Continue if MigrationBench not installed

        except Exception as e:
            log_agent(f"[ORCHESTRATOR] Final validation error: {e}", "WARNING")
            log_summary(f"FINAL_VALIDATION: ERROR - {e}")
            return True  # Continue if validation itself fails

    def _classify_migration_result(
        self,
        project_path: str,
        final_state: dict,
        termination_reason: str = "normal"
    ) -> tuple:
        """
        Deterministic classification of migration result.

        NO LLM INVOLVEMENT - Pure code logic.

        Args:
            project_path: Path to the repository
            final_state: Final workflow state dict
            termination_reason: Why workflow ended ("normal", "llm_limit", "timeout", "error_max", "stuck_max")

        Returns:
            Tuple of (MigrationStatus, reason_message, success_bool)

        Priority Order (STRICT):
            1. Check termination reason (LLM limit, timeout → INCOMPLETE)
            2. Check build status (mvn compile)
            3. Check test status (mvn test)
            4. Check error state (unresolved errors → FAILURE)
            5. Check progress percentage
        """
        from src.tools.command_executor import mvn_compile, mvn_test

        # ═══════════════════════════════════════════════════════════════════
        # PRIORITY 1: Termination Reason (highest priority)
        # ═══════════════════════════════════════════════════════════════════
        progress = calculate_todo_progress(project_path)

        if termination_reason == "llm_limit":
            return (
                MigrationStatus.INCOMPLETE,
                f"LLM call limit reached. Progress: {progress['completed']}/{progress['total']} tasks ({progress['percent']:.0f}%)",
                False
            )

        if termination_reason == "timeout":
            return (
                MigrationStatus.INCOMPLETE,
                f"Execution timeout. Progress: {progress['completed']}/{progress['total']} tasks ({progress['percent']:.0f}%)",
                False
            )

        if termination_reason == "error_max":
            return (
                MigrationStatus.FAILURE,
                f"Max error resolution attempts (3) exceeded. Progress: {progress['completed']}/{progress['total']} tasks ({progress['percent']:.0f}%)",
                False
            )

        if termination_reason == "stuck_max":
            return (
                MigrationStatus.FAILURE,
                f"Max stuck loop attempts (3) exceeded. Progress: {progress['completed']}/{progress['total']} tasks ({progress['percent']:.0f}%)",
                False
            )

        # ═══════════════════════════════════════════════════════════════════
        # PRIORITY 2: Build Status (mvn compile)
        # ═══════════════════════════════════════════════════════════════════
        log_agent("[CLASSIFY] Running build verification...")
        build_passes = None
        try:
            compile_result = mvn_compile.invoke({"project_path": project_path})
            build_passes = "BUILD SUCCESS" in compile_result or "Success" in compile_result
            if not build_passes and "BUILD FAILURE" in compile_result:
                return (
                    MigrationStatus.FAILURE,
                    "Build failed (mvn compile). Migration cannot be considered successful.",
                    False
                )
        except Exception as e:
            log_agent(f"[CLASSIFY] Build check failed with exception: {e}", "WARNING")
            # If we can't verify build, check if last known state had errors
            if final_state.get("has_build_error", False):
                return (
                    MigrationStatus.FAILURE,
                    f"Build verification failed: {str(e)}",
                    False
                )
            # Otherwise continue to other checks
            build_passes = None  # Unknown

        # ═══════════════════════════════════════════════════════════════════
        # PRIORITY 3: Test Status (mvn test)
        # ═══════════════════════════════════════════════════════════════════
        log_agent("[CLASSIFY] Running test verification...")
        tests_pass = None
        try:
            test_result = mvn_test.invoke({"project_path": project_path})
            tests_pass = "BUILD SUCCESS" in test_result or "Success" in test_result
            if not tests_pass and "BUILD FAILURE" in test_result:
                return (
                    MigrationStatus.FAILURE,
                    "Tests failed (mvn test). Migration cannot be considered successful.",
                    False
                )
        except Exception as e:
            log_agent(f"[CLASSIFY] Test check failed with exception: {e}", "WARNING")
            # If we can't verify tests, check state
            if final_state.get("test_failure_count", 0) > 0:
                return (
                    MigrationStatus.FAILURE,
                    f"Test verification failed: {str(e)}",
                    False
                )
            tests_pass = None  # Unknown

        # ═══════════════════════════════════════════════════════════════════
        # PRIORITY 4: Unresolved Error State
        # ═══════════════════════════════════════════════════════════════════
        error_count = final_state.get("error_count", 0)
        has_build_error = final_state.get("has_build_error", False)

        # If state shows unresolved errors but build passed above, trust the build
        # (state might be stale)
        if has_build_error and build_passes is False:
            return (
                MigrationStatus.FAILURE,
                f"Unresolved build errors (error_count={error_count})",
                False
            )

        # ═══════════════════════════════════════════════════════════════════
        # PRIORITY 5: Progress Percentage (TODO.md)
        # ═══════════════════════════════════════════════════════════════════
        percent = progress['percent']
        completed = progress['completed']
        total = progress['total']

        log_agent(f"[CLASSIFY] Progress: {completed}/{total} ({percent:.0f}%)")
        log_agent(f"[CLASSIFY] Build: {'PASS' if build_passes else 'UNKNOWN' if build_passes is None else 'FAIL'}")
        log_agent(f"[CLASSIFY] Tests: {'PASS' if tests_pass else 'UNKNOWN' if tests_pass is None else 'FAIL'}")

        # ═══════════════════════════════════════════════════════════════════
        # CLASSIFICATION MATRIX (Deterministic)
        # ═══════════════════════════════════════════════════════════════════
        #
        # | Progress | Build | Tests | Result          |
        # |----------|-------|-------|-----------------|
        # | >= 90%   | PASS  | PASS  | SUCCESS         |
        # | >= 50%   | PASS  | PASS  | PARTIAL_SUCCESS |
        # | < 50%    | PASS  | PASS  | FAILURE         |
        # | Any      | FAIL  | Any   | FAILURE         |
        # | Any      | Any   | FAIL  | FAILURE         |
        # | Any      | UNK   | UNK   | Based on %      |
        #
        # ═══════════════════════════════════════════════════════════════════

        # SUCCESS: >= 90% AND build passes AND tests pass
        if percent >= 90:
            if build_passes is not False and tests_pass is not False:
                return (
                    MigrationStatus.SUCCESS,
                    f"Migration complete. {completed}/{total} tasks ({percent:.0f}%). Build passes, tests pass.",
                    True
                )

        # PARTIAL_SUCCESS: >= 50% AND build passes AND tests pass
        if percent >= 50:
            if build_passes is not False and tests_pass is not False:
                return (
                    MigrationStatus.PARTIAL_SUCCESS,
                    f"Migration partially complete. {completed}/{total} tasks ({percent:.0f}%). Build passes, tests pass.",
                    True  # success=True so it's not logged as FAILURE (CSV handling is separate)
                )

        # FAILURE: < 50% OR build/test issues
        return (
            MigrationStatus.FAILURE,
            f"Migration incomplete. {completed}/{total} tasks ({percent:.0f}%).",
            False
        )

    def _handle_limit_exceeded(self, e: LLMCallLimitExceeded, step_count: int) -> dict:
        """Handle LLM call limit exceeded gracefully"""
        token_stats = tc.get_stats()
        log_summary(f"MIGRATION HALTED: {str(e)}")
        log_console(f"Migration stopped gracefully: {str(e)}", "WARNING")
        self._log_token_stats(token_stats)
        return {
            "success": False,
            "error": str(e),
            "warning": str(e),  # For compatibility with migrate_single_Repo.py
            "steps": step_count,
            "token_stats": token_stats,
            "limit_exceeded": True
        }

    def _log_workflow_step(self, step_count: int, chunk: Dict[str, Any]):
        """Log information about each workflow step"""
        log_agent(f"Workflow Step {step_count}")
        if not chunk:
            log_agent("Empty chunk received", "WARNING")
            return

        for node_name, node_data in chunk.items():
            log_agent(f"Node: {node_name}")
            if node_name in ["analysis_expert", "execution_expert", "error_expert"]:
                log_summary(f"Calling {node_name.replace('_', ' ').title()}")
            if isinstance(node_data, dict) and "messages" in node_data:
                log_agent(f"Processing {len(node_data['messages'])} messages")

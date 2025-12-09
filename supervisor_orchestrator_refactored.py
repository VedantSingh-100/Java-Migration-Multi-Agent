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
import json
from datetime import datetime
from typing import Dict, Any, List

import tiktoken
from dotenv import load_dotenv
from pydantic import PrivateAttr

from langchain_anthropic import ChatAnthropic
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

from src.tools import all_tools_flat
from src.tools.state_management import set_state_tracker
from prompts.prompt_loader import (
    get_supervisor_prompt,
    get_migration_request,
    get_analysis_expert_prompt,
    get_execution_expert_prompt,
    get_error_expert_prompt,
)

load_dotenv()
set_verbose(True)

# Anthropic API key from environment variable
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is required")

# Global token counter
tc = TokenCounter()
ENC = tiktoken.get_encoding("cl100k_base")


class LLMCallLimitExceeded(Exception):
    """Exception raised when the LLM call limit is exceeded during migration"""
    pass


class CircuitBreakerChatAnthropic(ChatAnthropic):
    """Wrapper around ChatAnthropic that enforces LLM call limits by checking BEFORE each call"""

    _token_counter: Any = PrivateAttr()
    _max_calls: int = PrivateAttr()

    def __init__(self, token_counter, max_calls, **kwargs):
        super().__init__(**kwargs)
        self._token_counter = token_counter
        self._max_calls = max_calls

    def _generate(self, *args, **kwargs):
        if self._token_counter.llm_calls >= self._max_calls:
            log_agent(f"Circuit breaker preventing LLM call: {self._token_counter.llm_calls}/{self._max_calls}", "ERROR")
            raise LLMCallLimitExceeded(
                f"LLM call limit of {self._max_calls} exceeded ({self._token_counter.llm_calls} calls already made)"
            )
        return super()._generate(*args, **kwargs)

    async def _agenerate(self, *args, **kwargs):
        if self._token_counter.llm_calls >= self._max_calls:
            log_agent(f"Circuit breaker preventing LLM call (async): {self._token_counter.llm_calls}/{self._max_calls}", "ERROR")
            raise LLMCallLimitExceeded(
                f"LLM call limit of {self._max_calls} exceeded ({self._token_counter.llm_calls} calls already made)"
            )
        return await super()._agenerate(*args, **kwargs)


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
        """Create circuit-breaker wrapped model instance"""
        return CircuitBreakerChatAnthropic(
            token_counter=tc,
            max_calls=MAX_LLM_CALLS,
            model="claude-sonnet-4-20250514",
            api_key=ANTHROPIC_API_KEY,
            callbacks=[LLMLogger(), tc],
            max_tokens=8192,
            max_retries=5,
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

        log_agent(f"[ROUTER] Phase: {current_phase} | Analysis: {analysis_done} | Execution: {execution_done} | Error: {has_build_error} (type={error_type}, count={error_count}, test_failures={test_failure_count})")

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

        # PRIORITY 2: Check for build errors (now with test failure retry logic)
        if has_build_error:
            if error_count >= 3:
                log_agent("[ROUTER] -> Max error attempts reached, MIGRATION FAILED", "ERROR")
                log_summary("MIGRATION FAILED: Max error attempts (3) reached")
                return "FAILED"

            # Differentiate between compile errors and test failures
            if error_type == 'test':
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
                # Compile errors go to error_expert immediately
                log_agent(f"[ROUTER] -> Compile error detected, routing to error_expert (attempt {error_count}/3)")
                return "error_expert"

        # PRIORITY 3: Route based on phase
        if not analysis_done:
            log_agent("[ROUTER] -> Routing to analysis_expert")
            return "analysis_expert"
        else:
            log_agent("[ROUTER] -> Routing to execution_expert")
            return "execution_expert"

    def _wrap_analysis_node(self, state: State):
        """Wrapper for analysis agent using AnalysisNodeWrapper logic"""
        log_agent("[WRAPPER] Running analysis_expert")

        analysis_agent = self.migration_workers[0]
        result = analysis_agent.invoke(state)

        # Auto-detect completion
        project_path = state.get("project_path", "")
        messages = result.get("messages", [])
        analysis_complete = detect_analysis_complete(project_path, messages)

        if analysis_complete:
            log_agent("[WRAPPER] Analysis AUTO-DETECTED as complete")
            log_summary("ANALYSIS PHASE: AUTO-COMPLETED (files created)")

        return {
            "messages": messages,
            "analysis_done": analysis_complete,
            "current_phase": "ANALYSIS_COMPLETE" if analysis_complete else "ANALYSIS"
        }

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

        # Check for stuck loop patterns
        if total_loops >= 3:
            is_stuck, stuck_reason = self.error_handler.detect_stuck_loop()
            if is_stuck:
                log_agent(f"[STUCK] Loop pattern detected: {stuck_reason}", "WARNING")
                log_console(f"Loop pattern: {stuck_reason}", "WARNING")

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

        # Detect if agent returned without making tool calls
        new_no_tool_loops = self._count_no_tool_response(messages, no_tool_loops)

        return {
            "messages": messages,
            "analysis_done": state.get("analysis_done", False),
            "execution_done": execution_complete,
            "current_phase": "EXECUTION_COMPLETE" if execution_complete else "EXECUTION",
            "last_todo_count": current_todo_count,
            "loops_without_progress": new_loops_without_progress,
            "total_execution_loops": total_loops,
            "stuck_intervention_active": new_loops_without_progress >= MAX_LOOPS_WITHOUT_PROGRESS,
            "has_build_error": has_error,
            "error_type": error_type,
            "error_count": state.get("error_count", 0) + (1 if has_error else 0),
            "last_error_message": error_msg if has_error else "",
            "no_tool_call_loops": new_no_tool_loops,
            "test_failure_count": new_test_failure_count,
            "last_test_failure_task": new_last_test_failure_task,
        }

    def _wrap_error_node(self, state: State):
        """Wrapper for error agent with error resolution tracking"""
        error_count = state.get("error_count", 0)
        project_path = state.get("project_path", "")
        prev_error_type = state.get("error_type", "none")

        log_agent(f"[WRAPPER] Running error_expert (attempt {error_count}/3, type={prev_error_type})")
        log_summary(f"ERROR RESOLUTION: error_expert attempting fix (attempt {error_count}/3)")

        error_agent = self.migration_workers[2]
        current_messages = state.get("messages", [])

        # Build clean error context
        current_error = self._extract_latest_error(current_messages)
        error_history = self.state_file_manager.read_file("ERROR_HISTORY.md")

        # Include error type in context to help error_expert
        error_type_hint = "TEST FAILURE" if prev_error_type == 'test' else "COMPILATION ERROR"
        clean_messages = [
            HumanMessage(content=f"""ERROR FIX REQUIRED - Project: {project_path}

## ERROR TYPE: {error_type_hint}

## CURRENT ERROR:
{current_error}

## PREVIOUS ATTEMPTS:
{error_history if error_history else 'No previous attempts - this is your first try.'}

Do NOT repeat failed approaches. Try something different.
Analyze the error, then EXECUTE the fix using your tools.
Run mvn_compile (for compile errors) or mvn_test (for test failures) to verify it works.""")
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
            # FIX: INCREMENT error_count when error_expert fails, reset to 0 when fixed
            # Bug was: error_count stayed at 1 forever, router's "error_count >= 3" check never triggered
            "error_count": state.get("error_count", 0) + 1 if still_has_error else 0,
            "last_error_message": error_msg if still_has_error else "",
            # Reset test failure count when error is resolved
            "test_failure_count": state.get("test_failure_count", 0) if still_has_error else 0,
            "last_test_failure_task": state.get("last_test_failure_task", "") if still_has_error else "",
        }

    def _apply_phase_transition(self, state: State) -> List[BaseMessage]:
        """Apply phase transition pruning from analysis to execution"""
        log_agent("[PRUNE] FIRST EXECUTION: Applying phase transition")

        try:
            project_path = state.get("project_path", "")

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
                "stuck_intervention_active": False,
                "has_build_error": False,
                "error_count": 0,
                "last_error_message": "",
                "error_type": "none",
                "test_failure_count": 0,
                "last_test_failure_task": "",
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

            return {
                "success": final_result.get("execution_done", False),
                "result": self._get_final_content(final_result),
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

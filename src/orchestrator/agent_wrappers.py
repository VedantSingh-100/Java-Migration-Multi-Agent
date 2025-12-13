"""
Agent Wrappers for Migration Orchestrator

This module provides wrapper functions for the three migration agents:
- Analysis Agent: Analyzes project and creates migration plan
- Execution Agent: Executes migration tasks
- Error Agent: Fixes build/test errors

Each wrapper handles:
- Auto-completion detection
- Progress tracking
- External memory injection
- Phase transitions
- Error detection

IMPORTANT: These wrappers modify what messages the AGENT SEES when invoked,
but the LangGraph State uses `add_messages` reducer which ACCUMULATES messages.
This means the state grows over time even though agents see pruned context.
"""

from datetime import datetime
from collections import Counter
from typing import List, Callable, Optional

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage

from src.utils.logging_config import log_agent, log_summary, log_console
from src.utils.completion_detector import detect_analysis_complete

from .constants import MAX_EXECUTION_LOOPS_PER_PHASE, MAX_LOOPS_WITHOUT_PROGRESS
from .state import State, calculate_todo_progress
from .error_handler import unified_classifier, MavenErrorType


class AnalysisNodeWrapper:
    """
    Wrapper for analysis agent with automatic completion detection.

    Detects when analysis is complete by checking for:
    - TODO.md exists with task markers
    - CURRENT_STATE.md exists with content
    """

    def __init__(self, analysis_agent):
        """
        Args:
            analysis_agent: The analysis agent instance
        """
        self.analysis_agent = analysis_agent

    def __call__(self, state: State) -> dict:
        """
        Wrap analysis agent execution with completion detection.

        Args:
            state: Current workflow state

        Returns:
            Updated state dict
        """
        log_agent("[WRAPPER] Running analysis_expert")

        # Run the agent
        result = self.analysis_agent.invoke(state)

        # Auto-detect completion
        project_path = state.get("project_path", "")
        messages = result.get("messages", [])

        analysis_complete = detect_analysis_complete(project_path, messages)

        if analysis_complete:
            log_agent("[WRAPPER] Analysis AUTO-DETECTED as complete")
            log_summary("ANALYSIS PHASE: AUTO-COMPLETED (files created)")

        # Update state with detection result
        return {
            "messages": messages,
            "analysis_done": analysis_complete,
            "current_phase": "ANALYSIS_COMPLETE" if analysis_complete else "ANALYSIS"
        }


class ExecutionNodeWrapper:
    """
    Wrapper for execution agent with:
    - Automatic completion detection
    - Stuck loop detection
    - Phase transition message pruning
    - External memory integration
    - Progress tracking
    """

    def __init__(self, execution_agent, state_file_manager, task_manager, error_handler):
        """
        Args:
            execution_agent: The execution agent instance
            state_file_manager: StateFileManager for reading state files
            task_manager: TaskManager for task operations
            error_handler: ErrorHandler for error detection
        """
        self.execution_agent = execution_agent
        self.state_file_manager = state_file_manager
        self.task_manager = task_manager
        self.error_handler = error_handler

    def __call__(self, state: State) -> dict:
        """
        Wrap execution agent with completion and stuck detection.

        Args:
            state: Current workflow state

        Returns:
            Updated state dict
        """
        project_path = state.get("project_path", "")
        total_loops = state.get("total_execution_loops", 0) + 1

        log_agent(f"[WRAPPER] Running execution_expert (loop #{total_loops})")

        # Check if we've hit max loops per phase
        if total_loops > MAX_EXECUTION_LOOPS_PER_PHASE:
            log_agent(f"[STUCK] Max execution loops ({MAX_EXECUTION_LOOPS_PER_PHASE}) exceeded - forcing completion", "WARNING")
            log_summary(f"WARNING: Execution phase exceeded {MAX_EXECUTION_LOOPS_PER_PHASE} loops - stopping")
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
                log_summary(f"LOOP DETECTED: {stuck_reason} - agent may be stuck")
                log_console(f"Loop pattern: {stuck_reason}", "WARNING")

        # Get current messages
        current_messages = state.get("messages", [])

        # FIRST EXECUTION: Apply phase transition pruning
        if total_loops == 1 and state.get("analysis_done", False):
            current_messages = self._apply_phase_transition(state, current_messages)

        # Check if stuck intervention is needed
        stuck_intervention = state.get("stuck_intervention_active", False)
        loops_without_progress = state.get("loops_without_progress", 0)

        if stuck_intervention:
            current_messages = self._inject_intervention(current_messages, loops_without_progress)

        # Run agent with potentially modified messages
        if current_messages != state.get("messages", []):
            state_with_messages = dict(state)
            state_with_messages["messages"] = current_messages
            result = self.execution_agent.invoke(state_with_messages)
        else:
            result = self.execution_agent.invoke(state)

        # Get current TODO progress
        todo_progress = calculate_todo_progress(project_path)
        current_todo_count = todo_progress['completed']
        last_todo_count = state.get("last_todo_count", 0)

        # Detect if progress was made
        if current_todo_count > last_todo_count:
            new_loops_without_progress = 0
            log_agent(f"[PROGRESS] TODO count increased: {last_todo_count} -> {current_todo_count}")
            log_summary(f"PROGRESS: Completed {current_todo_count - last_todo_count} new TODO items")
        else:
            new_loops_without_progress = loops_without_progress + 1
            log_agent(f"[PROGRESS] No progress - loop #{new_loops_without_progress} without TODO updates")

            if new_loops_without_progress >= MAX_LOOPS_WITHOUT_PROGRESS:
                log_agent(f"[STUCK] Agent stuck for {new_loops_without_progress} loops - intervention needed!", "WARNING")
                log_summary(f"STUCK DETECTED: {new_loops_without_progress} loops without progress")

        # Auto-detect completion
        messages = result.get("messages", [])
        execution_complete, completion_reason, completion_stats = self._is_migration_complete(project_path)

        if execution_complete:
            log_agent(f"[WRAPPER] Execution COMPLETE: {completion_reason}")
            log_summary(f"EXECUTION PHASE: COMPLETED - {completion_reason}")
            log_console(f"Migration complete: {completion_stats.get('completion_percentage', 0)}% of tasks done", "SUCCESS")
        else:
            if completion_stats.get('completed_tasks', 0) > 0:
                log_agent(f"[PROGRESS] {completion_reason}")

        # Detect build errors using unified classifier (Single Source of Truth)
        error_type, error_msg = unified_classifier.classify_from_messages(messages)
        has_error = error_type != MavenErrorType.SUCCESS

        if has_error:
            log_agent(f"[WRAPPER] Build error detected in execution output (type={error_type.value})")
            log_summary(f"BUILD ERROR ({error_type.value}): {error_msg[:100]}...")

        # Track test failure count for retry logic
        prev_test_failure_count = state.get("test_failure_count", 0)
        # Get current task for tracking
        visible_tasks_content = self.state_file_manager.read_file("VISIBLE_TASKS.md")
        current_task = self.task_manager.extract_current_task(visible_tasks_content) if visible_tasks_content else ""
        last_test_failure_task = state.get("last_test_failure_task", "")

        # Test failure tracking logic - use MavenErrorType enum
        if has_error and error_type == MavenErrorType.TEST_FAILURE:
            if current_task == last_test_failure_task:
                new_test_failure_count = prev_test_failure_count + 1
                log_agent(f"[WRAPPER] Test failure on same task (count: {new_test_failure_count})")
            else:
                new_test_failure_count = 1
                log_agent(f"[WRAPPER] Test failure on new task")
            new_last_test_failure_task = current_task
        elif not has_error:
            new_test_failure_count = 0
            new_last_test_failure_task = ""
        else:
            new_test_failure_count = prev_test_failure_count
            new_last_test_failure_task = last_test_failure_task

        # Determine if intervention needed for next loop
        needs_intervention = (new_loops_without_progress >= MAX_LOOPS_WITHOUT_PROGRESS and
                            not execution_complete)

        # Update state - store error_type.value (string) for serialization
        return {
            "messages": messages,
            "analysis_done": state.get("analysis_done", False),
            "execution_done": execution_complete,
            "current_phase": "EXECUTION_COMPLETE" if execution_complete else "EXECUTION",
            "last_todo_count": current_todo_count,
            "loops_without_progress": new_loops_without_progress,
            "total_execution_loops": total_loops,
            "stuck_intervention_active": needs_intervention,
            "has_build_error": has_error,
            "error_type": error_type.value,  # Store as string for serialization
            "error_count": state.get("error_count", 0) + (1 if has_error else 0),
            "last_error_message": error_msg if has_error else "",
            "test_failure_count": new_test_failure_count,
            "last_test_failure_task": new_last_test_failure_task,
        }

    def _apply_phase_transition(self, state: State, current_messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Apply phase transition pruning when moving from analysis to execution.

        Creates a clean execution context with only a HumanMessage.
        """
        log_agent("[PRUNE] FIRST EXECUTION: Applying phase transition message pruning")

        try:
            original_message_count = len(current_messages)
            log_agent(f"[PRUNE_DETAIL] BEFORE Phase Transition: {original_message_count} messages from analysis")

            # Log message types being removed
            self._log_message_types(current_messages)

            # Read current state files
            project_path = state.get("project_path", "")
            todo_content = self.state_file_manager.read_file("TODO.md", keep_beginning=True)

            # Get restricted view of tasks (only next 3)
            visible_tasks = self.task_manager.get_visible_tasks(todo_content, max_visible=3)

            # Create VISIBLE_TASKS.md file
            self.task_manager.create_visible_tasks_file(visible_tasks)

            log_agent(f"[PHASE_TRANSITION] Created VISIBLE_TASKS.md with next 3 tasks: {visible_tasks['remaining_count']} remaining")

            # Build clean execution context - ONLY HumanMessage
            clean_messages = [
                HumanMessage(content=f"""EXECUTION PHASE START - Project: {project_path}

Analysis is complete. Your task list is in VISIBLE_TASKS.md.

WORKFLOW:
1. Read VISIBLE_TASKS.md to see the current task
2. Execute that task
3. Commit with git_commit
4. Task list auto-updates after commit
5. Repeat

Start now: read VISIBLE_TASKS.md""")
            ]

            pruned_count = original_message_count - len(clean_messages)
            log_agent(f"[PRUNE] Pruned {original_message_count} -> {len(clean_messages)} messages (removed {pruned_count})")
            log_summary(f"MESSAGE PRUNING: Removed {pruned_count} analysis messages, created clean 1-message execution context")

            return clean_messages

        except Exception as e:
            log_agent(f"[PRUNE] Error during message pruning: {str(e)}", "ERROR")
            # Fallback - append a HumanMessage
            phase_transition_msg = HumanMessage(content="""EXECUTION PHASE - Analysis is complete.

Read VISIBLE_TASKS.md and execute the CURRENT TASK.
Do NOT repeat analysis. Just execute tasks and commit.""")
            return current_messages + [phase_transition_msg]

    def _inject_intervention(self, current_messages: List[BaseMessage], loops_without_progress: int) -> List[BaseMessage]:
        """Inject stuck intervention message"""
        intervention_msg = HumanMessage(content=f"""STUCK ALERT: {loops_without_progress} loops without progress.

Read VISIBLE_TASKS.md NOW. Execute ONE task. Commit. Repeat.
Do not read TODO.md directly - only VISIBLE_TASKS.md.""")

        log_agent("[STUCK] Injecting intervention message to agent")
        log_summary(f"STUCK INTERVENTION: Informing agent of {loops_without_progress} loops without progress")

        return current_messages + [intervention_msg]

    def _log_message_types(self, messages: List[BaseMessage]):
        """Log message type statistics"""
        msg_types = {}
        tool_calls = []
        for msg in messages:
            msg_type = getattr(msg, 'type', type(msg).__name__)
            msg_types[msg_type] = msg_types.get(msg_type, 0) + 1
            if isinstance(msg, ToolMessage):
                tool_name = getattr(msg, 'name', 'unknown')
                tool_calls.append(tool_name)

        log_agent(f"[PRUNE_DETAIL] Message types: {dict(msg_types)}")
        if tool_calls:
            tool_counts = Counter(tool_calls)
            top_tools = tool_counts.most_common(5)
            log_agent(f"[PRUNE_DETAIL] Top tools called: {dict(top_tools)}")

    def _is_migration_complete(self, project_path: str) -> tuple:
        """
        Check if migration is complete based on TODO.md status.

        Returns:
            (is_complete: bool, reason: str, stats: dict)
        """
        todo_progress = calculate_todo_progress(project_path)
        completed = todo_progress['completed']
        total = todo_progress['total']
        percent = todo_progress['percent']

        if total == 0:
            return (False, "TODO.md not created yet or empty", {'completed_tasks': 0, 'total_tasks': 0})

        if percent >= 100:
            return (True, f"All {total} tasks completed",
                   {'completed_tasks': completed, 'total_tasks': total, 'completion_percentage': 100})

        return (False, f"{completed}/{total} tasks complete ({percent:.0f}%)",
               {'completed_tasks': completed, 'total_tasks': total, 'completion_percentage': percent})


class ErrorNodeWrapper:
    """
    Wrapper for error agent with:
    - Error resolution tracking
    - Duplicate error detection
    - Clean context creation
    """

    def __init__(self, error_agent, state_file_manager, error_handler):
        """
        Args:
            error_agent: The error agent instance
            state_file_manager: StateFileManager for reading state files
            error_handler: ErrorHandler for error operations
        """
        self.error_agent = error_agent
        self.state_file_manager = state_file_manager
        self.error_handler = error_handler

    def __call__(self, state: State) -> dict:
        """
        Wrap error agent with error resolution tracking.

        Args:
            state: Current workflow state

        Returns:
            Updated state dict
        """
        error_count = state.get("error_count", 0)
        last_error_msg = state.get("last_error_message", "")
        project_path = state.get("project_path", "")
        prev_error_type = state.get("error_type", "none")

        log_agent(f"[WRAPPER] Running error_expert (attempt {error_count}/3, type={prev_error_type})")
        log_summary(f"ERROR RESOLUTION: error_expert attempting to fix {prev_error_type} errors (attempt {error_count}/3)")

        # Extract current error
        current_messages = state.get("messages", [])
        current_error = self._extract_latest_error(current_messages)

        # Read error history
        error_history = self.state_file_manager.read_file("ERROR_HISTORY.md")

        # Check if exact same error already tried
        if self._error_already_attempted(current_error, error_history):
            log_agent("[ERROR] Same error signature attempted before - escalating", "WARNING")
            log_summary("ERROR: Same error attempted before - max retries reached")
            return {
                "messages": current_messages,
                "analysis_done": state.get("analysis_done", False),
                "execution_done": state.get("execution_done", False),
                "has_build_error": True,
                "error_type": prev_error_type,
                "error_count": 3,
                "last_error_message": "Duplicate error - cannot resolve",
                "test_failure_count": state.get("test_failure_count", 0),
                "last_test_failure_task": state.get("last_test_failure_task", ""),
            }

        # Apply aggressive pruning for error expert
        log_agent(f"[MEMORY] ERROR EXPERT: Applying external memory system")

        try:
            original_message_count = len(current_messages)
            log_agent(f"[PRUNE_DETAIL] BEFORE Error Expert: {original_message_count} messages from execution")

            # Include error type in context to help error_expert with specific guidance
            # Map new MavenErrorType values to appropriate guidance
            error_type_hint, verification_cmd, extra_guidance = self._get_error_guidance(prev_error_type)

            # Add mandatory web search reminder
            # Trigger on first error_expert invocation since internal loops don't increment error_count
            web_search_reminder = ""
            if error_count >= 1:
                # Build a suggested search query from the error
                error_snippet = current_error[:200] if current_error else "unknown error"
                suggested_query = self._build_search_query(error_snippet)
                web_search_reminder = f"""
⚠️ MANDATORY WEB SEARCH REQUIRED ⚠️
════════════════════════════════════════════════════
You have attempted {error_count} fixes without success.

BEFORE trying another fix, you MUST:
1. Call web_search_tool with a specific query
2. Include the error message + framework versions + "fix"

SUGGESTED QUERY:
  web_search_tool("{suggested_query}")

DO NOT skip this step. Search first, then fix.
════════════════════════════════════════════════════
"""
                log_agent(f"[WEB_SEARCH_ENFORCE] Injecting mandatory web search reminder (attempt {error_count})")

            # Build clean error context
            clean_messages = [
                HumanMessage(content=f"""ERROR FIX REQUIRED - Project: {project_path}
{web_search_reminder}
## ERROR TYPE: {error_type_hint}

## CURRENT ERROR:
{current_error}

## PREVIOUS ATTEMPTS:
{error_history if error_history else 'No previous attempts - this is your first try.'}
{extra_guidance}

Do NOT repeat failed approaches. Try something different.
{'USE web_search_tool FIRST to find the correct solution.' if error_count >= 1 else ''}

Analyze the error, then EXECUTE the fix using your tools.
Run {verification_cmd} to verify it works.""")
            ]

            pruned_count = original_message_count - len(clean_messages)
            log_agent(f"[MEMORY] Error expert: {original_message_count} -> {len(clean_messages)} messages")
            log_summary(f"MEMORY: Error expert using clean 1-message context")

            current_messages = clean_messages

        except Exception as e:
            log_agent(f"[MEMORY] Error applying external memory: {str(e)}", "ERROR")

        # Run error agent with clean context
        state_with_context = dict(state)
        state_with_context["messages"] = current_messages
        result = self.error_agent.invoke(state_with_context)

        # Check if errors are resolved using unified classifier (Single Source of Truth)
        messages = result.get("messages", [])
        new_error_type, error_msg = unified_classifier.classify_from_messages(messages)
        still_has_error = new_error_type != MavenErrorType.SUCCESS

        # Log error attempt to history
        self.error_handler.log_error_attempt(
            error=current_error,
            attempt_num=error_count,
            was_successful=not still_has_error
        )

        if still_has_error:
            log_agent(f"[WRAPPER] Build error still present after error_expert (type={new_error_type.value})")
            log_summary(f"ERROR PERSISTS ({new_error_type.value}): {error_msg[:100]}...")
        else:
            log_agent("[WRAPPER] Build error RESOLVED by error_expert")
            log_summary("ERROR RESOLVED: Build errors fixed, returning to execution")

        # Update state - store error_type.value (string) for serialization
        return {
            "messages": messages,
            "analysis_done": state.get("analysis_done", False),
            "execution_done": state.get("execution_done", False),
            "has_build_error": still_has_error,
            "error_type": new_error_type.value if still_has_error else MavenErrorType.SUCCESS.value,
            # FIX: INCREMENT error_count when error_expert fails, reset to 0 when fixed
            # Bug was: error_count stayed at 1 forever, router's "error_count >= 3" check never triggered
            "error_count": state.get("error_count", 0) + 1 if still_has_error else 0,
            "last_error_message": error_msg if still_has_error else "",
            # Reset test failure count when error is resolved
            "test_failure_count": state.get("test_failure_count", 0) if still_has_error else 0,
            "last_test_failure_task": state.get("last_test_failure_task", "") if still_has_error else "",
        }

    def _get_error_guidance(self, error_type_str: str) -> tuple:
        """
        Get error-specific guidance based on MavenErrorType value string.

        Args:
            error_type_str: The error_type value string from state

        Returns:
            (error_type_hint: str, verification_cmd: str, extra_guidance: str)
        """
        # POM/Dependency errors
        if error_type_str in ['pom_syntax_error', 'artifact_not_found', 'dependency_conflict', 'version_mismatch']:
            return (
                "POM/DEPENDENCY ERROR",
                "mvn_compile",
                """
This is a POM configuration or dependency error. Common fixes include:
- Adding missing version tags to dependencies
- Fixing XML syntax errors (unclosed tags, typos)
- Adding missing properties in <properties> section
- Fixing parent POM references
- Resolving dependency version conflicts
- Checking artifact coordinates (groupId, artifactId)

Use read_file to examine pom.xml, then use find_replace or write_file to fix the issue."""
            )

        # Test failures
        elif error_type_str == 'test_failure':
            return (
                "TEST FAILURE",
                "mvn_test",
                """
This is a test failure. Analyze the test error and fix the test or the code being tested.
Do NOT delete tests - fix them or use @Disabled with documentation if truly incompatible."""
            )

        # Migration-specific errors
        elif error_type_str == 'jakarta_migration_error':
            return (
                "JAKARTA MIGRATION ERROR",
                "mvn_compile",
                """
This is a javax to jakarta namespace migration error. Common fixes:
- Replace javax.servlet.* with jakarta.servlet.*
- Replace javax.persistence.* with jakarta.persistence.*
- Replace javax.validation.* with jakarta.validation.*
- Update import statements in affected Java files
- Check for JPA/Hibernate compatibility issues"""
            )

        elif error_type_str == 'spring_migration_error':
            return (
                "SPRING MIGRATION ERROR",
                "mvn_compile",
                """
This is a Spring framework migration error. Common fixes:
- Update deprecated Spring APIs to new equivalents
- Check Spring Boot 3.x compatibility
- Update WebMvcConfigurer implementations
- Fix ErrorController changes (getErrorPath removed in 2.3+)
- Check for removed or moved Spring classes"""
            )

        elif error_type_str == 'java_version_error':
            return (
                "JAVA VERSION ERROR",
                "mvn_compile",
                """
This is a Java version compatibility error. Common fixes:
- Update source/target version in pom.xml
- Check maven-compiler-plugin configuration
- Verify JDK compatibility with libraries
- Update deprecated Java APIs for target version"""
            )

        # Infrastructure errors (usually auto-fixed, but if they reach error_expert...)
        elif error_type_str in ['ssl_certificate', 'auth_401', 'auth_403', 'network_timeout']:
            return (
                "INFRASTRUCTURE/NETWORK ERROR",
                "mvn_compile",
                """
This is a network/infrastructure error that wasn't auto-fixed. Options:
- Check repository URLs in pom.xml
- Verify artifact coordinates are correct
- Consider adding public repository fallbacks
- Check for corporate proxy/firewall issues"""
            )

        # Runtime errors (exceptions during build)
        elif error_type_str == 'runtime_error':
            return (
                "RUNTIME ERROR",
                "mvn_compile",
                """
This is a runtime error that occurred during the build process. Common causes:
- NullPointerException in plugin execution
- ClassNotFoundException - missing class at runtime
- NoClassDefFoundError - class available at compile but not runtime
- NoSuchMethodError - method signature changed between versions
- OutOfMemoryError - increase heap size with -Xmx
- Configuration errors in plugins (resources, exec, etc.)

Steps to fix:
1. Read the full stack trace to identify the failing plugin/class
2. Check if it's a dependency version mismatch
3. Verify plugin configurations in pom.xml
4. Check if required resources/files exist"""
            )

        # Generic build failure (catch-all)
        elif error_type_str == 'generic_build_failure':
            return (
                "BUILD FAILURE",
                "mvn_compile",
                """
This is a generic build failure without a specific error category. Steps to diagnose:
1. Read the full Maven output to identify the failing phase/goal
2. Check which plugin is failing (compiler, surefire, resources, etc.)
3. Look for any error messages or exceptions in the output
4. Try running with -X flag for debug output: mvn compile -X

Common causes:
- Plugin configuration errors
- Missing required files or resources
- Environment/path issues
- Maven version incompatibilities"""
            )

        # Default: Compilation error
        else:
            return (
                "COMPILATION ERROR",
                "mvn_compile",
                """
This is a compilation error. Common fixes include:
- Adding missing imports
- Fixing type mismatches
- Adding missing method implementations
- Updating deprecated API usage
- Fixing syntax errors"""
            )

    def _extract_latest_error(self, messages: List[BaseMessage]) -> str:
        """Extract the most recent error from messages"""
        for msg in reversed(messages):
            content = ""
            if isinstance(msg, dict):
                content = str(msg.get('content', ''))
            elif hasattr(msg, 'content'):
                content = str(msg.content)

            if 'BUILD FAILURE' in content or 'ERROR' in content or 'error:' in content:
                # Extract error portion
                lines = content.split('\n')
                error_lines = [l for l in lines if 'ERROR' in l or 'error' in l.lower()]
                return '\n'.join(error_lines[:10]) if error_lines else content[:500]

        return "Unknown error"

    def _error_already_attempted(self, error: str, error_history: str) -> bool:
        """Check if this error has been attempted before"""
        if not error_history:
            return False

        # Use first 100 chars as signature
        error_signature = error[:100] if error else ""
        if error_signature:
            # Count occurrences in history
            if error_signature in error_history:
                count = error_history.count(error_signature)
                if count >= 2:
                    return True

        return False

    def _build_search_query(self, error_text: str) -> str:
        """Build a web search query from error text.

        Extracts key error terms and adds migration context for better search results.
        """
        import re

        # Extract key error patterns
        query_parts = []

        # Common Java error patterns to extract
        patterns = [
            r'(NoClassDefFoundError[:\s]+[\w\.]+)',  # NoClassDefFoundError: class.name
            r'(ClassNotFoundException[:\s]+[\w\.]+)',  # ClassNotFoundException: class.name
            r'(NoSuchMethodError[:\s]+[\w\.]+)',  # NoSuchMethodError: method
            r'(NoSuchFieldError[:\s]+[\w\.]+)',  # NoSuchFieldError: field
            r'(UnsupportedClassVersionError)',  # JDK version mismatch
            r'(javax\.\w+)',  # javax package references
            r'(jakarta\.\w+)',  # jakarta package references
            r'(cglib\.[\w\.]+)',  # CGLIB proxy issues
            r'(Spring\s*Boot\s*[\d\.]+)',  # Spring Boot version
            r'(Spring\s*Framework\s*[\d\.]+)',  # Spring Framework version
        ]

        for pattern in patterns:
            matches = re.findall(pattern, error_text, re.IGNORECASE)
            query_parts.extend(matches[:2])  # Take at most 2 matches per pattern

        # If we found specific patterns, use them
        if query_parts:
            # Deduplicate while preserving order
            seen = set()
            unique_parts = []
            for p in query_parts:
                if p.lower() not in seen:
                    seen.add(p.lower())
                    unique_parts.append(p)

            base_query = " ".join(unique_parts[:4])  # Max 4 terms
            return f"{base_query} Java migration fix solution"

        # Fallback: extract first meaningful error line
        lines = error_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) > 20 and ('Error' in line or 'Exception' in line):
                # Clean up and truncate
                clean_line = re.sub(r'[^a-zA-Z0-9\.\s]', ' ', line)
                return f"{clean_line[:80]} Java fix solution"

        # Last resort: generic query
        return "Java Spring Boot migration error fix"

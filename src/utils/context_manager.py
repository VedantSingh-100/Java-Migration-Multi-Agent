"""
Smart Context Management for Multi-Agent Systems

Implements best practices for handling large contexts in multi-agent LLM workflows:
1. Agent-specific context filtering (each agent only sees relevant messages)
2. Tool result summarization (compress large outputs)
3. Semantic fact extraction (preserve key decisions/state)
4. File-based checkpointing (offload history to structured files)
5. Token-aware pruning (trim when approaching limits)

This is superior to naive "keep last N messages" approaches.
"""

import re
import hashlib
import os
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from pathlib import Path
from src.utils.logging_config import log_agent


class ContextManager:
    """Intelligent context management for multi-agent migration workflows"""

    def __init__(self, max_recent_messages: int = 5, max_tool_output_lines: int = 15):
        """
        Args:
            max_recent_messages: Number of recent messages to keep in full
            max_tool_output_lines: Maximum lines to keep from tool outputs
        """
        self.max_recent_messages = max_recent_messages
        self.max_tool_output_lines = max_tool_output_lines

        # FILE CACHING: Cache read_file/read_pom results to prevent re-reads
        self.file_cache: Dict[str, str] = {}  # {file_path: content}
        self.cache_hits = 0
        self.cache_misses = 0

        # WEB SEARCH OFFLOADING: Track offloaded content for metrics
        self.offloaded_content_count = 0
        self.total_bytes_offloaded = 0

        # Create external storage directory
        self.storage_dir = Path("./context_storage")
        self.storage_dir.mkdir(exist_ok=True)

        # TOOL CALL DEDUPLICATION: Track seen tool calls
        self.seen_tool_calls: Set[str] = set()  # {hash of tool call}
        self.deduped_calls = 0

        # Semantic memory: key facts extracted from conversation
        self.facts = {
            "java_version": {"from": None, "to": None},
            "spring_boot_version": {"from": None, "to": None},
            "build_status": None,
            "test_status": None,
            "errors": [],
            "completed_steps": [],
            "pending_steps": []
        }

        log_agent(f"ContextManager initialized: max_recent={max_recent_messages}, file_cache enabled, deduplication enabled")

    def compress_for_agent(
        self,
        messages: List[Any],
        agent_name: str,
        state_context: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """
        Compress message history for a specific agent.

        Strategy:
        1. Extract and update semantic facts from messages
        2. Filter messages relevant to agent role
        3. Compress tool outputs
        4. Keep recent messages + compressed older context
        5. Prepend fact summary as system message

        Args:
            messages: Full message history
            agent_name: Name of the agent receiving context (analysis_expert, execution_expert, error_expert)
            state_context: Additional state from State.context dict

        Returns:
            Compressed message list suitable for agent
        """
        if not messages:
            return []

        log_agent(f"[COMPRESS] Starting compression for {agent_name} ({len(messages)} messages)")

        # Step 1: Extract facts from ALL messages (to maintain semantic memory)
        self._extract_facts_from_messages(messages)

        # Step 2: Apply file caching and deduplication
        cached_messages = self._apply_file_cache_and_dedup(messages)
        log_agent(f"[COMPRESS] After cache/dedup: {len(cached_messages)} messages (cache hits: {self.cache_hits}, deduped: {self.deduped_calls})")

        # Step 3: Filter messages by agent role
        relevant_messages = self._filter_messages_for_agent(cached_messages, agent_name)
        log_agent(f"[COMPRESS] After role filtering: {len(relevant_messages)} messages for {agent_name}")

        # Step 4: Compress tool outputs in relevant messages
        compressed_messages = self._compress_tool_outputs(relevant_messages)

        # Step 5: Apply token-aware pruning (keep recent + compress old)
        pruned_messages = self._prune_with_recency(compressed_messages)
        log_agent(f"[COMPRESS] After recency pruning: {len(pruned_messages)} messages (kept last {self.max_recent_messages})")

        # Step 6: Prepend semantic fact summary
        fact_summary = self._generate_fact_summary()
        final_messages = self._prepend_fact_summary(pruned_messages, fact_summary)

        if self.facts["java_version"]["from"] or self.facts["build_status"]:
            log_agent(f"[COMPRESS] Facts: Java {self.facts['java_version']}, Build: {self.facts['build_status']}, Test: {self.facts['test_status']}")

        return final_messages

    def _apply_file_cache_and_dedup(self, messages: List[Any]) -> List[Any]:
        """Apply file caching and tool call deduplication"""
        processed = []

        for msg in messages:
            # Check if this is a tool call we can cache/dedupe
            if self._is_tool_message(msg):
                tool_name = self._get_message_name(msg)
                content = self._get_message_content(msg)

                # FILE CACHING: Check if this is a file read
                if tool_name in ["read_file", "read_pom"]:
                    file_path = self._extract_file_path_from_content(content)
                    if file_path:
                        if file_path in self.file_cache:
                            # Use cached content instead
                            self.cache_hits += 1
                            cached_msg = self._create_cached_message(msg, self.file_cache[file_path])
                            processed.append(cached_msg)
                            continue
                        else:
                            # Store in cache
                            self.file_cache[file_path] = content
                            self.cache_misses += 1

                # DEDUPLICATION: Check if we've seen this exact tool call
                tool_hash = self._hash_tool_call(msg)
                if tool_hash in self.seen_tool_calls:
                    # Skip duplicate tool call
                    self.deduped_calls += 1
                    continue
                else:
                    self.seen_tool_calls.add(tool_hash)

            processed.append(msg)

        return processed

    def _extract_file_path_from_content(self, content: str) -> Optional[str]:
        """Extract file path from tool result content"""
        # Try to find file paths in common formats
        import re
        patterns = [
            r'/Users/[^\s]+\.\w+',
            r'/[^\s]+/pom\.xml',
            r'file_path["\']?\s*[:\=]\s*["\']([^"\']+)["\']+'
        ]
        for pattern in patterns:
            match = re.search(pattern, str(content))
            if match:
                return match.group(0) if '(' not in pattern else match.group(1)
        return None

    def _hash_tool_call(self, msg: Any) -> str:
        """Create hash of tool call for deduplication"""
        tool_name = self._get_message_name(msg)
        content = self._get_message_content(msg)

        # For read operations, hash based on tool + path
        if tool_name in ["read_file", "read_pom", "file_exists", "list_java_files"]:
            file_path = self._extract_file_path_from_content(content)
            if file_path:
                return hashlib.md5(f"{tool_name}:{file_path}".encode()).hexdigest()

        # For other tools, hash the full content (less aggressive)
        return hashlib.md5(f"{tool_name}:{content[:200]}".encode()).hexdigest()

    def _create_cached_message(self, original_msg: Any, cached_content: str) -> Any:
        """Create a message with cached content marker"""
        marked_content = f"[FROM CACHE] {cached_content[:500]}..." if len(cached_content) > 500 else f"[FROM CACHE] {cached_content}"
        return self._create_compressed_message(original_msg, marked_content)

    def _extract_facts_from_messages(self, messages: List[Any]):
        """Extract key facts from messages to maintain semantic memory"""
        for msg in messages:
            content = self._get_message_content(msg)
            if not content:
                continue

            content_str = str(content).lower()

            # Extract Java version changes
            java_match = re.search(r'java\s+(?:version\s+)?(\d+).*?(?:to|→|->)\s*(\d+)', content_str)
            if java_match:
                self.facts["java_version"]["from"] = java_match.group(1)
                self.facts["java_version"]["to"] = java_match.group(2)

            # Extract Spring Boot version
            spring_match = re.search(r'spring\s*boot\s+(?:version\s+)?(\d+\.\d+(?:\.\d+)?).*?(?:to|→|->)\s*(\d+\.\d+(?:\.\d+)?)', content_str)
            if spring_match:
                self.facts["spring_boot_version"]["from"] = spring_match.group(1)
                self.facts["spring_boot_version"]["to"] = spring_match.group(2)

            # Track build/test status
            if "mvn compile" in content_str or "maven compile" in content_str:
                if "success" in content_str or "return code: 0" in content_str:
                    self.facts["build_status"] = "SUCCESS"
                elif "fail" in content_str or "error" in content_str:
                    self.facts["build_status"] = "FAILED"

            if "mvn test" in content_str:
                if "success" in content_str or "return code: 0" in content_str:
                    self.facts["test_status"] = "SUCCESS"
                elif "fail" in content_str or "error" in content_str:
                    self.facts["test_status"] = "FAILED"

            # Extract errors (keep last 5)
            if "error:" in content_str or "exception" in content_str:
                error_lines = [line for line in content_str.split('\n') if 'error' in line or 'exception' in line]
                for error_line in error_lines[:3]:  # Max 3 errors per message
                    if error_line not in self.facts["errors"]:
                        self.facts["errors"].append(error_line[:200])  # Truncate long errors
                        if len(self.facts["errors"]) > 5:
                            self.facts["errors"].pop(0)  # Keep only last 5

    def _filter_messages_for_agent(self, messages: List[Any], agent_name: str) -> List[Any]:
        """
        Filter messages based on agent role.

        - analysis_expert: Needs initial user request, POM analysis, dependency info
        - execution_expert: Needs analysis results, migration plan, execution feedback
        - error_expert: Needs error messages, failed compilation/test results
        - supervisor: Needs high-level status from all agents
        """
        if agent_name == "supervisor":
            # Supervisor gets condensed view: only agent responses, not tool calls
            return [msg for msg in messages if self._is_agent_response(msg)]

        # For worker agents, apply role-based filtering
        relevant_types = self._get_relevant_message_types(agent_name)
        filtered = []

        for msg in messages:
            msg_type = self._classify_message(msg)

            # Always keep: user messages, current agent's messages
            if msg_type in ["user", agent_name]:
                filtered.append(msg)
                continue

            # Keep messages relevant to agent role
            if msg_type in relevant_types:
                filtered.append(msg)

        return filtered

    def _get_relevant_message_types(self, agent_name: str) -> List[str]:
        """Define which message types are relevant for each agent"""
        relevance_map = {
            "analysis_expert": [
                "pom_analysis", "dependency_info", "file_read",
                "java_version", "spring_version", "user"
            ],
            "execution_expert": [
                "analysis_expert",  # Needs analysis results
                "migration_plan", "recipe_execution", "build_result",
                "user"
            ],
            "error_expert": [
                "execution_expert",  # Needs execution feedback
                "error", "compilation_failed", "test_failed",
                "build_result", "user"
            ],
        }
        return relevance_map.get(agent_name, ["user"])

    def _classify_message(self, msg: Any) -> str:
        """Classify message type for filtering purposes"""
        content = str(self._get_message_content(msg)).lower()
        msg_name = self._get_message_name(msg)

        # Check if it's from a specific agent
        for agent in ["analysis_expert", "execution_expert", "error_expert", "supervisor"]:
            if agent in msg_name or agent in content[:100]:
                return agent

        # Check message type by content
        if "pom.xml" in content or "dependency" in content:
            return "pom_analysis"
        if "error" in content or "exception" in content or "failed" in content:
            return "error"
        if "return code:" in content or "mvn compile" in content or "mvn test" in content:
            return "build_result"
        if "java version" in content:
            return "java_version"
        if "spring boot" in content or "spring framework" in content:
            return "spring_version"
        if "recipe" in content or "openrewrite" in content:
            return "recipe_execution"

        return "other"

    def _compress_tool_outputs(self, messages: List[Any]) -> List[Any]:
        """
        Compress tool outputs to prevent token bloat.

        Strategy:
        - For Maven logs: Keep only errors, warnings, and summary
        - For file reads: Keep first/last N lines if too long
        - For list operations: Truncate to key items
        - For web searches: Extract from spans and compress with LLM
        """
        compressed = []

        for msg in messages:
            # Debug: Log message structure for first few messages
            msg_str = str(msg)
            if "Web_Search" in msg_str and len(compressed) < 3:
                log_agent(f"[DEBUG] Found Web_Search in message string, checking structure...")
                log_agent(f"[DEBUG] Message type: {type(msg)}")
                log_agent(f"[DEBUG] Has 'spans' attr: {hasattr(msg, 'spans')}")
                if isinstance(msg, dict):
                    log_agent(f"[DEBUG] Is dict, has 'spans' key: {'spans' in msg}")
                    if 'spans' in msg:
                        log_agent(f"[DEBUG] Spans value: {msg['spans'][:200] if len(str(msg['spans'])) > 200 else msg['spans']}")

            # Check for web search in spans first (priority check)
            if self._has_web_search(msg):
                log_agent(f"[DEBUG] _has_web_search returned True!")
                web_search_output = self._extract_web_search_output(msg)
                if web_search_output:
                    compressed_content = self._compress_web_search_output(web_search_output, "web_search_tool")
                    # Create compressed message with updated span outputs
                    compressed_msg = self._compress_web_search_in_message(msg, compressed_content)
                    compressed.append(compressed_msg)
                    continue

            # Regular tool message compression
            if not self._is_tool_message(msg):
                compressed.append(msg)
                continue

            content = self._get_message_content(msg)
            tool_name = self._get_message_name(msg)

            # Compress based on tool type
            if tool_name in ["mvn_compile", "mvn_test", "mvn_rewrite_run", "run_command"]:
                compressed_content = self._compress_command_output(content)
            elif tool_name in ["read_file", "read_pom"]:
                compressed_content = self._compress_file_content(content)
            elif tool_name in ["list_java_files", "list_dependencies"]:
                compressed_content = self._compress_list_output(content)
            elif tool_name in ["web_search_tool", "call_openrewrite_agent"]:
                compressed_content = self._compress_web_search_output(content, tool_name)
            else:
                # Generic compression for unknown tools (safety net)
                if len(str(content)) > 5000:
                    compressed_content = str(content)[:5000] + "\n\n[... Truncated remaining content ...]\n"
                else:
                    compressed_content = content

            # Create compressed message (preserve structure but update content)
            compressed_msg = self._create_compressed_message(msg, compressed_content)
            compressed.append(compressed_msg)

        return compressed

    def _compress_command_output(self, content: str) -> str:
        """Compress command output to essential information only"""
        if not content:
            return content

        lines = str(content).split('\n')

        # Extract key information
        return_code = None
        errors = []
        warnings = []
        summary_lines = []

        for line in lines:
            line_lower = line.lower()

            # Capture return code
            if "return code:" in line_lower:
                return_code = line
                summary_lines.append(line)

            # Capture errors
            elif any(keyword in line_lower for keyword in ["error", "exception", "failed", "failure"]):
                if len(errors) < 10:  # Keep top 10 errors
                    errors.append(line)

            # Capture warnings
            elif "warning" in line_lower:
                if len(warnings) < 3:  # Keep top 3 warnings
                    warnings.append(line)

            # Capture summary lines
            elif any(keyword in line_lower for keyword in [
                "build success", "build failure", "tests run:", "failures:", "errors:", "skipped:",
                "total time:", "finished at:"
            ]):
                summary_lines.append(line)

        # Reconstruct compressed output
        result = []
        if return_code:
            result.append(return_code)
        if summary_lines:
            result.extend(summary_lines[:5])
        if errors:
            result.append(f"\n--- Errors ({len(errors)}) ---")
            result.extend(errors[:10])
        if warnings:
            result.append(f"\n--- Warnings ({len(warnings)}) ---")
            result.extend(warnings[:3])

        compressed = "\n".join(result)

        # If compression reduced significantly, add note
        original_lines = len(lines)
        compressed_lines = len(result)
        if compressed_lines < original_lines * 0.3:  # Compressed to <30%
            compressed = f"[Compressed from {original_lines} to {compressed_lines} lines]\n" + compressed

        return compressed if compressed else content[:1000]  # Fallback: first 1000 chars

    def _compress_file_content(self, content: str) -> str:
        """Compress file content if too long"""
        if not content:
            return content

        lines = str(content).split('\n')
        max_lines = 50  # Keep max 50 lines of file content

        if len(lines) <= max_lines:
            return content

        # Keep first 25 and last 25 lines
        head = lines[:25]
        tail = lines[-25:]
        compressed = head + [f"\n... [{len(lines) - 50} lines omitted] ...\n"] + tail

        return "\n".join(compressed)

    def _compress_list_output(self, content: str) -> str:
        """Compress list outputs (file lists, dependency lists)"""
        if not content:
            return content

        lines = str(content).split('\n')
        max_items = 30  # Keep max 30 items

        if len(lines) <= max_items:
            return content

        kept_lines = lines[:max_items]
        return "\n".join(kept_lines) + f"\n... and {len(lines) - max_items} more items"

    def _compress_web_search_output(self, content: str, tool_name: str) -> str:
        """
        Intelligently compress web search results using LLM-based extraction with offloading.
        Implements industry best practices: smart summarization + reversible storage.

        Based on patterns from Factory.ai, Manus, and Anthropic for handling large web content.
        """
        if not content:
            return content

        content_str = str(content)
        content_len = len(content_str)

        # Short content doesn't need compression
        if content_len < 3000:
            return content

        log_agent(f"[WEB_COMPRESS] Starting compression for {tool_name}: {content_len:,} chars")

        # Step 1: Save full content to external storage (offloading pattern)
        storage_path = self._save_to_external_storage(content_str, tool_name)

        # Step 2: Extract key facts using LLM (smart summarization)
        if content_len > 100000:  # Very large, sample for summary
            sample = content_str[:50000]
            log_agent(f"[WEB_COMPRESS] Content too large ({content_len:,} chars), sampling first 50k for extraction")
        else:
            sample = content_str

        try:
            key_facts = self._extract_key_facts_with_llm(sample, tool_name)
        except Exception as e:
            log_agent(f"[WEB_COMPRESS] LLM extraction failed: {e}, using fallback")
            key_facts = self._rule_based_extraction(sample)

        # Step 3: Build compressed result with reversibility
        compressed = f"""[{tool_name.upper()}] - SMART COMPRESSED: {content_len:,} → {len(key_facts):,} chars]

📋 KEY FACTS EXTRACTED:
{key_facts}

📁 Full content stored at: {storage_path}
💡 Agent can use read_file('{storage_path}') to retrieve complete details if needed

[Compression: {((1 - len(key_facts)/content_len) * 100):.1f}% reduction | Reversible: ✓]""".strip()

        compressed_len = len(compressed)
        reduction_pct = ((1 - compressed_len/content_len) * 100)

        log_agent(f"[WEB_COMPRESS] ✓ Compressed {content_len:,} → {compressed_len:,} chars ({reduction_pct:.1f}% reduction)")
        log_agent(f"[WEB_COMPRESS] ✓ Full content saved to: {storage_path}")

        return compressed

    def _save_to_external_storage(self, content: str, tool_name: str) -> str:
        """Save full content to disk for potential retrieval (offloading pattern)"""
        # Generate unique ID
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{tool_name}_{timestamp}_{content_hash}.txt"

        storage_path = self.storage_dir / filename

        # Save content
        try:
            storage_path.write_text(content, encoding='utf-8')

            # Update metrics
            self.offloaded_content_count += 1
            self.total_bytes_offloaded += len(content)

            log_agent(f"[OFFLOAD] Saved {len(content):,} bytes to {storage_path.name}")
            return str(storage_path)
        except Exception as e:
            log_agent(f"[OFFLOAD] Error saving to disk: {e}")
            return f"<storage_failed: {e}>"

    def _extract_key_facts_with_llm(self, content: str, tool_name: str) -> str:
        """Use LLM to extract actionable facts from web content (smart summarization)"""
        # Tailor extraction based on tool type
        if "openrewrite" in tool_name.lower():
            focus = "OpenRewrite recipes, POM.xml configurations, Maven plugin setup, migration steps, dependency changes"
            example_output = """- Recipe: org.openrewrite.java.migrate.JavaxToJakarta
- POM changes: Add jakarta.persistence-api:3.1.0, remove javax.persistence
- Plugin: rewrite-maven-plugin:5.0.0 required
- Command: mvn rewrite:run -Drewrite.activeRecipes=..."""
        else:
            focus = "library versions, API details, migration paths, breaking changes, configuration examples"
            example_output = """- Spring Boot 3.x requires Java 17+
- javax.* → jakarta.* namespace change
- New dependency: jakarta.servlet-api:6.0.0
- Breaking change: Configuration properties renamed"""

        prompt = f"""Extract ONLY the essential actionable information from this web search result.
This is for a Java/Spring Boot migration project.

Focus on: {focus}

Web content (truncated):
{content[:5000]}

Return a concise bullet-point summary (max 400 words) with:
- Specific versions, recipes, or dependencies mentioned
- Step-by-step instructions if present
- Important warnings or breaking changes
- Code/configuration snippets if critical

Example format:
{example_output}

Be technical and precise. Omit marketing content, navigation, or irrelevant details."""

        try:
            # Use Amazon Bedrock Claude for summarization
            from langchain_aws import ChatBedrock
            import os

            # Using us. prefix for cross-region inference profile
            llm = ChatBedrock(
                model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                region_name=os.environ.get("AWS_REGION", "us-east-1"),
                model_kwargs={
                    "max_tokens": 800,  # Cap summary length
                    "temperature": 0.0,
                },
            )

            summary = llm.invoke(prompt).content

            # Hard cap at 2000 chars for safety
            if len(summary) > 2000:
                summary = summary[:2000] + "\n... [truncated for length]"

            return summary

        except Exception as e:
            log_agent(f"[EXTRACT] LLM extraction error: {e}")
            # Fallback to rule-based if LLM fails
            raise e

    def _rule_based_extraction(self, content: str) -> str:
        """Fallback: Rule-based extraction if LLM unavailable"""
        lines = content.split('\n')

        important_lines = []
        code_blocks = []

        in_code_block = False
        code_buffer = []

        for line in lines:
            lower = line.lower()

            # Track code blocks
            if '```' in line or '<code>' in line:
                in_code_block = not in_code_block
                if not in_code_block and code_buffer:
                    # End of code block
                    code_blocks.append('\n'.join(code_buffer[:10]))  # Max 10 lines per block
                    code_buffer = []
                continue

            if in_code_block:
                code_buffer.append(line)
                continue

            # Extract lines with key patterns
            if any(pattern in lower for pattern in [
                'version:', 'step', 'recipe:', 'dependency', 'pom.xml',
                'migration', 'upgrade', 'example:', 'warning:', 'error:',
                '<dependency>', '<plugin>', 'openrewrite', 'spring boot',
                'javax.', 'jakarta.', 'java 17', 'java 21',
                'breaking change', 'deprecated', 'removed in',
                'mvn ', 'maven', 'gradle', 'configuration'
            ]):
                important_lines.append(line.strip())
                if len(important_lines) >= 50:  # Cap at 50 lines
                    break

        # Combine important lines and code blocks
        result = []
        if important_lines:
            result.append("KEY INFORMATION:")
            result.extend(important_lines[:50])

        if code_blocks:
            result.append("\nCODE EXAMPLES:")
            for i, block in enumerate(code_blocks[:3], 1):  # Max 3 code blocks
                result.append(f"\nExample {i}:")
                result.append(block)

        extracted = '\n'.join(result) if result else content[:2000]

        # Cap at 2000 chars
        if len(extracted) > 2000:
            extracted = extracted[:2000] + "\n... [truncated]"

        return extracted

    def _should_preserve_message(self, msg: Any) -> bool:
        """
        Check if message contains critical state information that must NEVER be compressed.

        These patterns indicate state transitions, agent handoffs, and completion markers
        that are essential for the supervisor to maintain context across compressions.
        """
        content = str(self._get_message_content(msg)).lower()

        # Critical patterns that indicate state information
        NEVER_COMPRESS_PATTERNS = [
            "transfer_to_",            # Agent handoffs
            "transfer_back_",          # Agent returns
            "analysis complete",       # Phase completion markers
            "execution complete",
            "migration complete",
            "phase complete",
            "todo.md",                 # State file writes
            "current_state.md",
            "phase 1",                 # Phase markers
            "phase 2",
            "phase 3",
            "phase 4",
            "✅",                       # Completion markers
            "❌",                       # Error/blocked markers
            "calling analysis_expert",
            "calling execution_expert",
            "calling error_expert",
            "finished analysis",
            "finished execution",
            "blocked transfer",        # Guarded handoff blocking
            "allowed transfer",        # Guarded handoff allowing
            "cannot transfer",         # State guard messages
            "already completed",       # Completion status
            "marked as complete",      # State tracker updates
            "current_phase",           # Phase tracking
            "agent_call_count",        # Call tracking
            "analysis_complete",       # State field updates
            "execution_complete",
        ]

        # Check content for critical patterns
        for pattern in NEVER_COMPRESS_PATTERNS:
            if pattern in content:
                return True

        # Check for tool calls to transfer functions
        if isinstance(msg, dict):
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name", "") if isinstance(tool_call, dict) else getattr(tool_call, "name", "")
                    if "transfer" in str(tool_name).lower():
                        return True
        elif hasattr(msg, "tool_calls"):
            tool_calls = getattr(msg, "tool_calls", [])
            if tool_calls:
                for tool_call in tool_calls:
                    tool_name = getattr(tool_call, "name", "")
                    if "transfer" in str(tool_name).lower():
                        return True

        return False

    def _prune_with_recency(self, messages: List[Any]) -> List[Any]:
        """
        Keep recent messages in full, compress older messages.

        Strategy: Keep last N messages fully, others get compressed.
        CRITICAL: Always preserve messages with state transition information.
        """
        if len(messages) <= self.max_recent_messages:
            return messages

        # Keep recent messages
        recent = messages[-self.max_recent_messages:]

        # For older messages, keep:
        # 1. User messages and agent responses
        # 2. CRITICAL: Messages with state transitions (agent handoffs, completions)
        older = messages[:-self.max_recent_messages]
        compressed_older = [
            msg for msg in older
            if (self._is_user_message(msg) or
                self._is_agent_response(msg) or
                self._should_preserve_message(msg))  # PRESERVE CRITICAL STATE
        ]

        return compressed_older + recent

    def _generate_fact_summary(self) -> str:
        """Generate a concise summary of extracted facts"""
        summary_parts = ["=== Migration Context ==="]

        if self.facts["java_version"]["from"] and self.facts["java_version"]["to"]:
            summary_parts.append(
                f"Java: {self.facts['java_version']['from']} → {self.facts['java_version']['to']}"
            )

        if self.facts["spring_boot_version"]["from"] and self.facts["spring_boot_version"]["to"]:
            summary_parts.append(
                f"Spring Boot: {self.facts['spring_boot_version']['from']} → {self.facts['spring_boot_version']['to']}"
            )

        if self.facts["build_status"]:
            summary_parts.append(f"Build Status: {self.facts['build_status']}")

        if self.facts["test_status"]:
            summary_parts.append(f"Test Status: {self.facts['test_status']}")

        if self.facts["errors"]:
            summary_parts.append(f"\nRecent Errors ({len(self.facts['errors'])}):")
            for i, error in enumerate(self.facts["errors"][-3:], 1):  # Last 3 errors
                summary_parts.append(f"  {i}. {error[:150]}...")

        summary_parts.append("=" * 30)

        return "\n".join(summary_parts)

    def _prepend_fact_summary(self, messages: List[Any], fact_summary: str) -> List[Any]:
        """Prepend fact summary as a system message"""
        # Create a system message with facts (implementation depends on message format)
        # For now, prepend to first user message if exists
        if not messages:
            return messages

        # Find first message and prepend summary to its content
        first_msg = messages[0]
        if self._is_user_message(first_msg):
            content = self._get_message_content(first_msg)
            updated_content = f"{fact_summary}\n\n{content}"
            updated_msg = self._create_message_with_content(first_msg, updated_content)
            return [updated_msg] + messages[1:]

        return messages

    # Helper methods for message inspection
    def _is_tool_message(self, msg: Any) -> bool:
        """Check if message is a tool result"""
        if isinstance(msg, dict):
            return msg.get("type") in ["tool", "function"]
        return hasattr(msg, "type") and msg.type in ["tool", "function"]

    def _is_user_message(self, msg: Any) -> bool:
        """Check if message is from user"""
        if isinstance(msg, dict):
            return msg.get("type") in ["human", "user"]
        return hasattr(msg, "type") and msg.type in ["human", "user"]

    def _is_agent_response(self, msg: Any) -> bool:
        """Check if message is an LLM/agent response"""
        if isinstance(msg, dict):
            return msg.get("type") in ["ai", "assistant"]
        return hasattr(msg, "type") and msg.type in ["ai", "assistant"]

    def _get_message_content(self, msg: Any) -> str:
        """Extract content from message"""
        if isinstance(msg, dict):
            content = msg.get("content", "")
            return str(content) if content is not None else ""
        content = getattr(msg, "content", "")
        return str(content) if content is not None else ""

    def _get_message_name(self, msg: Any) -> str:
        """Extract name from message"""
        if isinstance(msg, dict):
            name = msg.get("name", "")
            return str(name) if name is not None else ""
        name = getattr(msg, "name", "")
        return str(name) if name is not None else ""

    def _has_web_search(self, msg: Any) -> bool:
        """Check if message contains web search results in spans"""
        if isinstance(msg, dict):
            spans = msg.get("spans", [])
            if isinstance(spans, list):
                return any(span.get("name") == "Web_Search" for span in spans)
        elif hasattr(msg, "spans"):
            spans = getattr(msg, "spans", [])
            if isinstance(spans, list):
                return any(getattr(span, "name", "") == "Web_Search" or span.get("name") == "Web_Search" for span in spans)
        return False

    def _extract_web_search_output(self, msg: Any) -> str:
        """Extract web search output from spans"""
        if isinstance(msg, dict):
            spans = msg.get("spans", [])
            for span in spans:
                if isinstance(span, dict) and span.get("name") == "Web_Search":
                    return str(span.get("outputs", ""))
        elif hasattr(msg, "spans"):
            spans = getattr(msg, "spans", [])
            for span in spans:
                span_name = getattr(span, "name", "") if hasattr(span, "name") else span.get("name", "")
                if span_name == "Web_Search":
                    outputs = getattr(span, "outputs", "") if hasattr(span, "outputs") else span.get("outputs", "")
                    return str(outputs)
        return ""

    def _compress_web_search_in_message(self, msg: Any, compressed_output: str) -> Any:
        """Update web search output in spans while preserving message structure"""
        if isinstance(msg, dict):
            compressed = msg.copy()
            if "spans" in compressed and isinstance(compressed["spans"], list):
                compressed["spans"] = [
                    {**span, "outputs": compressed_output} if span.get("name") == "Web_Search" else span
                    for span in compressed["spans"]
                ]
            return compressed
        else:
            # For objects, try to update spans
            import copy
            compressed = copy.deepcopy(msg)
            if hasattr(compressed, "spans") and isinstance(compressed.spans, list):
                for span in compressed.spans:
                    span_name = getattr(span, "name", "") if hasattr(span, "name") else span.get("name", "")
                    if span_name == "Web_Search":
                        if hasattr(span, "outputs"):
                            span.outputs = compressed_output
                        elif isinstance(span, dict):
                            span["outputs"] = compressed_output
            return compressed

    def _create_compressed_message(self, original_msg: Any, new_content: str) -> Any:
        """Create a new message with compressed content while preserving critical metadata"""
        if isinstance(original_msg, dict):
            # For dict messages, just copy everything and update content
            compressed = original_msg.copy()
            compressed["content"] = new_content
            return compressed
        else:
            # For LangChain message objects, preserve all critical fields
            msg_dict = {
                "type": getattr(original_msg, "type", "unknown"),
                "content": new_content,
            }

            # Preserve all important fields that LangGraph needs
            important_fields = ["name", "id", "tool_call_id", "tool_calls", "additional_kwargs", "response_metadata"]
            for field in important_fields:
                if hasattr(original_msg, field):
                    value = getattr(original_msg, field, None)
                    if value is not None:
                        msg_dict[field] = value

            return msg_dict

    def _create_message_with_content(self, original_msg: Any, new_content: str) -> Any:
        """Create a message with updated content"""
        return self._create_compressed_message(original_msg, new_content)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about context management"""
        return {
            "facts_extracted": {
                k: v for k, v in self.facts.items()
                if v and (not isinstance(v, (list, dict)) or len(v) > 0)
            },
            "max_recent_messages": self.max_recent_messages,
            "max_tool_output_lines": self.max_tool_output_lines,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "deduped_calls": self.deduped_calls,
            "cached_files": len(self.file_cache),
            "offloaded_content_count": self.offloaded_content_count,
            "total_bytes_offloaded": self.total_bytes_offloaded,
        }

    def get_compression_report(self) -> str:
        """Generate a detailed compression report"""
        stats = self.get_stats()
        report = [
            "=" * 60,
            "CONTEXT OPTIMIZATION REPORT",
            "=" * 60,
            f"File Cache: {self.cache_hits} hits, {self.cache_misses} misses ({len(self.file_cache)} files cached)",
            f"Deduplicated: {self.deduped_calls} duplicate tool calls",
            f"Web Content Offloaded: {self.offloaded_content_count} items ({self.total_bytes_offloaded:,} bytes)",
            f"Max Recent Messages: {self.max_recent_messages}",
            f"Max Tool Output Lines: {self.max_tool_output_lines}",
            "",
            "Facts Extracted:",
        ]

        for key, value in stats["facts_extracted"].items():
            report.append(f"  - {key}: {value}")

        report.append("=" * 60)
        return "\n".join(report)
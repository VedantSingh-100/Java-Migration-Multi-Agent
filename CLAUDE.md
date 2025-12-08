# Java Migration Framework - Agent Reference Guide

This document provides comprehensive documentation of the Java Migration codebase for any AI agent working with this project.

---

## Project Overview

**Purpose**: Automated Java ecosystem migration framework using multi-agent LLM orchestration

**Target Migrations**:
- Java 8+ → Java 21
- Spring Boot 2.x → 3.x
- Spring Framework 5.x → 6.x
- javax.* → jakarta.* namespace
- JUnit 4 → JUnit 5

**Technology Stack**: Python, LangGraph, LangChain, OpenAI API (Claude via Vertex AI)

---

## File Structure

```
/home/vhsingh/Java_Migration/
├── migrate_single_Repo.py          # CLI entry point (134 lines)
├── supervisor_orchestrator.py      # Main orchestration engine (3612 lines)
├── CLAUDE.md                       # This documentation file
├── requirements.txt                # Python dependencies (to be populated)
├── .gitignore                      # Git ignore rules (to be populated)
├── prompts/
│   ├── prompt_loader.py            # YAML prompt loading utility (95 lines)
│   ├── supervisor_exprt.yaml       # Supervisor agent prompt (12.5 KB)
│   ├── analysis_expert.yaml        # Analysis agent prompt (11.6 KB)
│   ├── execution_export.yaml       # Execution agent prompt (13 KB)
│   └── error_expert.yaml           # Error agent prompt (9.7 KB)
├── src/
│   ├── tools/                      # Migration tools (10 modules, 52+ tools)
│   │   ├── __init__.py             # Exports all_tools_flat
│   │   ├── file_operations.py      # 6 tools: read/write/search files
│   │   ├── git_operations.py       # 7 tools: git version control
│   │   ├── maven_api.py            # 15 tools: POM/dependency management
│   │   ├── command_executor.py     # 9 tools: safe command execution
│   │   ├── openrewrite_client.py   # 4 tools: recipe discovery
│   │   ├── web_search_tools.py     # 2 tools: web search & AI agents
│   │   ├── guarded_handoff.py      # 3 tools: agent handoff control
│   │   ├── completion_tools.py     # 2 tools: phase completion signals
│   │   └── state_management.py     # 4 tools: migration state queries
│   └── utils/                      # Utility modules (7 modules)
│       ├── __init__.py             # Package marker
│       ├── LLMLogger.py            # LLM interaction callback handler
│       ├── Tokencounter.py         # Token usage and cost tracking
│       ├── completion_detector.py  # Artifact-based phase detection
│       ├── context_manager.py      # Context compression & fact extraction
│       ├── logging_config.py       # Three-stream logging system
│       ├── migration_state_tracker.py  # Anti-amnesia state tracking
│       └── repo_utils.py           # Repository cloning/preparation
└── .claude/
    └── settings.local.json         # IDE permissions
```

### Core File Descriptions

| File | Purpose |
|------|---------|
| `migrate_single_Repo.py` | CLI entry point. Accepts repo name, base commit, optional CSV path. Orchestrates the entire migration. |
| `supervisor_orchestrator.py` | Core multi-agent orchestration engine using LangGraph. Contains SupervisorMigrationOrchestrator class, CircuitBreakerChatOpenAI, state management, and workflow logic. |
| `prompts/prompt_loader.py` | Utility class for loading YAML prompts with variable substitution support. |
| `prompts/*.yaml` | System prompts for each specialized agent defining their roles, tools, and behavior. |

---

## Architecture

### Multi-Agent Orchestration Pattern

```
┌─────────────────────────────────────────────────────────────────────┐
│                migrate_single_Repo.py (CLI Entry)                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│        SupervisorMigrationOrchestrator (LangGraph Workflow)         │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Circuit Breaker: MAX_LLM_CALLS = 250 (cost control)           │ │
│  │  Token Counter: Tracks prompt/response tokens & cost           │ │
│  │  Context Manager: Smart message trimming to 140K tokens        │ │
│  │  Loop Detection: Detects stuck patterns (5 loops threshold)    │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  Analysis Expert │  │ Execution Expert │  │   Error Expert   │  │
│  │                  │  │                  │  │                  │  │
│  │  - Analyze repo  │  │  - Execute tasks │  │  - Diagnose      │  │
│  │  - Create plan   │  │  - Run recipes   │  │  - Fix errors    │  │
│  │  - TODO.md       │→ │  - Update state  │→ │  - Return fixes  │  │
│  │  - analysis.md   │  │  - Commit work   │  │                  │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
│                                                                     │
│  Shared State Files:                                                │
│  - TODO.md (task checklist, created by analysis, updated by exec)  │
│  - CURRENT_STATE.md (project status, append-only)                  │
│  - COMPLETED_ACTIONS.md (action audit trail, system-managed)       │
│  - analysis.md (detailed findings from analysis phase)             │
└─────────────────────────────────────────────────────────────────────┘
```

### Agent Roles

| Agent | Role | Key Responsibilities |
|-------|------|---------------------|
| **Supervisor** | Orchestrator | Routes between agents, validates progress, coordinates handoffs |
| **Analysis Expert** | Planner | POM analysis, dependency mapping, creates TODO.md and analysis.md |
| **Execution Expert** | Implementer | Executes migration tasks, runs OpenRewrite recipes, commits changes |
| **Error Expert** | Debugger | Diagnoses compilation/test failures, applies targeted fixes |

---

## Execution Flow

### Complete Migration Workflow

```
1. CLI Entry (migrate_single_Repo.py)
   ├── Parse arguments: repo, base_commit, [--csv]
   ├── Clone repository to /Java_Migration/repositories/{repo}/
   ├── Setup logging (agent, summary, console, LLM logs)
   └── Initialize SupervisorMigrationOrchestrator

2. ANALYSIS PHASE
   ├── Analysis Expert receives migration request
   ├── Discovers all pom.xml files (find_all_poms)
   ├── Analyzes Java, Spring Boot, Spring versions
   ├── Establishes build baseline (mvn compile, mvn test)
   ├── Records baseline test count
   ├── Scans for javax.* imports, deprecated APIs
   ├── Queries OpenRewrite for applicable recipes
   └── Creates: TODO.md, CURRENT_STATE.md, analysis.md

3. EXECUTION PHASE (loops until complete)
   ├── Execution Expert reads VISIBLE_TASKS.md (next 3 tasks only)
   ├── For each task:
   │   ├── Execute tracked tool (add_openrewrite_plugin, mvn_rewrite_run, etc.)
   │   ├── Verify success (read files, mvn compile, mvn test)
   │   ├── Mark task [x] in TODO.md (requires verification)
   │   └── Commit changes to git
   └── System regenerates VISIBLE_TASKS.md with next task

4. ERROR RESOLUTION (when build/test fails)
   ├── Supervisor detects build error
   ├── Routes to Error Expert
   ├── Error Expert diagnoses root cause
   ├── Applies targeted fixes
   ├── Returns control to Execution Expert
   └── Execution Expert retries

5. VALIDATION & COMPLETION
   ├── All tasks marked [x] in TODO.md
   ├── mvn compile passes
   ├── mvn test passes (test count preserved)
   ├── MigrationReport.md generated
   └── Token usage statistics reported
```

### Phase Transitions

```
INIT → ANALYSIS → ANALYSIS_COMPLETE → EXECUTION → ERROR_RESOLUTION → SUCCESS/FAILED
```

---

## Key Constants

Located in `supervisor_orchestrator.py`:

| Constant | Value | Purpose |
|----------|-------|---------|
| `MAX_LLM_CALLS` | 250 | Circuit breaker - hard stop after 250 LLM calls per migration |
| `MAX_CONTEXT_TOKENS` | 140,000 | Maximum LLM context window size |
| `SUMMARISE_TO_TOKENS` | 30,000 | Target token count when summarizing context |
| `MAX_LOOPS_WITHOUT_PROGRESS` | 5 | Stuck detection threshold |
| `MAX_EXECUTION_LOOPS_PER_PHASE` | 30 | Hard limit on execution iterations per phase |
| `EXECUTION_WINDOW_SIZE` | 5 | Messages to keep in execution agent context |
| `ERROR_WINDOW_SIZE` | 3 | Messages to keep in error agent context |

---

## State Management

### State Files Created During Migration

| File | Created By | Purpose | Update Rules |
|------|------------|---------|--------------|
| `TODO.md` | Analysis Expert | Migration task checklist | Format: `- [ ] task` / `- [x] task`. Only mark [x] after verified work. |
| `CURRENT_STATE.md` | Analysis Expert | Project status tracking | Append-only. Contains Java version, Spring version, test counts. |
| `COMPLETED_ACTIONS.md` | System | Action audit trail | System-managed only. Agents cannot modify. |
| `analysis.md` | Analysis Expert | Detailed findings | Created once during analysis. Contains code examples, patterns. |
| `VISIBLE_TASKS.md` | System | Next 3 tasks for agent | Auto-generated from TODO.md. Agents read this, not TODO.md directly. |
| `ERROR_HISTORY.md` | System | Error tracking | Prevents infinite retry loops by tracking attempted fixes. |

### External Memory Injection

The system injects context into agent prompts at position [1]:

```
EXTERNAL MEMORY (Updated: HH:MM:SS)

COMPLETED ACTIONS (DO NOT REPEAT):
- [ACTION] Description | timestamp
...

YOUR TASKS (Next 3 only):
✔ CURRENT TASK: First unchecked task
▪ UPCOMING: Task 2, Task 3

PROGRESS: 12/45 tasks complete

CURRENT STATE:
(Migration status from CURRENT_STATE.md)
```

---

## Tool Ecosystem

### Tracked Tools (Require Verification)

These tools are monitored to verify work before allowing TODO.md updates:

| Tool | Category | Purpose |
|------|----------|---------|
| `add_openrewrite_plugin` | Migration | Add OpenRewrite Maven plugin to pom.xml |
| `configure_openrewrite_recipes` | Migration | Configure recipes in pom.xml |
| `mvn_rewrite_run` | Migration | Execute all configured OpenRewrite recipes |
| `mvn_rewrite_run_recipe` | Migration | Execute specific recipe by name |
| `update_java_version` | Migration | Update Java version in pom.xml |
| `mvn_compile` | Verification | Compile project to verify syntax |
| `mvn_test` | Verification | Run tests to verify functionality |
| `git_commit` | Version Control | Commit changes with message |
| `write_file` | File Ops | Write content to file |
| `find_replace` | File Ops | Find and replace in file |

### Agent-Specific Tool Access

**Supervisor Tools** (Read-only):
- `read_file`, `file_exists`, `list_java_files`, `read_pom`
- `check_migration_state`, `guarded_handoff`

**Analysis Expert Tools**:
- Read: `read_file`, `read_pom`, `find_all_poms`, `list_java_files`, `search_files`
- Build: `mvn_compile`, `mvn_test`, `run_command`
- Discovery: `suggest_recipes_for_java_version`, `get_available_recipes`
- External: `call_openrewrite_agent`, `web_search_tool`
- Git: All git operations
- Write: ONLY `TODO.md`, `CURRENT_STATE.md`, `analysis.md`

**Execution Expert Tools**:
- File: `read_file`, `write_file`, `find_replace`, `file_exists`
- Maven: `update_java_version`, `add_openrewrite_plugin`, `configure_openrewrite_recipes`
- OpenRewrite: `mvn_rewrite_run`, `mvn_rewrite_run_recipe`
- Git: All git operations
- Build: `mvn_compile`, `mvn_test`
- Completion: `mark_execution_complete`

**Error Expert Tools**:
- Read: `read_file`, `file_exists`
- Diagnosis: `mvn_compile`, `mvn_test`
- Git: `git_status`, `get_log`, `list_branches` (read-only)
- Note: `run_command` and `write_file` are removed to prevent risky edits

### OpenRewrite Integration

| Tool | Purpose |
|------|---------|
| `call_openrewrite_agent` | Query OpenRewrite documentation (learning only, no code changes) |
| `mvn_rewrite_run` | Execute all configured recipes (makes actual code changes) |
| `mvn_rewrite_run_recipe` | Execute specific recipe by name |
| `mvn_rewrite_dry_run` | Preview recipe changes without applying |

**Important**: `call_openrewrite_agent` is for documentation queries only. Use `mvn_rewrite_run` or `mvn_rewrite_run_recipe` for actual code changes.

---

## Critical Rules and Constraints

### Test Preservation Requirements

From `supervisor_exprt.yaml`:
- **NEVER** delete test files
- **NEVER** delete test methods
- **NEVER** rename test methods
- **ONLY** modify test implementations
- Use `@Disabled` annotation for incompatible tests (with documentation)
- Final test count **MUST** equal baseline test count

### Work Verification Before Task Completion

The system enforces:
1. Agents must use a tracked tool (actual work) before marking tasks [x]
2. System verifies tool execution within 60 seconds of TODO.md update
3. If verification fails, TODO.md update is blocked with error message

**Correct Workflow**:
```
1. DO THE WORK (use tracked migration tool)
2. VERIFY success (read files, compile, test)
3. ONLY THEN mark task [x] in TODO.md
4. Commit the change
```

### Sequential Task Execution

- Agents can only see next 3 unchecked tasks via VISIBLE_TASKS.md
- Cannot skip ahead to later tasks
- Must complete current task before proceeding
- TODO.md is redirected to VISIBLE_TASKS.md when agents try to read it

### File Protection Mechanisms

| File | Protection |
|------|------------|
| `COMPLETED_ACTIONS.md` | Write blocked - system-managed only |
| `TODO.md` | Overwrite blocked after analysis - use find_replace to mark [x] |
| `CURRENT_STATE.md` | Overwrite blocked - append-only |
| `analysis.md` | Overwrite blocked after creation |

---

## src/tools/ - Complete Tools Reference

The `src/tools/` directory contains 10 Python modules with 52+ LangChain tools for migration operations.

### file_operations.py (6 tools)

| Tool | Signature | Purpose |
|------|-----------|---------|
| `read_file` | `(file_path: str) -> str` | Read file content with UTF-8 encoding |
| `write_file` | `(file_path: str, content: str) -> str` | Write content to file |
| `find_replace` | `(file_path: str, find_text: str, replace_text: str) -> str` | Find and replace all occurrences |
| `list_java_files` | `(directory: str) -> str` | Recursively find all .java files |
| `search_files` | `(directory: str, pattern: str) -> str` | Regex search in Java files (max 20 results) |
| `file_exists` | `(file_path: str) -> str` | Check if file exists ("True"/"False") |

### git_operations.py (7 tools)

| Tool | Signature | Purpose |
|------|-----------|---------|
| `list_branches` | `(repo_path: str) -> str` | List all repository branches |
| `create_branch` | `(repo_path: str, branch_name: str) -> str` | Create new branch |
| `checkout_branch` | `(repo_path: str, branch_name: str) -> str` | Switch to branch |
| `commit_changes` | `(repo_path: str, message: str, add_all: bool = True) -> str` | Stage and commit changes |
| `tag_checkpoint` | `(repo_path: str, tag_name: str, message: str = None) -> str` | Create annotated tag |
| `get_status` | `(repo_path: str) -> str` | Get git status output |
| `get_log` | `(repo_path: str, max_entries: int = 10) -> str` | Show recent commits |

### maven_api.py (15 tools)

**POM Discovery & Reading:**

| Tool | Purpose |
|------|---------|
| `find_all_poms(project_path)` | Discover all pom.xml files, marks parent POMs with `[PARENT POM]` |
| `read_pom(project_path)` | Read root pom.xml - returns groupId, artifactId, Java version, dependency count |
| `read_pom_by_path(pom_file_path)` | Read specific pom.xml - includes modules list for parent POMs |

**Java Version Management:**

| Tool | Purpose |
|------|---------|
| `get_java_version(project_path)` | Get current Java version from root pom (defaults to "8") |
| `update_java_version(project_path, java_version)` | Update Java version in ROOT pom.xml only |
| `update_java_version_in_pom(pom_file_path, java_version)` | Update Java version in specific pom.xml |
| `update_all_poms_java_version(project_path, java_version)` | Update Java version in ALL pom.xml files (recommended for multi-module) |

**Dependency Management:**

| Tool | Purpose |
|------|---------|
| `list_dependencies(project_path)` | List all dependencies with versions and scopes |
| `get_latest_version_from_maven_central(group_id, artifact_id)` | Query Maven Central for latest version |
| `update_dependencies(project_path)` | Get latest versions for all updatable dependencies (uses Maven + LLM) |
| `get_spring_boot_latest_version()` | Get latest Spring Boot 3.x version |
| `get_spring_framework_latest_version()` | Get latest Spring Framework 6.x version |

**OpenRewrite Plugin:**

| Tool | Purpose |
|------|---------|
| `add_openrewrite_plugin(project_path)` | Add OpenRewrite Maven plugin (v6.15.0) with default Java 21 recipe |
| `configure_openrewrite_recipes(project_path, recipes: List[str])` | Configure active recipes in pom.xml |
| `add_rewrite_dependency(project_path, dependency_artifact, version)` | Add OpenRewrite recipe dependency |

### command_executor.py (9 tools)

**Build Tools:**

| Tool | Purpose |
|------|---------|
| `mvn_compile(project_path)` | Run `mvn compile -B` with automatic SSL/403 error handling |
| `mvn_test(project_path)` | Run `mvn test -B` with automatic error handling |
| `run_command(command, cwd, timeout)` | Execute safe shell commands (see Command Safety below) |

**OpenRewrite Execution:**

| Tool | Purpose |
|------|---------|
| `mvn_rewrite_run(project_path)` | Execute ALL configured recipes (MODIFIES CODE) |
| `mvn_rewrite_dry_run(project_path)` | Preview changes without applying |
| `mvn_rewrite_run_recipe(project_path, recipe_name)` | Execute specific recipe by name |

**Git Integration:**

| Tool | Purpose |
|------|---------|
| `git_status(project_path)` | Get `git status --porcelain` output |
| `git_add_all(project_path)` | Stage all changes (`git add .`) |
| `git_commit(project_path, message)` | Commit staged changes |

### openrewrite_client.py (4 tools)

| Tool | Purpose |
|------|---------|
| `get_available_recipes()` | List commonly used recipes by category (Java, Spring Boot, Jakarta, JUnit) |
| `suggest_recipes_for_java_version(current_version, target_version)` | Suggest recipes for version migration |
| `create_rewrite_config(project_path, recipes)` | **DEPRECATED** - Use `configure_openrewrite_recipes` instead |
| `remove_rewrite_config(project_path)` | Remove rewrite.yml configuration file |

### web_search_tools.py (2 tools)

| Tool | Purpose |
|------|---------|
| `web_search_tool(query)` | Search internet and get AI-summarized answers |
| `call_openrewrite_agent(command)` | Query OpenRewrite RAG agent for migration guidance (DOCUMENTATION ONLY - does not execute) |

**Important**: `call_openrewrite_agent` is for learning about recipes. Use `mvn_rewrite_run*` tools to actually execute recipes.

### guarded_handoff.py (3 tools)

These tools prevent duplicate agent calls by checking state before routing:

| Tool | Purpose |
|------|---------|
| `guarded_analysis_handoff` | Transfer to analysis_expert (blocks if already completed) |
| `guarded_execution_handoff` | Transfer to execution_expert (blocks if not ready) |
| `guarded_error_handoff` | Transfer to error_expert (can be called multiple times) |

### completion_tools.py (2 tools)

| Tool | Purpose |
|------|---------|
| `mark_analysis_complete(summary)` | Signal analysis phase is done (creates TODO.md, analysis.md) |
| `mark_execution_complete(summary)` | Signal execution phase is done (ALL tasks [x], build passes, tests pass) |

**Warning**: Only call `mark_execution_complete` when truly finished - it prevents further execution_expert calls.

### state_management.py (4 tools)

| Tool | Purpose |
|------|---------|
| `check_migration_state()` | Get current phase, next action, completion status, history |
| `can_call_analysis_expert()` | Check if analysis_expert can be called (prevents duplicates) |
| `can_call_execution_expert()` | Check if execution_expert can be called |
| `can_call_error_expert()` | Check if error_expert can be called (always True, returns call count) |

**Usage**: Call these BEFORE routing to agents to prevent amnesia loops.

---

## src/utils/ - Utility Modules Reference

The `src/utils/` directory contains 7 utility modules for logging, state tracking, and context management.

### LLMLogger.py

**Class**: `LLMLogger(BaseCallbackHandler)`

LangChain callback handler for tracking all LLM interactions.

| Method | Purpose |
|--------|---------|
| `on_llm_start(serialized, prompts)` | Log when LLM invocation begins (captures model ID and prompt) |
| `on_llm_end(response)` | Log LLM responses (handles chat and completion formats) |
| `on_llm_error(error)` | Log LLM errors to LLM log, summary log, and console |

### Tokencounter.py

**Class**: `TokenCounter(BaseCallbackHandler)`

Token usage and cost tracking using tiktoken encoder (`cl100k_base`).

**Cost Configuration**:
```python
COST_PER_MILLION_PROMPT = 3.00    # $3 per 1M input tokens
COST_PER_MILLION_RESPONSE = 15.00 # $15 per 1M output tokens
```

| Method | Purpose |
|--------|---------|
| `on_llm_start()` | Count prompt tokens, increment llm_calls |
| `on_llm_end()` | Count response tokens |
| `calculate_cost()` | Returns (prompt_cost, response_cost, total_cost) in USD |
| `get_stats()` | Returns dict with all metrics |
| `report(log_func)` | Print formatted cost report |
| `reset()` | Clear all counters |

### completion_detector.py

Artifact-based detection of agent work completion (no LLM reliance).

| Function | Purpose |
|----------|---------|
| `detect_analysis_complete(project_path, messages)` | Check TODO.md and CURRENT_STATE.md exist with 50+ chars and task markers |
| `detect_execution_complete(project_path, messages)` | Check if 80% of TODO items are marked [x] |
| `get_todo_checked_count(project_path)` | Count checked items for progress tracking |
| `get_completion_status(project_path, messages, current_phase)` | Return comprehensive status dict |

### context_manager.py

**Class**: `ContextManager`

Intelligent context management with token-aware compression, fact extraction, and web content offloading.

**Key Features**:
- File caching to avoid re-processing identical reads
- Tool call deduplication via MD5 hashing
- Web content offloading to `./context_storage/`
- Semantic fact extraction (Java version, Spring version, build status, errors)

| Method | Purpose |
|--------|---------|
| `compress_for_agent(messages, agent_name, state_context)` | Main entry - 6-step compression pipeline |
| `_extract_facts_from_messages(messages)` | Regex-based extraction of migration facts |
| `_filter_messages_for_agent(messages, agent_name)` | Role-based message filtering |
| `_compress_tool_outputs(messages)` | Reduce token bloat in tool results |
| `_compress_web_search_output(content, tool_name)` | Save to disk, extract key facts via LLM |
| `_prune_with_recency(messages)` | Keep recent N messages, compress older |
| `get_compression_report()` | Generate detailed optimization metrics |

### logging_config.py

Three-stream structured logging using loguru.

| Function | Purpose |
|----------|---------|
| `setup_migration_logging(repo_name)` | Initialize logging - creates `logs/{repo}/` with 3 log files |
| `log_llm(message, level)` | Log LLM prompts/responses to `llm_interactions_{timestamp}.log` |
| `log_agent(message, level)` | Log agent events to `multiagent_process_{timestamp}.log` |
| `log_summary(message, level)` | Log high-level decisions to `summary_{timestamp}.log` |
| `log_console(message, level)` | Log to console only (no file) |

### migration_state_tracker.py

**Class**: `MigrationStateTracker`

Maintains authoritative state outside conversation to prevent "amnesia loops".

**State Tracked**:
```python
{
    "analysis_expert_called": bool,
    "analysis_expert_completed": bool,
    "execution_expert_called": bool,
    "execution_expert_completed": bool,
    "error_expert_call_count": int,
    "current_phase": str,  # INIT -> ANALYSIS -> EXECUTION -> ERROR_RECOVERY -> COMPLETE
    "phases_completed": list,
    "agent_call_history": list,  # [(timestamp, agent, action, details)]
    "duplicate_calls_prevented": int,
}
```

| Method | Purpose |
|--------|---------|
| `mark_agent_called(agent_name, details)` | Record agent invocation, update phase |
| `mark_agent_completed(agent_name, details)` | Mark agent finished, add to phases_completed |
| `can_call_agent(agent_name)` | **GATE FUNCTION** - returns (bool, reason) |
| `get_next_action()` | Returns CALL_ANALYSIS_EXPERT, CALL_EXECUTION_EXPERT, etc. |
| `get_state_summary()` | Full state snapshot with duration and history |

### repo_utils.py

Repository preparation utilities.

| Function | Purpose |
|----------|---------|
| `clone_and_prepare_repo(github_repo, base_commit, dest_dir, branch_name)` | Clone repo, checkout commit, create branch, create TODO.md and CURRENT_STATE.md |

---

## Command Safety System

The `command_executor.py` module implements strict command safety controls.

### Blocked Commands (NEVER executed)

```
rm, del, rmdir, unlink, format, fdisk, mkfs, sudo, su, doas,
chmod, chown, chgrp, kill, killall, pkill, shutdown, reboot, halt,
dd, shred, mount, umount, crontab, systemctl, service,
apt, yum, dnf, pacman, pip, npm, gem, cargo,
wget, curl, ssh, scp, ftp, telnet, nc, nmap, docker, podman
```

### Blocked Patterns

- `rm -rf` anywhere in command
- Device file writes (`>/dev/`)
- System config writes
- Command substitution (`$()`, backticks)
- Shell piping (`| sh`, `| bash`)

### Allowed Commands

```
mvn, maven, git, java, javac, ls, dir, cat, type, grep, find, findstr,
echo, pwd, cd, mkdir, cp, copy, head, tail, wc, sort, uniq, which, where, env, printenv
```

### Git Safety Restrictions

- Blocks: `clean -fd`, `reset --hard`, `push --force`
- Copy restrictions: Cannot copy to /etc/, /usr/, /var/, /sys/, /proc/

### Maven Error Handling

The system automatically handles:
- **SSL Errors**: Bypass verification, cache deps, run offline
- **403/401 Errors**: Add public repository fallbacks, cache deps, run offline
- **404/Missing**: Report to agent for resolution

---

## External Dependencies

### Python Packages Required

| Package | Purpose |
|---------|---------|
| `langchain` | LLM agent framework |
| `langchain_core` | Core LangChain components |
| `langchain_openai` | OpenAI/Vertex AI integration |
| `langgraph` | Multi-agent workflow orchestration |
| `pydantic` | Data validation and settings management |
| `tiktoken` | Token counting (cl100k_base encoder) |
| `pandas` | CSV file handling for migration tracking |
| `loguru` | Structured logging |
| `pyyaml` | YAML prompt file loading |
| `GitPython` | Git repository operations |
| `requests` | HTTP requests (Maven Central API) |
| `openai` | OpenAI client (via Vertex AI) |

### External Services

| Service | Integration |
|---------|-------------|
| **OpenAI API** | LLM calls via ChatOpenAI (default: Claude 3.7 Sonnet via Vertex AI) |
| **BNY Mellon Eliza** | Session management and JWT authentication |
| **Git/Bitbucket** | Repository operations |
| **Maven Central** | Latest dependency version queries |
| **OpenRewrite** | Recipe documentation and execution |

---

## Circuit Breaker and Cost Control

### CircuitBreakerChatOpenAI

Wraps the OpenAI client to enforce LLM call limits:

```python
class CircuitBreakerChatOpenAI(ChatOpenAI):
    def _generate(self, *args, **kwargs):
        if self._token_counter.llm_calls >= self._max_calls:
            raise LLMCallLimitExceeded(...)
        return super()._generate(...)
```

- Checked **BEFORE** each LLM call
- Raises `LLMCallLimitExceeded` exception for graceful shutdown
- Default limit: 250 calls per migration

### Token Usage Reporting

At migration end, outputs:
```
TOKEN USAGE & COST REPORT
=========================
LLM Calls:       127
Prompt tokens:   45,321
Response tokens: 23,456
Total tokens:    68,777
Total cost:      $0.2345
```

---

## Loop and Stuck Detection

### Detection Mechanisms

1. **Progress Tracking**: Monitors TODO.md completed/total tasks
2. **Action Window**: Tracks last 5 actions for patterns
3. **Stuck Detection**: Triggers if no progress after 5 loops

### Detection Patterns

- Same tool called 3+ times consecutively
- No completed actions in recent window
- Same TODO item attempted repeatedly

### Intervention

- `MAX_LOOPS_WITHOUT_PROGRESS = 5`: Alert threshold
- `MAX_EXECUTION_LOOPS_PER_PHASE = 30`: Hard stop
- Stuck detection triggers supervisory intervention

---

## Usage

### Command Line

```bash
python migrate_single_Repo.py <repo_name> <base_commit> [--csv <path>]
```

**Arguments**:
- `repo_name`: Name of repository to migrate
- `base_commit`: Git commit hash to start from
- `--csv`: Optional path to CSV tracking file

### Example

```bash
python migrate_single_Repo.py my-java-app abc123def --csv /path/to/tracking.csv
```

### Output

- Repositories cloned to: `/Java_Migration/repositories/{repo_name}/`
- Logs written to configured log directory
- CSV updated with migration status (attempted → migrated)

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `LLMCallLimitExceeded` | Exceeded 250 LLM calls | Migration too complex. Review TODO.md for optimization. |
| Stuck in execution loop | Same task failing repeatedly | Check ERROR_HISTORY.md. May need manual intervention. |
| TODO.md update blocked | No tracked tool executed | Ensure actual migration work before marking complete. |
| Build failures persisting | Error Expert can't fix | Check error patterns. May need recipe adjustment. |

### Debug Checklist

1. Check `COMPLETED_ACTIONS.md` for action history
2. Review `ERROR_HISTORY.md` for repeated errors
3. Examine TODO.md progress (completed vs total)
4. Review agent logs for detailed decisions
5. Check token usage if approaching limits

### Manual Recovery

If migration gets stuck:
1. Review state files (TODO.md, CURRENT_STATE.md)
2. Check git log for recent commits
3. Consider resetting to last known good commit
4. Simplify TODO.md tasks if too granular
5. Manually fix blocking error and restart

---

## Success Criteria

A migration is successful when:
- [ ] Java version 21 in all pom.xml files
- [ ] Spring Boot 3.x+ (latest stable)
- [ ] Spring Framework 6.x+ (compatible with Spring Boot 3.x)
- [ ] JUnit 5.x with tests migrated
- [ ] `mvn compile` passes
- [ ] `mvn test` passes
- [ ] Test count unchanged from baseline
- [ ] No deprecated API warnings
- [ ] MigrationReport.md created

---

## Key Code Locations

| Component | File | Lines (approx) |
|-----------|------|----------------|
| CLI entry point | `migrate_single_Repo.py` | 1-134 |
| CircuitBreakerChatOpenAI | `supervisor_orchestrator.py` | 22-52 |
| State class | `supervisor_orchestrator.py` | 200-240 |
| SupervisorMigrationOrchestrator | `supervisor_orchestrator.py` | 250-3600 |
| Tool wrapping/protection | `supervisor_orchestrator.py` | 963-1121 |
| External memory injection | `supervisor_orchestrator.py` | 1807-1933 |
| Loop detection | `supervisor_orchestrator.py` | 1935-1975 |
| Build error detection | `supervisor_orchestrator.py` | 2221-2241 |
| Error deduplication | `supervisor_orchestrator.py` | 2096-2107 |
| Phase routing | `supervisor_orchestrator.py` | 2500-2700 |
| Prompt loading | `prompts/prompt_loader.py` | 1-95 |

---

*This documentation was generated to help AI agents understand and work with the Java Migration framework effectively.*

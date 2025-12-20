# Multi-Agent Orchestration Problems

## Critical Error: "Received multiple non-consecutive system messages"

**Location:** Phase transition from analysis → execution (supervisor_orchestrator.py:2359-2425)

**Root Cause:**
The code creates messages in this order:
```python
current_messages = [
    HumanMessage(content="..."),      # Position 0
    SystemMessage(content="..."),      # Position 1 - TASKS
    SystemMessage(content="..."),      # Position 2 - COMPLETED_ACTIONS
    SystemMessage(content="..."),      # Position 3 - CURRENT_STATE
    SystemMessage(content="...")       # Position 4 - EXECUTION INSTRUCTIONS
]
```

**Problem:** Anthropic's API requires ALL SystemMessages to be at the START of the conversation, consecutively. Here, a HumanMessage comes first, then 4 SystemMessages follow - violating the API contract.

**Why it worked with BNY:** The BNY system used an OpenAI-compatible proxy (bnym_eliza) that internally transformed messages before sending to Claude. This proxy likely reordered or merged system messages.

**Fix:** Either:
1. Move all SystemMessage content to a single SystemMessage at position 0
2. Convert SystemMessages to HumanMessages (less ideal)
3. Merge all context into the existing system prompt via `_create_prompt_with_trimming`

---

## Bug #1: Checkbox Parsing Inconsistency

**Location:** supervisor_orchestrator.py:640, 1647

**Code:**
```python
# Line 640
if '- [x]' in line.lower() or '- [X]' in line:

# Line 1647
if '- [x]' in line.lower():
```

**Problem:**
- First check: `line.lower()` converts everything to lowercase, so `'- [X]' in line` is redundant
- Second check: Only checks lowercase, missing `[X]` case
- Inconsistent logic between locations

**Impact:** Tasks may be incorrectly marked as complete/incomplete depending on case used in markdown.

---

## Bug #2: Missing `_log_action_to_file()` Function

**Location:** supervisor_orchestrator.py:1301

**Code:**
```python
self._log_action_to_file(tool_name, args, "SUCCESS", result)  # Line 1301
```

**Problem:** The function `_log_action_to_file()` is called but never defined in the class.

**Impact:** Runtime AttributeError when trying to log actions.

---

## Bug #3: Task Extraction Fragility

**Location:** supervisor_orchestrator.py:1698-1720

**Code:**
```python
def _extract_current_task(self, todo_content: str) -> str:
    lines = todo_content.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith('- [ ]'):  # Unchecked task
            task = line.strip()[5:].strip()  # Remove "- [ ]" prefix
            ...
```

**Problems:**
1. Assumes exactly `- [ ]` format (5 chars) - breaks with `- [x]` or spacing variations
2. No handling for numbered tasks, different markdown formats
3. Silent failure returns empty string, masking issues

**Impact:** Current task extraction fails silently, execution agent gets no tasks.

---

## Bug #4: VISIBLE_TASKS.md Only Updates After git_commit

**Location:** supervisor_orchestrator.py (VISIBLE_TASKS update logic)

**Problem:** The sliding window of visible tasks only refreshes when `git_commit` is detected. If an agent:
- Completes a task without committing
- Uses a different commit method
- Fails mid-task

...the VISIBLE_TASKS.md becomes stale, showing already-completed tasks.

**Impact:** Agent cherry-picks tasks, loses sequential progress, or repeats completed work.

---

## Bug #5: Context Manager Doesn't Populate State

**Location:** src/utils/context_manager.py

**Problem:** The ContextManager has fields for `completed_steps` and `pending_steps` but they are never populated from the actual TODO.md/COMPLETED_ACTIONS.md files.

**Impact:** External memory injection is incomplete - agents don't get accurate state information.

---

## Bug #6: File Overwrite Risk in COMPLETED_ACTIONS.md

**Location:** supervisor_orchestrator.py (COMPLETED_ACTIONS writes)

**Code:**
```python
with open(completed_file, 'w') as f:  # Opens in 'w' mode
    f.write(content)
```

**Problem:** Using `'w'` mode overwrites the entire file. If there's a crash or concurrent access, previously logged actions could be lost.

**Recommended:** Use `'a'` (append) mode for logging actions, with periodic cleanup.

---

## Bug #7: Phase Awareness Lost During Context Trimming

**Location:** supervisor_orchestrator.py:271 (`_create_prompt_with_trimming`)

**Problem:** When message history exceeds `max_messages=30`, older messages are trimmed. This can remove:
- Phase transition markers
- Important context about completed work
- State information that prevents task repetition

**Impact:** Agent "forgets" what phase it's in, may repeat analysis during execution phase.

---

---

## Bug #8: TODO.md Truncation From Wrong End (NEW - CRITICAL)

**Location:** supervisor_orchestrator.py:1601-1607 (`_read_state_file`)

**Code:**
```python
if len(content) > MAX_SUMMARY_LENGTH:
    return content[-MAX_SUMMARY_LENGTH:]  # Returns LAST 2000 chars!
```

**Problem:** When TODO.md exceeds 2000 chars (common - analysis creates ~7000 char files), the function returns the LAST 2000 characters. This means `_get_visible_tasks()` only sees tasks from Phase 9/10 instead of Phase 1.

**Impact:** VISIBLE_TASKS.md shows tasks from the END of TODO.md instead of the BEGINNING. Agent works on wrong tasks.

**Fix Applied:** Added `keep_beginning` parameter to `_read_state_file()`. TODO.md calls now use `keep_beginning=True`.

---

## Summary: Priority Fixes

| Priority | Bug | Severity | Effort | Status |
|----------|-----|----------|--------|--------|
| P0 | Non-consecutive system messages (loop 1) | Blocking | Medium | FIXED |
| P0 | Non-consecutive system messages (loop 2+) | Blocking | Medium | FIXED |
| P0 | TODO.md truncation from wrong end | Blocking | Low | FIXED |
| P0 | Commit tool name mismatch (commit_changes vs git_commit) | Blocking | Low | FIXED |
| P0 | TODO.md writes not blocked (find_replace) | Blocking | Low | FIXED |
| P1 | Missing `_log_action_to_file()` | Crash | Low | FIXED |
| P1 | VISIBLE_TASKS.md staleness | Major | Medium | FIXED (AUTO_SYNC now triggers on both commit tools) |
| P1 | COMPLETED_ACTIONS.md useless format | Major | Medium | FIXED (Added TASK COMPLETIONS section) |
| P1 | Agent knows about TODO.md | Major | Medium | FIXED (execution_expert.yaml rewritten) |
| P2 | Checkbox parsing inconsistency | Moderate | Low | Open |
| P2 | Task extraction fragility | Moderate | Low | Open |
| P3 | Context manager state | Minor | Medium | Open |
| P3 | File overwrite risk | Minor | Low | Open |

---

## Fixes Applied (2025-12-07)

### Fix 1: Commit Tool Detection
- Added `commit_changes` to `TRACKED_TOOLS` constant
- Added `COMMIT_TOOLS = {'git_commit', 'commit_changes'}` set
- AUTO_SYNC now triggers on BOTH commit tools

### Fix 2: TODO.md Complete Block
- Changed PROTECTION 2 to completely block ALL TODO.md access (write_file AND find_replace)
- Returns clear message explaining the deterministic workflow
- Removed old PROTECTION 4 (was allowing writes with verification)

### Fix 3: COMPLETED_ACTIONS.md Redesign
- Added new `_log_task_completion()` method for TASK-level logging
- Added `=== TASK COMPLETIONS ===` section in header
- Header now shows `=== CURRENT TASK ===` section
- Task completions preserved across header updates
- AUTO_SYNC now logs task completions with commit hashes

### Fix 4: execution_expert.yaml Rewrite
- Removed all TODO.md references (40+ occurrences)
- Added clear TASK MANAGEMENT section at top
- Emphasized VISIBLE_TASKS.md as single source of truth
- Emphasized commit triggers auto-progress
- Added CRITICAL RULES section

### Fix 5: Streamlined Execution Agent Toolset
- REMOVED `check_migration_state` (supervisor tool, wrong agent)
- REMOVED discovery tools that were causing re-analysis loops:
  - `find_all_poms`, `list_java_files`, `search_files` (discovery)
  - `get_java_version`, `list_dependencies` (can use read_file)
- ADDED `read_pom` (for targeted reads before modifications)
- Execution agent now has focused toolset for execution only
- Analysis is done once by analysis_expert, not repeated

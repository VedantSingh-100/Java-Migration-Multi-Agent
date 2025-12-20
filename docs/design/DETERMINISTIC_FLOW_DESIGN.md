# Deterministic Task Flow Design

## Current Problems

### Problem 1: Agent Knows About TODO.md
- Execution prompt mentions TODO.md
- Agent actively tries to read/modify TODO.md
- Agent marks tasks in BOTH TODO.md and VISIBLE_TASKS.md inconsistently

### Problem 2: Commit Tool Name Mismatch
- `git_operations.py` has `commit_changes`
- `command_executor.py` has `git_commit`
- AUTO_SYNC checks for `git_commit` but agent uses `commit_changes`
- Result: AUTO_SYNC never triggers

### Problem 3: TODO.md Writes Not Blocked
- `read_file("TODO.md")` → Intercepted (returns VISIBLE_TASKS.md)
- `find_replace("TODO.md", ...)` → NOT intercepted
- Agent can directly modify TODO.md

### Problem 4: COMPLETED_ACTIONS.md is Useless
- Logs tool-level info: `"mvn_compile: FAILED (0.0s)"`
- Doesn't log task-level info: `"TASK COMPLETED: Establish build baseline"`
- Agent can't understand what was accomplished

### Problem 5: VISIBLE_TASKS.md Never Auto-Updates
- Since AUTO_SYNC doesn't trigger, VISIBLE_TASKS.md is stale
- Agent sees same task repeatedly
- Agent repeats work → infinite loop

---

## Target Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AGENT VIEW                                │
│                                                                  │
│  ┌──────────────────────┐    ┌───────────────────────────────┐  │
│  │   VISIBLE_TASKS.md   │    │    COMPLETED_ACTIONS.md       │  │
│  │   (Read-Only View)   │    │    (External Memory)          │  │
│  │                      │    │                               │  │
│  │  CURRENT TASK:       │    │  === PHASE: EXECUTION ===     │  │
│  │  - [ ] Run mvn test  │    │  COMPLETED TASKS:             │  │
│  │                      │    │  [x] Created migration branch │  │
│  │  UPCOMING:           │    │  [x] Build baseline - SUCCESS │  │
│  │  1. Add OpenRewrite  │    │                               │  │
│  │  2. Run migration    │    │  CURRENT: Run mvn test        │  │
│  │                      │    │                               │  │
│  │  Progress: 2/20      │    │  DO NOT REPEAT ABOVE TASKS    │  │
│  └──────────────────────┘    └───────────────────────────────┘  │
│                                                                  │
│  Agent can ONLY:                                                 │
│  - Read VISIBLE_TASKS.md                                         │
│  - Read COMPLETED_ACTIONS.md                                     │
│  - Execute tasks using tools                                     │
│  - Call commit_changes when done                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ commit_changes succeeds
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SYSTEM (DETERMINISTIC)                        │
│                                                                  │
│  1. Detect commit success                                        │
│  2. Read VISIBLE_TASKS.md → get current task                     │
│  3. Mark task [x] in TODO.md (system file)                       │
│  4. Append to COMPLETED_ACTIONS.md                               │
│  5. Regenerate VISIBLE_TASKS.md with next 3 tasks                │
│  6. Return control to agent                                      │
│                                                                  │
│  TODO.md is SYSTEM-ONLY - agent never sees or touches it         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Changes

### Change 1: Fix Commit Tool Detection

**File:** `supervisor_orchestrator.py`

**Current (BROKEN):**
```python
if tool_name == 'git_commit' and is_success:
```

**Fixed:**
```python
# Handle BOTH commit tools
COMMIT_TOOLS = {'git_commit', 'commit_changes'}
if tool_name in COMMIT_TOOLS and is_success:
```

**Also update TRACKED_TOOLS:**
```python
TRACKED_TOOLS = {
    ...
    'git_commit': 'COMMIT',
    'commit_changes': 'COMMIT',  # ADD THIS
    ...
}
```

---

### Change 2: Block ALL TODO.md Access

**File:** `supervisor_orchestrator.py` in `_wrap_tool_with_tracking`

**Current:** Only `read_file` is intercepted

**Fixed:** Block read_file, write_file, find_replace for TODO.md

```python
# Inside _wrap_tool_with_tracking, add to file protection:

PROTECTED_FILES = {'TODO.md'}  # Agent cannot access these

# For read_file
if tool_name == 'read_file':
    file_path = args[0] if args else kwargs.get('file_path', '')
    if any(protected in file_path for protected in PROTECTED_FILES):
        # Redirect to VISIBLE_TASKS.md
        if 'TODO.md' in file_path:
            new_path = file_path.replace('TODO.md', 'VISIBLE_TASKS.md')
            log_agent(f"[PROTECT] Redirecting TODO.md read to VISIBLE_TASKS.md")
            if args:
                args = (new_path,) + args[1:]
            else:
                kwargs['file_path'] = new_path

# For write_file and find_replace - BLOCK completely
if tool_name in ('write_file', 'find_replace'):
    file_path = args[0] if args else kwargs.get('file_path', '')
    if any(protected in file_path for protected in PROTECTED_FILES):
        return (
            f"BLOCKED: Cannot modify {file_path}. "
            f"This is a system-managed file. "
            f"Complete your task and commit - progress is tracked automatically."
        )
```

---

### Change 3: Redesign COMPLETED_ACTIONS.md Format

**New Format:**
```markdown
=== MIGRATION STATE ===
PHASE: EXECUTION
STARTED: 2025-12-07 17:00:00
LAST_UPDATE: 2025-12-07 17:15:00

=== COMPLETED TASKS ===
[x] 1. Create migration branch (commit: abc1234)
[x] 2. Establish build baseline - mvn compile SUCCESS
[x] 3. Establish test baseline - mvn test SUCCESS (1 test passed)

=== CURRENT TASK ===
[ ] 4. Add OpenRewrite plugin to pom.xml

=== TASK HISTORY ===
[17:05:00] Task 1 completed - Created branch 'migration-java21'
[17:10:00] Task 2 completed - Build SUCCESS
[17:15:00] Task 3 completed - Tests: 1 passed, 0 failed

=== IMPORTANT ===
- DO NOT repeat completed tasks
- System tracks progress automatically after each commit
- Focus on CURRENT TASK only
```

**Implementation:**
```python
def _update_completed_actions_structured(self, completed_task: str, commit_hash: str = None):
    """Update COMPLETED_ACTIONS.md with structured task completion info"""
    if not self.project_path:
        return

    filepath = os.path.join(self.project_path, "COMPLETED_ACTIONS.md")
    timestamp = datetime.now().strftime("%H:%M:%S")

    # Read current content
    current_content = ""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            current_content = f.read()

    # Parse and update sections
    # ... build structured content ...

    # Write back
    with open(filepath, 'w') as f:
        f.write(new_content)
```

---

### Change 4: Deterministic Progress Tracking

**File:** `supervisor_orchestrator.py` in `_wrap_tool_with_tracking`

**After commit success:**
```python
COMMIT_TOOLS = {'git_commit', 'commit_changes'}

if tool_name in COMMIT_TOOLS and is_success:
    log_agent(f"[AUTO_SYNC] Commit detected - updating progress")

    # Step 1: Get current task from VISIBLE_TASKS.md
    current_task = self._get_current_visible_task()

    if current_task:
        # Step 2: Mark in TODO.md (system operation)
        self._mark_task_complete_in_todo(current_task)

        # Step 3: Update COMPLETED_ACTIONS.md (structured)
        self._append_completed_task(current_task, commit_hash)

        # Step 4: Regenerate VISIBLE_TASKS.md
        self._regenerate_visible_tasks()

        log_agent(f"[AUTO_SYNC] Task marked complete: {current_task[:50]}...")
        log_agent(f"[AUTO_SYNC] VISIBLE_TASKS.md updated with next task")
```

---

### Change 5: Update Execution Expert Prompt

**File:** `prompts/execution_expert.yaml`

**Remove all references to TODO.md. New prompt structure:**

```yaml
execution_expert_prompt: |
  You are a Java Migration Execution Expert.

  YOUR WORKFLOW:
  1. Read VISIBLE_TASKS.md to see your CURRENT TASK
  2. Execute that task using your available tools
  3. When done, commit your changes with commit_changes
  4. The system automatically tracks progress
  5. Read VISIBLE_TASKS.md again for the next task

  IMPORTANT:
  - ONLY use VISIBLE_TASKS.md for task information
  - NEVER try to read or modify TODO.md (it doesn't exist for you)
  - After committing, VISIBLE_TASKS.md auto-updates with next task
  - Check COMPLETED_ACTIONS.md to see what was already done

  YOUR AVAILABLE TOOLS:
  - read_file, write_file, find_replace (for code changes)
  - mvn_compile, mvn_test (for validation)
  - commit_changes (to save progress - triggers auto-update)
  - run_command (for shell commands)

  DO NOT:
  - Try to mark tasks as complete manually
  - Access any file named TODO.md
  - Repeat work shown in COMPLETED_ACTIONS.md
```

---

### Change 6: Remove Duplicate Commit Tool

**Option A:** Keep only `commit_changes` (recommended)
- Remove `git_commit` from command_executor.py
- Update all references to use `commit_changes`

**Option B:** Keep both but handle in AUTO_SYNC
- Add `commit_changes` to COMMIT_TOOLS set
- Both tools trigger AUTO_SYNC

---

## Execution Order

1. **Fix commit detection** (supervisor_orchestrator.py:1314)
   - Add `commit_changes` to COMMIT_TOOLS check

2. **Block TODO.md writes** (supervisor_orchestrator.py ~line 975)
   - Intercept find_replace and write_file for TODO.md

3. **Update TRACKED_TOOLS** (supervisor_orchestrator.py:121)
   - Add `'commit_changes': 'COMMIT'`

4. **Redesign COMPLETED_ACTIONS.md**
   - Create `_update_completed_actions_structured()` method
   - Replace current `_log_action_to_file()` calls

5. **Update execution_expert.yaml**
   - Remove all TODO.md references
   - Emphasize VISIBLE_TASKS.md only workflow

6. **Test the flow**
   - Agent reads VISIBLE_TASKS.md
   - Agent executes task
   - Agent commits
   - System auto-updates everything
   - Agent sees next task

---

## Success Criteria

- [ ] Agent never mentions TODO.md
- [ ] Agent only reads VISIBLE_TASKS.md for tasks
- [ ] After commit, VISIBLE_TASKS.md shows next task automatically
- [ ] COMPLETED_ACTIONS.md shows clear task completion history
- [ ] No more "stuck" loops from repeated tasks
- [ ] Progress tracking is 100% deterministic (no agent involvement)

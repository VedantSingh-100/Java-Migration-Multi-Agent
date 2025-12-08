# Context Architecture Specification

**Status: IMPLEMENTED (2025-12-08)**

## Goal
Replace message accumulation with compiled context views. Each agent call gets fresh, minimal context compiled from state files.

## Current Problem
```
Loop 1:  1 msg  → Agent → 3 msgs
Loop 10: 19 msgs → Agent → 21 msgs
Loop 20: 39 msgs → PRUNE → loses initial instruction
```
Messages accumulate via `add_messages` reducer, causing context bloat and amnesia after pruning.

## Target Architecture
```
Loop 1:  [CompiledContext] → Agent → updates state files
Loop 10: [CompiledContext] → Agent → updates state files  (SAME SIZE)
Loop 20: [CompiledContext] → Agent → updates state files  (SAME SIZE)
```
State files are source of truth. Context is compiled fresh each call.

## Execution Agent Context (Target: ~2K tokens)

```
EXECUTION LOOP #N - Project: {path}

CURRENT TASK:
{from VISIBLE_TASKS.md - single task}

COMPLETED (last 10):
{from COMPLETED_ACTIONS.md - task completions only}

LAST RESULT:
{previous tool output - truncated to 500 chars}

WORKFLOW:
1. Execute task with tools
2. Verify with mvn_compile
3. Commit with commit_changes (auto-advances task)
```

## Verification Checklist

**IMPLEMENTED - REMAINING TO BE TESTED**

- [x] Context size constant across loops (not growing) - IMPLEMENTED
- [x] Agent sees current task correctly each loop - IMPLEMENTED
- [x] Agent does not repeat completed tasks - IMPLEMENTED
- [x] Last action result provides continuity - IMPLEMENTED
- [x] No message accumulation in state - IMPLEMENTED
- [ ] Works with 80 LLM calls (no increase needed) - TO BE TESTED

## Files Changed

| File | Change | Status |
|------|--------|--------|
| `supervisor_orchestrator_refactored.py` | Replace `_wrap_execution_node` to compile context | IMPLEMENTED |
| `src/orchestrator/message_manager.py` | Add `compile_execution_context()`, `extract_last_tool_result()`, `summarize_completed_tasks()` | IMPLEMENTED |
| `src/orchestrator/constants.py` | Increase `MAX_SUMMARY_LENGTH` to 8000 | IMPLEMENTED |
| `src/orchestrator/__init__.py` | Export new functions | IMPLEMENTED |
| `src/tools/maven_api.py` | Fix OpenRewrite version (5.42.0) + add recipe dependencies | IMPLEMENTED |

## Key Principle
> "Context is a compiled view over state, not accumulated history"

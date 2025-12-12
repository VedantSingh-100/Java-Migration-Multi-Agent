# Stuck Loop Detection Improvements

**Document Created:** December 12, 2025
**Status:** In Progress
**Related Log:** `logs/DickChesterwood__fleetman-webapp/summary_20251211_171031.log`

---

## 1. Problem Summary

### 1.1 The Bug We Hit

Migration of `DickChesterwood/fleetman-webapp` was progressing successfully (5/34 tasks, all builds passing) but was **incorrectly flagged as stuck** because `commit_changes` was called 3 times in 5 actions.

**Log Evidence:**
```
17:14:37 | [PROGRESS] TODO count increased: 4 -> 5        <- PROGRESS BEING MADE!
17:14:37 | [STUCK] Stuck attempt #1 (pattern=True, no_progress=False)  <- FLAGGED ANYWAY
17:14:37 | [ROUTER] -> STUCK LOOP detected (type=tool_loop, attempt=1/3)
...
17:17:27 | MIGRATION FAILED: Stuck in loop for 3 attempts
```

**Result:** A healthy migration was killed after only 5/34 tasks.

### 1.2 Root Causes Identified

| Bug | Location | Issue |
|-----|----------|-------|
| **Bug 1** | `error_handler.py:1076-1080` | Counts tool NAME only, not full signature (args, result) |
| **Bug 2** | `supervisor_orchestrator_refactored.py:700` | `is_stuck_now = pattern OR no_progress` - doesn't check if progress was made |
| **Bug 3** | `error_handler.py:1101-1114` | `track_action()` doesn't store tool arguments or result category |

---

## 2. Current Implementation Analysis

### 2.1 What's Well Implemented

#### No-Tool Response Handling
Location: `supervisor_orchestrator_refactored.py:1194-1464`

| Component | Function | Status |
|-----------|----------|--------|
| Count no-tool responses | `_count_no_tool_response()` | Working |
| Classify response sentiment | `_classify_no_tool_response()` | Working |
| Handle by category | `_handle_no_tool_response()` | Working |
| Generate directives | `_get_directive_for_response_type()` | Working |
| Progress check | Lines 651-658 | Working |

**Categories Implemented:**
- `acknowledging` - Agent confirms work (wasteful)
- `confused` - Agent stuck/lost (harmful) → Route to error_expert
- `thinking` - Agent reasoning (allow 2x, then force)
- `complete` - Agent claims done → Verify, then allow or route to error_expert
- `unknown` - Can't classify (graduated response)

**Key Insight Already Implemented (Line 651-658):**
```python
if progress_made:
    # Progress was made - this is a BENIGN acknowledgment
    log_agent(f"[NO_TOOL] Response type: {response_type} BUT progress made - BENIGN")
    new_no_tool_loops = 0  # Reset counter
```

#### Tool Success Detection
Location: `supervisor_orchestrator_refactored.py:1163-1191`

```python
def _determine_tool_success(self, tool_name: str, result: str) -> bool:
    failure_patterns = ['error', 'failed', 'no occurrences', 'no match', ...]
    success_patterns = ['success', 'committed', 'created', ...]
```

### 2.2 What's Buggy

#### Stuck Loop Detection
Location: `error_handler.py:1057-1099`

```python
def detect_stuck_loop(self) -> Tuple[bool, str]:
    # BUG: Only counts tool_name, not signature
    tool_counts = Counter(a.get('tool_name') for a in last_n)
    for tool, count in tool_counts.items():
        if count >= 3:  # BUG: Doesn't check if SUCCESS or FAILURE
            return (True, f"Tool '{tool}' called {count} times...")
```

**Problems:**
1. All `commit_changes` calls treated equally (healthy vs empty)
2. `find_replace` SUCCESS treated same as `find_replace` NO MATCH
3. No argument tracking (same pattern repeated vs different patterns)

#### Stuck State Logic
Location: `supervisor_orchestrator_refactored.py:697-700`

```python
stuck_via_pattern = is_stuck  # From detect_stuck_loop()
stuck_via_no_progress = new_loops_without_progress >= MAX_LOOPS_WITHOUT_PROGRESS
is_stuck_now = stuck_via_pattern or stuck_via_no_progress  # BUG: OR logic
```

**Problem:** Even if `progress_made=True`, the `stuck_via_pattern` triggers stuck state.

#### Action Tracking
Location: `error_handler.py:1101-1120`

```python
def track_action(self, tool_name: str, todo_item: str = None, logged_to_completed: bool = False):
    action = {
        'tool_name': tool_name,
        'todo_item': todo_item,
        'logged_to_completed': logged_to_completed,  # This is success, but...
        'timestamp': ...
    }
    # MISSING: args_hash, result_category
```

---

## 3. Desired Behavior (Scenarios)

### Scenario A: Healthy Migration (Should NOT Flag)

| Loop | Tools | Args Different? | Result | TODO | Stuck? |
|------|-------|-----------------|--------|------|--------|
| 1 | mvn_compile, commit | Yes | SUCCESS, "1 file changed" | 0→1 | No |
| 2 | git_status, commit | Yes | SUCCESS, "2 files changed" | 1→2 | No |
| 3 | add_plugin, commit | Yes | SUCCESS, "pom.xml modified" | 2→3 | No |
| 4 | mvn_compile, commit | Yes | SUCCESS, "verified build" | 3→4 | No |
| 5 | mvn_test, commit | Yes | SUCCESS, "tests pass" | 4→5 | No |

**Why NOT stuck:** Different args each time, SUCCESS results, progress made.

### Scenario B: Empty Commit Loop (SHOULD Flag)

| Loop | Tools | Args | Result | TODO | Stuck? |
|------|-------|------|--------|------|--------|
| 1 | commit | hash1 | "nothing to commit" | 5→5 | No |
| 2 | commit | hash1 | "nothing to commit" | 5→5 | No |
| 3 | commit | hash1 | "nothing to commit" | 5→5 | **YES** |

**Why stuck:** Same args, EMPTY_RESULT 3x, no progress.

### Scenario C: find_replace NO MATCH (SHOULD Flag)

| Loop | Tools | Args | Result | TODO | Stuck? |
|------|-------|------|--------|------|--------|
| 1 | find_replace | hash("CDATA") | NO_MATCH | 5→5 | No |
| 2 | find_replace | hash("CDATA") | NO_MATCH | 5→5 | No |
| 3 | find_replace | hash("CDATA") | NO_MATCH | 5→5 | **YES** |

**Why stuck:** Same args, NO_MATCH 3x, no progress.

### Scenario D: Progressive find_replace (Should NOT Flag)

| Loop | Tools | Args | Result | TODO | Stuck? |
|------|-------|------|--------|------|--------|
| 1 | find_replace | hash("pattern1") | SUCCESS | 5→5 | No |
| 2 | find_replace | hash("pattern2") | SUCCESS | 5→5 | No |
| 3 | find_replace | hash("pattern3") | SUCCESS | 5→6 | No |

**Why NOT stuck:** Different args each time, SUCCESS results.

### Scenario E: Confused Agent (Route to Error Expert)

| Loop | Response | Progress | Sentiment | Action |
|------|----------|----------|-----------|--------|
| 1 | "I'm not sure how to proceed..." | 5→5 | CONFUSED | Route to error_expert |

**Already implemented** via `_handle_no_tool_response()`.

### Scenario F: False Completion Claim (Route to Error Expert)

| Loop | Response | Progress | Sentiment | Verification | Action |
|------|----------|----------|-----------|--------------|--------|
| 1 | "Migration complete!" | 5→5 | COMPLETE | Build FAILS | Route to error_expert |

**Already implemented** via `_handle_no_tool_response()` line 1396-1402.

---

## 4. Solution Design

### 4.1 Core Principle: Signature-Based Detection

**Current:** Track `tool_name` only
**Proposed:** Track `(tool_name, args_hash, result_category)`

### 4.2 Result Categories

```python
class ResultCategory:
    SUCCESS = "success"        # Tool worked as expected
    NO_MATCH = "no_match"      # find_replace couldn't find pattern
    EMPTY_RESULT = "empty"     # commit with nothing to commit
    ERROR = "error"            # Tool failed with error
```

### 4.3 Enhanced Action Tracking

```python
def track_action(
    self,
    tool_name: str,
    args_hash: str,              # NEW: hash of tool arguments
    result_category: str,        # NEW: success/no_match/empty/error
    todo_item: str = None,
    logged_to_completed: bool = False
):
    action = {
        'tool_name': tool_name,
        'args_hash': args_hash,
        'result_category': result_category,
        'todo_item': todo_item,
        'logged_to_completed': logged_to_completed,
        'timestamp': datetime.now().strftime("%H:%M:%S")
    }
```

### 4.4 Enhanced Stuck Detection

```python
def detect_stuck_loop(self) -> Tuple[bool, str]:
    last_n = self.recent_actions[-self.action_window_size:]

    # Group by FULL signature
    signatures = Counter(
        (a['tool_name'], a['args_hash'], a['result_category'])
        for a in last_n
    )

    for (tool, args_hash, result_cat), count in signatures.items():
        if count >= 3:
            # Only stuck if result is PROBLEMATIC
            if result_cat in ['no_match', 'empty', 'error']:
                return (True, f"Tool '{tool}' {result_cat} {count}x with same args")

    return (False, "")
```

### 4.5 Result Categorization

```python
def categorize_tool_result(tool_name: str, result: str) -> str:
    result_lower = result.lower()

    # NO_MATCH patterns
    if any(p in result_lower for p in ['no match', 'no occurrences', 'not found']):
        return 'no_match'

    # EMPTY_RESULT patterns
    if any(p in result_lower for p in ['nothing to commit', 'no changes',
                                        'already up to date', 'working tree clean']):
        return 'empty'

    # ERROR patterns
    if any(p in result_lower for p in ['error', 'failed', 'exception',
                                        'build failure', 'return code: 1']):
        return 'error'

    return 'success'
```

### 4.6 Arguments Hashing

```python
import hashlib
import json

def hash_tool_args(args: dict) -> str:
    """Create stable hash of tool arguments for comparison."""
    # Sort keys for stable ordering
    sorted_args = json.dumps(args, sort_keys=True)
    return hashlib.md5(sorted_args.encode()).hexdigest()[:8]
```

### 4.7 Fixed Stuck State Logic

```python
# Line 700 fix
stuck_via_pattern = is_stuck  # From detect_stuck_loop()
stuck_via_no_progress = new_loops_without_progress >= MAX_LOOPS_WITHOUT_PROGRESS

# FIXED: Only consider stuck if NOT making progress
if progress_made:
    # Progress trumps pattern detection (false positive case)
    is_stuck_now = False
    if stuck_via_pattern:
        log_agent(f"[STUCK] Pattern detected but progress made - FALSE POSITIVE, ignoring")
else:
    is_stuck_now = stuck_via_pattern or stuck_via_no_progress
```

---

## 5. Implementation Plan

### Phase 1: Enhanced Action Tracking

**Files to modify:**
- `src/orchestrator/error_handler.py`
- `supervisor_orchestrator_refactored.py`

**Changes:**
1. Add `args_hash` and `result_category` to `track_action()` signature
2. Add `categorize_tool_result()` function
3. Add `hash_tool_args()` function
4. Update `_extract_and_track_tool_calls()` to:
   - Extract tool arguments from tool_calls
   - Hash the arguments
   - Categorize the result
   - Pass all to `track_action()`

### Phase 2: Enhanced Stuck Detection

**Files to modify:**
- `src/orchestrator/error_handler.py`

**Changes:**
1. Update `detect_stuck_loop()` to use full signature
2. Only flag as stuck if result_category is problematic
3. Update logging to show signature details

### Phase 3: Fixed Stuck State Logic

**Files to modify:**
- `supervisor_orchestrator_refactored.py`

**Changes:**
1. Fix line 700 to check `progress_made` before setting `is_stuck_now`
2. Add logging for false positive detection

### Phase 4: Testing

**Test scenarios:**
1. Run healthy migration - should NOT flag as stuck
2. Simulate empty commit loop - SHOULD flag
3. Simulate find_replace NO MATCH loop - SHOULD flag
4. Simulate progressive find_replace - should NOT flag

---

## 6. Files to Modify

| File | Changes Required |
|------|------------------|
| `src/orchestrator/error_handler.py` | `track_action()`, `detect_stuck_loop()`, new `categorize_tool_result()` |
| `supervisor_orchestrator_refactored.py` | `_extract_and_track_tool_calls()`, line 700 logic fix, new `hash_tool_args()` |

---

## 7. Acceptance Criteria

**Implementation Status (2025-12-12):**

- [x] Healthy migration with multiple commits does NOT trigger stuck detection
  - *Implemented via signature-based detection: different args_hash = different signature*
- [x] Empty commit loop (3x "nothing to commit") DOES trigger stuck detection
  - *Implemented via result_category='empty' in signature*
- [x] find_replace NO MATCH loop (3x same pattern) DOES trigger stuck detection
  - *Implemented via result_category='no_match' in signature*
- [x] Progressive find_replace (different patterns) does NOT trigger stuck detection
  - *Implemented via different args_hash for each call*
- [x] Progress being made (TODO count increasing) overrides pattern detection
  - *Implemented via progress_made check before is_stuck_now assignment*
- [x] Existing no-tool response handling continues to work
  - *Unchanged - no-tool response handling is separate from tool loop detection*
- [x] Existing confused/complete detection continues to work
  - *Unchanged - sentiment classification is separate from tool loop detection*

**Pending Validation:**
- [ ] Manual testing with actual migration run to verify real-world behavior

---

## 8. Risk Assessment

| Risk | Mitigation |
|------|------------|
| Breaking existing functionality | Backward-compatible changes, extensive testing |
| Missing edge cases | Document all scenarios, add logging for monitoring |
| Performance impact from hashing | MD5 hash is fast, truncate to 8 chars |

---

## 9. Progress Tracking

- [x] Analyzed latest run logs
- [x] Identified root causes
- [x] Reviewed current implementation
- [x] Researched industry patterns
- [x] Designed solution
- [x] Created this document
- [x] **Phase 1:** Enhanced Action Tracking (COMPLETED 2025-12-12)
  - Added `ToolResultCategory` class with SUCCESS, NO_MATCH, EMPTY_RESULT, ERROR
  - Added `categorize_tool_result()` function
  - Added `hash_tool_args()` function
  - Updated `track_action()` signature to accept `args_hash` and `result_category`
- [x] **Phase 2:** Enhanced Stuck Detection (COMPLETED 2025-12-12)
  - Updated `detect_stuck_loop()` to use full signature (tool_name, args_hash, result_category)
  - Only flags as stuck if result_category is PROBLEMATIC (no_match, empty, error)
  - SUCCESS with same tool/args is logged but NOT flagged as stuck
- [x] **Phase 3:** Fixed Stuck State Logic (COMPLETED 2025-12-12)
  - Fixed line 700 logic: progress now TRUMPS pattern detection
  - Added explicit false positive logging when pattern detected but progress made
  - Updated `_extract_and_track_tool_calls()` to pass args_hash and result_category
- [ ] **Phase 4:** Testing (PENDING - manual testing with actual migration)

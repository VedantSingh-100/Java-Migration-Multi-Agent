# Migration Framework Issues Analysis

## Issue 1: Detached HEAD After Verification

**Problem:** After running the verification script (`check_build_test_comprehensive.py`), repositories are left in detached HEAD state instead of on `migration-base` branch.

**Root Cause:** `eval/lang/java/eval/parse_repo.py` in `get_repo_test_files()` calls `repo.checkout(base_commit)` where `base_commit` is a commit hash. This leaves the repo in detached HEAD state. The function never restores the original branch after comparison.

**Impact:** 23 out of 36 repos (64%) were left in detached HEAD state.

**Type:** Implementation Bug

**Status:** FIXED - Added `_get_current_ref()` and `_restore_ref()` functions with try/finally block to always restore original branch.

---

## Issue 2: Java Version Regression (21 → 17)

**Problem:** Migrations go from Java 8 → Java 21 → back to Java 17 unexpectedly.

**Root Causes:**

1. **Prompt Examples Use Java 17:**
   - `execution_expert.yaml:72` - `call_openrewrite_agent("how to migrate Java 8 to Java 17")`
   - `error_expert.yaml:53,60,61` - Examples mention "JDK 17"

2. **Web Search Returns Outdated Information:**
   - Web search returns 2023 articles saying "Java 21 is not yet supported by maven-compiler-plugin; downgrade to Java 17"
   - This is FALSE - maven-compiler-plugin 3.12.0+ fully supports Java 21

3. **No Guardrails Against Downgrade:**
   - No validation prevents Java version from going DOWN (21 → 17)
   - `update_java_version` tool accepts any version without checking if it's a downgrade

**Type:** Prompt Bug + External Misinformation + Missing Guardrails

**Status:** FIXED (2024-12-20)

### Fix Implementation Details

#### A. CLI & Environment Configuration

| File | Change | Line(s) |
|------|--------|---------|
| `migrate_single_repo.py` | Added `--target-java-version` CLI argument (default: "21") | 137-141 |
| `migrate_single_repo.py` | Sets `TARGET_JAVA_VERSION` env var at startup | 145 |
| `migrate_single_repo.py` | Logs target version to summary | 153-155 |
| `prompts/prompt_loader.py` | Added `DEFAULT_TARGET_JAVA_VERSION` from env | 12 |
| `prompts/prompt_loader.py` | All `get_*_prompt()` functions now format with `{target_java_version}` | 64-92, 108-122 |

#### B. Prompt Fixes

| File | Change | Line(s) |
|------|--------|---------|
| `prompts/execution_expert.yaml` | Changed `"Java 8 to Java 17"` → `"Java {target_java_version}"` | 72 |
| `prompts/execution_expert.yaml` | Added `"NEVER downgrade!"` warning | 87 |
| `prompts/error_expert.yaml` | Changed `"JDK 17"` examples → `"Java {target_java_version}"` | 53, 60-61 |
| `prompts/error_expert.yaml` | Added **JAVA VERSION - CRITICAL GUARDRAIL** section with explicit anti-downgrade rules | 44-62 (new section) |

#### C. Tool Guardrails (Blocking Downgrades)

| File | Function | Guardrail Added |
|------|----------|-----------------|
| `src/tools/maven_api.py` | `check_java_version_downgrade()` | New helper function (lines 49-67) |
| `src/tools/maven_api.py` | `update_java_version()` | BLOCKS if requested < target (lines 339-342) |
| `src/tools/maven_api.py` | `update_java_version_in_pom()` | BLOCKS if requested < target (lines 245-248) |
| `src/tools/maven_api.py` | `update_all_poms_java_version()` | BLOCKS if requested < target (lines 295-298) |
| `src/tools/file_operations.py` | `_check_java_version_downgrade_in_replacement()` | New helper (lines 20-60) |
| `src/tools/file_operations.py` | `find_replace()` | BLOCKS Java version downgrade patterns (lines 144-147) |

#### D. Tools Updated to Use Target Version

| File | Function | Change |
|------|----------|--------|
| `src/tools/openrewrite_client.py` | `get_target_java_version()` | New helper (lines 10-12) |
| `src/tools/openrewrite_client.py` | `suggest_recipes_for_java_version()` | Uses `TARGET_JAVA_VERSION` env, dynamic recipe map (lines 88-122) |
| `src/orchestrator/tool_registry.py` | `_verify_java_21()` | Uses `TARGET_JAVA_VERSION` env instead of hardcoded "21" (lines 997-1008) |
| `src/utils/test_verifier.py` | `audit_bytecode_versions()` | Uses `TARGET_JAVA_VERSION` env for default (lines 124-129) |
| `src/utils/search_processor.py` | `SearchContext.from_environment()` | Already used `TARGET_JAVA_VERSION` ✓ (line 69) |

#### E. Usage Examples

```bash
# Default: Java 21
python migrate_single_repo.py owner/repo abc123

# Custom target: Java 17
python migrate_single_repo.py owner/repo abc123 --target-java-version 17

# Via environment variable
TARGET_JAVA_VERSION=17 python migrate_single_repo.py owner/repo abc123
```

#### F. Guardrail Behavior

When an agent attempts to downgrade Java version:

```
🚫 BLOCKED: Java version downgrade detected!
Requested: Java 17, Target: Java 21.
Downgrading is forbidden. Fix dependency versions instead of lowering Java version.
```

### Verification Checklist

- [ ] Run migration with `--target-java-version 21` and verify no downgrades occur
- [ ] Attempt manual downgrade via `update_java_version("path", "17")` - should be blocked
- [ ] Attempt downgrade via `find_replace` with `<java.version>21` → `<java.version>17` - should be blocked
- [ ] Verify prompts render with correct target version (check agent logs)
- [ ] Verify `suggest_recipes_for_java_version()` returns correct recipe for target
- [ ] Verify bytecode audit uses correct expected version

### Potential Loose Ends - NOW FIXED (2024-12-20)

1. **write_file() tool** - ✅ FIXED: Added `_check_java_version_in_content()` guardrail in `file_operations.py:76-108`
   - Scans pom.xml content for Java version patterns before writing
   - Blocks if any Java version found is below TARGET_JAVA_VERSION

2. **run_command() tool** - ✅ FIXED: Updated `command_executor.py` BLOCKED_COMMANDS and BLOCKED_PATTERNS
   - Added `sed`, `awk`, `perl`, `tee`, `patch`, `ed`, `vi`, `vim`, `nano` to BLOCKED_COMMANDS
   - Added patterns to block file redirections: `> pom.xml`, `>> *.xml`, `echo > pom`, `cat > pom`
   - Added patterns to block piping to text processors: `| sed`, `| awk`, `| perl`

3. **OpenRewrite recipes** - ✅ FIXED: Added recipe validation in `command_executor.py:888-1021`
   - `_check_openrewrite_recipes_for_downgrade()` checks pom.xml for downgrade recipes
   - `_check_recipe_for_downgrade()` checks individual recipe names
   - Both `mvn_rewrite_run` and `mvn_rewrite_run_recipe` now validate before execution

4. **find_replace patterns** - ✅ FIXED: Extended patterns in `file_operations.py:26-40`
   - Added `maven.compiler.release` pattern
   - Added whitespace-tolerant patterns with `\s*`
   - Added 1.x format patterns (e.g., `1.8` → `8`)

5. **Web search influence** - ⚠️ MITIGATED: Cannot fully prevent, but guardrails now block all bypass vectors
6. **Multi-module edge cases** - ⚠️ MITIGATED: Guardrails apply to all pom.xml writes/modifications

---

## Issue 3: Error Classification/Routing Issues

**Problem:** Multiple related issues causing migrations to fail prematurely or get stuck:

1. **Misleading Log Order** - "Calling Error Expert" appears AFTER "ERROR RESOLVED" but it's logging the COMPLETED chunk, not a new call
2. **Stuck Loop Counter Not Resetting Properly** - After error_expert successfully resolves issue, execution_expert re-detects stuck and increments counter
3. **Double Error Expert Calls** - 8 instances found where Error Expert was called twice consecutively

### Evidence from Logs

**A. Misleading Log Pattern (NOT a bug, just confusing logs):**
```
21:36:16 | [ERROR_CLASSIFIER] SUCCESS (return code 0, BUILD SUCCESS pattern found)
21:36:16 | ERROR RESOLVED: Build errors fixed
21:36:16 | Calling Error Expert  ← This logs the COMPLETED chunk, not a new call
21:36:47 | Calling Execution Expert  ← Router correctly sent to execution_expert next
```
The "Calling X Expert" log is printed by `_log_workflow_step()` for the chunk that JUST completed, not the next agent.

**B. REAL BUG - Stuck Loop Counter Issue:**
```
00:54:27 | STUCK LOOP: Routing to error_expert (attempt 2/3)
... error_expert runs, build succeeds at 00:56:26 ...
00:56:57 | MIGRATION FAILED: Stuck in loop for 3 attempts - Agent making progress
```
The stuck_loop_attempts went from 2 to 3 AFTER the error was resolved.

**C. Double Error Expert Calls (8 instances found):**
Lines 12965, 18278, 26608, 30012, 33655, 60964, 62120, 65999 in logs.

### Root Cause Analysis

**Primary Bug: `stuck_loop_attempts` counter not resetting properly**

Flow that causes the bug:
1. `execution_expert` detects stuck → sets `stuck_loop_attempts=2`
2. Router sends to `error_expert`
3. `error_expert` resolves issue → returns `stuck_loop_attempts=0` (line 1082 in orchestrator)
4. Router sends to `execution_expert`
5. `execution_expert` runs stuck detection again
6. **BUG**: If same pattern detected, `new_stuck_attempts = prev_stuck_attempts + 1` (line 872)
7. But `prev_stuck_attempts` might not be 0 due to state race condition
8. Counter reaches 3 → MIGRATION FAILED

**Code Location:** `supervisor_orchestrator_refactored.py`
- Lines 868-880: Stuck loop counter logic in `_wrap_execution_node()`
- Lines 1073-1085: Stuck loop reset logic in `_wrap_error_node()`

**Secondary Issue: State race between wrappers**
The state returned by `error_expert` might not be fully propagated before `execution_expert`'s stuck detection runs.

### Impact

| Issue | Frequency | Impact |
|-------|-----------|--------|
| Stuck loop false positive | ~5% of migrations | Premature failure at 12-32% progress |
| Double error_expert calls | 8 instances in test run | Wasted LLM calls |
| Misleading logs | Every error resolution | Developer confusion only |

**Type:** State Management Bug + Stuck Detection Logic Bug

**Status:** PENDING FIX

### Proposed Fix

1. **Fix stuck counter reset** - In `_wrap_execution_node()`, check if previous agent was error_expert and don't re-detect stuck for 1 cycle
2. **Add state flag** - `last_agent_was_error_expert` to prevent immediate re-stuck detection
3. **Fix log ordering** - Move "Calling X Expert" log to BEFORE node execution, not after
4. **Add progress tracking** - Reset stuck counter when TODO progress increases, not just when error resolves

---

## Summary Table

| Issue | Root Cause | Severity | Status |
|-------|-----------|----------|--------|
| Detached HEAD | Missing branch restore in parse_repo.py | High | FIXED |
| Java 21→17 | Prompts + Web search + No guardrails | High | FIXED (2024-12-20) |
| Error Routing | Stuck counter not resetting + State race | High | PENDING FIX |

### Issue 3 Quick Reference

| Sub-Issue | Actual Bug? | Location |
|-----------|-------------|----------|
| "Calling Error Expert" after SUCCESS | NO - Misleading log only | `_log_workflow_step()` line 2231 |
| Stuck counter not resetting | YES - Primary bug | `_wrap_execution_node()` lines 868-880 |
| Double error_expert calls | YES - Secondary bug | Router/state propagation |
| Premature failure (12-32% progress) | YES - Result of above | Stuck detection logic |

---

## Files Modified for Issue 2 Fix

### Initial Fix (2024-12-20)
```
prompts/prompt_loader.py
prompts/execution_expert.yaml
prompts/error_expert.yaml
migrate_single_repo.py
src/tools/maven_api.py
src/tools/file_operations.py
src/tools/openrewrite_client.py
src/orchestrator/tool_registry.py
src/utils/test_verifier.py
```

### Additional Guardrails (2024-12-20)

These files were updated to close bypass vectors:

| File | Changes |
|------|---------|
| `src/tools/file_operations.py` | Added `_check_java_version_in_content()` for write_file guardrail; Extended `JAVA_VERSION_PATTERNS` |
| `src/tools/command_executor.py` | Added `sed`, `awk`, `perl`, editors to BLOCKED_COMMANDS; Added file redirection patterns; Added `_check_openrewrite_recipes_for_downgrade()` and `_check_recipe_for_downgrade()` |

### Complete Guardrail Coverage

| Bypass Vector | Guardrail | Location |
|---------------|-----------|----------|
| `update_java_version()` | `check_java_version_downgrade()` | `maven_api.py:49-67` |
| `update_java_version_in_pom()` | `check_java_version_downgrade()` | `maven_api.py:245-248` |
| `update_all_poms_java_version()` | `check_java_version_downgrade()` | `maven_api.py:295-298` |
| `find_replace()` | `_check_java_version_downgrade_in_replacement()` | `file_operations.py:43-72` |
| `write_file()` | `_check_java_version_in_content()` | `file_operations.py:76-108, 184-187` |
| `run_command()` | BLOCKED_COMMANDS + BLOCKED_PATTERNS | `command_executor.py:154-215` |
| `mvn_rewrite_run()` | `_check_openrewrite_recipes_for_downgrade()` | `command_executor.py:888-948` |
| `mvn_rewrite_run_recipe()` | `_check_recipe_for_downgrade()` | `command_executor.py:970-1021` |

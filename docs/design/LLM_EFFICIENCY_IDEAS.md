# LLM Efficiency Improvement Ideas

## Problem
- ~11 LLM calls per task (should be ~6-7)
- ReAct pattern: every tool call = 1 LLM call
- Maven outputs are huge (500-2000 lines)
- Agent reads files to understand state (unnecessary)

## Proposed Optimizations

### 1. Maven Output Compression
**Location**: `src/tools/command_executor.py`
**Impact**: Medium
**Effort**: Low

Compress mvn_compile, mvn_test, mvn_rewrite_run outputs to just:
- SUCCESS or FAILURE
- If failure: only [ERROR] lines (first 10)

```python
def _compress_maven_output(output: str, return_code: int) -> str:
    if return_code == 0:
        return "BUILD SUCCESS"
    else:
        errors = [line for line in output.split('\n') if '[ERROR]' in line]
        return f"BUILD FAILURE:\n" + '\n'.join(errors[:10])
```

### 2. Composite Tools
**Location**: New file `src/tools/composite_tools.py`
**Impact**: High
**Effort**: Medium

| Tool | Replaces | Saves |
|------|----------|-------|
| `execute_recipe_verified(recipe)` | mvn_rewrite_run + mvn_compile | 1-2 calls |
| `commit_if_changes(message)` | git_status + commit | 1 call |
| `verify_migration_complete(type)` | multiple reads + grep | 2-3 calls |

### 3. Richer Task Descriptions
**Location**: `prompts/analysis_expert.yaml`
**Impact**: Medium
**Effort**: Low

Make analysis agent create explicit tasks:
```
Before: "Execute JUnit 4 to JUnit 5 migration using OpenRewrite"
After: "Execute JUnit 4 to 5: call mvn_rewrite_run_recipe with
'org.openrewrite.java.testing.junit5.JUnit4to5Migration',
verify with mvn_compile, then commit"
```

### 4. File State Injection
**Location**: `src/orchestrator/message_manager.py` (ExternalMemoryBuilder)
**Impact**: Medium
**Effort**: Low

Inject file state into context so agent doesn't read files:
```python
context += f"""
CURRENT STATE (no need to read files):
- Java version: {_get_java_version_from_pom()}
- Spring Boot: {_get_spring_boot_version()}
- OpenRewrite configured: YES/NO
"""
```

### 5. Early Success Exit
**Location**: `src/orchestrator/agent_wrappers.py`
**Impact**: Low-Medium
**Effort**: Medium

If task is "verify compilation" and mvn_compile SUCCESS, exit early without additional tool calls.

## Priority Order
1. Maven output compression (quick win)
2. File state injection (quick win)
3. Composite tools (highest impact)
4. Richer task descriptions (prompt change only)
5. Early success exit (more complex)

## Notes
- Recipe name validation is handled by RAG agent in BNY framework (not here)
- All changes should go in refactored code (`src/orchestrator/`, NOT `supervisor_orchestrator.py`)

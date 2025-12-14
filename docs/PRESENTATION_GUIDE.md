# Java Migration Framework: Multi-Agent LLM Orchestration System
## Complete Technical Presentation Guide

---

# Executive Summary

The Java Migration Framework is an enterprise-grade, multi-agent LLM orchestration system that automates complex Java ecosystem migrations. It transforms legacy Java 8/Spring Boot 2 applications to Java 21/Spring Boot 3 using a coordinated team of specialized AI agents, while maintaining strict cost control, test preservation guarantees, and deterministic behavior.

**What It Migrates:**
- Java 8+ → Java 21
- Spring Boot 2.x → Spring Boot 3.x
- Spring Framework 5.x → Spring Framework 6.x
- javax.* → jakarta.* namespace
- JUnit 4 → JUnit 5

**Key Differentiators:**
- Deterministic routing (no LLM calls to decide which agent runs)
- Signature-based loop detection (prevents stuck patterns without false positives)
- Test preservation guarantees (baseline capture + final validation)
- Circuit breaker cost control (hard limit at 500 LLM calls)
- Compiled context pattern (fresh context each loop, no message bloat)
- Intelligent web search with query optimization and deduplication

---

# Part 1: System Architecture

## 1.1 High-Level Architecture

The system follows a **Supervisor-Worker** pattern with three specialized agents coordinated by a central orchestrator:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         CLI Entry Point                                  │
│                    (migrate_single_Repo.py)                              │
└─────────────────────────────┬────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│              SupervisorMigrationOrchestrator                             │
│                  (LangGraph State Machine)                               │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  CONTROL MECHANISMS                                                 │  │
│  │  ├─ Circuit Breaker: MAX_LLM_CALLS = 500                           │  │
│  │  ├─ Token Counter: Real-time cost tracking                         │  │
│  │  ├─ Stuck Detector: Signature-based loop detection                 │  │
│  │  ├─ Context Manager: 140K token limit, smart pruning               │  │
│  │  ├─ Web Search Processor: Query optimization + deduplication       │  │
│  │  └─ Deterministic Router: Pure code-based agent selection          │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐   │
│  │ ANALYSIS EXPERT  │  │ EXECUTION EXPERT │  │    ERROR EXPERT      │   │
│  │                  │  │                  │  │                      │   │
│  │ • Discover POMs  │  │ • Execute tasks  │  │ • Diagnose failures  │   │
│  │ • Map deps       │  │ • Run recipes    │  │ • Apply fixes        │   │
│  │ • Create plan    │→ │ • Verify builds  │→ │ • Web search assist  │   │
│  │ • Set baseline   │  │ • Git commits    │  │ • Escalate strategy  │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────────┘   │
│                                                                          │
│  SHARED STATE FILES (Source of Truth):                                   │
│  ├─ TODO.md          : Task checklist created by analysis               │
│  ├─ CURRENT_STATE.md : Project status (append-only)                     │
│  ├─ COMPLETED_ACTIONS.md : Action audit trail (system-managed)          │
│  ├─ VISIBLE_TASKS.md : Next 3 tasks (auto-generated)                    │
│  └─ ERROR_HISTORY.md : Error tracking (prevents duplicate fixes)        │
└──────────────────────────────────────────────────────────────────────────┘
```

## 1.2 The Three Agents

### Analysis Expert
**Role:** The Planner and Architect

The Analysis Expert is the first agent to run. It performs comprehensive project discovery and creates the migration blueprint.

**Responsibilities:**
- Discover all pom.xml files (single vs multi-module detection)
- Extract current versions (Java, Spring Boot, Spring Framework, JUnit)
- Identify migration targets (javax imports, deprecated APIs, old annotations)
- Query OpenRewrite for applicable recipes
- Establish build baseline (compile + test counts)
- Create the migration plan (TODO.md with ordered tasks)

**Tools Available:**
- Read-only discovery: `find_all_poms`, `read_pom`, `list_dependencies`, `search_files`
- Recipe research: `call_openrewrite_agent`, `get_available_recipes`
- Build verification: `mvn_compile`, `mvn_test`
- Web search: `web_search_tool` for migration guidance
- State creation: Write access to TODO.md, CURRENT_STATE.md, analysis.md only

**Output Artifacts:**
- `TODO.md`: Ordered task checklist (e.g., "- [ ] Update Java version to 21")
- `CURRENT_STATE.md`: Baseline status (versions, test counts, build status)
- `analysis.md`: Detailed findings with code patterns discovered

### Execution Expert
**Role:** The Implementer

The Execution Expert takes the migration plan and executes it task by task, verifying each step before moving on.

**Responsibilities:**
- Read current task from VISIBLE_TASKS.md (only sees next 3 tasks)
- Execute migration using appropriate tools
- Verify success (compile, read files, run tests if needed)
- Commit changes with meaningful messages
- Mark tasks complete (via find_replace on TODO.md)

**Tools Available:**
- File operations: `read_file`, `write_file`, `find_replace`
- Maven/OpenRewrite: `update_java_version`, `add_openrewrite_plugin`, `mvn_rewrite_run`
- Build verification: `mvn_compile`, `mvn_test`
- Git operations: Full access (commit, branch, tag)
- Web search: `web_search_tool` for solutions

**Workflow Pattern:**
```
1. Read VISIBLE_TASKS.md → Get current task
2. Execute task using appropriate tool(s)
3. Verify success (compile/read/test)
4. Commit changes → AUTO_SYNC marks task [x]
5. Loop to next task
```

### Error Expert
**Role:** The Debugger and Fixer

The Error Expert activates when builds fail, tests fail, or the execution expert gets stuck. It diagnoses issues and applies targeted fixes.

**Responsibilities:**
- Diagnose root cause of failures (POM errors, compile errors, test failures)
- Apply targeted fixes using appropriate strategy
- **Use web search when stuck (MANDATORY after 1 failed attempt)**
- Escalate strategy if initial approaches fail
- Return control to execution expert after fix

**Tools Available:**
- File operations: `read_file`, `find_replace`, `write_file`
- Build verification: `mvn_compile`, `mvn_test`
- Research: `web_search_tool` (MANDATORY after 1 failure)
- Git: Read-only (status, log, branches)

**Escalation Strategy:**
- Attempt 1: Try different tool or approach
- Attempt 2: Rewrite entire file section vs targeted edit + **mandatory web search**
- Attempt 3: Skip task and move on (mark as SKIPPED)

---

# Part 2: The Complete Execution Flow

## 2.1 Phase Diagram

```
                    ┌──────────┐
                    │   INIT   │
                    └────┬─────┘
                         │
                         ▼
           ┌────────────────────────────┐
           │   ANALYSIS PHASE           │
           │   (analysis_expert runs)    │
           │   • Discover project       │
           │   • Create TODO.md         │
           │   • Set baseline           │
           └────────────┬───────────────┘
                        │
                        │ Auto-detects: TODO.md + CURRENT_STATE.md exist
                        ▼
           ┌────────────────────────────┐
           │   PHASE TRANSITION         │
           │   • Capture test baseline  │
           │   • Create VISIBLE_TASKS   │
           │   • Prune analysis msgs    │
           └────────────┬───────────────┘
                        │
                        ▼
           ┌────────────────────────────┐
           │   EXECUTION PHASE          │◄──────────────────┐
           │   (execution_expert runs)  │                   │
           │   • Execute current task   │                   │
           │   • Verify success         │                   │
           │   • Commit changes         │                   │
           └────────────┬───────────────┘                   │
                        │                                   │
          ┌─────────────┼─────────────┐                    │
          │             │             │                    │
     Build Error   Test Failure   Success                  │
          │             │             │                    │
          ▼             ▼             │                    │
    ┌───────────────────────┐        │                    │
    │   ERROR RESOLUTION    │        │                    │
    │   (error_expert runs) │────────┘                    │
    │   • Diagnose issue    │     Fix successful          │
    │   • Web search help   │                             │
    │   • Apply fix         │                             │
    │   • 3 attempts max    │                             │
    └───────────┬───────────┘                             │
                │                                         │
           3 failures                                     │
                │                                         │
                ▼                                         │
          ┌──────────┐                                    │
          │  FAILED  │                          All tasks done
          └──────────┘                                    │
                                                          │
                                                          ▼
                                              ┌────────────────────┐
                                              │   VALIDATION       │
                                              │   • Classify result│
                                              │   • Test invariance│
                                              │   • Generate report│
                                              └─────────┬──────────┘
                                                        │
                                          ┌─────────────┼─────────────┐
                                          │             │             │
                                          ▼             ▼             ▼
                                     ┌────────┐   ┌─────────┐   ┌────────┐
                                     │SUCCESS │   │PARTIAL  │   │FAILURE │
                                     │ ≥90%   │   │ ≥50%    │   │ <50%   │
                                     └────────┘   └─────────┘   └────────┘
```

## 2.2 Detailed Phase Walkthrough

### Phase 1: Initialization

**Entry:** User runs `python migrate_single_Repo.py <repo_name> <base_commit>`

**Actions:**
1. Clone repository to `/repositories/{repo_name}/`
2. Checkout specified commit
3. Create migration branch
4. Initialize logging (3 streams: LLM, agent, summary)
5. Initialize token counter
6. **Setup search context** (detect Java/Spring versions from pom.xml)
7. Create SupervisorMigrationOrchestrator instance
8. Begin migration workflow

### Phase 2: Analysis

**Agent:** analysis_expert

**Entry Condition:** Phase is INIT (first run) or analysis not yet complete

**Actions:**
1. Read all pom.xml files (multi-module detection)
2. Extract current versions:
   - Java version from properties
   - Spring Boot version from parent or dependency
   - Spring Framework version
   - JUnit version
3. Scan for migration targets:
   - `javax.*` imports → need jakarta migration
   - `@RunWith` annotations → need JUnit 5 migration
   - Deprecated APIs → need recipe application
4. Query OpenRewrite for applicable recipes
5. Run baseline build: `mvn compile` + `mvn test`
6. Record baseline test count
7. Create state files:
   - `TODO.md`: Ordered task list
   - `CURRENT_STATE.md`: Baseline status
   - `analysis.md`: Detailed findings

**Exit Condition:** Auto-detected when TODO.md and CURRENT_STATE.md exist with sufficient content (50+ characters, task markers present)

**Stuck Recovery:** If 5+ consecutive responses without tool calls → Reset with fresh context (max 2 resets)

### Phase 3: Phase Transition (Analysis → Execution)

**Triggered By:** analysis_done = True (auto-detected)

**Actions:**
1. Parse TODO.md to extract all tasks
2. Create VISIBLE_TASKS.md with next 3 tasks
3. Capture test baseline (method count per test file)
4. Prune accumulated analysis messages (prevent context bloat)
5. Set phase to EXECUTION

### Phase 4: Execution

**Agent:** execution_expert

**Entry Condition:** analysis_done = True AND execution_done = False

**Loop Structure:**
```
while tasks remain AND no_errors AND not_stuck:
    1. Compile fresh context (NOT accumulated)
    2. Read VISIBLE_TASKS.md → Extract current task
    3. Execute task using appropriate tool
    4. Verify success
    5. If success: Commit → AUTO_SYNC marks [x]
    6. Regenerate VISIBLE_TASKS.md
    7. Loop
```

**Tools Tracking:**
When a "tracked tool" is used (maven, openrewrite, git commit), the system:
- Records action to COMPLETED_ACTIONS.md
- On commit: Automatically marks current TODO item as [x]
- Updates VISIBLE_TASKS.md with next task

**Exit Conditions:**
- All tasks marked [x] → Success
- Build error detected → Route to error_expert
- Stuck loop detected → Route to error_expert
- Max loops reached (200) → Timeout
- LLM limit hit (500) → Graceful shutdown

### Phase 5: Error Resolution

**Agent:** error_expert

**Entry Conditions:**
- Build failure (compile error, POM error)
- Test failure (after 1 execution retry)
- Stuck loop detected (same tool+args+failure 3x)

**Workflow:**
```
Attempt 1:
  - Diagnose error type
  - Apply standard fix
  - Verify with mvn_compile/test

Attempt 2 (if attempt 1 failed):
  - Try different approach
  - MANDATORY WEB SEARCH ← System enforces this
  - Apply solution from search

Attempt 3 (if attempt 2 failed):
  - Skip task OR
  - Rewrite entire section
  - If still fails: Route to FAILED
```

**Return:** If fix successful → Back to execution_expert

### Phase 6: Validation and Classification

**Triggered By:** All tasks complete OR max errors reached OR LLM limit hit

**Deterministic Classification (NO LLM):**
```python
if llm_limit_exceeded or timeout:
    return INCOMPLETE

if build_fails:
    return FAILURE

if tests_fail:
    return FAILURE

progress = completed_tasks / total_tasks

if progress >= 0.90 and build_passes and tests_pass:
    return SUCCESS
elif progress >= 0.50 and build_passes and tests_pass:
    return PARTIAL_SUCCESS
else:
    return FAILURE
```

**Test Invariance Check:**
- Compare final test method count against baseline
- If test count changed → Force FAILURE (regardless of other success indicators)

---

# Part 3: Web Search Integration

## 3.1 The Web Search Challenge

Migration agents encounter unfamiliar errors that require external knowledge:
- Framework compatibility issues (CGLIB + JDK 17)
- Version-specific migration patterns
- Deprecated API replacements
- Error messages not in training data

**Without smart search:**
- Agents search for raw stack traces (poor results)
- Same search repeated multiple times (wasted API calls)
- Results are tangentially related (not actionable)

## 3.2 Web Search Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│ AGENT CALLS: web_search_tool("Spring Boot CGLIB error JDK 17...")      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1: web_search_tool() in web_search_tools.py                        │
│                                                                         │
│   - Gets SearchProcessor singleton                                      │
│   - Builds SearchContext from environment (Java version, Spring ver)   │
│   - Calls processor.search(query, search_fn)                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2: SearchProcessor.search()                                        │
│                                                                         │
│   a) OPTIMIZE QUERY:                                                    │
│      - Clean stack trace noise (line numbers, "at org.xxx")            │
│      - Add version context ("Java 21", "Spring Boot")                  │
│      - Add solution keywords ("fix solution")                          │
│                                                                         │
│   b) CHECK CACHE:                                                       │
│      - Hash normalized query                                            │
│      - If cached → return immediately (no API call)                    │
│                                                                         │
│   c) EXECUTE SEARCH (if not cached):                                   │
│      - Call Tavily API with optimized query                            │
│      - Get synthesized answer + 5 results                              │
│                                                                         │
│   d) TRUNCATE & CACHE:                                                  │
│      - Truncate to 10K chars (preserve structure)                      │
│      - Store in cache for deduplication                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Return to Agent                                                 │
│                                                                         │
│   Raw results returned - LLM extracts actionable steps                 │
│   (LLM handles synthesis better than regex rules)                      │
└─────────────────────────────────────────────────────────────────────────┘
```

## 3.3 Query Optimization

The SearchProcessor transforms raw agent queries into optimized search queries.

### Query Cleaning

**Problem:** Agents often search with raw error messages including stack traces.

**Before:**
```
NoClassDefFoundError: net/sf/cglib/proxy/Enhancer
    at org.springframework.aop.framework.CglibAopProxy.createEnhancer(CglibAopProxy.java:219)
    at org.springframework.aop.framework.CglibAopProxy.getProxy(CglibAopProxy.java:138)
```

**After Cleaning:**
```
NoClassDefFoundError net sf cglib proxy Enhancer springframework aop framework CglibAopProxy createEnhancer
```

**Cleaning Operations:**
1. Remove line numbers: `(ClassName.java:123)` → removed
2. Remove stack trace prefixes: `at org.xxx.yyy` → removed
3. Remove Maven markers: `[INFO]`, `[ERROR]` → removed
4. Collapse whitespace

### Context Enhancement

**Problem:** Agent queries lack version context needed for accurate results.

**Enhancement Rules:**
1. If Java version not in query → Add "Java {target_version}"
2. If "spring" mentioned but not "Spring Boot" → Add "Spring Boot"
3. If error keywords present but no solution keywords → Add "fix solution"

**Example:**
```
ORIGINAL: "NoClassDefFoundError cglib proxy Enhancer"
ENHANCED: "NoClassDefFoundError cglib proxy Enhancer Java 21 fix solution"
```

### Query Length Control

```
MAX_QUERY_LENGTH = 300 characters  (Tavily recommends < 400)
```

If enhanced query exceeds limit, truncate while preserving key terms.

## 3.4 Query Deduplication (Caching)

**Problem:** Agents often search for the same thing multiple times during error resolution.

**Solution:** Cache results by normalized query hash.

**Normalization Algorithm:**
```python
def _get_cache_key(self, query: str) -> str:
    # 1. Lowercase
    # 2. Split into words
    # 3. Sort words alphabetically
    # 4. Join back
    # 5. MD5 hash
    words = sorted(query.lower().split())
    normalized = ' '.join(words)
    return hashlib.md5(normalized.encode()).hexdigest()
```

**Result:** Queries with same words in different order → Same cache key → No duplicate API call

**Example:**
```
Query 1: "Spring Boot CGLIB error Java 17"
Query 2: "Java 17 CGLIB error Spring Boot"
→ Both produce same cache key → Second query returns cached result
```

**Cache Hit Response:**
```
[Cached Result - identical search performed earlier]

Summary: CGLIB requires special handling in Java 17+...
```

## 3.5 Search Context

The SearchProcessor uses migration context for smarter queries.

**SearchContext Fields:**
```python
@dataclass
class SearchContext:
    java_version_current: str = "8"     # Detected from pom.xml
    java_version_target: str = "21"     # Migration target
    spring_boot_version: str = "unknown"  # From parent pom
    project_path: str = ""              # For logging
```

**Auto-Detection from pom.xml:**
```python
def setup_search_context_from_pom(project_path: str):
    # Parse pom.xml
    # Extract java.version or maven.compiler.source
    # Extract Spring Boot parent version
    # Set environment variables
    os.environ["CURRENT_JAVA_VERSION"] = "8"
    os.environ["TARGET_JAVA_VERSION"] = "21"
    os.environ["CURRENT_SPRING_VERSION"] = "2.5.0"
```

**Context Usage:** Every search automatically includes detected versions for more relevant results.

## 3.6 Typical Search Queries

**Error Resolution Queries:**
```
"Spring Boot 1.5 NoClassDefFoundError cglib JDK 17 fix solution"
"javax to jakarta migration Spring Boot 3 fix solution"
"JUnit 4 SpringRunner migration JUnit 5 fix solution"
"AbstractMethodError hibernate Spring Boot 3 Java 21 fix solution"
```

**Recipe Research Queries:**
```
"OpenRewrite recipe Spring Boot 2 to 3 migration"
"Jakarta EE migration OpenRewrite recipe configuration"
"JUnit 5 migration best practices Spring Boot"
```

## 3.7 Search Results Format

**Tavily API Response Structure:**
```
Summary: [AI-synthesized answer from search results]

---

Results:

**Title 1**
Content snippet (first 500 chars)...
Source: https://example.com/article

---

**Title 2**
Content snippet...
Source: https://stackoverflow.com/question

---
```

**Result Truncation:**
- Max 10,000 characters returned to agent
- Truncates at natural break points (`---` or newline)
- Preserves at least 70% of content

## 3.8 Mandatory Web Search Enforcement

The system **forces** the error_expert to use web search after failed fix attempts.

**Trigger:** When `error_count >= 1` (at least one failed fix attempt)

**Injected Message:**
```
⚠️ MANDATORY WEB SEARCH REQUIRED ⚠️
════════════════════════════════════════════════════════════════════
You have attempted 1 fix(es) without success.

BEFORE trying another fix, you MUST:
1. Call web_search_tool with a specific query about this error
2. Include the error message + framework versions + "fix"

SUGGESTED QUERY:
  web_search_tool("NoClassDefFoundError cglib Spring Boot Java 21 fix")

DO NOT skip this step. Search first, then apply the solution.
════════════════════════════════════════════════════════════════════
```

**Query Builder:**
```python
def _build_search_query(self, error_snippet: str) -> str:
    # Extract error class (NoClassDefFoundError, etc.)
    # Add framework context
    # Add "fix" keyword
    return f"{error_class} {framework} Java {java_version} fix"
```

## 3.9 Search Statistics

The SearchProcessor tracks metrics:

```python
{
    'total_searches': 24,      # Total search() calls
    'cache_hits': 8,           # Returned from cache
    'cache_hit_rate': 0.33,    # 33% of searches were duplicates
    'cached_queries': 16,      # Unique queries stored
}
```

**Logging Output:**
```
[SEARCH_PROC] Search complete | 8432 chars | Cache: 8/24
```

## 3.10 Web Search in the Overall Flow

```
Error Detected (BUILD FAILURE)
        │
        ▼
┌───────────────────────────────────┐
│ error_expert Attempt 1            │
│ • Diagnose error                  │
│ • Apply standard fix              │
│ • mvn_compile → STILL FAILS       │
└───────────────────────────────────┘
        │
        ▼ (error_count = 1)
┌───────────────────────────────────┐
│ error_expert Attempt 2            │
│ ┌───────────────────────────────┐ │
│ │ MANDATORY WEB SEARCH          │ │
│ │ web_search_tool("...")        │ │
│ │      ↓                        │ │
│ │ SearchProcessor               │ │
│ │ • Optimize query              │ │
│ │ • Check cache                 │ │
│ │ • Execute Tavily search       │ │
│ │ • Return results              │ │
│ └───────────────────────────────┘ │
│ • Apply solution from search      │
│ • mvn_compile → SUCCESS!          │
└───────────────────────────────────┘
        │
        ▼
Return to execution_expert
```

---

# Part 4: State Management

## 4.1 The State Object

The system maintains state using a LangGraph AgentState with these key fields:

```
PHASE TRACKING
├─ current_phase: "INIT" | "ANALYSIS" | "EXECUTION" | "ERROR_RESOLUTION" | ...
├─ analysis_done: boolean
└─ execution_done: boolean

PROGRESS TRACKING
├─ last_todo_count: number (completed tasks at previous loop)
├─ loops_without_progress: number
└─ total_execution_loops: number

STUCK DETECTION
├─ is_stuck: boolean
├─ stuck_type: "tool_loop" | "no_progress" | "none"
├─ stuck_tool: string (which tool is stuck)
├─ stuck_loop_attempts: number (0-3)
└─ stuck_failed_approaches: JSON list of tried approaches

ERROR TRACKING
├─ has_build_error: boolean
├─ error_count: number (0-3)
├─ error_type: "success" | "pom" | "compile" | "test"
├─ test_failure_count: number
└─ last_test_failure_task: string

NO-TOOL TRACKING
├─ no_tool_call_loops: number
└─ thinking_loops: number
```

## 4.2 State Files (Source of Truth)

The system uses files as the authoritative source of truth, not just LLM memory:

| File | Created By | Purpose | Access Rules |
|------|------------|---------|--------------|
| **TODO.md** | analysis_expert | Task checklist | Read by system; mark [x] via find_replace only |
| **CURRENT_STATE.md** | analysis_expert | Status info | Append-only after creation |
| **COMPLETED_ACTIONS.md** | System | Audit trail | System-managed; read-only for agents |
| **VISIBLE_TASKS.md** | System | Next 3 tasks | Auto-generated; read-only for agents |
| **ERROR_HISTORY.md** | System | Error tracking | Append-only; prevents duplicate fixes |
| **analysis.md** | analysis_expert | Detailed findings | Read-only after creation |

**Why Files?**
- LLM memory can be pruned or summarized → data loss
- Files persist across context resets
- Files are human-readable for debugging
- Files prevent "amnesia" where agent forgets previous work

## 4.3 External Memory Injection

Before each execution_expert invocation, the system injects a context block at position [1]:

```
EXTERNAL MEMORY (Updated: 14:32:05)

COMPLETED ACTIONS (DO NOT REPEAT):
- [UPDATE_JAVA_VERSION] Updated pom.xml Java 8→21 | 14:28:12
- [MVN_REWRITE_RUN] Applied jakarta namespace recipes | 14:29:45
- [COMMIT] "feat: migrate to Java 21" | 14:30:01

YOUR TASKS (Next 3 only):
✔ CURRENT TASK: Update Spring Boot parent to 3.2.0
▪ UPCOMING: Run JUnit 5 migration recipe
▪ UPCOMING: Fix deprecated API calls

PROGRESS: 12/45 tasks (26%)

CURRENT STATE:
Java: 21, Spring Boot: 2.7.0→pending, Tests: 142 passing
```

This injection keeps agents context-aware without accumulating hundreds of messages.

---

# Part 5: Deterministic Routing

## 5.1 Why Deterministic?

Traditional multi-agent systems often use an LLM to decide which agent to call next. This has problems:
- Uses tokens for "routing" decisions
- Non-deterministic (same state might route differently)
- Can create infinite loops ("should I call execution?")
- Hard to debug and test

Our system uses **pure code-based routing** with no LLM involvement:

```
ROUTING PRIORITY ORDER:

1. execution_done = True → END (migration complete)

2. error_count >= 3 OR stuck_loop_attempts >= 3 → FAILED

3. has_build_error = True:
   ├─ test_violation → error_expert
   ├─ pom_error → error_expert
   ├─ test_failure AND test_failure_count == 0 → execution_expert (1 retry)
   ├─ test_failure AND test_failure_count > 0 → error_expert
   └─ compile_error → error_expert

4. is_stuck = True → error_expert

5. analysis_done = False → analysis_expert

6. Default → execution_expert
```

## 5.2 Benefits of Deterministic Routing

| Benefit | Explanation |
|---------|-------------|
| **Cost Savings** | No LLM tokens spent on "what next?" decisions |
| **Reproducibility** | Same state always routes to same agent |
| **Testability** | Can unit test routing logic |
| **Debuggability** | Clear audit trail of why agent was selected |
| **Speed** | No LLM latency for routing decisions |

---

# Part 6: Loop Detection and Stuck Handling

## 6.1 The Stuck Loop Problem

A common failure mode in agent systems is the "stuck loop" where:
- Agent tries the same action repeatedly
- Action keeps failing the same way
- Agent doesn't realize it's stuck
- Tokens wasted on futile attempts

## 6.2 Signature-Based Detection

Our system uses **signature-based detection** rather than simple repetition counting:

**Signature = (tool_name, args_hash, result_category)**

```
HEALTHY REPETITION (ALLOWED):
commit_changes("file1.java", "msg1") → SUCCESS
commit_changes("file2.java", "msg2") → SUCCESS
commit_changes("file3.java", "msg3") → SUCCESS
→ Different args each time = healthy variation

STUCK PATTERN (BLOCKED):
find_replace("pom.xml", "<old>", "<new>") → NO_MATCH
find_replace("pom.xml", "<old>", "<new>") → NO_MATCH
find_replace("pom.xml", "<old>", "<new>") → NO_MATCH
→ Same args, same failure 3x = STUCK
```

**Detection Algorithm:**
1. After each tool call, extract: tool name, argument hash, result category
2. Add to sliding window (max 10 recent actions)
3. Count occurrences of same signature
4. If same (tool + args + failed_result) appears 3+ times → Flag STUCK

**Result Categories:**
- SUCCESS: Tool worked
- EMPTY: No results but not error
- NO_MATCH: Search/replace found nothing
- ERROR: Tool returned error
- EXCEPTION: Tool threw exception

## 6.3 Progress-Based Detection

Beyond signature detection, the system tracks actual progress:

```
Loop N: 12/45 tasks complete
Loop N+1: 12/45 tasks complete (no progress)
Loop N+2: 12/45 tasks complete (no progress)
Loop N+3: 12/45 tasks complete (no progress)
Loop N+4: 12/45 tasks complete (no progress)
Loop N+5: 12/45 tasks complete → STUCK (5 loops without progress)
```

**Override Rule:** If progress is made (task count increased), don't flag as stuck even if signature patterns detected. This prevents false positives during healthy iteration.

## 6.4 Stuck Recovery Strategy

When stuck detected, error_expert receives escalating instructions:

**Attempt 1:** "Try a DIFFERENT tool or approach than previously attempted"
**Attempt 2:** "Use write_file to REWRITE the entire section instead of find_replace"
**Attempt 3:** "Skip this task and mark it [x] SKIPPED with explanation"

Failed approaches are tracked in `stuck_failed_approaches` JSON list to prevent repeating the same fix.

---

# Part 7: Error Handling

## 7.1 Error Detection

The system detects errors through output analysis of maven commands:

**POM Errors:**
```
[ERROR] Error parsing pom.xml
[ERROR] Malformed POM
[ERROR] Invalid XML
```

**Compile Errors:**
```
[ERROR] cannot find symbol
[ERROR] package does not exist
[ERROR] BUILD FAILURE
```

**Test Failures:**
```
Tests run: 50, Failures: 3, Errors: 0
[ERROR] There are test failures
```

## 7.2 Error Routing Logic

```
Error Detected
     │
     ▼
┌─────────────────┐
│ What type?      │
└────────┬────────┘
         │
    ┌────┴────┬────────────┐
    │         │            │
   POM    Compile       Test
    │         │            │
    ▼         ▼            ▼
error_expert  error_expert  Is this first
                           test failure?
                              │
                         ┌────┴────┐
                         │         │
                        YES       NO
                         │         │
                         ▼         ▼
                   execution    error_expert
                   (1 retry)
```

**Test Failure Retry:** The execution expert gets ONE chance to retry a test failure (common case: just needed a clean build). If it fails again, routes to error_expert.

## 7.3 Error Expert Workflow

```
Receive Error Context
       │
       ▼
┌──────────────────┐
│ Classify Error   │
│ • POM/XML issue  │
│ • Import missing │
│ • Test assertion │
│ • Configuration  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Attempt Fix      │
│ Based on type    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Verify Fix       │
│ mvn compile/test │
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
  Fixed    Failed
    │         │
    ▼         ▼
Return to  Attempt 2
execution  (WEB SEARCH)
              │
              ▼
         ┌────┴────┐
         │         │
       Fixed    Failed
         │         │
         ▼         ▼
      Return    Attempt 3
                  │
                  ▼
             ┌────┴────┐
             │         │
           Fixed    Failed
             │         │
             ▼         ▼
          Return   FAILURE
```

## 7.4 Error Deduplication

`ERROR_HISTORY.md` tracks all error fix attempts:
```
## Error: cannot find symbol: javax.servlet.http.HttpServletRequest
Attempted: 2024-01-15 14:32:05
Fix tried: Added jakarta.servlet dependency
Result: FAILED (different error)

## Error: package jakarta.servlet does not exist
Attempted: 2024-01-15 14:33:12
Fix tried: Updated import statements
Result: SUCCESS
```

This prevents the error_expert from trying the same fix twice for the same error.

---

# Part 8: Circuit Breakers and Cost Control

## 8.1 The LLM Call Limit

**Hard Limit:** 500 LLM calls per migration

**Implementation:**
```python
class CircuitBreakerChatBedrock(ChatBedrock):
    def _generate(self, *args, **kwargs):
        if self._token_counter.llm_calls >= MAX_LLM_CALLS:
            raise LLMCallLimitExceeded(
                f"Reached {MAX_LLM_CALLS} LLM calls"
            )
        return super()._generate(...)
```

**Check Point:** BEFORE every LLM invocation (prevents overage)

## 8.2 Token Tracking

Real-time tracking of:
- Prompt tokens (input to LLM)
- Response tokens (output from LLM)
- Total LLM calls
- Running cost estimate

**Cost Model:**
```
Prompt cost  = prompt_tokens / 1,000,000 × $3.00
Response cost = response_tokens / 1,000,000 × $15.00
Total cost = prompt_cost + response_cost
```

## 8.3 Context Window Management

**Limit:** 140,000 tokens maximum

**When approaching limit:**
1. Prune old messages (keep recent window)
2. Summarize verbose tool outputs
3. Offload web search results to files
4. Compress file read outputs

**Target after pruning:** 30,000 tokens

## 8.4 Graceful Shutdown

When limit hit (LLM calls or timeout):

```
1. Stop workflow streaming
2. Extract current state from last chunk
3. Run deterministic classification
4. Calculate actual progress percentage
5. Run final test validation
6. Return result with status:
   - INCOMPLETE if >50% done
   - FAILURE if <50% done
   - Include reason: "LLM limit reached" or "Timeout"
7. Log token usage statistics
```

**Partial Credit:** If migration is 60% complete when limit hit, it's marked PARTIAL_SUCCESS (not failure), allowing future resume or manual completion.

---

# Part 9: Tool Ecosystem

## 9.1 Tool Categories

**52+ tools across 10 modules:**

| Category | Count | Examples |
|----------|-------|----------|
| File Operations | 6 | read_file, write_file, find_replace, search_files |
| Git Operations | 7 | commit_changes, create_branch, get_status |
| Maven/POM | 15 | read_pom, update_java_version, list_dependencies |
| Command Execution | 9 | mvn_compile, mvn_test, run_command |
| OpenRewrite | 4 | mvn_rewrite_run, get_available_recipes |
| Web/Research | 2 | web_search_tool, call_openrewrite_agent |
| State Management | 4 | check_migration_state |
| Completion | 2 | mark_analysis_complete, mark_execution_complete |
| Handoff | 3 | guarded_analysis_handoff, guarded_execution_handoff |

## 9.2 Tool Wrapping

Every tool is wrapped to provide:

**1. Action Tracking:**
```
Tool: find_replace
Args: {"file": "pom.xml", "find": "<java.version>8", "replace": "<java.version>21"}
Result: SUCCESS - replaced 1 occurrence
Logged to: COMPLETED_ACTIONS.md
```

**2. Auto-Sync on Commit:**
When `commit_changes` or `git_commit` is called:
- Callback fires: `_on_commit_success()`
- System finds current task from VISIBLE_TASKS.md
- Auto-marks task [x] in TODO.md
- Regenerates VISIBLE_TASKS.md

**3. Access Control:**
| Agent | Blocked Tools |
|-------|---------------|
| analysis_expert | write_file (except state files) |
| execution_expert | None |
| error_expert | revert_test_files, full write_file |

## 9.3 Command Safety

The `run_command` tool has strict safety controls:

**Blocked Commands:**
```
rm, del, rmdir, sudo, chmod, chown, kill, shutdown,
docker, wget, curl, ssh, apt, pip, npm
```

**Blocked Patterns:**
```
rm -rf, >/dev/, command substitution $(), pipe to shell | sh
```

**Allowed Commands:**
```
mvn, git, java, javac, ls, cat, grep, find, echo, pwd
```

---

# Part 10: Test Preservation

## 10.1 The Test Preservation Problem

Migrations can accidentally:
- Delete test files
- Remove test methods
- Rename tests (breaks CI)
- Change test logic
- Remove @Test annotations

Any of these would break CI pipelines and is unacceptable.

## 10.2 Test Baseline Capture

Before execution phase starts:

```python
test_baseline = {
    "src/test/java/UserServiceTest.java": {
        "methods": ["testCreate", "testUpdate", "testDelete"],
        "count": 3
    },
    "src/test/java/OrderServiceTest.java": {
        "methods": ["testOrder", "testCancel"],
        "count": 2
    }
}
total_test_methods = 5
```

## 10.3 Preservation Rules

**FORBIDDEN (will trigger error_expert):**
- Delete test files
- Delete test methods
- Rename test methods
- Remove @Test annotations
- Change test assertions logic

**ALLOWED:**
- Update imports: `org.junit.Test` → `org.junit.jupiter.api.Test`
- Update assertions: `Assert.assertEquals` → `Assertions.assertEquals`
- Update annotations: `@Before` → `@BeforeEach`
- Update runners: Remove `@RunWith(SpringRunner.class)`

## 10.4 Final Validation

Before declaring SUCCESS:

```python
def verify_final_test_invariance():
    current_test_count = count_test_methods(project)

    if current_test_count != baseline_test_count:
        return FAILURE, f"Test count changed: {baseline} → {current}"

    return SUCCESS, "Test invariance maintained"
```

**Hard Rule:** If test count changed, result is FAILURE regardless of other success indicators.

---

# Part 11: Exit Conditions Summary

## 11.1 Success Conditions

| Condition | Result |
|-----------|--------|
| ≥90% tasks complete + build passes + tests pass + test count unchanged | SUCCESS |
| ≥50% tasks complete + build passes + tests pass + test count unchanged | PARTIAL_SUCCESS |

## 11.2 Failure Conditions

| Condition | Result |
|-----------|--------|
| error_count ≥ 3 (error expert failed 3 times) | FAILURE |
| stuck_loop_attempts ≥ 3 (stuck recovery failed 3 times) | FAILURE |
| Build fails at final validation | FAILURE |
| Tests fail at final validation | FAILURE |
| Test count changed from baseline | FAILURE |
| <50% tasks complete | FAILURE |

## 11.3 Incomplete Conditions

| Condition | Result |
|-----------|--------|
| LLM call limit reached (500 calls) | INCOMPLETE |
| Execution timeout (200 loops) | INCOMPLETE |
| Workflow recursion limit (500 iterations) | INCOMPLETE |

## 11.4 Circuit Breaker Triggers

| Breaker | Limit | Action |
|---------|-------|--------|
| LLM Calls | 500 | Raise LLMCallLimitExceeded |
| Execution Loops | 200 | Set EXECUTION_TIMEOUT phase |
| Error Attempts | 3 | Route to FAILED |
| Stuck Attempts | 3 | Route to FAILED |
| Context Tokens | 140,000 | Prune messages to 30,000 |
| Analysis No-Tool | 5 loops | Reset context (max 2 resets) |

---

# Part 12: Architectural Innovations

## 12.1 Compiled Context Pattern

**Traditional Approach:** Accumulate all messages in conversation history
- Problem: Context grows unbounded, pruning loses information

**Our Approach:** Compile fresh context each loop from state files
- Read VISIBLE_TASKS.md for current task
- Read COMPLETED_ACTIONS.md for history
- Read CURRENT_STATE.md for status
- Result: ~2KB fresh context vs 100KB accumulated

**Benefits:**
- No context bloat
- No amnesia after pruning
- Consistent agent behavior
- Lower token costs

## 12.2 Auto-Sync on Commit

**Problem:** Agents forget to mark tasks complete after doing work

**Solution:** Commit tools trigger automatic task marking

```
Agent calls: commit_changes("pom.xml updated")
                    │
                    ▼
            Callback fires: _on_commit_success()
                    │
                    ▼
            Extract current task from VISIBLE_TASKS.md
                    │
                    ▼
            Mark [x] in TODO.md via find_replace
                    │
                    ▼
            Regenerate VISIBLE_TASKS.md
```

**Benefits:**
- Work and tracking always synchronized
- No "I did the work but forgot to mark it" bugs
- Agent sees updated tasks next loop

## 12.3 Signature-Based Loop Detection

**Traditional:** Count how many times same tool called
- Problem: Flags healthy repetition (different files each time)

**Our Approach:** Hash arguments + categorize results
- Only flag when SAME tool + SAME args + SAME failure 3x
- Allow healthy variation (different args = OK)

**Benefits:**
- No false positives
- Catches actual stuck patterns
- Allows legitimate repetition

## 12.4 Intelligent Web Search

**Traditional:** Raw queries to search API
- Problem: Stack traces don't make good queries

**Our Approach:** Query optimization + context enhancement + deduplication
- Clean stack trace noise
- Add version context
- Add solution keywords
- Cache results to prevent duplicate API calls

**Benefits:**
- Better search results
- Lower API costs
- No duplicate searches

## 12.5 Test Preservation Guarantees

**Traditional:** Hope agents don't delete tests

**Our Approach:** Mathematical guarantee via baseline comparison
- Capture exact test count before migration
- Compare after migration
- Any difference = FAILURE

**Benefits:**
- Cannot accidentally break CI
- Measurable guarantee
- Human-verifiable

---

# Part 13: Observability and Debugging

## 13.1 Three-Stream Logging

| Log File | Contents | Use Case |
|----------|----------|----------|
| `llm_interactions_{timestamp}.log` | Full LLM prompts and responses | Debug agent behavior |
| `multiagent_process_{timestamp}.log` | Agent events, routing, tool calls | Debug workflow |
| `summary_{timestamp}.log` | High-level decisions, results | Executive overview |

## 13.2 State File Inspection

At any point, human can read:
- `TODO.md`: What's done, what's pending
- `COMPLETED_ACTIONS.md`: Every tool call with timestamp
- `CURRENT_STATE.md`: Project status
- `ERROR_HISTORY.md`: All errors and fix attempts

## 13.3 Token Usage Report

End of migration outputs:
```
TOKEN USAGE & COST REPORT
=========================
LLM Calls:       247
Prompt tokens:   145,321
Response tokens: 53,456
Total tokens:    198,777
Estimated cost:  $1.24
```

## 13.4 Search Statistics

```
[SEARCH_PROC] Search complete | 8432 chars | Cache: 8/24
```
- 24 total searches
- 8 were cache hits (33% saved)

---

# Part 14: Typical Migration Example

## 14.1 Sample Repository

**Project:** spring-boot-demo (single-module)
- Java 8
- Spring Boot 2.5.0
- JUnit 4
- 15 javax.* imports

## 14.2 Analysis Phase Output

**TODO.md Created:**
```markdown
# Migration Tasks for spring-boot-demo

## Phase 1: Java Version
- [ ] Update Java version from 8 to 21 in pom.xml

## Phase 2: Spring Boot
- [ ] Update Spring Boot parent from 2.5.0 to 3.2.0
- [ ] Add jakarta.servlet dependency

## Phase 3: Namespace Migration
- [ ] Run Jakarta namespace migration recipe
- [ ] Fix any remaining javax.* imports

## Phase 4: JUnit Migration
- [ ] Run JUnit 4 to JUnit 5 recipe
- [ ] Update test annotations

## Phase 5: Verification
- [ ] Run mvn compile and fix errors
- [ ] Run mvn test and verify all pass
```

## 14.3 Execution Phase Flow

```
Loop 1: VISIBLE_TASKS shows "Update Java version"
        → update_java_version(21)
        → mvn_compile (verify)
        → commit_changes → AUTO_SYNC marks [x]

Loop 2: VISIBLE_TASKS shows "Update Spring Boot parent"
        → find_replace(pom.xml, "2.5.0", "3.2.0")
        → mvn_compile
        → BUILD FAILURE: jakarta.servlet not found

Loop 3: Routes to error_expert
        → Diagnoses: Missing Jakarta servlet
        → Adds dependency
        → mvn_compile: SUCCESS
        → Returns to execution_expert

Loop 4: VISIBLE_TASKS shows "Add jakarta.servlet dependency"
        → Already done by error_expert
        → Skip and mark [x]

... continues until all tasks complete
```

## 14.4 Error Resolution with Web Search

```
Loop 8: VISIBLE_TASKS shows "Fix CGLIB compatibility"
        → mvn_compile
        → BUILD FAILURE: NoClassDefFoundError cglib

Loop 9: error_expert Attempt 1
        → Adds cglib dependency
        → mvn_compile: STILL FAILS

Loop 10: error_expert Attempt 2 (MANDATORY WEB SEARCH)
        → web_search_tool("NoClassDefFoundError cglib Spring Boot Java 21 fix")
        → SearchProcessor:
            - Cleans query
            - Adds context
            - Checks cache (miss)
            - Executes Tavily search
            - Returns: "Spring Boot 3.x uses spring-core proxy, remove cglib dependency"
        → Removes cglib, adds spring-core
        → mvn_compile: SUCCESS!

Loop 11: Back to execution_expert
        → Continues with next task
```

## 14.5 Final Result

```
Migration Complete!
==================
Status: SUCCESS
Progress: 9/9 tasks (100%)
Build: PASSING
Tests: 45/45 PASSING
Test count: PRESERVED (45 before, 45 after)
LLM calls: 127
Cost: $0.82
Web searches: 5 (2 cache hits)
```

---

# Conclusion

The Java Migration Framework represents a mature, production-ready approach to multi-agent LLM orchestration. Its key strengths are:

1. **Deterministic Behavior:** Routing and classification use code, not LLM decisions
2. **Cost Control:** Hard limits prevent runaway spending
3. **Test Safety:** Mathematical guarantees on test preservation
4. **Smart Loop Detection:** Signature-based detection prevents stuck patterns
5. **Intelligent Web Search:** Query optimization + deduplication saves API costs
6. **Graceful Degradation:** Partial success possible when limits hit
7. **Full Observability:** Three-stream logging and state files for debugging
8. **Modular Architecture:** Components can be tested and replaced independently

The system successfully balances the power of LLM agents with the control and reliability requirements of enterprise software migration.

---

*Document generated for presentation purposes. For technical implementation details, see CLAUDE.md and source code.*

# Issue: Test Preservation Verifier Gap - Lifecycle Methods Not Tracked

**Date:** December 13, 2025  
**Repo:** `lkrnac/blog-2014-12-06-mock-spring-bean`  
**Status:** Migration marked SUCCESS, but MigrationBench verification FAILED

---

## Summary

The test preservation commit blocker allowed commits that deleted `@After` lifecycle methods because it only tracks `@Test` annotated methods. MigrationBench's final verification caught the issue, but by then the damage was done.

---

## What Happened

### The Original Test File (`AddressServiceITest.java`)

```java
@RunWith(SpringJUnit4ClassRunner.class)
@SpringApplicationConfiguration(classes = { Application.class, AddressDaoMock.class })
public class AddressServiceITest {
    @Autowired
    private AddressService addressService;

    @Autowired
    private AddressDao addressDao;

    @Test
    public void testGetAddressForUser() {
        when(addressDao.readAddress("john")).thenReturn("5 Bright Corner");
        String actualAddress = addressService.getAddressForUser("john");
        Assert.assertEquals("5 Bright Corner", actualAddress);
    }

    @After
    public void resetMock() {
        reset(addressDao);  // <-- THIS METHOD WAS DELETED
    }
}
```

### After Migration (Modified)

```java
@SpringBootTest
@TestPropertySource(properties = "spring.main.allow-bean-definition-overriding=true")
public class AddressServiceITest {
    @Autowired
    private AddressService addressService;

    @MockBean
    private AddressDao addressDao;

    @Test
    public void testGetAddressForUser() {
        when(addressDao.getAddress("john")).thenReturn("3 Dark Corner");
        String actualAddress = addressService.getAddressForUser("john");
        assertEquals("3 Dark Corner", actualAddress);
    }
    // @After resetMock() method was DELETED
}
```

---

## Why The Commit Blocker Didn't Catch It

### Baseline Captured (`.test_baseline.json`)

```json
{
  "files": {
    "AddressServiceITest.java": {
      "method_count": 1,
      "methods": [
        {"name": "testGetAddressForUser", "annotations": ["@Test"]}
      ]
    }
  }
}
```

**Notice:** `resetMock()` with `@After` annotation was **never captured** in the baseline!

### Why?

The `TEST_METHOD_PATTERNS` in `test_verifier.py` only matches:
```python
TEST_METHOD_PATTERNS = [
    r'@Test\s*...',
    r'@ParameterizedTest\s*...',
    r'@RepeatedTest\s*...',
    r'@TestFactory\s*...',
]
```

It does **NOT** track:
- `@Before` / `@BeforeEach`
- `@After` / `@AfterEach`
- `@BeforeAll` / `@AfterAll`

---

## The Gap

| Check | What It Tracks | lkrnac Result |
|-------|----------------|---------------|
| **Commit Blocker** (`verify_test_preservation_before_commit`) | Only `@Test` methods | ✅ PASS (3→3 methods) |
| **MigrationBench** (`same_repo_test_files`) | Full test structure | ❌ FAIL (detected deletion) |

The commit blocker saw:
- Baseline: 3 test methods (`testGetAddressForUser`, `testGetUserDetails`, `contextLoads`)
- Current: 3 test methods (same names)
- Result: **"All good!"** ✅

But MigrationBench saw the actual file content changed significantly.

---

## Timeline

```
07:31:30 | TEST BASELINE CAPTURED: 3 test files (only @Test methods tracked)
07:3x:xx | Agent modifies test files (deletes @After methods, rewrites mock setup)
07:3x:xx | Commit blocker: "✅ Test preservation verified: 3 methods intact"
08:07:07 | MigrationBench: "❌ Test mismatch for files (len = 001/005)"
```

---

## Recommended Fixes

### Option 1: Track Lifecycle Methods Too
Add lifecycle annotations to `TEST_METHOD_PATTERNS`:
```python
TEST_METHOD_PATTERNS = [
    r'@Test\s*...',
    r'@Before\s*...',
    r'@After\s*...',
    r'@BeforeEach\s*...',
    r'@AfterEach\s*...',
    # ... etc
]
```

### Option 2: Use Content Hash Comparison
The baseline already captures `content_hash` but doesn't enforce it:
```json
"content_hash": "4238fed8bd55"
```

Add a check: if `content_hash` changed, block the commit (or at least warn).

### Option 3: Stricter File-Level Protection
Block ANY modifications to test files during migration, not just method deletions.

---

## Lessons Learned

1. **The commit blocker has a blind spot** - lifecycle methods can be deleted silently
2. **MigrationBench is the real source of truth** - it caught what the commit blocker missed
3. **Agent "fixed" tests by rewriting them** - should have preserved structure and only updated annotations
4. **Need alignment** between commit blocker and final verification logic

---

## Files Affected

- `src/test/java/net/lkrnac/blog/testing/mockbean/AddressServiceITest.java` - `@After resetMock()` deleted
- `src/test/java/net/lkrnac/blog/testing/mockbean/UserServiceITest.java` - mock setup rewritten
- `src/test/java/net/lkrnac/blog/testing/mockbean/ApplicationTests.java` - minor changes

---

## Related Code

- `migration/src/utils/test_verifier.py` - `TEST_METHOD_PATTERNS` (line ~94)
- `migration/src/tools/git_operations.py` - `commit_changes()` calls verifier
- `MigrationBench/src/migration_bench/lang/java/eval/parse_repo.py` - `same_repo_test_files()`

---
---

# Issue: Flaky AspectJ Build - False Positive Migration Success

**Date:** December 13, 2025  
**Repo:** `serpro69/kotlin-aspectj-maven-example`  
**Status:** Migration marked SUCCESS during execution, but verification FAILED intermittently

---

## Summary

The migration appeared successful during the agent's execution (tests passed), but the final MigrationBench verification failed with "Build failed with code 1". This is a **flaky build** caused by AspectJ weaving issues - the build can pass or fail on the same code depending on timing/caching.

---

## What Happened

### During Migration

The agent ran `mvn test` multiple times and saw:
```
Tests run: 2, Failures: 0, Errors: 0, Skipped: 0
BUILD SUCCESS
```

### During Final Verification

MigrationBench ran the same `mvn test` and got:
```
Build failed with code 1
Unable to determine test count
```

---

## Root Cause: jcabi-maven-plugin Uses Bundled AspectJ 1.9.1

The project uses `jcabi-maven-plugin` for AspectJ weaving. The problem:

### The pom.xml specifies AspectJ 1.9.25:
```xml
<aspectj.version>1.9.25</aspectj.version>
```

### But jcabi-maven-plugin bundles AspectJ 1.9.1 internally:

From the AspectJ dump file (`ajcore.20251213.070811.473.txt`):
```
---- AspectJ Properties ---
AspectJ Compiler 1.9.1 built on Friday Apr 20, 2018
```

### The Version Mismatch Causes:

1. AspectJ 1.9.1 doesn't fully support Java 21 bytecode
2. Weaving can fail intermittently depending on class loading order
3. The `BCException` in the dump file shows weaving failures

---

## The Flaky Behavior

| Run | Result | Why |
|-----|--------|-----|
| Agent's `mvn test` #1 | ✅ PASS | Cached classes, no re-weaving |
| Agent's `mvn test` #2 | ✅ PASS | Same cache |
| MigrationBench verification | ❌ FAIL | Clean build, AspectJ re-weaves, fails |

The build is **non-deterministic** because:
- AspectJ weaving depends on class file timestamps
- Maven's incremental compilation can skip weaving
- Clean builds force full weaving, exposing the version mismatch

---

## Evidence

### AspectJ Dump File Contents

```
---- Exception Information ---
java.lang.RuntimeException: BCException
  at org.aspectj.weaver.bcel.BcelWorld.resolve(BcelWorld.java:...)
  
---- AspectJ Properties ---
AspectJ Compiler 1.9.1 built on Friday Apr 20, 2018  <-- OLD VERSION
```

### The Aspect Being Woven

```kotlin
// FellowshipOfTheRingAspect intercepts getRingBearer()
@Around("execution(* FellowshipOfTheRing.getRingBearer(..))")
fun interceptRingBearer(): String = "Frodo"
```

### Test That Depends on Weaving

```java
@Test
void testFrodo() {
    assertEquals("Frodo", new FellowshipOfTheRing().getRingBearer());
    // Fails if aspect not woven: returns "council meeting is ongoing..."
}
```

---

## Why This Is a False Positive

The migration was marked "SUCCESS" because:
1. Tests passed during agent execution (cached build)
2. Commit blocker saw passing tests
3. Agent generated MigrationReport.md

But the migration is **actually broken** because:
1. The build is flaky
2. Clean builds fail
3. AspectJ weaving is incompatible with Java 21

---

## Recommended Fixes

### Option 1: Update jcabi-maven-plugin
Check if a newer version bundles AspectJ 1.9.25+:
```xml
<plugin>
    <groupId>com.jcabi</groupId>
    <artifactId>jcabi-maven-plugin</artifactId>
    <version>0.17.0</version>  <!-- or newer -->
</plugin>
```

### Option 2: Use aspectj-maven-plugin Instead
Replace jcabi with the official AspectJ plugin:
```xml
<plugin>
    <groupId>org.codehaus.mojo</groupId>
    <artifactId>aspectj-maven-plugin</artifactId>
    <version>1.14.0</version>
    <configuration>
        <complianceLevel>21</complianceLevel>
    </configuration>
</plugin>
```

### Option 3: Force Clean Builds in Verification
Always run `mvn clean test` instead of `mvn test` to catch flaky builds early.

---

## Lessons Learned

1. **Passing tests during migration ≠ stable build** - flaky builds can fool the agent
2. **AspectJ + Java 21 = version compatibility issues** - bundled AspectJ in plugins may be outdated
3. **Final verification should use clean builds** - to catch caching-related false positives
4. **Check plugin internals** - jcabi-maven-plugin's bundled AspectJ version matters

---

## Files Affected

- `pom.xml` - Updated `aspectj.version` to 1.9.25 (but jcabi ignores it)
- `ajcore.*.txt` - AspectJ dump files showing BCException
- Test classes depend on aspect weaving working correctly

---

## Related Code

- `jcabi-maven-plugin` - Bundles old AspectJ compiler
- `src/main/kotlin/.../DeepThoughtAspect.kt` - Kotlin aspect
- `src/main/java/.../FellowshipOfTheRingAspect.java` - Java aspect
- `src/test/java/.../JavaTest.java` - Test that fails when weaving breaks

---
---

# Issue: Kotlin jvmTarget Not Updated in Multi-Module Project

**Date:** December 13, 2025  
**Repo:** `m1l4n54v1c/meetup`  
**Status:** Migration completed, but verification FAILED with Java version mismatch

---

## Summary

In a multi-module Maven project with Kotlin, the agent updated the parent `<java.version>` property but missed the Kotlin plugin's `<jvmTarget>` configuration in a submodule. This resulted in mixed bytecode versions (Java 8 + Java 21) causing verification failure.

---

## What Happened

### Verification Error

```
Java version mismatch: expected 65, got {65, 52}
```

| Bytecode | Java Version |
|----------|--------------|
| 65 | Java 21 ✅ |
| 52 | Java 8 ❌ |

### Class File Analysis

```
./meetup-api/target/classes/.../CommentOnTopicCommand.class: version 52.0 (Java 1.8)  ❌
./meetup-api/target/classes/.../CreateMeetupCommand.class: version 52.0 (Java 1.8)    ❌
./meetup-query/target/classes/.../MeetupComment.class: version 65.0 (Java 21)         ✅
./meetup-command/target/classes/.../Meetup.class: version 65.0 (Java 21)              ✅
```

The `meetup-api` module (Kotlin) compiled to Java 8, while `meetup-query` and `meetup-command` (Java) compiled to Java 21.

---

## Root Cause

### Parent pom.xml (Updated Correctly)

```xml
<properties>
    <java.version>21</java.version>  ✅
</properties>
```

### meetup-api/pom.xml (Missed)

```xml
<plugin>
    <groupId>org.jetbrains.kotlin</groupId>
    <artifactId>kotlin-maven-plugin</artifactId>
    <configuration>
        <jvmTarget>1.8</jvmTarget>  <!-- ❌ Should be 21 -->
    </configuration>
</plugin>
```

The Kotlin plugin's `jvmTarget` does **NOT** inherit from `<java.version>`. It must be explicitly set.

---

## Why The Agent Missed It

1. Agent searched for `java.version` property → found and updated in parent pom.xml ✅
2. Agent didn't search for `jvmTarget` in Kotlin plugin configurations ❌
3. Multi-module projects require checking **all** submodule pom.xml files

---

## The Fix

Should have updated `meetup-api/pom.xml`:

```xml
<configuration>
    <jvmTarget>21</jvmTarget>
</configuration>
```

Or better, use a property:

```xml
<configuration>
    <jvmTarget>${java.version}</jvmTarget>
</configuration>
```

---

## Additional Issue: Multi-Module Dependency Resolution

The verification also failed with:

```
Could not find artifact io.axoniq.demo:meetup-api:jar:0.0.1-SNAPSHOT
```

This is because `mvn dependency:resolve` doesn't work well with multi-module projects - inter-module dependencies need `mvn install` first. This is a **verification limitation**, not a migration failure.

---

## Lessons Learned

1. **Kotlin projects need special handling** - `jvmTarget` must be explicitly updated
2. **Multi-module projects need comprehensive scanning** - check ALL submodule pom.xml files
3. **Java version property doesn't propagate to Kotlin** - they're separate configurations
4. **OpenRewrite RAG agent would help** - it knows about Kotlin + Java version alignment
5. **Verification needs multi-module awareness** - `mvn install` before `mvn dependency:resolve`

---

## Detection Pattern

When migrating, always search for:
- `<jvmTarget>` in kotlin-maven-plugin
- `<release>` or `<target>` in maven-compiler-plugin
- Any hardcoded Java version strings like `1.8`, `11`, `17`

---

## Files Affected

- `pom.xml` - Parent with `<java.version>21</java.version>` ✅
- `meetup-api/pom.xml` - Kotlin module with `<jvmTarget>1.8</jvmTarget>` ❌
- `meetup-query/pom.xml` - Java module (inherits correctly) ✅
- `meetup-command/pom.xml` - Java module (inherits correctly) ✅

---

## Related Code

- `kotlin-maven-plugin` - Has separate `jvmTarget` configuration
- `maven-compiler-plugin` - Uses `<release>` or `<source>/<target>`
- Multi-module reactor builds require `mvn install` for inter-module deps

---
---

# CRITICAL: Maven Incremental Compilation Hiding Breaking Changes

**Date:** December 13, 2025  
**Repo:** `software-space/qiyana-aggregator`  
**Status:** Migration marked SUCCESS, but build is **completely broken**  
**Severity:** 🔴 **CRITICAL** - Agent reports false success on broken builds

---

## Summary

Maven's incremental compilation allows builds to pass using **cached .class files** even after dependency changes introduce breaking API changes. The agent reports "all tests passing" but the code doesn't actually compile.

**This is not the agent lying - Maven is lying to the agent.**

---

## What Happened

### 1. Agent Updated Dependency

```xml
<!-- Before -->
<version>4.0.0-rc4</version>

<!-- After -->
<version>4.0.0-rc8</version>
```

The `orianna` library removed enum values in the newer version:
- `RANKED_SOLO_5x5`, `RANKED_FLEX_SR`, `TEAM_BUILDER_RANKED_SOLO`, etc.

### 2. Agent Ran `mvn test`

```
[INFO] --- compiler:3.11.0:compile (default-compile) @ data-aggregator ---
[INFO] Nothing to compile - all classes are up to date   <-- THE LIE
```

Maven's incremental compiler saw existing `.class` files and **skipped recompilation**.

### 3. Tests Passed Using OLD Classes

The tests ran using:
- OLD `.class` files (compiled against `orianna:4.0.0-rc4`)
- NEW `orianna:4.0.0-rc8` JAR (at runtime, but enums not used in tests)

Result: **Tests passed!**

### 4. Agent Committed "Success"

```
10:12:35 | [ERROR_CLASSIFIER] SUCCESS (return code 0, BUILD SUCCESS pattern found)
10:12:35 | MVN_COMMAND: Success
...
10:14:20 | FINAL RESULT: SUCCESS
```

### 5. MigrationBench Runs Clean Build → FAILS

```
mvn clean test
```

Forces recompilation → picks up NEW orianna → **20+ compilation errors**

```
[ERROR] cannot find symbol
[ERROR]   symbol:   variable RANKED_SOLO_5x5
[ERROR]   location: class com.merakianalytics.orianna.types.common.Queue
```

---

## The Evidence

From the logs:

```
10:12:32 | MVN_COMMAND: Executing main command: mvn test -B
10:12:35 | [ERROR_CLASSIFIER] SUCCESS (return code 0, BUILD SUCCESS pattern found)
```

But when we run `mvn clean test` now:

```
[ERROR] COMPILATION ERROR
[ERROR] cannot find symbol: variable RANKED_SOLO_5x5
[ERROR] cannot find symbol: variable RANKED_FLEX_SR
... (20+ more errors)
```

---

## Why This Is A Systemic Bug

This is the **SAME ROOT CAUSE** as the AspectJ flaky build:

| Issue | Caching Layer | Agent Sees | Reality |
|-------|---------------|------------|---------|
| AspectJ | Aspect weaving cache | Tests pass | Weaving broken |
| **This** | Maven incremental compile | Tests pass | Code doesn't compile |

**Maven's caching mechanisms are designed for developer productivity, but they hide breaking changes from automated migration tools.**

---

## The Fix Required

### Immediate: Force Clean Builds After Dependency Changes

When the agent modifies `pom.xml` (especially `<version>` tags), the next build MUST be:

```bash
mvn clean test   # NOT just "mvn test"
```

### In Code

The `mvn_test` tool should detect pom.xml changes and automatically add `clean`:

```python
def mvn_test(project_path: str, force_clean: bool = False) -> str:
    # Check if pom.xml was modified since last build
    if pom_modified_since_last_build() or force_clean:
        cmd = "mvn clean test -B"
    else:
        cmd = "mvn test -B"
```

### Or: Always Use Clean Builds

Simpler but slower - always run `mvn clean test` instead of `mvn test`.

---

## Detection Pattern

**Red flags that indicate this bug:**
1. `[INFO] Nothing to compile - all classes are up to date` after pom.xml changes
2. Agent commits with "tests passing" after dependency version bump
3. MigrationBench verification fails with compilation errors

---

## Lessons Learned

1. **Maven incremental compilation is dangerous for migrations** - it hides breaking changes
2. **Dependency updates MUST trigger clean builds** - cached classes are stale
3. **"Tests passing" after pom.xml change is suspicious** - should see recompilation
4. **The agent isn't lying** - it's reporting what Maven told it
5. **MigrationBench clean builds are essential** - they catch what incremental builds miss

---

## Files Affected

- `pom.xml` - Updated `orianna` from `4.0.0-rc4` to `4.0.0-rc8`
- `MatchesCollectionService.java` - References removed enum values
- All compilation now fails due to missing symbols

---

## Related Code

- `migration/src/tools/maven_api.py` - `mvn_test()` should force clean after pom changes
- `migration/src/orchestrator/error_handler.py` - SUCCESS classification is correct (Maven returned 0)
- Maven's `maven-compiler-plugin` - Incremental compilation logic

---

## Action Items

1. **URGENT**: Modify `mvn_test` to use `mvn clean test` after any pom.xml modification
2. **URGENT**: Add detection for "Nothing to compile" after dependency changes → force clean
3. Consider always using `mvn clean test` in migration context (safety over speed)

---
---

# CRITICAL: Analysis Agent Stuck in Infinite Text Loop - No Tool Calls

**Date:** December 13, 2025  
**Repo:** `epignosisx/spring-cloud-config-server-jwt`  
**Status:** Migration STUCK - agent generating text but not executing tools  
**Severity:** 🔴 **CRITICAL** - Agent burns tokens doing nothing

---

## Summary

The **analysis_expert** agent got stuck in an infinite loop where it generates verbose migration plans and "polite acknowledgments" but **never calls any tools**. After 40+ LLM calls and 14+ minutes, zero actual work was done.

---

## What Happened

### The Evidence

| Metric | Expected | Actual |
|--------|----------|--------|
| LLM Responses | ~5-10 for analysis | **40+** |
| Tool Calls | Multiple (read_file, etc.) | **0** |
| TODO.md created | Within 1-2 minutes | **Never** |
| Time elapsed | ~5 minutes | **14+ minutes** |

### The Loop Pattern

```
10:37:05 | "Certainly! I'll coordinate the migration process..."
10:38:00 | "Thank you for providing the analysis... Let's proceed..."
10:38:42 | "Thank you for providing the detailed migration plan. Let's proceed..."
10:39:17 | "Thank you for providing the detailed migration guide. Let's proceed..."
10:39:57 | "Thank you for providing the detailed migration guide. Let's proceed..."
10:40:47 | "Thank you for providing the detailed migration plan. Let's proceed..."
10:41:29 | "Thank you for providing the detailed migration plan. Let's proceed..."
10:42:12 | "Thank you for providing the detailed migration guide. Let's proceed..."
... (repeating every ~40 seconds)
```

The agent keeps saying **"Thank you... Let's proceed..."** but never actually proceeds with tool calls.

---

## Root Cause

The LLM is generating **text responses** instead of **tool calls**. Possible causes:

1. **Prompt not forcing tool use** - Agent allowed to respond with text instead of tools
2. **Self-conversation loop** - Agent responding to its own generated "plans" as if they're messages
3. **Model behavior** - LLM pattern-matching on "migration" and outputting verbose guides
4. **No tool call enforcement** - System accepts text-only responses without intervention

---

## The Logs

### LLM Interactions Log (23,192 lines, 0 tool calls)

```
2025-12-13 10:37:05.301 | Choice 1: Certainly! I'll coordinate the migration process...
2025-12-13 10:38:00.924 | Choice 1: Thank you for providing the analysis... Let's proceed...
2025-12-13 10:38:42.865 | Choice 1: Thank you for providing the detailed migration plan...
2025-12-13 10:39:17.170 | Choice 1: Thank you for providing the detailed migration guide...
```

### What the Agent Generated (But Never Executed)

```xml
<step>
1. Navigate to Project Directory
<command>
cd /Users/xfmlc5g/Repos/deepwiki-experiment/migration/repositories/...
</command>
</step>

<step>
2. Check Java Version
<command>
java -version
</command>
</step>
```

The agent wrote out commands **in text** instead of **calling tools to execute them**.

---

## Why This Is Critical

1. **Token waste** - 40+ LLM calls doing nothing = ~$1-2 burned
2. **Time waste** - 14+ minutes with zero progress
3. **Silent failure** - No error raised, migration appears "running"
4. **Undetected** - Without manual log inspection, this looks like slow progress

---

## Detection Pattern

**Red flags that indicate this bug:**
1. `llm_interactions.log` growing but `multiagent_process.log` not updating
2. LLM response count >> tool call count
3. TODO.md not created after 2+ minutes
4. Repeated phrases like "Thank you for providing..." or "Let's proceed..."

---

## The Fix Required

### 1. Force Tool Calls in Analysis Phase

```python
# If agent returns text-only response in analysis phase, retry with tool enforcement
if not response.tool_calls and phase == "analysis":
    inject_message("You MUST call a tool. Do not respond with text. Call read_file or list_dir NOW.")
```

### 2. Detect Repetitive Text Patterns

```python
# Track last N responses, detect loops
if is_repetitive_pattern(last_5_responses):
    log_error("Agent stuck in text loop")
    raise AgentLoopException()
```

### 3. Tool Call Timeout

```python
# If no tool calls after N LLM responses, abort
if llm_call_count > 10 and tool_call_count == 0:
    raise NoToolCallsException("Agent not calling tools")
```

### 4. Dynamic Tool Binding (from LangGraph research)

Force agents to use tools by removing the ability to respond with text-only:
- Analysis agent: Only has analysis tools, MUST call them
- Execution agent: Only has execution tools, MUST call them

---

## Lessons Learned

1. **Cannot rely on prompts alone** - Agent ignores "you must call tools" instructions
2. **Need architectural enforcement** - Physically prevent text-only responses
3. **Monitor tool call rate** - 0 tool calls after multiple LLM calls = stuck
4. **Log inspection is essential** - Silent failures hide in verbose logs

---

## Related Issues

This is related to the previously documented **execution agent loop problem** where:
- Execution agent repeated analysis work 30+ times
- Phase-awareness messages had no effect
- Solution requires dynamic tool binding, not just prompts

See: `LANGGRAPH_SOLUTIONS_REFERENCE.md` for architectural fixes.

---

## Files Affected

- `logs/epignosisx__spring-cloud-config-server-jwt/llm_interactions_*.log` - 23K+ lines of text loops
- `logs/epignosisx__spring-cloud-config-server-jwt/multiagent_process_*.log` - Only 34 lines (stuck at start)
- No TODO.md created
- No actual migration work done

---

## Action Items

1. **URGENT**: Add tool call monitoring - abort if 0 tool calls after N LLM responses
2. **URGENT**: Detect repetitive response patterns and break loops
3. Implement dynamic tool binding to force tool use
4. Add real-time alerting when agent appears stuck

---
---

# CRITICAL: Error Agent Called on False Positive - Corrupts Working Code

**Date:** December 13, 2025  
**Repo:** `SimpleProgramming/spring-boot-ehCache-CacheManager`  
**Status:** Migration FAILED - error agent corrupted valid pom.xml  
**Severity:** 🔴 **CRITICAL** - Working build destroyed by "fix"

---

## Summary

The stuck loop detector triggered a **false positive** when the build was actually passing. The error agent was called to "fix" a non-existent problem and proceeded to **corrupt the pom.xml file**, breaking a working build.

**The error agent made things worse, not better.**

---

## Timeline

| Time | Event |
|------|-------|
| **11:00 - 11:06** | Build was ACTUALLY WORKING. `mvn test` passing. |
| **11:06:24** | ⚠️ STUCK LOOP DETECTED: "TODO item failed 3 times without success" |
| **11:07:23** | Error agent called (attempt 1/3) - **BUT THERE WAS NO REAL ERROR** |
| **11:07:59** | Error agent reads pom.xml (it's VALID at this point) |
| **11:08:13** | Error agent WRITES corrupted pom.xml |
| **11:08:43** | ERROR_CLASSIFIER still says SUCCESS (using cached results) |
| **11:08:43** | Migration FAILED after 3 stuck loop attempts |

---

## The Corruption

### Before (Valid pom.xml)
```xml
<name>api-programming-tips</name>
```

### After Error Agent "Fixed" It
```xml
<n>api-programming-tips</n>   <!-- CORRUPTED: <name> truncated to <n> -->
```

### Also Missing Version
```xml
<dependency>
    <groupId>jakarta.cache</groupId>
    <artifactId>cache-api</artifactId>
    <!-- NO VERSION TAG - Maven can't resolve -->
</dependency>
```

### Maven Error After Corruption
```
Malformed POM: Unrecognised tag: 'n' (position: START_TAG seen ...</version>\n\t<n>... @15:5)
'dependencies.dependency.version' for jakarta.cache:cache-api:jar is missing
```

---

## Root Cause Chain

```
1. STUCK LOOP DETECTION TRIGGERED INCORRECTLY
   ↓
   Build was passing, but loop detector thought "TODO item failed"
   ↓
2. ERROR AGENT CALLED WHEN NOTHING WAS BROKEN
   ↓
   Error agent sees "fix Jakarta changes" as the task
   ↓
3. ERROR AGENT TRIES TO "FIX" A NON-EXISTENT PROBLEM
   ↓
   Uses write_file to rewrite entire pom.xml
   ↓
4. ERROR AGENT CORRUPTS THE FILE
   ↓
   - Writes `<n>` instead of `<name>` (LLM output truncation)
   - Removes version from jakarta.cache dependency
   ↓
5. ERROR_CLASSIFIER DOESN'T DETECT THE CORRUPTION
   ↓
   Still says "SUCCESS" because using cached/incremental results
   ↓
6. MIGRATION FAILS AFTER 3 ATTEMPTS
```

---

## Why ERROR_CLASSIFIER Kept Saying SUCCESS

From the logs at **11:08:43**:
```
[ERROR_CLASSIFIER] SUCCESS (return code 0, BUILD SUCCESS pattern found)
MIGRATION FAILED: Stuck in loop for 3 attempts
```

The ERROR_CLASSIFIER never actually ran a fresh build after the corruption. It was:
1. Using cached incremental compilation results
2. Pattern matching on old output that still had "BUILD SUCCESS"
3. Not running `mvn clean` to force recompilation

---

## The Error Agent's Self-Report (Damning Evidence)

The error agent's own report admitted:
```
Compilation Status: Not verified (mvn_compile not called)
Test Results: Not verified (mvn_test not called)
```

**It claimed "COMPLETE" without ever verifying the build worked!**

---

## Three Bugs Combined

| Bug | Description |
|-----|-------------|
| **Stuck Loop False Positive** | Triggered when build was actually passing |
| **Error Agent Made It Worse** | "Fixed" a non-existent problem and corrupted the file |
| **No Verification After Changes** | Claimed success without running `mvn compile` |

---

## Why This Is Critical

1. **Destroys working code** - A passing build was broken
2. **Wastes time & money** - $1.82 spent making things worse
3. **Silent corruption** - No immediate error raised
4. **Cascading failure** - Each "fix" attempt made it worse

---

## The Fixes Required

### 1. Don't Call Error Agent on Passing Builds

```python
# Before calling error_expert, verify build is actually broken
result = run_mvn_clean_test()
if result.success:
    log("Build passing, stuck loop is false positive")
    continue_execution()  # Don't call error agent!
```

### 2. Error Agent MUST Verify After Changes

```python
# After error agent makes changes, ALWAYS verify
if error_agent_made_changes:
    result = run_mvn_clean_compile()
    if not result.success:
        revert_changes()
        log_error("Error agent broke the build, reverting")
```

### 3. Improve Stuck Loop Detection

```python
# Check if TODO is actually failing or just slow
if todo_marked_complete_recently:
    reset_stuck_counter()  # Not actually stuck
```

### 4. Validate XML Before Writing

```python
# Before writing pom.xml, validate it's well-formed
if not is_valid_xml(content):
    raise InvalidXMLError("Cannot write malformed pom.xml")
```

---

## Lessons Learned

1. **Stuck loop detection has false positives** - Need better heuristics
2. **Error agent can make things worse** - Must verify after changes
3. **"Fix" without verification is dangerous** - Always run build after changes
4. **LLM output can be truncated** - `<name>` became `<n>`
5. **write_file for entire files is risky** - Prefer targeted find_replace

---

## Files Affected

- `pom.xml` - Corrupted by error agent (tag truncation + missing version)
- Build completely broken after "fix"

---

## Action Items

1. **URGENT**: Error agent MUST run `mvn compile` after any file changes
2. **URGENT**: Don't call error agent if last build was successful
3. Add XML validation before writing pom.xml
4. Improve stuck loop detection to reduce false positives
5. Consider using find_replace instead of write_file for pom.xml changes

---
---

# CRITICAL: Agent Runs Blocking Server Commands - Infinite Timeout Loop

**Date:** December 13, 2025  
**Repo:** `jobmission/oauth2-client`  
**Status:** Migration STUCK - agent repeatedly tries to start web server  
**Severity:** 🔴 **CRITICAL** - Burns tokens on 5-minute timeout loops

---

## Summary

The execution agent interpreted a TODO step like "Verify application runs" as "start the Spring Boot server". It repeatedly runs `mvn spring-boot:run`, which is a **blocking command** that never completes. Each attempt times out after 5 minutes, then the agent tries again on a different port.

**The agent doesn't understand blocking vs non-blocking commands.**

---

## The Loop

| Time | Command | Result |
|------|---------|--------|
| 12:37:58 | `mvn spring-boot:run --server.port=10481` | Starts... |
| 12:42:58 | TIMEOUT after 300s | Agent confused |
| 12:43:17 | `mvn spring-boot:run --server.port=10481` | Tries again |
| 12:43:20 | SUCCESS (but blocking) | Server running |
| 12:43:28 | `mvn spring-boot:run --server.port=10482` | Tries new port |
| ... | Repeat forever | ∞ |

---

## The Evidence

```
190,359 lines in llm_interactions.log
```

The agent also tried:
- `curl -I http://localhost:10481` → BLOCKED (curl not allowed)
- `lsof -i :10481` → BLOCKED (lsof not allowed)
- Various `cd` + path combinations (confused about working directory)

---

## Root Cause

1. **TODO has vague verification step** - "Verify application starts" or similar
2. **Agent interprets literally** - Tries to actually start the server
3. **spring-boot:run is blocking** - Command never returns, runs server
4. **300s timeout triggers** - But agent doesn't learn, tries again
5. **Port already in use** - Agent tries different ports, still blocks

---

## Why This Is Different From Other Stuck Issues

| Issue Type | Behavior | Tool Calls |
|------------|----------|------------|
| Text Loop (epignosisx) | Generates "Thank you..." text | **0 tool calls** |
| Blocking Command (this) | Runs `spring-boot:run` | **Many tool calls** (wrong ones) |

The agent IS making tool calls, just the **wrong** ones.

---

## Commands That Should NEVER Be Run During Migration

```
mvn spring-boot:run          # Starts server (blocking)
mvn jetty:run                # Starts server (blocking)
mvn tomcat7:run              # Starts server (blocking)
java -jar *.jar              # Starts application (blocking)
./gradlew bootRun            # Starts server (blocking)
```

---

## The Fix Required

### 1. Blocklist Server-Starting Commands

```python
BLOCKED_COMMANDS = [
    'spring-boot:run',
    'jetty:run',
    'tomcat:run',
    'bootRun',
    '-jar',  # when running JARs
]

def validate_command(cmd):
    for blocked in BLOCKED_COMMANDS:
        if blocked in cmd:
            return False, f"Blocked: {blocked} is a server command that never completes"
    return True, None
```

### 2. Better TODO Wording

Instead of:
```
- [ ] Verify application starts
```

Use:
```
- [ ] Verify build passes with `mvn clean test`
```

### 3. Timeout Detection + Abort

```python
if command_timed_out and 'spring-boot:run' in command:
    log_error("Agent tried to start server - this is not a migration task")
    skip_to_next_todo_item()
```

### 4. Add to Prompt

```
NEVER run server-starting commands like:
- mvn spring-boot:run
- mvn jetty:run
- java -jar

These commands block forever. Use `mvn test` to verify the application works.
```

---

## Lessons Learned

1. **Agents interpret TODO items literally** - "verify it runs" → starts server
2. **Blocking commands need blocklist** - Can't trust agent judgment
3. **Timeout is not enough** - Agent just retries on timeout
4. **Verification = tests, not server startup** - Must be explicit

---

## Files Affected

- `logs/jobmission__oauth2-client/llm_interactions_*.log` - 190K+ lines
- Multiple 5-minute timeout cycles wasted
- ~$5+ in tokens burned on nothing

---

## Action Items

1. **URGENT**: Add blocklist for server-starting Maven goals
2. **URGENT**: Detect "spring-boot:run" timeout → abort, not retry
3. Update analysis agent to never generate "verify app starts" TODOs
4. Add explicit prompt instruction: "NEVER run spring-boot:run"

---
---

# CRITICAL: Analysis Agent "I Can't Execute" Text Loop

**Date:** December 13, 2025  
**Repo:** `spring-cloud-services-samples/traveler`  
**Status:** Migration STUCK - analysis agent refuses to act  
**Severity:** 🔴 **CRITICAL** - Agent believes it cannot take actions

---

## Summary

The analysis agent completed its work (created TODO.md, analysis.md, CURRENT_STATE.md) but then got stuck in a text loop saying **"As an AI language model, I don't have the ability to execute these commands"**. It doesn't know how to signal completion and hand off to the execution agent.

**The agent forgot it has tools and thinks it's just a chatbot.**

---

## The Loop Pattern

```
13:19:31 | "As an AI language model, I don't have the ability to execute..."
13:20:16 | "As an AI language model, I don't have the ability to execute..."
13:20:56 | "As an AI language model, I don't have the ability to execute..."
13:21:40 | "As an AI language model, I don't have the ability to execute..."
13:22:05 | "As an AI language model, I don't have the ability to execute..."
13:22:47 | "As an AI language model, I don't have the ability to execute..."
13:23:09 | "As an AI language model, I don't have the ability to execute..."
... (continues every ~40 seconds)
```

---

## Evidence

| Metric | Value |
|--------|-------|
| LLM interactions | **40,701 lines** |
| Phase | Stuck in **INIT** |
| Agent | `analysis_expert` |
| Last tool call | 13:17:10 |
| Tool calls after 13:17 | **0** |

---

## What Happened

```
13:10:16 | [WRAPPER] Running analysis_expert
13:14:15 | [ANALYSIS_WRITE] CURRENT_STATE.md ✅
13:14:26 | [ANALYSIS_WRITE] analysis.md ✅
13:14:39 | [ANALYSIS_WRITE] TODO.md ✅
13:16:43 | [ANALYSIS_BLOCK] Tried to write pom.xml → BLOCKED
13:17:10 | [ANALYSIS_WRITE] Updated TODO.md ✅
13:17:59 | [STATE_TOOL] Phase: INIT, Next: CALL_ANALYSIS_EXPERT
         | ... then pure text generation, no more tool calls ...
```

The analysis agent:
1. ✅ Did its job (created all required files)
2. ✅ Was correctly blocked from writing pom.xml
3. ❌ Doesn't know how to signal "I'm done"
4. ❌ Starts generating "I can't do anything" text

---

## Root Cause

1. **No clear completion signal** - Agent doesn't know how to end analysis phase
2. **Identity confusion** - Agent thinks it's a "language model" not an agent with tools
3. **Self-defeating loop** - Each response reinforces "I can't act"
4. **No phase transition** - Supervisor keeps calling analysis_expert

---

## Text Loop Variants Documented

| Variant | Repo | Pattern |
|---------|------|---------|
| "Thank you... relay to execution_expert" | epignosisx, hzpz | Politeness loop |
| **"I'm an AI, I can't execute"** | traveler | Identity crisis |
| "Thank you for the detailed plan" | hzpz | Acknowledgment loop |

All are the same underlying bug: **agent generates text instead of calling tools**.

---

## The Fix Required

### 1. Auto-Complete Analysis When Files Exist

```python
# If all required files exist, force transition
if all([
    file_exists("TODO.md"),
    file_exists("analysis.md"),
    file_exists("CURRENT_STATE.md")
]):
    force_phase_transition("ANALYSIS" -> "EXECUTION")
```

### 2. Detect "I can't" Pattern

```python
REFUSAL_PATTERNS = [
    "as an ai language model",
    "i don't have the ability",
    "i cannot execute",
    "should be done by the human",
]

if any(p in response.lower() for p in REFUSAL_PATTERNS):
    inject_message("You ARE an agent with tools. Call a tool NOW.")
```

### 3. Force Tool Calls

```python
if consecutive_text_responses > 3:
    raise AgentRefusalException("Agent refusing to use tools")
```

### 4. Better Prompt

```
You are NOT a passive language model. You are an AGENT with tools.
You MUST use your tools to complete tasks.
NEVER say "I can't execute" - you CAN and MUST execute using your tools.
```

---

## Lessons Learned

1. **Agents can forget their identity** - Start acting like chatbots
2. **"I can't" is a red flag** - Should trigger intervention
3. **Analysis completion needs explicit signal** - Can't rely on agent judgment
4. **All text loops share same root cause** - Agent not calling tools

---

## Files Affected

- `logs/spring-cloud-services-samples__traveler/llm_interactions_*.log` - 40K+ lines
- Analysis files were created successfully (TODO.md, etc.)
- Migration never progressed past analysis phase

---

## Action Items

1. **URGENT**: Detect "as an AI language model" pattern → force tool call
2. **URGENT**: Auto-complete analysis when required files exist
3. Add identity reinforcement to prompts
# Enhanced Web Search Architecture Plan

## Problem Analysis

Based on the log analysis of `lkrnac__blog-2014-12-06-mock-spring-bean`:

1. **Generic queries** - Agent searched for stack trace patterns instead of root causes
2. **No query optimization** - Raw error messages passed directly as queries
3. **Repeated identical searches** - 24 searches, many duplicates with same results
4. **Results not actionable** - Summaries were tangentially related, not specific
5. **No multi-hop reasoning** - Agent didn't decompose "CGLIB + JDK 17" into version compatibility query

## Architecture: Agentic Search Pipeline

Based on research from [Azure AI Search](https://learn.microsoft.com/en-us/azure/search/agentic-retrieval-overview), [Haystack Agentic RAG](https://haystack.deepset.ai/tutorials/36_building_fallbacks_with_conditional_routing), and [Tavily Best Practices](https://docs.tavily.com/documentation/best-practices/best-practices-search):

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ENHANCED WEB SEARCH TOOL                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   1. Query   │───▶│  2. Query    │───▶│  3. Multi-   │              │
│  │  Analyzer    │    │  Decomposer  │    │  Query Exec  │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│         │                   │                   │                        │
│         ▼                   ▼                   ▼                        │
│  - Detect query type   - Break into        - Execute in parallel        │
│  - Extract entities      sub-queries       - Deduplicate results        │
│  - Identify intent     - Rewrite for web   - Score & rank               │
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │  4. Result   │───▶│  5. Answer   │───▶│  6. Action   │              │
│  │  Grader     │    │  Synthesizer │    │  Extractor   │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│         │                   │                   │                        │
│         ▼                   ▼                   ▼                        │
│  - Check relevance     - Combine results   - Extract specific           │
│  - Trigger re-search   - Cite sources        fix commands               │
│    if low quality      - Summarize          - Provide code snippets     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Query Intelligence Layer

#### 1.1 Query Analyzer (Pre-processing)

Classify incoming queries and extract structured information:

```python
class QueryAnalyzer:
    """Analyze and classify incoming search queries."""

    QUERY_TYPES = {
        "error_diagnosis": ["Exception", "Error", "failed", "cannot", "NoClassDefFound"],
        "version_compatibility": ["version", "upgrade", "migrate", "compatibility", "JDK"],
        "how_to": ["how to", "how do I", "guide", "tutorial"],
        "api_usage": ["API", "method", "function", "class", "annotation"]
    }

    def analyze(self, query: str, context: dict = None) -> dict:
        """
        Analyze query and return structured metadata.

        Returns:
            {
                "query_type": "error_diagnosis",
                "entities": {
                    "framework": "Spring Boot",
                    "version": "1.1.9",
                    "java_version": "21",
                    "error_class": "NoClassDefFoundError",
                    "error_component": "cglib.proxy.Enhancer"
                },
                "intent": "fix_runtime_error",
                "complexity": "multi_hop"  # requires version compatibility check
            }
        """
```

#### 1.2 Query Decomposer

Based on [LangChain Query Decomposition](https://python.langchain.com/v0.1/docs/use_cases/query_analysis/techniques/decomposition/):

```python
class QueryDecomposer:
    """Decompose complex queries into focused sub-queries."""

    DECOMPOSITION_PROMPT = """You are a search query optimizer for Java/Spring migration issues.

Given this error or question, decompose it into 2-3 focused search queries.

RULES:
1. Queries must be AFFIRMATIVE sentences, not questions
2. Each query should be under 100 characters
3. Include version numbers when relevant
4. Focus on ROOT CAUSE, not symptoms

ERROR/QUESTION: {input}

CONTEXT (if available):
- Java version: {java_version}
- Spring Boot version: {spring_version}
- Error type: {error_type}

OUTPUT FORMAT (JSON):
{
    "root_cause_hypothesis": "Brief hypothesis of what's wrong",
    "sub_queries": [
        "query 1 - most specific to error",
        "query 2 - broader context/compatibility",
        "query 3 - alternative solution path"
    ]
}
"""

    def decompose(self, query: str, context: dict) -> list[str]:
        # Use fast LLM to decompose
        # Returns list of optimized sub-queries
```

**Example Transformation:**

| Original Query | Decomposed Sub-Queries |
|----------------|------------------------|
| `Spring Boot AbstractApplicationContext invokeBeanFactoryPostProcessors test failure fix solution` | 1. `Spring Boot 1.x CGLIB NoClassDefFoundError JDK 17 incompatibility` <br> 2. `Spring Boot version compatibility Java 21 minimum requirements` <br> 3. `Spring CGLIB proxy Enhancer initialization failure fix` |

#### 1.3 Query Rewriter

Transform queries for optimal web search (based on [HuggingFace Agentic RAG](https://huggingface.co/learn/cookbook/en/agent_rag)):

```python
class QueryRewriter:
    """Rewrite queries for optimal web search results."""

    REWRITE_PROMPT = """Convert this query to a web-search-optimized format.

RULES:
1. Use AFFIRMATIVE sentences (not questions)
2. Include technology names and versions
3. Add "fix" or "solution" for error queries
4. Keep under 100 characters
5. Remove stack trace noise, keep error class names

INPUT: {query}
OUTPUT: Single optimized search query
"""

    def rewrite(self, query: str) -> str:
        # For simple queries, apply rule-based optimization
        # For complex queries, use LLM rewriting
```

---

### Phase 2: Search Execution Layer

#### 2.1 Multi-Query Executor

Execute sub-queries in parallel with deduplication:

```python
class MultiQueryExecutor:
    """Execute multiple queries with deduplication and caching."""

    def __init__(self):
        self.cache = {}  # query_hash -> results
        self.seen_urls = set()

    async def execute(self, queries: list[str]) -> list[dict]:
        """
        Execute queries concurrently, deduplicate results.

        Uses Tavily's advanced search with:
        - search_depth="advanced" for query-aligned content
        - max_results=5 per query
        - include_answer=True for synthesis
        """
        results = await asyncio.gather(*[
            self._search_single(q) for q in queries
            if self._get_cache_key(q) not in self.cache
        ])

        # Deduplicate by URL
        unique_results = []
        for r in results:
            if r['url'] not in self.seen_urls:
                unique_results.append(r)
                self.seen_urls.add(r['url'])

        return self._rank_by_relevance(unique_results)
```

#### 2.2 Enhanced Tavily Integration

```python
def tavily_search_enhanced(query: str, context: dict) -> dict:
    """
    Enhanced Tavily search with optimal parameters.

    Based on: https://docs.tavily.com/documentation/api-reference/endpoint/search
    """
    client = TavilyClient(api_key=TAVILY_API_KEY)

    response = client.search(
        query=query,
        search_depth="advanced",      # Query-aligned content, not generic summaries
        max_results=5,
        include_answer=True,          # Get synthesized answer
        include_raw_content=False,    # Don't need full page (latency)
        topic="general",              # Could be "news" for recent issues
    )

    return response
```

---

### Phase 3: Result Processing Layer

#### 3.1 Result Grader

Based on [Haystack Agentic RAG](https://sajalsharma.com/posts/comprehensive-agentic-rag/):

```python
class ResultGrader:
    """Grade search results for relevance and trigger re-search if needed."""

    GRADING_PROMPT = """Grade these search results for relevance to the query.

QUERY: {query}
CONTEXT: Java {java_version}, Spring Boot {spring_version}

RESULTS:
{results}

For each result, score 1-5:
- 5: Directly answers the query with actionable fix
- 4: Highly relevant, provides useful context
- 3: Somewhat relevant, partial information
- 2: Tangentially related
- 1: Not relevant

OUTPUT (JSON):
{
    "scores": [{"url": "...", "score": N, "reason": "..."}],
    "overall_quality": "high|medium|low",
    "should_retry": true/false,
    "retry_suggestion": "alternative query if should_retry"
}
"""

    def grade(self, query: str, results: list, context: dict) -> dict:
        # Grade results
        # If overall_quality is "low", trigger re-search with retry_suggestion
```

#### 3.2 Answer Synthesizer

Combine multiple results into actionable guidance:

```python
class AnswerSynthesizer:
    """Synthesize search results into actionable fix instructions."""

    SYNTHESIS_PROMPT = """You are a Java migration expert. Synthesize these search results into actionable guidance.

ORIGINAL QUERY: {query}
PROJECT CONTEXT:
- Current Java version: {java_version}
- Current Spring Boot: {spring_version}
- Target: Java 21, Spring Boot 3.x

SEARCH RESULTS:
{results}

OUTPUT FORMAT:
## Root Cause
[1-2 sentences explaining the actual problem]

## Solution
[Step-by-step fix, be specific with version numbers]

## Code/Config Changes
```xml or java
[Specific changes if applicable]
```

## Sources
- [Source 1](url)
- [Source 2](url)
"""
```

#### 3.3 Action Extractor

Extract concrete actions for the agent:

```python
class ActionExtractor:
    """Extract actionable commands from synthesized answers."""

    def extract(self, synthesis: str) -> dict:
        """
        Extract structured actions.

        Returns:
            {
                "primary_action": "upgrade_spring_boot",
                "version_changes": {
                    "spring_boot": "2.7.18"  # or "3.2.0" if JDK 17+
                },
                "file_changes": [
                    {"file": "pom.xml", "change": "update parent version"}
                ],
                "commands": ["mvn clean install -U"],
                "confidence": 0.85
            }
        """
```

---

### Phase 4: Caching & Deduplication

#### 4.1 Search Cache

```python
class SearchCache:
    """Cache search results to avoid redundant API calls."""

    def __init__(self, ttl_seconds: int = 3600):
        self.cache = {}
        self.ttl = ttl_seconds

    def _normalize_query(self, query: str) -> str:
        """Normalize query for cache key."""
        # Lowercase, remove extra spaces, sort words
        words = sorted(query.lower().split())
        return " ".join(words)

    def get(self, query: str) -> Optional[dict]:
        key = self._normalize_query(query)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['results']
        return None

    def set(self, query: str, results: dict):
        key = self._normalize_query(query)
        self.cache[key] = {
            'results': results,
            'timestamp': time.time()
        }
```

#### 4.2 Session Deduplication

```python
class SessionDeduplicator:
    """Track searches within a session to avoid repetition."""

    def __init__(self):
        self.searched_queries = []
        self.searched_errors = set()  # Track error signatures

    def is_duplicate(self, query: str, error_signature: str = None) -> bool:
        """Check if this is a semantically duplicate search."""

        # Check exact error signature (e.g., "NoClassDefFoundError:cglib.proxy.Enhancer")
        if error_signature and error_signature in self.searched_errors:
            return True

        # Check semantic similarity with previous queries
        for prev_query in self.searched_queries:
            if self._semantic_similarity(query, prev_query) > 0.8:
                return True

        return False

    def record(self, query: str, error_signature: str = None):
        self.searched_queries.append(query)
        if error_signature:
            self.searched_errors.add(error_signature)
```

---

### Phase 5: Integration

#### 5.1 Enhanced Web Search Tool

Replace the current `web_search_tool` in `src/tools/web_search_tools.py`:

```python
@tool
def web_search_tool(query: str) -> str:
    """
    Intelligent web search for Java migration issues.

    This tool automatically:
    1. Analyzes and classifies your query
    2. Decomposes complex queries into focused sub-queries
    3. Executes parallel searches with deduplication
    4. Grades results and re-searches if quality is low
    5. Synthesizes actionable fix instructions

    Args:
        query: Your search query or error message

    Returns:
        Synthesized answer with root cause, solution, and sources
    """
    # Get project context from environment
    context = {
        "java_version": os.getenv("CURRENT_JAVA_VERSION", "unknown"),
        "spring_version": os.getenv("CURRENT_SPRING_VERSION", "unknown"),
        "project_path": os.getenv("MIGRATION_REPO_PATH", "")
    }

    # Initialize pipeline components
    analyzer = QueryAnalyzer()
    decomposer = QueryDecomposer()
    executor = MultiQueryExecutor()
    grader = ResultGrader()
    synthesizer = AnswerSynthesizer()

    # Check for duplicate search
    if deduplicator.is_duplicate(query):
        return "⚠️ Similar search already performed. Check previous results or try a different approach."

    # Step 1: Analyze query
    analysis = analyzer.analyze(query, context)

    # Step 2: Decompose into sub-queries
    sub_queries = decomposer.decompose(query, analysis)
    log_agent(f"[WEB_SEARCH] Decomposed into {len(sub_queries)} sub-queries")

    # Step 3: Execute searches (with caching)
    all_results = []
    for sq in sub_queries:
        cached = cache.get(sq)
        if cached:
            all_results.extend(cached)
        else:
            results = tavily_search_enhanced(sq, context)
            cache.set(sq, results)
            all_results.extend(results.get('results', []))

    # Step 4: Grade results
    grading = grader.grade(query, all_results, context)

    # Step 4b: Re-search if quality is low
    if grading['should_retry'] and grading.get('retry_suggestion'):
        log_agent(f"[WEB_SEARCH] Low quality results, retrying with: {grading['retry_suggestion']}")
        retry_results = tavily_search_enhanced(grading['retry_suggestion'], context)
        all_results.extend(retry_results.get('results', []))

    # Step 5: Synthesize answer
    synthesis = synthesizer.synthesize(query, all_results, context)

    # Record for deduplication
    deduplicator.record(query, analysis.get('error_signature'))

    return synthesis
```

---

## File Changes Required

| File | Changes |
|------|---------|
| `src/tools/web_search_tools.py` | Replace with enhanced implementation |
| `src/tools/__init__.py` | No changes needed |
| `src/utils/search_cache.py` | **NEW FILE** - Search caching |
| `src/utils/query_intelligence.py` | **NEW FILE** - QueryAnalyzer, QueryDecomposer, QueryRewriter |
| `src/utils/result_processing.py` | **NEW FILE** - ResultGrader, AnswerSynthesizer, ActionExtractor |
| `requirements.txt` | Add `asyncio` (if not present) |

---

## Expected Improvements

| Metric | Before | After |
|--------|--------|-------|
| Queries per error | 5-10 (repeated) | 2-3 (decomposed, unique) |
| Result relevance | ~30% actionable | ~80% actionable |
| Time to correct fix | Often never | Usually 1st or 2nd attempt |
| API costs | High (redundant calls) | Lower (caching + dedup) |

---

## Testing Plan

1. **Unit tests** for each component (QueryAnalyzer, Decomposer, etc.)
2. **Integration test** with known error patterns:
   - CGLIB + JDK 17 incompatibility
   - javax to jakarta migration
   - JUnit 4 to 5 runner issues
3. **Regression test** on previous failed migrations
4. **A/B comparison** - run same repo with old vs new search

---

## Implementation Priority

1. **P0 (Immediate)**: Query Decomposer + Rewriter - biggest impact
2. **P1 (High)**: Caching + Deduplication - reduce waste
3. **P2 (Medium)**: Result Grader + Re-search - improve quality
4. **P3 (Later)**: Action Extractor - enhance automation

---

## Sources

- [Azure AI Agentic Retrieval](https://learn.microsoft.com/en-us/azure/search/agentic-retrieval-overview)
- [Tavily Best Practices](https://docs.tavily.com/documentation/best-practices/best-practices-search)
- [LangChain Query Decomposition](https://python.langchain.com/v0.1/docs/use_cases/query_analysis/techniques/decomposition/)
- [Haystack Agentic RAG](https://sajalsharma.com/posts/comprehensive-agentic-rag/)
- [HuggingFace Agent RAG Cookbook](https://huggingface.co/learn/cookbook/en/agent_rag)
- [Tavily Search Depth Explanation](https://help.tavily.com/articles/6938147944-basic-vs-advanced-search-what-s-the-difference)

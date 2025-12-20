# BNY Web Search Wrapper - Step-by-Step Implementation Guide

## Current State

You have:
```
┌─────────────────────────────────────────────────┐
│           BNY Web Search Agent                  │
│           (BLACK BOX)                           │
│                                                 │
│  Input: query string                            │
│  Output: search results (unknown format)        │
│                                                 │
│  You don't control the scraping/ranking        │
└─────────────────────────────────────────────────┘
```

## Target Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     SMART WEB SEARCH WRAPPER                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    PRE-PROCESSING                                │ │
│  │                                                                  │ │
│  │  Agent Query ──▶ Deduplication ──▶ Query Optimizer ──▶ Queries  │ │
│  │                  (did we search                                  │ │
│  │                   this before?)                                  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              │                                        │
│                              ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    BNY AGENT CALL                               │ │
│  │                    (Black Box)                                   │ │
│  │                                                                  │ │
│  │           call_bny_web_search(optimized_query)                  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              │                                        │
│                              ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                   POST-PROCESSING                                │ │
│  │                                                                  │ │
│  │  Raw Results ──▶ Result Grader ──▶ Synthesizer ──▶ Actionable   │ │
│  │                  (is this useful?)   (extract      Output       │ │
│  │                                       fixes)                    │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Create the Search Processor Module

Create a new file: `src/utils/search_processor.py`

```python
"""
Search Pre/Post Processing for BNY Web Search Agent

This module wraps the BNY black-box web search with:
1. PRE: Query optimization, deduplication, decomposition
2. POST: Result grading, synthesis, action extraction
"""

import os
import re
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from src.utils.logging_config import log_agent, log_summary


@dataclass
class SearchContext:
    """Migration context for search optimization"""
    java_version_current: str = "8"
    java_version_target: str = "21"
    spring_boot_version: str = "unknown"
    error_type: str = "unknown"
    error_class: str = ""
    project_path: str = ""


@dataclass
class SearchResult:
    """Structured search result"""
    original_query: str
    optimized_queries: List[str]
    raw_response: str
    synthesized_answer: str
    confidence: float
    suggested_actions: List[str]
    was_cached: bool = False


class SearchProcessor:
    """
    Wraps BNY web search agent with intelligent pre/post processing.

    Usage:
        processor = SearchProcessor()
        result = processor.search(
            query="Spring Boot AbstractApplicationContext error",
            context=SearchContext(java_version_current="8", spring_boot_version="1.1.9"),
            bny_search_fn=call_bny_web_search  # Your BNY agent function
        )
    """

    def __init__(self, llm=None):
        """
        Args:
            llm: LLM for query optimization and synthesis (optional, uses rules if None)
        """
        self.llm = llm

        # Session-level caching
        self.query_cache: Dict[str, str] = {}  # normalized_query -> response
        self.searched_errors: set = set()  # error signatures already searched
        self.search_history: List[dict] = []  # all searches this session

        # Metrics
        self.cache_hits = 0
        self.total_searches = 0
        self.optimizations_applied = 0

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def search(
        self,
        query: str,
        context: SearchContext,
        bny_search_fn: callable
    ) -> SearchResult:
        """
        Main entry point - wraps BNY search with pre/post processing.

        Args:
            query: Raw query from agent
            context: Migration context (versions, error type)
            bny_search_fn: Function to call BNY web search agent

        Returns:
            SearchResult with synthesized, actionable response
        """
        self.total_searches += 1
        log_agent(f"[SEARCH_PROC] Processing query: {query[:80]}...")

        # ─────────────────────────────────────────────────────────────────────
        # PRE-PROCESSING
        # ─────────────────────────────────────────────────────────────────────

        # Step 1: Check deduplication
        error_sig = self._extract_error_signature(query)
        if self._is_duplicate_search(query, error_sig):
            log_agent(f"[SEARCH_PROC] Duplicate search detected, returning cached")
            return self._get_cached_result(query)

        # Step 2: Optimize query
        optimized_queries = self._optimize_query(query, context)
        log_agent(f"[SEARCH_PROC] Optimized into {len(optimized_queries)} queries")

        # ─────────────────────────────────────────────────────────────────────
        # CALL BNY AGENT (Black Box)
        # ─────────────────────────────────────────────────────────────────────

        all_results = []
        for opt_query in optimized_queries:
            # Check cache first
            cache_key = self._normalize_query(opt_query)
            if cache_key in self.query_cache:
                self.cache_hits += 1
                all_results.append(self.query_cache[cache_key])
                log_agent(f"[SEARCH_PROC] Cache hit for: {opt_query[:50]}...")
            else:
                # Call BNY agent
                try:
                    result = bny_search_fn(opt_query)
                    all_results.append(result)
                    self.query_cache[cache_key] = result
                    log_agent(f"[SEARCH_PROC] BNY search returned {len(str(result))} chars")
                except Exception as e:
                    log_agent(f"[SEARCH_PROC] BNY search failed: {e}")
                    all_results.append(f"Search failed: {e}")

        raw_response = "\n\n---\n\n".join(all_results)

        # ─────────────────────────────────────────────────────────────────────
        # POST-PROCESSING
        # ─────────────────────────────────────────────────────────────────────

        # Step 3: Grade results
        grade = self._grade_results(raw_response, query, context)
        log_agent(f"[SEARCH_PROC] Result grade: {grade['quality']} (score: {grade['score']:.2f})")

        # Step 4: If low quality, try alternative query
        if grade['quality'] == 'low' and grade.get('alternative_query'):
            log_agent(f"[SEARCH_PROC] Low quality, trying alternative: {grade['alternative_query'][:50]}...")
            alt_result = bny_search_fn(grade['alternative_query'])
            raw_response += f"\n\n---\n\n[ALTERNATIVE SEARCH]\n{alt_result}"

        # Step 5: Synthesize actionable answer
        synthesis = self._synthesize_answer(raw_response, query, context)

        # Step 6: Extract suggested actions
        actions = self._extract_actions(synthesis, context)

        # Record for deduplication
        self._record_search(query, error_sig, synthesis)

        return SearchResult(
            original_query=query,
            optimized_queries=optimized_queries,
            raw_response=raw_response,
            synthesized_answer=synthesis,
            confidence=grade['score'],
            suggested_actions=actions,
            was_cached=False
        )

    # =========================================================================
    # PRE-PROCESSING: Query Optimization
    # =========================================================================

    def _optimize_query(self, query: str, context: SearchContext) -> List[str]:
        """
        Transform raw query into 1-3 optimized search queries.

        Strategies:
        1. Extract error class + add version context
        2. Create root cause query (compatibility check)
        3. Create solution-focused query
        """
        optimized = []

        # Strategy 1: Clean and enhance the original query
        primary = self._clean_query(query)
        primary = self._add_version_context(primary, context)
        optimized.append(primary)

        # Strategy 2: If error detected, create compatibility query
        error_class = self._extract_error_class(query)
        if error_class:
            compat_query = self._create_compatibility_query(error_class, context)
            if compat_query and compat_query != primary:
                optimized.append(compat_query)

        # Strategy 3: Create solution-focused query
        if "error" in query.lower() or "exception" in query.lower():
            solution_query = self._create_solution_query(query, context)
            if solution_query and solution_query not in optimized:
                optimized.append(solution_query)

        self.optimizations_applied += len(optimized) - 1
        return optimized[:3]  # Max 3 queries

    def _clean_query(self, query: str) -> str:
        """Remove noise from query (stack traces, line numbers, etc.)"""
        # Remove line numbers
        query = re.sub(r'\([\w]+\.java:\d+\)', '', query)

        # Remove "at org.xxx.yyy" stack trace lines
        query = re.sub(r'\bat\s+[\w\.]+\([\w\.]+:\d+\)', '', query)

        # Remove excessive whitespace
        query = re.sub(r'\s+', ' ', query).strip()

        # Truncate if too long (Tavily recommends < 400 chars)
        if len(query) > 300:
            query = query[:300]

        return query

    def _add_version_context(self, query: str, context: SearchContext) -> str:
        """Add version information to query for better results"""
        additions = []

        # Add Java version if migrating
        if context.java_version_target and context.java_version_target != "unknown":
            if "java" not in query.lower():
                additions.append(f"Java {context.java_version_target}")

        # Add Spring Boot version if known
        if context.spring_boot_version and context.spring_boot_version != "unknown":
            if "spring" not in query.lower():
                additions.append(f"Spring Boot {context.spring_boot_version}")

        # Add "fix" or "solution" if not present
        if not any(word in query.lower() for word in ["fix", "solution", "resolve", "how to"]):
            additions.append("fix")

        if additions:
            return f"{query} {' '.join(additions)}"
        return query

    def _extract_error_class(self, query: str) -> Optional[str]:
        """Extract the primary error class from query"""
        patterns = [
            r'(NoClassDefFoundError)',
            r'(ClassNotFoundException)',
            r'(NoSuchMethodError)',
            r'(NoSuchFieldError)',
            r'(IllegalAccessError)',
            r'(IncompatibleClassChangeError)',
            r'(UnsupportedClassVersionError)',
            r'(ExceptionInInitializerError)',
            r'(InaccessibleObjectException)',
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _create_compatibility_query(self, error_class: str, context: SearchContext) -> str:
        """Create a query focused on version compatibility"""
        # Map common errors to compatibility queries
        if "NoClassDefFound" in error_class or "ClassNotFound" in error_class:
            return f"{error_class} Java {context.java_version_target} Spring Boot compatibility"

        if "UnsupportedClassVersion" in error_class:
            return f"Spring Boot minimum Java version requirements {context.java_version_target}"

        if "InaccessibleObject" in error_class:
            return f"Java {context.java_version_target} module system --add-opens Spring fix"

        return f"{error_class} Java {context.java_version_target} compatibility fix"

    def _create_solution_query(self, query: str, context: SearchContext) -> str:
        """Create a solution-focused query"""
        # Extract the component causing issues
        components = []

        if "cglib" in query.lower():
            components.append("CGLIB proxy")
        if "hibernate" in query.lower():
            components.append("Hibernate")
        if "javax" in query.lower():
            components.append("javax to jakarta")
        if "junit" in query.lower():
            components.append("JUnit")

        if components:
            return f"{' '.join(components)} Java {context.java_version_target} migration upgrade solution"

        return f"Java migration error Java {context.java_version_target} solution"

    # =========================================================================
    # POST-PROCESSING: Result Grading
    # =========================================================================

    def _grade_results(self, results: str, query: str, context: SearchContext) -> dict:
        """
        Grade search results for relevance and actionability.

        Returns:
            {
                'quality': 'high' | 'medium' | 'low',
                'score': 0.0-1.0,
                'reasons': [...],
                'alternative_query': str or None
            }
        """
        score = 0.0
        reasons = []

        results_lower = results.lower()

        # Check for version-specific content
        if context.java_version_target in results:
            score += 0.2
            reasons.append(f"Contains Java {context.java_version_target} reference")

        if context.spring_boot_version in results:
            score += 0.1
            reasons.append(f"Contains Spring Boot {context.spring_boot_version} reference")

        # Check for actionable content
        action_indicators = [
            ("upgrade", 0.15, "Mentions upgrade"),
            ("pom.xml", 0.1, "Contains POM reference"),
            ("<dependency>", 0.1, "Contains dependency XML"),
            ("version", 0.1, "Mentions versions"),
            ("fix", 0.1, "Contains fix"),
            ("solution", 0.1, "Contains solution"),
            ("add-opens", 0.15, "Contains module fix"),
            ("migrate", 0.1, "Contains migration guidance"),
        ]

        for indicator, points, reason in action_indicators:
            if indicator in results_lower:
                score += points
                reasons.append(reason)

        # Check for code examples
        if "```" in results or "<dependency>" in results:
            score += 0.15
            reasons.append("Contains code examples")

        # Penalize if results seem generic
        generic_indicators = [
            "no results found",
            "search failed",
            "could not find",
        ]
        for indicator in generic_indicators:
            if indicator in results_lower:
                score -= 0.3
                reasons.append(f"Generic response: {indicator}")

        # Determine quality level
        if score >= 0.6:
            quality = 'high'
        elif score >= 0.3:
            quality = 'medium'
        else:
            quality = 'low'

        # Suggest alternative query if low quality
        alternative = None
        if quality == 'low':
            alternative = self._suggest_alternative_query(query, context)

        return {
            'quality': quality,
            'score': min(1.0, max(0.0, score)),
            'reasons': reasons,
            'alternative_query': alternative
        }

    def _suggest_alternative_query(self, query: str, context: SearchContext) -> str:
        """Suggest an alternative query when results are poor"""
        # Try a more specific compatibility query
        return f"Spring Boot {context.spring_boot_version} Java {context.java_version_target} upgrade migration guide"

    # =========================================================================
    # POST-PROCESSING: Answer Synthesis
    # =========================================================================

    def _synthesize_answer(self, results: str, query: str, context: SearchContext) -> str:
        """
        Synthesize search results into actionable guidance.

        Uses LLM if available, otherwise rule-based extraction.
        """
        if self.llm:
            return self._synthesize_with_llm(results, query, context)
        else:
            return self._synthesize_rule_based(results, query, context)

    def _synthesize_with_llm(self, results: str, query: str, context: SearchContext) -> str:
        """Use LLM to synthesize results"""
        prompt = f"""You are a Java migration expert. Synthesize these search results into actionable guidance.

QUERY: {query}

PROJECT CONTEXT:
- Current Java: {context.java_version_current}
- Target Java: {context.java_version_target}
- Spring Boot: {context.spring_boot_version}

SEARCH RESULTS:
{results[:8000]}

Provide a CONCISE answer (max 500 words) with:

## Root Cause
[1-2 sentences - what's actually wrong]

## Solution
[Numbered steps to fix]

## Required Changes
```xml or java
[Specific code/config if applicable]
```

Be specific with version numbers. No fluff."""

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            return content[:3000]  # Cap at 3000 chars
        except Exception as e:
            log_agent(f"[SEARCH_PROC] LLM synthesis failed: {e}, using rule-based")
            return self._synthesize_rule_based(results, query, context)

    def _synthesize_rule_based(self, results: str, query: str, context: SearchContext) -> str:
        """Rule-based synthesis when LLM unavailable"""
        lines = results.split('\n')

        # Extract key information
        versions_mentioned = set()
        code_snippets = []
        key_facts = []

        in_code = False
        code_buffer = []

        for line in lines:
            # Track code blocks
            if '```' in line or '<dependency>' in line:
                in_code = not in_code
                if not in_code and code_buffer:
                    code_snippets.append('\n'.join(code_buffer[:15]))
                    code_buffer = []
                continue

            if in_code:
                code_buffer.append(line)
                continue

            # Extract versions
            version_matches = re.findall(r'(\d+\.\d+(?:\.\d+)?)', line)
            versions_mentioned.update(version_matches)

            # Extract key facts
            lower = line.lower()
            if any(kw in lower for kw in ['upgrade', 'migrate', 'requires', 'minimum', 'fix', 'solution']):
                if len(line.strip()) > 20 and len(line.strip()) < 200:
                    key_facts.append(line.strip())

        # Build synthesis
        synthesis_parts = []

        synthesis_parts.append("## Search Results Summary\n")

        if key_facts:
            synthesis_parts.append("**Key Findings:**")
            for fact in key_facts[:10]:  # Top 10 facts
                synthesis_parts.append(f"- {fact}")

        if versions_mentioned:
            synthesis_parts.append(f"\n**Versions Mentioned:** {', '.join(sorted(versions_mentioned)[:10])}")

        if code_snippets:
            synthesis_parts.append("\n**Code Examples:**")
            for i, snippet in enumerate(code_snippets[:3], 1):
                synthesis_parts.append(f"\nExample {i}:\n```\n{snippet}\n```")

        if not key_facts and not code_snippets:
            synthesis_parts.append("\n*No specific actionable information found. Consider searching with different terms.*")

        return '\n'.join(synthesis_parts)

    # =========================================================================
    # POST-PROCESSING: Action Extraction
    # =========================================================================

    def _extract_actions(self, synthesis: str, context: SearchContext) -> List[str]:
        """Extract suggested actions from synthesized answer"""
        actions = []

        lower = synthesis.lower()

        # Detect upgrade recommendations
        if "upgrade spring boot" in lower or "update spring boot" in lower:
            version_match = re.search(r'spring boot (\d+\.\d+(?:\.\d+)?)', lower)
            if version_match:
                actions.append(f"Upgrade Spring Boot to {version_match.group(1)}")
            else:
                actions.append("Upgrade Spring Boot to compatible version")

        if "upgrade java" in lower or "java 17" in lower or "java 21" in lower:
            actions.append(f"Ensure Java {context.java_version_target} compatibility")

        if "--add-opens" in synthesis:
            actions.append("Add JVM --add-opens flags for module access")

        if "javax" in lower and "jakarta" in lower:
            actions.append("Migrate javax.* to jakarta.* namespace")

        if "pom.xml" in lower:
            actions.append("Update pom.xml dependencies")

        if not actions:
            actions.append("Review search results for specific guidance")

        return actions

    # =========================================================================
    # DEDUPLICATION
    # =========================================================================

    def _extract_error_signature(self, query: str) -> str:
        """Extract a signature for error deduplication"""
        # Extract error class + component
        error_class = self._extract_error_class(query) or "unknown"

        # Extract component (cglib, hibernate, etc.)
        component = "general"
        components = ["cglib", "hibernate", "junit", "spring", "jackson", "jakarta"]
        for comp in components:
            if comp in query.lower():
                component = comp
                break

        return f"{error_class}:{component}"

    def _normalize_query(self, query: str) -> str:
        """Normalize query for cache key"""
        # Lowercase, remove extra spaces, sort words
        words = sorted(query.lower().split())
        return hashlib.md5(' '.join(words).encode()).hexdigest()

    def _is_duplicate_search(self, query: str, error_sig: str) -> bool:
        """Check if this is a duplicate search"""
        # Check error signature
        if error_sig in self.searched_errors:
            return True

        # Check query cache
        cache_key = self._normalize_query(query)
        if cache_key in self.query_cache:
            return True

        return False

    def _get_cached_result(self, query: str) -> SearchResult:
        """Return cached result for duplicate query"""
        cache_key = self._normalize_query(query)
        cached = self.query_cache.get(cache_key, "No cached result found")

        return SearchResult(
            original_query=query,
            optimized_queries=[],
            raw_response=cached,
            synthesized_answer=f"[CACHED RESULT]\n\n{cached[:2000]}",
            confidence=0.5,
            suggested_actions=["Review previous search results"],
            was_cached=True
        )

    def _record_search(self, query: str, error_sig: str, synthesis: str):
        """Record search for deduplication"""
        self.searched_errors.add(error_sig)

        self.search_history.append({
            'timestamp': datetime.now().isoformat(),
            'query': query[:200],
            'error_sig': error_sig,
            'synthesis_len': len(synthesis)
        })

    # =========================================================================
    # METRICS
    # =========================================================================

    def get_stats(self) -> dict:
        """Get search processor statistics"""
        return {
            'total_searches': self.total_searches,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / max(1, self.total_searches),
            'optimizations_applied': self.optimizations_applied,
            'unique_errors_searched': len(self.searched_errors),
            'cached_queries': len(self.query_cache),
        }
```

---

## Step 2: Integrate with web_search_tools.py

Modify `src/tools/web_search_tools.py`:

```python
"""
Web Search and OpenRewrite Agent Tools

Wrapped with SearchProcessor for intelligent pre/post processing.
"""

import os
from langchain_core.tools import tool
from src.utils.logging_config import log_agent, log_summary
from src.utils.search_processor import SearchProcessor, SearchContext

# Initialize processor (singleton for session)
_search_processor = None

def get_search_processor():
    """Get or create search processor singleton"""
    global _search_processor
    if _search_processor is None:
        # Optionally pass LLM for synthesis
        try:
            from langchain_aws import ChatBedrock
            llm = ChatBedrock(
                model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",  # Fast model for processing
                region_name=os.environ.get("AWS_REGION", "us-east-1"),
                model_kwargs={"max_tokens": 1000, "temperature": 0.0},
            )
            _search_processor = SearchProcessor(llm=llm)
        except Exception as e:
            log_agent(f"[SEARCH] LLM init failed, using rule-based: {e}")
            _search_processor = SearchProcessor(llm=None)
    return _search_processor


def _call_bny_web_search(query: str) -> str:
    """
    Call BNY Web Search Agent (BLACK BOX).

    TODO: Replace this with actual BNY agent call:
        from bny_framework import WebSearchAgent
        agent = WebSearchAgent(agent_id="your-agent-id")
        return agent.search(query)
    """
    # Current implementation uses Tavily as placeholder
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    if not TAVILY_API_KEY:
        return "Web search not configured"

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)

        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True
        )

        # Format response
        answer = response.get("answer", "")
        results = []
        for r in response.get("results", []):
            results.append(f"**{r.get('title', 'Untitled')}**\n{r.get('content', '')[:500]}\nSource: {r.get('url', '')}")

        output = ""
        if answer:
            output += f"Summary: {answer}\n\n"
        output += "\n---\n".join(results)

        return output

    except Exception as e:
        return f"Search failed: {e}"


@tool
def web_search_tool(query: str) -> str:
    """
    Search the internet for Java migration solutions and error fixes.

    This tool automatically:
    - Optimizes your query for better results
    - Deduplicates repeated searches
    - Grades result quality
    - Synthesizes actionable guidance

    Args:
        query: Your search query or error message

    Returns:
        Synthesized answer with root cause, solution, and actions
    """
    log_agent(f"[WEB_SEARCH] Incoming query: {query[:100]}...")

    # Build context from environment
    context = SearchContext(
        java_version_current=os.getenv("CURRENT_JAVA_VERSION", "8"),
        java_version_target=os.getenv("TARGET_JAVA_VERSION", "21"),
        spring_boot_version=os.getenv("CURRENT_SPRING_VERSION", "unknown"),
        project_path=os.getenv("MIGRATION_REPO_PATH", ""),
    )

    # Get processor and search
    processor = get_search_processor()
    result = processor.search(
        query=query,
        context=context,
        bny_search_fn=_call_bny_web_search  # <-- YOUR BNY AGENT GOES HERE
    )

    # Log stats
    stats = processor.get_stats()
    log_summary(f"[WEB_SEARCH] Query processed | Cache hits: {stats['cache_hits']} | Optimizations: {stats['optimizations_applied']}")

    # Build response
    response = f"""## Search Results

**Original Query:** {result.original_query[:100]}...
**Optimized Queries:** {len(result.optimized_queries)}
**Confidence:** {result.confidence:.0%}
**Cached:** {'Yes' if result.was_cached else 'No'}

---

{result.synthesized_answer}

---

## Suggested Actions
"""
    for i, action in enumerate(result.suggested_actions, 1):
        response += f"{i}. {action}\n"

    return response
```

---

## Step 3: Set Environment Variables

The processor uses environment variables for context. Set these in your orchestrator:

```python
# In supervisor_orchestrator.py, during initialization:

def _set_migration_context(self, project_path: str):
    """Set environment variables for search context"""
    import os

    # These get picked up by SearchProcessor
    os.environ["MIGRATION_REPO_PATH"] = project_path
    os.environ["CURRENT_JAVA_VERSION"] = self._detect_java_version(project_path)
    os.environ["TARGET_JAVA_VERSION"] = "21"
    os.environ["CURRENT_SPRING_VERSION"] = self._detect_spring_version(project_path)
```

---

## Step 4: Integrate with Context Manager

The context manager already compresses web search results. Update it to recognize the new format:

In `src/utils/context_manager.py`, update `_compress_web_search_output`:

```python
def _compress_web_search_output(self, content: str, tool_name: str) -> str:
    """Compress web search results - NEW processed format is already concise"""

    # Check if this is already processed by SearchProcessor
    if "## Search Results" in content and "## Suggested Actions" in content:
        # Already optimized by SearchProcessor, just ensure reasonable length
        if len(content) > 3000:
            # Keep summary and actions, trim middle
            lines = content.split('\n')

            # Find key sections
            summary_end = 0
            actions_start = len(lines)

            for i, line in enumerate(lines):
                if line.startswith('---') and summary_end == 0:
                    summary_end = i
                if '## Suggested Actions' in line:
                    actions_start = i

            # Keep first section + actions
            kept = lines[:summary_end+5] + ['...'] + lines[actions_start:]
            return '\n'.join(kept)
        return content

    # Fall back to existing compression for raw results
    return self._existing_compression_logic(content)
```

---

## Step 5: How It All Flows

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ERROR EXPERT CALLS: web_search_tool("Spring Boot cglib error...")          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: web_search_tool() in web_search_tools.py                           │
│                                                                             │
│   - Gets SearchProcessor singleton                                          │
│   - Builds SearchContext from environment                                   │
│   - Calls processor.search()                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: SearchProcessor.search() - PRE-PROCESSING                          │
│                                                                             │
│   2a. Check deduplication (have we searched this error before?)            │
│       → If yes, return cached result immediately                           │
│                                                                             │
│   2b. Optimize query:                                                       │
│       - Clean: Remove stack trace noise                                    │
│       - Enhance: Add "Java 21" + "fix"                                     │
│       - Decompose: Create 2-3 focused sub-queries                          │
│                                                                             │
│   INPUT:  "Spring Boot AbstractApplicationContext cglib error test"        │
│   OUTPUT: ["CGLIB NoClassDefFoundError Java 21 Spring Boot fix",           │
│            "Spring Boot 1.x Java 21 compatibility minimum version",        │
│            "Spring CGLIB proxy upgrade migration"]                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Call BNY Agent (BLACK BOX)                                         │
│                                                                             │
│   For each optimized query:                                                 │
│       - Check cache first                                                   │
│       - If not cached: call _call_bny_web_search(query)                    │
│       - Store in cache                                                      │
│                                                                             │
│   You replace _call_bny_web_search with your BNY agent call               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: SearchProcessor.search() - POST-PROCESSING                         │
│                                                                             │
│   4a. Grade results:                                                        │
│       - Check for version references                                        │
│       - Check for actionable content (code, deps, steps)                   │
│       - Score 0.0-1.0                                                       │
│       - If score < 0.3: suggest alternative query and re-search            │
│                                                                             │
│   4b. Synthesize answer:                                                    │
│       - Use LLM (if available) or rules                                    │
│       - Extract: Root Cause, Solution, Code Changes                        │
│       - Cap at 3000 chars                                                   │
│                                                                             │
│   4c. Extract actions:                                                      │
│       - "Upgrade Spring Boot to 2.7.x"                                     │
│       - "Add --add-opens JVM flags"                                        │
│       - etc.                                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: Return to Agent                                                     │
│                                                                             │
│   ## Search Results                                                         │
│   **Confidence:** 75%                                                       │
│                                                                             │
│   ## Root Cause                                                             │
│   Spring Boot 1.x uses an old CGLIB version incompatible with Java 21...   │
│                                                                             │
│   ## Solution                                                               │
│   1. Upgrade Spring Boot to 2.7.x or 3.x                                   │
│   2. Update pom.xml parent version                                         │
│                                                                             │
│   ## Suggested Actions                                                      │
│   1. Upgrade Spring Boot to 2.7.18                                         │
│   2. Update pom.xml dependencies                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: Context Manager Compression                                        │
│                                                                             │
│   When context is compressed for next agent call:                          │
│   - Recognizes "## Search Results" format                                  │
│   - Keeps summary + actions (already concise)                              │
│   - Truncates middle if > 3000 chars                                       │
│   - Stores full result in external storage for retrieval                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 6: Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/utils/search_processor.py` | **CREATE** | New file with SearchProcessor class |
| `src/tools/web_search_tools.py` | **MODIFY** | Replace with wrapped version |
| `src/utils/context_manager.py` | **MODIFY** | Add recognition of processed format |
| `supervisor_orchestrator.py` | **MODIFY** | Set environment variables for context |

---

## Step 7: Replace with BNY Agent

When ready to use actual BNY agent, update `_call_bny_web_search`:

```python
def _call_bny_web_search(query: str) -> str:
    """
    Call BNY Web Search Agent.
    """
    # TODO: Get your actual BNY agent import and ID
    from bny_eliza_framework import AgentClient

    client = AgentClient()
    response = client.call_agent(
        agent_id="your-web-search-agent-id",
        input=query
    )

    return response.output  # or however BNY returns results
```

The wrapper handles everything else - the BNY agent is truly a black box.

---

## Expected Improvements

| Before | After |
|--------|-------|
| Raw error text sent as query | Cleaned, enhanced, decomposed queries |
| 24 duplicate searches | ~5 unique searches (deduplication) |
| Generic results | Version-specific, actionable results |
| Long context bloat | Concise synthesized answers |
| No quality check | Auto-retry on low-quality results |

---

## Testing

```python
# Test the processor directly
from src.utils.search_processor import SearchProcessor, SearchContext

processor = SearchProcessor()
context = SearchContext(
    java_version_current="8",
    java_version_target="21",
    spring_boot_version="1.1.9"
)

# Mock BNY search
def mock_bny_search(query):
    return f"Mock results for: {query}"

result = processor.search(
    query="NoClassDefFoundError: org.springframework.cglib.proxy.Enhancer",
    context=context,
    bny_search_fn=mock_bny_search
)

print(f"Optimized queries: {result.optimized_queries}")
print(f"Confidence: {result.confidence}")
print(f"Actions: {result.suggested_actions}")
```

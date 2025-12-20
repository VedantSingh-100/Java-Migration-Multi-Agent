"""
Search Pre-Processing for Web Search Agent (Simplified)

This module wraps the web search agent with MINIMAL processing:
1. Query optimization (add version context)
2. Query deduplication (avoid duplicate API calls)
3. Length truncation (stay within context limits)

Philosophy: Let the LLM handle noise filtering - it's better at it than regex.

Usage:
    processor = get_search_processor()
    result = processor.search(query, bny_search_fn=my_search_function)
"""

import os
import re
import hashlib
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime

from src.utils.logging_config import log_agent, log_summary


# =============================================================================
# CONSTANTS
# =============================================================================

MAX_QUERY_LENGTH = 300  # Tavily recommends < 400 chars
MAX_RESULT_LENGTH = 10000  # Max chars to return to LLM
MIN_QUERY_LENGTH = 10

# Patterns for error class extraction (useful for query optimization)
ERROR_CLASS_PATTERNS = [
    r'(NoClassDefFoundError)',
    r'(ClassNotFoundException)',
    r'(NoSuchMethodError)',
    r'(NoSuchFieldError)',
    r'(IllegalAccessError)',
    r'(IncompatibleClassChangeError)',
    r'(UnsupportedClassVersionError)',
    r'(ExceptionInInitializerError)',
    r'(InaccessibleObjectException)',
    r'(AbstractMethodError)',
]

# Components for query enhancement
COMPONENT_KEYWORDS = ['cglib', 'hibernate', 'junit', 'spring', 'jackson', 'jakarta', 'javax', 'mockito']


# =============================================================================
# DATA CLASS
# =============================================================================

@dataclass
class SearchContext:
    """Migration context for query optimization."""
    java_version_current: str = "8"
    java_version_target: str = "21"
    spring_boot_version: str = "unknown"
    project_path: str = ""

    @classmethod
    def from_environment(cls) -> 'SearchContext':
        """Create context from environment variables."""
        return cls(
            java_version_current=os.getenv("CURRENT_JAVA_VERSION", "8"),
            java_version_target=os.getenv("TARGET_JAVA_VERSION", "21"),
            spring_boot_version=os.getenv("CURRENT_SPRING_VERSION", "unknown"),
            project_path=os.getenv("MIGRATION_REPO_PATH", ""),
        )


# =============================================================================
# SEARCH PROCESSOR (Simplified)
# =============================================================================

class SearchProcessor:
    """
    Minimal search processor - query optimization and deduplication only.

    Removed:
    - Noise filtering (LLM handles this better)
    - Result grading (unnecessary complexity)
    - Key findings extraction (LLM does this naturally)
    - Rule-based synthesis (just return raw results)

    Kept:
    - Query optimization (add version context)
    - Exact query deduplication (save API calls)
    - Length truncation (stay in context)
    """

    def __init__(self):
        # Query cache: normalized_query_hash -> raw_response
        self._query_cache: Dict[str, str] = {}

        # Metrics
        self._cache_hits = 0
        self._total_searches = 0

        log_agent("[SEARCH_PROC] SearchProcessor initialized (simplified)")

    def search(
        self,
        query: str,
        bny_search_fn: Callable[[str], str],
        context: Optional[SearchContext] = None,
    ) -> str:
        """
        Execute search with query optimization and deduplication.

        Args:
            query: Raw query from agent
            bny_search_fn: Function to call the actual search
            context: Migration context (reads from env if None)

        Returns:
            Raw search results (let LLM handle synthesis)
        """
        self._total_searches += 1

        if context is None:
            context = SearchContext.from_environment()

        log_agent(f"[SEARCH_PROC] Query: {query[:80]}...")
        log_agent(f"[SEARCH_PROC] Context: Java {context.java_version_current}→{context.java_version_target}")

        # Step 1: Optimize query
        optimized_query = self._optimize_query(query, context)
        log_agent(f"[SEARCH_PROC] Optimized: {optimized_query[:80]}...")

        # Step 2: Check cache (exact query deduplication)
        cache_key = self._get_cache_key(optimized_query)
        if cache_key in self._query_cache:
            self._cache_hits += 1
            log_agent("[SEARCH_PROC] Cache hit - returning cached result")
            log_summary(f"[SEARCH_PROC] Cache hit | Total: {self._total_searches} | Hits: {self._cache_hits}")
            cached = self._query_cache[cache_key]
            return f"[Cached Result - identical search performed earlier]\n\n{cached}"

        # Step 3: Execute search
        try:
            raw_result = bny_search_fn(optimized_query)
            log_agent(f"[SEARCH_PROC] Got {len(raw_result)} chars from search")
        except Exception as e:
            log_agent(f"[SEARCH_PROC] Search failed: {e}")
            return f"Search failed: {e}"

        # Step 4: Truncate if too long (preserve structure)
        result = self._truncate_result(raw_result)

        # Step 5: Cache and return
        self._query_cache[cache_key] = result

        log_summary(f"[SEARCH_PROC] Search complete | {len(result)} chars | Cache: {self._cache_hits}/{self._total_searches}")

        return result

    def _optimize_query(self, query: str, context: SearchContext) -> str:
        """
        Optimize query for better search results.

        - Clean up stack traces and noise
        - Add version context
        - Add "fix" or "solution" for error queries
        """
        # Clean: remove stack trace noise from query itself
        cleaned = self._clean_query(query)

        # Enhance: add version context if not present
        enhanced = self._add_context(cleaned, context)

        # Truncate to max length
        if len(enhanced) > MAX_QUERY_LENGTH:
            enhanced = enhanced[:MAX_QUERY_LENGTH]

        return enhanced

    def _clean_query(self, query: str) -> str:
        """Remove stack trace lines and noise from the query."""
        # Remove line numbers like (ClassName.java:123)
        query = re.sub(r'\([\w]+\.java:\d+\)', '', query)

        # Remove "at org.xxx.yyy" stack trace patterns
        query = re.sub(r'\bat\s+[\w.]+\([\w.]+:\d+\)', '', query)

        # Remove standalone "at " prefixes
        query = re.sub(r'^\s*at\s+', '', query, flags=re.MULTILINE)

        # Remove [INFO], [ERROR] etc maven output markers
        query = re.sub(r'\[\s*(INFO|ERROR|WARNING|DEBUG)\s*\]', '', query)

        # Collapse whitespace
        query = re.sub(r'\s+', ' ', query).strip()

        return query

    def _add_context(self, query: str, context: SearchContext) -> str:
        """Add version context to query for better results."""
        additions = []
        query_lower = query.lower()

        # Add Java version if not mentioned
        if context.java_version_target and context.java_version_target != "unknown":
            if context.java_version_target not in query:
                additions.append(f"Java {context.java_version_target}")

        # Add Spring Boot if relevant and not mentioned
        if context.spring_boot_version != "unknown":
            if "spring" in query_lower and "spring boot" not in query_lower:
                additions.append("Spring Boot")

        # Add solution-oriented keyword if this looks like an error query
        error_indicators = ['error', 'exception', 'failed', 'cannot', 'unable', 'noclassdef']
        if any(ind in query_lower for ind in error_indicators):
            if not any(sol in query_lower for sol in ['fix', 'solution', 'resolve', 'how to']):
                additions.append("fix solution")

        if additions:
            return f"{query} {' '.join(additions)}"
        return query

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key from query."""
        # Normalize: lowercase, sort words, hash
        words = sorted(query.lower().split())
        normalized = ' '.join(words)
        return hashlib.md5(normalized.encode()).hexdigest()

    def _truncate_result(self, result: str) -> str:
        """Truncate result while preserving useful structure."""
        if len(result) <= MAX_RESULT_LENGTH:
            return result

        # Try to truncate at a natural break point
        truncated = result[:MAX_RESULT_LENGTH]

        # Find last complete section (---) or paragraph
        last_break = truncated.rfind('\n---')
        if last_break > MAX_RESULT_LENGTH * 0.7:  # At least 70% of content
            truncated = truncated[:last_break]
        else:
            # Fall back to last newline
            last_newline = truncated.rfind('\n')
            if last_newline > MAX_RESULT_LENGTH * 0.9:
                truncated = truncated[:last_newline]

        return truncated + "\n\n[Results truncated...]"

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            'total_searches': self._total_searches,
            'cache_hits': self._cache_hits,
            'cache_hit_rate': self._cache_hits / max(1, self._total_searches),
            'cached_queries': len(self._query_cache),
        }

    def reset(self):
        """Reset processor state for new session."""
        self._query_cache.clear()
        self._cache_hits = 0
        self._total_searches = 0
        log_agent("[SEARCH_PROC] Processor state reset")


# =============================================================================
# SINGLETON
# =============================================================================

_search_processor: Optional[SearchProcessor] = None


def get_search_processor() -> SearchProcessor:
    """Get or create the search processor singleton."""
    global _search_processor
    if _search_processor is None:
        _search_processor = SearchProcessor()
    return _search_processor


def reset_search_processor():
    """Reset the search processor singleton."""
    global _search_processor
    if _search_processor is not None:
        _search_processor.reset()
    _search_processor = None


# =============================================================================
# ENVIRONMENT SETUP HELPER
# =============================================================================

def setup_search_context_from_pom(project_path: str) -> Dict[str, str]:
    """
    Detect Java/Spring versions from pom.xml and set environment variables.

    Call this from the orchestrator's _set_project_path() method.
    """
    import xml.etree.ElementTree as ET
    from pathlib import Path

    versions = {
        'CURRENT_JAVA_VERSION': '8',
        'TARGET_JAVA_VERSION': os.environ.get('TARGET_JAVA_VERSION', '21'),
        'CURRENT_SPRING_VERSION': 'unknown',
        'MIGRATION_REPO_PATH': project_path,
    }

    pom_path = Path(project_path) / "pom.xml"

    if not pom_path.exists():
        log_agent(f"[SEARCH_CONTEXT] No pom.xml at {project_path}, using defaults")
        _set_env_vars(versions)
        return versions

    try:
        tree = ET.parse(pom_path)
        root = tree.getroot()

        # Handle Maven namespace
        namespace = ""
        if root.tag.startswith("{"):
            namespace = root.tag.split("}")[0] + "}"

        # Detect Java version
        props = root.find(f"{namespace}properties")
        if props is not None:
            for tag in ['java.version', 'maven.compiler.source', 'maven.compiler.target']:
                elem = props.find(f"{namespace}{tag}")
                if elem is not None and elem.text:
                    java_version = elem.text
                    if java_version == "1.8":
                        java_version = "8"
                    versions['CURRENT_JAVA_VERSION'] = java_version
                    break

        # Detect Spring Boot version from parent
        parent = root.find(f"{namespace}parent")
        if parent is not None:
            artifact_id = parent.find(f"{namespace}artifactId")
            version = parent.find(f"{namespace}version")
            if artifact_id is not None and artifact_id.text:
                if "spring-boot" in artifact_id.text.lower():
                    if version is not None and version.text:
                        versions['CURRENT_SPRING_VERSION'] = version.text

        log_agent(f"[SEARCH_CONTEXT] Detected: Java={versions['CURRENT_JAVA_VERSION']}, Spring={versions['CURRENT_SPRING_VERSION']}")

    except Exception as e:
        log_agent(f"[SEARCH_CONTEXT] Error parsing pom.xml: {e}")

    _set_env_vars(versions)
    return versions


def _set_env_vars(versions: Dict[str, str]):
    """Set environment variables for search context."""
    for key, value in versions.items():
        os.environ[key] = value

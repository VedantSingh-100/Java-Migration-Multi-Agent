"""
Web Search and OpenRewrite Agent Tools

These tools provide web search and RAG capabilities for the migration agents.
- web_search_tool: Search the internet for Java migration solutions
- call_openrewrite_agent: Query OpenRewrite documentation

Integration:
- Uses simplified SearchProcessor for query optimization and deduplication
- Raw results passed to LLM - it handles noise filtering better than regex

🔔 REMINDER: BNY Internal Integration
TODO: Replace _raw_web_search with BNY internal framework
"""

import os
from langchain_core.tools import tool
from src.utils.logging_config import log_agent, log_summary
from src.utils.search_processor import get_search_processor

# Tavily API configuration
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# =============================================================================
# RAW SEARCH FUNCTION (Black Box - Replace with BNY Agent)
# =============================================================================

def _raw_web_search(query: str) -> str:
    """
    Execute raw web search using Tavily (or BNY agent).

    This is the BLACK BOX function that the SearchProcessor wraps.
    Replace this with BNY agent call when available.

    Args:
        query: Optimized search query from SearchProcessor

    Returns:
        Raw search results as string
    """
    if not TAVILY_API_KEY:
        log_agent("[WEB_SEARCH_RAW] TAVILY_API_KEY not set")
        return "Web search not configured (TAVILY_API_KEY not set)"

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)

        log_agent(f"[WEB_SEARCH_RAW] Executing: {query[:80]}...")

        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True
        )

        # Extract the synthesized answer if available
        answer = response.get("answer", "")

        # Format individual results
        results = []
        for r in response.get("results", []):
            title = r.get("title", "Untitled")
            content = r.get("content", "")[:500]
            url = r.get("url", "")
            results.append(f"**{title}**\n{content}...\nSource: {url}")

        formatted_results = "\n\n---\n\n".join(results) if results else "No results found"

        # Combine answer with detailed results
        output = ""
        if answer:
            output += f"Summary: {answer}\n\n---\n\n"
        output += f"Results:\n\n{formatted_results}"

        log_agent(f"[WEB_SEARCH_RAW] Found {len(results)} results")
        return output

    except ImportError:
        log_agent("[WEB_SEARCH_RAW] tavily-python not installed")
        return (
            "Web search library not installed.\n"
            "General migration tips:\n"
            "- Spring Boot 1.5.x only works with JDK 8\n"
            "- Spring Boot 2.x works with JDK 8-17\n"
            "- Spring Boot 3.x requires JDK 17+\n"
        )
    except Exception as e:
        log_agent(f"[WEB_SEARCH_RAW] Error: {str(e)}")
        return f"Search failed: {str(e)}"


# =============================================================================
# MAIN WEB SEARCH TOOL (With Smart Processing)
# =============================================================================

@tool
def web_search_tool(query: str) -> str:
    """
    Search the internet for Java migration solutions and error fixes.

    Use this tool when:
    - You encounter unfamiliar errors
    - Your fix attempts have failed 2-3 times
    - You need current best practices for Spring Boot/JUnit migration

    The tool automatically optimizes your query and deduplicates searches.

    Args:
        query: Search query - can be raw error message or specific question

    Returns:
        Raw search results - extract actionable steps yourself

    Example queries:
        - "Spring Boot 1.5 NoClassDefFoundError cglib JDK 17"
        - "javax to jakarta migration Spring Boot 3"
        - "JUnit 4 to JUnit 5 SpringRunner migration"
    """
    log_agent(f"[WEB_SEARCH] Incoming query: {query[:100]}...")

    # Check if search is configured
    if not TAVILY_API_KEY:
        log_agent("[WEB_SEARCH] TAVILY_API_KEY not set")
        return (
            "Web search not configured (TAVILY_API_KEY not set).\n\n"
            "General migration tips:\n"
            "- Spring Boot 1.5.x only works with JDK 8\n"
            "- Spring Boot 2.x works with JDK 8-17\n"
            "- Spring Boot 3.x requires JDK 17+\n"
        )

    try:
        # Get search processor (singleton)
        processor = get_search_processor()

        # Execute search with query optimization and deduplication
        result = processor.search(
            query=query,
            bny_search_fn=_raw_web_search,
        )

        # Log stats
        stats = processor.get_stats()
        log_summary(
            f"[WEB_SEARCH] Complete | "
            f"Cache: {stats['cache_hits']}/{stats['total_searches']}"
        )

        return result

    except Exception as e:
        log_agent(f"[WEB_SEARCH] Error: {str(e)}, falling back to raw search")
        return _raw_web_search(query)


# =============================================================================
# OPENREWRITE AGENT TOOL
# =============================================================================

@tool
def call_openrewrite_agent(command: str) -> str:
    """
    [DOCUMENTATION QUERY ONLY] Query for OpenRewrite recipe guidance and recommendations.

    Use this to understand which recipes to use for specific migration tasks.
    This tool does NOT execute recipes - use mvn_rewrite_run for that.

    Args:
        command: Your question about OpenRewrite recipes

    Example queries:
        - "how to migrate Java 8 to target version"
        - "which recipe fixes javax to jakarta imports"
        - "Spring Boot 2 to 3 migration recipes"
    """
    log_agent(f"[OPENREWRITE_RAG] Query: {command[:100]}...")

    # Get target Java version from environment
    target_java_version = os.environ.get("TARGET_JAVA_VERSION", "21")

    # TODO: Replace with BNY OpenRewrite RAG Agent when available
    # This is a knowledge-based fallback

    response = f"""OpenRewrite Recipe Guidance for: "{command[:80]}..."

📋 **Common Migration Recipes:**

**Java Version Upgrade (TARGET: Java {target_java_version}):**
- `org.openrewrite.java.migrate.UpgradeToJava{target_java_version}` - Upgrade to Java {target_java_version} (YOUR TARGET)

**Spring Boot Migration:**
- `org.openrewrite.java.spring.boot2.UpgradeSpringBoot_2_7` - Upgrade to 2.7.x
- `org.openrewrite.java.spring.boot3.UpgradeSpringBoot_3_0` - Upgrade to 3.0.x (requires JDK 17+)
- `org.openrewrite.java.spring.boot3.UpgradeSpringBoot_3_2` - Upgrade to 3.2.x

**Jakarta EE Migration (javax → jakarta):**
- `org.openrewrite.java.migrate.jakarta.JavaxMigrationToJakarta` - Full javax to jakarta
- `org.openrewrite.java.migrate.jakarta.JakartaEE10` - Jakarta EE 10

**JUnit Migration:**
- `org.openrewrite.java.testing.junit5.JUnit4to5Migration` - Full JUnit 4 to 5
- `org.openrewrite.java.testing.junit5.JUnit5BestPractices` - Apply JUnit 5 best practices

**Usage:**
1. Add recipes with: configure_openrewrite_recipes(project_path, ["recipe.name"])
2. Execute with: mvn_rewrite_run(project_path)
3. Or run specific recipe: mvn_rewrite_run_recipe(project_path, "recipe.name")

📖 **Documentation:** https://docs.openrewrite.org/
"""

    log_agent("[OPENREWRITE_RAG] Returned recipe guidance")
    return response


# =============================================================================
# EXPORTS
# =============================================================================

web_search_toolkit = [web_search_tool, call_openrewrite_agent]

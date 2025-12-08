"""
Web Search and OpenRewrite Agent Tools

NOTE: These tools are currently disabled as they require external services.
- web_search_tool: Requires web search API (Tavily, etc.)
- call_openrewrite_agent: Requires RAG infrastructure

The tools return informative messages directing agents to use alternative approaches.
"""

from langchain_core.tools import tool


@tool
def web_search_tool(query: str) -> str:
    """
    Use this tool to search the internet for information.

    NOTE: This tool is currently disabled. Web search functionality is not available.
    Please use your existing knowledge or ask the user for guidance.
    """
    return (
        "Web search is currently disabled. "
        "Please proceed with your existing knowledge about Java migration best practices, "
        "or consult the OpenRewrite documentation at https://docs.openrewrite.org/ manually."
    )


@tool
def call_openrewrite_agent(command: str) -> str:
    """
    [DOCUMENTATION QUERY ONLY] Query for OpenRewrite recipe guidance and recommendations.

    NOTE: The RAG agent is currently disabled. This tool provides general guidance instead.

    For OpenRewrite recipes, you can:
    1. Use get_available_recipes() to see common recipes
    2. Use suggest_recipes_for_java_version() for version migration recipes
    3. Consult https://docs.openrewrite.org/ for detailed documentation

    Common recipes for Java migration:
    - org.openrewrite.java.migrate.UpgradeToJava21 (Java version upgrade)
    - org.openrewrite.java.spring.boot3.UpgradeSpringBoot_3_0 (Spring Boot 2 to 3)
    - org.openrewrite.java.migrate.jakarta.JavaxMigrationToJakarta (javax to jakarta)
    - org.openrewrite.java.testing.junit5.JUnit4to5Migration (JUnit 4 to 5)
    """
    return f"""OpenRewrite RAG agent is currently disabled.

For your query: "{command[:100]}..."

Please use the following approach instead:
1. Check available recipes with get_available_recipes()
2. Use suggest_recipes_for_java_version() for version-specific recipes
3. Configure recipes using configure_openrewrite_recipes()
4. Execute with mvn_rewrite_run() or mvn_rewrite_run_recipe()

Common OpenRewrite recipes:
- Java 21: org.openrewrite.java.migrate.UpgradeToJava21
- Spring Boot 3: org.openrewrite.java.spring.boot3.UpgradeSpringBoot_3_0
- Jakarta EE: org.openrewrite.java.migrate.jakarta.JavaxMigrationToJakarta
- JUnit 5: org.openrewrite.java.testing.junit5.JUnit4to5Migration

For detailed documentation, visit: https://docs.openrewrite.org/
"""


web_search_toolkit = [web_search_tool, call_openrewrite_agent]

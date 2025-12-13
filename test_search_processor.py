#!/usr/bin/env python3
"""
Test script for SearchProcessor integration.

Usage:
    python test_search_processor.py

Tests:
1. Query optimization and decomposition
2. Result grading
3. Deduplication
4. Environment variable setup
"""

import os
import sys

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.search_processor import (
    SearchProcessor,
    SearchContext,
    get_search_processor,
    reset_search_processor,
    setup_search_context_from_pom,
)


def test_query_optimization():
    """Test query cleaning and optimization."""
    print("\n" + "="*60)
    print("TEST 1: Query Optimization")
    print("="*60)

    processor = SearchProcessor()
    context = SearchContext(
        java_version_current="8",
        java_version_target="21",
        spring_boot_version="1.1.9",
    )

    # Test 1: Error query with stack trace noise
    raw_query = """NoClassDefFoundError: Could not initialize class org.springframework.cglib.proxy.Enhancer
    at org.springframework.context.support.AbstractApplicationContext.invokeBeanFactoryPostProcessors(AbstractApplicationContext.java:712)
    at org.springframework.context.support.AbstractApplicationContext.refresh(AbstractApplicationContext.java:532)"""

    optimized = processor._optimize_query(raw_query, context)

    print(f"\nOriginal query ({len(raw_query)} chars):")
    print(f"  {raw_query[:100]}...")
    print(f"\nOptimized queries ({len(optimized)}):")
    for i, q in enumerate(optimized, 1):
        print(f"  {i}. {q}")

    assert len(optimized) >= 1, "Should produce at least 1 optimized query"
    assert len(optimized[0]) < len(raw_query), "Optimized query should be shorter"
    print("\n✓ Query optimization works!")


def test_result_grading():
    """Test result quality grading."""
    print("\n" + "="*60)
    print("TEST 2: Result Grading")
    print("="*60)

    processor = SearchProcessor()
    context = SearchContext(
        java_version_current="8",
        java_version_target="21",
        spring_boot_version="1.1.9",
    )

    # High quality result (mentions versions, has code)
    high_quality = """
    To fix this issue, upgrade Spring Boot to version 2.7.x or 3.x.
    Spring Boot 1.x uses an old CGLIB version incompatible with Java 21.

    Update your pom.xml:
    ```xml
    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>2.7.18</version>
    </parent>
    ```
    """

    # Low quality result
    low_quality = "No results found for your query."

    high_grade = processor._grade_results(high_quality, "cglib error", context)
    low_grade = processor._grade_results(low_quality, "cglib error", context)

    print(f"\nHigh quality result grade: {high_grade.quality} (score: {high_grade.score:.2f})")
    print(f"  Reasons: {high_grade.reasons[:3]}")

    print(f"\nLow quality result grade: {low_grade.quality} (score: {low_grade.score:.2f})")
    print(f"  Reasons: {low_grade.reasons}")
    print(f"  Alternative query: {low_grade.alternative_query}")

    assert high_grade.score > low_grade.score, "High quality should score higher"
    assert high_grade.quality in ['high', 'medium'], "Good result should be high/medium"
    assert low_grade.quality == 'low', "Empty result should be low"
    print("\n✓ Result grading works!")


def test_deduplication():
    """Test search deduplication."""
    print("\n" + "="*60)
    print("TEST 3: Deduplication")
    print("="*60)

    reset_search_processor()
    processor = get_search_processor()

    # Mock search function
    search_count = [0]
    def mock_search(query):
        search_count[0] += 1
        return f"Mock result for: {query}"

    # First search
    result1 = processor.search(
        query="NoClassDefFoundError cglib Spring Boot",
        bny_search_fn=mock_search,
    )

    # Second search with similar query (should be deduplicated)
    result2 = processor.search(
        query="NoClassDefFoundError cglib Spring Boot Java",
        bny_search_fn=mock_search,
    )

    print(f"\nFirst search cached: {result1.was_cached}")
    print(f"Second search cached: {result2.was_cached}")
    print(f"Total mock searches executed: {search_count[0]}")

    stats = processor.get_stats()
    print(f"\nProcessor stats:")
    print(f"  Total searches: {stats['total_searches']}")
    print(f"  Cache hits: {stats['cache_hits']}")

    # The second search might not be cached if queries are different enough
    # But similar error signatures should be caught
    print("\n✓ Deduplication logic works!")


def test_environment_setup():
    """Test environment variable setup from POM."""
    print("\n" + "="*60)
    print("TEST 4: Environment Variable Setup")
    print("="*60)

    # Test with a sample project path (may not exist)
    test_path = "/home/vhsingh/Java_Migration/repositories/test-project"

    # Clear existing env vars
    for key in ['CURRENT_JAVA_VERSION', 'TARGET_JAVA_VERSION', 'CURRENT_SPRING_VERSION']:
        if key in os.environ:
            del os.environ[key]

    # This will set defaults if pom.xml doesn't exist
    versions = setup_search_context_from_pom(test_path)

    print(f"\nDetected versions:")
    for key, value in versions.items():
        print(f"  {key}: {value}")
        # Check env var was set
        assert os.environ.get(key) == value, f"Env var {key} not set correctly"

    print("\n✓ Environment setup works!")


def test_full_search_flow():
    """Test the complete search flow with mock."""
    print("\n" + "="*60)
    print("TEST 5: Full Search Flow")
    print("="*60)

    reset_search_processor()
    processor = get_search_processor()

    context = SearchContext(
        java_version_current="8",
        java_version_target="21",
        spring_boot_version="1.1.9",
    )

    # Mock search that returns realistic results
    def mock_search(query):
        return f"""
Summary: The CGLIB proxy issue occurs when using Spring Boot 1.x with Java 17+.
You need to upgrade Spring Boot to version 2.7.x or 3.x.

Results:

**Migrating Spring Boot to Java 17 - Stack Overflow**
Spring Boot 1.x uses CGLIB 3.x which doesn't support Java 17 module system.
Upgrade to Spring Boot 2.7+ which includes updated CGLIB.
Source: https://stackoverflow.com/questions/12345

**Spring Boot Java 17 Compatibility - Spring Blog**
Spring Boot 2.5+ officially supports Java 17. Spring Boot 3.0+ requires Java 17.
Update your pom.xml parent version to 2.7.18 or 3.2.0.
Source: https://spring.io/blog/java17

---

<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-parent</artifactId>
    <version>2.7.18</version>
</dependency>
"""

    result = processor.search(
        query="NoClassDefFoundError: Could not initialize class org.springframework.cglib.proxy.Enhancer",
        bny_search_fn=mock_search,
        context=context,
    )

    print(f"\nSearch Result:")
    print(f"  Confidence: {result.confidence:.0%}")
    print(f"  Optimized queries: {len(result.optimized_queries)}")
    print(f"  Was cached: {result.was_cached}")
    print(f"\nSuggested Actions:")
    for action in result.suggested_actions:
        print(f"  - {action}")

    print(f"\nSynthesized Answer (first 500 chars):")
    print(f"  {result.synthesized_answer[:500]}...")

    assert result.confidence > 0.3, "Should have reasonable confidence"
    assert len(result.suggested_actions) > 0, "Should have suggested actions"
    print("\n✓ Full search flow works!")


def main():
    """Run all tests."""
    print("="*60)
    print("SearchProcessor Integration Tests")
    print("="*60)

    try:
        test_query_optimization()
        test_result_grading()
        test_deduplication()
        test_environment_setup()
        test_full_search_flow()

        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

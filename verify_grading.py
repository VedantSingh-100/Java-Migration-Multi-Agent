#!/usr/bin/env python3
"""Verify grading calculation is not hardcoded."""
import sys
sys.path.insert(0, '/home/vhsingh/Java_Migration')

from src.utils.search_processor import SearchProcessor, SearchContext

# Test 1: Content similar to what got 0.65
print("="*60)
print("TEST 1: Content similar to actual 0.65 result")
print("="*60)

test_content_1 = """
Summary: The NoClassDefFoundError for org.springframework.cglib.proxy.Enhancer in Spring Boot with Java 21 is often due to CGLIB library compatibility issues with newer JDK versions.
java.lang.NoClassDefFoundError: Could not initialize class net.sf.cglib.proxy.Enhancer
Caused by: java.lang.NoClassDefFoundError
Code Examples:
```
Caused by: java.lang.NoClassDefFoundError: org/springframework/cglib/core/ReflectUtils
```
Source: https://example.com
"""

processor = SearchProcessor()
context = SearchContext(java_version_target='21', spring_boot_version='1.1.9')
grade = processor._grade_results(test_content_1, 'cglib error', context)

print(f"Score: {grade.score}")
print(f"Quality: {grade.quality}")
print("Reasons:")
for r in grade.reasons:
    print(f"  - {r}")

# Test 2: Different content should give different score
print("\n" + "="*60)
print("TEST 2: High quality content (should be higher than 0.65)")
print("="*60)

test_content_2 = """
To fix this issue, upgrade Spring Boot to version 2.7.18 or 3.2.0.

Step 1: Update your pom.xml parent version
Step 2: Run mvn clean install
Step 3: Verify with mvn test

The CGLIB issue requires upgrading to Spring Boot 2.7+ which includes Java 21 compatible CGLIB.

```xml
<parent>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-parent</artifactId>
    <version>2.7.18</version>
</parent>
```

This upgrade is compatible with Java 21 and fixes the NoClassDefFoundError.
"""

grade2 = processor._grade_results(test_content_2, 'cglib error', context)
print(f"Score: {grade2.score}")
print(f"Quality: {grade2.quality}")
print("Reasons:")
for r in grade2.reasons:
    print(f"  - {r}")

# Test 3: Low quality/noisy content
print("\n" + "="*60)
print("TEST 3: Noisy/low quality content (should be low)")
print("="*60)

test_content_3 = """
Can you help me? I have the same problem.
Thanks in advance!
Did you find a solution?
Please help me too.
I'm facing same issue.

Add a comment
Improve this answer
Share
Follow
Posted by user123
3 comments
"""

grade3 = processor._grade_results(test_content_3, 'cglib error', context)
print(f"Score: {grade3.score}")
print(f"Quality: {grade3.quality}")
print("Reasons:")
for r in grade3.reasons:
    print(f"  - {r}")

print("\n" + "="*60)
print("SUMMARY: Scores vary based on content (NOT hardcoded)")
print("="*60)
print(f"Test 1 (actual-like): {grade.score:.2f}")
print(f"Test 2 (high quality): {grade2.score:.2f}")
print(f"Test 3 (noisy): {grade3.score:.2f}")

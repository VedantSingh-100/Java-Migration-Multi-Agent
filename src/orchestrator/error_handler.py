"""
Error Handler for Migration Orchestrator

This module handles:
- Unified error classification (MavenErrorType enum)
- Build/test error detection
- Error history tracking (ERROR_HISTORY.md)
- Stuck loop detection
- Error resolution tracking

The UnifiedErrorClassifier provides a SINGLE SOURCE OF TRUTH for all
error classification in the system. Both command_executor.py and
agent_wrappers.py use this classifier for consistent error handling.

Errors are tracked in ERROR_HISTORY.md to prevent infinite retry loops
and help diagnose persistent issues.
"""

import os
import re
import hashlib
import json
from enum import Enum
from datetime import datetime
from collections import Counter
from typing import List, Tuple, Optional, Dict, Any

from langchain_core.messages import BaseMessage

from src.utils.logging_config import log_agent, log_summary
from .constants import MAX_ERROR_ATTEMPTS, MAX_LOOPS_WITHOUT_PROGRESS


# =============================================================================
# TOOL RESULT CATEGORIZATION - For Signature-Based Loop Detection
# =============================================================================

class ToolResultCategory:
    """
    Categories for tool execution results.

    Used in signature-based loop detection to distinguish between:
    - Successful tool executions (not a problem to repeat with different args)
    - No-match results (may indicate stuck searching for non-existent pattern)
    - Empty results (may indicate stuck committing nothing)
    - Errors (may indicate fundamental problem)
    """
    SUCCESS = "success"        # Tool worked as expected
    NO_MATCH = "no_match"      # find_replace couldn't find pattern
    EMPTY_RESULT = "empty"     # commit with nothing to commit
    ERROR = "error"            # Tool failed with error


def categorize_tool_result(tool_name: str, result: str) -> str:
    """
    Categorize a tool result for signature-based loop detection.

    Args:
        tool_name: Name of the tool that was called
        result: The result string returned by the tool

    Returns:
        One of ToolResultCategory values: 'success', 'no_match', 'empty', 'error'
    """
    if not result:
        return ToolResultCategory.EMPTY_RESULT

    result_lower = result.lower()

    # NO_MATCH patterns - search/replace couldn't find target
    no_match_patterns = [
        'no match', 'no matches', 'no occurrences', 'not found',
        'pattern not found', 'could not find', 'string not found',
        '0 occurrences', 'zero occurrences', 'no results'
    ]
    if any(p in result_lower for p in no_match_patterns):
        return ToolResultCategory.NO_MATCH

    # EMPTY_RESULT patterns - operation had nothing to do
    empty_patterns = [
        'nothing to commit', 'no changes', 'already up to date',
        'working tree clean', 'no files to commit', 'nothing added',
        'no modifications', 'no staged changes', 'already exists',
        'no difference', 'identical'
    ]
    if any(p in result_lower for p in empty_patterns):
        return ToolResultCategory.EMPTY_RESULT

    # ERROR patterns - tool failed
    error_patterns = [
        'error', 'failed', 'exception', 'build failure', 'return code: 1',
        'compilation error', 'test failure', 'cannot', 'unable to',
        'permission denied', 'not accessible', 'timeout', 'timed out'
    ]
    # Be careful not to match "error_expert" or similar non-error strings
    for p in error_patterns:
        if p in result_lower:
            # Make sure it's not part of a compound word
            if p == 'error' and 'error_' in result_lower:
                continue
            return ToolResultCategory.ERROR

    return ToolResultCategory.SUCCESS


def hash_tool_args(args: Dict[str, Any]) -> str:
    """
    Create stable hash of tool arguments for comparison.

    This allows us to detect when the same tool is called with the
    exact same arguments multiple times (a sign of being stuck).

    Args:
        args: Dictionary of tool arguments

    Returns:
        8-character MD5 hash of the sorted JSON representation
    """
    if not args:
        return "no_args_"

    try:
        # Sort keys for stable ordering
        # Handle non-serializable values by converting to string
        serializable_args = {}
        for k, v in args.items():
            try:
                json.dumps(v)
                serializable_args[k] = v
            except (TypeError, ValueError):
                serializable_args[k] = str(v)

        sorted_args = json.dumps(serializable_args, sort_keys=True)
        return hashlib.md5(sorted_args.encode()).hexdigest()[:8]
    except Exception:
        # Fallback for any serialization issues
        return hashlib.md5(str(args).encode()).hexdigest()[:8]


# =============================================================================
# MAVEN ERROR TYPE ENUM - Single Source of Truth
# =============================================================================

class MavenErrorType(Enum):
    """
    Comprehensive error classification for Maven build outputs.

    This enum is the SINGLE SOURCE OF TRUTH for error classification across
    the entire migration system. Used by:
    - command_executor.py (for auto-fix decisions)
    - agent_wrappers.py (for routing decisions)
    - supervisor_orchestrator.py (for state management)

    Categories:
    - Infrastructure: Auto-fixable network/auth issues
    - Dependency: Artifact resolution issues
    - Build: Compilation and test failures
    - Migration: Spring/Jakarta specific issues
    - Control: Success and unknown states
    """

    # Infrastructure errors (auto-fixable)
    SSL_CERTIFICATE = "ssl_certificate"
    AUTH_401 = "auth_401"
    AUTH_403 = "auth_403"
    NETWORK_TIMEOUT = "network_timeout"

    # Dependency errors
    ARTIFACT_NOT_FOUND = "artifact_not_found"
    DEPENDENCY_CONFLICT = "dependency_conflict"
    VERSION_MISMATCH = "version_mismatch"

    # Build errors (need error_expert)
    COMPILATION_ERROR = "compilation_error"
    TEST_FAILURE = "test_failure"
    POM_SYNTAX_ERROR = "pom_syntax_error"

    # Migration-specific errors
    JAVA_VERSION_ERROR = "java_version_error"
    SPRING_MIGRATION_ERROR = "spring_migration_error"
    JAKARTA_MIGRATION_ERROR = "jakarta_migration_error"

    # Runtime/execution errors
    RUNTIME_ERROR = "runtime_error"

    # Generic build failure (catch-all)
    GENERIC_BUILD_FAILURE = "generic_build_failure"

    # Control states
    UNKNOWN = "unknown"
    SUCCESS = "success"

    @classmethod
    def from_string(cls, value: str) -> 'MavenErrorType':
        """
        Convert string to enum with flexible matching.

        Supports:
        1. Exact enum value match (e.g., "compilation_error")
        2. Alias mapping for common variations (e.g., "compile" -> "compilation_error")
        3. Partial match extraction (e.g., "compilation_error:" -> "compilation_error")

        Falls back to UNKNOWN if no match found.
        """
        if not value:
            return cls.UNKNOWN

        # Clean up the value - handle LLM responses that might have extra text
        value_lower = value.lower().strip()

        # Remove common LLM response artifacts
        # Handle cases like "compilation_error." or "compilation_error:"
        value_lower = value_lower.rstrip('.:,;')

        # If LLM returns something like "The error is compilation_error", extract the category
        for member in cls:
            if member.value in value_lower:
                return member

        # Check for exact match
        for member in cls:
            if member.value == value_lower:
                return member

        # Check aliases (imported at module level, but referenced here)
        # LLM_RESPONSE_ALIASES is defined below the class, so we use a direct reference
        from src.orchestrator.error_handler import LLM_RESPONSE_ALIASES
        if value_lower in LLM_RESPONSE_ALIASES:
            canonical_value = LLM_RESPONSE_ALIASES[value_lower]
            for member in cls:
                if member.value == canonical_value:
                    log_summary(f"[ERROR_CLASSIFIER] Alias match: '{value}' -> '{canonical_value}'")
                    return member

        # Try partial word matching as last resort
        # E.g., "compile error" should match "compilation_error"
        words = value_lower.replace('_', ' ').replace('-', ' ').split()
        for word in words:
            if word in LLM_RESPONSE_ALIASES:
                canonical_value = LLM_RESPONSE_ALIASES[word]
                for member in cls:
                    if member.value == canonical_value:
                        log_summary(f"[ERROR_CLASSIFIER] Partial word match: '{word}' in '{value}' -> '{canonical_value}'")
                        return member

        log_summary(f"[ERROR_CLASSIFIER] Unknown error type string: '{value}', defaulting to UNKNOWN")
        return cls.UNKNOWN

    @classmethod
    def get_category(cls, error_type: 'MavenErrorType') -> str:
        """Get the category of an error type."""
        infrastructure = {cls.SSL_CERTIFICATE, cls.AUTH_401, cls.AUTH_403, cls.NETWORK_TIMEOUT}
        dependency = {cls.ARTIFACT_NOT_FOUND, cls.DEPENDENCY_CONFLICT, cls.VERSION_MISMATCH}
        build = {cls.COMPILATION_ERROR, cls.TEST_FAILURE, cls.POM_SYNTAX_ERROR, cls.RUNTIME_ERROR, cls.GENERIC_BUILD_FAILURE}
        migration = {cls.JAVA_VERSION_ERROR, cls.SPRING_MIGRATION_ERROR, cls.JAKARTA_MIGRATION_ERROR}

        if error_type in infrastructure:
            return "infrastructure"
        elif error_type in dependency:
            return "dependency"
        elif error_type in build:
            return "build"
        elif error_type in migration:
            return "migration"
        elif error_type == cls.SUCCESS:
            return "success"
        else:
            return "unknown"

    @classmethod
    def is_auto_fixable(cls, error_type: 'MavenErrorType') -> bool:
        """Check if this error type can be automatically fixed."""
        auto_fixable = {cls.SSL_CERTIFICATE, cls.AUTH_401, cls.AUTH_403, cls.NETWORK_TIMEOUT}
        return error_type in auto_fixable

    @classmethod
    def requires_error_expert(cls, error_type: 'MavenErrorType') -> bool:
        """Check if this error type requires routing to error_expert."""
        requires_expert = {
            cls.ARTIFACT_NOT_FOUND, cls.DEPENDENCY_CONFLICT, cls.VERSION_MISMATCH,
            cls.COMPILATION_ERROR, cls.TEST_FAILURE, cls.POM_SYNTAX_ERROR,
            cls.JAVA_VERSION_ERROR, cls.SPRING_MIGRATION_ERROR, cls.JAKARTA_MIGRATION_ERROR,
            cls.RUNTIME_ERROR, cls.GENERIC_BUILD_FAILURE, cls.UNKNOWN
        }
        return error_type in requires_expert


# =============================================================================
# PATTERN DEFINITIONS - Comprehensive regex patterns for error detection
# =============================================================================

# Priority order: Migration-specific → Infrastructure → Build → Fallback
# More specific patterns are checked first

# Migration-specific errors (highest priority - most actionable)
JAKARTA_MIGRATION_PATTERNS = [
    r'package\s+javax\.[a-z]+.*does\s+not\s+exist',
    r'cannot\s+access\s+javax\.',
    r'cannot\s+find\s+symbol.*javax\.',
    r'class\s+file\s+for\s+javax\.',
    r'javax\.servlet',
    r'javax\.persistence',
    r'javax\.validation',
    r'javax\.annotation',
    r'javax\.inject',
    r'javax\.enterprise',
    r'javax\.ws\.rs',
    r'javax\.xml\.bind',
    r'javax\.mail',
    r'javax\.activation',
    r'javax\.transaction',
    r'javax\.jms',
    r'javax\.websocket',
    r'javax\.json',
    r'javax\.faces',
    r'javax\.el',
]

SPRING_MIGRATION_PATTERNS = [
    r'org\.springframework\.boot\..*does\s+not\s+exist',
    r'cannot\s+find\s+symbol.*springframework',
    r'WebMvcConfigurerAdapter',  # Deprecated in Spring 5
    r'ErrorController.*getErrorPath',  # Changed in Spring Boot 2.3+
    r'SpringBootServletInitializer.*does\s+not\s+exist',
    r'org\.springframework\.boot\.context\.web',  # Package moved
    r'antMatchers.*cannot\s+find',  # Changed in Spring Security 6
    r'authorizeRequests.*cannot\s+find',  # Changed to authorizeHttpRequests
    # Spring Boot 3.x specific
    r'BeanCreationException',
    r'UnsatisfiedDependencyException',
    r'NoSuchBeanDefinitionException',
    r'BeanDefinitionStoreException',
    r'ApplicationContextException',
    r'Consider\s+defining\s+a\s+bean',
    r'No\s+qualifying\s+bean',
    r'Failed\s+to\s+determine\s+a\s+suitable\s+driver\s+class',
    r'Unable\s+to\s+find\s+main\s+class',
    r'spring-boot-maven-plugin.*repackage.*failed',
    r'spring-boot-maven-plugin.*Unable',
    # Spring Security changes
    r'mvcMatchers.*cannot\s+find',
    r'csrf\(\)\.disable\(\)',
    r'and\(\).*cannot\s+find\s+symbol',
]

JAVA_VERSION_PATTERNS = [
    r'source\s+option\s+\d+\s+is\s+no\s+longer\s+supported',
    r'target\s+option\s+\d+\s+is\s+no\s+longer\s+supported',
    r'release\s+version\s+\d+\s+not\s+supported',
    r'java\.lang\.UnsupportedClassVersionError',
    r'class\s+file\s+has\s+wrong\s+version\s+\d+\.\d+',
    r'unsupported\s+class\s+file.*version\s+\d+',
    r'has\s+been\s+compiled\s+by\s+a\s+more\s+recent\s+version',
    # Java module system errors
    r'module\s+\S+\s+does\s+not\s+export',
    r'package\s+\S+\s+is\s+declared\s+in\s+module',
    r'cannot\s+access\s+class.*because\s+module',
    r'IllegalAccessError.*module',
    # Additional version patterns
    r'Fatal\s+error\s+compiling.*release\s+version',
    r'invalid\s+flag.*--release',
    r'option\s+--release\s+not\s+allowed',
]

# Infrastructure errors (auto-fixable)
SSL_PATTERNS = [
    r'SSLHandshakeException',
    r'SSLException',
    r'PKIX\s+path\s+building\s+failed',
    r'unable\s+to\s+find\s+valid\s+certification\s+path',
    r'SSL\s+peer\s+shut\s+down\s+incorrectly',
    r'sun\.security\.validator\.ValidatorException',
    r'certificate.*expired',
    r'certificate.*untrusted',
    r'Received\s+fatal\s+alert',
]

AUTH_401_PATTERNS = [
    r'401\s*Unauthorized',
    r'Status\s*code:\s*401',
    r'authentication.*required',
    r'Not\s+authorized',
    r'access\s+denied.*401',
    r'HTTP\s+response\s+code:\s*401',
]

AUTH_403_PATTERNS = [
    r'403\s*Forbidden',
    r'Status\s*code:\s*403',
    r'Access.*denied',
    r'not\s+permitted',
    r'authorization.*failed',
    r'HTTP\s+response\s+code:\s*403',
]

NETWORK_PATTERNS = [
    r'Connection\s+timed\s+out',
    r'Network\s+is\s+unreachable',
    r'Could\s+not\s+connect\s+to',
    r'Read\s+timed\s+out',
    r'Connection\s+refused',
    r'UnknownHostException',
    r'No\s+route\s+to\s+host',
    r'SocketTimeoutException',
]

# Dependency errors
ARTIFACT_NOT_FOUND_PATTERNS = [
    r'Could\s+not\s+find\s+artifact',
    r'Could\s+not\s+resolve\s+dependencies',
    r'Failure\s+to\s+find',
    r'was\s+not\s+found\s+in',
    r'artifact.*not\s+found',
    r'Cannot\s+resolve.*artifact',
    r'missing\s+artifact',
]

DEPENDENCY_CONFLICT_PATTERNS = [
    r'Detected\s+both\s+.*\s+and\s+.*',
    r'version\s+conflict',
    r'incompatible\s+versions',
    r'requires\s+version\s+.*\s+but\s+.*\s+found',
    r'Multiple\s+bindings\s+were\s+found',
    r'conflicting\s+dependency',
]

VERSION_MISMATCH_PATTERNS = [
    r'class\s+file\s+has\s+wrong\s+version',
    r'unsupported\s+class\s+file.*version',
    r'invalid\s+target\s+release',
    r'invalid\s+source\s+release',
    r'source\s+release\s+\d+\s+requires\s+target\s+release',
    r'major\s+version\s+\d+\s+is\s+newer',
]

# Build errors
COMPILATION_PATTERNS = [
    # Core compilation errors
    r'cannot\s+find\s+symbol',
    r'COMPILATION\s+ERROR',
    r'package\s+.*\s+does\s+not\s+exist',
    r'class\s+.*\s+does\s+not\s+exist',
    r'incompatible\s+types',
    r'method\s+.*\s+cannot\s+be\s+applied',
    r'non-static\s+.*\s+cannot\s+be\s+referenced',
    r'unreported\s+exception',
    r'illegal\s+start\s+of\s+expression',
    r'reached\s+end\s+of\s+file\s+while\s+parsing',
    r'unclosed\s+string\s+literal',
    r'\[ERROR\].*\.java:\[\d+,\d+\]',  # javac error format
    r'error:\s+\[',  # javac error format
    r'cannot\s+be\s+applied\s+to',
    r'is\s+not\s+abstract\s+and\s+does\s+not\s+override',
    r'has\s+private\s+access',
    r'cannot\s+access',
    r'bad\s+operand',
    # Plugin-specific compilation failures
    r'maven-compiler-plugin.*Compilation\s+failure',
    r'maven-compiler-plugin.*compile.*failed',
    r'Fatal\s+error\s+compiling',
    # Annotation processing errors
    r'annotation\s+processing.*error',
    r'error:\s+annotation\s+processor',
    r'Annotation\s+processing\s+found\s+errors',
    # Additional compilation patterns
    r'type\s+.*\s+does\s+not\s+take\s+parameters',
    r'constructor\s+.*\s+cannot\s+be\s+applied',
    r'variable\s+.*\s+might\s+not\s+have\s+been\s+initialized',
    r'variable\s+.*\s+is\s+already\s+defined',
    r'duplicate\s+class',
    r'illegal\s+character',
    r'unmappable\s+character\s+for\s+encoding',
    r'class\s+is\s+public.*should\s+be\s+declared\s+in\s+a\s+file',
    # Generic error indicators
    r'error:\s+cannot\s+find',
    r'error:\s+incompatible',
    r'error:\s+method\s+does\s+not\s+override',
    r'error:\s+no\s+suitable',
]

TEST_FAILURE_PATTERNS = [
    # Maven surefire/failsafe patterns
    r'Tests\s+run:.*Failures:\s*[1-9]',
    r'Tests\s+run:.*Errors:\s*[1-9]',
    r'There\s+are\s+test\s+failures',
    r'Failed\s+tests:',
    r'Tests\s+in\s+error:',
    r'Test\s+.*\s+FAILED',
    r'testCompile.*FAILED',
    # Surefire plugin specific
    r'maven-surefire-plugin.*test.*failed',
    r'maven-failsafe-plugin.*failed',
    r'Surefire.*There\s+are\s+test\s+failures',
    # Assertion errors
    r'java\.lang\.AssertionError',
    r'org\.junit\..*AssertionFailedError',
    r'expected:<.*>\s+but\s+was:<.*>',
    r'org\.opentest4j\.AssertionFailedError',
    # Additional test patterns
    r'org\.junit\.ComparisonFailure',
    r'junit\.framework\.AssertionFailedError',
    r'org\.hamcrest\..*Mismatch',
    r'org\.assertj\..*AssertionError',
    r'org\.mockito\.exceptions',
    # Test execution errors
    r'Could\s+not\s+start\s+test',
    r'Test\s+ignored',
    r'Skipped\s+tests.*due\s+to\s+errors',
    r'Exception\s+in\s+thread.*during\s+test',
]

POM_SYNTAX_PATTERNS = [
    # XML parsing errors
    r'non-?parseable\s+pom',
    r'malformed\s+pom',
    r'Error\s+parsing',
    r'XML\s+parsing\s+error',
    r'premature\s+end\s+of\s+file',
    r'cvc-complex-type',
    r'element.*not\s+allowed\s+here',
    r'content\s+is\s+not\s+allowed\s+in\s+prolog',
    r'must\s+be\s+terminated\s+by\s+the\s+matching',
    r'unrecognised\s+tag',
    r'unrecognized\s+tag',
    r'problems?\s+were\s+encountered\s+while\s+processing\s+the\s+pom',
    r'the\s+build\s+could\s+not\s+read',
    r'dependencies\.dependency\.version.*is\s+missing',
    r'invalid\s+pom',
    # Additional POM errors
    r'Unknown\s+packaging',
    r'Duplicate\s+declaration\s+of\s+plugin',
    r'\'dependencies\.dependency\.(groupId|artifactId)\'\s+is\s+missing',
    r'\'build\.plugins\.plugin\.(groupId|artifactId)\'\s+is\s+missing',
    r'Project\s+ID\s+must\s+not\s+be\s+null',
    r'Parent\s+POM.*not\s+found',
    r'Could\s+not\s+find\s+the\s+selected\s+project',
    r'Invalid\s+packaging\s+for\s+parent',
    r'Circular\s+dependency',
]

# Runtime/Execution errors (during Maven build)
RUNTIME_ERROR_PATTERNS = [
    # Common Java runtime exceptions during build
    r'NullPointerException',
    r'IllegalArgumentException',
    r'IllegalStateException',
    r'ClassNotFoundException',
    r'NoClassDefFoundError',
    r'NoSuchMethodError',
    r'AbstractMethodError',
    r'UnsupportedOperationException',
    r'ClassCastException',
    r'ArrayIndexOutOfBoundsException',
    r'StringIndexOutOfBoundsException',
    r'NumberFormatException',
    # Resource/memory errors
    r'OutOfMemoryError',
    r'Java\s+heap\s+space',
    r'GC\s+overhead\s+limit\s+exceeded',
    r'Metaspace',
    r'PermGen\s+space',
    r'StackOverflowError',
    r'Unable\s+to\s+allocate',
    # Plugin execution failures (catch-all for various plugins)
    r'Failed\s+to\s+execute\s+goal.*plugin',
    r'Execution\s+.*\s+of\s+goal\s+.*\s+failed',
    r'MojoExecutionException',
    r'MojoFailureException',
    r'PluginExecutionException',
    # Resource plugin errors
    r'maven-resources-plugin.*failed',
    r'Mark\s+invalid',
    r'Invalid\s+or\s+corrupt\s+jarfile',
    r'Could\s+not\s+find\s+or\s+load\s+main\s+class',
    # Exec plugin errors
    r'exec-maven-plugin.*Exception',
    r'An\s+exception\s+occurred\s+while\s+executing',
    # Javadoc errors
    r'maven-javadoc-plugin.*error',
    r'Exit\s+code:\s*1.*javadoc',
    r'javadoc:\s+error',
    # Code quality plugin errors
    r'maven-checkstyle-plugin.*violations',
    r'maven-pmd-plugin.*violations',
    r'spotbugs.*violations',
    r'You\s+have\s+\d+\s+(Checkstyle|PMD|SpotBugs)\s+violations',
]

# Generic build failure patterns (catch-all, checked LAST before LLM)
# These are broad patterns that indicate something failed but don't provide specific info
GENERIC_BUILD_FAILURE_PATTERNS = [
    r'BUILD\s+FAILURE',
    r'Failed\s+to\s+execute\s+goal',
    r'\[ERROR\].*failed',
    r'Execution\s+default.*failed',
    r'goal\s+.*\s+failed',
    r'Return\s+code:\s*[1-9]',  # Non-zero return code
]

# Success patterns
SUCCESS_PATTERNS = [
    r'BUILD\s+SUCCESS',
    r'Return\s+code:\s*0',
]

# LLM Classification prompt template
LLM_CLASSIFICATION_PROMPT = """Analyze this Maven build error and classify it into ONE of these categories:

Categories:
- ssl_certificate: SSL/TLS certificate issues
- auth_401: Authentication required (401)
- auth_403: Authorization denied (403)
- network_timeout: Network connectivity issues
- artifact_not_found: Missing Maven artifacts
- dependency_conflict: Version conflicts between dependencies
- version_mismatch: Java version incompatibility
- compilation_error: Java compilation failures (syntax errors, missing symbols, type errors)
- test_failure: Test execution failures (assertion errors, test exceptions)
- pom_syntax_error: POM XML syntax errors (malformed XML, missing elements)
- java_version_error: Java version configuration issues
- spring_migration_error: Spring framework migration issues
- jakarta_migration_error: javax to jakarta namespace issues
- runtime_error: Runtime exceptions during build (NullPointer, ClassNotFound, etc.)
- generic_build_failure: General build failure (when no specific category fits)
- unknown: Cannot determine error type

Maven Output:
```
{error_output}
```

Respond with ONLY the category name (e.g., "compilation_error"), nothing else."""

# Aliases for flexible LLM response parsing
# Maps common variations to canonical MavenErrorType values
LLM_RESPONSE_ALIASES = {
    # Compilation errors
    "compilation": "compilation_error",
    "compile": "compilation_error",
    "compile_error": "compilation_error",
    "compiler_error": "compilation_error",
    "syntax_error": "compilation_error",
    "syntax": "compilation_error",

    # Test failures
    "test": "test_failure",
    "test_error": "test_failure",
    "test_fail": "test_failure",
    "tests_failed": "test_failure",
    "unit_test_failure": "test_failure",

    # POM errors
    "pom": "pom_syntax_error",
    "pom_error": "pom_syntax_error",
    "xml_error": "pom_syntax_error",
    "pom_xml_error": "pom_syntax_error",

    # Dependency errors
    "artifact": "artifact_not_found",
    "missing_artifact": "artifact_not_found",
    "dependency_not_found": "artifact_not_found",
    "missing_dependency": "artifact_not_found",
    "dependency": "artifact_not_found",
    "conflict": "dependency_conflict",

    # Infrastructure errors
    "ssl": "ssl_certificate",
    "certificate": "ssl_certificate",
    "tls": "ssl_certificate",
    "401": "auth_401",
    "unauthorized": "auth_401",
    "authentication": "auth_401",
    "403": "auth_403",
    "forbidden": "auth_403",
    "authorization": "auth_403",
    "network": "network_timeout",
    "timeout": "network_timeout",
    "connection": "network_timeout",

    # Version errors
    "version": "version_mismatch",
    "java_version": "java_version_error",
    "jdk_version": "java_version_error",

    # Migration errors
    "spring": "spring_migration_error",
    "spring_boot": "spring_migration_error",
    "jakarta": "jakarta_migration_error",
    "javax": "jakarta_migration_error",

    # Runtime errors
    "runtime": "runtime_error",
    "exception": "runtime_error",
    "nullpointer": "runtime_error",
    "null_pointer": "runtime_error",

    # Generic/unknown
    "build_failure": "generic_build_failure",
    "build": "generic_build_failure",
    "failure": "generic_build_failure",
    "other": "unknown",
    "none": "unknown",
}


# =============================================================================
# UNIFIED ERROR CLASSIFIER
# =============================================================================

class UnifiedErrorClassifier:
    """
    Single unified error classifier for the entire migration system.

    Uses a layered approach:
    1. Check return code (0 = success)
    2. Pattern matching (fast, regex-based)
    3. LLM fallback (for unrecognized errors)

    Pattern matching priority (most specific first):
    1. Migration-specific (jakarta, spring, java version)
    2. Infrastructure (SSL, auth, network)
    3. Dependency (artifact, conflict, version)
    4. Build (compilation, test, POM)
    """

    # Singleton instance
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._llm_cache: Dict[str, MavenErrorType] = {}
        self._classifier_llm = None  # Lazy initialization
        self._initialized = True
        log_summary("[ERROR_CLASSIFIER] UnifiedErrorClassifier initialized")

    def _get_llm(self):
        """Lazy initialization of LLM for classification using Amazon Bedrock."""
        if self._classifier_llm is None:
            try:
                from langchain_aws import ChatBedrock
                # Use Claude 3.5 Sonnet on Bedrock for classification
                # Using us. prefix for cross-region inference profile
                self._classifier_llm = ChatBedrock(
                    model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                    region_name=os.environ.get("AWS_REGION", "us-east-1"),
                    model_kwargs={
                        "max_tokens": 50,
                        "temperature": 0.0,
                    },
                )
                log_summary("[ERROR_CLASSIFIER] LLM initialized for fallback classification (Claude 3.5 Sonnet)")
            except Exception as e:
                log_summary(f"[ERROR_CLASSIFIER] Failed to initialize LLM: {e}")
                self._classifier_llm = None
        return self._classifier_llm

    def classify(self, output: str, return_code: int) -> MavenErrorType:
        """
        Classify Maven output into an error type.

        Args:
            output: Combined stdout + stderr from Maven command
            return_code: Process return code (0 = success)

        Returns:
            MavenErrorType enum value
        """
        # Step 1: Check return code
        if return_code == 0:
            # Also verify no error patterns in successful builds
            if self._has_success_pattern(output):
                log_summary("[ERROR_CLASSIFIER] SUCCESS (return code 0, BUILD SUCCESS pattern found)")
                return MavenErrorType.SUCCESS
            # Return code 0 but no success pattern - still treat as success
            log_summary("[ERROR_CLASSIFIER] SUCCESS (return code 0)")
            return MavenErrorType.SUCCESS

        # Step 2: Pattern matching (fast path)
        pattern_result = self._pattern_match(output)
        if pattern_result is not None:
            log_summary(f"[ERROR_CLASSIFIER] Pattern match: {pattern_result.value}")
            return pattern_result

        # Step 3: LLM fallback (slow path, cached)
        llm_result = self._llm_classify(output)
        log_summary(f"[ERROR_CLASSIFIER] LLM fallback: {llm_result.value}")
        return llm_result

    def _has_success_pattern(self, output: str) -> bool:
        """Check if output contains success patterns."""
        for pattern in SUCCESS_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE):
                return True
        return False

    def _pattern_match(self, output: str) -> Optional[MavenErrorType]:
        """
        Pattern-based classification with priority ordering.

        Priority (most specific first):
        1. Migration-specific (jakarta, spring, java version)
        2. Infrastructure (SSL, auth, network)
        3. Dependency (artifact, conflict, version)
        4. Build (compilation, test, POM)
        """
        # Migration-specific errors (highest priority - most actionable)
        for pattern in JAKARTA_MIGRATION_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE):
                return MavenErrorType.JAKARTA_MIGRATION_ERROR

        for pattern in SPRING_MIGRATION_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE):
                return MavenErrorType.SPRING_MIGRATION_ERROR

        for pattern in JAVA_VERSION_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE):
                return MavenErrorType.JAVA_VERSION_ERROR

        # Infrastructure errors (can be auto-fixed)
        for pattern in SSL_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE):
                return MavenErrorType.SSL_CERTIFICATE

        for pattern in AUTH_401_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE):
                return MavenErrorType.AUTH_401

        for pattern in AUTH_403_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE):
                return MavenErrorType.AUTH_403

        for pattern in NETWORK_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE):
                return MavenErrorType.NETWORK_TIMEOUT

        # Dependency errors
        for pattern in ARTIFACT_NOT_FOUND_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE):
                return MavenErrorType.ARTIFACT_NOT_FOUND

        for pattern in DEPENDENCY_CONFLICT_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE):
                return MavenErrorType.DEPENDENCY_CONFLICT

        for pattern in VERSION_MISMATCH_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE):
                return MavenErrorType.VERSION_MISMATCH

        # Build errors
        for pattern in POM_SYNTAX_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE):
                return MavenErrorType.POM_SYNTAX_ERROR

        for pattern in TEST_FAILURE_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE):
                return MavenErrorType.TEST_FAILURE

        for pattern in COMPILATION_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE):
                return MavenErrorType.COMPILATION_ERROR

        # Runtime/execution errors (various plugins and exceptions)
        for pattern in RUNTIME_ERROR_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE):
                return MavenErrorType.RUNTIME_ERROR

        # Generic build failure (catch-all, checked LAST before LLM)
        # This ensures we classify as GENERIC_BUILD_FAILURE instead of UNKNOWN
        # when there's a clear failure but no specific pattern matches
        for pattern in GENERIC_BUILD_FAILURE_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE):
                return MavenErrorType.GENERIC_BUILD_FAILURE

        # No pattern matched - will go to LLM fallback
        return None

    def _llm_classify(self, error_output: str) -> MavenErrorType:
        """
        Use LLM to classify unrecognized errors.
        Results are cached by error signature.
        """
        # Create cache key from first 500 chars
        cache_key = hashlib.md5(error_output[:500].encode()).hexdigest()

        if cache_key in self._llm_cache:
            log_summary(f"[ERROR_CLASSIFIER] LLM cache hit for {cache_key[:8]}")
            return self._llm_cache[cache_key]

        llm = self._get_llm()
        if llm is None:
            log_summary("[ERROR_CLASSIFIER] LLM not available, returning UNKNOWN")
            return MavenErrorType.UNKNOWN

        try:
            # Extract most relevant error context
            context = self._extract_error_context(error_output, max_chars=2000)
            prompt = LLM_CLASSIFICATION_PROMPT.format(error_output=context)

            response = llm.invoke(prompt)
            category = response.content.strip().lower()

            # Map to enum
            error_type = MavenErrorType.from_string(category)
            self._llm_cache[cache_key] = error_type

            log_summary(f"[ERROR_CLASSIFIER] LLM classified '{cache_key[:8]}' as: {error_type.value}")
            return error_type

        except Exception as e:
            log_summary(f"[ERROR_CLASSIFIER] LLM fallback failed: {e}")
            return MavenErrorType.UNKNOWN

    def _extract_error_context(self, output: str, max_chars: int = 2000) -> str:
        """
        Extract the most relevant error context from Maven output.
        Focuses on lines containing ERROR, FAILURE, exception info, etc.
        """
        lines = output.split('\n')
        error_lines = []

        # Keywords that indicate relevant error information
        error_keywords = [
            'ERROR', 'FAILURE', 'Failed', 'Exception', 'cannot', 'missing',
            'not found', 'does not exist', 'incompatible', 'invalid',
            'javax.', 'jakarta.', 'springframework', 'Tests run:'
        ]

        for i, line in enumerate(lines):
            if any(kw in line for kw in error_keywords):
                # Include some context around error lines
                start = max(0, i - 1)
                end = min(len(lines), i + 2)
                for j in range(start, end):
                    if lines[j] not in error_lines:
                        error_lines.append(lines[j])

        if error_lines:
            result = '\n'.join(error_lines)
        else:
            # No error keywords found, take last portion of output
            result = output[-max_chars:] if len(output) > max_chars else output

        # Truncate if still too long
        if len(result) > max_chars:
            result = result[:max_chars] + "\n... (truncated)"

        return result

    def classify_from_messages(self, messages: List[BaseMessage]) -> Tuple[MavenErrorType, str]:
        """
        Classify error from LangGraph message history.
        Finds the most recent build tool result and classifies it.

        Args:
            messages: List of LangGraph messages

        Returns:
            (MavenErrorType, error_summary) - The error type and a summary of the error
        """
        for msg in reversed(messages):
            msg_content = ""
            msg_name = ""

            if isinstance(msg, dict):
                msg_content = str(msg.get('content', ''))
                msg_name = msg.get('name', '')
            elif hasattr(msg, 'content'):
                msg_content = str(msg.content)
                msg_name = getattr(msg, 'name', '')

            # Check if this is a build-related message
            if not self._is_build_tool_result(msg_content, msg_name):
                continue

            # Found a build tool result - classify it
            # Determine return code from content
            return_code = 0 if 'Return code: 0' in msg_content or 'BUILD SUCCESS' in msg_content else 1

            error_type = self.classify(msg_content, return_code)

            # Extract error summary
            error_summary = self._extract_error_context(msg_content, max_chars=800) if error_type != MavenErrorType.SUCCESS else ""

            return error_type, error_summary

        # No build tool results found
        return MavenErrorType.SUCCESS, ""

    def _is_build_tool_result(self, content: str, name: str) -> bool:
        """
        Check if a message is a build tool result.

        IMPORTANT: Must be strict to avoid false positives from AI text responses
        that merely mention Maven commands (e.g., "I'll run mvn compile...").
        Only match actual build output, not casual mentions.
        """
        # Check by name first (most reliable)
        if name:
            name_lower = name.lower()
            build_tool_names = ['mvn_compile', 'mvn_test', 'mvn_rewrite', 'maven']
            if any(tool in name_lower for tool in build_tool_names):
                return True

        # Check by content patterns - STRICT patterns only
        # These patterns appear in ACTUAL Maven output, not in AI text responses
        #
        # REMOVED: 'mvn compile', 'mvn test' - these match AI text too easily
        # e.g., "I'll run mvn compile" or "<invoke name="mvn_compile">" in AI response
        # would falsely trigger classification
        build_content_indicators = [
            'Return code:',           # All our Maven tools include this
            'BUILD SUCCESS',          # Actual Maven final status
            'BUILD FAILURE',          # Actual Maven final status
            '[INFO] BUILD',           # Actual Maven output line
            '[INFO] --- maven-compiler-plugin',   # Actual plugin execution
            '[INFO] --- maven-surefire-plugin',   # Actual test plugin execution
            '[ERROR] Failed to execute goal',     # Actual Maven error
            '[INFO] Compiling ',      # Actual compilation output (note the space)
            'Tests run:',             # Actual test summary line
        ]

        return any(indicator in content for indicator in build_content_indicators)


# Create singleton instance for easy import
unified_classifier = UnifiedErrorClassifier()


class ErrorHandler:
    """
    Handles error detection, tracking, and resolution for the migration process.

    Features:
    - Build/test error detection from messages
    - Error history logging to ERROR_HISTORY.md
    - Stuck loop detection
    - Action tracking for progress monitoring
    """

    def __init__(self, project_path: str = None, action_window_size: int = 10):
        """
        Args:
            project_path: Path to project directory
            action_window_size: Number of recent actions to track for loop detection
        """
        self.project_path = project_path
        self.action_window_size = action_window_size
        self.recent_actions = []

    def set_project_path(self, project_path: str):
        """Update the project path"""
        self.project_path = project_path

    def detect_build_error(self, messages: List[BaseMessage]) -> Tuple[bool, str, str]:
        """
        Check if the MOST RECENT build tool result contains errors.

        DEPRECATED: This method now delegates to UnifiedErrorClassifier.
        Use unified_classifier.classify_from_messages() directly for new code.

        Returns:
            (has_error: bool, error_summary: str, error_type: str)
            error_type: Now returns MavenErrorType.value for backwards compatibility
        """
        # Delegate to unified classifier
        error_type, error_summary = unified_classifier.classify_from_messages(messages)

        # Convert to legacy format for backwards compatibility
        has_error = error_type != MavenErrorType.SUCCESS

        # Map new error types to legacy categories for backwards compatibility
        legacy_type = self._map_to_legacy_error_type(error_type)

        log_agent(f"[ERROR_DETECT] Unified classifier result: {error_type.value} -> legacy: {legacy_type}")

        return has_error, error_summary, legacy_type

    def _map_to_legacy_error_type(self, error_type: MavenErrorType) -> str:
        """
        Map new MavenErrorType to legacy error type strings for backwards compatibility.

        Legacy types: 'compile', 'test', 'pom', 'none'
        """
        if error_type == MavenErrorType.SUCCESS:
            return 'none'
        elif error_type == MavenErrorType.TEST_FAILURE:
            return 'test'
        elif error_type == MavenErrorType.POM_SYNTAX_ERROR:
            return 'pom'
        elif error_type in {MavenErrorType.ARTIFACT_NOT_FOUND, MavenErrorType.DEPENDENCY_CONFLICT}:
            return 'pom'  # Dependency issues are handled like POM errors
        else:
            # All other errors (compilation, migration, infrastructure, unknown) -> 'compile'
            return 'compile'

    def log_error_attempt(self, error: str, attempt_num: int,
                          was_successful: bool, attempted_fixes: List[str] = None):
        """
        Record error resolution attempt to ERROR_HISTORY.md.

        Args:
            error: Error message/description
            attempt_num: Which attempt number (1, 2, 3)
            was_successful: Whether the error was resolved
            attempted_fixes: List of fix descriptions
        """
        if not self.project_path:
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        error_snippet = error[:200] if error else "Unknown error"

        # Check if this is a new error or continuation
        error_history = self._read_error_history()
        if f"## Error #{attempt_num}" not in error_history:
            # New error - create header
            self._append_to_error_history(
                f"\n## [x] Error #{attempt_num}: {error_snippet} [{timestamp}]"
            )

        # Record attempt result with details
        status = "RESOLVED" if was_successful else "FAILED"
        self._append_to_error_history(
            f"- [x] Attempt {attempt_num}: {status} [{timestamp}]"
        )

        # Log what fixes were attempted (if provided)
        if attempted_fixes:
            for fix in attempted_fixes:
                self._append_to_error_history(f"  - Tried: {fix}")

        log_agent(f"[ERROR_HISTORY] Logged error attempt #{attempt_num}: {status}")

    def is_error_duplicate(self, error_message: str) -> bool:
        """
        Check if this error has already been attempted too many times.

        Args:
            error_message: The error message to check

        Returns:
            True if this error should be skipped (already maxed attempts)
        """
        error_history = self._read_error_history()
        error_signature = error_message[:100] if error_message else ""

        # Count how many times this error signature appears
        if error_signature:
            count = error_history.count(error_signature)
            if count >= MAX_ERROR_ATTEMPTS:
                log_agent(f"[ERROR_DEDUPE] Error already attempted {count} times - skipping")
                return True

        return False

    def detect_stuck_loop(self) -> Tuple[bool, str]:
        """
        Detect if agent is stuck in a repetitive loop using signature-based detection.

        SIGNATURE-BASED DETECTION:
        A "signature" is (tool_name, args_hash, result_category).
        This allows us to distinguish between:
        - Healthy: commit_changes called 5x with DIFFERENT args and SUCCESS results
        - Stuck: commit_changes called 3x with SAME args and EMPTY results

        Patterns checked:
        1. Same FULL signature repeated (3+ times) with PROBLEMATIC result
        2. No completed actions logged in recent window
        3. Same TODO item attempted repeatedly without completion

        Returns:
            (is_stuck: bool, reason: str)
        """
        if len(self.recent_actions) < self.action_window_size:
            return (False, "Not enough action history yet")

        # Get last N actions
        last_n = self.recent_actions[-self.action_window_size:]

        # Pattern 1: SIGNATURE-BASED detection (NEW IMPLEMENTATION)
        # Group by FULL signature: (tool_name, args_hash, result_category)
        signatures = Counter(
            (
                a.get('tool_name', 'unknown'),
                a.get('args_hash', 'unknown'),
                a.get('result_category', ToolResultCategory.SUCCESS)
            )
            for a in last_n
        )

        for (tool, args_hash, result_cat), count in signatures.items():
            if count >= 3:
                # Key insight: Only flag as stuck if result is PROBLEMATIC
                # SUCCESS with same tool/args is unusual but not necessarily stuck
                # (e.g., reading same file multiple times)
                if result_cat in [ToolResultCategory.NO_MATCH, ToolResultCategory.EMPTY_RESULT, ToolResultCategory.ERROR]:
                    reason = f"Tool '{tool}' with same args ({args_hash}) returned '{result_cat}' {count}x"
                    log_agent(f"[LOOP_DETECT] STUCK: {reason}", "WARNING")
                    return (True, reason)
                else:
                    # Same signature with SUCCESS - log warning but don't flag as stuck
                    log_agent(
                        f"[LOOP_DETECT] Tool '{tool}' called {count}x with same args but SUCCESS - not stuck",
                        "DEBUG"
                    )

        # Pattern 2: No completed actions logged (unchanged - still valid check)
        completions_logged = sum(1 for a in last_n if a.get('logged_to_completed'))
        if completions_logged == 0:
            log_agent(f"[LOOP_DETECT] No actions logged to COMPLETED_ACTIONS in last {self.action_window_size} calls", "WARNING")
            return (True, f"No progress: 0 completions in last {self.action_window_size} actions")

        # Pattern 3: Same TODO item FAILED repeatedly (unchanged - still valid check)
        todo_failures = Counter(
            a.get('todo_item')
            for a in last_n
            if a.get('todo_item') and not a.get('logged_to_completed')
        )
        for todo_item, count in todo_failures.items():
            if count >= 3:
                log_agent(f"[LOOP_DETECT] TODO '{todo_item[:50]}...' FAILED {count} times", "WARNING")
                return (True, f"TODO item failed {count} times without success")

        return (False, "Agent making progress")

    def track_action(
        self,
        tool_name: str,
        args_hash: str = None,
        result_category: str = None,
        todo_item: str = None,
        logged_to_completed: bool = False
    ):
        """
        Track an action for signature-based loop detection.

        The full signature (tool_name, args_hash, result_category) is used to
        distinguish between:
        - Healthy repetition: same tool, different args, SUCCESS results
        - Stuck loop: same tool, same args, NO_MATCH/EMPTY/ERROR results

        Args:
            tool_name: Name of the tool that was called
            args_hash: MD5 hash of tool arguments (use hash_tool_args())
            result_category: Result category (use categorize_tool_result())
            todo_item: Current TODO item being worked on (if known)
            logged_to_completed: Whether this action was logged to COMPLETED_ACTIONS.md
        """
        action = {
            'tool_name': tool_name,
            'args_hash': args_hash or 'unknown',
            'result_category': result_category or ToolResultCategory.SUCCESS,
            'todo_item': todo_item,
            'logged_to_completed': logged_to_completed,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        self.recent_actions.append(action)

        log_agent(f"[ACTION_TRACK] {tool_name} | args={args_hash or 'n/a'} | result={result_category or 'n/a'}")

        # Keep only last 40 actions in memory (4x window size for history)
        if len(self.recent_actions) > self.action_window_size * 4:
            self.recent_actions = self.recent_actions[-self.action_window_size * 4:]

    def get_error_count_from_state(self, state: dict) -> int:
        """Get current error count from state"""
        return state.get('error_count', 0)

    def has_max_error_attempts(self, state: dict) -> bool:
        """Check if maximum error attempts have been reached"""
        return state.get('error_count', 0) >= MAX_ERROR_ATTEMPTS

    def should_route_to_error_agent(self, state: dict) -> bool:
        """
        Determine if error agent should be invoked.

        Args:
            state: Current workflow state

        Returns:
            True if error agent should handle the error
        """
        has_error = state.get('has_build_error', False)
        error_count = state.get('error_count', 0)

        if not has_error:
            return False

        if error_count >= MAX_ERROR_ATTEMPTS:
            log_agent(f"[ERROR] Max error attempts ({MAX_ERROR_ATTEMPTS}) reached - not routing to error agent")
            return False

        return True

    def _read_error_history(self) -> str:
        """Read ERROR_HISTORY.md content"""
        if not self.project_path:
            return ""

        error_history_path = os.path.join(self.project_path, "ERROR_HISTORY.md")
        if not os.path.exists(error_history_path):
            return ""

        try:
            with open(error_history_path, 'r') as f:
                return f.read()
        except Exception:
            return ""

    def _append_to_error_history(self, content: str):
        """Append content to ERROR_HISTORY.md"""
        if not self.project_path:
            return

        error_history_path = os.path.join(self.project_path, "ERROR_HISTORY.md")
        try:
            with open(error_history_path, 'a') as f:
                f.write(content + "\n")
        except Exception as e:
            log_agent(f"[ERROR_HISTORY] Error writing to file: {e}")


class StuckDetector:
    """
    Detects when the migration process is stuck and not making progress.

    Tracks progress through TODO completion and action logging.
    """

    def __init__(self, max_loops_without_progress: int = MAX_LOOPS_WITHOUT_PROGRESS):
        """
        Args:
            max_loops_without_progress: Number of loops without progress before considered stuck
        """
        self.max_loops_without_progress = max_loops_without_progress
        self.last_todo_count = 0
        self.loops_without_progress = 0

    def check_progress(self, current_todo_count: int) -> Tuple[bool, str]:
        """
        Check if progress is being made based on TODO completion.

        Args:
            current_todo_count: Current number of completed tasks

        Returns:
            (made_progress: bool, status_message: str)
        """
        if current_todo_count > self.last_todo_count:
            self.loops_without_progress = 0
            self.last_todo_count = current_todo_count
            return (True, f"Progress made: {current_todo_count} tasks completed")
        else:
            self.loops_without_progress += 1
            if self.loops_without_progress >= self.max_loops_without_progress:
                return (False, f"No progress for {self.loops_without_progress} loops (stuck)")
            else:
                return (True, f"No change this loop ({self.loops_without_progress}/{self.max_loops_without_progress})")

    def is_stuck(self) -> bool:
        """Check if the process is considered stuck"""
        return self.loops_without_progress >= self.max_loops_without_progress

    def reset(self):
        """Reset the stuck detector state"""
        self.last_todo_count = 0
        self.loops_without_progress = 0


def initialize_error_history_file(project_path: str):
    """
    Initialize ERROR_HISTORY.md file.

    Args:
        project_path: Path to project directory
    """
    error_history_path = os.path.join(project_path, "ERROR_HISTORY.md")

    if not os.path.exists(error_history_path):
        header = """# Error History

This file tracks error resolution attempts during migration.
It helps prevent infinite retry loops and diagnose persistent issues.

"""
        try:
            with open(error_history_path, 'w') as f:
                f.write(header)
            log_agent(f"[ERROR_HISTORY] Initialized ERROR_HISTORY.md")
        except Exception as e:
            log_agent(f"[ERROR_HISTORY] Error initializing file: {e}")

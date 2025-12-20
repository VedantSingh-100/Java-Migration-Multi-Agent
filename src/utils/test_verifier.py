"""
Test Method Preservation Verifier & Build Verification Utilities

This module provides:
1. Test method preservation verification during migration
2. Bytecode version audit to catch Kotlin/AspectJ/Scala issues
3. Technology detection for proactive migration research

CRITICAL RULES ENFORCED:
- Test method names must remain identical
- Test method count must remain constant
- Test files must not be deleted
- New test methods should not be added

ADDITIONAL VERIFICATIONS (2025-12-13):
- Bytecode version consistency across all .class files
- Detection of Kotlin, AspectJ, Scala for special handling

Usage:
    verifier = TestMethodVerifier(project_path)
    verifier.capture_baseline()  # Call at migration start

    # Before each commit:
    is_valid, violations = verifier.verify_preservation()
    if not is_valid:
        # Block commit, report violations

See: docs/Recent_Issues.md for why these verifications are needed
"""

import os
import re
import json
import glob
import struct
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, asdict

from src.utils.logging_config import log_agent, log_summary


# =============================================================================
# BYTECODE VERSION AUDIT
# =============================================================================

# Java bytecode major version mapping
JAVA_BYTECODE_VERSIONS = {
    45: 1,   # Java 1.1
    46: 2,   # Java 1.2
    47: 3,   # Java 1.3
    48: 4,   # Java 1.4
    49: 5,   # Java 5
    50: 6,   # Java 6
    51: 7,   # Java 7
    52: 8,   # Java 8
    53: 9,   # Java 9
    54: 10,  # Java 10
    55: 11,  # Java 11
    56: 12,  # Java 12
    57: 13,  # Java 13
    58: 14,  # Java 14
    59: 15,  # Java 15
    60: 16,  # Java 16
    61: 17,  # Java 17
    62: 18,  # Java 18
    63: 19,  # Java 19
    64: 20,  # Java 20
    65: 21,  # Java 21
    66: 22,  # Java 22
    67: 23,  # Java 23
}


def read_class_major_version(class_file_path: str) -> Optional[int]:
    """
    Read the major version from a Java .class file.

    Class file format:
    - Bytes 0-3: Magic number (0xCAFEBABE)
    - Bytes 4-5: Minor version
    - Bytes 6-7: Major version

    Returns:
        Major version number or None if file can't be read
    """
    try:
        with open(class_file_path, 'rb') as f:
            # Read magic number (4 bytes)
            magic = f.read(4)
            if magic != b'\xCA\xFE\xBA\xBE':
                return None  # Not a valid class file

            # Read minor version (2 bytes) - skip
            f.read(2)

            # Read major version (2 bytes, big-endian)
            major_bytes = f.read(2)
            major_version = struct.unpack('>H', major_bytes)[0]

            return major_version
    except Exception:
        return None


def audit_bytecode_versions(project_path: str, expected_java_version: int = None) -> Tuple[bool, str, Dict]:
    """
    Audit all compiled .class files to verify bytecode version consistency.

    This catches issues like:
    - Kotlin compiling to Java 8 when jvmTarget not updated
    - AspectJ using old bundled compiler
    - Mixed bytecode versions in multi-module projects

    Args:
        project_path: Path to the project root
        expected_java_version: Expected Java version (default from TARGET_JAVA_VERSION env or 21)

    Returns:
        (all_match, message, details_dict)
    """
    import os

    # Use environment variable if not specified
    if expected_java_version is None:
        target_version = os.environ.get("TARGET_JAVA_VERSION", "21")
        expected_java_version = int(target_version) if target_version.isdigit() else 21

    project_dir = Path(project_path)
    if not project_dir.exists():
        return False, f"Error: Project path not found: {project_path}", {}

    expected_bytecode = expected_java_version + 44  # Java 21 = bytecode 65

    # Find all .class files in target directories
    class_patterns = [
        '**/target/classes/**/*.class',
        '**/target/test-classes/**/*.class',
        '**/build/classes/**/*.class',  # Gradle projects
    ]

    class_files = []
    for pattern in class_patterns:
        class_files.extend(project_dir.glob(pattern))

    if not class_files:
        log_agent(f"[BYTECODE_AUDIT] No .class files found in {project_path}")
        return True, "No compiled .class files found. Run 'mvn compile' first.", {}

    # Audit each class file
    version_counts: Dict[int, int] = {}
    mismatches: List[Dict] = []

    for class_file in class_files:
        major_version = read_class_major_version(str(class_file))
        if major_version is None:
            continue

        version_counts[major_version] = version_counts.get(major_version, 0) + 1

        if major_version != expected_bytecode:
            java_version = JAVA_BYTECODE_VERSIONS.get(major_version, f"unknown({major_version})")
            rel_path = str(class_file.relative_to(project_dir))
            mismatches.append({
                "file": rel_path,
                "bytecode_version": major_version,
                "java_version": java_version,
            })

    total_files = sum(version_counts.values())
    all_match = len(mismatches) == 0

    # Build result details
    details = {
        "total_files": total_files,
        "expected_java": expected_java_version,
        "expected_bytecode": expected_bytecode,
        "version_distribution": {
            JAVA_BYTECODE_VERSIONS.get(bv, f"unknown({bv})"): cnt
            for bv, cnt in version_counts.items()
        },
        "mismatches": mismatches[:20],  # Limit to first 20
        "mismatch_count": len(mismatches),
    }

    if all_match:
        msg = f"✅ BYTECODE AUDIT PASSED: All {total_files} files compiled for Java {expected_java_version}"
        log_summary(f"BYTECODE_AUDIT: ✅ All {total_files} files match Java {expected_java_version}")
    else:
        # Group by Java version for cleaner output
        by_version: Dict[int, int] = {}
        for m in mismatches:
            ver = m['java_version']
            by_version[ver] = by_version.get(ver, 0) + 1

        msg = f"❌ BYTECODE AUDIT FAILED: {len(mismatches)} files have wrong bytecode version\n"
        msg += f"Expected: Java {expected_java_version} (bytecode {expected_bytecode})\n"
        msg += "Found:\n"
        for java_ver, cnt in sorted(by_version.items()):
            msg += f"  - Java {java_ver}: {cnt} files\n"
        msg += "\nCommon causes:\n"
        msg += "  - Kotlin: jvmTarget not set to " + str(expected_java_version) + "\n"
        msg += "  - AspectJ: bundled compiler version doesn't support Java " + str(expected_java_version) + "\n"
        msg += "  - Scala: scalac target version mismatch\n"
        msg += "\nUse web_search to find fix for your specific technology."

        log_summary(f"BYTECODE_AUDIT: ❌ {len(mismatches)} files have wrong version")

    return all_match, msg, details


# =============================================================================
# TECHNOLOGY DETECTION
# =============================================================================

# Technology detection signals
TECHNOLOGY_SIGNALS = {
    "kotlin": {
        "file_patterns": ["**/*.kt", "**/*.kts"],
        "pom_patterns": ["kotlin-maven-plugin", "kotlin-stdlib", "org.jetbrains.kotlin"],
        "migration_concerns": [
            "jvmTarget must be explicitly set (doesn't inherit java.version)",
            "kotlin.version may need updating for Java 21 compatibility",
        ],
        "research_queries": [
            "Kotlin jvmTarget Java 21 Maven configuration",
            "kotlin-maven-plugin Java 21 settings",
        ]
    },
    "aspectj": {
        "file_patterns": ["**/*.aj"],
        "pom_patterns": ["aspectj-maven-plugin", "jcabi-maven-plugin", "org.aspectj"],
        "migration_concerns": [
            "AspectJ compiler version must support Java 21 bytecode",
            "jcabi-maven-plugin bundles old AspectJ 1.9.1 internally",
            "Aspect weaving can fail silently with version mismatches",
        ],
        "research_queries": [
            "AspectJ Java 21 compatibility",
            "jcabi-maven-plugin AspectJ version bundled",
        ]
    },
    "scala": {
        "file_patterns": ["**/*.scala"],
        "pom_patterns": ["scala-maven-plugin", "org.scala-lang"],
        "migration_concerns": [
            "Scala version must be compatible with Java 21",
            "Scala 3.x recommended for Java 21",
        ],
        "research_queries": [
            "Scala Java 21 compatibility",
            "scala-maven-plugin Java 21",
        ]
    },
    "groovy": {
        "file_patterns": ["**/*.groovy"],
        "pom_patterns": ["groovy-maven-plugin", "gmavenplus-plugin", "org.codehaus.groovy"],
        "migration_concerns": [
            "Groovy version must support Java 21",
            "targetBytecode configuration required",
        ],
        "research_queries": [
            "Groovy Java 21 compatibility",
            "GMavenPlus Java 21 configuration",
        ]
    },
    "lombok": {
        "file_patterns": [],
        "pom_patterns": ["lombok", "org.projectlombok"],
        "migration_concerns": [
            "Lombok version must support Java 21",
            "Annotation processing may need updates",
        ],
        "research_queries": [
            "Lombok Java 21 compatibility",
        ]
    },
}


def detect_project_technologies(project_path: str) -> Tuple[List[str], str, Dict]:
    """
    Detect non-standard JVM technologies that require special migration handling.

    Scans for Kotlin, AspectJ, Scala, Groovy, Lombok, etc. and provides:
    - Detection signals (what was found)
    - Known migration concerns
    - Suggested research queries for web search

    Args:
        project_path: Path to the project root

    Returns:
        (detected_tech_names, message, details_dict)
    """
    project_dir = Path(project_path)
    if not project_dir.exists():
        return [], f"Error: Project path not found: {project_path}", {}

    # Read all pom.xml content
    pom_content = ""
    pom_files = list(project_dir.glob("**/pom.xml"))
    for pom_file in pom_files:
        try:
            pom_content += pom_file.read_text(encoding='utf-8')
        except Exception:
            pass

    detected = []
    details = {"technologies": {}}

    for tech_name, signals in TECHNOLOGY_SIGNALS.items():
        found_signals = []

        # Check file patterns
        for pattern in signals.get("file_patterns", []):
            matches = list(project_dir.glob(pattern))
            if matches:
                found_signals.append(f"Found {len(matches)} {pattern.split('*')[-1]} files")

        # Check pom.xml patterns
        for pom_pattern in signals.get("pom_patterns", []):
            if pom_pattern.lower() in pom_content.lower():
                found_signals.append(f"Found '{pom_pattern}' in pom.xml")

        if found_signals:
            detected.append(tech_name)
            details["technologies"][tech_name] = {
                "signals": found_signals,
                "concerns": signals.get("migration_concerns", []),
                "research_queries": signals.get("research_queries", []),
            }

    # Build message
    if not detected:
        msg = "✅ No special technologies detected. Standard Java/Maven project."
        log_summary(f"TECH_DETECT: No special technologies detected")
    else:
        msg = f"⚠️ DETECTED {len(detected)} TECHNOLOGY/TECHNOLOGIES REQUIRING ATTENTION:\n"
        all_queries = []

        for tech_name in detected:
            tech_info = details["technologies"][tech_name]
            msg += f"\n📦 {tech_name.upper()}\n"
            msg += f"   Signals: {', '.join(tech_info['signals'])}\n"
            msg += f"   Concerns:\n"
            for concern in tech_info['concerns']:
                msg += f"     ⚠️ {concern}\n"
            all_queries.extend(tech_info['research_queries'])

        msg += f"\n🔍 RECOMMENDED RESEARCH (use web_search):\n"
        for i, query in enumerate(all_queries[:5], 1):
            msg += f"   {i}. {query}\n"

        log_summary(f"TECH_DETECT: Found {len(detected)} technologies: {detected}")

    details["detected"] = detected
    details["pom_count"] = len(pom_files)

    return detected, msg, details


@dataclass
class TestMethod:
    """Represents a test method signature."""
    name: str
    file_path: str
    line_number: int
    annotations: List[str]  # @Test, @ParameterizedTest, etc.

    def signature(self) -> str:
        """Return unique signature for comparison."""
        return f"{self.file_path}::{self.name}"


@dataclass
class TestFile:
    """Represents a test file with its methods."""
    file_path: str
    relative_path: str
    methods: List[TestMethod]
    method_count: int
    content_hash: str  # Hash of file content for change detection

    def method_names(self) -> Set[str]:
        """Return set of method names in this file."""
        return {m.name for m in self.methods}


@dataclass
class VerificationResult:
    """Result of test preservation verification."""
    is_valid: bool
    violations: List[str]
    renamed_methods: List[Tuple[str, str, str]]  # (file, old_name, new_name)
    deleted_methods: List[Tuple[str, str]]  # (file, method_name)
    added_methods: List[Tuple[str, str]]  # (file, method_name)
    deleted_files: List[str]
    added_files: List[str]
    baseline_method_count: int
    current_method_count: int


class TestMethodVerifier:
    """
    Verifies test method preservation during migration.

    Captures baseline test signatures at migration start and verifies
    they remain unchanged before commits.
    """

    # Patterns to identify test files
    TEST_FILE_PATTERNS = [
        r'.*Test\.java$',
        r'.*Tests\.java$',
        r'.*TestCase\.java$',
        r'.*IT\.java$',  # Integration tests
        r'.*ITCase\.java$',
    ]

    # Patterns to identify test methods (JUnit 4 and 5)
    TEST_METHOD_PATTERNS = [
        r'@Test\s*(?:\([^)]*\))?\s*(?:public\s+)?void\s+(\w+)\s*\(',
        r'@ParameterizedTest\s*(?:\([^)]*\))?\s*(?:public\s+)?void\s+(\w+)\s*\(',
        r'@RepeatedTest\s*(?:\([^)]*\))?\s*(?:public\s+)?void\s+(\w+)\s*\(',
        r'@TestFactory\s*(?:\([^)]*\))?\s*(?:public\s+)?\w+\s+(\w+)\s*\(',
    ]

    # Annotation patterns to capture
    ANNOTATION_PATTERNS = [
        r'@Test',
        r'@ParameterizedTest',
        r'@RepeatedTest',
        r'@TestFactory',
        r'@Disabled',
        r'@Ignore',
        r'@BeforeEach',
        r'@AfterEach',
        r'@BeforeAll',
        r'@AfterAll',
        r'@Before',
        r'@After',
    ]

    def __init__(self, project_path: str):
        """
        Initialize the verifier.

        Args:
            project_path: Path to the project root
        """
        self.project_path = project_path
        self.baseline: Optional[Dict[str, TestFile]] = None
        self.baseline_captured_at: Optional[str] = None
        self.baseline_file = os.path.join(project_path, ".test_baseline.json")

    def capture_baseline(self) -> Dict[str, TestFile]:
        """
        Capture baseline test signatures at migration start.

        Should be called ONCE at the beginning of migration,
        before any changes are made.

        Returns:
            Dictionary mapping file paths to TestFile objects
        """
        log_agent("[TEST_VERIFIER] Capturing baseline test signatures...")

        self.baseline = {}
        test_files = self._find_test_files()

        total_methods = 0
        for file_path in test_files:
            test_file = self._analyze_test_file(file_path)
            if test_file:
                self.baseline[test_file.relative_path] = test_file
                total_methods += test_file.method_count

        self.baseline_captured_at = datetime.now().isoformat()

        # Save baseline to file for persistence across restarts
        self._save_baseline()

        log_agent(f"[TEST_VERIFIER] Baseline captured: {len(self.baseline)} test files, {total_methods} test methods")
        log_summary(f"TEST BASELINE: {len(self.baseline)} files, {total_methods} methods")

        return self.baseline

    def load_baseline(self) -> bool:
        """
        Load baseline from file if it exists.

        Returns:
            True if baseline was loaded, False otherwise
        """
        if not os.path.exists(self.baseline_file):
            return False

        try:
            with open(self.baseline_file, 'r') as f:
                data = json.load(f)

            self.baseline_captured_at = data.get('captured_at')
            self.baseline = {}

            for rel_path, file_data in data.get('files', {}).items():
                methods = [
                    TestMethod(
                        name=m['name'],
                        file_path=m['file_path'],
                        line_number=m['line_number'],
                        annotations=m['annotations']
                    )
                    for m in file_data['methods']
                ]
                self.baseline[rel_path] = TestFile(
                    file_path=file_data['file_path'],
                    relative_path=rel_path,
                    methods=methods,
                    method_count=file_data['method_count'],
                    content_hash=file_data['content_hash']
                )

            log_agent(f"[TEST_VERIFIER] Loaded baseline from file: {len(self.baseline)} test files")
            return True

        except Exception as e:
            log_agent(f"[TEST_VERIFIER] Failed to load baseline: {e}", "WARNING")
            return False

    def verify_preservation(self) -> VerificationResult:
        """
        Verify that test methods have been preserved since baseline.

        Should be called before commits to catch violations early.

        Returns:
            VerificationResult with details of any violations
        """
        if self.baseline is None:
            # Try to load from file
            if not self.load_baseline():
                log_agent("[TEST_VERIFIER] No baseline captured - skipping verification", "WARNING")
                return VerificationResult(
                    is_valid=True,
                    violations=["No baseline captured - verification skipped"],
                    renamed_methods=[],
                    deleted_methods=[],
                    added_methods=[],
                    deleted_files=[],
                    added_files=[],
                    baseline_method_count=0,
                    current_method_count=0
                )

        log_agent("[TEST_VERIFIER] Verifying test preservation...")

        # Get current state
        current_files = {}
        test_files = self._find_test_files()
        for file_path in test_files:
            test_file = self._analyze_test_file(file_path)
            if test_file:
                current_files[test_file.relative_path] = test_file

        violations = []
        renamed_methods = []
        deleted_methods = []
        added_methods = []
        deleted_files = []
        added_files = []

        baseline_method_count = sum(f.method_count for f in self.baseline.values())
        current_method_count = sum(f.method_count for f in current_files.values())

        # Check for deleted files
        for rel_path in self.baseline:
            if rel_path not in current_files:
                deleted_files.append(rel_path)
                violations.append(f"❌ TEST FILE DELETED: {rel_path}")
                # All methods in this file are effectively deleted
                for method in self.baseline[rel_path].methods:
                    deleted_methods.append((rel_path, method.name))

        # Check for added files
        for rel_path in current_files:
            if rel_path not in self.baseline:
                added_files.append(rel_path)
                violations.append(f"⚠️ TEST FILE ADDED: {rel_path}")
                for method in current_files[rel_path].methods:
                    added_methods.append((rel_path, method.name))

        # Check each baseline file for method changes
        for rel_path, baseline_file in self.baseline.items():
            if rel_path not in current_files:
                continue  # Already handled as deleted file

            current_file = current_files[rel_path]
            baseline_names = baseline_file.method_names()
            current_names = current_file.method_names()

            # Methods that were removed
            removed = baseline_names - current_names
            for method_name in removed:
                deleted_methods.append((rel_path, method_name))
                violations.append(f"❌ TEST METHOD DELETED: {rel_path}::{method_name}")

            # Methods that were added
            added = current_names - baseline_names
            for method_name in added:
                added_methods.append((rel_path, method_name))

            # Check for potential renames (removed + added in same file)
            if removed and added:
                # Heuristic: if same number removed and added, likely a rename
                if len(removed) == len(added):
                    for old_name, new_name in zip(sorted(removed), sorted(added)):
                        renamed_methods.append((rel_path, old_name, new_name))
                        violations.append(f"❌ TEST METHOD RENAMED: {rel_path}::{old_name} → {new_name}")
                else:
                    # Different counts - just report additions
                    for method_name in added:
                        violations.append(f"⚠️ TEST METHOD ADDED: {rel_path}::{method_name}")
            elif added:
                for method_name in added:
                    violations.append(f"⚠️ TEST METHOD ADDED: {rel_path}::{method_name}")

        # Check total method count
        if current_method_count != baseline_method_count:
            violations.append(
                f"❌ TEST COUNT CHANGED: {baseline_method_count} → {current_method_count} "
                f"(diff: {current_method_count - baseline_method_count:+d})"
            )

        is_valid = len(deleted_methods) == 0 and len(renamed_methods) == 0 and len(deleted_files) == 0

        result = VerificationResult(
            is_valid=is_valid,
            violations=violations,
            renamed_methods=renamed_methods,
            deleted_methods=deleted_methods,
            added_methods=added_methods,
            deleted_files=deleted_files,
            added_files=added_files,
            baseline_method_count=baseline_method_count,
            current_method_count=current_method_count
        )

        if is_valid:
            log_agent(f"[TEST_VERIFIER] ✅ Test preservation verified: {current_method_count} methods intact")
        else:
            log_agent(f"[TEST_VERIFIER] ❌ Test preservation VIOLATED: {len(violations)} issues found", "ERROR")
            for v in violations[:5]:  # Log first 5
                log_agent(f"[TEST_VERIFIER]   {v}", "ERROR")
            if len(violations) > 5:
                log_agent(f"[TEST_VERIFIER]   ... and {len(violations) - 5} more", "ERROR")
            log_summary(f"TEST VIOLATION: {len(violations)} issues - {violations[0] if violations else 'unknown'}")

        return result

    def get_violation_message(self, result: VerificationResult) -> str:
        """
        Generate a human-readable violation message for the agent.

        Args:
            result: VerificationResult from verify_preservation()

        Returns:
            Formatted message explaining the violations
        """
        if result.is_valid:
            return ""

        lines = [
            "=" * 70,
            "🚨 TEST PRESERVATION VIOLATION DETECTED 🚨",
            "=" * 70,
            "",
            "Your changes violate the test preservation rules.",
            "Test methods must remain IDENTICAL to baseline.",
            "",
        ]

        if result.renamed_methods:
            lines.append("RENAMED METHODS (FORBIDDEN):")
            for file, old, new in result.renamed_methods:
                lines.append(f"  ❌ {file}: {old} → {new}")
            lines.append("")

        if result.deleted_methods:
            lines.append("DELETED METHODS (FORBIDDEN):")
            for file, method in result.deleted_methods[:10]:
                lines.append(f"  ❌ {file}::{method}")
            if len(result.deleted_methods) > 10:
                lines.append(f"  ... and {len(result.deleted_methods) - 10} more")
            lines.append("")

        if result.deleted_files:
            lines.append("DELETED TEST FILES (FORBIDDEN):")
            for file in result.deleted_files:
                lines.append(f"  ❌ {file}")
            lines.append("")

        # Collect affected files for revert commands
        affected_files = set()
        for file, _, _ in result.renamed_methods:
            affected_files.add(file)
        for file, _ in result.deleted_methods:
            affected_files.add(file)
        for file in result.deleted_files:
            affected_files.add(file)

        lines.extend([
            "=" * 70,
            "🔧 HOW TO FIX (FOLLOW THESE STEPS):",
            "=" * 70,
            "",
            "STEP 1: REVERT the test file changes using these EXACT commands:",
            "",
        ])

        # Add specific git checkout commands
        for file in sorted(affected_files):
            lines.append(f"    git checkout HEAD -- {file}")

        lines.extend([
            "",
            "STEP 2: Understand WHY the test was failing:",
            "  - If it's a compilation error: Fix the APPLICATION code, not the test",
            "  - If test references old API: Update imports/method calls in APPLICATION code",
            "  - If test expects old behavior: The APPLICATION code change broke it - fix app code",
            "",
            "STEP 3: If a test TRULY cannot work after migration (rare):",
            "  - Add @Disabled(\"Reason: <explain why>\") annotation BEFORE @Test",
            "  - Example:",
            "      @Disabled(\"Incompatible with Jakarta namespace - requires manual review\")",
            "      @Test",
            "      public void originalTestName() { ... }",
            "",
            "⚠️  NEVER rename, delete, or rewrite test method signatures!",
            "",
            f"Baseline: {result.baseline_method_count} methods",
            f"Current:  {result.current_method_count} methods",
            "=" * 70,
        ])

        return "\n".join(lines)

    def _find_test_files(self) -> List[str]:
        """Find all test files in the project."""
        test_files = []

        # Common test directories
        test_dirs = [
            os.path.join(self.project_path, "src", "test", "java"),
            os.path.join(self.project_path, "src", "test"),
            os.path.join(self.project_path, "test"),
        ]

        for test_dir in test_dirs:
            if not os.path.exists(test_dir):
                continue

            for root, dirs, files in os.walk(test_dir):
                for file in files:
                    if file.endswith('.java'):
                        file_path = os.path.join(root, file)
                        # Check if it matches test file patterns
                        for pattern in self.TEST_FILE_PATTERNS:
                            if re.match(pattern, file):
                                test_files.append(file_path)
                                break

        return test_files

    def _analyze_test_file(self, file_path: str) -> Optional[TestFile]:
        """Analyze a test file and extract method signatures."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            log_agent(f"[TEST_VERIFIER] Error reading {file_path}: {e}", "WARNING")
            return None

        methods = []
        lines = content.split('\n')

        # Pattern to find method declarations (without requiring @Test on same line)
        method_pattern = r'(?:public\s+)?void\s+(\w+)\s*\('

        for i, line in enumerate(lines):
            # Skip non-method lines quickly
            if 'void' not in line:
                continue

            match = re.search(method_pattern, line)
            if not match:
                continue

            method_name = match.group(1)

            # Check preceding lines (up to 10) for test annotations
            # This handles @Test, @Disabled, @ParameterizedTest etc on separate lines
            annotations = []
            has_test_annotation = False

            for j in range(max(0, i - 10), i + 1):
                line_j = lines[j]
                for ann_pattern in self.ANNOTATION_PATTERNS:
                    if re.search(ann_pattern, line_j):
                        ann_name = ann_pattern.replace('\\', '')
                        annotations.append(ann_name)
                        # Check if it's a test-indicating or lifecycle annotation
                        if ann_name in ['@Test', '@ParameterizedTest', '@RepeatedTest', '@TestFactory',
                                        '@Before', '@After', '@BeforeEach', '@AfterEach',
                                        '@BeforeAll', '@AfterAll']:
                            has_test_annotation = True

            # Only include methods that have a test annotation
            if has_test_annotation:
                methods.append(TestMethod(
                    name=method_name,
                    file_path=file_path,
                    line_number=i + 1,
                    annotations=list(set(annotations))
                ))

        if not methods:
            return None

        relative_path = os.path.relpath(file_path, self.project_path)
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]

        return TestFile(
            file_path=file_path,
            relative_path=relative_path,
            methods=methods,
            method_count=len(methods),
            content_hash=content_hash
        )

    def _save_baseline(self):
        """Save baseline to JSON file for persistence."""
        if self.baseline is None:
            return

        data = {
            'captured_at': self.baseline_captured_at,
            'project_path': self.project_path,
            'files': {}
        }

        for rel_path, test_file in self.baseline.items():
            data['files'][rel_path] = {
                'file_path': test_file.file_path,
                'method_count': test_file.method_count,
                'content_hash': test_file.content_hash,
                'methods': [
                    {
                        'name': m.name,
                        'file_path': m.file_path,
                        'line_number': m.line_number,
                        'annotations': m.annotations
                    }
                    for m in test_file.methods
                ]
            }

        try:
            with open(self.baseline_file, 'w') as f:
                json.dump(data, f, indent=2)
            log_agent(f"[TEST_VERIFIER] Baseline saved to {self.baseline_file}")
        except Exception as e:
            log_agent(f"[TEST_VERIFIER] Failed to save baseline: {e}", "WARNING")


def verify_test_preservation_before_commit(project_path: str) -> Tuple[bool, str]:
    """
    Convenience function to verify test preservation before a commit.

    Args:
        project_path: Path to the project

    Returns:
        (is_valid, message) - True if tests preserved, message with details
    """
    verifier = TestMethodVerifier(project_path)

    if not verifier.load_baseline():
        # No baseline - can't verify
        return True, "No test baseline found - verification skipped"

    result = verifier.verify_preservation()

    if result.is_valid:
        return True, f"✅ Test preservation verified: {result.current_method_count} methods intact"
    else:
        return False, verifier.get_violation_message(result)


def verify_final_test_invariance(project_path: str, base_commit: str) -> Tuple[bool, str]:
    """
    Final verification using MigrationBench's exact evaluation logic.
    Called before marking migration as complete.

    This uses the SAME function that check_build_test_comprehensive.py uses,
    guaranteeing alignment between prevention and evaluation.

    Args:
        project_path: Path to the repository
        base_commit: Original commit hash before migration started

    Returns:
        (is_valid, message) - True if test methods match baseline
    """
    try:
        from eval.lang.java.eval import parse_repo

        log_agent(f"[TEST_VERIFIER] Running final test invariance check against base_commit: {base_commit}")

        # Run the exact same check that evaluation uses
        all_same, num_files, tests_same = parse_repo.same_repo_test_files(
            project_path,
            lhs_branch=base_commit
        )

        if tests_same:
            log_agent(f"[TEST_VERIFIER] ✅ Final test invariance verified: {num_files} test files match baseline")
            log_summary(f"FINAL_TEST_CHECK: PASS - {num_files} test files match baseline")
            return True, f"✅ Final test invariance verified: {num_files} test files match baseline"
        else:
            log_agent(f"[TEST_VERIFIER] ❌ Final test invariance FAILED: test methods changed", "ERROR")
            log_summary(f"FINAL_TEST_CHECK: FAIL - test methods have changed compared to {base_commit}")
            return False, (
                "=" * 70 + "\n"
                "❌ FINAL TEST INVARIANCE CHECK FAILED\n"
                "=" * 70 + "\n\n"
                "Test methods have changed compared to base_commit.\n"
                "This migration will FAIL evaluation.\n\n"
                "The evaluation script uses parse_repo.same_repo_test_files()\n"
                f"to compare current state against base_commit: {base_commit}\n\n"
                "You must revert test file changes before completing.\n"
                "Test method NAMES must remain identical to baseline.\n"
                "Only update test IMPLEMENTATIONS, not method signatures.\n"
                "=" * 70
            )

    except ImportError as e:
        log_agent(f"[TEST_VERIFIER] MigrationBench eval module not available: {e}", "WARNING")
        log_summary(f"FINAL_TEST_CHECK: FALLBACK - MigrationBench not available, using local verifier")
        # Fall back to local verifier if MigrationBench not installed
        return verify_test_preservation_before_commit(project_path)

    except Exception as e:
        log_agent(f"[TEST_VERIFIER] Final verification error: {e}", "ERROR")
        log_summary(f"FINAL_TEST_CHECK: ERROR - {e}")
        return False, f"Final test verification failed: {e}"

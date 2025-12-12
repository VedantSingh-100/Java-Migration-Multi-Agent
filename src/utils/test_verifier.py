"""
Test Method Preservation Verifier

This module provides deterministic verification that test methods are preserved
during migration. It captures baseline test signatures and verifies they remain
unchanged throughout the migration process.

CRITICAL RULES ENFORCED:
- Test method names must remain identical
- Test method count must remain constant
- Test files must not be deleted
- New test methods should not be added

Usage:
    verifier = TestMethodVerifier(project_path)
    verifier.capture_baseline()  # Call at migration start

    # Before each commit:
    is_valid, violations = verifier.verify_preservation()
    if not is_valid:
        # Block commit, report violations
"""

import os
import re
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, asdict

from src.utils.logging_config import log_agent, log_summary


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

        lines.extend([
            "HOW TO FIX:",
            "1. REVERT your test file changes",
            "2. Fix the APPLICATION code instead to make tests pass",
            "3. If a test truly cannot work, use @Disabled annotation",
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

        for i, line in enumerate(lines):
            for pattern in self.TEST_METHOD_PATTERNS:
                match = re.search(pattern, line)
                if match:
                    method_name = match.group(1)

                    # Extract annotations from preceding lines
                    annotations = []
                    for j in range(max(0, i - 5), i + 1):
                        for ann_pattern in self.ANNOTATION_PATTERNS:
                            if re.search(ann_pattern, lines[j]):
                                annotations.append(ann_pattern.replace('\\', ''))

                    methods.append(TestMethod(
                        name=method_name,
                        file_path=file_path,
                        line_number=i + 1,
                        annotations=list(set(annotations))
                    ))
                    break

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

#!/usr/bin/env python3
"""
Proper evaluation of migrated repositories with Maven in PATH.
This script validates migrated repos against MigrationBench-like criteria.
"""

import os
import subprocess
import json
from pathlib import Path
from datetime import datetime

# Configuration
MAVEN_HOME = "/home/vhsingh/apache-maven-3.9.11"
JAVA_HOME = "/usr/lib/jvm/java-21-openjdk"
REPOS_DIR = "/home/vhsingh/Java_Migration/repositories_sonnet35"

# MigrationBench dependency version requirements (subset for common deps)
REQUIRED_VERSIONS = {
    "org.springframework.boot:spring-boot-starter-parent": "3.0.0",  # Major version 3
    "org.springframework.boot:spring-boot-starter": "3.0.0",
    "org.springframework.boot:spring-boot-starter-web": "3.0.0",
    "org.springframework.boot:spring-boot-starter-test": "3.0.0",
    "org.junit.jupiter:junit-jupiter": "5.0.0",
    "org.junit.jupiter:junit-jupiter-api": "5.0.0",
}

def run_command(cmd, cwd, timeout=300):
    """Run a command with Maven in PATH."""
    env = os.environ.copy()
    env["PATH"] = f"{MAVEN_HOME}/bin:" + env.get("PATH", "")
    env["JAVA_HOME"] = JAVA_HOME
    env["MAVEN_HOME"] = MAVEN_HOME

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    except Exception as e:
        return -2, "", str(e)

def check_git_branch(repo_path):
    """Check if repo is on migration-base branch."""
    code, stdout, _ = run_command("git branch --show-current", repo_path, timeout=10)
    if code == 0:
        return stdout.strip() == "migration-base"
    return False

def check_build_success(repo_path):
    """Run mvn clean verify and check if it succeeds."""
    code, stdout, stderr = run_command("mvn clean verify -B -DskipTests=false", repo_path, timeout=600)
    success = code == 0 and "BUILD SUCCESS" in stdout
    return success, stdout + stderr

def check_java_version_in_pom(repo_path):
    """Check if pom.xml specifies Java 17 or 21."""
    pom_path = os.path.join(repo_path, "pom.xml")
    if not os.path.exists(pom_path):
        return False, "No pom.xml"

    with open(pom_path, 'r') as f:
        content = f.read()

    # Check for Java 17 or 21
    if "<java.version>17</java.version>" in content or "<java.version>21</java.version>" in content:
        return True, "Java 17/21 found"
    if "<maven.compiler.source>17</maven.compiler.source>" in content or "<maven.compiler.source>21</maven.compiler.source>" in content:
        return True, "Java 17/21 found"
    if ">17<" in content or ">21<" in content:
        return True, "Possibly Java 17/21"
    return False, "Java version not 17/21"

def check_spring_boot_version(repo_path):
    """Check if Spring Boot version is 3.x."""
    pom_path = os.path.join(repo_path, "pom.xml")
    if not os.path.exists(pom_path):
        return False, "No pom.xml"

    with open(pom_path, 'r') as f:
        content = f.read()

    # Check for Spring Boot 3.x
    import re
    match = re.search(r'spring-boot-starter-parent.*?<version>(\d+\.\d+\.\d+)', content, re.DOTALL)
    if match:
        version = match.group(1)
        major = int(version.split('.')[0])
        return major >= 3, f"Spring Boot {version}"

    # Check in properties
    match = re.search(r'<spring-boot\.version>(\d+\.\d+\.\d+)', content)
    if match:
        version = match.group(1)
        major = int(version.split('.')[0])
        return major >= 3, f"Spring Boot {version}"

    return None, "Spring Boot version not found"

def count_tests(output):
    """Count tests from Maven output."""
    import re
    match = re.search(r'Tests run:\s*(\d+)', output)
    if match:
        return int(match.group(1))
    return 0

def evaluate_repo(repo_path, repo_name):
    """Evaluate a single repository."""
    result = {
        "repo": repo_name,
        "path": repo_path,
        "on_migration_branch": False,
        "build_success": False,
        "java_version_ok": False,
        "spring_boot_3x": None,
        "tests_run": 0,
        "details": "",
        "errors": []
    }

    # Check branch
    result["on_migration_branch"] = check_git_branch(repo_path)
    if not result["on_migration_branch"]:
        result["errors"].append("Not on migration-base branch")
        return result

    # Check Java version in pom
    java_ok, java_detail = check_java_version_in_pom(repo_path)
    result["java_version_ok"] = java_ok
    result["details"] += f"Java: {java_detail}. "

    # Check Spring Boot version
    sb_ok, sb_detail = check_spring_boot_version(repo_path)
    result["spring_boot_3x"] = sb_ok
    result["details"] += f"SpringBoot: {sb_detail}. "

    # Build and test
    print(f"  Building {repo_name}...")
    build_ok, build_output = check_build_success(repo_path)
    result["build_success"] = build_ok

    if build_ok:
        result["tests_run"] = count_tests(build_output)
        result["details"] += f"Tests: {result['tests_run']}. "
    else:
        # Extract error
        if "TIMEOUT" in build_output:
            result["errors"].append("Build timeout")
        elif "BUILD FAILURE" in build_output:
            result["errors"].append("Build failure")
        else:
            result["errors"].append("Unknown build error")

    return result

def main():
    """Run evaluation on all repos."""
    print("=" * 60)
    print("PROPER EVALUATION WITH MAVEN IN PATH")
    print("=" * 60)
    print(f"Maven: {MAVEN_HOME}")
    print(f"Java: {JAVA_HOME}")
    print(f"Repos: {REPOS_DIR}")
    print("=" * 60)

    results = []
    repos = sorted([d for d in os.listdir(REPOS_DIR) if os.path.isdir(os.path.join(REPOS_DIR, d))])

    print(f"\nFound {len(repos)} repositories to evaluate\n")

    for i, repo_name in enumerate(repos, 1):
        repo_path = os.path.join(REPOS_DIR, repo_name)
        print(f"[{i}/{len(repos)}] Evaluating: {repo_name}")

        result = evaluate_repo(repo_path, repo_name)
        results.append(result)

        status = "✓ PASS" if result["build_success"] else "✗ FAIL"
        print(f"  {status}: {result['details']}")
        if result["errors"]:
            print(f"  Errors: {result['errors']}")
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    on_branch = sum(1 for r in results if r["on_migration_branch"])
    build_pass = sum(1 for r in results if r["build_success"])
    java_ok = sum(1 for r in results if r["java_version_ok"])
    spring_ok = sum(1 for r in results if r["spring_boot_3x"])

    total = len(results)

    print(f"Total repos:           {total}")
    print(f"On migration branch:   {on_branch}/{total} ({100*on_branch/total:.1f}%)")
    print(f"Build success:         {build_pass}/{total} ({100*build_pass/total:.1f}%)")
    print(f"Java 17/21 in pom:     {java_ok}/{total} ({100*java_ok/total:.1f}%)")
    print(f"Spring Boot 3.x:       {spring_ok}/{total} ({100*spring_ok/total:.1f}%)" if spring_ok else "Spring Boot 3.x: N/A")

    # MigrationBench-style metrics
    print("\n" + "=" * 60)
    print("MIGRATIONBENCH-STYLE METRICS")
    print("=" * 60)

    # Minimal migration: build success
    minimal = build_pass
    print(f"Minimal Migration (build success): {minimal}/{total} ({100*minimal/total:.1f}%)")

    # Maximal migration: build + java version + spring boot 3.x (approximation)
    maximal = sum(1 for r in results if r["build_success"] and r["java_version_ok"] and r["spring_boot_3x"])
    print(f"Maximal Migration (approx):        {maximal}/{total} ({100*maximal/total:.1f}%)")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": total,
            "on_migration_branch": on_branch,
            "build_success": build_pass,
            "java_version_ok": java_ok,
            "spring_boot_3x": spring_ok,
            "minimal_migration": minimal,
            "maximal_migration": maximal,
        },
        "results": results
    }

    output_file = f"proper_eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()

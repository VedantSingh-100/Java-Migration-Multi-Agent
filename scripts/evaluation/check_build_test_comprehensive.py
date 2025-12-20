#!/usr/bin/env python3
"""
Comprehensive Build & Test Evaluation Script
Uses the migration-bench evaluation framework adapted for local CSV-based workflow
"""

import os
import sys
import json
import logging
import re
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional
from colorama import Fore, Style, init
from tqdm import tqdm

# Add eval directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import from correct submodule paths
from eval.eval import final_eval
from eval.common import maven_utils, hash_utils, git_repo, eval_utils, utils
from eval.lang.java.eval import parse_repo

# Initialize colorama
init()

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise, set to INFO for debugging
    format=utils.LOGGING_FORMAT
)


class ComprehensiveEvaluator:
    """Comprehensive evaluation for migrated Java repositories."""

    def __init__(self, csv_path: str = "migration/selected.csv",
                 repos_dir: str = "migration/repositories",
                 results_dir: str = "evaluation_results"):
        """Initialize evaluator with paths."""
        self.csv_path = csv_path
        self.repos_dir = repos_dir
        self.results_dir = results_dir

        # Create results directory
        os.makedirs(results_dir, exist_ok=True)

        # Load CSV
        self.df = pd.read_csv(csv_path)

        # Results storage
        self.results = []
        self.summary_stats = {
            'total': 0,
            'build_success': 0,
            'java_version_correct': 0,
            'test_count_maintained': 0,
            'test_methods_invariant': 0,
            'maximal_migration': 0,
            'fully_passing': 0,
        }

        # Migration metrics stats
        self.metrics_stats = {
            'total_llm_calls': 0,
            'total_duration_seconds': 0,
            'total_cost_usd': 0.0,
            'total_tokens': 0,
        }

    def get_repo_path(self, repo_name: str) -> str:
        """Convert repo name to local path."""
        # Repo format: owner/name -> owner_name (single underscore)
        safe_name = repo_name.replace("/", "_")
        return os.path.join(self.repos_dir, safe_name)

    def get_log_dir(self, repo_name: str) -> str:
        """Get log directory for a repository."""
        # Logs use double underscore
        safe_name = repo_name.replace("/", "__")
        return os.path.join("logs", safe_name)

    def extract_migration_metrics(self, repo_name: str) -> Optional[Dict]:
        """Extract migration metrics from log files."""
        log_dir = self.get_log_dir(repo_name)

        if not os.path.exists(log_dir):
            return None

        # Find the most recent summary log
        summary_logs = [f for f in os.listdir(log_dir) if f.startswith('summary_') and f.endswith('.log')]
        if not summary_logs:
            return None

        # Get most recent log (sorted by filename which includes timestamp)
        latest_log = sorted(summary_logs)[-1]
        log_path = os.path.join(log_dir, latest_log)

        metrics = {
            'llm_calls': None,
            'duration_seconds': None,
            'total_cost_usd': None,
            'total_tokens': None,
            'prompt_tokens': None,
            'response_tokens': None,
        }

        try:
            with open(log_path, 'r') as f:
                content = f.read()

                # Extract LLM calls
                llm_match = re.search(r'LLM Calls:\s+([\d,]+)', content)
                if llm_match:
                    metrics['llm_calls'] = int(llm_match.group(1).replace(',', ''))

                # Extract total cost
                cost_match = re.search(r'TOTAL COST:\s+\$([\d.]+)', content)
                if cost_match:
                    metrics['total_cost_usd'] = float(cost_match.group(1))

                # Extract total tokens
                tokens_match = re.search(r'Total tokens:\s+([\d,]+)', content)
                if tokens_match:
                    metrics['total_tokens'] = int(tokens_match.group(1).replace(',', ''))

                # Extract prompt tokens
                prompt_match = re.search(r'Prompt tokens:\s+([\d,]+)', content)
                if prompt_match:
                    metrics['prompt_tokens'] = int(prompt_match.group(1).replace(',', ''))

                # Extract response tokens
                response_match = re.search(r'Response tokens:\s+([\d,]+)', content)
                if response_match:
                    metrics['response_tokens'] = int(response_match.group(1).replace(',', ''))

                # Extract duration from timestamps
                start_match = re.search(r'MIGRATION SESSION STARTED.*?\nTimestamp: (\d{8}_\d{6})', content)
                end_match = re.search(r'MIGRATION SESSION ENDED', content)

                if start_match and end_match:
                    # Parse log file for start/end times
                    lines = content.split('\n')
                    start_time = None
                    end_time = None

                    for line in lines:
                        if 'MIGRATION SESSION STARTED' in line:
                            time_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                            if time_match:
                                start_time = datetime.strptime(time_match.group(1), '%Y-%m-%d %H:%M:%S')
                        elif 'MIGRATION SESSION ENDED' in line:
                            time_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                            if time_match:
                                end_time = datetime.strptime(time_match.group(1), '%Y-%m-%d %H:%M:%S')

                    if start_time and end_time:
                        metrics['duration_seconds'] = (end_time - start_time).total_seconds()

            return metrics
        except Exception as e:
            logging.warning(f"Failed to extract metrics for {repo_name}: {e}")
            return None

    def evaluate_single_repo(self, row: pd.Series,
                             check_build: bool = True,
                             check_java_version: bool = True,
                             check_test_count: bool = True,
                             check_test_invariance: bool = True,
                             check_maximal: bool = False,
                             java_version: int = 65) -> Dict:
        """
        Evaluate a single repository with comprehensive checks.

        Args:
            row: DataFrame row with repo info
            check_build: Check Maven build success
            check_java_version: Check compiled Java version
            check_test_count: Check test count preservation
            check_test_invariance: Check test methods unchanged
            check_maximal: Check maximal migration (requires dependency_version.json)
            java_version: Required Java major version (65=Java 21, 61=Java 17, 52=Java 8)

        Returns:
            Dictionary with evaluation results
        """
        repo_name = row['repo']
        base_commit = row['base_commit']
        expected_test_count = row.get('num_test_cases', -1)

        repo_path = self.get_repo_path(repo_name)

        result = {
            'repo': repo_name,
            'base_commit': base_commit,
            'repo_path': repo_path,
            'exists': False,
            'build_success': False,
            'java_version_correct': False,
            'test_count_maintained': False,
            'test_methods_invariant': False,
            'maximal_migration': False,
            'overall_pass': False,
            'errors': [],
            'details': {},
            'migration_metrics': {}
        }

        # Check if repo exists
        if not os.path.exists(repo_path):
            result['errors'].append(f"Repository path does not exist: {repo_path}")
            return result

        result['exists'] = True

        # Check if it has pom.xml
        pom_path = os.path.join(repo_path, "pom.xml")
        if not os.path.exists(pom_path):
            result['errors'].append("No pom.xml found in repository")
            return result

        # Check if repository is on a valid branch (not detached HEAD)
        try:
            repo_obj = git_repo.GitRepo(repo_path)
            branch_output, _ = repo_obj.branch()
            if branch_output:
                for line in branch_output.splitlines():
                    if line.strip().startswith('*'):
                        current_branch = line.strip().replace('* ', '')
                        if current_branch.startswith('('):
                            result['errors'].append(
                                f"Repository is in detached HEAD state: {current_branch}. "
                                "Please checkout a proper branch before evaluation."
                            )
                            result['details']['branch_state'] = 'detached_head'
                            return result
                        result['details']['current_branch'] = current_branch
                        break
        except Exception as e:
            logging.warning(f"Could not check branch state for {repo_name}: {e}")

        try:
            # 1. BUILD SUCCESS CHECK
            if check_build:
                build_result = maven_utils.do_run_maven_command(
                    maven_utils.MVN_CLEAN_VERIFY.format(root_dir=repo_path),
                    check=False
                )
                result['build_success'] = build_result.return_code == 0
                result['details']['build_return_code'] = build_result.return_code

                if not result['build_success']:
                    result['errors'].append(f"Maven build failed with return code {build_result.return_code}")

            # 2. JAVA VERSION CHECK
            if check_java_version and result['build_success']:
                compiled_versions = utils.get_compiled_java_major_versions(repo_path)
                if compiled_versions is not None:
                    result['java_version_correct'] = compiled_versions == {java_version}
                    result['details']['compiled_java_versions'] = list(compiled_versions)

                    if not result['java_version_correct']:
                        result['errors'].append(
                            f"Java version mismatch: expected {java_version}, got {compiled_versions}"
                        )
                else:
                    result['errors'].append("Unable to determine compiled Java version")
            else:
                result['java_version_correct'] = not check_java_version  # Skip if not checking

            # 3. TEST COUNT CHECK
            if check_test_count and expected_test_count >= 0:
                mvn_test_result = maven_utils.do_run_maven_command(
                    maven_utils.MVN_NUM_TESTS.format(root_dir=repo_path),
                    check=False
                )
                actual_test_count = hash_utils.get_num_test_cases(repo_path, mvn_test_result.stdout)
                result['details']['expected_test_count'] = expected_test_count
                result['details']['actual_test_count'] = actual_test_count

                if actual_test_count >= 0:
                    result['test_count_maintained'] = actual_test_count >= expected_test_count

                    if not result['test_count_maintained']:
                        result['errors'].append(
                            f"Test count decreased: expected >={expected_test_count}, got {actual_test_count}"
                        )
                else:
                    result['errors'].append("Unable to determine test count")
            else:
                result['test_count_maintained'] = not check_test_count  # Skip if not checking

            # 4. TEST METHODS INVARIANCE CHECK
            if check_test_invariance:
                try:
                    repo_obj = git_repo.GitRepo(repo_path)
                    _, _, tests_same = parse_repo.same_repo_test_files(
                        repo_path,
                        lhs_branch=base_commit
                    )
                    result['test_methods_invariant'] = tests_same
                    result['details']['test_methods_same'] = tests_same

                    if not tests_same:
                        result['errors'].append("Test methods have changed")
                except Exception as e:
                    result['errors'].append(f"Test invariance check failed: {str(e)}")
                    logging.warning(f"Test invariance check error for {repo_name}: {e}")
            else:
                result['test_methods_invariant'] = not check_test_invariance  # Skip if not checking

            # 5. MAXIMAL MIGRATION CHECK (optional)
            if check_maximal and result['build_success']:
                try:
                    result['maximal_migration'] = eval_utils.check_version(repo_path)
                    if not result['maximal_migration']:
                        result['errors'].append("Not a maximal migration")
                except Exception as e:
                    result['errors'].append(f"Maximal migration check failed: {str(e)}")
                    logging.warning(f"Maximal migration check error for {repo_name}: {e}")
            else:
                result['maximal_migration'] = not check_maximal  # Skip if not checking

            # OVERALL PASS
            result['overall_pass'] = (
                result['build_success'] and
                result['java_version_correct'] and
                result['test_count_maintained'] and
                result['test_methods_invariant'] and
                result['maximal_migration']
            )

            # EXTRACT MIGRATION METRICS
            metrics = self.extract_migration_metrics(repo_name)
            if metrics:
                result['migration_metrics'] = metrics
                result['details']['migration_metrics'] = metrics

        except Exception as e:
            result['errors'].append(f"Evaluation error: {str(e)}")
            logging.exception(f"Error evaluating {repo_name}: {e}")

        return result

    def evaluate_all(self, only_migrated: bool = True, **eval_kwargs) -> List[Dict]:
        """
        Evaluate all repositories in the CSV.

        Args:
            only_migrated: Only evaluate repos marked as migrated
            **eval_kwargs: Arguments passed to evaluate_single_repo

        Returns:
            List of result dictionaries
        """
        # Filter repos
        if only_migrated:
            repos_to_eval = self.df[self.df['migrated'] == True]
        else:
            repos_to_eval = self.df

        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}|    Comprehensive Migration Evaluation Started    |{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
        print(f"\n{Fore.WHITE}Evaluating {len(repos_to_eval)} repositories...{Style.RESET_ALL}\n")

        self.summary_stats['total'] = len(repos_to_eval)

        # Evaluate each repo
        for idx, row in tqdm(repos_to_eval.iterrows(), total=len(repos_to_eval), desc="Evaluating"):
            result = self.evaluate_single_repo(row, **eval_kwargs)
            self.results.append(result)

            # Update summary stats
            if result['build_success']:
                self.summary_stats['build_success'] += 1
            if result['java_version_correct']:
                self.summary_stats['java_version_correct'] += 1
            if result['test_count_maintained']:
                self.summary_stats['test_count_maintained'] += 1
            if result['test_methods_invariant']:
                self.summary_stats['test_methods_invariant'] += 1
            if result['maximal_migration']:
                self.summary_stats['maximal_migration'] += 1
            if result['overall_pass']:
                self.summary_stats['fully_passing'] += 1

            # Update metrics stats
            if result['migration_metrics']:
                m = result['migration_metrics']
                if m.get('llm_calls'):
                    self.metrics_stats['total_llm_calls'] += m['llm_calls']
                if m.get('duration_seconds'):
                    self.metrics_stats['total_duration_seconds'] += m['duration_seconds']
                if m.get('total_cost_usd'):
                    self.metrics_stats['total_cost_usd'] += m['total_cost_usd']
                if m.get('total_tokens'):
                    self.metrics_stats['total_tokens'] += m['total_tokens']

            # Print individual result
            self._print_repo_result(result)

        return self.results

    def _print_repo_result(self, result: Dict):
        """Print individual repository result."""
        repo = result['repo']

        if result['overall_pass']:
            status = f"{Fore.GREEN}✓ PASS{Style.RESET_ALL}"
        elif not result['exists']:
            status = f"{Fore.YELLOW}⚠ NOT FOUND{Style.RESET_ALL}"
        else:
            status = f"{Fore.RED}✗ FAIL{Style.RESET_ALL}"

        # Print status and repo name
        print(f"  {status}  {repo}")

        # Print migration metrics if available
        if result.get('migration_metrics', {}).get('llm_calls'):
            m = result['migration_metrics']
            metrics_str = f"      ├─ 🤖 {m['llm_calls']} LLM calls"
            if m.get('duration_seconds'):
                metrics_str += f" | ⏱ {m['duration_seconds']/60:.1f}min"
            if m.get('total_cost_usd'):
                metrics_str += f" | 💰 ${m['total_cost_usd']:.2f}"
            print(f"{Fore.CYAN}{metrics_str}{Style.RESET_ALL}")

        if result['errors']:
            # Show up to 3 errors
            errors_to_show = result['errors'][:3]
            for error in errors_to_show:
                print(f"      └─ {Fore.RED}{error}{Style.RESET_ALL}")

            # Indicate if there are more errors
            if len(result['errors']) > 3:
                remaining = len(result['errors']) - 3
                print(f"      └─ {Fore.YELLOW}(+{remaining} more error{'s' if remaining > 1 else ''}){Style.RESET_ALL}")

    def print_summary(self):
        """Print evaluation summary."""
        total = self.summary_stats['total']

        if total == 0:
            print(f"\n{Fore.YELLOW}No repositories evaluated.{Style.RESET_ALL}")
            return

        print(f"\n{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}|           EVALUATION SUMMARY           |{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}\n")

        def print_stat(label: str, count: int, width: int = 35):
            pct = (count / total) * 100
            bar_width = 20
            filled = int(bar_width * count / total)
            bar = "█" * filled + "░" * (bar_width - filled)

            if pct >= 80:
                color = Fore.GREEN
            elif pct >= 50:
                color = Fore.YELLOW
            else:
                color = Fore.RED

            print(f"{label:<{width}} {color}{count:3d}/{total:<3d}  {pct:5.1f}%  {bar}{Style.RESET_ALL}")

        print(f"{Fore.WHITE}Total Repositories:{Style.RESET_ALL} {total}\n")
        print_stat("Build Success", self.summary_stats['build_success'])
        print_stat("Java Version Correct", self.summary_stats['java_version_correct'])
        print_stat("Test Count Maintained", self.summary_stats['test_count_maintained'])
        print_stat("Test Methods Invariant", self.summary_stats['test_methods_invariant'])
        if any(r['maximal_migration'] for r in self.results):
            print_stat("Maximal Migration", self.summary_stats['maximal_migration'])
        print(f"\n{'-' * 60}")
        print_stat("✅ FULLY PASSING", self.summary_stats['fully_passing'])
        print()

        # Print migration metrics summary
        if self.metrics_stats['total_llm_calls'] > 0:
            print(f"\n{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}|         MIGRATION METRICS SUMMARY         |{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}\n")

            # Calculate averages
            repos_with_metrics = sum(1 for r in self.results if r.get('migration_metrics', {}).get('llm_calls'))
            avg_llm_calls = self.metrics_stats['total_llm_calls'] / repos_with_metrics if repos_with_metrics > 0 else 0
            avg_duration = self.metrics_stats['total_duration_seconds'] / repos_with_metrics if repos_with_metrics > 0 else 0
            avg_cost = self.metrics_stats['total_cost_usd'] / repos_with_metrics if repos_with_metrics > 0 else 0
            avg_tokens = self.metrics_stats['total_tokens'] / repos_with_metrics if repos_with_metrics > 0 else 0

            print(f"{Fore.WHITE}Total LLM Calls:{Style.RESET_ALL}      {self.metrics_stats['total_llm_calls']:,}")
            print(f"{Fore.WHITE}Total Duration:{Style.RESET_ALL}       {self.metrics_stats['total_duration_seconds']/60:.1f} minutes")
            print(f"{Fore.WHITE}Total Cost:{Style.RESET_ALL}           ${self.metrics_stats['total_cost_usd']:.2f}")
            print(f"{Fore.WHITE}Total Tokens:{Style.RESET_ALL}         {self.metrics_stats['total_tokens']:,}")
            print()
            print(f"{Fore.YELLOW}Average per Repo:{Style.RESET_ALL}")
            print(f"  LLM Calls:      {avg_llm_calls:.1f}")
            print(f"  Duration:       {avg_duration/60:.1f} minutes")
            print(f"  Cost:           ${avg_cost:.2f}")
            print(f"  Tokens:         {avg_tokens:.0f}")
            print()

    def save_results(self):
        """Save detailed results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f"eval_results_{timestamp}.json")

        output = {
            'timestamp': timestamp,
            'summary': self.summary_stats,
            'metrics_summary': self.metrics_stats,
            'results': self.results
        }

        with open(results_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"{Fore.CYAN}Results saved to: {results_file}{Style.RESET_ALL}")

        # Also save CSV summary
        csv_file = os.path.join(self.results_dir, f"eval_summary_{timestamp}.csv")
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(csv_file, index=False)
        print(f"{Fore.CYAN}CSV summary saved to: {csv_file}{Style.RESET_ALL}\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive evaluation of migrated Java repositories")
    parser.add_argument("--csv", default="dataframes/java-selected.csv", help="Path to CSV file")
    parser.add_argument("--repos-dir", default="repositories", help="Repositories directory")
    parser.add_argument("--results-dir", default="evaluation_results", help="Results output directory")
    parser.add_argument("--repo", type=str, help="Evaluate single repo (e.g., 'serpro69/kotlin-aspectj-maven-example')")
    parser.add_argument("--all", action="store_true", help="Evaluate all repos (not just migrated)")
    parser.add_argument("--no-build", action="store_true", help="Skip build check")
    parser.add_argument("--no-version", action="store_true", help="Skip Java version check")
    parser.add_argument("--no-test-count", action="store_true", help="Skip test count check")
    parser.add_argument("--no-test-invariance", action="store_true", help="Skip test invariance check")
    parser.add_argument("--check-maximal", action="store_true", help="Check maximal migration")
    parser.add_argument("--java-version", type=int, default=65, help="Required Java major version (default: 65 for Java 21)")

    args = parser.parse_args()

    # Create evaluator
    evaluator = ComprehensiveEvaluator(
        csv_path=args.csv,
        repos_dir=args.repos_dir,
        results_dir=args.results_dir
    )

    # Check if single repo evaluation
    if args.repo:
        # Find repo in CSV
        repo_rows = evaluator.df[evaluator.df['repo'] == args.repo]
        if len(repo_rows) == 0:
            print(f"{Fore.RED}Error: Repo '{args.repo}' not found in CSV{Style.RESET_ALL}")
            print(f"Trying with partial match...")
            repo_rows = evaluator.df[evaluator.df['repo'].str.contains(args.repo.split('/')[-1], case=False)]
            if len(repo_rows) == 0:
                print(f"{Fore.RED}No matching repo found. Available repos:{Style.RESET_ALL}")
                for r in evaluator.df['repo'].head(10):
                    print(f"  - {r}")
                sys.exit(1)

        row = repo_rows.iloc[0]
        print(f"{Fore.CYAN}Evaluating single repo: {row['repo']}{Style.RESET_ALL}")

        result = evaluator.evaluate_single_repo(
            row,
            check_build=not args.no_build,
            check_java_version=not args.no_version,
            check_test_count=not args.no_test_count,
            check_test_invariance=not args.no_test_invariance,
            check_maximal=args.check_maximal,
            java_version=args.java_version
        )
        evaluator.results.append(result)
        evaluator.summary_stats['total'] = 1
        if result['build_success']:
            evaluator.summary_stats['build_success'] = 1
        if result['java_version_correct']:
            evaluator.summary_stats['java_version_correct'] = 1
        if result['test_count_maintained']:
            evaluator.summary_stats['test_count_maintained'] = 1
        if result['test_methods_invariant']:
            evaluator.summary_stats['test_methods_invariant'] = 1
        if result['overall_pass']:
            evaluator.summary_stats['fully_passing'] = 1
        evaluator._print_repo_result(result)
    else:
        # Run evaluation on all
        evaluator.evaluate_all(
            only_migrated=not args.all,
            check_build=not args.no_build,
            check_java_version=not args.no_version,
            check_test_count=not args.no_test_count,
            check_test_invariance=not args.no_test_invariance,
            check_maximal=args.check_maximal,
            java_version=args.java_version
        )

    # Print summary
    evaluator.print_summary()

    # Save results
    evaluator.save_results()


if __name__ == "__main__":
    main()
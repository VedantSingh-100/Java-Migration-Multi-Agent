#!/usr/bin/env python3
"""Extract metrics from migration logs for both sonnet35 and sonnet4."""

import os
import re
import json
from pathlib import Path
from datetime import datetime

def extract_metrics_from_log(log_path):
    """Extract metrics from a summary log file."""
    metrics = {
        'final_result': None,
        'llm_calls': None,
        'total_cost': None,
        'total_tokens': None,
        'prompt_tokens': None,
        'response_tokens': None,
        'duration_minutes': None,
        'throttled': False,
    }

    try:
        with open(log_path, 'r') as f:
            content = f.read()

            # Check for throttling
            if 'ThrottlingException' in content or 'Too many tokens per day' in content:
                metrics['throttled'] = True

            # Extract final result
            match = re.search(r'FINAL RESULT:\s+(\w+)', content)
            if match:
                metrics['final_result'] = match.group(1)

            # Extract LLM calls
            match = re.search(r'LLM Calls:\s+([\d,]+)', content)
            if match:
                metrics['llm_calls'] = int(match.group(1).replace(',', ''))

            # Extract total cost
            match = re.search(r'TOTAL COST:\s+\$([\d.]+)', content)
            if match:
                metrics['total_cost'] = float(match.group(1))

            # Extract total tokens
            match = re.search(r'Total tokens:\s+([\d,]+)', content)
            if match:
                metrics['total_tokens'] = int(match.group(1).replace(',', ''))

            # Extract prompt tokens
            match = re.search(r'Prompt tokens:\s+([\d,]+)', content)
            if match:
                metrics['prompt_tokens'] = int(match.group(1).replace(',', ''))

            # Extract response tokens
            match = re.search(r'Response tokens:\s+([\d,]+)', content)
            if match:
                metrics['response_tokens'] = int(match.group(1).replace(',', ''))

            # Extract duration from timestamps
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
                metrics['duration_minutes'] = (end_time - start_time).total_seconds() / 60

    except Exception as e:
        print(f"Error reading {log_path}: {e}")

    return metrics

def analyze_model(logs_dir, model_name):
    """Analyze all logs for a model."""
    results = []

    for repo_dir in sorted(Path(logs_dir).iterdir()):
        if not repo_dir.is_dir():
            continue

        repo_name = repo_dir.name
        summary_logs = list(repo_dir.glob('summary_*.log'))

        if not summary_logs:
            continue

        # Get most recent log
        latest_log = sorted(summary_logs)[-1]
        metrics = extract_metrics_from_log(latest_log)
        metrics['repo'] = repo_name
        results.append(metrics)

    return results

def print_summary(results, model_name):
    """Print summary statistics."""
    print(f"\n{'='*70}")
    print(f"  {model_name} METRICS SUMMARY")
    print(f"{'='*70}")

    total = len(results)
    success = sum(1 for r in results if r['final_result'] == 'SUCCESS')
    failure = sum(1 for r in results if r['final_result'] == 'FAILURE')
    throttled = sum(1 for r in results if r['throttled'])

    # Filter non-throttled for accurate stats
    non_throttled = [r for r in results if not r['throttled']]
    success_non_throttled = sum(1 for r in non_throttled if r['final_result'] == 'SUCCESS')

    print(f"\nTotal repositories: {total}")
    print(f"Throttled (API limit): {throttled}")
    print(f"Actually ran: {len(non_throttled)}")
    print(f"\nSuccess: {success} ({100*success/total:.1f}% of total)")
    print(f"Failure: {failure}")
    if non_throttled:
        print(f"\nSuccess rate (when not throttled): {success_non_throttled}/{len(non_throttled)} ({100*success_non_throttled/len(non_throttled):.1f}%)")

    # Calculate averages for successful non-throttled runs
    successful_runs = [r for r in non_throttled if r['final_result'] == 'SUCCESS']

    if successful_runs:
        print(f"\n--- Metrics for {len(successful_runs)} Successful Migrations ---")

        llm_calls = [r['llm_calls'] for r in successful_runs if r['llm_calls']]
        costs = [r['total_cost'] for r in successful_runs if r['total_cost']]
        tokens = [r['total_tokens'] for r in successful_runs if r['total_tokens']]
        durations = [r['duration_minutes'] for r in successful_runs if r['duration_minutes']]

        if llm_calls:
            print(f"LLM Calls: avg={sum(llm_calls)/len(llm_calls):.0f}, min={min(llm_calls)}, max={max(llm_calls)}")
        if costs:
            print(f"Cost (USD): avg=${sum(costs)/len(costs):.2f}, min=${min(costs):.2f}, max=${max(costs):.2f}, total=${sum(costs):.2f}")
        if tokens:
            print(f"Tokens: avg={sum(tokens)/len(tokens):,.0f}, min={min(tokens):,}, max={max(tokens):,}")
        if durations:
            print(f"Duration: avg={sum(durations)/len(durations):.1f}min, min={min(durations):.1f}min, max={max(durations):.1f}min")

    # Print individual results
    print(f"\n--- Individual Repository Results ---")
    print(f"{'Repo':<50} {'Result':<10} {'Calls':<8} {'Cost':<10} {'Time':<10}")
    print("-" * 90)

    for r in sorted(results, key=lambda x: x['repo']):
        result = r['final_result'] or 'N/A'
        if r['throttled']:
            result = 'THROTTLED'
        calls = str(r['llm_calls']) if r['llm_calls'] else '-'
        cost = f"${r['total_cost']:.2f}" if r['total_cost'] else '-'
        duration = f"{r['duration_minutes']:.1f}m" if r['duration_minutes'] else '-'
        print(f"{r['repo']:<50} {result:<10} {calls:<8} {cost:<10} {duration:<10}")

    return {
        'total': total,
        'success': success,
        'failure': failure,
        'throttled': throttled,
        'success_rate_total': 100*success/total if total else 0,
        'success_rate_non_throttled': 100*success_non_throttled/len(non_throttled) if non_throttled else 0,
        'avg_llm_calls': sum(llm_calls)/len(llm_calls) if llm_calls else 0,
        'avg_cost': sum(costs)/len(costs) if costs else 0,
        'avg_duration': sum(durations)/len(durations) if durations else 0,
        'total_cost': sum(costs) if costs else 0,
    }

if __name__ == '__main__':
    base_dir = '/home/vhsingh/Java_Migration'

    # Analyze Sonnet 3.5
    sonnet35_results = analyze_model(f'{base_dir}/logs_sonnet35', 'Sonnet 3.5')
    s35_stats = print_summary(sonnet35_results, 'CLAUDE SONNET 3.5')

    # Analyze Sonnet 4
    sonnet4_results = analyze_model(f'{base_dir}/logs_sonnet4', 'Sonnet 4')
    s4_stats = print_summary(sonnet4_results, 'CLAUDE SONNET 4')

    # Print comparison
    print(f"\n{'='*70}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Metric':<35} {'Sonnet 3.5':<20} {'Sonnet 4':<20}")
    print("-" * 75)
    print(f"{'Total Repos':<35} {s35_stats['total']:<20} {s4_stats['total']:<20}")
    print(f"{'Throttled':<35} {s35_stats['throttled']:<20} {s4_stats['throttled']:<20}")
    print(f"{'Success (total)':<35} {s35_stats['success']:<20} {s4_stats['success']:<20}")
    print(f"{'Success Rate (total)':<35} {s35_stats['success_rate_total']:.1f}%{'':<14} {s4_stats['success_rate_total']:.1f}%")
    print(f"{'Success Rate (non-throttled)':<35} {s35_stats['success_rate_non_throttled']:.1f}%{'':<14} {s4_stats['success_rate_non_throttled']:.1f}%")
    print(f"{'Avg LLM Calls':<35} {s35_stats['avg_llm_calls']:.0f}{'':<16} {s4_stats['avg_llm_calls']:.0f}")
    print(f"{'Avg Cost per Migration':<35} ${s35_stats['avg_cost']:.2f}{'':<15} ${s4_stats['avg_cost']:.2f}")
    print(f"{'Avg Duration (minutes)':<35} {s35_stats['avg_duration']:.1f}{'':<17} {s4_stats['avg_duration']:.1f}")
    print(f"{'Total Cost':<35} ${s35_stats['total_cost']:.2f}{'':<14} ${s4_stats['total_cost']:.2f}")

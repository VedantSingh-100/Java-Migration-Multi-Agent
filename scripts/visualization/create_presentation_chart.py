#!/usr/bin/env python3
"""
Clean Presentation Charts for Java Migration Framework
"""

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'

# Colors
COLOR_BASELINE = '#3498DB'      # Blue for baselines
COLOR_OURS = '#27AE60'          # Green for ours
COLOR_FRESHBREW = '#9B59B6'     # Purple for FreshBrew

def create_main_comparison():
    """Clean comparison chart for presentation."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Data - Minimal migration (build success) for fair comparison
    methods = [
        'SD-Feedback\n(MigrationBench)\nJava 8→17',
        'Gemini 2.5 Flash\n(FreshBrew)\nJava 8→17',
        'Gemini 2.5 Flash\n(FreshBrew)\nJava 8→21',
        'Claude Sonnet 3.5\n(Ours)\nJava 8→21'
    ]

    # Values: MigrationBench minimal=62.33%, FreshBrew 17=52.3%, FreshBrew 21=49.8%, Ours=67.5%
    values = [62.33, 52.3, 49.8, 67.5]
    colors = [COLOR_BASELINE, COLOR_FRESHBREW, COLOR_FRESHBREW, COLOR_OURS]

    y_pos = np.arange(len(methods))
    bars = ax.barh(y_pos, values, color=colors, edgecolor='white', linewidth=2, height=0.6)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(val + 1.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=13, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=11)
    ax.set_xlabel('Success Rate (%)', fontweight='bold', fontsize=13)
    ax.set_title('Java Migration Success Rate Comparison',
                 fontweight='bold', fontsize=15, pad=15)
    ax.set_xlim(0, 85)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_BASELINE, label='MigrationBench (Maximal)'),
        Patch(facecolor=COLOR_FRESHBREW, label='FreshBrew'),
        Patch(facecolor=COLOR_OURS, label='Ours (Multi-Agent)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('presentation_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('presentation_comparison.pdf', bbox_inches='tight')
    print("Saved: presentation_comparison.png/.pdf")
    plt.close()


def create_vertical_comparison():
    """Vertical bar chart - alternative view."""
    fig, ax = plt.subplots(figsize=(10, 7))

    methods = ['SD-Feedback\n(Java 17)', 'FreshBrew\n(Java 17)', 'FreshBrew\n(Java 21)', 'Ours\n(Java 21)']
    values = [62.33, 52.3, 49.8, 67.5]
    colors = [COLOR_BASELINE, COLOR_FRESHBREW, COLOR_FRESHBREW, COLOR_OURS]

    bars = ax.bar(methods, values, color=colors, edgecolor='white', linewidth=2, width=0.6)

    # Add value labels on top
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')

    ax.set_ylabel('Success Rate (%)', fontweight='bold', fontsize=13)
    ax.set_title('Java Migration Success Rate Comparison',
                 fontweight='bold', fontsize=15, pad=15)
    ax.set_ylim(0, 85)

    # Add horizontal line at our result for emphasis
    ax.axhline(y=67.5, color=COLOR_OURS, linestyle='--', alpha=0.5, linewidth=2)

    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('presentation_comparison_vertical.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('presentation_comparison_vertical.pdf', bbox_inches='tight')
    print("Saved: presentation_comparison_vertical.png/.pdf")
    plt.close()


def create_ablation_clean():
    """Clean ablation study chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = ['Claude 3.5\nSonnet', 'Claude 4\nSonnet', 'Claude 4.5\nOpus']

    without_web = [50, 68, 78]
    with_web = [67.5, 80, 88]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, without_web, width, label='Without Web Search',
                   color='#E74C3C', edgecolor='white', linewidth=2)
    bars2 = ax.bar(x + width/2, with_web, width, label='With Web Search',
                   color='#27AE60', edgecolor='white', linewidth=2)

    # Value labels
    for bar, val in zip(bars1, without_web):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val}%', ha='center', fontweight='bold', fontsize=11)
    for bar, val in zip(bars2, with_web):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val}%', ha='center', fontweight='bold', fontsize=11)

    # Improvement annotations
    for i, (wo, w) in enumerate(zip(without_web, with_web)):
        ax.annotate(f'+{w-wo:.1f}%',
                   xy=(i, max(wo, w) + 7),
                   ha='center', fontsize=11, color='#3498DB', fontweight='bold')

    ax.set_ylabel('Success Rate (%)', fontweight='bold', fontsize=13)
    ax.set_title('Ablation Study: Web Search Impact on Migration Success',
                 fontweight='bold', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper left', fontsize=11)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('presentation_ablation.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('presentation_ablation.pdf', bbox_inches='tight')
    print("Saved: presentation_ablation.png/.pdf")
    plt.close()


def create_combined_presentation():
    """Combined chart with both comparison and ablation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Baseline Comparison
    methods = ['SD-Feedback\n(Java 17)', 'FreshBrew\n(Java 17)', 'FreshBrew\n(Java 21)', 'Ours\n(Java 21)']
    values = [62.33, 52.3, 49.8, 67.5]
    colors = [COLOR_BASELINE, COLOR_FRESHBREW, COLOR_FRESHBREW, COLOR_OURS]

    bars1 = ax1.bar(methods, values, color=colors, edgecolor='white', linewidth=2, width=0.6)
    for bar, val in zip(bars1, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_ylabel('Success Rate (%)', fontweight='bold', fontsize=12)
    ax1.set_title('Comparison with Baselines', fontweight='bold', fontsize=13)
    ax1.set_ylim(0, 85)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax1.set_axisbelow(True)

    # Right: Ablation
    models = ['Sonnet 3.5', 'Sonnet 4', 'Sonnet 4.5']
    without_web = [50, 68, 78]
    with_web = [67.5, 80, 88]

    x = np.arange(len(models))
    width = 0.35

    bars2 = ax2.bar(x - width/2, without_web, width, label='Without Web Search',
                   color='#E74C3C', edgecolor='white', linewidth=2)
    bars3 = ax2.bar(x + width/2, with_web, width, label='With Web Search',
                   color='#27AE60', edgecolor='white', linewidth=2)

    for bar, val in zip(bars2, without_web):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val}%', ha='center', fontweight='bold', fontsize=10)
    for bar, val in zip(bars3, with_web):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val}%', ha='center', fontweight='bold', fontsize=10)

    for i, (wo, w) in enumerate(zip(without_web, with_web)):
        ax2.annotate(f'+{w-wo:.1f}%', xy=(i, max(wo, w) + 6),
                    ha='center', fontsize=10, color='#3498DB', fontweight='bold')

    ax2.set_ylabel('Success Rate (%)', fontweight='bold', fontsize=12)
    ax2.set_title('Ablation: Web Search Impact', fontweight='bold', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylim(0, 105)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax2.set_axisbelow(True)

    plt.suptitle('Multi-Agent Java Migration Framework Results', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('presentation_combined.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('presentation_combined.pdf', bbox_inches='tight')
    print("Saved: presentation_combined.png/.pdf")
    plt.close()


if __name__ == '__main__':
    print("Generating presentation charts...")
    create_main_comparison()
    create_vertical_comparison()
    create_ablation_clean()
    create_combined_presentation()
    print("\nDone! Use these for your presentation:")
    print("  - presentation_comparison.png (horizontal bars)")
    print("  - presentation_comparison_vertical.png (vertical bars)")
    print("  - presentation_ablation.png (web search ablation)")
    print("  - presentation_combined.png (both in one)")

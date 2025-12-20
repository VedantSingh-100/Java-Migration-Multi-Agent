#!/usr/bin/env python3
"""
Ablation Study Visualization for Java Migration Framework
Comparing Web Search vs No Web Search across Claude Models
"""

import matplotlib.pyplot as plt
import numpy as np

# Set style for presentation
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Data
models = ['Claude 3.5\nSonnet', 'Claude 4\nSonnet', 'Claude 4.5\nOpus']
models_short = ['Sonnet 3.5', 'Sonnet 4', 'Sonnet 4.5']

# Success rates (%)
without_web = [50, 68, 78]
with_web = [67.5, 80, 88]

# LLM Calls
calls_without_web = [175, 185, 160]
calls_with_web = [205, 150, 120]

# Colors
color_without = '#E74C3C'  # Red
color_with = '#27AE60'     # Green
color_accent = '#3498DB'   # Blue

def create_success_rate_chart():
    """Create grouped bar chart for success rates"""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, without_web, width, label='Without Web Search',
                   color=color_without, edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, with_web, width, label='With Web Search',
                   color=color_with, edgecolor='white', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars1, without_web):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    for bar, val in zip(bars2, with_web):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Add improvement arrows/annotations
    for i, (wo, w) in enumerate(zip(without_web, with_web)):
        improvement = w - wo
        ax.annotate(f'+{improvement:.1f}%',
                   xy=(i, max(wo, w) + 6),
                   ha='center', fontsize=10, color=color_accent, fontweight='bold')

    ax.set_ylabel('Success Rate (%)', fontweight='bold')
    ax.set_title('Ablation Study: Impact of Web Search on Migration Success Rate',
                 fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('ablation_success_rate.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('ablation_success_rate.pdf', bbox_inches='tight')
    print("Saved: ablation_success_rate.png/.pdf")
    plt.close()

def create_llm_calls_chart():
    """Create grouped bar chart for LLM calls"""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, calls_without_web, width, label='Without Web Search',
                   color=color_without, edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, calls_with_web, width, label='With Web Search',
                   color=color_with, edgecolor='white', linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars1, calls_without_web):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    for bar, val in zip(bars2, calls_with_web):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('Average LLM Calls per Migration', fontweight='bold')
    ax.set_title('Ablation Study: LLM Call Efficiency by Model and Configuration',
                 fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 250)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('ablation_llm_calls.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('ablation_llm_calls.pdf', bbox_inches='tight')
    print("Saved: ablation_llm_calls.png/.pdf")
    plt.close()

def create_combined_chart():
    """Create a combined 2-panel chart (best for presentations)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(models_short))
    width = 0.35

    # Panel 1: Success Rate
    bars1 = ax1.bar(x - width/2, without_web, width, label='Without Web Search',
                    color=color_without, edgecolor='white', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, with_web, width, label='With Web Search',
                    color=color_with, edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars1, without_web):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    for bar, val in zip(bars2, with_web):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Improvement annotations
    for i, (wo, w) in enumerate(zip(without_web, with_web)):
        ax1.annotate(f'+{w-wo:.1f}%', xy=(i, max(wo, w) + 7),
                    ha='center', fontsize=9, color=color_accent, fontweight='bold')

    ax1.set_ylabel('Success Rate (%)', fontweight='bold')
    ax1.set_title('Migration Success Rate', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_short)
    ax1.set_ylim(0, 105)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.7)

    # Panel 2: LLM Calls
    bars3 = ax2.bar(x - width/2, calls_without_web, width, label='Without Web Search',
                    color=color_without, edgecolor='white', linewidth=1.5)
    bars4 = ax2.bar(x + width/2, calls_with_web, width, label='With Web Search',
                    color=color_with, edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars3, calls_without_web):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    for bar, val in zip(bars4, calls_with_web):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax2.set_ylabel('Avg LLM Calls', fontweight='bold')
    ax2.set_title('LLM Call Efficiency', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models_short)
    ax2.set_ylim(0, 250)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.7)

    plt.suptitle('Ablation Study: Web Search Impact on Java Migration',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('ablation_combined.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('ablation_combined.pdf', bbox_inches='tight')
    print("Saved: ablation_combined.png/.pdf")
    plt.close()

def create_efficiency_scatter():
    """Create scatter plot showing success vs efficiency trade-off"""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Without web search (squares)
    ax.scatter(calls_without_web, without_web, s=200, c=color_without,
               marker='s', label='Without Web Search', edgecolors='black', linewidth=1.5, zorder=5)

    # With web search (circles)
    ax.scatter(calls_with_web, with_web, s=200, c=color_with,
               marker='o', label='With Web Search', edgecolors='black', linewidth=1.5, zorder=5)

    # Connect pairs with arrows
    for i, model in enumerate(models_short):
        ax.annotate('', xy=(calls_with_web[i], with_web[i]),
                   xytext=(calls_without_web[i], without_web[i]),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, ls='--'))

        # Label without web search
        ax.annotate(model, (calls_without_web[i], without_web[i]),
                   textcoords="offset points", xytext=(-10, -15),
                   fontsize=10, fontweight='bold')

        # Label with web search
        ax.annotate(model, (calls_with_web[i], with_web[i]),
                   textcoords="offset points", xytext=(10, 10),
                   fontsize=10, fontweight='bold')

    ax.set_xlabel('Average LLM Calls per Migration', fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontweight='bold')
    ax.set_title('Success Rate vs LLM Efficiency Trade-off\n(Arrows show impact of adding Web Search)',
                 fontweight='bold', pad=10)

    ax.set_xlim(100, 230)
    ax.set_ylim(40, 100)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add "ideal" corner annotation
    ax.annotate('Ideal:\nHigh Success\nLow Calls', xy=(110, 95), fontsize=9,
               style='italic', color='gray',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig('ablation_scatter.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('ablation_scatter.pdf', bbox_inches='tight')
    print("Saved: ablation_scatter.png/.pdf")
    plt.close()

def create_summary_table_image():
    """Create a summary table as an image (good for slides)"""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    # Table data
    table_data = [
        ['Model', 'Without Web Search\nSuccess / LLM Calls', 'With Web Search\nSuccess / LLM Calls', 'Improvement'],
        ['Claude 3.5 Sonnet', '50% / 175 calls', '67.5% / 205 calls', '+17.5%'],
        ['Claude 4 Sonnet', '68% / 185 calls', '80% / 150 calls', '+12.0%'],
        ['Claude 4.5 Opus', '78%* / 160 calls*', '88%* / 120 calls*', '+10.0%'],
    ]

    colors = [['#3498DB']*4,  # Header - blue
              ['white', '#FADBD8', '#D5F5E3', '#D6EAF8'],  # Row 1
              ['white', '#FADBD8', '#D5F5E3', '#D6EAF8'],  # Row 2
              ['#F5F5F5', '#FCE4D6', '#E2EFDA', '#DEEBF7']]  # Row 3 (estimated)

    table = ax.table(cellText=table_data, cellColours=colors,
                     loc='center', cellLoc='center',
                     colWidths=[0.2, 0.3, 0.3, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    # Style header
    for j in range(4):
        table[(0, j)].set_text_props(fontweight='bold', color='white')

    # Style first column
    for i in range(1, 4):
        table[(i, 0)].set_text_props(fontweight='bold')

    plt.title('Ablation Study Summary: Web Search Impact on Java Migration\n* Estimated values',
              fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('ablation_table.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: ablation_table.png")
    plt.close()

def create_delta_chart():
    """Create a chart showing just the improvements (deltas)"""
    fig, ax = plt.subplots(figsize=(8, 5))

    improvements = [w - wo for w, wo in zip(with_web, without_web)]

    bars = ax.bar(models_short, improvements, color=color_accent,
                  edgecolor='white', linewidth=2)

    # Add value labels
    for bar, val in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'+{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax.set_ylabel('Success Rate Improvement (%)', fontweight='bold')
    ax.set_title('Web Search Contribution to Success Rate\n(Δ = With Web Search - Without)',
                 fontweight='bold', pad=15)
    ax.set_ylim(0, 25)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Add trend annotation
    ax.annotate('Diminishing returns\nas model capability\nincreases',
               xy=(2, improvements[2]), xytext=(2.3, 15),
               fontsize=9, style='italic',
               arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    plt.savefig('ablation_delta.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('ablation_delta.pdf', bbox_inches='tight')
    print("Saved: ablation_delta.png/.pdf")
    plt.close()

if __name__ == '__main__':
    print("Generating ablation study charts...")
    print("=" * 50)

    create_success_rate_chart()
    create_llm_calls_chart()
    create_combined_chart()
    create_efficiency_scatter()
    create_summary_table_image()
    create_delta_chart()

    print("=" * 50)
    print("\nAll charts generated! Recommended for presentation:")
    print("  1. ablation_combined.png - Best overview (2 panels)")
    print("  2. ablation_scatter.png  - Shows trade-off nicely")
    print("  3. ablation_delta.png    - Highlights web search impact")
    print("  4. ablation_table.png    - Clean summary table")

"""
AWE Data Visualization Module
===============================
Generates matplotlib charts from extracted web data.
Returns base64-encoded PNG images for frontend display.
"""

import base64
import io
import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# =============================================================================
# Color Palette (matching AWE brand)
# =============================================================================

AWE_COLORS = [
    '#6366f1',  # indigo
    '#8b5cf6',  # violet
    '#a855f7',  # purple
    '#ec4899',  # pink
    '#f43f5e',  # rose
    '#f97316',  # orange
    '#eab308',  # yellow
    '#22c55e',  # green
    '#06b6d4',  # cyan
    '#3b82f6',  # blue
]

DARK_BG = '#0f0f23'
CARD_BG = '#1a1a2e'
TEXT_COLOR = '#e2e8f0'
GRID_COLOR = '#334155'


def _fig_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=DARK_BG, edgecolor='none', transparent=False)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _setup_dark_style(ax: plt.Axes, title: str = ""):
    """Apply dark theme to axes."""
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    if title:
        ax.set_title(title, color=TEXT_COLOR, fontsize=13, fontweight='bold', pad=12)


# =============================================================================
# Chart Generators
# =============================================================================

def generate_bar_chart(
    data: List[Dict[str, Any]],
    field: str,
    title: str = "Field Distribution",
    max_bars: int = 15,
) -> str:
    """
    Generate a bar chart showing frequency distribution of a field.
    
    Args:
        data: List of extracted data items
        field: Key to analyze
        title: Chart title
        max_bars: Maximum number of bars to show
    
    Returns:
        Base64-encoded PNG string
    """
    # Count values
    values = []
    for item in data:
        val = item.get(field)
        if val is not None:
            if isinstance(val, list):
                values.extend([str(v) for v in val])
            else:
                values.append(str(val)[:50])

    if not values:
        return _generate_empty_chart(f"No data for field: {field}")

    counter = Counter(values)
    most_common = counter.most_common(max_bars)
    labels, counts = zip(*most_common)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(DARK_BG)
    _setup_dark_style(ax, title)

    # Truncate long labels
    labels = [l[:25] + '...' if len(l) > 25 else l for l in labels]

    bars = ax.barh(range(len(labels)), counts, color=AWE_COLORS[:len(labels)], 
                   edgecolor='none', height=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Count', fontsize=10)
    ax.invert_yaxis()

    # Add value labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                str(count), va='center', color=TEXT_COLOR, fontsize=9)

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_pie_chart(
    data: List[Dict[str, Any]],
    field: str,
    title: str = "Category Breakdown",
    max_slices: int = 8,
) -> str:
    """
    Generate a pie chart showing category breakdown.
    
    Returns:
        Base64-encoded PNG string
    """
    values = []
    for item in data:
        val = item.get(field)
        if val is not None:
            values.append(str(val)[:40])

    if not values:
        return _generate_empty_chart(f"No data for field: {field}")

    counter = Counter(values)
    most_common = counter.most_common(max_slices)
    
    # Group remaining into "Other"
    remaining = sum(counter.values()) - sum(c for _, c in most_common)
    if remaining > 0:
        most_common.append(("Other", remaining))

    labels, sizes = zip(*most_common)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    colors = AWE_COLORS[:len(labels)]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, autopct='%1.1f%%',
        colors=colors, startangle=90, pctdistance=0.75,
        wedgeprops={'edgecolor': DARK_BG, 'linewidth': 2}
    )

    for txt in autotexts:
        txt.set_color('white')
        txt.set_fontsize(9)

    ax.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5),
              facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
              fontsize=9)
    ax.set_title(title, color=TEXT_COLOR, fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_field_completeness_chart(
    data: List[Dict[str, Any]],
    title: str = "Field Completeness",
) -> str:
    """
    Generate a chart showing how complete each field is across all items.
    
    Returns:
        Base64-encoded PNG string
    """
    if not data:
        return _generate_empty_chart("No data available")

    # Count non-empty values per field
    all_fields = set()
    for item in data:
        all_fields.update(item.keys())

    field_completeness = {}
    total = len(data)
    for field_name in all_fields:
        count = sum(1 for item in data if item.get(field_name) not in (None, "", [], {}))
        field_completeness[field_name] = (count / total) * 100

    # Sort by completeness
    sorted_fields = sorted(field_completeness.items(), key=lambda x: x[1], reverse=True)[:15]
    fields, pcts = zip(*sorted_fields)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(DARK_BG)
    _setup_dark_style(ax, title)

    colors = ['#22c55e' if p >= 80 else '#f97316' if p >= 50 else '#f43f5e' for p in pcts]
    bars = ax.barh(range(len(fields)), pcts, color=colors, edgecolor='none', height=0.6)
    ax.set_yticks(range(len(fields)))
    ax.set_yticklabels(fields, fontsize=9)
    ax.set_xlabel('Completeness (%)', fontsize=10)
    ax.set_xlim(0, 110)
    ax.invert_yaxis()

    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{pct:.0f}%', va='center', color=TEXT_COLOR, fontsize=9)

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_word_frequency_chart(
    data: List[Dict[str, Any]],
    fields: Optional[List[str]] = None,
    title: str = "Top Words",
    max_words: int = 20,
) -> str:
    """
    Generate a word frequency bar chart from text fields.
    
    Returns:
        Base64-encoded PNG string
    """
    import re as regex

    # Common stop words to filter
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'shall', 'can', 'it', 'its', 'this',
        'that', 'these', 'those', 'i', 'we', 'you', 'he', 'she', 'they', 'them',
        'my', 'your', 'his', 'her', 'our', 'their', 'not', 'no', 'nor',
    }

    words = []
    for item in data:
        target_fields = fields if fields else list(item.keys())
        for f in target_fields:
            val = item.get(f)
            if isinstance(val, str):
                tokens = regex.findall(r'\b[a-zA-Z]{3,}\b', val.lower())
                words.extend([w for w in tokens if w not in stop_words])

    if not words:
        return _generate_empty_chart("No text content found")

    counter = Counter(words)
    most_common = counter.most_common(max_words)
    w_labels, w_counts = zip(*most_common)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(DARK_BG)
    _setup_dark_style(ax, title)

    gradient_colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(w_labels)))
    bars = ax.bar(range(len(w_labels)), w_counts, color=gradient_colors, edgecolor='none')
    ax.set_xticks(range(len(w_labels)))
    ax.set_xticklabels(w_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Frequency', fontsize=10)

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_summary_dashboard(
    data: List[Dict[str, Any]],
    url: str = "",
) -> str:
    """
    Generate a summary dashboard with multiple mini-charts.
    
    Returns:
        Base64-encoded PNG string
    """
    if not data:
        return _generate_empty_chart("No data available for dashboard")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(f'Extraction Dashboard{f" — {url[:50]}" if url else ""}',
                 color=TEXT_COLOR, fontsize=15, fontweight='bold', y=0.98)

    # 1. Items count by type (top-left)
    ax1 = axes[0, 0]
    _setup_dark_style(ax1, f"Total Items: {len(data)}")
    all_fields = set()
    for item in data:
        all_fields.update(item.keys())
    field_counts = {f: sum(1 for i in data if i.get(f) not in (None, "", [], {})) for f in all_fields}
    sorted_fc = sorted(field_counts.items(), key=lambda x: x[1], reverse=True)[:8]
    if sorted_fc:
        f_names, f_counts = zip(*sorted_fc)
        ax1.barh(range(len(f_names)), f_counts, color=AWE_COLORS[:len(f_names)], height=0.5)
        ax1.set_yticks(range(len(f_names)))
        ax1.set_yticklabels(f_names, fontsize=8)
        ax1.invert_yaxis()

    # 2. Field completeness (top-right)
    ax2 = axes[0, 1]
    _setup_dark_style(ax2, "Field Completeness")
    total = len(data)
    completeness = {f: (sum(1 for i in data if i.get(f) not in (None, "", [], {})) / total) * 100 
                   for f in all_fields}
    sorted_comp = sorted(completeness.items(), key=lambda x: x[1], reverse=True)[:8]
    if sorted_comp:
        c_names, c_pcts = zip(*sorted_comp)
        colors = ['#22c55e' if p >= 80 else '#f97316' if p >= 50 else '#f43f5e' for p in c_pcts]
        ax2.barh(range(len(c_names)), c_pcts, color=colors, height=0.5)
        ax2.set_yticks(range(len(c_names)))
        ax2.set_yticklabels(c_names, fontsize=8)
        ax2.set_xlim(0, 110)
        ax2.invert_yaxis()

    # 3. Data length distribution (bottom-left)
    ax3 = axes[1, 0]
    _setup_dark_style(ax3, "Content Length Distribution")
    str_fields = [f for f in all_fields if any(isinstance(i.get(f), str) for i in data)]
    if str_fields:
        lens = [len(str(data[0].get(f, ""))) for f in str_fields[:8]]
        ax3.bar(range(len(str_fields[:8])), lens, color=AWE_COLORS[:len(str_fields[:8])])
        ax3.set_xticks(range(len(str_fields[:8])))
        ax3.set_xticklabels([f[:12] for f in str_fields[:8]], rotation=30, ha='right', fontsize=8)

    # 4. Statistics text (bottom-right)
    ax4 = axes[1, 1]
    _setup_dark_style(ax4, "Quick Stats")
    ax4.axis('off')
    stats_text = [
        f"📊 Total Items: {len(data)}",
        f"📋 Fields Found: {len(all_fields)}",
        f"✅ Avg Completeness: {sum(completeness.values())/max(len(completeness), 1):.0f}%",
        f"📝 Text Fields: {len(str_fields)}",
    ]
    for i, line in enumerate(stats_text):
        ax4.text(0.1, 0.8 - i * 0.2, line, transform=ax4.transAxes,
                color=TEXT_COLOR, fontsize=12, verticalalignment='top')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return _fig_to_base64(fig)


def _generate_empty_chart(message: str) -> str:
    """Generate a placeholder chart with an error message."""
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.axis('off')
    ax.text(0.5, 0.5, message, transform=ax.transAxes,
            color=TEXT_COLOR, fontsize=14, ha='center', va='center')
    return _fig_to_base64(fig)


# =============================================================================
# Main API function
# =============================================================================

def generate_visualization(
    data: List[Dict[str, Any]],
    chart_type: str = "dashboard",
    field: Optional[str] = None,
    title: Optional[str] = None,
    url: str = "",
) -> Dict[str, Any]:
    """
    Main entry point for generating visualizations.
    
    Args:
        data: Extracted data items
        chart_type: One of 'bar', 'pie', 'completeness', 'words', 'dashboard'
        field: Field to analyze (required for bar/pie)
        title: Custom chart title
        url: Source URL for context
    
    Returns:
        Dict with chart_type, image_base64, and metadata
    """
    generators = {
        "bar": lambda: generate_bar_chart(data, field or _pick_best_field(data), title or "Field Distribution"),
        "pie": lambda: generate_pie_chart(data, field or _pick_best_field(data), title or "Category Breakdown"),
        "completeness": lambda: generate_field_completeness_chart(data, title or "Field Completeness"),
        "words": lambda: generate_word_frequency_chart(data, title=title or "Top Words"),
        "dashboard": lambda: generate_summary_dashboard(data, url),
    }

    gen = generators.get(chart_type, generators["dashboard"])
    image_b64 = gen()

    return {
        "chart_type": chart_type,
        "image_base64": image_b64,
        "items_analyzed": len(data),
        "field": field,
    }


def _pick_best_field(data: List[Dict[str, Any]]) -> str:
    """Pick the best field for charting by finding the most populated categorical field."""
    if not data:
        return ""
    
    field_scores = {}
    for field_name in data[0].keys():
        values = [item.get(field_name) for item in data if item.get(field_name)]
        if not values:
            continue
        # Prefer fields with moderate cardinality (not too unique, not too uniform)
        unique_ratio = len(set(str(v) for v in values)) / len(values)
        if 0.1 < unique_ratio < 0.9:
            field_scores[field_name] = len(values) * (1 - abs(unique_ratio - 0.5))

    if field_scores:
        return max(field_scores, key=field_scores.get)
    return list(data[0].keys())[0] if data[0] else ""


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    test_data = [
        {"name": "Alice", "department": "Engineering", "role": "Senior", "email": "alice@ex.com"},
        {"name": "Bob", "department": "Engineering", "role": "Junior", "email": "bob@ex.com"},
        {"name": "Carol", "department": "Marketing", "role": "Senior", "email": "carol@ex.com"},
        {"name": "Dave", "department": "Sales", "role": "Lead", "email": "dave@ex.com"},
        {"name": "Eve", "department": "Engineering", "role": "Junior", "email": ""},
    ]
    result = generate_visualization(test_data, "dashboard")
    print(f"Chart type: {result['chart_type']}")
    print(f"Image base64 length: {len(result['image_base64'])}")
    print("✅ Visualization generated successfully")

"""
Visualization tools for Dempster-Shafer Theory results.

This module provides functions to plot mass functions, belief/plausibility intervals,
and other relevant visualizations using matplotlib.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Set, FrozenSet, List, Tuple, Optional, Any
import sys
import os

# Add parent directory to path to import dst_core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dst_core.basic import MassFunction


def plot_mass_function(mass_function: MassFunction, title: str = "Mass Function",
                      save_path: Optional[str] = None):
    """
    Plot a mass function as a bar chart.

    Args:
        mass_function: The MassFunction object to plot.
        title: The title for the plot.
        save_path: (Optional) Path to save the plot image file. If None, displays the plot.
    """
    focal_sets = sorted(mass_function.masses.keys(), key=lambda x: (len(x), str(x)))
    masses = [mass_function.masses[fs] for fs in focal_sets]

    # Create labels for focal sets
    labels = []
    for fs in focal_sets:
        if not fs:
            labels.append("∅")
        elif fs == mass_function.frame:
            labels.append("Ω")
        else:
            labels.append("{" + ", ".join(sorted(fs)) + "}")

    x_pos = np.arange(len(labels))

    plt.figure(figsize=(max(6, len(labels) * 0.8), 4))
    bars = plt.bar(x_pos, masses, align='center', alpha=0.7)
    plt.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
    plt.xticks(x_pos, labels, rotation=45, ha='right')
    plt.ylabel('Mass')
    plt.title(title)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close() # Close the plot to free memory
        if "verbose" not in save_path: # Avoid excessive printing during main run
             print(f"Plot saved to {save_path}")
    else:
        plt.show()

def plot_belief_plausibility(mass_function: MassFunction,
                             subsets: Optional[List[Set[str]]] = None,
                             title: str = "Belief and Plausibility",
                             save_path: Optional[str] = None):
    """
    Plot belief and plausibility intervals for specified subsets.

    Args:
        mass_function: The MassFunction object.
        subsets: (Optional) A list of subsets (as sets) to plot. If None, plots for all singletons.
        title: The title for the plot.
        save_path: (Optional) Path to save the plot image file. If None, displays the plot.
    """
    if subsets is None:
        # Default to plotting singletons
        subsets = [{element} for element in mass_function.frame]

    subset_labels = []
    beliefs = []
    plausibilities = []

    for subset in sorted(subsets, key=lambda x: str(x)):
        bel = mass_function.belief(subset)
        pl = mass_function.plausibility(subset)
        beliefs.append(bel)
        plausibilities.append(pl)
        subset_labels.append("{" + ", ".join(sorted(subset)) + "}")

    y_pos = np.arange(len(subset_labels))
    interval_lengths = [pl - bel for bel, pl in zip(beliefs, plausibilities)]

    plt.figure(figsize=(6, max(4, len(subset_labels) * 0.5)))
    plt.barh(y_pos, interval_lengths, left=beliefs, height=0.6, align='center',
             color='lightblue', edgecolor='grey', label='Uncertainty [Bel, Pl]')
    plt.plot(beliefs, y_pos, '|', color='blue', markersize=10, label='Belief')
    plt.plot(plausibilities, y_pos, '|', color='red', markersize=10, label='Plausibility')

    # Add text labels for belief and plausibility values
    for i, (bel, pl) in enumerate(zip(beliefs, plausibilities)):
        plt.text(bel, y_pos[i] + 0.1, f'{bel:.3f}', va='bottom', ha='center', color='blue', fontsize=8)
        plt.text(pl, y_pos[i] + 0.1, f'{pl:.3f}', va='bottom', ha='center', color='red', fontsize=8)
    plt.yticks(y_pos, subset_labels)
    plt.xlabel('Value')
    plt.title(title)
    plt.xlim(0, 1)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close() # Close the plot to free memory
        if "verbose" not in save_path: # Avoid excessive printing during main run
            print(f"Plot saved to {save_path}")
    else:
        plt.show()

def plot_metrics_comparison(all_metrics: Dict[str, Dict[str, float]],
                           save_path: Optional[str] = None):
    """
    Plot a comparison of classification metrics across different combination rules.

    Args:
        all_metrics: Dictionary where keys are rule names (e.g., "dempster") and values
                     are dictionaries of metrics (e.g., {"accuracy": 0.9, "f1_macro": 0.85}).
        save_path: (Optional) Path to save the plot image file. If None, displays the plot.
    """
    rules = list(all_metrics.keys())
    if not rules:
        print("No metrics data to plot.")
        return

    # Extract metrics - assuming common metrics like accuracy, precision_macro, recall_macro, f1_macro
    metric_names = sorted([k for k in all_metrics[rules[0]].keys() if isinstance(all_metrics[rules[0]][k], (int, float))])
    if not metric_names:
        print("No numeric metrics found to plot.")
        return

    n_rules = len(rules)
    n_metrics = len(metric_names)

    bar_width = 0.8 / n_metrics
    index = np.arange(n_rules)

    plt.figure(figsize=(max(6, n_rules * 1.5), 5))

    for i, metric in enumerate(metric_names):
        values = [all_metrics[rule].get(metric, 0.0) for rule in rules]
        bars = plt.bar(index + i * bar_width, values, bar_width, label=metric.replace("_", " ").title())
        # Add value labels on top of bars
        plt.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)

    plt.xlabel('Combination Rule')
    plt.ylabel('Score')
    plt.title('Comparison of Classification Metrics by Combination Rule')
    plt.xticks(index + bar_width * (n_metrics - 1) / 2, [r.capitalize() for r in rules])
    plt.ylim(0, 1.1) # Extend y-axis slightly above 1.0
    plt.legend(loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close() # Close the plot to free memory
        if "verbose" not in save_path: # Avoid excessive printing during main run
            print(f"Plot saved to {save_path}")
    else:
        plt.show()

# Example Usage (can be removed or placed under if __name__ == "__main__")
# if __name__ == "__main__":
#     # Example metrics data
#     metrics_data = {
#         "dempster": {"accuracy": 0.95, "precision_macro": 0.94, "recall_macro": 0.95, "f1_macro": 0.945},
#         "yager": {"accuracy": 0.93, "precision_macro": 0.92, "recall_macro": 0.93, "f1_macro": 0.925},
#         "pcr5": {"accuracy": 0.96, "precision_macro": 0.96, "recall_macro": 0.96, "f1_macro": 0.96}
#     }
#     plot_metrics_comparison(metrics_data, save_path="/home/ubuntu/metrics_comparison_plot.png")



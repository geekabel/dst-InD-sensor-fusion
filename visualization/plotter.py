"""
Visualization tools for Dempster-Shafer Theory results.

This module provides functions to plot mass functions, belief/plausibility intervals,
and other relevant visualizations using matplotlib.
"""

import matplotlib.pyplot as plt
import numpy as np
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
    plt.bar(x_pos, masses, align='center', alpha=0.7)
    plt.xticks(x_pos, labels, rotation=45, ha='right')
    plt.ylabel('Mass')
    plt.title(title)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close() # Close the plot to free memory
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
    # Plot the interval bars (Plausibility - Belief)
    plt.barh(y_pos, interval_lengths, left=beliefs, height=0.6, align='center', 
             color='lightblue', edgecolor='grey', label='Uncertainty [Bel, Pl]')
    # Plot belief markers
    plt.plot(beliefs, y_pos, '|', color='black', markersize=10, label='Belief')
    # Plot plausibility markers
    plt.plot(plausibilities, y_pos, '|', color='black', markersize=10, label='Plausibility')
    
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
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

# Example Usage (can be removed or placed under if __name__ == "__main__")
# if __name__ == "__main__":
#     frame = {"A", "B", "C"}
#     m1_dict = {
#         frozenset({"A"}): 0.6,
#         frozenset({"B", "C"}): 0.3,
#         frozenset(frame): 0.1
#     }
#     m1 = MassFunction(frame, m1_dict)
#     
#     print(m1)
#     plot_mass_function(m1, title="Example Mass Function m1", save_path="/home/ubuntu/m1_plot.png")
#     plot_belief_plausibility(m1, title="Belief/Plausibility for m1 Singletons", save_path="/home/ubuntu/m1_bel_pl_plot.png")
#     
#     # Example with specific subsets
#     plot_belief_plausibility(m1, subsets=[{"A"}, {"B"}, {"A", "B"}], title="Belief/Plausibility for Specific Subsets", save_path="/home/ubuntu/m1_bel_pl_specific_plot.png")

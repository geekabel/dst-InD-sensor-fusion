"""
Module for handling the classification task using Dempster-Shafer fusion results.

This includes decision-making strategies based on fused belief functions and
functions for evaluating the classification performance against ground truth.
"""

import numpy as np
import pandas as pd
from typing import Dict, Set, FrozenSet, List, Tuple, Optional, Any, Literal
from collections import Counter
import sys
import os

# Add parent directory to path to import dst_core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dst_core.basic import MassFunction


DecisionMethod = Literal["max_bel", "max_pl", "max_pignistic"]


def make_decision(mass_function: MassFunction, method: DecisionMethod = "max_pignistic") -> Optional[str]:
    """
    Make a classification decision based on the fused mass function.

    Args:
        mass_function: The fused MassFunction object.
        method: The decision-making method to use (
            "max_bel": Maximum belief on singletons,
            "max_pl": Maximum plausibility on singletons,
            "max_pignistic": Maximum pignistic probability on singletons
        ).

    Returns:
        The predicted class label (str) or None if no decision can be made.
    """
    singletons = {frozenset({element}) for element in mass_function.frame}
    
    if method == "max_bel":
        max_singleton = None
        max_belief = -1.0
        for singleton in singletons:
            belief = mass_function.belief(singleton)
            # Handle ties by returning None or choosing arbitrarily (here, first one)
            if belief > max_belief:
                max_belief = belief
                max_singleton = singleton
            elif belief == max_belief:
                # In case of a tie, we might return None or choose one
                # For simplicity, let's return the first one found
                pass 
        
        return list(max_singleton)[0] if max_singleton else None

    elif method == "max_pl":
        max_singleton = None
        max_plausibility = -1.0
        for singleton in singletons:
            plausibility = mass_function.plausibility(singleton)
            # Handle ties
            if plausibility > max_plausibility:
                max_plausibility = plausibility
                max_singleton = singleton
            elif plausibility == max_plausibility:
                pass # Return first one found
                
        return list(max_singleton)[0] if max_singleton else None

    elif method == "max_pignistic":
        pignistic_probs = mass_function.pignistic_transformation()
        if not pignistic_probs:
            return None
        # Find the element with the maximum probability
        max_element = max(pignistic_probs.items(), key=lambda item: item[1])
        # Check for ties - if multiple elements have the max probability, return None or choose one
        max_prob = max_element[1]
        tied_elements = [el for el, prob in pignistic_probs.items() if np.isclose(prob, max_prob)]
        if len(tied_elements) > 1:
             # Handle ties, e.g., return None or the first one
             return tied_elements[0] # Return the first one for simplicity
        return max_element[0]
    
    else:
        raise ValueError(f"Unknown decision method: {method}")


def calculate_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    """
    Calculate the classification accuracy.

    Args:
        predictions: List of predicted class labels.
        ground_truth: List of true class labels.

    Returns:
        Accuracy value (float between 0 and 1).
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Length of predictions and ground truth must match.")
    if not predictions: # Avoid division by zero
        return 0.0
        
    correct_count = sum(1 for pred, true in zip(predictions, ground_truth) if pred == true)
    return correct_count / len(predictions)


def calculate_confusion_matrix(predictions: List[str], ground_truth: List[str], 
                              labels: List[str]) -> pd.DataFrame:
    """
    Calculate the confusion matrix.

    Args:
        predictions: List of predicted class labels.
        ground_truth: List of true class labels.
        labels: List of all possible class labels in the desired order.

    Returns:
        A pandas DataFrame representing the confusion matrix.
        Rows represent true labels, columns represent predicted labels.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Length of predictions and ground truth must match.")
        
    matrix = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    
    for true, pred in zip(ground_truth, predictions):
        if true in labels and pred in labels:
            matrix.loc[true, pred] += 1
            
    return matrix


def calculate_precision_recall_f1(confusion_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate precision, recall, and F1-score for each class from a confusion matrix.

    Args:
        confusion_matrix: A pandas DataFrame confusion matrix.

    Returns:
        A pandas DataFrame with precision, recall, and F1-score for each class.
    """
    labels = confusion_matrix.index.tolist()
    metrics = pd.DataFrame(index=labels, columns=["Precision", "Recall", "F1-Score"], dtype=float)
    
    for label in labels:
        true_positives = confusion_matrix.loc[label, label]
        predicted_positives = confusion_matrix[label].sum()
        actual_positives = confusion_matrix.loc[label].sum()
        
        precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
        recall = true_positives / actual_positives if actual_positives > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics.loc[label, "Precision"] = precision
        metrics.loc[label, "Recall"] = recall
        metrics.loc[label, "F1-Score"] = f1
        
    # Calculate macro averages
    metrics.loc["Macro Avg"] = metrics.mean()
    
    return metrics

# Example Usage
# if __name__ == "__main__":
#     # Example Mass Function
#     frame = {"car", "pedestrian", "bicycle"}
#     m_dict = {
#         frozenset({"car"}): 0.5,
#         frozenset({"pedestrian"}): 0.2,
#         frozenset({"car", "pedestrian"}): 0.1,
#         frozenset(frame): 0.2
#     }
#     m = MassFunction(frame, m_dict)
#     
#     # Decision Making
#     decision_bel = make_decision(m, method="max_bel")
#     decision_pl = make_decision(m, method="max_pl")
#     decision_pig = make_decision(m, method="max_pignistic")
#     print(f"Decision (Max Bel): {decision_bel}")
#     print(f"Decision (Max Pl): {decision_pl}")
#     print(f"Decision (Max Pignistic): {decision_pig}")
#     
#     # Evaluation Example
#     preds = ["car", "pedestrian", "car", "bicycle", "pedestrian"]
#     truths = ["car", "car", "car", "bicycle", "pedestrian"]
#     all_labels = ["car", "pedestrian", "bicycle"]
#     
#     accuracy = calculate_accuracy(preds, truths)
#     print(f"\nAccuracy: {accuracy:.4f}")
#     
#     conf_matrix = calculate_confusion_matrix(preds, truths, all_labels)
#     print("\nConfusion Matrix:")
#     print(conf_matrix)
#     
#     perf_metrics = calculate_precision_recall_f1(conf_matrix)
#     print("\nPerformance Metrics:")
#     print(perf_metrics)


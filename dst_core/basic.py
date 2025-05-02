"""
Basic implementation of Dempster-Shafer Theory for belief functions.

This module provides the core functionality for Dempster-Shafer Theory (DST),
including basic belief assignment (BBA) operations, combination rules,
and belief/plausibility calculations.

References:
    - Shafer, G. (1976). A Mathematical Theory of Evidence. Princeton University Press.
    - Smets, P. (1990). The combination of evidence in the transferable belief model.
      IEEE Transactions on Pattern Analysis and Machine Intelligence, 12(5), 447-458.
"""

import math
import copy
import itertools
from typing import Dict, Set, FrozenSet, List, Tuple, Union, Optional, Any


class MassFunction:
    """
    A class representing a mass function (Basic Belief Assignment) in Dempster-Shafer Theory.

    A mass function assigns belief masses to subsets of the frame of discernment.
    The sum of all masses must equal 1.
    """

    def __init__(self, frame_of_discernment: Set[str], mass_dict: Optional[Dict[FrozenSet[str], float]] = None):
        """
        Initialize a mass function with a frame of discernment and optional mass dictionary.

        Args:
            frame_of_discernment: Set of all possible hypotheses (the universe)
            mass_dict: Dictionary mapping subsets (as frozensets) to mass values
        """
        self.frame = frozenset(frame_of_discernment)
        self.masses = {}

        # Initialize with empty mass function if no dictionary provided
        if mass_dict is None:
            # By default, assign all mass to the universal set (total ignorance)
            self.masses[self.frame] = 1.0
        else:
            # Validate and copy the provided mass dictionary
            total_mass = 0.0
            for subset, mass in mass_dict.items():
                if not isinstance(subset, frozenset):
                    subset = frozenset(subset)

                # Ensure subset is valid (contained in frame)
                if not subset.issubset(self.frame):
                    raise ValueError(f"Subset {subset} is not contained in the frame of discernment {self.frame}")

                # Ensure mass is valid
                if mass < 0:
                    raise ValueError(f"Mass value {mass} for subset {subset} is negative")

                if mass > 0:  # Only store non-zero masses
                    self.masses[subset] = mass
                    total_mass += mass

            # Check if masses sum to 1 (allowing for small floating-point errors)
            if not math.isclose(total_mass, 1.0, abs_tol=1e-10):
                raise ValueError(f"Mass values must sum to 1.0, got {total_mass}")

    def __str__(self) -> str:
        """String representation of the mass function."""
        result = "Mass Function:\n"
        for subset, mass in sorted(self.masses.items(), key=lambda x: (len(x[0]), str(x[0]))):
            subset_str = "{" + ", ".join(sorted(subset)) + "}" if subset else "∅"
            result += f"m({subset_str}) = {mass:.4f}\n"
        return result

    def focal_sets(self) -> List[FrozenSet[str]]:
        """Return the list of focal sets (sets with non-zero mass)."""
        return list(self.masses.keys())

    def belief(self, subset: Union[Set[str], FrozenSet[str]]) -> float:
        """
        Calculate the belief value for a subset.

        Belief is the sum of masses of all subsets that are contained in the given subset.

        Args:
            subset: The subset to calculate belief for

        Returns:
            The belief value for the subset
        """
        if not isinstance(subset, frozenset):
            subset = frozenset(subset)

        return sum(mass for s, mass in self.masses.items() if s and s.issubset(subset))

    def plausibility(self, subset: Union[Set[str], FrozenSet[str]]) -> float:
        """
        Calculate the plausibility value for a subset.

        Plausibility is the sum of masses of all subsets that have non-empty intersection
        with the given subset.

        Args:
            subset: The subset to calculate plausibility for

        Returns:
            The plausibility value for the subset
        """
        if not isinstance(subset, frozenset):
            subset = frozenset(subset)

        return sum(mass for s, mass in self.masses.items() if s and s.intersection(subset))

    def uncertainty_interval(self, subset: Union[Set[str], FrozenSet[str]]) -> Tuple[float, float]:
        """
        Calculate the uncertainty interval [Belief, Plausibility] for a subset.

        Args:
            subset: The subset to calculate the uncertainty interval for

        Returns:
            A tuple (belief, plausibility) representing the uncertainty interval
        """
        return (self.belief(subset), self.plausibility(subset))

    def combine_dempster(self, other: 'MassFunction') -> 'MassFunction':
        """
        Combine this mass function with another using Dempster's rule of combination.

        Dempster's rule is a normalized conjunctive combination that redistributes
        the mass of the empty set (conflict) proportionally among other focal sets.

        Args:
            other: Another mass function to combine with

        Returns:
            A new mass function representing the combination

        Raises:
            ValueError: If the frames of discernment are different or if there's complete conflict
        """
        # Check if frames are compatible
        if self.frame != other.frame:
            raise ValueError("Cannot combine mass functions with different frames of discernment")

        # Calculate combined masses
        combined_masses = {}
        conflict = 0.0

        # For each pair of focal sets from both mass functions
        for s1, m1 in self.masses.items():
            for s2, m2 in other.masses.items():
                # Calculate intersection
                intersection = s1.intersection(s2)

                # Calculate mass product
                mass_product = m1 * m2

                if not intersection:
                    # This is a conflicting combination
                    conflict += mass_product
                else:
                    # Add to the combined mass
                    if intersection in combined_masses:
                        combined_masses[intersection] += mass_product
                    else:
                        combined_masses[intersection] = mass_product

        # Check for complete conflict
        if math.isclose(conflict, 1.0, abs_tol=1e-10):
            raise ValueError("Cannot combine completely conflicting mass functions using Dempster's rule")

        # Normalize the combined masses
        if not math.isclose(conflict, 0.0, abs_tol=1e-10):
            normalization_factor = 1.0 / (1.0 - conflict)
            for subset in combined_masses:
                combined_masses[subset] *= normalization_factor

        return MassFunction(self.frame, combined_masses)

    def combine_yager(self, other: 'MassFunction') -> 'MassFunction':
        """
        Combine this mass function with another using Yager's rule of combination.

        Yager's rule assigns the mass of the empty set (conflict) to the universal set
        instead of normalizing.

        Args:
            other: Another mass function to combine with

        Returns:
            A new mass function representing the combination

        Raises:
            ValueError: If the frames of discernment are different
        """
        # Check if frames are compatible
        if self.frame != other.frame:
            raise ValueError("Cannot combine mass functions with different frames of discernment")

        # Calculate combined masses
        combined_masses = {}
        conflict = 0.0

        # For each pair of focal sets from both mass functions
        for s1, m1 in self.masses.items():
            for s2, m2 in other.masses.items():
                # Calculate intersection
                intersection = s1.intersection(s2)

                # Calculate mass product
                mass_product = m1 * m2

                if not intersection:
                    # This is a conflicting combination
                    conflict += mass_product
                else:
                    # Add to the combined mass
                    if intersection in combined_masses:
                        combined_masses[intersection] += mass_product
                    else:
                        combined_masses[intersection] = mass_product

        # Assign conflict to the universal set (Yager's approach)
        if not math.isclose(conflict, 0.0, abs_tol=1e-10):
            if self.frame in combined_masses:
                combined_masses[self.frame] += conflict
            else:
                combined_masses[self.frame] = conflict

        return MassFunction(self.frame, combined_masses)

    def combine_pcr5(self, other: 'MassFunction') -> 'MassFunction':
        """
        Combine this mass function with another using PCR5 rule (Proportional Conflict Redistribution).

        PCR5 redistributes the conflict proportionally to the masses of the elements involved
        in the conflict.

        Args:
            other: Another mass function to combine with

        Returns:
            A new mass function representing the combination

        Raises:
            ValueError: If the frames of discernment are different
        """
        # Check if frames are compatible
        if self.frame != other.frame:
            raise ValueError("Cannot combine mass functions with different frames of discernment")

        # Calculate combined masses using conjunctive rule
        combined_masses = {}

        # For each pair of focal sets from both mass functions
        for s1, m1 in self.masses.items():
            for s2, m2 in other.masses.items():
                # Calculate intersection
                intersection = s1.intersection(s2)

                # Calculate mass product
                mass_product = m1 * m2

                if not intersection:
                    # This is a conflicting combination
                    # In PCR5, we redistribute this conflict proportionally
                    # to the masses of the elements involved
                    if m1 + m2 > 0:
                        factor1 = (m1**2 * m2) / (m1 + m2)
                        factor2 = (m2**2 * m1) / (m1 + m2)

                        # Add to the combined masses
                        if s1 in combined_masses:
                            combined_masses[s1] += factor1
                        else:
                            combined_masses[s1] = factor1

                        if s2 in combined_masses:
                            combined_masses[s2] += factor2
                        else:
                            combined_masses[s2] = factor2
                else:
                    # Add to the combined mass
                    if intersection in combined_masses:
                        combined_masses[intersection] += mass_product
                    else:
                        combined_masses[intersection] = mass_product

        return MassFunction(self.frame, combined_masses)

    def discount(self, reliability: float) -> 'MassFunction':
        """
        Apply Shafer's classical discounting to this mass function.

        Discounting reduces the mass of each focal set by a factor (reliability)
        and assigns the remaining mass to the universal set.

        Args:
            reliability: A value between 0 and 1 representing the reliability of the source

        Returns:
            A new discounted mass function

        Raises:
            ValueError: If reliability is not between 0 and 1
        """
        if not 0 <= reliability <= 1:
            raise ValueError("Reliability must be between 0 and 1")

        if reliability == 1:
            # No discounting needed
            return copy.deepcopy(self)

        if reliability == 0:
            # Complete discounting, return vacuous mass function
            return MassFunction(self.frame)

        # Apply discounting
        discounted_masses = {}
        universal_mass = 0.0

        for subset, mass in self.masses.items():
            if subset == self.frame:
                # Universal set gets special treatment
                universal_mass += mass
            else:
                # Discount other focal sets
                discounted_mass = reliability * mass
                if not math.isclose(discounted_mass, 0.0, abs_tol=1e-10):
                    discounted_masses[subset] = discounted_mass

                # Add remaining mass to universal set
                universal_mass += (1 - reliability) * mass

        # Add universal set mass
        if not math.isclose(universal_mass, 0.0, abs_tol=1e-10):
            discounted_masses[self.frame] = universal_mass

        return MassFunction(self.frame, discounted_masses)

    def contextual_discount(self, discount_factors: Dict[FrozenSet[str], float]) -> 'MassFunction':
        """
        Apply contextual discounting to this mass function.

        Contextual discounting allows different reliability factors for different subsets
        of the frame of discernment.

        Args:
            discount_factors: Dictionary mapping subsets to reliability factors (0 to 1)

        Returns:
            A new contextually discounted mass function

        Raises:
            ValueError: If any reliability factor is not between 0 and 1
        """
        # Validate discount factors
        for subset, factor in discount_factors.items():
            if not 0 <= factor <= 1:
                raise ValueError(f"Reliability factor {factor} for subset {subset} must be between 0 and 1")

        # Start with a copy of the original mass function
        result = copy.deepcopy(self)

        # Apply each discount factor
        for subset, factor in discount_factors.items():
            if not isinstance(subset, frozenset):
                subset = frozenset(subset)

            # Create a simple mass function for this subset
            simple_mass = {
                subset: factor,
                self.frame: 1 - factor
            }
            simple_bba = MassFunction(self.frame, simple_mass)

            # Combine with the result so far
            result = result.combine_dempster(simple_bba)

        return result

    def pignistic_transformation(self) -> Dict[str, float]:
        """
        Calculate the pignistic transformation of this mass function.

        The pignistic transformation converts a mass function to a probability
        distribution by distributing the mass of each focal set equally among
        its elements.

        Returns:
            A dictionary mapping singleton elements to their pignistic probabilities
        """
        result = {element: 0.0 for element in self.frame}

        for subset, mass in self.masses.items():
            if not subset:  # Skip empty set
                continue

            # Distribute mass equally among elements
            share = mass / len(subset)
            for element in subset:
                result[element] += share

        return result

    def max_bel(self) -> Tuple[FrozenSet[str], float]:
        """
        Find the subset with the maximum belief.

        Returns:
            A tuple (subset, belief_value) for the subset with maximum belief
        """
        max_subset = None
        max_belief = -1.0

        for subset in self.focal_sets():
            belief = self.belief(subset)
            if belief > max_belief:
                max_belief = belief
                max_subset = subset

        return (max_subset, max_belief)

    def max_pl(self) -> Tuple[FrozenSet[str], float]:
        """
        Find the subset with the maximum plausibility.

        Returns:
            A tuple (subset, plausibility_value) for the subset with maximum plausibility
        """
        max_subset = None
        max_plausibility = -1.0

        for subset in self.focal_sets():
            plausibility = self.plausibility(subset)
            if plausibility > max_plausibility:
                max_plausibility = plausibility
                max_subset = subset

        return (max_subset, max_plausibility)

    def max_pignistic(self) -> Tuple[str, float]:
        """
        Find the singleton element with the maximum pignistic probability.

        Returns:
            A tuple (element, probability) for the element with maximum pignistic probability
        """
        pignistic = self.pignistic_transformation()
        max_element = max(pignistic.items(), key=lambda x: x[1])
        return max_element


def create_simple_mass_function(frame: Set[str],
                               focal_element: Set[str],
                               mass_value: float) -> MassFunction:
    """
    Create a simple mass function with one focal element and the universal set.

    Args:
        frame: The frame of discernment
        focal_element: The focal element to assign mass to
        mass_value: The mass to assign to the focal element

    Returns:
        A simple mass function
    """
    if not 0 <= mass_value <= 1:
        raise ValueError("Mass value must be between 0 and 1")

    focal_element = frozenset(focal_element)

    # Create mass dictionary
    mass_dict = {
        focal_element: mass_value,
        frozenset(frame): 1 - mass_value
    }

    # If mass_value is 1, remove the universal set
    if math.isclose(mass_value, 1.0, abs_tol=1e-10):
        del mass_dict[frozenset(frame)]

    return MassFunction(frame, mass_dict)


def create_categorical_mass_function(frame: Set[str],
                                    element: Union[str, Set[str]]) -> MassFunction:
    """
    Create a categorical mass function that assigns all mass to one element.

    Args:
        frame: The frame of discernment
        element: The element to assign all mass to (can be a singleton or a subset)

    Returns:
        A categorical mass function
    """
    if isinstance(element, str):
        element = {element}

    return create_simple_mass_function(frame, element, 1.0)


def create_vacuous_mass_function(frame: Set[str]) -> MassFunction:
    """
    Create a vacuous mass function that assigns all mass to the universal set.

    Args:
        frame: The frame of discernment

    Returns:
        A vacuous mass function
    """
    return MassFunction(frame)

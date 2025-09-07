# -*- coding: utf-8 -*-
"""
src/simulation/policy.py

This module implements the Descriptive Policy Function π(·|S_t), which maps a given
system state to a probability distribution over the discrete action space.
Currently, it uses a heuristic-based approach as defined in the data_generation.py script.
"""

import numpy as np
from typing import List, Tuple

# Import project-specific modules
from src import config
from src.core import SystemState, Action


class DescriptivePolicyFunction:
    """
    Implements the descriptive policy function π(·|S_t).
    This function models the decision-making logic of the kNN-MT system, determining
    which action to take based on the current system state.
    Corresponds to Section 4.2 in the theoretical framework.
    """

    def __init__(self):
        """
        Initializes the policy function.
        It loads the action space and heuristic parameters from the central config file.
        """
        print("[INFO] Initializing Descriptive Policy Function...")
        self.action_space: List[Action] = config.ACTION_SPACE
        self.num_actions: int = len(self.action_space)

        # Load heuristic thresholds from config
        policy_params = config.POLICY_PARAMS
        self.high_pressure_threshold = policy_params["high_pressure_threshold"]
        self.medium_pressure_threshold = policy_params["medium_pressure_threshold"]
        self.uncertainty_threshold = policy_params["uncertainty_threshold"]
        self.confidence_threshold = policy_params["confidence_threshold"]
        
        print(f"\t- Policy type: Heuristic-based")
        print(f"\t- Action space size: {self.num_actions}")
        print(f"\t- High pressure threshold: {self.high_pressure_threshold}")

    def compute_action_probabilities(self, system_state: SystemState) -> np.ndarray:
        """
        Computes the probability distribution over the action space for a given state.
        This implementation uses a set of heuristic rules based on state vector norms
        and context metrics, directly mirroring the logic from data_generation.py.

        Args:
            system_state (SystemState): The current state of the system, S_t.

        Returns:
            np.ndarray: A probability vector of shape (num_actions,).
        """
        pressure_norm = system_state.pressure_state.norm()
        uncertainty = system_state.context_state.predictive_uncertainty
        confidence = system_state.context_state.predictive_confidence

        # Heuristic rule-based policy:
        # These rules define the system's reactive behavior to different conditions.
        if pressure_norm > self.high_pressure_threshold:

            # High pressure: strongly prefer low-cost actions (skip or fast retrieval).
            probs = np.array([0.7, 0.2, 0.1, 0.0, 0.0])
        elif pressure_norm > self.medium_pressure_threshold:
            # Medium-high pressure: lean towards low-cost actions.
            probs = np.array([0.4, 0.3, 0.2, 0.1, 0.0])
        elif uncertainty > self.uncertainty_threshold and pressure_norm < 0.3:
            # Low pressure but high uncertainty: model is confused, so prefer
            # expensive but accurate retrieval actions to correct potential errors.
            probs = np.array([0.1, 0.1, 0.2, 0.3, 0.3])
        elif confidence < self.confidence_threshold:
            # Low confidence: model is not sure about its top choice, use kNN to help.
            probs = np.array([0.2, 0.2, 0.3, 0.2, 0.1])
        else:
            # Normal conditions: a balanced mix of actions.
            probs = np.array([0.3, 0.3, 0.2, 0.15, 0.05])

        # Ensure the probability distribution is valid (sums to 1)
        if len(probs) != self.num_actions:
            # Fallback in case of mismatch between rules and ACTION_SPACE size
            probs = np.ones(self.num_actions)
        
        return probs / np.sum(probs)

    def sample_action(self, system_state: SystemState) -> Tuple[Action, int]:
        """
        Samples an action from the policy distribution for a given state.

        Args:
            system_state (SystemState): The current state of the system, S_t.

        Returns:
            Tuple[Action, int]: A tuple containing the selected Action object and its index.
        """
        # Get the probability distribution over actions
        probabilities = self.compute_action_probabilities(system_state)
        
        # Sample an action index based on the distribution
        action_idx = np.random.choice(self.num_actions, p=probabilities)
        
        # Return the corresponding Action object and its index
        return self.action_space[action_idx], action_idx

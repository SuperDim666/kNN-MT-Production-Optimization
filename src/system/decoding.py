# -*- coding: utf-8 -*-
"""
src/system/decoding.py

Encapsulates the generic decoding algorithms used in the kNN-MT system,
primarily the beam search implementation. This module separates the search
strategy from the core translation model logic.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Callable, Optional

# Import core data structures for type hinting.
from src.core import GenerativeContextVector, ErrorStateVector

@dataclass
class BeamHypothesis:
    """
    Represents a single hypothesis (a potential translation path) within the beam
    search algorithm. It tracks the sequence of generated tokens, its cumulative
    probability, and the history of all relevant state vectors.

    Attributes:
        tokens (List[int]): The sequence of generated token IDs.
        log_prob (float): The cumulative log probability of this token sequence.
        hidden_states (List[torch.Tensor]): History of decoder hidden states for each step.
        context_states (List[GenerativeContextVector]): History of generative context vectors.
        error_states (List[ErrorStateVector]): History of error state vectors.
        query_embeddings (List[np.ndarray]): History of embeddings used for kNN queries.
    """
    tokens: List[int]
    log_prob: float

    # Use field(default_factory=list) to ensure each instance gets a new list,
    # avoiding issues with mutable default arguments.
    hidden_states: List[torch.Tensor] = field(default_factory=list)
    context_states: List[GenerativeContextVector] = field(default_factory=list)
    error_states: List[ErrorStateVector] = field(default_factory=list)
    query_embeddings: List[np.ndarray] = field(default_factory=list)
    
    # Optional field for past key values, useful for models that support caching
    past_key_values: Optional[tuple] = None 

    def __lt__(self, other: 'BeamHypothesis') -> bool:
        """
        Comparison method for use in priority queues.
        Note: We want to find the hypothesis with the highest log_prob, so the
        comparison is inverted for min-heap implementations that pop the smallest item.
        """
        return self.log_prob < other.log_prob

    def score(self, length_penalty: float = 1.0) -> float:
        """
        Calculates the length-normalized score for the hypothesis.
        This helps to prevent the search from favoring shorter sequences.

        Args:
            length_penalty (float): Factor to penalize longer sequences. Often denoted as
                                  alpha in literature.

        Returns:
            float: The normalized score.
        """
        length = len(self.tokens)
        if length == 0:
            return -float('inf')
        return self.log_prob / (length ** length_penalty)


class BeamSearchDecoder:
    """
    A generic beam search decoder.
    It orchestrates the step-by-step expansion of hypotheses and manages the active
    and completed beams. The actual logic for a single decoding step is provided
    by an external `step_function`, making this decoder highly modular.
    """

    def __init__(self, beam_size: int, eos_token_id: int, pad_token_id: int,
                 length_penalty: float = 1.0, early_stopping: bool = True):
        """
        Initializes the BeamSearchDecoder.

        Args:
            beam_size (int): The number of hypotheses to maintain (k in beam search).
            eos_token_id (int): The token ID for the end-of-sentence token.
            pad_token_id (int): The token ID for padding.
            length_penalty (float): The length penalty factor.
            early_stopping (bool): If True, stop when `beam_size` completed hypotheses are found.
        """
        self.beam_size = beam_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping

    def search(self,
               initial_beams: List[BeamHypothesis],
               step_function: Callable[[List[BeamHypothesis], int], List[BeamHypothesis]],
               max_length: int) -> List[BeamHypothesis]:
        """
        Executes the beam search algorithm.

        Args:
            initial_beams (List[BeamHypothesis]): The starting beams, typically one empty
                                                  hypothesis at the beginning.
            step_function (Callable): A function that performs a single decoding step.
                                      Its signature must be:
                                      step_function(beams_at_t, step_t) -> beams_at_t+1
            max_length (int): The maximum number of tokens to generate to prevent infinite loops.

        Returns:
            A list of completed hypotheses, sorted by their final score in descending order.
        """
        active_beams = initial_beams
        completed_hypotheses: List[BeamHypothesis] = []

        for step in range(max_length):
            if not active_beams:
                break  # Stop if there are no more active hypotheses to expand

            # Execute one step of decoding using the provided function
            active_beams = step_function(active_beams, step)

            # Separate completed hypotheses from still-active ones
            new_active_beams: List[BeamHypothesis] = []
            for beam in active_beams:
                # A hypothesis is considered complete if its last generated token is the EOS token
                if beam.tokens[-1] == self.eos_token_id:
                    completed_hypotheses.append(beam)
                else:
                    new_active_beams.append(beam)

            active_beams = new_active_beams

            # Check for the early stopping condition
            if self.early_stopping and len(completed_hypotheses) >= self.beam_size:
                break

        # If any beams were still active when max_length was reached, they are also considered "complete"
        completed_hypotheses.extend(active_beams)

        # Sort all completed hypotheses by their final length-normalized score
        completed_hypotheses.sort(key=lambda h: h.score(self.length_penalty), reverse=True)

        # Return the top `beam_size` best hypotheses found
        return completed_hypotheses[:self.beam_size]

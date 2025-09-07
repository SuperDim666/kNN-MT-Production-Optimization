# -*- coding: utf-8 -*-
"""
src/core/structs.py

Defines the core data structures for the PAEC (Production-Aware Exposure Compensation) project.
These classes are direct Python implementations of the mathematical constructs defined in the
research's theoretical framework document (definition_framework_v2.md).
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum

# ==============================================================================
# 1. State Space Vector Components (S_t)
# ==============================================================================

@dataclass
class ErrorStateVector:
    """
    Represents the Error State Vector E_t in R^3.
    Quantifies the deviation of the current partial translation from an ideal one.
    Corresponds to Section 3.1 in the theoretical framework.

    Attributes:
        semantic_drift (float): ε_t^(sem) - Semantic deviation from the source. Range: [0, 2].
        fluency_degradation (float): ε_t^(flu) - Linguistic incoherence of the generated sequence. Range: [0, E[-log(P_{min})]).
        faithfulness_mismatch (float): ε_t^(fth) - Failure to translate key factual entities. Range: [0, 1].
    """
    semantic_drift: float
    fluency_degradation: float
    faithfulness_mismatch: float

    def to_vector(self) -> np.ndarray:
        """Converts the dataclass to a NumPy vector."""
        return np.array([
            self.semantic_drift,
            self.fluency_degradation,
            self.faithfulness_mismatch
        ], dtype=np.float32)

    def norm(self) -> float:
        """Calculates the L2 norm (Euclidean magnitude) of the error vector."""
        return np.linalg.norm(self.to_vector())

@dataclass
class ResourcePressureVector:
    """
    Represents the Resource Pressure State Vector Φ_t in (0, 1)^3.
    Quantifies the normalized, dynamic pressure on the system's computational resources.
    Corresponds to Section 3.2 in the theoretical framework.

    Attributes:
        latency_pressure (float): φ_t^(lat) - Pressure from response time.
        memory_pressure (float): φ_t^(mem) - Pressure from memory consumption.
        throughput_pressure (float): φ_t^(thr) - Pressure from request rate.
    """
    latency_pressure: float
    memory_pressure: float
    throughput_pressure: float

    def to_vector(self) -> np.ndarray:
        """Converts the dataclass to a NumPy vector."""
        return np.array([
            self.latency_pressure,
            self.memory_pressure,
            self.throughput_pressure
        ], dtype=np.float32)

    def norm(self) -> float:
        """Calculates the L2 norm of the pressure vector."""
        return np.linalg.norm(self.to_vector())

@dataclass
class GenerativeContextVector:
    """
    Represents the Generative Context State Vector H_t in R^3.
    A low-dimensional projection of the NMT model's internal state, capturing
    information relevant to the immediate generation quality.
    Corresponds to Section 3.3 in the theoretical framework.

    Attributes:
        predictive_uncertainty (float): H_t^(unc) - Shannon entropy of the next-token distribution.
        predictive_confidence (float): H_t^(cnf) - Maximum probability in the next-token distribution.
        retrieval_relevance (float): H_t^(rel) - Quality of available information in the kNN datastore.
    """
    predictive_uncertainty: float
    predictive_confidence: float
    retrieval_relevance: float

    def to_vector(self) -> np.ndarray:
        """Converts the dataclass to a NumPy vector."""
        return np.array([
            self.predictive_uncertainty,
            self.predictive_confidence,
            self.retrieval_relevance
        ], dtype=np.float32)

# ==============================================================================
# 2. Total System State Vector (S_t)
# ==============================================================================

@dataclass
class SystemState:
    """
    Represents the total system state vector S_t = (E_t, Φ_t, H_t).
    This vector fully encapsulates the system's state at a discrete time step t.
    Corresponds to Section 2 in the theoretical framework.

    Attributes:
        error_state (ErrorStateVector): The error component E_t.
        pressure_state (ResourcePressureVector): The resource pressure component Φ_t.
        context_state (GenerativeContextVector): The generative context component H_t.
        timestamp (float): The time at which the state was recorded.
    """
    error_state: ErrorStateVector
    pressure_state: ResourcePressureVector
    context_state: GenerativeContextVector
    timestamp: float

    def to_vector(self) -> np.ndarray:
        """
        Concatenates all component vectors into a single 9-dimensional NumPy vector.
        This unified vector is suitable for input to machine learning models (e.g., the
        dynamics model T or the policy function π).
        """
        return np.concatenate([
            self.error_state.to_vector(),
            self.pressure_state.to_vector(),
            self.context_state.to_vector()
        ], axis=0)

# ==============================================================================
# 3. Action Space and Decoding Strategy
# ==============================================================================

@dataclass
class Action:
    """
    Represents a single action a_i from the discrete Action Space A.
    An action is a triplet that uniquely defines a kNN retrieval and fusion operation.
    Corresponds to Section 4.1 in the theoretical framework.

    Attributes:
        k (int): The number of nearest neighbors to retrieve. k=0 signifies skipping retrieval.
        index_type (str): The type of FAISS index to use ('none', 'hnsw', 'ivf_pq', 'exact').
        lambda_weight (float): The interpolation weight for combining kNN and NMT distributions.
    """
    k: int
    index_type: str
    lambda_weight: float

    def __post_init__(self):
        """
        Ensures logical consistency after initialization. If k is 0, no retrieval
        is performed, so index_type and lambda_weight should be set to 'none' and 0.0
        respectively.
        """
        if self.k == 0:
            self.index_type = 'none'
            self.lambda_weight = 0.0

class DecodingStrategy(Enum):
    """
    Enumeration for the different decoding strategies to be tested in experiments.
    This allows for type-safe and clear selection of the generation method.
    """
    BEAM_SEARCH = "beam_search"
    # Future strategies like NUCLEUS_SAMPLING could be added here.

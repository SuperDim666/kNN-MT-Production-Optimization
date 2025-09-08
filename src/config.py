# -*- coding: utf-8 -*-
"""
src/config.py

Central configuration file for the PAEC (Production-Aware Exposure Compensation) project.
This file consolidates all hyperparameters, model identifiers, file paths, and simulation
parameters to facilitate easy management and experimentation.
"""

import os
from pathlib import Path
import hashlib
import json
from collections import OrderedDict
from src.core import Action

# --- Project Directory Structure ---
# Defines the root directory of the project to construct absolute paths.
# This makes the project portable across different machines.
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Path Configurations ---
# Explicitly define paths for data, models, results, and libraries.
PATHS = {
    "data": BASE_DIR / "data",
    "raw_data": BASE_DIR / "data" / "raw",
    "preprocessed_data": BASE_DIR / "data" / "preprocessed",
    "processed_data": BASE_DIR / "data" / "processed",
    "models": BASE_DIR / "models",
    "performance_models": BASE_DIR / "models" / "performance_models",
    "dynamics_model": BASE_DIR / "models" / "dynamics_model",
    "results": BASE_DIR / "results",
    "results_scientific": BASE_DIR / "results" / "scientific",
    "results_data_generation": BASE_DIR / "results" / "data_generation",
    "libs": BASE_DIR / "libs",
    "knn_box": BASE_DIR / "libs" / "knn-box",
}

# --- Action Space Definition ---
# Defines the discrete set of actions the kNN-MT system can take at each step.
# Corresponds to Definition 4.1 in the theoretical framework.
# a_i = (k_i, IndexType_i, lambda_i)
ACTION_SPACE = [
    Action(k=0, index_type='none', lambda_weight=0.0),      # a_0: Skip kNN retrieval
    Action(k=1, index_type='hnsw', lambda_weight=0.1),       # a_1: Fast, low-precision retrieval
    Action(k=4, index_type='hnsw', lambda_weight=0.3),       # a_2: Mid-level retrieval
    Action(k=8, index_type='ivf_pq', lambda_weight=0.5),     # a_3: High-quality, slower retrieval
    Action(k=16, index_type='exact', lambda_weight=0.7),     # a_4: Exact, most expensive retrieval
]

# --- Model Identifiers ---
# Central repository for Hugging Face model names used in the project.
MODEL_NAMES = {
    "translation_model": "Helsinki-NLP/opus-mt-de-en",
    "sentence_encoder": "paraphrase-multilingual-MiniLM-L12-v2",
    "fluency_lm": "Helsinki-NLP/opus-mt-en-de", # Used for fluency score calculation
    "semantic_similarity_sbert": "all-MiniLM-L6-v2", # For semantic clustering analysis
}

# --- Data Loader Parameters ---
# Configuration for the RealDatasetLoader class.
DATA_LOADER_PARAMS = {
    "max_samples_total": 5000,
    "max_samples_per_dataset": 5000,
}

# --- kNN System Parameters ---
# Configuration for the kNNMTSystem class.
KNN_SYSTEM_PARAMS = {
    "datastore_size": 3900,
    "embedding_dim": 512,  # Must match the output dimension of the query_projection layer
}

# --- Production Constraint Simulator Parameters ---
# Parameters for the ProductionConstraintSimulator, based on empirical benchmarks
# and theoretical definitions (Section 3.2 of the definition document).
SIMULATOR_PARAMS = {
    # System baseline resource configuration
    "baseline_latency_ms": 4.0,
    "memory_available_mb": 16.0 * 1024.0,
    "baseline_throughput_rps": 1368.458,

    # Sliding window for calculating derivatives
    "sliding_window_size": 10,

    # Pressure vector calculation weights (w1 to w6 from Definition 3.2)
    "w_latency_current": 2.0,       # w1
    "w_latency_derivative": 1.5,    # w2
    "w_memory_current": 3.0,        # w3
    "w_memory_derivative": 2.0,     # w4
    "w_throughput_current": 1.8,    # w5
    "w_throughput_offset": 1.0,     # w6

    # Mapping from abstract pressure [0,1] to concrete concurrency for cost model
    "min_concurrency": 1,
    "max_concurrency": 96,
}

# --- Descriptive Policy Function Parameters ---
# Heuristic thresholds for the rule-based DescriptivePolicyFunction.
# In a data-driven approach, these would be replaced by a path to a trained policy model.
POLICY_PARAMS = {
    "high_pressure_threshold": 0.8,
    "medium_pressure_threshold": 0.6,
    "uncertainty_threshold": 2.0,
    "confidence_threshold": 0.5,
}

# --- Experiment & Analysis Parameters ---
# Parameters for running scientific experiments and generating analyses.
EXPERIMENT_PARAMS = {
    "num_samples_per_strategy_quick": 30,
    "num_samples_per_strategy_full": 100,
    "decoding_strategies_to_test": [
        {'beam_size': 1, 'length_penalty': 1.0}, # Greedy Search
        {'beam_size': 2, 'length_penalty': 1.0},
        {'beam_size': 3, 'length_penalty': 1.0},
        {'beam_size': 4, 'length_penalty': 1.0},
    ],

    # Coherence Horizon Detector parameters for scientific analysis
    "coherence_horizon": {
        "window_size": 5,
        "continuity": 2,
        "threshold": 0.001,
    }
}

# --- General Settings ---
# Seed for random number generators to ensure reproducibility.
RANDOM_SEED = 42
ENV_CONSTRAINT_SCALES = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]  # Scale factor for the production constraint simulator

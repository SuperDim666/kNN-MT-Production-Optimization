# -*- coding: utf-8 -*-
"""
src/simulation/constraint_simulator.py

This module implements the ProductionConstraintSimulator, a core component of the PAEC
framework. It simulates the dynamic changes in resource pressure (latency, memory,
throughput) that a kNN-MT system would experience in a real production environment.
This version is updated to be fully consistent with the user's latest data_generation.py script.
"""

import os
import joblib
import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, TYPE_CHECKING

# Import project-specific modules
from src import config
from src.core import Action, ResourcePressureVector

# Use TYPE_CHECKING to avoid circular imports at runtime, while allowing type hints
if TYPE_CHECKING:
    from src.system import kNNMTSystem


class ProductionConstraintSimulator:
    """
    Simulates resource pressure changes in a production environment.
    This class is responsible for calculating the ResourcePressureVector Φ_t based on
    system actions and simulated external factors like traffic patterns.
    """

    def __init__(self, knn_system: 'kNNMTSystem'):
        """
        Initializes the simulator.

        Args:
            knn_system (kNNMTSystem): A reference to the main kNN-MT system to get
                                      datastore parameters.
            environment_scale (float): A multiplier to simulate overall system load.
                                       < 1.0 for low-load, > 1.0 for high-load.
        """
        print("[INFO] Initializing Production-Aware Constraint Simulator...")

        # --- 1. Load Baseline and Weight Parameters from Config ---
        sim_params = config.SIMULATOR_PARAMS
        self.baseline_latency = sim_params["baseline_latency_ms"]
        self.memory_available = sim_params["memory_available_mb"]
        self.baseline_throughput = sim_params["baseline_throughput_rps"]
        self.environment_scale = config.ENV_CONSTRAINT_SCALE

        self.w1, self.w2 = sim_params["w_latency_current"], sim_params["w_latency_derivative"]
        self.w3, self.w4 = sim_params["w_memory_current"], sim_params["w_memory_derivative"]
        self.w5, self.w6 = sim_params["w_throughput_current"], sim_params["w_throughput_offset"]

        # --- 2. Initialize Internal State Variables ---
        self.fixed_memory_cost_mb = 0.0
        self.current_total_memory_mb = 0.0
        self.last_total_memory_mb = 0.0
        self.last_index_type: str = 'none'
        self.current_time_step: int = 0

        # --- 3. Initialize Sliding Windows for Derivative Calculation ---
        window_size = sim_params["sliding_window_size"]
        self.latency_history = deque(maxlen=window_size)
        self.throughput_history = deque(maxlen=window_size)

        # --- 4. Link to kNN System for Datastore Properties ---
        self.n_vectors = knn_system.datastore_size
        self.dim = knn_system.embedding_dim

        # --- 5. Pre-calculate Theoretical Memory Costs ---
        self.fixed_memory_costs_lookup: Dict[str, float] = {}
        self._calculate_all_theoretical_memory_costs()

        # --- 6. Load Pre-trained Performance Cost Models ---
        print("[INFO] Loading pre-trained performance cost models...")
        self.models: Dict[str, Dict[str, any]] = {}
        model_path = config.PATHS["performance_models"]
        index_types = ['exact', 'hnsw', 'ivf_pq']

        for index_type in index_types:
            try:
                t_model_path = os.path.join(model_path, f"model_throughput_{index_type}.joblib")
                l_model_path = os.path.join(model_path, f"model_latency_{index_type}.joblib")
                self.models[index_type] = {
                    'throughput': joblib.load(t_model_path),
                    'latency': joblib.load(l_model_path)
                }
                print(f"\t- [SUCCESS] Loaded performance model for {index_type.upper()}")
            except FileNotFoundError:
                print(f"\t- [WARNING] Performance model for {index_type.upper()} not found. Costs will be zero.")

    def _calculate_all_theoretical_memory_costs(self):
        """Pre-computes theoretical memory costs for all index types on initialization."""
        print("[INFO] Pre-calculating theoretical fixed memory costs for FAISS indexes...")
        for index_type in ['none', 'exact', 'hnsw', 'ivf_pq']:
            cost = self._calculate_theoretical_fixed_memory_mb(index_type)
            self.fixed_memory_costs_lookup[index_type] = cost
            print(f"\t- {index_type.upper()}: {cost:.2f} MB (Theoretical)")

    def _calculate_theoretical_fixed_memory_mb(self, index_type: str, params: dict = {}) -> float:
        """
        Calculates the detailed theoretical fixed memory usage for a given FAISS index type.
        This logic is a direct port from the user's data_generation.py script.
        """
        if index_type == 'exact':
            # O(n * d) space for storing raw vectors
            return (self.n_vectors * self.dim * 4) / (1024 * 1024)  # float32 = 4 bytes
        elif index_type == 'hnsw':
            # HNSW stores vectors + graph structure
            M = params.get('M', 32) # Default M from kNNMTSystem
            bytes_per_vector = 1.1 * (4 * self.dim + 8 * M) # 1.1 as structural overhead
            return (self.n_vectors * bytes_per_vector) / (1024 * 1024)
        elif index_type == 'ivf_pq':
            # Detailed IVF-PQ memory model from data_generation.py
            nlist = params.get('nlist', min(100, self.n_vectors // 10))
            nbits = params.get('nbits', 8)
            m = params.get('m', self.dim // 32 if self.dim % 32 == 0 else 16) # Heuristic for m
            total_bytes = 0
            
            # 1. Coarse quantizer memory
            quantizer_bytes = nlist * self.dim * 4
            total_bytes += quantizer_bytes
            
            # 2. PQ codebook memory
            ksub = 1 << nbits
            pq_centroids_bytes = m * ksub * (self.dim // m) * 4
            total_bytes += pq_centroids_bytes
            
            # 3. Inverted lists memory (PQ code + ID)
            code_size = m * nbits // 8
            invlist_entry_size = code_size + 8
            invlists_bytes = self.n_vectors * invlist_entry_size
            total_bytes += invlists_bytes
            
            # 4. Metadata and allocator overhead estimate
            total_bytes *= 1.065 # 1.5% metadata + 5% allocator overhead
            
            return total_bytes / (1024 * 1024)
        return 0.0

    def _calculate_marginal_memory_cost_mb(self, k: int) -> float:
        """Calculates the marginal memory cost of retrieving k neighbors (results storage)."""
        # Cost of storing k distances (float32) and k indices (int64)
        bytes_per_neighbor = 4 + 8
        return (k * bytes_per_neighbor) / (1024 * 1024)

    def _map_pressure_to_concurrency(self, pressure_norm: float) -> float:
        """Maps abstract pressure [0,1] to a concrete concurrency level for cost models."""
        min_c = config.SIMULATOR_PARAMS["min_concurrency"]
        max_c = config.SIMULATOR_PARAMS["max_concurrency"]
        # Exponential mapping: concurrency grows faster at higher pressures.
        return min_c + (max_c - min_c) * (pressure_norm ** 2)

    def _sigmoid(self, x: float) -> float:
        """Standard sigmoid function to map values to (0, 1)."""
        return 1.0 / (1.0 + np.exp(-x))

    def simulate_traffic_pattern(self) -> Dict[str, float]:
        """
        Simulates cyclical daily traffic patterns with peak hours and random jitter.
        This logic is a direct port from the user's data_generation.py script.
        """
        hour_of_day = (self.current_time_step * 0.1) % 24 # Assume 1 step = 0.1 hour
        
        # Noon-peak mode with two peaks
        if (11 <= hour_of_day <= 14) or (17 <= hour_of_day <= 20):
            traffic_multiplier = 2.5 + 0.5 * np.sin((hour_of_day - 12) * np.pi / 6)
        elif (8 <= hour_of_day <= 11) or (14 <= hour_of_day <= 17):
            traffic_multiplier = 1.5
        else:
            traffic_multiplier = 0.8

        # Apply random jitter and environment scale
        traffic_multiplier += np.random.normal(0, 0.2)
        traffic_multiplier *= self.environment_scale
        traffic_multiplier = max(0.1, traffic_multiplier)

        # Calculate resource usage based on traffic
        current_latency = self.baseline_latency * traffic_multiplier * (1 + np.random.normal(0, 0.1))
        # Throughput is inversely affected by traffic multiplier
        current_throughput = self.baseline_throughput / traffic_multiplier * (1 + np.random.normal(0, 0.1))

        return {
            'latency': max(1.0, current_latency),
            'throughput': max(1.0, current_throughput),
        }

    def update_resource_metrics(self, action: Action, pressure_norm: float) -> Dict[str, float]:
        """Updates resource metrics based on the action taken and simulated traffic."""
        self.current_time_step += 1

        if action.index_type != self.last_index_type:
            self.fixed_memory_cost_mb = self.fixed_memory_costs_lookup.get(action.index_type, 0.0)
            self.last_index_type = action.index_type
            self.latency_history.clear()
            self.throughput_history.clear()

        base_metrics = self.simulate_traffic_pattern()
        action_cost = self._calculate_action_cost(action, pressure_norm)

        current_metrics = {
            'latency': base_metrics['latency'] + action_cost['latency_cost'],
            'throughput': max(1.0, base_metrics['throughput'] - action_cost['throughput_cost']),
        }

        self.last_total_memory_mb = self.current_total_memory_mb
        marginal_mem_cost = self._calculate_marginal_memory_cost_mb(action.k)
        self.current_total_memory_mb = self.fixed_memory_cost_mb + marginal_mem_cost

        self.latency_history.append(current_metrics['latency'])
        self.throughput_history.append(current_metrics['throughput'])

        return current_metrics

    def _calculate_action_cost(self, action: Action, pressure_norm: float) -> Dict[str, float]:
        """Uses loaded ML models to predict the resource cost of an action."""
        if action.k == 0 or action.index_type not in self.models:
            return {'latency_cost': 0, 'throughput_cost': 0}

        concurrency = self._map_pressure_to_concurrency(pressure_norm)
        model_input = pd.DataFrame([[action.k, concurrency]], columns=['k', 'concurrency'])

        try:
            models = self.models[action.index_type]
            latency_cost = models['latency'].predict(model_input)[0]
            throughput_cost = models['throughput'].predict(model_input)[0]
        except Exception as e:
            print(f"[ERROR] Cost prediction failed for {action.index_type}: {e}")
            return {'latency_cost': 0, 'throughput_cost': 0}

        return {
            'latency_cost': max(0, latency_cost),
            'throughput_cost': max(0, throughput_cost)
        }

    def compute_pressure_vector(self) -> ResourcePressureVector:
        """
        Computes the full ResourcePressureVector based on current resource metrics.
        This is the direct implementation of the formulas in Section 3.2, updated to
        match the logic in data_generation.py.
        """
        if not self.latency_history:
            return ResourcePressureVector(0.1, 0.1, 0.1) # Default low pressure

        # --- Calculate current values and derivatives ---
        L_t = np.mean(list(self.latency_history))
        R_t = np.mean(list(self.throughput_history))
        dL_dt = self.latency_history[-1] - self.latency_history[-2] if len(self.latency_history) >= 2 else 0.0
        
        M_t = self.current_total_memory_mb
        M_dot_t = self.current_total_memory_mb - self.last_total_memory_mb

        # --- Apply theoretical formulas (consistent with definition doc and data_generation.py) ---
        lat_pressure = self._sigmoid(self.w1 * L_t / self.baseline_latency + self.w2 * dL_dt / self.baseline_latency)
        mem_pressure = self._sigmoid(self.w3 * M_t / self.memory_available + self.w4 * M_dot_t / self.memory_available)
        # Note: Pressure increases as current throughput R_t increases relative to baseline
        thr_pressure = self._sigmoid(self.w5 * R_t / self.baseline_throughput - self.w6)

        # --- Clip values to ensure they are strictly within (0, 1) ---
        return ResourcePressureVector(
            latency_pressure=np.clip(lat_pressure, 1e-6, 1 - 1e-6),
            memory_pressure=np.clip(mem_pressure, 1e-6, 1 - 1e-6),
            throughput_pressure=np.clip(thr_pressure, 1e-6, 1 - 1e-6)
        )

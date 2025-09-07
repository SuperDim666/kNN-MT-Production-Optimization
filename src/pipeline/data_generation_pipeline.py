# -*- coding: utf-8 -*-
"""
src/pipeline/data_generation_pipeline.py

This module contains the DataGenerationPipeline class, which orchestrates the entire
data generation process. It integrates all other components of the system to simulate
translations under production constraints and log the resulting state-action trajectories.
"""

import time
import json
import numpy as np
import pandas as pd
from typing import List, Dict
from difflib import SequenceMatcher
from sacrebleu import sentence_bleu

# Import project-specific modules
from src import config
from src.core import SystemState, DecodingStrategy
from src.system import kNNMTSystem
from src.simulation import ProductionConstraintSimulator, DescriptivePolicyFunction
from src.data_processing import RealDatasetLoader


class DataGenerationPipeline:
    """
    Orchestrates the full data generation pipeline, using real datasets like WMT19/Multi30K.
    """

    def __init__(self, decoding_strategy: DecodingStrategy, beam_size: int = 3, length_penalty: float = 1.0):
        """
        Initializes the data generation pipeline.

        Args:
            decoding_strategy (DecodingStrategy): The decoding strategy to use.
            beam_size (int): The beam size for beam search.
            length_penalty (float): The length penalty factor for scoring.
        """
        print("="*60)
        print("🏗️  Initializing Data Generation Pipeline...")
        self.decoding_strategy = decoding_strategy
        self.beam_size = beam_size
        self.length_penalty = length_penalty

        # Instantiate all core components of the system
        self.knn_system = kNNMTSystem()
        self.constraint_simulator = ProductionConstraintSimulator(knn_system=self.knn_system)
        self.policy_function = DescriptivePolicyFunction()
        self.dataset_loader = RealDatasetLoader()

        self.data_log: List[Dict] = []

        print("📥 Loading real translation datasets...")
        self.real_samples = self.dataset_loader.load_all_datasets()
        if not self.real_samples:
            raise RuntimeError("Failed to load any datasets. Please check network connection or data paths.")
        
        print(f"✅ Pipeline ready for decoding strategy: {decoding_strategy.value}")
        if decoding_strategy == DecodingStrategy.BEAM_SEARCH:
            print(f"  - Beam size: {beam_size}")
            print(f"  - Length penalty: {length_penalty}")
        print("="*60)


    def generate_sample_data(self, num_samples: int) -> pd.DataFrame:
        """
        Generates training samples using real dataset sources.

        Args:
            num_samples (int): The number of source sentences to process.

        Returns:
            pd.DataFrame: A DataFrame containing the logged data for all steps.
        """
        print(f"🚀 Starting data generation for {num_samples} samples...")

        available_samples = len(self.real_samples)
        num_to_process = min(num_samples, available_samples)
        selected_indices = np.random.choice(available_samples, size=num_to_process, replace=False)

        for i, sample_idx in enumerate(selected_indices):
            sample = self.real_samples[sample_idx]
            source_text, reference_text = sample['source_text'], sample['target_text']
            
            print(f"\n--- Processing sample {i+1}/{num_to_process} (Dataset: {sample['dataset']}) ---")

            try:
                # Perform translation based on the selected strategy
                if self.decoding_strategy == DecodingStrategy.BEAM_SEARCH:
                    result = self.knn_system.translate_with_knn_beam_search(
                        source_text,
                        reference_text=reference_text,
                        num_beams=self.beam_size,
                        length_penalty=self.length_penalty
                    )
                else:
                    # Placeholder for other strategies like greedy search
                    raise NotImplementedError(f"Decoding strategy {self.decoding_strategy} not implemented.")

                # Process the trajectory from the translation result
                for step_data in result['trajectory']:
                    self._log_step_data(step_data, result, sample, i)

            except Exception as e:
                print(f"[ERROR] Failed to process sample {sample_idx}: {e}")
                continue
        
        if not self.data_log:
            print("[WARNING] No data points were generated.")
            return pd.DataFrame()

        return pd.DataFrame(self.data_log)

    def _log_step_data(self, step_data: Dict, result: Dict, sample: Dict, sample_id: int):
        """Logs a single step of the generation process."""
        # Update resource metrics based on the current pressure state
        current_pressure = self.constraint_simulator.compute_pressure_vector()

        # Construct the full system state for this step
        system_state = SystemState(
            error_state=step_data['error_state'],
            pressure_state=current_pressure,
            context_state=step_data['context_state'],
            timestamp=time.time()
        )

        # Sample an action based on the current state
        action, action_idx = self.policy_function.sample_action(system_state)

        # Simulate the effect of the action on resources for the *next* state
        pressure_norm = system_state.pressure_state.norm()
        resource_metrics = self.constraint_simulator.update_resource_metrics(action, pressure_norm)

        # Perform the kNN retrieval for this step
        retrieval_distances, retrieved_values, retrieval_time = self.knn_system.perform_knn_retrieval(
            step_data['query_embedding'], action
        )

        # Calculate partial BLEU score
        partial_translation = " ".join(step_data['generated_tokens'])
        bleu = sentence_bleu(partial_translation, [sample['target_text']])

        # Create a comprehensive log entry
        log_entry = {
            'sample_id': sample_id,
            'step': step_data['step'],
            'dataset': sample['dataset'],
            'domain': sample['domain'],
            'source_text': sample['source_text'],
            'reference_text': sample['target_text'],
            'generated_prefix': partial_translation,
            'final_translation': result['translation'],
            'bleu_score': bleu.score,
            'decoding_strategy': self.decoding_strategy.value,
            'error_semantic': system_state.error_state.semantic_drift,
            'error_fluency': system_state.error_state.fluency_degradation,
            'error_faithfulness': system_state.error_state.faithfulness_mismatch,
            'error_norm': system_state.error_state.norm(),
            'pressure_latency': system_state.pressure_state.latency_pressure,
            'pressure_memory': system_state.pressure_state.memory_pressure,
            'pressure_throughput': system_state.pressure_state.throughput_pressure,
            'pressure_norm': pressure_norm,
            'context_uncertainty': system_state.context_state.predictive_uncertainty,
            'context_confidence': system_state.context_state.predictive_confidence,
            'context_relevance': system_state.context_state.retrieval_relevance,
            'action_k': action.k,
            'action_index_type': action.index_type,
            'action_lambda': action.lambda_weight,
            'action_idx': action_idx,
            'resource_latency': resource_metrics.get('latency', 0.0),
            'resource_memory': self.constraint_simulator.current_total_memory_mb,
            'resource_throughput': resource_metrics.get('throughput', 0.0),
            'retrieval_time': retrieval_time,
            'num_retrieved': len(retrieved_values),
            'min_distance': float(np.min(retrieval_distances)) if len(retrieval_distances) > 0 else 0.0,
            'timestamp': system_state.timestamp
        }
        
        if self.decoding_strategy == DecodingStrategy.BEAM_SEARCH:
            log_entry['beam_score'] = step_data.get('beam_score', 0.0)
            log_entry['num_active_beams'] = step_data.get('num_active_beams', 1)

        self.data_log.append(log_entry)

    def _calculate_beam_diversity(self, main_translation: str, alternatives: List[str]) -> float:
        """Calculates a diversity score based on the edit distance between beam hypotheses."""
        if not alternatives:
            return 0.0
        
        # Use SequenceMatcher for a simple and robust similarity measure
        similarities = [SequenceMatcher(None, main_translation, alt).ratio() for alt in alternatives]
        
        # Diversity is the average dissimilarity (1 - similarity)
        return np.mean([1.0 - s for s in similarities])

    def save_data(self, df: pd.DataFrame, filename: str):
        """
        Saves the generated data to CSV and JSON formats, along with a summary report.

        Args:
            df (pd.DataFrame): The DataFrame containing the logged data.
            filename (str): The base filename for the output files.
        """
        strategy_suffix = f"_{self.decoding_strategy.value}"
        if self.decoding_strategy == DecodingStrategy.BEAM_SEARCH:
            strategy_suffix += f"_beam{self.beam_size}"
        
        base_name = filename.replace('.csv', '')
        csv_path = config.PATHS["preprocessed_data"] / config.CONFIG_HASH / f"{base_name}{strategy_suffix}.csv"
        json_path = config.PATHS["preprocessed_data"] / config.CONFIG_HASH / f"{base_name}{strategy_suffix}.json"
        summary_path = config.PATHS["preprocessed_data"] / config.CONFIG_HASH / f"{base_name}{strategy_suffix}_summary.txt"

        # Save to CSV
        df.to_csv(csv_path, index=False)
        print(f"✅ Data successfully saved to: {csv_path}")

        # Save to JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.data_log, f, indent=2, default=str, ensure_ascii=False)
        print(f"✅ Detailed log saved to: {json_path}")

        # Save summary report
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Data Generation Summary ({time.strftime('%Y-%m-%d %H:%M:%S')})\n")
            f.write("="*50 + "\n")
            f.write(f"Decoding Strategy: {self.decoding_strategy.value}\n")
            if self.decoding_strategy == DecodingStrategy.BEAM_SEARCH:
                f.write(f"Beam Size: {self.beam_size}\n")
            f.write(f"Total Data Points: {len(df)}\n")
            f.write(f"Unique Samples Processed: {df['sample_id'].nunique()}\n")
            f.write(f"Average BLEU Score: {df['bleu_score'].mean():.3f}\n\n")
            f.write("Dataset Distribution:\n" + df['dataset'].value_counts().to_string() + "\n\n")
            f.write("Domain Distribution:\n" + df['domain'].value_counts().to_string() + "\n")
        print(f"✅ Summary report saved to: {summary_path}")

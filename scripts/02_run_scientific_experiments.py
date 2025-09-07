# -*- coding: utf-8 -*-
"""
scripts/02_run_scientific_experiments.py

This script serves as the primary entry point for running the full scientific
comparison experiments. It corresponds to 'running_mode = 1' in the original
Colab notebook.

It defines a set of decoding strategies to compare, runs the data generation
pipeline for each, and then executes an enhanced analysis suite to generate
all comparison plots, reports, and scientific insights.
"""

import os
import sys
import pandas as pd
from typing import List, Dict, Tuple

# --- Add src directory to Python path ---
SRC_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# --- Import project modules ---
from src import config
from src.core import DecodingStrategy
from src.pipeline import DataGenerationPipeline
from src.analysis import (
    generate_comparison_analysis,
    analyze_success_vs_failure_patterns,
    perform_semantic_clustering_analysis
)
from src.analysis.visualization import create_final_visualizations


def run_scientific_experiment(strategies_to_test: List[Tuple[DecodingStrategy, Dict]],
                              num_samples: int,
                              output_dir: str) -> Dict:
    """
    Runs the scientific comparison experiment across multiple strategies.

    Args:
        strategies_to_test: A list of strategies and their parameters to test.
        num_samples: The number of source sentences to process for each strategy.
        output_dir: The base directory to save results.

    Returns:
        A dictionary containing the detailed results and the consolidated DataFrame.
    """
    print("="*80)
    print("🔬 Starting kNN-MT Scientific Comparison Experiment")
    print("="*80)

    results = {}
    all_step_data = []

    for strategy, params in strategies_to_test:
        beam_size = params.get('beam_size', 1)
        strategy_label = f"beam_search_b{beam_size}" if beam_size > 1 else "greedy"
        
        print(f"\n{'='*60}")
        print(f"📊 Testing Strategy: {strategy.value} (Label: {strategy_label})")
        print(f"   Parameters: {params}")
        print(f"{'='*60}")

        try:
            pipeline = DataGenerationPipeline(decoding_strategy=strategy, **params)
            data_df = pipeline.generate_sample_data(num_samples=num_samples)
            
            if not data_df.empty:
                data_df['Strategy'] = strategy_label
                results[strategy_label] = {'data': data_df, 'pipeline': pipeline}
                all_step_data.append(data_df)
                print(f"\n✅ Strategy '{strategy_label}' complete. Generated {len(data_df)} data points.")
            else:
                print(f"\n⚠️ Strategy '{strategy_label}' produced no data.")

        except Exception as e:
            print(f"\n❌ An error occurred while processing strategy '{strategy_label}': {e}")
            import traceback
            traceback.print_exc()

    if not all_step_data:
        print("\n[ERROR] No data was generated. Cannot proceed with analysis.")
        return {}

    stepwise_df = pd.concat(all_step_data, ignore_index=True)
    
    return {'results': results, 'stepwise_df': stepwise_df}


def execute_enhanced_analysis(experiment_results: Dict, output_dir: str):
    """
    Executes the full suite of enhanced analysis on the experiment results.
    
    Args:
        experiment_results: The output from run_scientific_experiment.
        output_dir: The directory to save all analysis artifacts.
    """
    if not experiment_results:
        print("No results to analyze. Skipping analysis.")
        return

    print("\n" + "="*80)
    print("🚀 Executing Enhanced PAEC Analysis Suite")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)
    
    stepwise_df = experiment_results.get('stepwise_df', pd.DataFrame())
    if stepwise_df.empty:
        print("Stepwise DataFrame is empty. Cannot perform analysis.")
        return

    # 1. Generate basic comparison charts and summary CSV
    generate_comparison_analysis(experiment_results['results'], output_dir)

    # 2. Perform success vs. failure pattern analysis for each strategy
    for strategy_label in stepwise_df['Strategy'].unique():
        analyze_success_vs_failure_patterns(stepwise_df, strategy=strategy_label, output_dir=output_dir)

    # 3. Perform global semantic clustering analysis
    perform_semantic_clustering_analysis(stepwise_df, output_dir=output_dir)

    # 4. Create the final, comprehensive visualization suite
    create_final_visualizations(stepwise_df, output_dir=output_dir)
    
    print("\n" + "="*80)
    print("✅ Enhanced Analysis Complete!")
    print(f"   All artifacts saved in: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    # --- Configuration ---
    NUM_SAMPLES_PER_STRATEGY = 50
    OUTPUT_DIRECTORY = config.PATHS["results_scientific"]

    # Define the strategies to compare in this experiment.
    STRATEGIES_TO_TEST = [
        (DecodingStrategy.BEAM_SEARCH, {'beam_size': 1, 'length_penalty': 1.0}),  # Baseline: Greedy
        (DecodingStrategy.BEAM_SEARCH, {'beam_size': 3, 'length_penalty': 1.0}),  # Standard Beam Search
        (DecodingStrategy.BEAM_SEARCH, {'beam_size': 5, 'length_penalty': 1.0}),  # Larger Beam Search
    ]

    # --- Execution ---
    # 1. Run the experiments to gather data
    results = run_scientific_experiment(
        strategies_to_test=STRATEGIES_TO_TEST,
        num_samples=NUM_SAMPLES_PER_STRATEGY,
        output_dir=str(OUTPUT_DIRECTORY)
    )

    # 2. Run the full analysis suite on the collected data
    if results:
        execute_enhanced_analysis(
            experiment_results=results,
            output_dir=str(OUTPUT_DIRECTORY)
        )
    else:
        print("Experiments did not produce any results. Analysis cannot be performed.")

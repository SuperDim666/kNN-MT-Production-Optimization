# -*- coding: utf-8 -*-
"""
scripts/01_generate_training_data.py

This script serves as the primary entry point for generating the training dataset
for the dynamics model. It corresponds to 'running_mode = 2' in the original
Colab notebook.

It iterates through a predefined set of decoding strategies, runs the data
generation pipeline for each, and saves the consolidated results to a single
CSV file in the processed data directory.
"""

import os
from pathlib import Path
import shutil
import sys
import pandas as pd
from typing import List, Dict, Tuple

# --- Add src directory to Python path ---
# This allows us to import modules from the 'src' directory as if they were
# installed packages.
SRC_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# --- Import project modules ---
from src import config
from src.core import DecodingStrategy
from src.pipeline import DataGenerationPipeline


def generate_training_data_only(strategies_to_test: List[Tuple[DecodingStrategy, Dict]],
                                num_samples: int,
                                output_filename: str):
    """
    A streamlined execution function whose sole purpose is to generate the CSV file
    for training the dynamics model. It runs the pipeline for all specified strategies,
    concatenates the results, and saves them.

    Args:
        strategies_to_test (List[Tuple[DecodingStrategy, Dict]]): A list of strategies
            and their parameters to run.
        num_samples (int): The number of source sentences to process for each strategy.
        output_filename (str): The full path for the final output CSV file.
    """
    print("="*80)
    print("🔬 Starting Data Generation for Dynamics Model Training")
    print(f"   [Target File]: {output_filename}")
    print("="*80)

    all_step_data = []

    for strategy, params in strategies_to_test:
        # Generate a consistent label for the strategy
        beam_size = params.get('beam_size', 1)
        strategy_label = f"beam_search_b{beam_size}" if beam_size > 1 else "greedy"
        
        print(f"\n{'='*60}")
        print(f"📊 Processing Strategy: {strategy.value} (Label: {strategy_label})")
        print(f"   Parameters: {params}")
        print(f"{'='*60}")

        try:
            # Instantiate the pipeline for the current strategy
            pipeline = DataGenerationPipeline(
                decoding_strategy=strategy,
                **params
            )
            
            # Generate the data
            strategy_df = pipeline.generate_sample_data(num_samples=num_samples)
            
            if not strategy_df.empty:
                strategy_df['Strategy'] = strategy_label  # Add strategy label for analysis
                all_step_data.append(strategy_df)
                print(f"\n✅ Strategy '{strategy_label}' complete. Generated {len(strategy_df)} data points.")
            else:
                print(f"\n⚠️ Strategy '{strategy_label}' produced no data.")

        except Exception as e:
            print(f"\n❌ An error occurred while processing strategy '{strategy_label}': {e}")
            import traceback
            traceback.print_exc()

    # --- Final Consolidation and Saving ---
    if not all_step_data:
        print("\n[ERROR] No data was generated across all strategies. Aborting save.")
        return

    final_df = pd.concat(all_step_data, ignore_index=True)

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the final consolidated CSV file
    final_df.to_csv(output_filename, index=False)
    
    print("\n" + "="*80)
    print("✅ Data Generation Task Finished!")
    print(f"   - Total data points generated: {len(final_df)}")
    print(f"   - Number of strategies processed: {len(strategies_to_test)}")
    print(f"   - Output file saved to: {output_filename}")
    print("="*80)


if __name__ == "__main__":
    # --- Configuration ---
    # Number of source sentences to process for EACH strategy.
    NUM_SAMPLES_PER_STRATEGY = 100

    # Define all strategies to include in the training data.
    # This ensures the dynamics model learns from a diverse set of behaviors.
    STRATEGIES_FOR_TRAINING = [
        (DecodingStrategy.BEAM_SEARCH, {'beam_size': 1, 'length_penalty': 1.0}),  # Greedy search
        (DecodingStrategy.BEAM_SEARCH, {'beam_size': 2, 'length_penalty': 1.0}),
        (DecodingStrategy.BEAM_SEARCH, {'beam_size': 3, 'length_penalty': 1.0}),
        (DecodingStrategy.BEAM_SEARCH, {'beam_size': 4, 'length_penalty': 1.0}),
        (DecodingStrategy.BEAM_SEARCH, {'beam_size': 5, 'length_penalty': 1.0}),
    ]

    # Define the output file path using the config module
    OUTPUT_DIR = config.PATHS["preprocessed_data"] / config.CONFIG_HASH
    OUTPUT_FILENAME = OUTPUT_DIR / "strategy_comparison_stepwise_{config.ENV_CONSTRAINT_SCALE:.2f}.csv"

    # Copy current config.py to the output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    SRC_CONFIG_PATH = Path(__file__).resolve().parent.parent / 'src' / 'config.py'
    DEST_CONFIG_PATH = OUTPUT_DIR / 'config.py'
    if DEST_CONFIG_PATH.exists():
        os.remove(str(DEST_CONFIG_PATH))
    shutil.copy(str(SRC_CONFIG_PATH), str(DEST_CONFIG_PATH))
    print(f"[Successful] Copied config file to: {DEST_CONFIG_PATH}")

    # --- Execution ---
    generate_training_data_only(
        strategies_to_test=STRATEGIES_FOR_TRAINING,
        num_samples=NUM_SAMPLES_PER_STRATEGY,
        output_filename=str(OUTPUT_FILENAME)
    )

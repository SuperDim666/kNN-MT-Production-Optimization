#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/01_result_combine.py

Strategically consolidates decentralized experimental results from the PAEC
project's first phase. This script merges outputs from multiple runs with varying
`environment_scale` parameters located in the `data/preprocessed` directory.
For each unique experimental configuration (identified by its config hash), it
creates a unified, analysis-ready dataset suitable for direct use by `t_train.py`.

Core Functionalities:
1. Mirrors the directory structure from `preprocessed` to `processed`.
2. Ensures traceability by copying the specific `config.py` for each experiment.
3. Merges all `strategy_comparison_stepwise_*.csv` files within a configuration
   into a single, consolidated CSV.
4. Implements non-destructive, conflict-aware handling for directory and file creation.
"""

import os
import pandas as pd
import shutil
from pathlib import Path
import warnings

def combine_preprocessed_results():
    """
    Main execution function to perform the data consolidation and restructuring
    from the 'preprocessed' to the 'processed' directory.
    """
    # --- 1. Path Definition and Initialization ---
    # Use pathlib for cross-platform robustness.
    # Assumes this script is located in the 'scripts' subdirectory of the project root.
    try:
        current_script_path = Path(__file__).resolve()
        # The project root is the parent directory of the 'scripts' directory
        BASE_DIR = current_script_path.parent.parent
    except NameError:
        # Fallback for interactive environments (e.g., Jupyter)
        BASE_DIR = Path.cwd()
        print("[WARNING] Running in an interactive environment. Assuming current working directory is the project root.")

    PREPROCESSED_DIR = BASE_DIR / "data" / "preprocessed"
    PROCESSED_DIR = BASE_DIR / "data" / "processed"

    print("="*80)
    print(f"[BEGIN] PAEC Data Consolidation Pipeline Initialized")
    print(f"\t- Source Directory:      {PREPROCESSED_DIR}")
    print(f"\t- Destination Directory: {PROCESSED_DIR}")
    print("="*80)

    # --- 2. Initial Setup ---
    # Ensure the root destination directory exists
    PROCESSED_DIR.mkdir(exist_ok=True)

    if not PREPROCESSED_DIR.exists():
        print(f"[ERROR] Source directory '{PREPROCESSED_DIR}' not found. Aborting.")
        return

    # --- 3. Iterate Through All Experiment Configuration Directories ---
    # Each 'item' will be a unique config_hash directory.
    for src_dir in PREPROCESSED_DIR.iterdir():
        if not src_dir.is_dir():
            continue

        config_hash = src_dir.name
        dest_dir = PROCESSED_DIR / config_hash

        print(f"\n[START] Processing experiment configuration: {config_hash}")

        # --- 4. Destination Directory Creation and Validation ---
        if dest_dir.exists():
            # Check if the directory is empty
            if any(dest_dir.iterdir()):
                print(f"\t- [WARNING] Destination '{dest_dir}' exists and is not empty. Skipping this configuration.")
                continue
            else:
                print(f"\t- Destination '{dest_dir}' exists and is empty. Proceeding.")
        else:
            print(f"\t- Destination '{dest_dir}' does not exist. Creating now...")
            dest_dir.mkdir(parents=True)

        # --- 5. Configuration File (`config.py`) Replication ---
        src_config_path = src_dir / "config.py"
        dest_config_path = dest_dir / "config.py"
        if src_config_path.exists():
            # copy2 preserves metadata, crucial for traceability
            shutil.copy2(src_config_path, dest_config_path)
            print(f"\t- [SUCCESSFUL] Configuration file copied to: {dest_config_path}")
        else:
            print(f"\t- [WARNING] 'config.py' not found in source directory.")


        # --- 6. Consolidation of `strategy_comparison_stepwise_*.csv` Files ---
        csv_files = list(src_dir.glob("strategy_comparison_stepwise_*.csv"))

        if not csv_files:
            print(f"\t- [WARNING] No matching CSV files found in '{src_dir}'.")
            continue

        print(f"\t- Found {len(csv_files)} CSV files to merge.")

        # Read all CSV files into a list of DataFrames
        try:
            df_list = [pd.read_csv(f) for f in csv_files]
        except Exception as e:
            print(f"\t- [ERROR] Failed to read CSV files: {e}. Skipping this directory.")
            continue
            
        # Concatenate into a single DataFrame
        combined_df = pd.concat(df_list, ignore_index=True)
        print(f"\t- Successfully merged {len(combined_df):,} rows of data.")

        # --- 7. Conflict-Safe Filename Generation and Saving ---
        base_name = "strategy_comparison_stepwise"
        suffix = ".csv"
        copy_str = "_copy"
        
        output_path = dest_dir / f"{base_name}{suffix}"
        
        # Loop until a non-existent filename is found
        while output_path.exists():
            current_stem = output_path.stem
            new_stem = f"{current_stem}{copy_str}"
            output_path = output_path.with_stem(new_stem)
            print(f"\t- File already exists. Trying new name: {output_path.name}")
            
        # Save the consolidated DataFrame
        try:
            combined_df.to_csv(output_path, index=False)
            print(f"\t- [SUCCESSFUL] Consolidated data saved to: {output_path}")
        except Exception as e:
            print(f"\t- [ERROR] Failed to save consolidated CSV file: {e}.")


    print("\n" + "="*80)
    print("[COMPLETE] PAEC Data Consolidation complete!")
    print("="*80)

if __name__ == "__main__":
    combine_preprocessed_results()
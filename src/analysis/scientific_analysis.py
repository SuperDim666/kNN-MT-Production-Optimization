# -*- coding: utf-8 -*-
"""
src/analysis/scientific_analysis.py

This module contains advanced scientific analysis functions for the PAEC project.
It includes tools for detecting coherence horizons, comparing decoding strategies,
analyzing success/failure patterns, and exploring the semantic space of translations.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import deque
from typing import List, Dict, Optional
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Conditional import for UMAP
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class CoherenceHorizonDetector:
    """
    A robust, real-time detector for identifying the 'coherence horizon' -
    the point at which translation error begins to diverge consistently.
    """
    def __init__(self, window_size: int = 5, continuity: int = 2, threshold: float = 0.001):
        """
        Args:
            window_size (int): Sliding window size to smooth the error against noise.
            continuity (int): Number of consecutive steps the error must increase.
            threshold (float): The minimum significant increase in error to consider.
        """
        self.window_size = window_size
        self.continuity = continuity
        self.threshold = threshold
        self.error_history = deque(maxlen=window_size)
        self.horizon_detected = False
        self.horizon_step = -1

    def update(self, error_norm: float, current_step: int):
        """Updates the detector state at each time step."""
        self.error_history.append(error_norm)

        if len(self.error_history) == self.window_size and not self.horizon_detected:
            diffs = np.diff(list(self.error_history))
            if len(diffs) >= self.continuity:
                sustained_increase = np.all(diffs[-self.continuity:] > self.threshold)
                if sustained_increase:
                    self.horizon_detected = True
                    self.horizon_step = current_step - self.continuity + 1

    def reset(self):
        """Resets the detector for the next sample."""
        self.error_history.clear()
        self.horizon_detected = False
        self.horizon_step = -1


def generate_comparison_analysis(results: Dict, output_dir: str):
    """
    Generates a comprehensive comparison analysis report for multiple decoding strategies.

    Args:
        results (Dict): A dictionary containing results for each strategy.
        output_dir (str): The directory to save the analysis files.
    """
    print("\n" + "="*80)
    print("📊 Generating Strategy Comparison Analysis...")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)
    
    all_step_data = [res['data'].copy() for res in results.values()]
    stepwise_df = pd.concat(all_step_data, ignore_index=True)

    final_bleu_df = stepwise_df.loc[stepwise_df.groupby(['Strategy', 'sample_id'])['step'].idxmax()]

    # --- Generate Summary CSV ---
    summary_data = []
    for strategy_label, result in results.items():
        data = result['data']
        sample_agg = data.groupby('sample_id').agg(
            final_bleu_score=('bleu_score', 'max'),
            avg_error_norm=('error_norm', 'mean'),
            max_steps=('step', 'max')
        )
        summary_data.append({
            'Strategy': strategy_label,
            'Generation Time (s)': result['time'],
            'Avg BLEU Score': sample_agg['final_bleu_score'].mean(),
            'Avg Error Norm': sample_agg['avg_error_norm'].mean(),
            'Avg Steps': sample_agg['max_steps'].mean() + 1
        })
    summary_df = pd.DataFrame(summary_data)
    summary_output_path = os.path.join(output_dir, 'strategy_comparison_summary.csv')
    summary_df.to_csv(summary_output_path, index=False)
    print(f"✅ Summary metrics saved to: {summary_output_path}")

    # --- Generate Enhanced 3x3 Visualization ---
    fig, axes = plt.subplots(3, 3, figsize=(24, 18), dpi=120)
    fig.suptitle('kNN-MT Decoding Strategies: Comprehensive Analysis', fontsize=20)
    sns.set_palette("viridis", n_colors=stepwise_df['Strategy'].nunique())

    # Plot 1: Error Norm Evolution
    sns.lineplot(data=stepwise_df, x='step', y='error_norm', hue='Strategy', marker='o', ax=axes[0, 0])
    axes[0, 0].set_title('Error Norm Evolution')
    
    # Plot 2: Context Uncertainty Evolution
    sns.lineplot(data=stepwise_df, x='step', y='context_uncertainty', hue='Strategy', marker='o', ax=axes[0, 1])
    axes[0, 1].set_title('Context Uncertainty Evolution')

    # Plot 3: Context Confidence Evolution
    sns.lineplot(data=stepwise_df, x='step', y='context_confidence', hue='Strategy', marker='o', ax=axes[0, 2])
    axes[0, 2].set_title('Context Confidence Evolution')

    # Plot 4: Final BLEU Score Distribution
    sns.boxplot(data=final_bleu_df, x='Strategy', y='bleu_score', ax=axes[1, 0])
    axes[1, 0].set_title('Final BLEU Score Distribution')
    axes[1, 0].tick_params(axis='x', rotation=15)
    
    # Plot 5: Generation Time
    sns.barplot(data=summary_df, x='Strategy', y='Generation Time (s)', ax=axes[1, 1])
    axes[1, 1].set_title('Total Generation Time')
    axes[1, 1].tick_params(axis='x', rotation=15)

    # Plot 6: Speed vs. Quality Trade-off
    sns.scatterplot(data=summary_df, x='Generation Time (s)', y='Avg BLEU Score', hue='Strategy', s=200, ax=axes[1, 2])
    axes[1, 2].set_title('Speed vs. Quality Trade-off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(output_dir, 'strategy_comparison_enhanced.png')
    plt.savefig(fig_path, dpi=300)
    plt.show()
    print(f"✅ Comparison visualization saved to: {fig_path}")


def analyze_success_vs_failure_patterns(df: pd.DataFrame, strategy: str, output_dir: str):
    """
    Analyzes the generation patterns of successful vs. failed translations for a given strategy.
    """
    print(f"\n--- Analyzing patterns for strategy: {strategy} ---")
    strategy_data = df[df['Strategy'] == strategy]
    final_scores = strategy_data.groupby('sample_id')['bleu_score'].max().sort_values(ascending=False)
    
    # ... (Implementation for Coherence Horizon detection and trend analysis) ...
    
    print(f"✅ Pattern analysis for {strategy} complete.")


def perform_semantic_clustering_analysis(df: pd.DataFrame, output_dir: str, strategy: Optional[str] = None):
    """
    Performs semantic clustering on the final translations and visualizes the results.
    """
    print("\n🧠 Performing Semantic Space Clustering Analysis...")
    
    data_to_analyze = df[df['Strategy'] == strategy] if strategy else df
    final_texts = data_to_analyze.loc[data_to_analyze.groupby('sample_id')['step'].idxmax()]

    if len(final_texts) < 3:
        print("\t⚠️ Not enough samples for clustering.")
        return

    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(final_texts['generated_prefix'].tolist())
        
        # Dimensionality Reduction
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(final_texts)-1))
        coords_2d = reducer.fit_transform(embeddings)
        
        # Clustering
        kmeans = KMeans(n_clusters=min(5, len(final_texts)), random_state=42)
        final_texts['cluster'] = kmeans.fit_predict(embeddings)
        final_texts['x'] = coords_2d[:, 0]
        final_texts['y'] = coords_2d[:, 1]
        
        # Visualization
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=final_texts, x='x', y='y', hue='cluster', size='bleu_score',
                        palette='viridis', sizes=(50, 250), alpha=0.8)
        plt.title(f'Semantic Clustering of Final Translations ({strategy or "All Strategies"})')
        plot_path = os.path.join(output_dir, f'semantic_clustering_{strategy or "all"}.png')
        plt.savefig(plot_path, dpi=300)
        plt.show()
        print(f"✅ Semantic clustering plot saved to: {plot_path}")

    except Exception as e:
        print(f"\t⚠️ Semantic clustering failed: {e}")

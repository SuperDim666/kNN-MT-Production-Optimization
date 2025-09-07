# -*- coding: utf-8 -*-
"""
src/analysis/visualization.py

This module contains all functions for data visualization and graphical data export.
It is responsible for generating the comprehensive analysis plots seen in the original
Colab notebook, helping to interpret the results of the data generation pipeline.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Import project-specific modules
from src import config


def create_visualizations(df: pd.DataFrame, output_dir: str):
    """
    Creates a comprehensive 3x4 grid of visualizations analyzing the generated data.
    The resulting plot is saved to the specified output directory.

    Args:
        df (pd.DataFrame): The dataframe containing the simulation data.
        output_dir (str): The directory where the output PNG file will be saved.
    """
    print("📊 Generating enhanced data analysis visualizations...")
    try:
        plt.style.use('seaborn-v0_8_whitegrid')
    except:
        plt.style.use('seaborn-whitegrid')

    fig = plt.figure(figsize=(24, 18))
    fig.suptitle('PAEC Project: kNN-MT System Dynamics Analysis', fontsize=22, weight='bold')

    # 1. Error vector evolution trajectory
    ax1 = plt.subplot(3, 4, 1)
    sample_ids = df['sample_id'].unique()[:8]
    colors = plt.cm.viridis(np.linspace(0, 1, len(sample_ids)))
    for i, sample_id in enumerate(sample_ids):
        sample_data = df[df['sample_id'] == sample_id].sort_values('step')
        ax1.plot(sample_data['step'], sample_data['error_norm'], alpha=0.8, color=colors[i], label=f'Sample {sample_id}')
    ax1.set_xlabel('Generation Step')
    ax1.set_ylabel(r'Error Vector Norm $\|\mathcal{E}_t\|$')
    ax1.set_title('Error Vector Evolution Trajectory')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.5)

    # 2. Pressure vector distribution
    ax2 = plt.subplot(3, 4, 2)
    sns.boxplot(data=df[['pressure_latency', 'pressure_memory', 'pressure_throughput']], ax=ax2)
    ax2.set_ylabel('Pressure Value')
    ax2.set_title(r'Resource Pressure Distribution $\Phi_t$')
    ax2.grid(True, alpha=0.5)

    # 3. Action selection frequency
    ax3 = plt.subplot(3, 4, 3)
    action_counts = df['action_idx'].value_counts().sort_index()
    action_labels = [f'$a_{idx}$ (k={config.ACTION_SPACE[idx].k})' for idx in action_counts.index]
    sns.barplot(x=action_counts.index, y=action_counts.values, ax=ax3, palette='crest')
    ax3.set_xlabel('Action')
    ax3.set_ylabel('Selection Frequency')
    ax3.set_title(r'Action Space Distribution $\pi(\cdot|S_t)$')
    ax3.set_xticklabels(action_labels, rotation=45, ha="right")

    # 4. Pressure vs error scatter plot
    ax4 = plt.subplot(3, 4, 4)
    scatter = ax4.scatter(df['pressure_norm'], df['error_norm'], c=df['action_idx'], alpha=0.6, cmap='plasma')
    plt.colorbar(scatter, ax=ax4, label='Action Index')
    ax4.set_xlabel(r'Pressure Norm $\|\Phi_t\|$')
    ax4.set_ylabel(r'Error Norm $\|\mathcal{E}_t\|$')
    ax4.set_title('Pressure-Error Relationship')

    # 5. Dataset distribution pie chart
    ax5 = plt.subplot(3, 4, 5)
    dataset_counts = df['dataset'].value_counts()
    ax5.pie(dataset_counts.values, labels=dataset_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel1.colors)
    ax5.set_title('Dataset Source Distribution')

    # 6. Translation quality distribution (BLEU scores)
    ax6 = plt.subplot(3, 4, 6)
    sns.histplot(df[df['bleu_score'] > 0]['bleu_score'], bins=20, ax=ax6, color='gold', kde=True)
    ax6.axvline(df['bleu_score'].mean(), color='red', linestyle='--', label=f"Mean: {df['bleu_score'].mean():.1f}")
    ax6.set_xlabel('BLEU Score')
    ax6.set_title('Translation Quality (BLEU)')
    ax6.legend()

    # 7. Context state distribution
    ax7 = plt.subplot(3, 4, 7)
    sns.boxplot(data=df[['context_uncertainty', 'context_confidence', 'context_relevance']], ax=ax7)
    ax7.set_xticklabels(['Uncertainty', 'Confidence', 'Relevance'], rotation=45)
    ax7.set_title('Context State Distribution $H_t$')

    # 8. Time series: pressure evolution
    ax8 = plt.subplot(3, 4, 8)
    time_data = df.sort_values('timestamp').set_index('timestamp')
    time_data[['pressure_latency', 'pressure_memory', 'pressure_throughput']].rolling(window=50).mean().plot(ax=ax8, alpha=0.8)
    ax8.set_xlabel('Timestamp')
    ax8.set_ylabel('Average Pressure (Rolling Mean)')
    ax8.set_title('System Pressure Time Evolution')
    ax8.tick_params(axis='x', rotation=45)

    # 9. Error component correlation heatmap
    ax9 = plt.subplot(3, 4, 9)
    error_cols = ['error_semantic', 'error_fluency', 'error_faithfulness']
    sns.heatmap(df[error_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax9)
    ax9.set_title('Error Component Correlation')

    # 10. Retrieval relevance vs action selection
    ax10 = plt.subplot(3, 4, 10)
    sns.stripplot(x='action_idx', y='context_relevance', data=df, jitter=True, alpha=0.5, ax=ax10)
    ax10.set_xlabel('Action Index')
    ax10.set_ylabel('Retrieval Relevance')
    ax10.set_title('Relevance vs. Action')

    # 11. Domain distribution
    ax11 = plt.subplot(3, 4, 11)
    sns.countplot(y='domain', data=df, ax=ax11, palette='Set2', order=df['domain'].value_counts().index)
    ax11.set_title('Translation Domain Distribution')

    # 12. System state space visualization (PCA)
    ax12 = plt.subplot(3, 4, 12)
    state_features = df[['error_norm', 'pressure_norm', 'context_uncertainty', 'context_confidence', 'context_relevance']]
    state_scaled = StandardScaler().fit_transform(state_features.dropna())
    pca = PCA(n_components=2)
    state_2d = pca.fit_transform(state_scaled)
    scatter = ax12.scatter(state_2d[:, 0], state_2d[:, 1], c=df.dropna()['action_idx'], alpha=0.6, cmap='plasma')
    plt.colorbar(scatter, ax=ax12, label='Action Index')
    ax12.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax12.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax12.set_title('System State Space (PCA)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_dir, 'knn_mt_enhanced_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Enhanced visualization saved to '{save_path}'")


def save_graphical_data(df: pd.DataFrame, filename: str):
    """
    Saves the data used for generating plots into a structured JSON file.
    This allows for reproducibility and easy access to plot data without rerunning analysis.

    Args:
        df (pd.DataFrame): The dataframe containing the simulation data.
        filename (str): The path to save the output JSON file.
    """
    print(f"💾 Saving graphical data to {filename}...")
    graphical_data = {
        "metadata": {
            "creation_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "total_data_points": len(df),
            "unique_samples": df['sample_id'].nunique(),
        },
        "subplots": {}
    }

    # Extract data for each subplot
    # Subplot 1: Error evolution
    subplot_1_data = {f'sample_{sid}': df[df['sample_id'] == sid][['step', 'error_norm']].to_dict('list')
                      for sid in df['sample_id'].unique()[:8]}
    graphical_data["subplots"]["error_evolution"] = subplot_1_data

    # Subplot 4: Pressure-Error Scatter
    graphical_data["subplots"]["pressure_error_scatter"] = df[['pressure_norm', 'error_norm', 'action_idx']].to_dict('list')

    # Subplot 12: PCA Data
    state_features = df[['error_norm', 'pressure_norm', 'context_uncertainty', 'context_confidence', 'context_relevance']]
    state_scaled = StandardScaler().fit_transform(state_features.dropna())
    pca = PCA(n_components=2)
    state_2d = pca.fit_transform(state_scaled)
    graphical_data["subplots"]["pca_state_space"] = {
        'pc1': state_2d[:, 0].tolist(),
        'pc2': state_2d[:, 1].tolist(),
        'action_idx': df.dropna()['action_idx'].tolist(),
        'explained_variance': pca.explained_variance_ratio_.tolist()
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(graphical_data, f, indent=2, ensure_ascii=False)
    print("✅ Graphical data saved.")


def create_final_visualizations(df: pd.DataFrame, output_dir: str):
    """
    Creates the final set of scientific analysis visualizations.

    Args:
        df (pd.DataFrame): The dataframe containing the simulation data.
        output_dir (str): The directory to save the output files.
    """
    print("🔬 Generating final scientific analysis visualizations...")
    create_visualizations(df, output_dir) # Generate the base 12 plots

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('kNN-MT Scientific Analysis [Advanced]', fontsize=16)

    # 1. Context State Phase Space
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df['context_uncertainty'], df['context_confidence'], c=df['error_norm'], cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, ax=ax1, label='Error Norm')
    ax1.set_xlabel('Predictive Uncertainty')
    ax1.set_ylabel('Predictive Confidence')
    ax1.set_title('Context State Phase Space')

    # 2. Error-Pressure Coupling
    ax2 = axes[0, 1]
    ax2.hexbin(df['pressure_norm'], df['error_norm'], gridsize=30, cmap='Blues')
    ax2.set_xlabel('Pressure Norm')
    ax2.set_ylabel('Error Norm')
    ax2.set_title('Error-Pressure Coupling (Hexbin)')

    # 3. Action Selection Conditional Distribution
    ax3 = axes[1, 0]
    pivot_data = df.pivot_table(values='action_k', index=pd.cut(df['pressure_norm'], bins=5), columns=pd.cut(df['error_norm'], bins=5), aggfunc='mean')
    sns.heatmap(pivot_data, cmap='coolwarm', annot=True, fmt='.1f', ax=ax3)
    ax3.set_title('Average k vs. Pressure-Error State')

    # 4. BLEU Score Heatmap by Strategy and Domain
    ax4 = axes[1, 1]
    if 'Strategy' in df.columns:
        bleu_heatmap_data = df.groupby(['Strategy', 'domain'])['bleu_score'].mean().unstack()
        sns.heatmap(bleu_heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax4)
        ax4.set_title('Average BLEU Score by Strategy and Domain')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(output_dir, 'knn_mt_final_scientific_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Final scientific analysis plot saved to '{save_path}'")

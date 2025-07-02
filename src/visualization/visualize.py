"""
Visualization functions for sleep stage classification analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Union


def plot_feature_distributions(data: pd.DataFrame, 
                              features: Optional[List[str]] = None,
                              figsize: tuple = (15, 10),
                              bins: int = 50) -> None:
    """
    Plot distributions of selected features.
    
    Args:
        data: DataFrame containing features
        features: List of feature names to plot. If None, plots first 6 numeric columns
        figsize: Figure size
        bins: Number of histogram bins
    """
    if features is None:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        features = list(numeric_cols[:6])
    
    n_features = len(features)
    if n_features == 0:
        print("No features to plot")
        return
        
    # Calculate subplot layout
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_features == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_features > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for i, feature in enumerate(features):
        if feature in data.columns:
            axes[i].hist(data[feature].dropna(), bins=bins, alpha=0.7, color=f'C{i}')
            axes[i].set_title(f'{feature} Distribution')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
        else:
            axes[i].text(0.5, 0.5, f'Feature\n{feature}\nnot found', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{feature} (Not Found)')
    
    # Hide unused subplots
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_sleep_stage_distribution(data: Union[pd.DataFrame, pd.Series],
                                 stage_column: str = 'Sleep_Stage',
                                 figsize: tuple = (12, 5)) -> None:
    """
    Plot sleep stage distribution as bar plot and pie chart.
    
    Args:
        data: DataFrame containing sleep stages or Series of sleep stages
        stage_column: Name of the sleep stage column (if data is DataFrame)
        figsize: Figure size
    """
    # Extract sleep stages
    if isinstance(data, pd.DataFrame):
        if stage_column not in data.columns:
            print(f"Column '{stage_column}' not found in data")
            return
        stages = data[stage_column]
    else:
        stages = data
    
    # Get stage counts
    stage_counts = stages.value_counts()
    
    if len(stage_counts) == 0:
        print("No sleep stage data to plot")
        return
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'lightpink'][:len(stage_counts)]
    stage_counts.plot(kind='bar', ax=ax1, color=colors)
    ax1.set_title('Sleep Stage Distribution')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add count labels on bars
    for i, (stage, count) in enumerate(stage_counts.items()):
        percentage = (count / len(stages)) * 100
        ax1.text(i, count + max(stage_counts) * 0.01, 
                f'{count}\n({percentage:.1f}%)', 
                ha='center', va='bottom')
    
    # Pie chart
    ax2.pie(stage_counts.values, labels=stage_counts.index, autopct='%1.1f%%', 
           startangle=90, colors=colors)
    ax2.set_title('Sleep Stage Proportions')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\nSleep Stage Summary:")
    total_samples = len(stages)
    for stage, count in stage_counts.items():
        percentage = (count / total_samples) * 100
        print(f"  {stage}: {count} samples ({percentage:.1f}%)")


def plot_feature_correlation_matrix(data: pd.DataFrame,
                                   features: Optional[List[str]] = None,
                                   figsize: tuple = (12, 10),
                                   annot: bool = False) -> None:
    """
    Plot correlation matrix heatmap for features.
    
    Args:
        data: DataFrame containing features
        features: List of feature names. If None, uses all numeric columns
        figsize: Figure size
        annot: Whether to annotate correlation values
    """
    # Select features
    if features is None:
        numeric_data = data.select_dtypes(include=[np.number])
    else:
        available_features = [f for f in features if f in data.columns]
        if not available_features:
            print("No valid features found for correlation matrix")
            return
        numeric_data = data[available_features].select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        print("No numeric data available for correlation matrix")
        return
    
    # Calculate correlation matrix
    correlation_matrix = numeric_data.corr()
    
    # Create plot
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=annot, cmap='coolwarm', center=0,
                square=True, linewidths=0.1, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()


def plot_features_by_sleep_stage(data: pd.DataFrame,
                                features: List[str],
                                stage_column: str = 'Sleep_Stage',
                                figsize: tuple = (15, 10)) -> None:
    """
    Plot box plots of features grouped by sleep stage.
    
    Args:
        data: DataFrame containing features and sleep stages
        features: List of feature names to plot
        stage_column: Name of the sleep stage column
        figsize: Figure size
    """
    if stage_column not in data.columns:
        print(f"Sleep stage column '{stage_column}' not found")
        return
    
    available_features = [f for f in features if f in data.columns]
    if not available_features:
        print("No valid features found")
        return
    
    n_features = len(available_features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_features == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_features > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for i, feature in enumerate(available_features):
        data.boxplot(column=feature, by=stage_column, ax=axes[i])
        axes[i].set_title(f'{feature} by Sleep Stage')
        axes[i].set_xlabel('Sleep Stage')
        axes[i].set_ylabel(feature)
        axes[i].tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Feature Distributions by Sleep Stage')
    plt.tight_layout()
    plt.show()
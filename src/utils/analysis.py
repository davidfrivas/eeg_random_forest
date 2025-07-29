"""
Analysis utilities for sleep stage classification.
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union
from collections import Counter
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class SleepStageAnalyzer:
    """Comprehensive analysis utilities for sleep stage classification."""
    
    def __init__(self):
        pass
    
    def analyze_transitions(self, stages: Union[pd.Series, np.ndarray]) -> Dict[str, int]:
        """
        Analyze sleep stage transitions.
        
        Args:
            stages: Array or Series of sleep stages
            
        Returns:
            Dictionary of transition counts
        """
        if isinstance(stages, pd.Series):
            stages = stages.values
            
        transitions = {}
        for i in range(len(stages) - 1):
            current = stages[i]
            next_stage = stages[i + 1]
            transition = f"{current} → {next_stage}"
            transitions[transition] = transitions.get(transition, 0) + 1
            
        return transitions
    
    def compare_transitions(self, y_true: Union[pd.Series, np.ndarray], 
                          y_pred: Union[pd.Series, np.ndarray],
                          top_n: int = 10) -> pd.DataFrame:
        """
        Compare transitions between true and predicted labels.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            top_n: Number of top transitions to return
            
        Returns:
            DataFrame with transition comparison
        """
        true_transitions = self.analyze_transitions(y_true)
        pred_transitions = self.analyze_transitions(y_pred)
        
        all_transition_types = set(list(true_transitions.keys()) + list(pred_transitions.keys()))
        
        comparison_data = []
        for transition in all_transition_types:
            true_count = true_transitions.get(transition, 0)
            pred_count = pred_transitions.get(transition, 0)
            diff = abs(true_count - pred_count)
            relative_error = diff / max(true_count, 1) if true_count > 0 else float('inf')
            
            comparison_data.append({
                'transition': transition,
                'true_count': true_count,
                'pred_count': pred_count,
                'absolute_diff': diff,
                'relative_error': relative_error
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('true_count', ascending=False)
        
        return comparison_df.head(top_n)
    
    def create_performance_summary(self, results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """
        Create a comprehensive performance summary table.
        
        Args:
            results_dict: Dictionary of model results
            
        Returns:
            DataFrame with performance metrics
        """
        summary_data = {
            'Metric': [
                'Overall Accuracy',
                'Balanced Accuracy',
                'Wake F1-Score',
                'NREM F1-Score', 
                'REM F1-Score',
                'Macro Avg F1-Score',
                'Weighted Avg F1-Score'
            ]
        }
        
        for model_name, results in results_dict.items():
            f1_scores = list(results['per_class_metrics']['f1_score'].values())
            support_values = list(results['per_class_metrics']['support'].values())
            
            model_metrics = [
                f"{results['accuracy']:.4f}",
                f"{results['balanced_accuracy']:.4f}",
                f"{results['per_class_metrics']['f1_score'].get('Wake', 0):.4f}",
                f"{results['per_class_metrics']['f1_score'].get('Non REM', 0):.4f}",
                f"{results['per_class_metrics']['f1_score'].get('REM', 0):.4f}",
                f"{np.mean(f1_scores):.4f}",
                f"{np.average(f1_scores, weights=support_values):.4f}"
            ]
            
            summary_data[model_name] = model_metrics
        
        return pd.DataFrame(summary_data)
    
    def evaluate_feature_combinations(self, prepared_data: Dict, 
                                    classifier_class,
                                    test_size: float = 0.3,
                                    random_state: int = 42) -> pd.DataFrame:
        """
        Evaluate different feature combinations.
        
        Args:
            prepared_data: Dictionary of prepared feature combinations
            classifier_class: SleepStageClassifier class
            test_size: Test set proportion
            random_state: Random state for reproducibility
            
        Returns:
            DataFrame with feature combination results
        """
        combination_results = {}
        
        for combination, (X_combo, y_combo) in prepared_data.items():
            logger.info(f"Testing {combination} features ({X_combo.shape[1]} features)...")
            
            try:
                # Quick train/test split for this combination
                X_train_combo, X_test_combo, y_train_combo, y_test_combo = train_test_split(
                    X_combo, y_combo, test_size=test_size, random_state=random_state, 
                    stratify=y_combo
                )
                
                # Train a quick standard model
                temp_classifier = classifier_class()
                temp_classifier.feature_names = X_combo.columns.tolist()
                
                temp_results = temp_classifier.train_standard_model(
                    X_train_combo, y_train_combo, X_test_combo, y_test_combo,
                    apply_postprocessing=False,
                    save_model=False,
                    model_dir=None
                )
                
                combination_results[combination] = {
                    'n_features': X_combo.shape[1],
                    'balanced_accuracy': temp_results['balanced_accuracy'],
                    'accuracy': temp_results['accuracy'],
                    'wake_f1': temp_results['per_class_metrics']['f1_score'].get('Wake', 0),
                    'nrem_f1': temp_results['per_class_metrics']['f1_score'].get('Non REM', 0),
                    'rem_f1': temp_results['per_class_metrics']['f1_score'].get('REM', 0)
                }
                
                logger.info(f"  {combination}: Balanced Accuracy = {temp_results['balanced_accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {combination}: {e}")
                combination_results[combination] = {
                    'n_features': X_combo.shape[1],
                    'balanced_accuracy': 0,
                    'accuracy': 0,
                    'wake_f1': 0,
                    'nrem_f1': 0,
                    'rem_f1': 0
                }
        
        combo_df = pd.DataFrame(combination_results).T
        combo_df = combo_df.sort_values('balanced_accuracy', ascending=False)
        
        return combo_df
    
    def plot_transition_heatmap(self, y_true: Union[pd.Series, np.ndarray], 
                          y_pred: Union[pd.Series, np.ndarray],
                          save_path: str = None) -> plt.Figure:
      """
      Create a heatmap of sleep stage transitions.
      
      Args:
          y_true: True labels
          y_pred: Predicted labels
          save_path: Path to save the plot
          
      Returns:
          Figure object
      """
      # Convert to numpy arrays to avoid indexing issues
      if isinstance(y_true, pd.Series):
          y_true = y_true.values
      if isinstance(y_pred, pd.Series):
          y_pred = y_pred.values
      
      # Ensure arrays are the same length
      min_length = min(len(y_true), len(y_pred))
      y_true = y_true[:min_length]
      y_pred = y_pred[:min_length]
      
      # Get all unique stages
      stages = sorted(set(list(y_true) + list(y_pred)))
      
      # Create transition matrices
      true_matrix = np.zeros((len(stages), len(stages)))
      pred_matrix = np.zeros((len(stages), len(stages)))
      
      stage_to_idx = {stage: i for i, stage in enumerate(stages)}
      
      # Fill true transition matrix
      for i in range(len(y_true) - 1):
          curr_stage = y_true[i]
          next_stage = y_true[i + 1]
          
          # Skip if stage not in our mapping
          if curr_stage in stage_to_idx and next_stage in stage_to_idx:
              curr_idx = stage_to_idx[curr_stage]
              next_idx = stage_to_idx[next_stage]
              true_matrix[curr_idx, next_idx] += 1
      
      # Fill predicted transition matrix
      for i in range(len(y_pred) - 1):
          curr_stage = y_pred[i]
          next_stage = y_pred[i + 1]
          
          # Skip if stage not in our mapping
          if curr_stage in stage_to_idx and next_stage in stage_to_idx:
              curr_idx = stage_to_idx[curr_stage]
              next_idx = stage_to_idx[next_stage]
              pred_matrix[curr_idx, next_idx] += 1
      
      # Normalize by row sums (avoid division by zero)
      true_row_sums = true_matrix.sum(axis=1, keepdims=True)
      true_row_sums[true_row_sums == 0] = 1  # Avoid division by zero
      true_matrix_norm = true_matrix / true_row_sums
      
      pred_row_sums = pred_matrix.sum(axis=1, keepdims=True)
      pred_row_sums[pred_row_sums == 0] = 1  # Avoid division by zero
      pred_matrix_norm = pred_matrix / pred_row_sums
      
      # Create subplots
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
      
      # True transitions
      sns.heatmap(true_matrix_norm, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=stages, yticklabels=stages, ax=ax1,
                cbar_kws={'label': 'Transition Probability'})
      ax1.set_title('True Sleep Stage Transitions')
      ax1.set_xlabel('Next Stage')
      ax1.set_ylabel('Current Stage')
      
      # Predicted transitions
      sns.heatmap(pred_matrix_norm, annot=True, fmt='.3f', cmap='Reds',
                xticklabels=stages, yticklabels=stages, ax=ax2,
                cbar_kws={'label': 'Transition Probability'})
      ax2.set_title('Predicted Sleep Stage Transitions')
      ax2.set_xlabel('Next Stage')
      ax2.set_ylabel('Current Stage')
      
      plt.tight_layout()
      
      if save_path:
          plt.savefig(save_path, dpi=300, bbox_inches='tight')
          logger.info(f"Saved transition heatmap to {save_path}")
      
      # Log some statistics for debugging
      logger.info(f"Transition analysis completed:")
      logger.info(f"  - True transitions analyzed: {len(y_true) - 1}")
      logger.info(f"  - Predicted transitions analyzed: {len(y_pred) - 1}")
      logger.info(f"  - Unique stages found: {len(stages)} - {stages}")
      
      return fig
    
    def plot_feature_combination_comparison(self, combo_df: pd.DataFrame,
                                          save_path: str = None) -> plt.Figure:
        """
        Plot feature combination comparison.
        
        Args:
            combo_df: DataFrame with feature combination results
            save_path: Path to save the plot
            
        Returns:
            Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Balanced accuracy by combination
        axes[0, 0].bar(combo_df.index, combo_df['balanced_accuracy'], 
                      color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Balanced Accuracy by Feature Combination')
        axes[0, 0].set_ylabel('Balanced Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Accuracy vs number of features
        axes[0, 1].scatter(combo_df['n_features'], combo_df['balanced_accuracy'], 
                          s=100, c='lightcoral', alpha=0.7)
        for i, combo in enumerate(combo_df.index):
            axes[0, 1].annotate(combo, (combo_df.loc[combo, 'n_features'], 
                               combo_df.loc[combo, 'balanced_accuracy']),
                              xytext=(5, 5), textcoords='offset points', fontsize=9)
        axes[0, 1].set_xlabel('Number of Features')
        axes[0, 1].set_ylabel('Balanced Accuracy')
        axes[0, 1].set_title('Accuracy vs Number of Features')
        
        # F1 scores by class
        f1_data = combo_df[['wake_f1', 'nrem_f1', 'rem_f1']]
        f1_data.plot(kind='bar', ax=axes[1, 0], alpha=0.7)
        axes[1, 0].set_title('F1 Scores by Sleep Stage')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend(['Wake', 'NREM', 'REM'])
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Performance radar chart (simplified)
        metrics = ['balanced_accuracy', 'wake_f1', 'nrem_f1', 'rem_f1']
        best_combo = combo_df.index[0]  # Best performing combination
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        values = combo_df.loc[best_combo, metrics].tolist()
        values += values[:1]  # Complete the circle
        
        axes[1, 1] = plt.subplot(2, 2, 4, projection='polar')
        axes[1, 1].plot(angles, values, 'o-', linewidth=2, label=best_combo)
        axes[1, 1].fill(angles, values, alpha=0.25)
        axes[1, 1].set_xticks(angles[:-1])
        axes[1, 1].set_xticklabels(['Bal. Acc.', 'Wake F1', 'NREM F1', 'REM F1'])
        axes[1, 1].set_title(f'Best Model Performance: {best_combo}')
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature combination plot to {save_path}")
        
        return fig
    
    def generate_comprehensive_report(self, results_dict: Dict[str, Dict],
                                    y_true: Union[pd.Series, np.ndarray],
                                    y_pred_dict: Dict[str, Union[pd.Series, np.ndarray]],
                                    combo_df: pd.DataFrame = None,
                                    output_path: str = None) -> str:
        """
        Generate a comprehensive analysis report.
        
        Args:
            results_dict: Dictionary of model results
            y_true: True labels
            y_pred_dict: Dictionary of predictions for each model
            combo_df: Feature combination results
            output_path: Path to save the report
            
        Returns:
            Report as string
        """
        report_lines = [
            "Sleep Stage Classification - Comprehensive Analysis Report",
            "=" * 65,
            "",
            f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "EXECUTIVE SUMMARY",
            "-" * 20,
        ]
        
        # Performance summary
        summary_df = self.create_performance_summary(results_dict)
        report_lines.extend([
            "",
            "Model Performance Summary:",
            summary_df.to_string(index=False),
            ""
        ])
        
        # Best model identification
        best_model = None
        best_balanced_acc = 0
        for model_name, results in results_dict.items():
            if results['balanced_accuracy'] > best_balanced_acc:
                best_balanced_acc = results['balanced_accuracy']
                best_model = model_name
        
        report_lines.extend([
            f"Best Performing Model: {best_model}",
            f"Best Balanced Accuracy: {best_balanced_acc:.4f}",
            ""
        ])
        
        # Transition analysis for best model
        if best_model in y_pred_dict:
            transition_df = self.compare_transitions(y_true, y_pred_dict[best_model])
            report_lines.extend([
                "SLEEP STAGE TRANSITION ANALYSIS",
                "-" * 35,
                "",
                f"Top 10 Sleep Stage Transitions ({best_model} Model):",
                transition_df.to_string(index=False),
                ""
            ])
        
        # Feature combination analysis
        if combo_df is not None:
            report_lines.extend([
                "FEATURE COMBINATION ANALYSIS",
                "-" * 30,
                "",
                "Feature Combination Performance Ranking:",
                combo_df.round(4).to_string(),
                ""
            ])
        
        # Detailed model results
        for model_name, results in results_dict.items():
            report_lines.extend([
                f"DETAILED RESULTS - {model_name.upper()} MODEL",
                "-" * 50,
                "",
                "Classification Report:",
                results['classification_report'],
                ""
            ])
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Saved comprehensive report to {output_path}")
        
        return report_text
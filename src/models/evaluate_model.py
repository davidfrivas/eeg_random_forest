"""
Model evaluation utilities.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix, balanced_accuracy_score,
    precision_recall_fscore_support, accuracy_score
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate trained models and generate reports."""
    
    def __init__(self, save_plots: bool = True):
        self.save_plots = save_plots
    
    def evaluate_model(self, y_true: Union[pd.Series, np.ndarray], 
                      y_pred: np.ndarray,
                      model: Optional[object] = None,
                      feature_names: Optional[List[str]] = None,
                      output_dir: Optional[Path] = None) -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model: Trained model (for feature importance)
            feature_names: List of feature names
            output_dir: Directory to save plots and reports
            
        Returns:
            Dictionary containing evaluation metrics
        """
        results = {}
        
        # Convert to numpy arrays if needed
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
        
        # Basic metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        # Classification report
        report = classification_report(y_true, y_pred)
        results['classification_report'] = report
        
        # Detailed metrics per class
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=np.unique(y_true)
        )
        
        results['per_class_metrics'] = {
            'precision': dict(zip(np.unique(y_true), precision)),
            'recall': dict(zip(np.unique(y_true), recall)),
            'f1_score': dict(zip(np.unique(y_true), f1)),
            'support': dict(zip(np.unique(y_true), support))
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm
        
        # Feature importance (if model provided)
        if model is not None and hasattr(model, 'feature_importances_'):
            importance_df = self._get_feature_importance(model, feature_names)
            results['feature_importance'] = importance_df
            logger.info(f"Calculated feature importance for {len(importance_df)} features")
        
        # Generate visualizations if requested
        if self.save_plots and output_dir:
            self._generate_plots(y_true, y_pred, results, output_dir)
        
        # Log results
        logger.info(f"Model Evaluation Results:")
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
        logger.info(f"\nClassification Report:\n{report}")
        
        return results
    
    def compare_models(self, results_list: List[Dict], 
                      model_names: List[str],
                      output_dir: Optional[Path] = None) -> pd.DataFrame:
        """
        Compare multiple model results.
        
        Args:
            results_list: List of evaluation result dictionaries
            model_names: Names of the models
            output_dir: Directory to save comparison plots
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        for results, name in zip(results_list, model_names):
            comparison_data.append({
                'model': name,
                'accuracy': results['accuracy'],
                'balanced_accuracy': results['balanced_accuracy'],
                **{f'{stage}_f1': results['per_class_metrics']['f1_score'].get(stage, 0)
                   for stage in ['Wake', 'Non REM', 'REM']}
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if output_dir and self.save_plots:
            self._plot_model_comparison(comparison_df, output_dir)
        
        return comparison_df
    
    def _get_feature_importance(self, model, feature_names: Optional[List[str]]) -> pd.DataFrame:
        """Extract and format feature importance with proper feature names."""
        importances = model.feature_importances_
        
        # Use provided feature names, or create generic ones if none provided
        if feature_names is not None and len(feature_names) == len(importances):
            names = feature_names
            logger.info(f"Using provided feature names: {len(names)} features")
        else:
            names = [f'feature_{i}' for i in range(len(importances))]
            if feature_names is not None:
                logger.warning(f"Feature name count mismatch: provided {len(feature_names)}, "
                            f"expected {len(importances)}. Using generic names.")
            else:
                logger.warning("No feature names provided, using generic names.")
        
        importance_df = pd.DataFrame({
            'feature': names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Log top features for debugging
        logger.info(f"Top 5 important features: {importance_df.head()['feature'].tolist()}")
        
        return importance_df
    
    def _generate_plots(self, y_true, y_pred, results: Dict, output_dir: Path):
        """Generate and save evaluation plots."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Confusion matrix
        self._plot_confusion_matrix(
            results['confusion_matrix'], 
            np.unique(y_true), 
            output_dir / 'confusion_matrix.png'
        )
        
        # Feature importance (if available)
        if 'feature_importance' in results:
            self._plot_feature_importance(
                results['feature_importance'],
                output_dir / 'feature_importance.png'
            )
        
        # Class distribution comparison - FIXED
        self._plot_class_distribution(
            y_true, y_pred, 
            output_dir / 'class_distribution.png'
        )
    
    def _plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], 
                             save_path: Path):
        """Plot and save confusion matrix."""
        # Normalize by row (true labels)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix (Normalized)')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion matrix to {save_path}")
    
    def _plot_feature_importance(self, importance_df: pd.DataFrame, 
                               save_path: Path, top_n: int = 20):
        """Plot and save feature importance."""
        plt.figure(figsize=(12, 8))
        
        # Plot top N features
        top_features = importance_df.head(top_n)
        
        # Create horizontal bar plot
        plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()  # Highest importance at top
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved feature importance plot to {save_path}")
    
    def _plot_class_distribution(self, y_true, y_pred, save_path: Path):
        """Plot class distribution comparison - FIXED VERSION."""
        try:
            # Get unique classes from both true and predicted labels
            all_classes = sorted(set(list(y_true) + list(y_pred)))
            
            # Count occurrences
            true_counts = pd.Series(y_true).value_counts()
            pred_counts = pd.Series(y_pred).value_counts()
            
            # Ensure all classes are represented (fill missing with 0)
            for cls in all_classes:
                if cls not in true_counts:
                    true_counts[cls] = 0
                if cls not in pred_counts:
                    pred_counts[cls] = 0
            
            # Sort by class name for consistent ordering
            true_counts = true_counts.reindex(all_classes)
            pred_counts = pred_counts.reindex(all_classes)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # True distribution
            colors1 = plt.cm.Set3(np.linspace(0, 1, len(all_classes)))
            wedges1, texts1, autotexts1 = ax1.pie(
                true_counts.values, 
                labels=true_counts.index, 
                autopct='%1.1f%%',
                colors=colors1,
                startangle=90
            )
            ax1.set_title('True Class Distribution')
            
            # Predicted distribution
            colors2 = plt.cm.Set3(np.linspace(0, 1, len(all_classes)))
            wedges2, texts2, autotexts2 = ax2.pie(
                pred_counts.values, 
                labels=pred_counts.index, 
                autopct='%1.1f%%',
                colors=colors2,
                startangle=90
            )
            ax2.set_title('Predicted Class Distribution')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved class distribution plot to {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating class distribution plot: {e}")
            logger.error(f"y_true shape: {np.array(y_true).shape}, unique values: {np.unique(y_true)}")
            logger.error(f"y_pred shape: {np.array(y_pred).shape}, unique values: {np.unique(y_pred)}")
            
            # Create a fallback simple bar plot
            try:
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                
                all_classes = sorted(set(list(y_true) + list(y_pred)))
                true_counts = [sum(y_true == cls) for cls in all_classes]
                pred_counts = [sum(y_pred == cls) for cls in all_classes]
                
                x = np.arange(len(all_classes))
                width = 0.35
                
                ax.bar(x - width/2, true_counts, width, label='True', alpha=0.8)
                ax.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
                
                ax.set_xlabel('Sleep Stage')
                ax.set_ylabel('Count')
                ax.set_title('Class Distribution Comparison')
                ax.set_xticks(x)
                ax.set_xticklabels(all_classes, rotation=45)
                ax.legend()
                
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Saved fallback class distribution plot to {save_path}")
                
            except Exception as e2:
                logger.error(f"Failed to create fallback plot: {e2}")
    
    def _plot_model_comparison(self, comparison_df: pd.DataFrame, output_dir: Path):
        """Plot model comparison metrics."""
        # Metrics to compare
        metrics = ['accuracy', 'balanced_accuracy', 'Wake_f1', 'Non REM_f1', 'REM_f1']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i < len(axes) and metric in comparison_df.columns:
                ax = axes[i]
                sns.barplot(data=comparison_df, x='model', y=metric, ax=ax)
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved model comparison plot to {output_dir / 'model_comparison.png'}")
    
    def generate_report(self, results: Dict, model_name: str, 
                       output_file: Path) -> None:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results: Evaluation results dictionary
            model_name: Name of the model
            output_file: Path to save the report
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write(f"Sleep Stage Classification - Model Evaluation Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Model: {model_name}\n\n")
            
            # Overall metrics
            f.write("Overall Performance:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}\n\n")
            
            # Per-class metrics
            f.write("Per-Class Performance:\n")
            f.write("-" * 22 + "\n")
            for stage in results['per_class_metrics']['precision']:
                precision = results['per_class_metrics']['precision'][stage]
                recall = results['per_class_metrics']['recall'][stage]
                f1 = results['per_class_metrics']['f1_score'][stage]
                support = results['per_class_metrics']['support'][stage]
                
                f.write(f"{stage}:\n")
                f.write(f"  Precision: {precision:.4f}\n")
                f.write(f"  Recall: {recall:.4f}\n")
                f.write(f"  F1-Score: {f1:.4f}\n")
                f.write(f"  Support: {support}\n\n")
            
            # Classification report
            f.write("Detailed Classification Report:\n")
            f.write("-" * 32 + "\n")
            f.write(results['classification_report'])
            f.write("\n")
            
            # Feature importance (if available)
            if 'feature_importance' in results:
                f.write("Top 15 Most Important Features:\n")
                f.write("-" * 33 + "\n")
                top_features = results['feature_importance'].head(15)
                for _, row in top_features.iterrows():
                    f.write(f"{row['feature']}: {row['importance']:.6f}\n")
                f.write("\n")
            
            # Confusion matrix
            f.write("Confusion Matrix:\n")
            f.write("-" * 17 + "\n")
            f.write(str(results['confusion_matrix']))
            f.write("\n")
        
        logger.info(f"Saved evaluation report to {output_file}")


def evaluate_saved_model(model_path: Path, test_data: pd.DataFrame,
                        feature_names: List[str], output_dir: Path) -> Dict:
    """
    Evaluate a saved model on test data.
    
    Args:
        model_path: Path to the saved model
        test_data: Test dataset with features and Sleep_Stage
        feature_names: List of feature names to use
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary containing evaluation metrics
    """
    import joblib
    
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    # Load feature names if available
    feature_names_path = model_path.parent / f"{model_path.stem.replace('_model', '')}_feature_names.joblib"
    if feature_names_path.exists():
        saved_feature_names = joblib.load(feature_names_path)
        logger.info(f"Loaded saved feature names: {len(saved_feature_names)} features")
        feature_names = saved_feature_names
    
    # Prepare test data
    X_test = test_data[feature_names]
    y_test = test_data['Sleep_Stage']
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    evaluator = ModelEvaluator(save_plots=True)
    results = evaluator.evaluate_model(y_test, y_pred, model, feature_names, output_dir)
    
    # Generate report
    model_name = model_path.stem
    evaluator.generate_report(results, model_name, output_dir / f"{model_name}_report.txt")
    
    return results
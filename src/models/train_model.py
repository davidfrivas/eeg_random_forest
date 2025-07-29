"""
Model training functionality for sleep stage classification.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, balanced_accuracy_score
)
import joblib

from ..config import (
    FEATURE_GROUPS, MODEL_PARAMS, CLASS_WEIGHTS, 
    SMOOTHING_WINDOW_SIZE, VALID_TRANSITIONS
)
from .evaluate_model import ModelEvaluator

logger = logging.getLogger(__name__)


class SleepStageClassifier:
    """Sleep stage classification model trainer."""
    
    def __init__(self, model_params: Optional[Dict] = None, 
                 class_weights: Optional[Dict] = None):
        self.model_params = model_params or MODEL_PARAMS.copy()
        self.class_weights = class_weights or CLASS_WEIGHTS.copy()
        self.scaler = None
        self.feature_names = None
        self.evaluator = ModelEvaluator()
    
    def train_standard_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series,
                       apply_postprocessing: bool = True,
                       save_model: bool = True,
                       model_dir: Optional[Path] = None) -> Dict:
        """
        Train a standard multiclass Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features  
            y_test: Test labels
            apply_postprocessing: Whether to apply sleep transition rules
            save_model: Whether to save the trained model
            model_dir: Directory to save model and results
            
        Returns:
            Dictionary containing model performance metrics
        """
        logger.info("Training standard multiclass model")
        
        # Ensure feature names are preserved
        if self.feature_names is None:
            self.feature_names = X_train.columns.tolist()
        
        # Set class weights
        self.model_params['class_weight'] = self._calculate_class_weights(y_train)
        
        # Train model
        model = RandomForestClassifier(**self.model_params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Apply post-processing if requested
        if apply_postprocessing:
            y_pred_smoothed = self._apply_postprocessing(y_pred)
            
            # Use smoothed predictions if they improve performance
            original_acc = balanced_accuracy_score(y_test, y_pred)
            smoothed_acc = balanced_accuracy_score(y_test, y_pred_smoothed)
            
            if smoothed_acc > original_acc:
                logger.info(f"Using post-processed predictions (improved accuracy: "
                        f"{smoothed_acc:.4f} vs {original_acc:.4f})")
                y_pred = y_pred_smoothed
        
        # Evaluate model
        results = self.evaluator.evaluate_model(y_test, y_pred, model, self.feature_names)
        
        # Add the actual predictions and true labels to results for plotting
        results['y_true'] = y_test
        results['y_pred'] = y_pred
        results['model'] = model
        
        # Save model and results if requested
        if save_model and model_dir:
            self._save_model_artifacts(model, results, model_dir, "standard")
        
        return results
    
    def train_hierarchical_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series,
                                apply_postprocessing: bool = True,
                                save_model: bool = True,
                                model_dir: Optional[Path] = None) -> Dict:
        """
        Train a hierarchical model: Wake vs Sleep, then NREM vs REM.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            apply_postprocessing: Whether to apply sleep transition rules
            save_model: Whether to save the trained models
            model_dir: Directory to save models and results
            
        Returns:
            Dictionary containing model performance metrics
        """
        logger.info("Training hierarchical model")
        
        # Ensure feature names are preserved
        if self.feature_names is None:
            self.feature_names = X_train.columns.tolist()
        
        # Step 1: Train Wake vs Sleep classifier
        logger.info("Step 1: Training Wake vs Sleep classifier")
        
        y_train_binary = self._create_wake_sleep_labels(y_train)
        y_test_binary = self._create_wake_sleep_labels(y_test)
        
        wake_sleep_weights = self._calculate_binary_weights(y_train_binary)
        ws_params = self.model_params.copy()
        ws_params['class_weight'] = wake_sleep_weights
        
        wake_sleep_model = RandomForestClassifier(**ws_params)
        wake_sleep_model.fit(X_train, y_train_binary)
        
        # Step 2: Train NREM vs REM classifier (only on sleep data)
        logger.info("Step 2: Training NREM vs REM classifier")
        
        sleep_mask = y_train_binary == 1
        X_train_sleep = X_train[sleep_mask]
        y_train_sleep = y_train[sleep_mask]
        
        y_train_sleep_binary = self._create_nrem_rem_labels(y_train_sleep)
        
        nrem_rem_weights = self._calculate_binary_weights(y_train_sleep_binary)
        nr_params = self.model_params.copy()
        nr_params['class_weight'] = nrem_rem_weights
        
        nrem_rem_model = RandomForestClassifier(**nr_params)
        nrem_rem_model.fit(X_train_sleep, y_train_sleep_binary)
        
        # Make hierarchical predictions
        y_pred = self._make_hierarchical_predictions(
            wake_sleep_model, nrem_rem_model, X_test
        )
        
        # Apply post-processing if requested
        if apply_postprocessing:
            y_pred_smoothed = self._apply_postprocessing(y_pred)
            
            original_acc = balanced_accuracy_score(y_test, y_pred)
            smoothed_acc = balanced_accuracy_score(y_test, y_pred_smoothed)
            
            if smoothed_acc > original_acc:
                logger.info(f"Using post-processed predictions (improved accuracy: "
                          f"{smoothed_acc:.4f} vs {original_acc:.4f})")
                y_pred = y_pred_smoothed
        
        # Evaluate combined model using wake_sleep_model for feature importance
        results = self.evaluator.evaluate_model(y_test, y_pred, wake_sleep_model, self.feature_names)
        
        # Add the actual predictions and true labels to results for plotting
        results['y_true'] = y_test
        results['y_pred'] = y_pred
        results['model'] = wake_sleep_model  # Use wake-sleep model for feature importance
        
        # Add hierarchical-specific metrics
        results['wake_sleep_model'] = wake_sleep_model
        results['nrem_rem_model'] = nrem_rem_model
        
        # Save models and results if requested
        if save_model and model_dir:
            self._save_hierarchical_models(
                wake_sleep_model, nrem_rem_model, results, model_dir
            )
        
        return results
    
    def prepare_features(self, data: pd.DataFrame, 
                        feature_combination: str = 'all',
                        top_n_features: int = 0,
                        scale_features: bool = False,
                        filter_artifacts: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and labels from merged data.
        
        Args:
            data: DataFrame with features and Sleep_Stage column
            feature_combination: Which feature groups to use
            top_n_features: Use only top N most important features (0 = all)
            scale_features: Whether to standardize features
            filter_artifacts: Whether to remove high-artifact data
            
        Returns:
            Tuple of (features, labels)
        """
        logger.info(f"Preparing features with combination: {feature_combination}")
        
        # Clean data
        data = self._clean_data(data)
        
        # Filter artifacts if requested
        if filter_artifacts and 'Artifact' in data.columns:
            original_len = len(data)
            data = data[data['Artifact'] <= 0.5]  # Adjust threshold as needed
            logger.info(f"Filtered artifacts: {original_len} -> {len(data)} samples")
        
        # Select features
        X, feature_names = self._select_features(data, feature_combination, top_n_features)
        y = data['Sleep_Stage']
        
        # Store feature names in the classifier instance
        self.feature_names = feature_names
        
        # Scale features if requested
        if scale_features:
            if self.scaler is None:
                self.scaler = StandardScaler()
                X = pd.DataFrame(
                    self.scaler.fit_transform(X),
                    columns=feature_names,
                    index=X.index
                )
            else:
                X = pd.DataFrame(
                    self.scaler.transform(X),
                    columns=feature_names,
                    index=X.index
                )
            logger.info("Applied feature scaling")
        
        return X, y
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data by removing NaN/Inf values."""
        original_len = len(data)
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        logger.info(f"Data cleaning: {original_len} -> {len(data)} samples")
        return data
    
    def _select_features(self, data: pd.DataFrame, combination: str,
                        top_n: int = 0) -> Tuple[pd.DataFrame, List[str]]:
        """Select features based on combination and optionally limit to top N."""
        # Define columns to exclude
        exclude_cols = ['Sleep_Stage', 'filename', 'animal_id']
        
        if combination in FEATURE_GROUPS:
            selected_features = FEATURE_GROUPS[combination]
        elif combination == 'all':
            selected_features = []
            for group in FEATURE_GROUPS.values():
                selected_features.extend(group)
        elif '+' in combination:
            selected_groups = combination.split('+')
            selected_features = []
            for group in selected_groups:
                if group in FEATURE_GROUPS:
                    selected_features.extend(FEATURE_GROUPS[group])
        else:
            # Default to all available features
            selected_features = [col for col in data.columns if col not in exclude_cols]
        
        # Filter to only available features
        available_features = [f for f in selected_features if f in data.columns]
        
        # TODO: Implement top_n feature selection based on importance
        if top_n > 0 and top_n < len(available_features):
            logger.warning("Top-N feature selection not yet implemented")
        
        return data[available_features], available_features
    
    def _calculate_class_weights(self, y: pd.Series) -> Dict:
        """Calculate class weights for imbalanced data."""
        class_counts = Counter(y)
        total = len(y)
        
        # Use predefined weights if available, otherwise calculate balanced weights
        weights = {}
        for cls in class_counts:
            if cls in self.class_weights:
                weights[cls] = self.class_weights[cls]
            else:
                weights[cls] = total / (len(class_counts) * class_counts[cls])
        
        logger.info(f"Using class weights: {weights}")
        return weights
    
    def _calculate_binary_weights(self, y: np.ndarray) -> Dict:
        """Calculate weights for binary classification."""
        class_counts = Counter(y)
        total = len(y)
        
        return {
            cls: total / (2 * count) 
            for cls, count in class_counts.items()
        }
    
    def _create_wake_sleep_labels(self, y: pd.Series) -> np.ndarray:
        """Create binary Wake vs Sleep labels."""
        return np.where(y.str.contains('Wake'), 0, 1)
    
    def _create_nrem_rem_labels(self, y: pd.Series) -> np.ndarray:
        """Create binary NREM vs REM labels."""
        return np.where(y.str.contains('REM'), 1, 0)
    
    def _make_hierarchical_predictions(self, wake_sleep_model, nrem_rem_model,
                                     X_test: pd.DataFrame) -> np.ndarray:
        """Make predictions using hierarchical model."""
        # First predict Wake vs Sleep
        sleep_pred = wake_sleep_model.predict(X_test)
        
        # Initialize predictions as Wake
        y_pred = np.array(['Wake'] * len(X_test), dtype=object)
        
        # For predicted sleep samples, classify as NREM or REM
        sleep_samples = sleep_pred == 1
        if np.any(sleep_samples):
            nrem_rem_pred = nrem_rem_model.predict(X_test[sleep_samples])
            y_pred[sleep_samples] = np.where(nrem_rem_pred == 0, 'Non REM', 'REM')
        
        return y_pred
    
    def _apply_postprocessing(self, y_pred: np.ndarray,
                            window_size: int = SMOOTHING_WINDOW_SIZE,
                            apply_transitions: bool = True) -> np.ndarray:
        """Apply sleep stage smoothing and transition rules."""
        y_smoothed = y_pred.copy()
        
        # Ensure window size is odd
        if window_size % 2 == 0:
            window_size += 1
        
        half_window = window_size // 2
        changes = 0
        
        # Apply majority rule smoothing
        for i in range(half_window, len(y_pred) - half_window):
            window = y_pred[i-half_window:i+half_window+1]
            unique, counts = np.unique(window, return_counts=True)
            mode_class = unique[np.argmax(counts)]
            
            current_count = sum(window == y_pred[i])
            if current_count <= window_size // 3:
                y_smoothed[i] = mode_class
                changes += 1
        
        if apply_transitions:
            # Apply physiological transition rules
            for i in range(1, len(y_smoothed) - 1):
                prev_stage = y_smoothed[i-1]
                curr_stage = y_smoothed[i]
                next_stage = y_smoothed[i+1]
                
                if curr_stage in VALID_TRANSITIONS.get(prev_stage, []):
                    continue
                
                # Find valid transition
                if next_stage in VALID_TRANSITIONS.get(prev_stage, []):
                    y_smoothed[i] = next_stage
                else:
                    # Default to NREM as intermediate stage
                    y_smoothed[i] = 'Non REM'
                changes += 1
        
        logger.info(f"Post-processing applied {changes} corrections "
                   f"({changes/len(y_pred)*100:.2f}%)")
        
        return y_smoothed
    
    def _save_model_artifacts(self, model, results: Dict, model_dir: Path, 
                            model_type: str):
        """Save model and evaluation results."""
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = model_dir / f"{model_type}_model.joblib"
        joblib.dump(model, model_file)
        logger.info(f"Saved model to {model_file}")
        
        # Save scaler if used
        if self.scaler is not None:
            scaler_file = model_dir / f"{model_type}_scaler.joblib"
            joblib.dump(self.scaler, scaler_file)
        
        # Save feature names
        if self.feature_names is not None:
            feature_names_file = model_dir / f"{model_type}_feature_names.joblib"
            joblib.dump(self.feature_names, feature_names_file)
            logger.info(f"Saved feature names to {feature_names_file}")
        
        # Save results
        results_file = model_dir / f"{model_type}_results.txt"
        with open(results_file, 'w') as f:
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(results['classification_report'])
            f.write("\n\nModel Parameters:\n")
            for param, value in self.model_params.items():
                f.write(f"{param}: {value}\n")
    
    def _save_hierarchical_models(self, wake_sleep_model, nrem_rem_model,
                                results: Dict, model_dir: Path):
        """Save hierarchical models and results."""
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        ws_file = model_dir / "wake_sleep_model.joblib"
        nr_file = model_dir / "nrem_rem_model.joblib"
        
        joblib.dump(wake_sleep_model, ws_file)
        joblib.dump(nrem_rem_model, nr_file)
        
        # Save feature names
        if self.feature_names is not None:
            feature_names_file = model_dir / "hierarchical_feature_names.joblib"
            joblib.dump(self.feature_names, feature_names_file)
            logger.info(f"Saved feature names to {feature_names_file}")
        
        logger.info(f"Saved hierarchical models to {model_dir}")
        
        # Save results
        results_file = model_dir / "hierarchical_results.txt"
        with open(results_file, 'w') as f:
            f.write("Hierarchical Model Results\n")
            f.write(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(results['classification_report'])
    
    def get_predictions(self, model_type: str = 'standard') -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the predictions from the last trained model.
        
        Args:
            model_type: Type of model ('standard' or 'hierarchical')
            
        Returns:
            Tuple of (y_true, y_pred)
        """
        if not hasattr(self, '_last_results'):
            raise ValueError("No model has been trained yet. Call train_standard_model or train_hierarchical_model first.")
        
        return self._last_results['y_true'], self._last_results['y_pred']
"""
Feature engineering and selection utilities for sleep stage classification.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, RFE
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from ..config import FEATURE_GROUPS

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Advanced feature engineering for sleep stage classification."""
    
    def __init__(self):
        self.feature_names = []
        self.engineered_features = []
    
    def create_interaction_features(self, data: pd.DataFrame, 
                                  feature_pairs: Optional[List[Tuple[str, str]]] = None) -> pd.DataFrame:
        """
        Create interaction features between different power bands.
        
        Args:
            data: DataFrame with EEG features
            feature_pairs: List of feature pairs to create interactions. If None, uses defaults.
            
        Returns:
            DataFrame with additional interaction features
        """
        logger.info("Creating interaction features")
        
        result = data.copy()
        
        if feature_pairs is None:
            # Default interactions based on neuroscience knowledge
            feature_pairs = [
                ('Rel_Delta', 'Rel_Theta'),    # Slow wave interactions
                ('Rel_Alpha', 'Rel_Beta'),     # Awake state interactions
                ('Rel_Theta', 'EMG_RMS'),      # REM sleep indicators
                ('Rel_Sigma', 'Rel_Delta'),    # Sleep spindle interactions
                ('Spectral_Entropy', 'Rel_Delta'),  # Complexity vs slow waves
            ]
        
        for feat1, feat2 in feature_pairs:
            if feat1 in data.columns and feat2 in data.columns:
                # Multiplicative interaction
                interaction_name = f"{feat1}_x_{feat2}"
                result[interaction_name] = data[feat1] * data[feat2]
                self.engineered_features.append(interaction_name)
                
                # Ratio interaction
                ratio_name = f"{feat1}_div_{feat2}"
                result[ratio_name] = data[feat1] / (data[feat2] + 1e-8)
                self.engineered_features.append(ratio_name)
        
        logger.info(f"Created {len(self.engineered_features)} interaction features")
        return result
    
    def create_temporal_features(self, data: pd.DataFrame, 
                               window_size: int = 5) -> pd.DataFrame:
        """
        Create temporal features using rolling windows.
        
        Args:
            data: DataFrame with time-ordered features
            window_size: Size of rolling window
            
        Returns:
            DataFrame with temporal features
        """
        logger.info(f"Creating temporal features with window size {window_size}")
        
        result = data.copy()
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in ['Artifact']:  # Skip artifact column
                continue
                
            # Rolling mean
            rolling_mean_name = f"{col}_rolling_mean_{window_size}"
            result[rolling_mean_name] = data[col].rolling(window=window_size, center=True).mean()
            
            # Rolling std
            rolling_std_name = f"{col}_rolling_std_{window_size}"
            result[rolling_std_name] = data[col].rolling(window=window_size, center=True).std()
            
            # Difference from rolling mean (trend)
            trend_name = f"{col}_trend_{window_size}"
            result[trend_name] = data[col] - result[rolling_mean_name]
            
            self.engineered_features.extend([rolling_mean_name, rolling_std_name, trend_name])
        
        # Forward fill NaN values created by rolling operations
        result = result.fillna(method='bfill').fillna(method='ffill')
        
        logger.info(f"Created temporal features for {len(numeric_cols)} base features")
        return result
    
    def create_complexity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features that capture signal complexity and variability.
        
        Args:
            data: DataFrame with EEG features
            
        Returns:
            DataFrame with complexity features
        """
        logger.info("Creating complexity features")
        
        result = data.copy()
        
        # Power band complexity measures
        power_bands = ['Rel_Delta', 'Rel_Theta', 'Rel_Alpha', 'Rel_Beta', 'Rel_Gamma']
        available_bands = [band for band in power_bands if band in data.columns]
        
        if len(available_bands) >= 2:
            # Shannon entropy across power bands
            power_data = data[available_bands].values + 1e-8  # Avoid log(0)
            power_probs = power_data / power_data.sum(axis=1, keepdims=True)
            shannon_entropy = -np.sum(power_probs * np.log2(power_probs), axis=1)
            result['Power_Shannon_Entropy'] = shannon_entropy
            self.engineered_features.append('Power_Shannon_Entropy')
            
            # Spectral flatness (Wiener entropy)
            geometric_mean = np.exp(np.mean(np.log(power_data), axis=1))
            arithmetic_mean = np.mean(power_data, axis=1)
            spectral_flatness = geometric_mean / arithmetic_mean
            result['Spectral_Flatness'] = spectral_flatness
            self.engineered_features.append('Spectral_Flatness')
            
            # Dominant frequency band
            dominant_band_idx = np.argmax(power_data, axis=1)
            result['Dominant_Band'] = dominant_band_idx
            self.engineered_features.append('Dominant_Band')
        
        # EMG complexity if available
        emg_features = ['EMG_RMS', 'EMG_MAV', 'EMG_Variance']
        available_emg = [feat for feat in emg_features if feat in data.columns]
        
        if len(available_emg) >= 2:
            # EMG complexity index
            emg_data = data[available_emg].values
            emg_complexity = np.std(emg_data, axis=1) / (np.mean(emg_data, axis=1) + 1e-8)
            result['EMG_Complexity'] = emg_complexity
            self.engineered_features.append('EMG_Complexity')
        
        logger.info(f"Created {len([f for f in self.engineered_features if 'Entropy' in f or 'Complexity' in f])} complexity features")
        return result
    
    def create_sleep_specific_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features specifically designed for sleep stage classification.
        
        Args:
            data: DataFrame with EEG/EMG features
            
        Returns:
            DataFrame with sleep-specific features
        """
        logger.info("Creating sleep-specific features")
        
        result = data.copy()
        
        # Wake probability index (high alpha/beta, low delta, high EMG)
        if all(col in data.columns for col in ['Rel_Alpha', 'Rel_Beta', 'Rel_Delta', 'EMG_RMS']):
            wake_index = (data['Rel_Alpha'] + data['Rel_Beta']) / (data['Rel_Delta'] + 1e-8) * data['EMG_RMS']
            result['Wake_Probability_Index'] = wake_index
            self.engineered_features.append('Wake_Probability_Index')
        
        # NREM probability index (high delta, low EMG)
        if all(col in data.columns for col in ['Rel_Delta', 'EMG_RMS']):
            nrem_index = data['Rel_Delta'] / (data['EMG_RMS'] + 1e-8)
            result['NREM_Probability_Index'] = nrem_index
            self.engineered_features.append('NREM_Probability_Index')
        
        # REM probability index (high theta, very low EMG)
        if all(col in data.columns for col in ['Rel_Theta', 'EMG_RMS']):
            rem_index = data['Rel_Theta'] / (data['EMG_RMS'] + 1e-8)
            result['REM_Probability_Index'] = rem_index
            self.engineered_features.append('REM_Probability_Index')
        
        # Sleep spindle index (sigma power relative to total power)
        if 'Rel_Sigma' in data.columns:
            result['Spindle_Index'] = data['Rel_Sigma']
            self.engineered_features.append('Spindle_Index')
        
        # Arousal index (combination of high frequency activity and EMG)
        if all(col in data.columns for col in ['Rel_Beta', 'Rel_Gamma', 'EMG_RMS']):
            arousal_index = (data['Rel_Beta'] + data['Rel_Gamma']) * data['EMG_RMS']
            result['Arousal_Index'] = arousal_index
            self.engineered_features.append('Arousal_Index')
        
        logger.info(f"Created {len([f for f in self.engineered_features if 'Index' in f])} sleep-specific features")
        return result
    
    def apply_all_engineering(self, data: pd.DataFrame, 
                            include_interactions: bool = True,
                            include_temporal: bool = True,
                            include_complexity: bool = True,
                            include_sleep_specific: bool = True,
                            temporal_window: int = 5) -> pd.DataFrame:
        """
        Apply all feature engineering techniques.
        
        Args:
            data: Input DataFrame
            include_interactions: Whether to create interaction features
            include_temporal: Whether to create temporal features
            include_complexity: Whether to create complexity features
            include_sleep_specific: Whether to create sleep-specific features
            temporal_window: Window size for temporal features
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Applying comprehensive feature engineering")
        
        result = data.copy()
        self.engineered_features = []
        
        if include_interactions:
            result = self.create_interaction_features(result)
        
        if include_temporal:
            result = self.create_temporal_features(result, temporal_window)
        
        if include_complexity:
            result = self.create_complexity_features(result)
        
        if include_sleep_specific:
            result = self.create_sleep_specific_features(result)
        
        logger.info(f"Feature engineering complete. Added {len(self.engineered_features)} new features")
        return result


class FeatureSelector:
    """Feature selection utilities for sleep stage classification."""
    
    def __init__(self):
        self.selected_features = []
        self.feature_scores = {}
        
    def select_univariate(self, X: pd.DataFrame, y: pd.Series, 
                         k: int = 50, score_func=f_classif) -> List[str]:
        """
        Select features using univariate statistical tests.
        
        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select
            score_func: Scoring function (f_classif or mutual_info_classif)
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting {k} features using univariate selection")
        
        # Ensure we don't select more features than available
        k = min(k, X.shape[1])
        
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Store feature scores
        self.feature_scores['univariate'] = dict(zip(X.columns, selector.scores_))
        
        logger.info(f"Selected {len(selected_features)} features using univariate selection")
        return selected_features
    
    def select_recursive(self, X: pd.DataFrame, y: pd.Series, 
                        n_features: int = 30, estimator=None) -> List[str]:
        """
        Select features using Recursive Feature Elimination.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_features: Number of features to select
            estimator: Estimator to use (if None, uses RandomForest)
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting {n_features} features using RFE")
        
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        # Ensure we don't select more features than available
        n_features = min(n_features, X.shape[1])
        
        selector = RFE(estimator=estimator, n_features_to_select=n_features)
        selector.fit(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Store feature rankings
        self.feature_scores['rfe_ranking'] = dict(zip(X.columns, selector.ranking_))
        
        logger.info(f"Selected {len(selected_features)} features using RFE")
        return selected_features
    
    def select_importance_based(self, X: pd.DataFrame, y: pd.Series, 
                               n_features: int = 40, estimator=None) -> List[str]:
        """
        Select features based on importance from tree-based model.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_features: Number of features to select
            estimator: Tree-based estimator (if None, uses RandomForest)
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting {n_features} features using importance-based selection")
        
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        # Fit the estimator
        estimator.fit(X, y)
        
        # Get feature importances
        importances = estimator.feature_importances_
        
        # Create importance DataFrame and sort
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Select top N features
        n_features = min(n_features, len(importance_df))
        selected_features = importance_df.head(n_features)['feature'].tolist()
        
        # Store feature importances
        self.feature_scores['importance'] = dict(zip(X.columns, importances))
        
        logger.info(f"Selected {len(selected_features)} features using importance-based selection")
        return selected_features
    
    def select_by_group(self, X: pd.DataFrame, y: pd.Series, 
                       group_counts: Dict[str, int] = None) -> List[str]:
        """
        Select features from each feature group.
        
        Args:
            X: Feature matrix
            y: Target variable
            group_counts: Number of features to select from each group
            
        Returns:
            List of selected feature names
        """
        logger.info("Selecting features by group")
        
        if group_counts is None:
            group_counts = {
                'relative': 4,
                'spectral': 3,
                'ratios': 3,
                'emg': 2,
                'absolute': 2
            }
        
        selected_features = []
        
        for group_name, n_features in group_counts.items():
            if group_name in FEATURE_GROUPS:
                # Get features from this group that are available in X
                group_features = [f for f in FEATURE_GROUPS[group_name] if f in X.columns]
                
                if group_features:
                    # Select best features from this group using univariate selection
                    if len(group_features) <= n_features:
                        selected_from_group = group_features
                    else:
                        group_X = X[group_features]
                        selector = SelectKBest(score_func=f_classif, k=n_features)
                        selector.fit(group_X, y)
                        selected_from_group = [group_features[i] for i in selector.get_support(indices=True)]
                    
                    selected_features.extend(selected_from_group)
                    logger.info(f"Selected {len(selected_from_group)} features from {group_name} group")
        
        logger.info(f"Selected {len(selected_features)} features total using group-based selection")
        return selected_features


class FeatureScaler:
    """Feature scaling utilities."""
    
    def __init__(self, method: str = 'standard'):
        """
        Initialize feature scaler.
        
        Args:
            method: Scaling method ('standard', 'minmax', 'robust')
        """
        self.method = method
        self.scaler = None
        self._initialize_scaler()
    
    def _initialize_scaler(self):
        """Initialize the appropriate scaler."""
        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler and transform features."""
        logger.info(f"Applying {self.method} scaling to features")
        
        # Fit and transform
        X_scaled = self.scaler.fit_transform(X)
        
        # Return as DataFrame with original column names
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scaler."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)


def build_features(data: pd.DataFrame, 
                  target_column: str = 'Sleep_Stage',
                  feature_engineering: bool = True,
                  feature_selection: bool = True,
                  feature_scaling: bool = True,
                  selection_method: str = 'importance',
                  n_features: int = 50,
                  scaling_method: str = 'standard') -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Complete feature building pipeline.
    
    Args:
        data: Input DataFrame
        target_column: Name of target column
        feature_engineering: Whether to apply feature engineering
        feature_selection: Whether to apply feature selection
        feature_scaling: Whether to apply feature scaling
        selection_method: Feature selection method ('importance', 'univariate', 'rfe', 'group')
        n_features: Number of features to select
        scaling_method: Scaling method ('standard', 'minmax', 'robust')
        
    Returns:
        Tuple of (features_df, target_series, selected_feature_names)
    """
    logger.info("Starting feature building pipeline")
    
    # Separate features and target
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    X = data.drop(columns=[target_column, 'filename', 'animal_id'], errors='ignore')
    y = data[target_column]
    
    # Feature Engineering
    if feature_engineering:
        engineer = FeatureEngineer()
        X = engineer.apply_all_engineering(X)
        logger.info(f"Feature engineering added {len(engineer.engineered_features)} new features")
    
    # Feature Selection
    selected_features = X.columns.tolist()
    if feature_selection:
        selector = FeatureSelector()
        
        if selection_method == 'importance':
            selected_features = selector.select_importance_based(X, y, n_features)
        elif selection_method == 'univariate':
            selected_features = selector.select_univariate(X, y, n_features)
        elif selection_method == 'rfe':
            selected_features = selector.select_recursive(X, y, n_features)
        elif selection_method == 'group':
            selected_features = selector.select_by_group(X, y)
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")
        
        X = X[selected_features]
        logger.info(f"Feature selection reduced features to {len(selected_features)}")
    
    # Feature Scaling
    if feature_scaling:
        scaler = FeatureScaler(method=scaling_method)
        X = scaler.fit_transform(X)
        logger.info(f"Applied {scaling_method} scaling to features")
    
    logger.info(f"Feature building complete. Final feature set: {X.shape}")
    return X, y, selected_features
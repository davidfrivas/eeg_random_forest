"""
Feature extraction from EEG/EMG data files.
"""
import os
import re
import logging
import numpy as np
import pandas as pd
from scipy.signal import welch
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import mne

from ..config import (
    FREQUENCY_BANDS, EPOCH_DURATION, EEG_FILTER_RANGE, 
    EMG_FILTER_RANGE, ARTIFACT_THRESHOLD_STD
)

logger = logging.getLogger(__name__)


class EEGFeatureExtractor:
    """Extract features from EEG/EMG data files."""
    
    def __init__(self, sampling_rate: Optional[int] = None):
        self.sampling_rate = sampling_rate
        self.frequency_bands = FREQUENCY_BANDS
        
    def extract_features_from_file(self, edf_file: Path, 
                                 output_dir: Path) -> Optional[Path]:
        """
        Extract features from a single EDF file.
        
        Args:
            edf_file: Path to the EDF file
            output_dir: Directory to save extracted features
            
        Returns:
            Path to the saved feature file, or None if processing failed
        """
        try:
            logger.info(f"Processing {edf_file}")
            raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)
            
            # Define EEG and EMG channels
            eeg_channels = raw.ch_names[:2]  # First two channels as EEG
            emg_channels = [raw.ch_names[-1]]  # Last channel as EMG
            
            if not eeg_channels or not emg_channels:
                logger.warning(f"Skipping {edf_file}: Missing EEG or EMG channels")
                return None
            
            # Apply filtering
            raw.filter(*EEG_FILTER_RANGE, picks=eeg_channels, verbose=False)
            raw.filter(EMG_FILTER_RANGE[0], EMG_FILTER_RANGE[1], 
                      picks=emg_channels, verbose=False)
            
            # Get sampling frequency
            fs = raw.info['sfreq']
            if self.sampling_rate is None:
                self.sampling_rate = fs
            
            # Create epochs
            epochs = mne.make_fixed_length_epochs(
                raw, duration=EPOCH_DURATION, overlap=0, preload=True, verbose=False
            )
            
            # Extract features
            features = self._extract_all_features(epochs, eeg_channels, emg_channels, fs)
            
            # Save features
            output_file = self._save_features(features, edf_file, output_dir)
            logger.info(f"Saved features to {output_file}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error processing {edf_file}: {e}")
            return None
    
    def _extract_all_features(self, epochs, eeg_channels: List[str], 
                            emg_channels: List[str], fs: float) -> pd.DataFrame:
        """Extract all features from epochs."""
        # EEG features
        eeg_data = epochs.get_data(picks=eeg_channels)
        freqs, psd = welch(eeg_data, fs=fs, nperseg=int(fs * 4), 
                          noverlap=int(fs * 2), axis=2)
        
        eeg_features = self._extract_eeg_features(psd, freqs)
        
        # EMG features
        emg_data = epochs.get_data(picks=emg_channels)
        emg_features = self._extract_emg_features(emg_data, fs)
        
        # Combine features
        features = {**eeg_features, **emg_features}
        
        # Add artifact detection
        features['Artifact'] = self._detect_artifacts(
            eeg_features['Delta_Power'], emg_features['EMG_RMS']
        )
        
        return pd.DataFrame(features)
    
    def _extract_eeg_features(self, psd: np.ndarray, freqs: np.ndarray) -> Dict:
        """Extract EEG spectral features."""
        features = {}
        
        # Calculate power in each frequency band
        band_powers = {}
        for band_name, (low, high) in self.frequency_bands.items():
            power = self._bandpower(psd, freqs, [low, high])
            band_powers[band_name] = power
            features[f'{band_name.capitalize()}_Power'] = power.mean(axis=1)
        
        # Calculate total power for normalization
        total_power = self._bandpower(psd, freqs, [0.5, 45])
        
        # Relative power features
        for band_name, power in band_powers.items():
            rel_power = np.divide(power, total_power, 
                               out=np.zeros_like(power), where=total_power!=0)
            features[f'Rel_{band_name.capitalize()}'] = rel_power.mean(axis=1)
        
        # Power ratios
        features['Theta_Delta_Ratio'] = self._safe_divide(
            band_powers['theta'], band_powers['delta']
        )
        features['Alpha_Delta_Ratio'] = self._safe_divide(
            band_powers['alpha'], band_powers['delta']
        )
        features['Beta_Theta_Ratio'] = self._safe_divide(
            band_powers['beta'], band_powers['theta']
        )
        features['Beta_Delta_Ratio'] = self._safe_divide(
            band_powers['beta'], band_powers['delta']
        )
        
        # Spectral features
        spectral_features = self._extract_spectral_features(psd, freqs)
        features.update(spectral_features)
        
        return features
    
    def _extract_emg_features(self, emg_data: np.ndarray, fs: float) -> Dict:
        """Extract EMG time-domain features."""
        features = {}
        
        # Basic EMG features
        features['EMG_RMS'] = np.sqrt(np.mean(emg_data ** 2, axis=2)).flatten()
        features['EMG_MAV'] = np.mean(np.abs(emg_data), axis=2).flatten()
        features['EMG_Variance'] = np.var(emg_data, axis=2).flatten()
        
        # Zero crossing rate
        emg_zcr = np.zeros(emg_data.shape[0])
        for i in range(emg_data.shape[0]):
            sign_changes = np.diff(np.signbit(emg_data[i, 0, :]))
            emg_zcr[i] = np.sum(sign_changes) / emg_data.shape[2]
        
        features['EMG_ZCR'] = emg_zcr
        
        return features
    
    def _extract_spectral_features(self, psd: np.ndarray, freqs: np.ndarray) -> Dict:
        """Extract spectral edge frequency and entropy."""
        features = {}
        
        spectral_edge_95 = []
        spectral_edge_50 = []
        spec_entropy = []
        
        for i in range(psd.shape[0]):
            epoch_psd = psd[i]
            spectral_edge_95.append(self._spectral_edge_freq(epoch_psd, freqs, 0.95))
            spectral_edge_50.append(self._spectral_edge_freq(epoch_psd, freqs, 0.50))
            spec_entropy.append(self._spectral_entropy(epoch_psd, freqs))
        
        features['Spectral_Edge_95'] = spectral_edge_95
        features['Spectral_Edge_50'] = spectral_edge_50
        features['Spectral_Entropy'] = spec_entropy
        
        return features
    
    def _bandpower(self, psd: np.ndarray, freqs: np.ndarray, 
                  band: List[float]) -> np.ndarray:
        """Calculate power in a frequency band."""
        idx = np.logical_and(freqs >= band[0], freqs <= band[1])
        return np.trapz(psd[:, :, idx], freqs[idx], axis=2)
    
    def _safe_divide(self, numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        """Safely divide arrays, handling division by zero."""
        result = np.divide(numerator, denominator,
                          out=np.zeros_like(numerator),
                          where=denominator!=0)
        return result.mean(axis=1)
    
    def _spectral_edge_freq(self, psd: np.ndarray, freqs: np.ndarray, 
                           edge: float = 0.95) -> float:
        """Calculate spectral edge frequency."""
        if psd.ndim > 1:
            psd = np.mean(psd, axis=0)
        
        c_sum = np.cumsum(psd)
        normalized_c_sum = c_sum / c_sum[-1] if c_sum[-1] > 0 else c_sum
        
        try:
            idx = np.where(normalized_c_sum >= edge)[0][0]
            return freqs[idx]
        except IndexError:
            return freqs[-1]
    
    def _spectral_entropy(self, psd: np.ndarray, freqs: np.ndarray) -> float:
        """Calculate spectral entropy."""
        if psd.ndim > 1:
            psd = np.mean(psd, axis=0)
        
        norm_psd = psd / np.sum(psd) if np.sum(psd) > 0 else psd
        entropy = -np.sum(norm_psd * np.log2(norm_psd + 1e-16))
        max_entropy = np.log2(len(norm_psd))
        
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def _detect_artifacts(self, delta_power: np.ndarray, emg_rms: np.ndarray) -> np.ndarray:
        """Detect artifacts based on power thresholds."""
        eeg_threshold = ARTIFACT_THRESHOLD_STD * np.std(delta_power)
        emg_threshold = ARTIFACT_THRESHOLD_STD * np.std(emg_rms)
        
        artifact_mask = ~((delta_power < eeg_threshold) & (emg_rms < emg_threshold))
        return artifact_mask.astype(int)
    
    def _save_features(self, features: pd.DataFrame, edf_file: Path, 
                      output_dir: Path) -> Path:
        """Save features to CSV file."""
        # Extract animal ID from filename
        base_filename = edf_file.stem
        match = re.match(r"([a-zA-Z0-9-]+)_(wm-day\d+|baseline)", base_filename)
        animal_id = match.group(1) if match else "unknown"
        
        # Create animal-specific directory
        animal_dir = output_dir / animal_id
        animal_dir.mkdir(parents=True, exist_ok=True)
        
        # Save features
        output_file = animal_dir / f"{base_filename}.csv"
        features.to_csv(output_file, index=False)
        
        return output_file


def extract_features_from_directory(input_dir: Path, output_dir: Path) -> List[Path]:
    """
    Extract features from all EDF files in a directory.
    
    Args:
        input_dir: Directory containing EDF files
        output_dir: Directory to save extracted features
        
    Returns:
        List of paths to saved feature files
    """
    extractor = EEGFeatureExtractor()
    
    # Find all EDF files
    edf_files = list(input_dir.rglob("*.edf"))
    logger.info(f"Found {len(edf_files)} EDF files")
    
    # Extract features from each file
    feature_files = []
    for edf_file in edf_files:
        feature_file = extractor.extract_features_from_file(edf_file, output_dir)
        if feature_file:
            feature_files.append(feature_file)
    
    logger.info(f"Successfully processed {len(feature_files)} files")
    return feature_files
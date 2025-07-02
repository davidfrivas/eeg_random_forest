"""
Data processing modules for EEG/EMG feature extraction and merging.
"""

from .extract_features import EEGFeatureExtractor, extract_features_from_directory
from .merge_data import DataMerger, merge_data_directory

__all__ = [
    'EEGFeatureExtractor',
    'extract_features_from_directory',
    'DataMerger',
    'merge_data_directory'
]
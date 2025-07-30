"""
Data processing modules for EEG/EMG feature extraction, merging, and cleaning.
"""

from .extract_features import EEGFeatureExtractor, extract_features_from_directory
from .merge_data import DataMerger, merge_data_directory
from .resolve_mismatch import MismatchResolver, resolve_all_mismatches, diagnose_file
from .clean_data import SleepStageDataCleaner, clean_all_merged_files, clean_single_merged_file
from .data_discovery import (
    extract_animal_id,
    discover_data_files,
    discover_edf_files,
    discover_label_files,
    print_data_summary,
    match_edf_and_labels,
    validate_file_pairs
)

__all__ = [
    # Feature extraction
    'EEGFeatureExtractor',
    'extract_features_from_directory',
    
    # Data merging
    'DataMerger',
    'merge_data_directory',
    
    # Mismatch resolution
    'MismatchResolver',
    'resolve_all_mismatches',
    'diagnose_file',
    
    # Data cleaning
    'SleepStageDataCleaner',
    'clean_all_merged_files',
    'clean_single_merged_file',
    
    # Data discovery
    'extract_animal_id',
    'discover_data_files',
    'discover_edf_files', 
    'discover_label_files',
    'print_data_summary',
    'match_edf_and_labels',
    'validate_file_pairs'
]
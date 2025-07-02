"""
Configuration settings for the sleep stage classifier.
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Model and output directories
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR,
                  EXTERNAL_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Signal processing parameters
SAMPLING_RATE = 400  # Hz (adjust based on your data)
EPOCH_DURATION = 4   # seconds

# Frequency bands for spectral analysis
FREQUENCY_BANDS = {
    'delta': [0.5, 4],
    'theta': [4, 8],
    'alpha': [8, 13],
    'beta': [13, 30],
    'sigma': [12, 16],  # Sleep spindles
    'gamma': [30, 45]   # Higher frequency activity
}

# EEG/EMG channel configuration
EEG_FILTER_RANGE = [0.5, 45]  # Hz
EMG_FILTER_RANGE = [10, None]  # High-pass at 10 Hz

# Feature groups for model training
FEATURE_GROUPS = {
    'absolute': ['Delta_Power', 'Theta_Power', 'Alpha_Power', 'Beta_Power', 'Sigma_Power', 'Gamma_Power'],
    'relative': ['Rel_Delta', 'Rel_Theta', 'Rel_Alpha', 'Rel_Beta', 'Rel_Sigma', 'Rel_Gamma'],
    'ratios': ['Theta_Delta_Ratio', 'Alpha_Delta_Ratio', 'Beta_Theta_Ratio', 'Beta_Delta_Ratio'],
    'spectral': ['Spectral_Edge_95', 'Spectral_Edge_50', 'Spectral_Entropy'],
    'emg': ['EMG_RMS', 'EMG_MAV', 'EMG_Variance', 'EMG_ZCR']
}

# Sleep stage standardization
SLEEP_STAGE_MAPPING = {
    "Wake": "Wake",
    "Wake ": "Wake",
    "REM": "REM",
    "REM ": "REM",
    "Non REM": "Non REM",
    "Non REM ": "Non REM",
}

# Sleep stages to exclude from analysis
EXCLUDE_STAGES = {"Unknown", "Artifact"}
EXCLUDE_SUFFIX = " X"

# Model training parameters
MODEL_PARAMS = {
    'n_estimators': 300,
    'max_depth': None,
    'min_samples_leaf': 5,
    'random_state': 42,
    'n_jobs': -1
}

# Class weights for imbalanced data
CLASS_WEIGHTS = {
    'Non REM': 1.2,
    'REM': 4.2,
    'Wake': 0.5
}

# Post-processing parameters
SMOOTHING_WINDOW_SIZE = 5  # Must be odd
VALID_TRANSITIONS = {
    'Wake': ['Wake', 'Non REM'],
    'Non REM': ['Wake', 'Non REM', 'REM'],
    'REM': ['Non REM', 'REM', 'Wake']
}

# Artifact detection thresholds (in standard deviations)
ARTIFACT_THRESHOLD_STD = 3

# Cloud storage configuration (optional)
GCS_CONFIG = {
    'project_id': os.getenv('GCP_PROJECT_ID'),
    'bucket_name': os.getenv('GCS_BUCKET_NAME'),
    'region': os.getenv('GCP_REGION', 'us-east1')
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': False
        }
    }
}
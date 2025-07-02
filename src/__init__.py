"""
Sleep Stage Classification Package

A machine learning pipeline for classifying sleep stages from EEG/EMG data.
"""

__version__ = "0.1.0"
__author__ = "David Rivas"
__email__ = "David.Rivas@nyspi.columbia.edu"

# Import main components for easy access
from .config import *

# Define what gets imported with "from src import *"
__all__ = [
    'PROJECT_ROOT',
    'DATA_DIR',
    'MODELS_DIR',
    'FREQUENCY_BANDS',
    'FEATURE_GROUPS',
    'MODEL_PARAMS'
]
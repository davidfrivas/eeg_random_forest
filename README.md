# Random Forest Sleep Stage Classifier

A machine learning pipeline for classifying sleep stages from mouse EEG/EMG data using Random Forest models with hierarchical and standard classification approaches.

## Table of contents

| **Section** | **Description** |
|:-|:-|
| [Project Organization](#project-organization) | File structure of the project |
| [Features](#features) | What this pipeline accomplishes |
| [Installation](#installation) | How to install this repo |
| [Quick Start](#quick-start) | How to start using this pipeline |
| [Usage Examples](#usage-examples) | Examples using this pipeline |
| [Configuration](#configuration) | How to configure to your needs |
| [Data Format](#data-format) | Expected data format |
| [Sleep Stage Classification](#sleep-stage-classification) | The stages of sleep this pipeline classifies |
| [Development](#development) | How to implement this pipeline in your development |
| [Contributing](#contributing) | How you can contribute to the project |
| [License](#license) | Project licensing |
| [Citation](#citation) | How to cite |
| [Troubleshooting](#troubleshooting) | Some troubleshooting tips |
| [Support](#support) | Get support using this pipeline |

## Project Organization

```
├── README.md          <- The top-level README for developers using this project.
├── data/
│   ├── external/      <- Data from third party sources.
│   ├── interim/       <- Intermediate data that has been transformed.
│   ├── processed/     <- The final, canonical data sets for modeling.
│   └── raw/           <- The original, immutable data dump.
│
├── docs/              <- Documentation
│
├── models/            <- Trained and serialized models, model predictions
│
├── notebooks/         <- Jupyter notebooks for exploration
│
├── references/        <- Data dictionaries, manuals, and explanatory materials
│
├── reports/           <- Generated analysis as TXT, PNG, PDF, etc.
│   └── figures/       <- Generated graphics and figures
│
├── requirements.txt   <- The requirements file for reproducing the environment
├── setup.py           <- Makes project pip installable (pip install -e .)
├── src/               <- Source code for use in this project
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── config.py      <- Configuration settings
│   │
│   ├── data/          <- Scripts to download or generate data
│   │   ├── clean_data.py
│   │   ├── extract_features.py
│   │   ├── merge_data.py
│   │   └── resolve_mismatch.py
│   │
│   ├── features/      <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models/        <- Scripts to train models and then use trained models
│   │   ├── predict_model.py
│   │   ├── train_model.py
│   │   └── evaluate_model.py
│   │
│   ├── utils/         <- Scripts to analyze model performance
│   │   └── analysis.py
│   │
│   └── visualization/ <- Scripts to create results visualizations
│       └── visualize.py
│
└── scripts/           <- Command-line scripts for running the pipeline
    ├── extract_features.py
    ├── merge_datasets.py
    ├── train_model.py
    └── evaluate_model.py
```

## Features

- **EEG/EMG Feature Extraction**: Comprehensive spectral and time-domain feature extraction from EDF files
- **Flexible Model Training**: Support for both standard multiclass and hierarchical (Wake vs Sleep → NREM vs REM) classification
- **Post-processing**: Sleep stage transition rules and smoothing for biologically plausible predictions
- **Comprehensive Evaluation**: Detailed performance metrics, confusion matrices, and feature importance analysis
- **Modular Design**: Clean separation of concerns with reusable components

## Installation

1. Clone the repository:
```bash
git clone https://github.com/davidfrivas/eeg_random_forest.git
cd eeg_random_forest
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package and dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

4. (Optional) Copy and configure environment variables:
```bash
cp .env.example .env
# Edit .env with your specific settings
```

## Quick Start

### 1. Prepare Your Data

Place your raw EDF files in the `data/raw/` directory and sleep stage annotation files in `data/interim/labels/`.

### 2. Extract Features

```bash
python scripts/extract_features.py --input-dir data/raw --output-dir data/interim/features
```

### 3. Merge Features with Labels

```bash
python scripts/merge_datasets.py \
    --features-dir data/interim/features \
    --labels-dir data/interim/labels \
    --output-dir data/processed
```

### 4. Train Models

```bash
# Train both standard and hierarchical models
python scripts/train_model.py \
    --data-dir data/processed \
    --output-dir models \
    --model-type both \
    --features all \
    --scale-features \
    --filter-artifacts \
    --apply-postprocessing \
    --subject-split
```

### 5. Evaluate Models

```bash
python scripts/evaluate_model.py \
    --model-path models/standard/standard_model.joblib \
    --test-data data/processed/test_data.csv \
    --output-dir reports
```

## Usage Examples

### Feature Extraction

```python
from src.data.extract_features import EEGFeatureExtractor
from pathlib import Path

# Extract features from a single file
extractor = EEGFeatureExtractor()
feature_file = extractor.extract_features_from_file(
    edf_file=Path("data/raw/animal_001.edf"),
    output_dir=Path("data/interim/features")
)
```

### Model Training

```python
from src.models.train_model import SleepStageClassifier
import pandas as pd

# Load merged data
data = pd.read_csv("data/processed/merged_data.csv")

# Initialize classifier
classifier = SleepStageClassifier()

# Prepare features
X, y = classifier.prepare_features(
    data, 
    feature_combination='all',
    scale_features=True,
    filter_artifacts=True
)

# Train hierarchical model
results = classifier.train_hierarchical_model(
    X_train, y_train, X_test, y_test,
    apply_postprocessing=True,
    save_model=True,
    model_dir=Path("models/hierarchical")
)
```

### Model Evaluation

```python
from src.models.evaluate_model import evaluate_saved_model
from pathlib import Path

# Evaluate a saved model
results = evaluate_saved_model(
    model_path=Path("models/standard/standard_model.joblib"),
    test_data=pd.read_csv("data/processed/test_data.csv"),
    feature_names=['Rel_Delta', 'Rel_Theta', 'Rel_Alpha', 'Spectral_Entropy'],
    output_dir=Path("reports/evaluation")
)

print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
```

## Configuration

The project uses a centralized configuration system in `src/config.py`. Key settings include:

- **Frequency Bands**: Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), etc.
- **Feature Groups**: Absolute power, relative power, power ratios, spectral features, EMG features
- **Model Parameters**: Random Forest hyperparameters, class weights
- **Post-processing**: Smoothing window size, transition rules

You can modify these settings or override them through environment variables.

## Data Format

### Input Data
- **EDF Files**: Raw EEG/EMG data in European Data Format
- **Annotation Files**: Sleep stage labels in text format (supports multiple formats)

### Expected Directory Structure
```
data/raw/
├── animal_001/
│   ├── animal_001_baseline.edf
│   └── animal_001_wm-day1.edf
└── animal_002/
    ├── animal_002_baseline.edf
    └── animal_002_wm-day1.edf

data/interim/labels/
├── animal_001/
│   ├── animal_001_baseline_epochs.txt
│   └── animal_001_wm-day1_epochs.txt
└── animal_002/
    ├── animal_002_baseline_epochs.txt
    └── animal_002_wm-day1_epochs.txt
```

## Sleep Stage Classification

The system supports three sleep stages:
- **Wake**: Active wakefulness
- **Non REM**: Non-rapid eye movement sleep
- **REM**: Rapid eye movement sleep

### Model Types

1. **Standard Model**: Direct multiclass classification
2. **Hierarchical Model**: Two-step classification
   - Step 1: Wake vs Sleep
   - Step 2: NREM vs REM (for sleep samples)

### Post-processing

Optional post-processing applies biological constraints:
- **Smoothing**: Majority vote filtering to reduce isolated misclassifications
- **Transition Rules**: Enforce physiologically valid sleep stage transitions

## Feature Engineering

### EEG Features
- **Spectral Power**: Absolute and relative power in frequency bands
- **Power Ratios**: Cross-band power relationships
- **Spectral Edge Frequency**: Frequency containing specified percentage of total power
- **Spectral Entropy**: Measure of spectral flatness

### EMG Features
- **RMS**: Root mean square amplitude
- **MAV**: Mean absolute value
- **Variance**: Signal variance
- **Zero Crossing Rate**: Frequency of signal sign changes

## Model Performance

Typical performance metrics:
- **Balanced Accuracy**: 0.80-0.90 (depending on data quality and features)
- **Wake Detection**: High precision and recall (F1 > 0.85)
- **NREM Detection**: Good performance (F1 > 0.80)
- **REM Detection**: More challenging due to class imbalance (F1 > 0.70)

## Advanced Usage

### Custom Feature Selection

```python
# Use only spectral features
X, y = classifier.prepare_features(data, feature_combination='spectral')

# Use combined relative power and spectral features
X, y = classifier.prepare_features(data, feature_combination='relative+spectral')
```

### Hyperparameter Tuning

```python
from src.models.train_model import SleepStageClassifier

# Custom model parameters
custom_params = {
    'n_estimators': 500,
    'max_depth': 30,
    'min_samples_leaf': 3,
    'random_state': 42
}

classifier = SleepStageClassifier(model_params=custom_params)
```

### Cloud Integration

The system supports Google Cloud Platform integration for large-scale processing:

```python
# Set environment variables for GCP
os.environ['GCP_PROJECT_ID'] = 'your-project'
os.environ['GCS_BUCKET_NAME'] = 'your-bucket'

# Use cloud storage for data
from src.utils.cloud_utils import upload_to_gcs, download_from_gcs
```

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Formatting

```bash
black src/
flake8 src/
```

### Using Make

```bash
# Install dependencies
make requirements

# Extract features and merge data
make data

# Train models
make train

# Run evaluation
make evaluate

# Clean up
make clean
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{eeg_random_forest,
  title={Random Forest Sleep Stage Classifier},
  author={David Rivas},
  year={2025},
  url={https://github.com/davidfrivas/eeg_random_forest}
}
```

## Troubleshooting

### Common Issues

1. **MNE Import Error**: Install MNE with `pip install mne`
2. **Memory Issues**: Reduce batch size or use feature selection
3. **File Format Issues**: Ensure EDF files are properly formatted
4. **Label Mismatch**: Check that feature and label files have matching epochs

### Performance Optimization

- Use feature selection to reduce dimensionality
- Apply artifact filtering to improve data quality
- Consider subject-specific splitting for better generalization
- Tune hyperparameters for your specific dataset

## Support

For questions and support:
- Open an issue on GitHub
- Check the documentation in `docs/`
- Review the example notebook in `notebooks/`
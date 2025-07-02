sleep-stage-classification/
├── README.md
├── requirements.txt
├── setup.py
├── Makefile
├── .env.example
├── .gitignore
│
├── data/
│   ├── external/           # Data from third party sources
│   ├── interim/            # Intermediate data that has been transformed
│   ├── processed/          # The final, canonical data sets for modeling
│   └── raw/                # The original, immutable data dump
│
├── docs/                   # Documentation
│
├── models/                 # Trained and serialized models, model predictions
│
├── notebooks/              # Jupyter notebooks for exploration
│   └── exploratory/
│
├── references/             # Data dictionaries, manuals, and explanatory materials
│
├── reports/                # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures/            # Generated graphics and figures
│
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration settings
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── extract_features.py    # EEG/EMG feature extraction
│   │   ├── merge_data.py         # Merge features with sleep stage labels
│   │   └── preprocessing.py      # Data cleaning and preprocessing
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py     # Feature engineering and selection
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py        # Model training logic
│   │   ├── predict_model.py      # Model prediction logic
│   │   └── evaluate_model.py     # Model evaluation utilities
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── visualize.py          # Plotting and visualization
│   │
│   └── utils/
│       ├── __init__.py
│       ├── file_utils.py         # File handling utilities
│       └── signal_processing.py  # Signal processing utilities
│
├── scripts/
│   ├── extract_features.py       # Script to extract features from raw EDF files
│   ├── merge_datasets.py         # Script to merge features with labels
│   ├── train_model.py            # Script to train models
│   └── evaluate_model.py         # Script to evaluate trained models
│
└── tests/
    ├── __init__.py
    └── test_data_processing.py
"""
Train sleep stage classification models.
"""
import logging
import click
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.train_model import SleepStageClassifier
from src.config import PROCESSED_DATA_DIR, MODELS_DIR
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--data-file', '-d',
              type=click.Path(exists=True, path_type=Path),
              help='Path to merged data file (CSV)')
@click.option('--data-dir', 
              type=click.Path(exists=True, path_type=Path),
              default=PROCESSED_DATA_DIR,
              help='Directory containing merged data files')
@click.option('--output-dir', '-o',
              type=click.Path(path_type=Path),
              default=MODELS_DIR,
              help='Directory to save trained models')
@click.option('--model-type', '-t',
              type=click.Choice(['standard', 'hierarchical', 'both']),
              default='both',
              help='Type of model to train')
@click.option('--features', '-f',
              type=click.Choice(['all', 'relative', 'spectral', 'ratios', 'emg', 'relative+spectral']),
              default='all',
              help='Feature combination to use')
@click.option('--scale-features', is_flag=True,
              help='Standardize features before training')
@click.option('--filter-artifacts', is_flag=True,
              help='Filter high-artifact data points')
@click.option('--apply-postprocessing', is_flag=True,
              help='Apply sleep stage transition rules')
@click.option('--test-size', type=float, default=0.3,
              help='Proportion of data to use for testing')
@click.option('--subject-split', is_flag=True,
              help='Split by subject (animal) rather than randomly')
def main(data_file: Path, data_dir: Path, output_dir: Path, model_type: str,
         features: str, scale_features: bool, filter_artifacts: bool,
         apply_postprocessing: bool, test_size: float, subject_split: bool):
    """Train sleep stage classification models."""
    
    # Load data
    if data_file:
        logger.info(f"Loading data from {data_file}")
        data = pd.read_csv(data_file)
    else:
        logger.info(f"Loading data from directory {data_dir}")
        # Combine all CSV files in the directory
        csv_files = list(data_dir.rglob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {data_dir}")
        
        data_frames = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            # Add file information
            df['filename'] = csv_file.name
            df['animal_id'] = csv_file.parent.name
            data_frames.append(df)
        
        data = pd.concat(data_frames, ignore_index=True)
    
    logger.info(f"Loaded {len(data)} samples")
    
    # Initialize classifier
    classifier = SleepStageClassifier()
    
    # Prepare features
    X, y = classifier.prepare_features(
        data, 
        feature_combination=features,
        scale_features=scale_features,
        filter_artifacts=filter_artifacts
    )
    
    # Split data
    if subject_split and 'animal_id' in data.columns:
        # Split by subject
        subjects = data['animal_id'].unique()
        train_subjects, test_subjects = train_test_split(
            subjects, test_size=test_size, random_state=42
        )
        
        train_mask = data['animal_id'].isin(train_subjects)
        test_mask = data['animal_id'].isin(test_subjects)
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        logger.info(f"Subject-based split: {len(train_subjects)} train, {len(test_subjects)} test subjects")
    else:
        # Random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        logger.info("Random split")
    
    logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train models
    results = {}
    
    if model_type in ['standard', 'both']:
        logger.info("Training standard model...")
        results['standard'] = classifier.train_standard_model(
            X_train, y_train, X_test, y_test,
            apply_postprocessing=apply_postprocessing,
            save_model=True,
            model_dir=output_dir / "standard"
        )
    
    if model_type in ['hierarchical', 'both']:
        logger.info("Training hierarchical model...")
        results['hierarchical'] = classifier.train_hierarchical_model(
            X_train, y_train, X_test, y_test,
            apply_postprocessing=apply_postprocessing,
            save_model=True,
            model_dir=output_dir / "hierarchical"
        )
    
    # Compare models if both were trained
    if len(results) > 1:
        logger.info("\nModel Comparison:")
        for name, result in results.items():
            logger.info(f"{name.capitalize()}: {result['balanced_accuracy']:.4f}")
    
    logger.info(f"Training complete. Models saved to {output_dir}")


if __name__ == "__main__":
    main()
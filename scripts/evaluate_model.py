"""
Evaluate trained models on test data.
"""
import logging
import click
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.evaluate_model import evaluate_saved_model
from src.config import MODELS_DIR, REPORTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--model-path', '-m',
              type=click.Path(exists=True, path_type=Path),
              required=True,
              help='Path to the saved model (.joblib)')
@click.option('--test-data', '-d',
              type=click.Path(exists=True, path_type=Path),
              required=True,
              help='Path to test data (CSV)')
@click.option('--output-dir', '-o',
              type=click.Path(path_type=Path),
              default=REPORTS_DIR,
              help='Directory to save evaluation results')
@click.option('--feature-names',
              type=str,
              help='Comma-separated list of feature names to use')
def main(model_path: Path, test_data: Path, output_dir: Path, feature_names: str):
    """Evaluate a trained model on test data."""
    
    logger.info(f"Evaluating model {model_path} on {test_data}")
    
    # Load test data
    data = pd.read_csv(test_data)
    
    # Parse feature names
    if feature_names:
        features = [f.strip() for f in feature_names.split(',')]
    else:
        # Use all numeric columns except Sleep_Stage
        exclude_cols = ['Sleep_Stage', 'filename', 'animal_id', 'Artifact']
        features = [col for col in data.columns 
                   if col not in exclude_cols and data[col].dtype in ['float64', 'int64']]
    
    logger.info(f"Using {len(features)} features")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate model
    results = evaluate_saved_model(model_path, data, features, output_dir)
    
    logger.info(f"Evaluation complete. Results saved to {output_dir}")
    logger.info(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")


if __name__ == "__main__":
    main()
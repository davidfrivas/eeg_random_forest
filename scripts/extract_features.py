"""
Extract features from raw EDF files.
"""
import logging
import click
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.extract_features import extract_features_from_directory
from src.config import RAW_DATA_DIR, INTERIM_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--input-dir', '-i', 
              type=click.Path(exists=True, path_type=Path),
              default=RAW_DATA_DIR,
              help='Directory containing EDF files')
@click.option('--output-dir', '-o',
              type=click.Path(path_type=Path),
              default=INTERIM_DATA_DIR / "features",
              help='Directory to save extracted features')
def main(input_dir: Path, output_dir: Path):
    """Extract features from EDF files."""
    logger.info(f"Extracting features from {input_dir} to {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    feature_files = extract_features_from_directory(input_dir, output_dir)
    
    logger.info(f"Successfully extracted features from {len(feature_files)} files")
    logger.info(f"Features saved to {output_dir}")


if __name__ == "__main__":
    main()
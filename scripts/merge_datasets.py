"""
Merge extracted features with sleep stage labels.
"""
import logging
import click
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.merge_data import merge_data_directory
from src.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--features-dir', '-f',
              type=click.Path(exists=True, path_type=Path),
              default=INTERIM_DATA_DIR / "features",
              help='Directory containing feature CSV files')
@click.option('--labels-dir', '-l',
              type=click.Path(exists=True, path_type=Path),
              default=INTERIM_DATA_DIR / "labels",
              help='Directory containing label TXT files')
@click.option('--output-dir', '-o',
              type=click.Path(path_type=Path),
              default=PROCESSED_DATA_DIR,
              help='Directory to save merged files')
def main(features_dir: Path, labels_dir: Path, output_dir: Path):
    """Merge feature files with sleep stage labels."""
    logger.info(f"Merging features from {features_dir} with labels from {labels_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    merged_files = merge_data_directory(features_dir, labels_dir, output_dir)
    
    logger.info(f"Successfully merged {len(merged_files)} files")
    logger.info(f"Merged data saved to {output_dir}")


if __name__ == "__main__":
    main()
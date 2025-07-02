import click
from pathlib import Path
import pandas as pd
from src.features import build_features

@click.command()
@click.option('--input-file', type=Path, required=True)
@click.option('--output-file', type=Path, required=True)
@click.option('--n-features', type=int, default=50)
def main(input_file, output_file, n_features):
    """Build features from merged data."""
    data = pd.read_csv(input_file)
    
    X, y, features = build_features(
        data, 
        n_features=n_features,
        selection_method='importance'
    )
    
    # Combine and save
    result = X.copy()
    result['Sleep_Stage'] = y
    result.to_csv(output_file, index=False)
    
    print(f"Built {len(features)} features, saved to {output_file}")

if __name__ == "__main__":
    main()
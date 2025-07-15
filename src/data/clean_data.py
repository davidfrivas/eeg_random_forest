"""
Clean and standardize sleep stage data from merged files.
"""
import os
import re
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from ..config import SLEEP_STAGE_MAPPING, EXCLUDE_STAGES, EXCLUDE_SUFFIX

logger = logging.getLogger(__name__)


class SleepStageDataCleaner:
    """Clean and standardize sleep stage data."""
    
    def __init__(self, log_cleaning: bool = True):
        self.log_cleaning = log_cleaning
        self.cleaning_log = []
        self.cleaned_files = []
        self.failed_files = []
    
    def clean_all_merged_files(self, merged_files_dir: Path, output_dir: Path) -> Dict[str, List[Path]]:
        """
        Clean all merged files in a directory.
        
        Args:
            merged_files_dir: Directory containing merged CSV files
            output_dir: Directory to save cleaned files
            
        Returns:
            Dictionary with 'cleaned' and 'failed' file lists
        """
        # Find all merged CSV files
        merged_files = self._find_merged_files(merged_files_dir)
        
        if not merged_files:
            print("No merged files found to clean")
            return {'cleaned': [], 'failed': []}
        
        print(f"Found {len(merged_files)} merged files to clean")
        
        # Clean each file
        for merged_file in merged_files:
            try:
                cleaned_file = self._clean_single_file(merged_file, output_dir)
                if cleaned_file:
                    self.cleaned_files.append(cleaned_file)
                    print(f"Cleaned: {merged_file.name}")
                else:
                    self.failed_files.append(merged_file)
                    print(f"Failed: {merged_file.name}")
            except Exception as e:
                self.failed_files.append(merged_file)
                print(f"Error cleaning {merged_file.name}: {e}")
                logger.error(f"Error cleaning {merged_file}: {e}")
        
        # Save cleaning log
        if self.cleaning_log:
            self._save_cleaning_log(output_dir)
        
        print(f"\nCLEANING RESULTS:")
        print(f"Successfully cleaned: {len(self.cleaned_files)} files")
        print(f"Failed to clean: {len(self.failed_files)} files")
        
        return {'cleaned': self.cleaned_files, 'failed': self.failed_files}
    
    def clean_single_merged_file(self, merged_file: Path, output_dir: Path) -> Optional[Path]:
        """
        Clean a single merged file.
        
        Args:
            merged_file: Path to merged CSV file
            output_dir: Directory to save cleaned file
            
        Returns:
            Path to cleaned file if successful, None otherwise
        """
        try:
            cleaned_file = self._clean_single_file(merged_file, output_dir)
            if cleaned_file:
                print(f"Successfully cleaned: {merged_file.name}")
                return cleaned_file
            else:
                print(f"Failed to clean: {merged_file.name}")
                return None
        except Exception as e:
            print(f"Error cleaning {merged_file.name}: {e}")
            logger.error(f"Error cleaning {merged_file}: {e}")
            return None
    
    def _find_merged_files(self, merged_files_dir: Path) -> List[Path]:
        """Find all merged CSV files in directory."""
        merged_files = []
        
        # Look for files with common merged suffixes
        patterns = ["*_merged.csv", "*_resolved.csv", "*_fixed_merged.csv", "*_gap_fixed_merged.csv"]
        
        for pattern in patterns:
            found_files = list(merged_files_dir.rglob(pattern))
            merged_files.extend(found_files)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_files = []
        for f in merged_files:
            if f not in seen:
                seen.add(f)
                unique_files.append(f)
        
        return unique_files
    
    def _clean_single_file(self, merged_file: Path, output_dir: Path) -> Optional[Path]:
        """Clean a single merged file."""
        print(f"Processing: {merged_file.name}")
        
        # Load merged data
        try:
            df = pd.read_csv(merged_file)
        except Exception as e:
            print(f"Failed to load file: {e}")
            return None
        
        # Check if file has Sleep_Stage column
        if 'Sleep_Stage' not in df.columns:
            print(f"No Sleep_Stage column found")
            return None
        
        original_count = len(df)
        original_stages = df['Sleep_Stage'].value_counts().to_dict()
        
        print(f"  Original: {original_count} rows")
        print(f"  Original sleep stages: {dict(sorted(original_stages.items()))}")
        
        # Clean sleep stages
        df_cleaned = self._clean_sleep_stages(df)
        
        cleaned_count = len(df_cleaned)
        cleaned_stages = df_cleaned['Sleep_Stage'].value_counts().to_dict()
        rows_removed = original_count - cleaned_count
        
        print(f"  Cleaned: {cleaned_count} rows ({rows_removed} rows removed)")
        print(f"  Cleaned sleep stages: {dict(sorted(cleaned_stages.items()))}")
        
        # Log cleaning details
        if self.log_cleaning:
            self._log_cleaning_details(merged_file, original_count, cleaned_count, 
                                     original_stages, cleaned_stages)
        
        # Save cleaned file
        output_file = self._save_cleaned_file(df_cleaned, merged_file, output_dir)
        
        return output_file
    
    def _clean_sleep_stages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize sleep stage labels.
        
        This function:
        1. Strips whitespace from sleep stage labels
        2. Excludes unwanted stages (Unknown, Artifact, etc.)
        3. Excludes stages ending with specified suffix (" X")
        4. Standardizes stage names using mapping
        
        Args:
            df: DataFrame with Sleep_Stage column
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Strip whitespace from sleep stage labels
        df_clean['Sleep_Stage'] = df_clean['Sleep_Stage'].astype(str).str.strip()
        
        # Create exclusion mask
        exclude_mask = (
            df_clean['Sleep_Stage'].isin(EXCLUDE_STAGES) |  # Exclude specific stages
            df_clean['Sleep_Stage'].str.endswith(EXCLUDE_SUFFIX)  # Exclude stages ending with suffix
        )
        
        # Apply exclusion mask (keep rows that are NOT excluded)
        df_clean = df_clean[~exclude_mask].copy()
        
        # Standardize stage names using mapping
        df_clean['Sleep_Stage'] = df_clean['Sleep_Stage'].map(SLEEP_STAGE_MAPPING).fillna(df_clean['Sleep_Stage'])
        
        return df_clean
    
    def _save_cleaned_file(self, df: pd.DataFrame, original_file: Path, output_dir: Path) -> Path:
        """Save cleaned data to file."""
        # Extract animal ID from filename
        base_filename = original_file.stem
        
        # Remove existing suffixes to get clean base name
        for suffix in ['_merged', '_resolved', '_fixed_merged', '_gap_fixed_merged']:
            if base_filename.endswith(suffix):
                base_filename = base_filename[:-len(suffix)]
                break
        
        # Extract animal ID
        match = re.match(r"([a-zA-Z0-9-]+)_(wm-day\d+|baseline)", base_filename)
        animal_id = match.group(1) if match else "unknown"
        
        # Create animal-specific directory
        animal_dir = output_dir / animal_id
        animal_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cleaned file
        output_file = animal_dir / f"{base_filename}_cleaned.csv"
        df.to_csv(output_file, index=False)
        
        return output_file
    
    def _log_cleaning_details(self, file_path: Path, original_count: int, cleaned_count: int,
                            original_stages: Dict, cleaned_stages: Dict):
        """Log detailed cleaning information."""
        
        # Calculate removed stages
        removed_stages = {}
        for stage, count in original_stages.items():
            if stage not in cleaned_stages:
                removed_stages[stage] = count
            else:
                diff = count - cleaned_stages.get(stage, 0)
                if diff > 0:
                    removed_stages[stage] = diff
        
        self.cleaning_log.append({
            'file': str(file_path),
            'original_rows': original_count,
            'cleaned_rows': cleaned_count,
            'rows_removed': original_count - cleaned_count,
            'original_stages': str(original_stages),
            'cleaned_stages': str(cleaned_stages),
            'removed_stages': str(removed_stages)
        })
    
    def _save_cleaning_log(self, output_dir: Path):
        """Save cleaning log to CSV file."""
        if self.cleaning_log:
            log_df = pd.DataFrame(self.cleaning_log)
            log_file = output_dir / "cleaning_log.csv"
            log_df.to_csv(log_file, index=False)
            print(f"Cleaning log saved to: {log_file}")
    
    def get_cleaning_summary(self) -> Dict:
        """Get summary of cleaning results."""
        if not self.cleaning_log:
            return {}
        
        log_df = pd.DataFrame(self.cleaning_log)
        
        summary = {
            'total_files_cleaned': len(self.cleaning_log),
            'total_original_rows': log_df['original_rows'].sum(),
            'total_cleaned_rows': log_df['cleaned_rows'].sum(),
            'total_rows_removed': log_df['rows_removed'].sum(),
            'removal_percentage': (log_df['rows_removed'].sum() / log_df['original_rows'].sum()) * 100
        }
        
        return summary


def clean_all_merged_files(merged_files_dir: Path, output_dir: Path, 
                          log_cleaning: bool = True) -> Dict[str, List[Path]]:
    """
    Convenience function to clean all merged files in a directory.
    
    Args:
        merged_files_dir: Directory containing merged CSV files
        output_dir: Directory to save cleaned files
        log_cleaning: Whether to log cleaning details
        
    Returns:
        Dictionary with 'cleaned' and 'failed' file lists
    """
    cleaner = SleepStageDataCleaner(log_cleaning=log_cleaning)
    return cleaner.clean_all_merged_files(merged_files_dir, output_dir)


def clean_single_merged_file(merged_file: Path, output_dir: Path) -> Optional[Path]:
    """
    Convenience function to clean a single merged file.
    
    Args:
        merged_file: Path to merged CSV file
        output_dir: Directory to save cleaned file
        
    Returns:
        Path to cleaned file if successful, None otherwise
    """
    cleaner = SleepStageDataCleaner()
    return cleaner.clean_single_merged_file(merged_file, output_dir)

# Example usage
if __name__ == "__main__":
    # Example paths - adjust to your setup
    merged_files_dir = Path("../data/processed")
    output_dir = Path("../data/processed")
    
    # Clean all merged files
    results = clean_all_merged_files(merged_files_dir, output_dir)
    
    print(f"\nCleaned files: {len(results['cleaned'])}")
    for file in results['cleaned']:
        print(f"{file.name}")
    
    print(f"\nFailed files: {len(results['failed'])}")
    for file in results['failed']:
        print(f"{file.name}")
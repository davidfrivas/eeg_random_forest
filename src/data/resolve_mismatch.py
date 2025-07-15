"""
Resolve row differences in mismatched files.
Handles large timestamp gaps and single-row differences separately.
"""
import os
import re
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from datetime import datetime, timedelta

from ..config import EPOCH_DURATION

logger = logging.getLogger(__name__)


class MismatchResolver:
    """Resolve files with row count mismatches."""
    
    def __init__(self, epoch_duration: Optional[float] = None):
        self.epoch_duration = epoch_duration or EPOCH_DURATION  # Use config default if not specified
        self.fixed_files = []
        self.failed_files = []
    
    def resolve_all_mismatches(self, mismatch_log_file: Path, output_dir: Path) -> Dict[str, List[Path]]:
        """
        Resolve all mismatches from the mismatch log.
        
        Args:
            mismatch_log_file: Path to the mismatch log CSV file
            output_dir: Directory to save fixed merged files
            
        Returns:
            Dictionary with 'fixed' and 'failed' file lists
        """
        # Load mismatch log
        if not mismatch_log_file.exists():
            print(f"Mismatch log not found: {mismatch_log_file}")
            return {'fixed': [], 'failed': []}
        
        mismatch_log = pd.read_csv(mismatch_log_file)
        print(f"Loaded mismatch log with {len(mismatch_log)} entries")
        
        # Process each file
        for _, row in mismatch_log.iterrows():
            feature_file = Path(row['feature_file'])
            label_file = Path(row['label_file'])
            difference = row['difference']
            
            print(f"\n{'='*60}")
            print(f"Processing: {feature_file.name}")
            print(f"Difference: {difference}")
            
            try:
                success = self._resolve_single_file(feature_file, label_file, difference, output_dir)
                if success:
                    self.fixed_files.append(feature_file)
                    print(f"Successfully resolved")
                else:
                    self.failed_files.append(feature_file)
                    print(f"Failed to resolve")
            except Exception as e:
                self.failed_files.append(feature_file)
                print(f"Error: {e}")
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS:")
        print(f"Fixed: {len(self.fixed_files)} files")
        print(f"Failed: {len(self.failed_files)} files")
        
        return {'fixed': self.fixed_files, 'failed': self.failed_files}
    
    def _resolve_single_file(self, feature_file: Path, label_file: Path, 
                           difference: int, output_dir: Path) -> bool:
        """
        Resolve a single file by applying fixes in order:
        1. Fix large timestamp gaps (if any)
        2. Fix single-row differences (if any)
        
        Args:
            feature_file: Path to feature CSV file
            label_file: Path to label TXT file
            difference: Row count difference (features - labels)
            output_dir: Output directory
            
        Returns:
            True if successfully resolved, False otherwise
        """
        # Load data
        df_features = pd.read_csv(feature_file)
        df_labels = self._load_sleep_stages(label_file)
        
        if df_labels is None:
            print(f"Failed to load labels")
            return False
        
        original_features_count = len(df_features)
        original_labels_count = len(df_labels)
        
        print(f"  Original counts: {original_features_count} features, {original_labels_count} labels")
        print(f"  Difference: {difference}")
        
        # Step 1: Handle large timestamp gaps (if difference > 1)
        if abs(difference) > 1:
            print(f"\n  STEP 1: Fixing large timestamp gaps...")
            df_features, df_labels = self._fix_timestamp_gaps(df_features, df_labels)
            
            current_diff = len(df_features) - len(df_labels)
            print(f"  After gap fix: {len(df_features)} features, {len(df_labels)} labels (diff: {current_diff})")
        else:
            print(f"\n  STEP 1: No large gaps detected (difference = {difference})")
            current_diff = difference
        
        # Step 2: Handle single-row differences (if needed)
        if abs(current_diff) == 1:
            print(f"\n  STEP 2: Fixing single-row difference...")
            df_features, df_labels = self._fix_single_row_difference(df_features, df_labels)
            
            final_diff = len(df_features) - len(df_labels)
            print(f"  After single-row fix: {len(df_features)} features, {len(df_labels)} labels (diff: {final_diff})")
        elif current_diff == 0:
            print(f"\n  STEP 2: No single-row difference to fix")
            final_diff = 0
        else:
            print(f"\n  STEP 2: Cannot fix difference of {current_diff} (not ±1)")
            final_diff = current_diff
        
        # Step 3: Verify and merge
        if final_diff == 0:
            print(f"\n  STEP 3: Merging data...")
            df_features['Sleep_Stage'] = df_labels['Sleep_Stage']
            
            # Save merged file
            output_file = self._save_merged_file(df_features, feature_file, output_dir)
            
            rows_removed = original_features_count - len(df_features)
            print(f"  Final result: {len(df_features)} rows merged ({rows_removed} rows removed)")
            print(f"  Saved to: {output_file.name}")
            
            return True
        else:
            print(f"\n  STEP 3: Cannot merge - final difference is {final_diff}")
            return False
    
    def _fix_timestamp_gaps(self, df_features: pd.DataFrame, df_labels: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fix large timestamp gaps by removing corresponding feature rows.
        
        This function:
        1. Identifies missing timestamps in the label file
        2. Removes corresponding rows from the feature file
        3. Keeps the label file unchanged
        
        Args:
            df_features: Feature DataFrame
            df_labels: Label DataFrame with Start_Time column
            
        Returns:
            Tuple of (fixed_features, original_labels)
        """
        if 'Start_Time' not in df_labels.columns:
            print(f"No Start_Time column in labels")
            return df_features, df_labels
        
        # Find missing timestamp indices
        missing_indices = self._find_missing_timestamp_indices(df_labels)
        
        if len(missing_indices) == 0:
            print(f"No timestamp gaps found")
            return df_features, df_labels
        
        print(f"    Found {len(missing_indices)} missing timestamp positions")
        
        # Show gap details
        self._show_gap_details(df_labels, missing_indices)
        
        # Remove corresponding rows from features
        df_features_fixed = df_features.drop(index=missing_indices).reset_index(drop=True)
        
        removed_count = len(missing_indices)
        print(f"    Removed {removed_count} rows from features to match timestamp gaps")
        
        return df_features_fixed, df_labels
    
    def _fix_single_row_difference(self, df_features: pd.DataFrame, df_labels: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fix single-row differences (±1).
        
        Args:
            df_features: Feature DataFrame
            df_labels: Label DataFrame
            
        Returns:
            Tuple of (fixed_features, fixed_labels)
        """
        diff = len(df_features) - len(df_labels)
        
        if diff == 1:
            # Features has one extra row - remove the last row
            print(f"    Features has 1 extra row - removing last row")
            df_features = df_features.iloc[:-1].copy()
            
        elif diff == -1:
            # Labels has one extra row - remove duplicate or last row
            print(f"    Labels has 1 extra row - checking for duplicates")
            
            if 'Start_Time' in df_labels.columns:
                # Check for duplicate timestamps
                duplicates = df_labels['Start_Time'].duplicated()
                if duplicates.any():
                    duplicate_idx = df_labels[duplicates].index[0]
                    df_labels = df_labels.drop(index=duplicate_idx).reset_index(drop=True)
                    print(f"    Removed duplicate timestamp at index {duplicate_idx}")
                else:
                    # No duplicates - remove last row
                    df_labels = df_labels.iloc[:-1].copy()
                    print(f"    No duplicates found - removed last row from labels")
            else:
                # No timestamp column - just remove last row
                df_labels = df_labels.iloc[:-1].copy()
                print(f"    Removed last row from labels")
        
        elif diff == 0:
            print(f"    No single-row difference to fix")
        
        else:
            print(f"    Cannot fix difference of {diff} (not ±1)")
        
        return df_features, df_labels
    
    def _find_missing_timestamp_indices(self, df_labels: pd.DataFrame) -> List[int]:
        """
        Find indices where timestamps are missing in the expected sequence.
        
        This creates a complete expected sequence and identifies which positions
        are missing in the actual label file.
        
        Args:
            df_labels: DataFrame with Start_Time column
            
        Returns:
            List of indices where timestamps are missing
        """
        try:
            # Convert timestamps to datetime
            timestamps = pd.to_datetime(df_labels['Start_Time'])
            
            # Calculate expected full sequence
            start_time = timestamps.iloc[0]
            # Generate expected timestamps for 21600 epochs (24 hours * 60 min * 60 sec / 4 sec)
            expected_timestamps = [
                start_time + timedelta(seconds=i * self.epoch_duration) 
                for i in range(21600)
            ]
            
            # Convert to set for faster lookup
            actual_timestamps_set = set(timestamps)
            
            # Find missing indices
            missing_indices = []
            for i, expected_time in enumerate(expected_timestamps):
                if expected_time not in actual_timestamps_set:
                    missing_indices.append(i)
            
            return missing_indices
            
        except Exception as e:
            print(f"    Error finding missing timestamps: {e}")
            return []
    
    def _show_gap_details(self, df_labels: pd.DataFrame, missing_indices: List[int]):
        """Show details about timestamp gaps."""
        if len(missing_indices) == 0:
            return
        
        timestamps = pd.to_datetime(df_labels['Start_Time'])
        start_time = timestamps.iloc[0]
        
        # Group consecutive missing indices into gaps
        gaps = []
        current_gap_start = missing_indices[0]
        current_gap_end = missing_indices[0]
        
        for i in range(1, len(missing_indices)):
            if missing_indices[i] == current_gap_end + 1:
                # Consecutive missing index
                current_gap_end = missing_indices[i]
            else:
                # Gap ended, start new gap
                gaps.append((current_gap_start, current_gap_end))
                current_gap_start = missing_indices[i]
                current_gap_end = missing_indices[i]
        
        # Add the last gap
        gaps.append((current_gap_start, current_gap_end))
        
        print(f"    Gap analysis:")
        for gap_start, gap_end in gaps:
            gap_size = gap_end - gap_start + 1
            gap_start_time = start_time + timedelta(seconds=gap_start * self.epoch_duration)
            gap_end_time = start_time + timedelta(seconds=gap_end * self.epoch_duration)
            
            print(f"      Gap: {gap_size} missing epochs at indices {gap_start}-{gap_end}")
            print(f"           Time range: {gap_start_time} to {gap_end_time}")
    
    def _load_sleep_stages(self, txt_file: Path) -> Optional[pd.DataFrame]:
        """Load sleep stage labels with automatic format detection."""
        try:
            with open(txt_file, 'r') as f:
                first_line = f.readline().strip()
            
            # Check format based on header
            if "start, duration, annotation" in first_line:
                return pd.read_csv(txt_file, comment='#', header=None,
                                 names=["Start_Time", "Epoch_Duration", "Sleep_Stage"])
            elif "Onset,Annotation" in first_line:
                df = pd.read_csv(txt_file, comment='#', header=None,
                               names=["Start_Time", "Sleep_Stage"])
                df["Epoch_Duration"] = EPOCH_DURATION
                return df
            else:
                # Try to infer format
                sample = pd.read_csv(txt_file, comment='#', nrows=5, header=None)
                
                if sample.shape[1] == 2:
                    df = pd.read_csv(txt_file, comment='#', header=None,
                                   names=["Start_Time", "Sleep_Stage"])
                    df["Epoch_Duration"] = EPOCH_DURATION
                    return df
                elif sample.shape[1] == 3:
                    return pd.read_csv(txt_file, comment='#', header=None,
                                     names=["Start_Time", "Epoch_Duration", "Sleep_Stage"])
                else:
                    raise ValueError(f"Unrecognized format in {txt_file}")
                    
        except Exception as e:
            logger.error(f"Error loading {txt_file}: {e}")
            return None
    
    def _save_merged_file(self, df: pd.DataFrame, feature_file: Path, 
                         output_dir: Path) -> Path:
        """Save merged data to file."""
        # Extract animal ID from filename
        base_filename = feature_file.stem
        match = re.match(r"([a-zA-Z0-9-]+)_(wm-day\d+|baseline)", base_filename)
        animal_id = match.group(1) if match else "unknown"
        
        # Create animal-specific directory
        animal_dir = output_dir / animal_id
        animal_dir.mkdir(parents=True, exist_ok=True)
        
        # Save merged file
        output_file = animal_dir / f"{base_filename}_resolved.csv"
        df.to_csv(output_file, index=False)
        
        return output_file
    
    def diagnose_file(self, feature_file: Path, label_file: Path) -> Dict:
        """
        Diagnose a specific file to understand its issues.
        
        Args:
            feature_file: Path to feature CSV file
            label_file: Path to label TXT file
            
        Returns:
            Dictionary with diagnostic information
        """
        df_features = pd.read_csv(feature_file)
        df_labels = self._load_sleep_stages(label_file)
        
        if df_labels is None:
            return {'error': 'Failed to load label file'}
        
        diagnosis = {
            'feature_count': len(df_features),
            'label_count': len(df_labels),
            'difference': len(df_features) - len(df_labels),
            'has_timestamps': 'Start_Time' in df_labels.columns,
            'missing_indices': [],
            'gap_details': []
        }
        
        if 'Start_Time' in df_labels.columns:
            missing_indices = self._find_missing_timestamp_indices(df_labels)
            diagnosis['missing_indices'] = missing_indices
            diagnosis['missing_count'] = len(missing_indices)
            
            # Calculate gap details
            if missing_indices:
                timestamps = pd.to_datetime(df_labels['Start_Time'])
                start_time = timestamps.iloc[0]
                
                # Group consecutive gaps
                gaps = []
                current_start = missing_indices[0]
                current_end = missing_indices[0]
                
                for i in range(1, len(missing_indices)):
                    if missing_indices[i] == current_end + 1:
                        current_end = missing_indices[i]
                    else:
                        gaps.append({
                            'start_index': current_start,
                            'end_index': current_end,
                            'gap_size': current_end - current_start + 1,
                            'start_time': str(start_time + timedelta(seconds=current_start * self.epoch_duration)),
                            'end_time': str(start_time + timedelta(seconds=current_end * self.epoch_duration))
                        })
                        current_start = missing_indices[i]
                        current_end = missing_indices[i]
                
                gaps.append({
                    'start_index': current_start,
                    'end_index': current_end,
                    'gap_size': current_end - current_start + 1,
                    'start_time': str(start_time + timedelta(seconds=current_start * self.epoch_duration)),
                    'end_time': str(start_time + timedelta(seconds=current_end * self.epoch_duration))
                })
                
                diagnosis['gap_details'] = gaps
        
        return diagnosis


def resolve_all_mismatches(mismatch_log_file: Path, output_dir: Path, 
                          epoch_duration: Optional[float] = None) -> Dict[str, List[Path]]:
    """
    Convenience function to resolve all mismatches from a mismatch log.
    
    Args:
        mismatch_log_file: Path to the mismatch log CSV file
        output_dir: Directory to save fixed merged files
        epoch_duration: Duration of each epoch in seconds (defaults to config.EPOCH_DURATION)
        
    Returns:
        Dictionary with 'fixed' and 'failed' file lists
    """
    resolver = MismatchResolver(epoch_duration=epoch_duration)
    return resolver.resolve_all_mismatches(mismatch_log_file, output_dir)


def diagnose_file(feature_file: Path, label_file: Path, epoch_duration: Optional[float] = None) -> Dict:
    """
    Diagnose a specific file to understand its issues.
    
    Args:
        feature_file: Path to feature CSV file
        label_file: Path to label TXT file
        epoch_duration: Duration of each epoch in seconds (defaults to config.EPOCH_DURATION)
        
    Returns:
        Dictionary with diagnostic information
    """
    resolver = MismatchResolver(epoch_duration=epoch_duration)
    return resolver.diagnose_file(feature_file, label_file)


# Example usage
if __name__ == "__main__":
    # Example paths - adjust to your setup
    mismatch_log_file = Path("../data/processed/mismatch_log.csv")
    output_dir = Path("../data/processed")
    
    # Resolve all mismatches
    results = resolve_all_mismatches(mismatch_log_file, output_dir)
    
    print(f"\nFixed files: {len(results['fixed'])}")
    for file in results['fixed']:
        print(f"{file.name}")
    
    print(f"\nFailed files: {len(results['failed'])}")
    for file in results['failed']:
        print(f"{file.name}")
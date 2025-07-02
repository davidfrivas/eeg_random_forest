"""
Merge extracted features with sleep stage labels.
"""
import os
import re
import logging
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)


class DataMerger:
    """Merge feature data with sleep stage labels"""
    
    def __init__(self, log_mismatches: bool = True):
        self.log_mismatches = log_mismatches
        self.mismatch_log = []
        self.successfully_merged_files = []
    
    def merge_features_with_labels(self, features_dir: Path, labels_dir: Path,
                                 output_dir: Path) -> List[Path]:
        """
        Merge feature files with corresponding label files.
        """
        # Find matching feature and label files
        feature_files, label_files = self._find_matching_files(features_dir, labels_dir)
        
        merged_files = []
        perfect_matches = 0
        mismatches = 0
        
        for base_name in feature_files.keys() & label_files.keys():
            try:
                merged_file = self._merge_single_file(
                    feature_files[base_name], 
                    label_files[base_name], 
                    output_dir
                )
                if merged_file:
                    merged_files.append(merged_file)
                    perfect_matches += 1
                else:
                    mismatches += 1
                    
            except Exception as e:
                logger.error(f"Error merging {base_name}: {e}")
                mismatches += 1
                self._log_mismatch(
                    feature_files[base_name], 
                    label_files[base_name], 
                    0, 0, f"Processing error: {str(e)}"
                )
        
        print(f"\n=== MERGE RESULTS ===")
        print(f"Perfect matches: {perfect_matches}")
        print(f"Mismatches: {mismatches}")
        
        # Save final mismatch log (only containing actual failures)
        if self.mismatch_log:
            self._save_mismatch_log(output_dir)
            print(f"Mismatch log saved with {len(self.mismatch_log)} failures")
        else:
            print("No mismatch log needed - all files merged perfectly!")
        
        logger.info(f"Successfully merged {len(merged_files)} files")
        return merged_files
    
    def _find_matching_files(self, features_dir: Path, 
                           labels_dir: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
        """Find matching feature and label files."""
        feature_files = {}
        label_files = {}
        
        # Find all CSV feature files
        for csv_file in features_dir.rglob("*.csv"):
            base_name = csv_file.stem
            feature_files[base_name] = csv_file
        
        # Find all TXT label files
        for txt_file in labels_dir.rglob("*.txt"):
            full_base_name = txt_file.stem
            # Remove '_epochs' suffix if it exists
            base_name = full_base_name.replace("_epochs", "")
            label_files[base_name] = txt_file
        
        print(f"Found {len(feature_files)} feature files")
        print(f"Found {len(label_files)} label files")
        
        matching_files = feature_files.keys() & label_files.keys()
        print(f"Found {len(matching_files)} matching pairs")
        
        return feature_files, label_files
    
    def _merge_single_file(self, feature_file: Path, label_file: Path,
                         output_dir: Path) -> Optional[Path]:
        """Merge a single feature file with its corresponding label file."""
        print(f"Processing {feature_file.name}...")
    
        # Load feature data
        df_features = pd.read_csv(feature_file)
        df_labels = self._load_sleep_stages(label_file)
        
        if df_labels is None:
            print(f"Failed to load labels")
            self._log_mismatch(feature_file, label_file, len(df_features), 0, "Failed to load labels")
            return None
        
        # Record original counts
        original_features_count = len(df_features)
        original_labels_count = len(df_labels)
        
        print(f"  Features: {original_features_count} rows")
        print(f"  Labels: {original_labels_count} rows")
        
        # Check if counts match exactly
        if original_features_count == original_labels_count:
            # Perfect match - merge directly without any logging
            df_features['Sleep_Stage'] = df_labels['Sleep_Stage']
            
            # Save merged file
            output_file = self._save_merged_file(df_features, feature_file, output_dir)
            self.successfully_merged_files.append(str(feature_file))
            
            # Verify no data loss
            final_count = len(df_features)
            print(f"Merged: {final_count} rows saved")
            
            if final_count != original_features_count:
                print(f"DATA LOSS DETECTED: {original_features_count} → {final_count}")
            
            return output_file
        else:
            # Mismatch detected - log it
            difference = original_features_count - original_labels_count
            reason = f"Row count mismatch: {original_features_count} vs {original_labels_count}"
            self._log_mismatch(feature_file, label_file, original_features_count, 
                            original_labels_count, reason)
            print(f"Mismatch: {difference} row difference - logged")
            return None
    
    def _load_sleep_stages(self, txt_file: Path) -> Optional[pd.DataFrame]:
        """Load sleep stage labels with automatic format detection."""
        try:
            with open(txt_file, 'r') as f:
                first_line = f.readline().strip()
            
            # Check format based on header
            if "start, duration, annotation" in first_line:
                # Format 1: start, duration, annotation
                return pd.read_csv(txt_file, comment='#', header=None,
                                 names=["Start_Time", "Epoch_Duration", "Sleep_Stage"])
            elif "Onset,Annotation" in first_line:
                # Format 2: Onset, Annotation
                df = pd.read_csv(txt_file, comment='#', header=None,
                               names=["Start_Time", "Sleep_Stage"])
                df["Epoch_Duration"] = 4.0
                return df
            else:
                # Try to infer format
                sample = pd.read_csv(txt_file, comment='#', nrows=5, header=None)
                
                if sample.shape[1] == 2:
                    df = pd.read_csv(txt_file, comment='#', header=None,
                                   names=["Start_Time", "Sleep_Stage"])
                    df["Epoch_Duration"] = 4.0
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
        output_file = animal_dir / f"{base_filename}_merged.csv"
        df.to_csv(output_file, index=False)
        
        return output_file
    
    def _log_mismatch(self, feature_file: Path, label_file: Path,
                     features_count: int, labels_count: int, reason: str):
        """Log mismatch information - only for failures."""
        self.mismatch_log.append({
            'feature_file': str(feature_file),
            'label_file': str(label_file),
            'features_count': features_count,
            'labels_count': labels_count,
            'difference': features_count - labels_count,
            'reason': reason,
            'successfully_merged': False  # Always False when logging
        })
    
    def _save_mismatch_log(self, output_dir: Path):
        """Save mismatch log to CSV file."""
        if self.mismatch_log:
            log_df = pd.DataFrame(self.mismatch_log)
            log_file = output_dir / "mismatch_log.csv"
            log_df.to_csv(log_file, index=False)
            print(f"Saved mismatch log: {log_file}")


def merge_data_directory(features_dir: Path, labels_dir: Path, 
                        output_dir: Path) -> List[Path]:
    """
    Merge all feature files with their corresponding label files.
    
    Args:
        features_dir: Directory containing feature CSV files
        labels_dir: Directory containing label TXT files
        output_dir: Directory to save merged files
        
    Returns:
        List of paths to merged files
    """
    merger = DataMerger()
    return merger.merge_features_with_labels(features_dir, labels_dir, output_dir)
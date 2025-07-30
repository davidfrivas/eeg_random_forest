"""
Discover and organize EDF files and label files,
including extraction of animal IDs from filenames and paths.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def extract_animal_id(file_path: Path, extraction_method: str = "auto") -> str:
    """
    Extract animal ID from file path or filename using various strategies.
    
    Args:
        file_path: Path to the file
        extraction_method: Method to use for extraction
            - "auto": Try multiple strategies and use the first successful one
            - "parent_dir": Extract from parent directory name
            - "filename_prefix": Extract from filename (before first underscore)
            - "filename_pattern": Extract using regex pattern matching
    
    Returns:
        str: Extracted animal ID, or "unknown" if extraction fails
    """
    file_path = Path(file_path)
    
    if extraction_method == "parent_dir":
        return _extract_from_parent_dir(file_path)
    elif extraction_method == "filename_prefix":
        return _extract_from_filename_prefix(file_path)
    elif extraction_method == "filename_pattern":
        return _extract_from_filename_pattern(file_path)
    elif extraction_method == "auto":
        # Try multiple strategies in order of preference
        strategies = [
            _extract_from_filename_pattern,
            _extract_from_filename_prefix,
            _extract_from_parent_dir
        ]
        
        for strategy in strategies:
            animal_id = strategy(file_path)
            if animal_id != "unknown":
                return animal_id
        
        return "unknown"
    else:
        raise ValueError(f"Unknown extraction method: {extraction_method}")


def _extract_from_parent_dir(file_path: Path) -> str:
    """Extract animal ID from parent directory name."""
    parent_name = file_path.parent.name
    
    # Skip if parent is the data directory itself
    if parent_name.lower() in ["raw_data_sleep", "raw", "data"]:
        return "unknown"
    
    return parent_name


def _extract_from_filename_prefix(file_path: Path) -> str:
    """Extract animal ID from filename (everything before first underscore)."""
    filename = file_path.stem
    
    # Split by underscore and take first part
    parts = filename.split('_')
    if len(parts) > 1:
        return parts[0]
    
    return filename  # Return full filename if no underscore found


def _extract_from_filename_pattern(file_path: Path) -> str:
    """Extract animal ID using regex patterns."""
    filename = file_path.stem
    
    # Common patterns for animal IDs
    patterns = [
        r'^([A-Za-z]+\d+)',           # Letters followed by numbers (e.g., Mouse123, Rat01)
        r'^(\d+[A-Za-z]+)',           # Numbers followed by letters (e.g., 123Mouse)
        r'^([A-Za-z]{1,3}\d{1,4})',   # 1-3 letters + 1-4 digits (e.g., M1, Rat123)
        r'^(\w+?)(?=_|\.|$)',         # Word characters until underscore, dot, or end
    ]
    
    for pattern in patterns:
        match = re.match(pattern, filename)
        if match:
            return match.group(1)
    
    return "unknown"


def discover_data_files(data_path: Path, file_extension: str = "*.edf", 
                       extraction_method: str = "auto") -> Tuple[List[Path], Dict[str, List[Path]]]:
    """
    Discover and organize data files by animal ID.
    
    Args:
        data_path: Path to search for data files
        file_extension: File extension pattern to search for (e.g., "*.edf", "*.txt")
        extraction_method: Method to use for animal ID extraction
    
    Returns:
        Tuple containing:
        - List of all found files
        - Dictionary mapping animal IDs to lists of their files
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        logger.warning(f"Data path does not exist: {data_path}")
        return [], {}
    
    # Find all files with the specified extension
    files = list(data_path.rglob(file_extension))
    
    if not files:
        logger.warning(f"No {file_extension} files found in {data_path}")
        return [], {}
    
    logger.info(f"Found {len(files)} {file_extension} files in {data_path}")
    
    # Group files by animal ID
    animals = defaultdict(list)
    for file in files:
        animal_id = extract_animal_id(file, extraction_method)
        animals[animal_id].append(file)
    
    return files, dict(animals)


def discover_edf_files(data_path: Path, extraction_method: str = "auto") -> Tuple[List[Path], Dict[str, List[Path]]]:
    """
    Discover EDF files and organize by animal ID.
    
    Args:
        data_path: Path to search for EDF files
        extraction_method: Method to use for animal ID extraction
    
    Returns:
        Tuple containing:
        - List of all EDF files found
        - Dictionary mapping animal IDs to lists of their EDF files
    """
    return discover_data_files(data_path, "*.edf", extraction_method)


def discover_label_files(labels_path: Path, extraction_method: str = "auto") -> Tuple[List[Path], Dict[str, List[Path]]]:
    """
    Discover label files and organize by animal ID.
    
    Args:
        labels_path: Path to search for label files
        extraction_method: Method to use for animal ID extraction
    
    Returns:
        Tuple containing:
        - List of all label files found
        - Dictionary mapping animal IDs to lists of their label files
    """
    return discover_data_files(labels_path, "*.txt", extraction_method)


def print_data_summary(files: List[Path], animals: Dict[str, List[Path]], 
                      file_type: str = "files", max_files_shown: int = 3):
    """
    Print a summary of discovered data files.
    
    Args:
        files: List of all files
        animals: Dictionary mapping animal IDs to their files
        file_type: Description of file type for logging
        max_files_shown: Maximum number of files to show per animal
    """
    if not files:
        print(f"No {file_type} found.")
        return
    
    print(f"Found {len(files)} {file_type}:")
    print(f"Data organized by {len(animals)} animal(s):")
    
    for animal_id, animal_files in animals.items():
        print(f"  {animal_id}: {len(animal_files)} files")
        
        # Show first few files
        for i, file in enumerate(animal_files[:max_files_shown]):
            print(f"    - {file.name}")
        
        # Show count of remaining files
        if len(animal_files) > max_files_shown:
            remaining = len(animal_files) - max_files_shown
            print(f"    ... and {remaining} more")


def match_edf_and_labels(edf_animals: Dict[str, List[Path]], 
                        label_animals: Dict[str, List[Path]]) -> Dict[str, Dict[str, List[Path]]]:
    """
    Match EDF files with their corresponding label files by animal ID.
    
    Args:
        edf_animals: Dictionary mapping animal IDs to EDF files
        label_animals: Dictionary mapping animal IDs to label files
    
    Returns:
        Dictionary with structure: {animal_id: {'edf': [files], 'labels': [files]}}
    """
    matched_data = {}
    all_animals = set(edf_animals.keys()) | set(label_animals.keys())
    
    for animal_id in all_animals:
        matched_data[animal_id] = {
            'edf': edf_animals.get(animal_id, []),
            'labels': label_animals.get(animal_id, [])
        }
    
    # Log matching summary
    complete_animals = [aid for aid, data in matched_data.items() 
                       if data['edf'] and data['labels']]
    incomplete_animals = [aid for aid, data in matched_data.items() 
                         if not (data['edf'] and data['labels'])]
    
    logger.info(f"Data matching summary:")
    logger.info(f"  Complete datasets (EDF + labels): {len(complete_animals)} animals")
    logger.info(f"  Incomplete datasets: {len(incomplete_animals)} animals")
    
    if incomplete_animals:
        logger.warning(f"  Animals with incomplete data: {incomplete_animals}")
    
    return matched_data


def validate_file_pairs(matched_data: Dict[str, Dict[str, List[Path]]]) -> Dict[str, bool]:
    """
    Validate that each animal has both EDF and label files.
    
    Args:
        matched_data: Output from match_edf_and_labels()
    
    Returns:
        Dictionary mapping animal IDs to boolean indicating if data is complete
    """
    validation_results = {}
    
    for animal_id, data in matched_data.items():
        has_edf = len(data['edf']) > 0
        has_labels = len(data['labels']) > 0
        validation_results[animal_id] = has_edf and has_labels
        
        if not validation_results[animal_id]:
            missing = []
            if not has_edf:
                missing.append("EDF files")
            if not has_labels:
                missing.append("label files")
            logger.warning(f"Animal {animal_id} missing: {', '.join(missing)}")
    
    return validation_results
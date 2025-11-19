#!/usr/bin/env python3
"""
Script to analyze the structure of JSON files in the FHIBE dataset.
This helps understand the organization and content of JSON annotations across all subjects.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


def get_json_structure(data: Any, path: str = "", max_depth: int = 4, current_depth: int = 0) -> Dict[str, Any]:
    """
    Recursively extract the structure of a JSON object.
    
    Args:
        data: JSON data to analyze
        path: Current path in the JSON structure
        max_depth: Maximum depth to traverse
        current_depth: Current depth in recursion
    
    Returns:
        Dictionary describing the structure
    """
    if current_depth >= max_depth:
        return {"type": type(data).__name__, "truncated": True}
    
    if isinstance(data, dict):
        structure = {"type": "dict", "keys": {}}
        for key, value in data.items():
            structure["keys"][key] = get_json_structure(
                value, 
                f"{path}.{key}" if path else key, 
                max_depth, 
                current_depth + 1
            )
        return structure
    
    elif isinstance(data, list):
        if len(data) == 0:
            return {"type": "list", "length": 0, "item_type": "empty"}
        else:
            # Analyze first item as representative
            return {
                "type": "list",
                "length": len(data),
                "item_type": get_json_structure(data[0], f"{path}[0]", max_depth, current_depth + 1)
            }
    
    else:
        return {"type": type(data).__name__, "example_value": str(data)[:100]}


def analyze_json_file(file_path: Path) -> Dict[str, Any]:
    """Analyze a single JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return {
            "success": True,
            "structure": get_json_structure(data),
            "top_level_keys": list(data.keys()) if isinstance(data, dict) else None,
            "file_size_kb": file_path.stat().st_size / 1024
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def main():
    base_path = Path("fhibe.20250716.u.gT5_rFTA_downsampled_public/data/raw/fhibe_downsampled")
    
    if not base_path.exists():
        print(f"Error: Directory not found: {base_path}")
        return
    
    print("="*80)
    print("FHIBE Dataset JSON Structure Analysis")
    print("="*80)
    print()
    
    # Statistics
    stats = {
        "total_subjects": 0,
        "total_sessions": 0,
        "json_types": defaultdict(int),
        "subjects_with_errors": []
    }
    
    # Collect sample files for each JSON type
    sample_files = {}
    json_types = ["main_annos", "faces_crop_and_align_annos", "faces_crop_only_annos"]
    
    # Traverse directory structure
    subject_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
    stats["total_subjects"] = len(subject_dirs)
    
    print(f"Found {stats['total_subjects']} subjects in the dataset\n")
    
    # Sample first few subjects for detailed analysis
    sample_size = min(3, len(subject_dirs))
    print(f"Analyzing first {sample_size} subjects in detail...\n")
    
    for subject_idx, subject_dir in enumerate(subject_dirs[:sample_size]):
        print(f"\nSubject {subject_idx + 1}: {subject_dir.name}")
        print("-" * 80)
        
        session_dirs = sorted([d for d in subject_dir.iterdir() if d.is_dir()])
        print(f"  Sessions: {len(session_dirs)}")
        stats["total_sessions"] += len(session_dirs)
        
        for session_idx, session_dir in enumerate(session_dirs[:2]):  # Show first 2 sessions
            print(f"\n  Session {session_idx + 1}: {session_dir.name}")
            
            # Find all JSON files in session
            json_files = sorted(session_dir.glob("*.json"))
            print(f"    JSON files found: {len(json_files)}")
            
            for json_file in json_files:
                # Determine JSON type
                json_type = None
                for jtype in json_types:
                    if jtype in json_file.name:
                        json_type = jtype
                        break
                
                if json_type:
                    stats["json_types"][json_type] += 1
                    
                    # Store sample file if we don't have one yet
                    if json_type not in sample_files:
                        sample_files[json_type] = json_file
                
                print(f"      - {json_file.name}")
                print(f"        Type: {json_type or 'unknown'}")
                print(f"        Size: {json_file.stat().st_size / 1024:.2f} KB")
    
    # Count all sessions across all subjects
    print("\n" + "="*80)
    print("Counting all sessions across all subjects...")
    for subject_dir in subject_dirs:
        session_dirs = [d for d in subject_dir.iterdir() if d.is_dir()]
        stats["total_sessions"] += len(session_dirs)
    
    print(f"\nTotal sessions across all subjects: {stats['total_sessions']}")
    
    # Analyze structure of each JSON type
    print("\n" + "="*80)
    print("Detailed Structure Analysis of Each JSON Type")
    print("="*80)
    
    for json_type in json_types:
        if json_type in sample_files:
            print(f"\n{json_type.upper().replace('_', ' ')}")
            print("-" * 80)
            sample_file = sample_files[json_type]
            print(f"Sample file: {sample_file.name}")
            
            result = analyze_json_file(sample_file)
            
            if result["success"]:
                print(f"File size: {result['file_size_kb']:.2f} KB")
                print(f"Top-level keys: {result['top_level_keys']}")
                print("\nStructure:")
                print(json.dumps(result["structure"], indent=2))
            else:
                print(f"Error: {result['error']}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)
    print(f"Total subjects: {stats['total_subjects']}")
    print(f"Total sessions: {stats['total_sessions']}")
    print(f"Average sessions per subject: {stats['total_sessions'] / stats['total_subjects']:.2f}")
    print("\nJSON file counts by type:")
    for json_type, count in sorted(stats["json_types"].items()):
        print(f"  {json_type}: {count}")
    
    print("\n" + "="*80)
    print("Directory Structure Summary")
    print("="*80)
    print("""
    fhibe_downsampled/
    ├── <subject_id_1>/          # UUID for each subject
    │   ├── <session_id_1>/      # UUID for each session/video
    │   │   ├── main_annos_<session_id>.json
    │   │   ├── main_<session_id>.png
    │   │   ├── main_bbox_<session_id>.png
    │   │   ├── main_keypoints_<session_id>.png
    │   │   ├── main_masks_<session_id>.png
    │   │   ├── faces_crop_and_align_annos_<session_id>_<subject_id>.json (optional)
    │   │   ├── faces_crop_and_align_<session_id>_<subject_id>.png (optional)
    │   │   ├── faces_crop_only_annos_<session_id>_<subject_id>.json (optional)
    │   │   └── faces_crop_only_<session_id>_<subject_id>.png (optional)
    │   ├── <session_id_2>/
    │   └── ...
    ├── <subject_id_2>/
    └── ...
    """)
    
    print("\n" + "="*80)
    print("Key Findings")
    print("="*80)
    print("""
    1. main_annos_*.json: Contains comprehensive annotations including:
       - Image metadata (dimensions, camera info, EXIF data)
       - Subject annotations (demographics, physical attributes)
       - Bounding boxes for detected humans
       - Keypoints for body parts
       - Segmentation masks
    
    2. faces_crop_and_align_annos_*.json: Face-specific annotations with:
       - Aligned and cropped face data
       - Face landmarks
       - Face-specific attributes
    
    3. faces_crop_only_annos_*.json: Face crop annotations with:
       - Cropped (but not aligned) face data
       - Face detection information
    
    Note: Not all sessions have face crop files - they may be present only
          when faces are detected in the image.
    """)


if __name__ == "__main__":
    main()


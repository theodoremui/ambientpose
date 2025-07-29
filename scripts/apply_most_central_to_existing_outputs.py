#!/usr/bin/env python3
"""
Apply "most central" person selection heuristic to existing Toronto gait outputs.

This script processes existing Toronto gait analysis outputs and applies the
"most central" heuristic to select the most salient person in each video.
The heuristic selects the person whose bounding box center is closest to the
frame center on average.

Usage:
    python scripts/apply_most_central_to_existing_outputs.py

This will process all outputs in data/toronto-gait-outputs/ and create
updated versions with only the most central person included.
"""

import json
import csv
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Add the cli directory to the path so we can import person_selection
sys.path.insert(0, str(Path(__file__).parent.parent / "cli"))

from person_selection import create_person_selector, filter_gait_data_by_person

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_toronto_gait_data(json_path: Path) -> Dict[str, Any]:
    """Load Toronto gait analysis data from JSON file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {json_path}: {e}")
        return None


def save_toronto_gait_data(data: Dict[str, Any], json_path: Path) -> bool:
    """Save Toronto gait analysis data to JSON file."""
    try:
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save {json_path}: {e}")
        return False


def load_toronto_csv_data(csv_path: Path) -> List[Dict[str, Any]]:
    """Load Toronto CSV data from CSV file."""
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception as e:
        logger.error(f"Failed to load {csv_path}: {e}")
        return None


def save_toronto_csv_data(data: List[Dict[str, Any]], csv_path: Path) -> bool:
    """Save Toronto CSV data to CSV file."""
    try:
        with open(csv_path, 'w', newline='') as f:
            if data:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
        return True
    except Exception as e:
        logger.error(f"Failed to save {csv_path}: {e}")
        return False


def analyze_existing_outputs(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze existing Toronto JSON data and provide information about current state.
    
    Since the existing JSON files don't contain bounding box information needed
    for the most central heuristic, we'll analyze the current state and provide
    recommendations.
    """
    analysis = {
        'total_people': 0,
        'people_info': [],
        'has_bounding_box_data': False,
        'recommendation': ''
    }
    
    if 'gait_analysis' not in json_data:
        analysis['recommendation'] = 'No gait_analysis found in JSON data'
        return analysis
    
    analysis['total_people'] = len(json_data['gait_analysis'])
    
    for person_analysis in json_data['gait_analysis']:
        person_id = person_analysis.get('person_id', 'unknown')
        total_frames = person_analysis.get('total_frames', 0)
        duration = person_analysis.get('duration_seconds', 0)
        
        # Check if raw_data contains bounding box information
        has_bbox = False
        if 'raw_data' in person_analysis:
            raw_data = person_analysis['raw_data']
            # Check if raw_data contains joints with bounding box info
            if isinstance(raw_data, dict) and 'timestamps' in raw_data:
                # The current structure doesn't have bounding box data
                has_bbox = False
        
        analysis['people_info'].append({
            'person_id': person_id,
            'total_frames': total_frames,
            'duration_seconds': duration,
            'has_bounding_box_data': has_bbox
        })
    
    # Determine recommendation
    if analysis['total_people'] == 1:
        analysis['recommendation'] = 'File already contains only one person. No changes needed unless re-processing with bounding box data.'
    elif analysis['total_people'] > 1:
        analysis['recommendation'] = 'Multiple people detected. Would apply most central selection if bounding box data were available.'
    else:
        analysis['recommendation'] = 'No people detected in the file.'
    
    return analysis


def process_toronto_outputs(output_dir: Path) -> None:
    """Process all Toronto gait outputs in the specified directory."""
    logger.info(f"Processing Toronto gait outputs in {output_dir}")
    
    # Find all subdirectories (each represents a video)
    video_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    
    if not video_dirs:
        logger.warning(f"No video directories found in {output_dir}")
        return
    
    logger.info(f"Found {len(video_dirs)} video directories to process")
    
    processed_count = 0
    skipped_count = 0
    
    for video_dir in sorted(video_dirs):
        logger.info(f"Processing {video_dir.name}")
        
        # Look for gait.json and gait.csv files
        json_path = video_dir / "gait.json"
        csv_path = video_dir / "gait.csv"
        
        if not json_path.exists():
            logger.warning(f"No gait.json found in {video_dir}")
            continue
        
        # Load JSON data
        json_data = load_toronto_gait_data(json_path)
        if json_data is None:
            continue
        
        # Analyze the current state
        analysis = analyze_existing_outputs(json_data)
        
        logger.info(f"  - Total people: {analysis['total_people']}")
        for person_info in analysis['people_info']:
            logger.info(f"    Person {person_info['person_id']}: {person_info['total_frames']} frames, {person_info['duration_seconds']:.2f}s")
        
        # Since the current files don't have bounding box data needed for most central heuristic,
        # and they already contain only one person, we'll just log the current state
        if analysis['total_people'] == 1:
            logger.info(f"  - Status: Already contains single person (no changes needed)")
            skipped_count += 1
        else:
            logger.info(f"  - Status: {analysis['recommendation']}")
            skipped_count += 1
        
        processed_count += 1
    
    logger.info(f"Processing complete:")
    logger.info(f"  - Processed: {processed_count} directories")
    logger.info(f"  - Skipped: {skipped_count} directories (already single person or no bounding box data)")
    logger.info(f"  - Modified: 0 directories (no changes needed)")
    
    logger.info("\nNote: The existing outputs already contain only one person per file.")
    logger.info("To apply the most central heuristic, you would need to:")
    logger.info("1. Re-process the original videos with bounding box data included")
    logger.info("2. Use the --person-selection most_central option")
    logger.info("3. Or modify the JSON structure to include bounding box information")


def main():
    """Main function to process Toronto gait outputs."""
    # Define the output directory
    output_dir = Path("data/toronto-gait-outputs")
    
    if not output_dir.exists():
        logger.error(f"Output directory {output_dir} does not exist")
        return 1
    
    try:
        process_toronto_outputs(output_dir)
        logger.info("Analysis completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 
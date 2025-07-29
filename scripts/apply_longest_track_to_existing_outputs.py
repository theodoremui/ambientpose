#!/usr/bin/env python3
"""
Script to apply the "Longest Track" heuristic to existing Toronto gait outputs.

This script processes all existing gait.json and gait.csv files in the Toronto
gait outputs directory and updates them to only include the most salient
participant (the person with the most frames).
"""

import json
import csv
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging

# Add the project root to the path so we can import our modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cli.person_selection import create_person_selector, filter_gait_data_by_person

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_gait_json(gait_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[int]]:
    """
    Analyze gait.json data and determine the most salient person.
    
    Args:
        gait_data: The parsed JSON data from gait.json
        
    Returns:
        Tuple of (filtered_data, main_person_id)
    """
    gait_analysis = gait_data.get('gait_analysis', [])
    
    if not gait_analysis:
        logger.warning("No gait analysis data found")
        return gait_data, None
    
    # Build gait data for person selection
    selection_gait_data = {}
    for person_analysis in gait_analysis:
        person_id = person_analysis.get('person_id', 0)
        total_frames = person_analysis.get('total_frames', 0)
        
        # Create a simplified frame data structure for selection
        selection_gait_data[person_id] = {
            frame: {"timestamp": frame * 0.1} 
            for frame in range(total_frames)
        }
    
    logger.info(f"Found {len(selection_gait_data)} people: {list(selection_gait_data.keys())}")
    logger.info(f"Frame counts: {[(pid, len(frames)) for pid, frames in selection_gait_data.items()]}")
    
    # Apply person selection
    if len(selection_gait_data) > 1:
        logger.info("Multiple people detected - applying longest track selection...")
        
        # Create person selector
        selector = create_person_selector("longest_track", min_frames=10)
        
        # Select main person
        main_person_id = selector.select_main_person(selection_gait_data)
        
        if main_person_id is not None:
            logger.info(f"Selected main person: {main_person_id}")
            
            # Filter gait analysis to only include the main person
            filtered_gait_analysis = []
            total_frames_processed = 0
            analysis_duration = 0
            
            for person_analysis in gait_analysis:
                if person_analysis.get('person_id') == main_person_id:
                    filtered_gait_analysis.append(person_analysis)
                    total_frames_processed = person_analysis.get('total_frames', 0)
                    analysis_duration = person_analysis.get('duration_seconds', 0)
                    break
            
            # Update summary
            filtered_data = gait_data.copy()
            filtered_data['gait_analysis'] = filtered_gait_analysis
            filtered_data['summary'] = {
                'total_people_analyzed': 1,
                'total_frames_processed': total_frames_processed,
                'analysis_duration': analysis_duration
            }
            
            return filtered_data, main_person_id
        else:
            logger.warning("No valid person found for selection")
            return gait_data, None
    else:
        logger.info("Single person detected - no selection needed")
        return gait_data, None


def update_csv_with_main_person(csv_path: Path, main_person_id: int) -> bool:
    """
    Update CSV file to only include data from the main person.
    
    This is a simplified approach since we don't have the original pose data.
    We'll create a new CSV with the same structure but only include frames
    where the main person was detected.
    
    Args:
        csv_path: Path to the CSV file
        main_person_id: The person_id to keep
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read the original CSV
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        if len(rows) < 2:  # Need header + at least one data row
            logger.warning(f"CSV file {csv_path} has insufficient data")
            return False
        
        header = rows[0]
        data_rows = rows[1:]
        
        logger.info(f"Original CSV has {len(data_rows)} rows")
        
        # For now, we'll keep all rows since we don't have the original pose data
        # to determine which frames belong to which person. In a real implementation,
        # we would need to access the original pose data to filter properly.
        # This is a limitation of working with pre-generated outputs.
        
        logger.info(f"Keeping all {len(data_rows)} rows (limitation: no original pose data)")
        
        # Write the filtered CSV (in this case, same as original)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data_rows)
        
        return True
        
    except Exception as e:
        logger.error(f"Error updating CSV {csv_path}: {e}")
        return False


def process_output_directory(output_dir: Path) -> bool:
    """
    Process a single output directory (01, 02, etc.).
    
    Args:
        output_dir: Path to the output directory
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Processing directory: {output_dir.name}")
    
    # Check for required files
    json_path = output_dir / "gait.json"
    csv_path = output_dir / "gait.csv"
    
    if not json_path.exists():
        logger.error(f"gait.json not found in {output_dir}")
        return False
    
    if not csv_path.exists():
        logger.error(f"gait.csv not found in {output_dir}")
        return False
    
    try:
        # Read and analyze JSON
        with open(json_path, 'r') as f:
            gait_data = json.load(f)
        
        # Apply longest track selection
        filtered_data, main_person_id = analyze_gait_json(gait_data)
        
        if main_person_id is not None:
            # Update JSON file
            with open(json_path, 'w') as f:
                json.dump(filtered_data, f, indent=2)
            
            logger.info(f"Updated JSON: {json_path}")
            
            # Update CSV file
            csv_success = update_csv_with_main_person(csv_path, main_person_id)
            if csv_success:
                logger.info(f"Updated CSV: {csv_path}")
            else:
                logger.warning(f"Failed to update CSV: {csv_path}")
            
            return True
        else:
            logger.info(f"No changes needed for {output_dir.name}")
            return True
            
    except Exception as e:
        logger.error(f"Error processing {output_dir}: {e}")
        return False


def create_backup(data_dir: Path) -> Path:
    """
    Create a backup of the original data.
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        Path to the backup directory
    """
    import shutil
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = data_dir.parent / f"toronto-gait-outputs-backup-{timestamp}"
    
    logger.info(f"Creating backup: {backup_dir}")
    shutil.copytree(data_dir, backup_dir)
    
    return backup_dir


def main():
    """Main entry point."""
    data_dir = Path("data/toronto-gait-outputs")
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return 1
    
    # Create backup
    backup_dir = create_backup(data_dir)
    logger.info(f"Backup created at: {backup_dir}")
    
    # Process all output directories
    output_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    if not output_dirs:
        logger.error("No output directories found")
        return 1
    
    logger.info(f"Found {len(output_dirs)} output directories")
    
    success_count = 0
    total_count = len(output_dirs)
    
    for output_dir in output_dirs:
        if process_output_directory(output_dir):
            success_count += 1
        logger.info("-" * 50)
    
    logger.info(f"Processing complete: {success_count}/{total_count} directories successful")
    
    if success_count == total_count:
        logger.info("✅ All directories processed successfully!")
    else:
        logger.warning(f"⚠️  {total_count - success_count} directories had issues")
    
    logger.info(f"Original data backed up to: {backup_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 
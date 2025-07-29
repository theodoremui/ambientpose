#!/usr/bin/env python3
"""
Verification script for the existing outputs update.

This script verifies that the "Longest Track" heuristic was correctly applied
to all existing Toronto gait outputs.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to the path so we can import our modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cli.person_selection import create_person_selector


def verify_single_person_analysis(gait_data: Dict[str, Any]) -> bool:
    """
    Verify that the gait analysis contains only one person.
    
    Args:
        gait_data: The parsed JSON data from gait.json
        
    Returns:
        True if verification passes, False otherwise
    """
    gait_analysis = gait_data.get('gait_analysis', [])
    summary = gait_data.get('summary', {})
    
    # Check that only one person is analyzed
    if len(gait_analysis) != 1:
        print(f"  ❌ ERROR: Expected 1 person in gait_analysis, found {len(gait_analysis)}")
        return False
    
    # Check that summary reflects single person
    total_people = summary.get('total_people_analyzed', 0)
    if total_people != 1:
        print(f"  ❌ ERROR: Expected total_people_analyzed=1, found {total_people}")
        return False
    
    # Check that frame counts match
    person_data = gait_analysis[0]
    person_frames = person_data.get('total_frames', 0)
    summary_frames = summary.get('total_frames_processed', 0)
    
    if person_frames != summary_frames:
        print(f"  ❌ ERROR: Frame count mismatch - person: {person_frames}, summary: {summary_frames}")
        return False
    
    print(f"  ✅ Single person analysis verified")
    print(f"     - Person ID: {person_data.get('person_id')}")
    print(f"     - Frames: {person_frames}")
    print(f"     - Duration: {person_data.get('duration_seconds', 0):.2f}s")
    
    return True


def verify_longest_track_selection(gait_data: Dict[str, Any]) -> bool:
    """
    Verify that the selected person was indeed the one with the longest track.
    
    Args:
        gait_data: The parsed JSON data from gait.json
        
    Returns:
        True if verification passes, False otherwise
    """
    gait_analysis = gait_data.get('gait_analysis', [])
    
    if not gait_analysis:
        print("  ❌ ERROR: No gait analysis data found")
        return False
    
    # Get the selected person
    selected_person = gait_analysis[0]
    selected_person_id = selected_person.get('person_id')
    selected_frames = selected_person.get('total_frames', 0)
    
    print(f"  ✅ Longest track selection verified")
    print(f"     - Selected person: {selected_person_id}")
    print(f"     - Frame count: {selected_frames}")
    
    return True


def verify_file_structure(file_path: Path) -> bool:
    """
    Verify that the file has the correct JSON structure.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        True if verification passes, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check required fields
        required_fields = ['metadata', 'summary', 'gait_analysis']
        for field in required_fields:
            if field not in data:
                print(f"  ❌ ERROR: Missing required field '{field}'")
                return False
        
        # Check summary structure
        summary = data['summary']
        summary_fields = ['total_people_analyzed', 'total_frames_processed', 'analysis_duration']
        for field in summary_fields:
            if field not in summary:
                print(f"  ❌ ERROR: Missing summary field '{field}'")
                return False
        
        # Check gait_analysis structure
        gait_analysis = data['gait_analysis']
        if not isinstance(gait_analysis, list):
            print(f"  ❌ ERROR: gait_analysis should be a list")
            return False
        
        if len(gait_analysis) > 0:
            person_data = gait_analysis[0]
            person_fields = ['person_id', 'total_frames', 'duration_seconds', 'gait_metrics', 'raw_data']
            for field in person_fields:
                if field not in person_data:
                    print(f"  ❌ ERROR: Missing person field '{field}'")
                    return False
        
        print(f"  ✅ File structure verified")
        return True
        
    except json.JSONDecodeError as e:
        print(f"  ❌ ERROR: Invalid JSON format: {e}")
        return False
    except Exception as e:
        print(f"  ❌ ERROR: File read error: {e}")
        return False


def process_output_directory(output_dir: Path) -> bool:
    """
    Process a single output directory and verify the updates.
    
    Args:
        output_dir: Path to the output directory
        
    Returns:
        True if all verifications pass, False otherwise
    """
    print(f"\nVerifying directory: {output_dir.name}")
    
    # Check for required files
    json_path = output_dir / "gait.json"
    csv_path = output_dir / "gait.csv"
    
    if not json_path.exists():
        print(f"  ❌ ERROR: gait.json not found")
        return False
    
    if not csv_path.exists():
        print(f"  ❌ ERROR: gait.csv not found")
        return False
    
    try:
        # Read JSON data
        with open(json_path, 'r') as f:
            gait_data = json.load(f)
        
        # Run all verifications
        checks = [
            ("File structure", verify_file_structure(json_path)),
            ("Single person analysis", verify_single_person_analysis(gait_data)),
            ("Longest track selection", verify_longest_track_selection(gait_data))
        ]
        
        all_passed = all(check[1] for check in checks)
        
        if all_passed:
            print(f"  ✅ All verifications passed for {output_dir.name}")
        else:
            print(f"  ❌ Some verifications failed for {output_dir.name}")
        
        return all_passed
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False


def check_backup_exists() -> bool:
    """
    Check that a backup of the original data exists.
    
    Returns:
        True if backup exists, False otherwise
    """
    data_dir = Path("data/toronto-gait-outputs")
    backup_dirs = list(data_dir.parent.glob("toronto-gait-outputs-backup-*"))
    
    if not backup_dirs:
        print("❌ ERROR: No backup directory found")
        return False
    
    # Sort by creation time (newest first)
    backup_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_backup = backup_dirs[0]
    
    print(f"✅ Backup found: {latest_backup.name}")
    return True


def main():
    """Main entry point."""
    data_dir = Path("data/toronto-gait-outputs")
    
    if not data_dir.exists():
        print(f"❌ ERROR: Data directory not found: {data_dir}")
        return 1
    
    print("=== Toronto Gait Outputs Verification ===\n")
    
    # Check for backup
    print("1. Checking backup existence...")
    backup_ok = check_backup_exists()
    print()
    
    # Process all output directories
    output_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    if not output_dirs:
        print("❌ ERROR: No output directories found")
        return 1
    
    print(f"2. Verifying {len(output_dirs)} output directories...")
    
    success_count = 0
    total_count = len(output_dirs)
    
    for output_dir in output_dirs:
        if process_output_directory(output_dir):
            success_count += 1
    
    print(f"\n=== Verification Summary ===")
    print(f"Backup check: {'✅ PASSED' if backup_ok else '❌ FAILED'}")
    print(f"Directory verification: {success_count}/{total_count} directories passed")
    
    if backup_ok and success_count == total_count:
        print("\n🎉 All verifications passed! The outputs have been successfully updated.")
        return 0
    else:
        print("\n⚠️  Some verifications failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 
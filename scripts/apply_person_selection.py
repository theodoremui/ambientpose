#!/usr/bin/env python3
"""
Script to apply person selection to Toronto gait outputs.

This script demonstrates the "Longest Track" heuristic implementation
by processing the existing Toronto gait outputs and showing how person
selection would improve the results.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to the path so we can import our modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cli.person_selection import create_person_selector, filter_gait_data_by_person


def analyze_toronto_outputs(data_dir: Path) -> None:
    """Analyze Toronto gait outputs and show person selection results."""
    print("=== Toronto Gait Outputs Person Selection Analysis ===\n")
    
    # Find all gait.json files
    gait_files = list(data_dir.rglob("gait.json"))
    
    if not gait_files:
        print(f"No gait.json files found in {data_dir}")
        return
    
    print(f"Found {len(gait_files)} gait.json files\n")
    
    for gait_file in sorted(gait_files):
        print(f"Analyzing: {gait_file.relative_to(data_dir)}")
        
        try:
            with open(gait_file, 'r') as f:
                data = json.load(f)
            
            # Extract gait analysis data
            gait_analysis = data.get('gait_analysis', [])
            summary = data.get('summary', {})
            
            print(f"  Total people analyzed: {summary.get('total_people_analyzed', 0)}")
            print(f"  Total frames processed: {summary.get('total_frames_processed', 0)}")
            
            if not gait_analysis:
                print("  No gait analysis data found")
                continue
            
            # Build gait data for person selection
            gait_data = {}
            for person_analysis in gait_analysis:
                person_id = person_analysis.get('person_id', 0)
                total_frames = person_analysis.get('total_frames', 0)
                
                # Create a simplified frame data structure for selection
                gait_data[person_id] = {
                    frame: {"timestamp": frame * 0.1} 
                    for frame in range(total_frames)
                }
            
            print(f"  People detected: {list(gait_data.keys())}")
            print(f"  Frame counts: {[(pid, len(frames)) for pid, frames in gait_data.items()]}")
            
            # Apply person selection
            if len(gait_data) > 1:
                print("  Multiple people detected - applying person selection...")
                
                # Create person selector
                selector = create_person_selector("longest_track", min_frames=10)
                
                # Select main person
                main_person_id = selector.select_main_person(gait_data)
                
                if main_person_id is not None:
                    print(f"  Selected main person: {main_person_id}")
                    print(f"  Main person frames: {len(gait_data[main_person_id])}")
                    
                    # Show what would be filtered out
                    other_people = [pid for pid in gait_data.keys() if pid != main_person_id]
                    if other_people:
                        print(f"  Would filter out: {other_people}")
                else:
                    print("  No valid person found for selection")
            else:
                print("  Single person detected - no selection needed")
            
            print()
            
        except Exception as e:
            print(f"  Error analyzing {gait_file}: {e}")
            print()


def demonstrate_person_selection() -> None:
    """Demonstrate person selection with example data."""
    print("=== Person Selection Demonstration ===\n")
    
    # Create example gait data with multiple people
    example_gait_data = {
        1: {frame: {"timestamp": frame * 0.1} for frame in range(8)},   # 8 frames
        2: {frame: {"timestamp": frame * 0.1} for frame in range(12)},  # 12 frames (longest)
        3: {frame: {"timestamp": frame * 0.1} for frame in range(6)}    # 6 frames
    }
    
    print("Example gait data:")
    for person_id, frames in example_gait_data.items():
        print(f"  Person {person_id}: {len(frames)} frames")
    
    print("\nApplying longest track selection...")
    
    # Create selector
    selector = create_person_selector("longest_track", min_frames=5)
    
    # Select main person
    main_person_id = selector.select_main_person(example_gait_data)
    
    if main_person_id is not None:
        print(f"Selected person {main_person_id} as main person")
        
        # Filter data
        filtered_data = filter_gait_data_by_person(example_gait_data, main_person_id)
        print(f"Filtered data contains {len(filtered_data)} person(s)")
        
        # Show what was filtered out
        original_count = len(example_gait_data)
        filtered_count = len(filtered_data)
        print(f"Filtered out {original_count - filtered_count} person(s)")
    else:
        print("No valid person selected")
    
    print()


def main():
    """Main entry point."""
    data_dir = Path("data/toronto-gait-outputs")
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Please ensure the Toronto gait outputs are available.")
        return 1
    
    # Demonstrate person selection
    demonstrate_person_selection()
    
    # Analyze existing outputs
    analyze_toronto_outputs(data_dir)
    
    print("=== Summary ===")
    print("The 'Longest Track' heuristic has been implemented and can:")
    print("1. Select the person with the most frames as the main subject")
    print("2. Filter out other people to focus analysis on the main person")
    print("3. Improve gait analysis consistency by avoiding mixed data")
    print("\nTo use this in practice:")
    print("  python cli/detect.py --video <video> --toronto-gait-format --person-selection longest_track")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 
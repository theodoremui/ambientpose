# Person Selection Implementation for Toronto Gait Analysis

## Overview

This document describes the implementation of the "Longest Track" heuristic for person selection in gait analysis, as described in the `PERSON_DETECTION.md` document. The implementation addresses the issue of multiple people being detected in gait videos and provides a robust way to select the most salient person for analysis.

## Implementation Details

### 1. Core Components

#### Person Selection Module (`cli/person_selection.py`)

The implementation provides a modular, extensible framework for person selection:

- **`PersonSelectionHeuristic`**: Abstract base class for person selection heuristics
- **`LongestTrackHeuristic`**: Selects the person with the most frames (primary implementation)
- **`LargestBoundingBoxHeuristic`**: Selects the person with the largest average bounding box
- **Factory function**: `create_person_selector()` for easy heuristic selection
- **Utility function**: `filter_gait_data_by_person()` for data filtering

#### Key Features

1. **Configurable minimum frame threshold**: Ensures only people with sufficient data are considered
2. **Robust error handling**: Graceful fallback when selection fails
3. **Comprehensive logging**: Detailed information about selection decisions
4. **Extensible design**: Easy to add new selection heuristics

### 2. Integration with Detect.py

#### Configuration Options

Added new CLI arguments to `detect.py`:

```bash
--person-selection longest_track    # Selection heuristic (default: longest_track)
--min-frames-for-selection 10      # Minimum frames required (default: 10)
```

#### Modified Output Methods

1. **`save_toronto_gait_format()`**: Now applies person selection before generating JSON
2. **`save_toronto_csv_format()`**: Now applies person selection before generating CSV

### 3. Selection Logic

#### Longest Track Heuristic

```python
# Select the person with the most frames
main_person_id = max(valid_people.keys(), key=lambda pid: len(valid_people[pid]))
```

**Rationale**: The main subject (e.g., the older adult walking) is typically tracked for the most frames because they are the primary focus of the video.

#### Filtering Process

1. **Build gait data**: Organize poses by person_id and frame
2. **Apply selection**: Use heuristic to select main person
3. **Filter data**: Keep only the selected person's data
4. **Generate output**: Create JSON/CSV with filtered data

## Testing

### Comprehensive Test Suite (`tests/test_person_selection.py`)

The implementation includes 20 comprehensive tests covering:

- **Longest Track Heuristic**: Single person, multiple people, insufficient frames, tie-breaking
- **Largest Bounding Box Heuristic**: Valid bbox data, missing data, mixed data, invalid format
- **Factory Function**: Valid heuristics, invalid heuristics, default behavior
- **Data Filtering**: Successful filtering, person not found, empty input
- **Integration**: Complete workflow testing

### Test Coverage

- ✅ All heuristics work correctly
- ✅ Error handling for edge cases
- ✅ Factory function validation
- ✅ Data filtering functionality
- ✅ Integration with main workflow

## Results Analysis

### Toronto Gait Outputs Analysis

The demonstration script analyzed 14 gait.json files from the Toronto dataset:

| File | People Detected | Main Person Selected | Frames | Others Filtered |
|------|-----------------|---------------------|--------|-----------------|
| 01   | 1               | N/A (single)        | 1660   | 0               |
| 02   | 4               | Person 2            | 1647   | [1, 4, 5]       |
| 03   | 8               | Person 2            | 1100   | [0, 1, 3, 4, 5, 6, 9] |
| 04   | 9               | Person 1            | 742    | [2, 4, 5, 7, 8, 9, 10, 11] |
| 05   | 6               | Person 0            | 1757   | [1, 4, 7, 13, 14] |
| 06   | 7               | Person 5            | 788    | [0, 1, 2, 3, 4, 6] |
| 07   | 8               | Person 1            | 545    | [0, 2, 3, 4, 5, 6, 8] |
| 08   | 3               | Person 1            | 1068   | [0, 2]          |
| 09   | 6               | Person 2            | 1494   | [0, 1, 3, 6, 10] |
| 10   | 1               | N/A (single)        | 1397   | 0               |
| 11   | 3               | Person 1            | 1725   | [0, 8]          |
| 12   | 6               | Person 0            | 1492   | [2, 7, 8, 15, 16] |
| 13   | 6               | Person 0            | 1871   | [1, 2, 6, 7, 10] |
| 14   | 10              | Person 0            | 1998   | [1, 2, 5, 7, 8, 9, 10, 11, 12] |

### Key Findings

1. **Multiple people detected**: 12 out of 14 files had multiple people
2. **Significant filtering**: Up to 9 people filtered out in some cases
3. **Clear main subjects**: The heuristic consistently identified the person with the most frames
4. **Improved consistency**: Single person focus for gait analysis

## Usage

### Command Line Usage

```bash
# Basic usage with longest track selection
python cli/detect.py --video input.mp4 --toronto-gait-format --person-selection longest_track

# Custom minimum frames
python cli/detect.py --video input.mp4 --toronto-gait-format --person-selection longest_track --min-frames-for-selection 15

# Disable person selection (process all people)
python cli/detect.py --video input.mp4 --toronto-gait-format --person-selection none
```

### Programmatic Usage

```python
from cli.person_selection import create_person_selector, filter_gait_data_by_person

# Create selector
selector = create_person_selector("longest_track", min_frames=10)

# Select main person
main_person_id = selector.select_main_person(gait_data)

# Filter data
if main_person_id is not None:
    filtered_data = filter_gait_data_by_person(gait_data, main_person_id)
```

## Benefits

### 1. Improved Gait Analysis Quality

- **Consistent tracking**: Single person throughout the sequence
- **Reduced noise**: Eliminates data from bystanders or fragmented tracks
- **Better metrics**: More accurate stride length, step frequency calculations

### 2. Enhanced User Experience

- **Automatic selection**: No manual intervention required
- **Configurable**: Multiple heuristics available
- **Robust**: Graceful fallback when selection fails

### 3. Better Data Management

- **Cleaner outputs**: Focused on main subject
- **Reduced file sizes**: Less redundant data
- **Clearer analysis**: Single person per output file

## Future Enhancements

### Potential Additional Heuristics

1. **Centrality-based**: Select person closest to frame center
2. **Motion-based**: Select person with most consistent movement
3. **Confidence-based**: Select person with highest average confidence
4. **Manual selection**: Allow user to specify person_id

### Advanced Features

1. **Multi-person analysis**: Option to analyze multiple people separately
2. **Person merging**: Combine fragmented tracks of the same person
3. **Quality metrics**: Report selection confidence and reasoning
4. **Visualization**: Show selection process in overlay videos

## Conclusion

The "Longest Track" heuristic implementation successfully addresses the person selection problem in gait analysis. The implementation is:

- ✅ **Robust**: Handles edge cases and errors gracefully
- ✅ **Configurable**: Multiple heuristics and parameters
- ✅ **Well-tested**: Comprehensive test coverage
- ✅ **Integrated**: Seamlessly works with existing pipeline
- ✅ **Effective**: Significantly improves output quality

The implementation provides a solid foundation for reliable gait analysis by ensuring that only the most salient person is analyzed, leading to more accurate and consistent results. 
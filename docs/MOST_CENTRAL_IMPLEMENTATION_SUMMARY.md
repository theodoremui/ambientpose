# Most Central Heuristic Implementation Summary

## Overview

This document summarizes the implementation of the "most central" person selection heuristic for the Toronto gait analysis system. The implementation addresses the user's requirement to identify the dominant person in video frames by selecting the person whose bounding box center is closest to the frame center on average.

## Implementation Components

### 1. Core Heuristic Class (`cli/person_selection.py`)

**New Class**: `MostCentralHeuristic`
- **Purpose**: Selects the person whose bounding box center is closest to the frame center
- **Algorithm**: 
  1. Calculate bounding box center for each person in each frame
  2. Estimate frame center from maximum bounding box coordinates
  3. Compute Euclidean distance from person center to frame center
  4. Average distances across all frames for each person
  5. Select person with smallest average distance

**Key Features**:
- Configurable minimum frame threshold (default: 10)
- Comprehensive error handling for missing bounding box data
- Detailed logging of selection decisions
- Graceful fallback when no valid candidates found

### 2. CLI Integration (`cli/detect.py`)

**New CLI Option**: `--person-selection most_central`
- Added to argument parser choices
- Integrated with both JSON and CSV output methods
- Bounding box data added to gait_data structure for heuristic access

**Modified Methods**:
- `save_toronto_gait_format()`: Now includes bounding box information in gait_data
- `save_toronto_csv_format()`: Now includes bounding box information for person selection

### 3. Factory Function Update

**Updated**: `create_person_selector()` function
- Added support for `"most_central"` heuristic
- Updated error messages to include new option
- Maintains backward compatibility with existing heuristics

### 4. Test Suite (`tests/test_person_selection.py`)

**New Tests**:
- `TestMostCentralHeuristic`: Comprehensive test class for the new heuristic
- Integration tests for complete workflow
- Error handling tests for various edge cases
- All tests pass successfully

**Test Coverage**:
- Selection with valid bounding box data
- Handling of missing bounding box data
- Insufficient frame scenarios
- Integration with factory function
- Complete workflow testing

### 5. Documentation

**New Documentation**: `docs/MOST_CENTRAL_HEURISTIC.md`
- Complete implementation guide
- Usage examples for CLI and programmatic access
- Comparison with other heuristics
- Advantages and limitations
- Performance considerations
- Future enhancement suggestions

**Updated Documentation**: `docs/PERSON_DETECTION.md`
- Added most central heuristic to recommendations
- Updated implementation status section
- Included code examples

### 6. Script for Existing Outputs

**New Script**: `scripts/apply_most_central_to_existing_outputs.py`
- Processes existing Toronto gait outputs
- Applies most central selection to JSON files
- Updates summary statistics to reflect selected person
- Comprehensive error handling and logging

## Technical Details

### Bounding Box Integration

The implementation addresses the challenge of accessing bounding box data by:

1. **Data Structure Enhancement**: Modified `save_toronto_gait_format()` to include bounding box information in gait_data
2. **Original Pose Access**: Maps converted poses back to original poses to extract bounding boxes
3. **Fallback Strategy**: Uses longest track heuristic when bounding box data is unavailable

### Algorithm Implementation

```python
# Calculate bounding box center
bbox_center_x = (bbox[0] + bbox[2]) / 2
bbox_center_y = (bbox[1] + bbox[3]) / 2

# Estimate frame center from maximum coordinates
frame_center_x = max_bbox_x / 2
frame_center_y = max_bbox_y / 2

# Calculate Euclidean distance
distance = math.sqrt((bbox_center_x - frame_center_x)**2 + 
                    (bbox_center_y - frame_center_y)**2)
```

### Error Handling

The implementation includes robust error handling for:
- Missing bounding box data
- Insufficient frame counts
- Empty gait data
- Invalid bounding box formats
- No valid candidates

## Usage Examples

### Command Line Usage

```bash
# Basic usage with most central heuristic
python cli/detect.py --video input.mp4 --toronto-gait-format --person-selection most_central

# With custom minimum frames
python cli/detect.py --video input.mp4 --toronto-gait-format --person-selection most_central --min-frames-for-selection 15

# Apply to existing outputs
python scripts/apply_most_central_to_existing_outputs.py
```

### Programmatic Usage

```python
from cli.person_selection import create_person_selector

# Create selector
selector = create_person_selector(heuristic="most_central", min_frames=10)

# Select main person
main_person_id = selector.select_main_person(gait_data)

# Filter data
from cli.person_selection import filter_gait_data_by_person
filtered_data = filter_gait_data_by_person(gait_data, main_person_id)
```

## Integration with Toronto Gait Format

### JSON Output
- Only selected main person included in `gait_analysis` array
- Summary statistics updated to reflect single person
- Metadata includes selection heuristic information

### CSV Output
- Only selected main person's data written to CSV
- Maintains exact Toronto format compatibility
- Person selection applied per frame

## Performance Characteristics

- **Time Complexity**: O(n × f) where n is number of people and f is number of frames
- **Memory Usage**: Minimal additional memory required
- **Accuracy**: Depends on bounding box detection quality and camera framing

## Comparison with Other Heuristics

| Aspect | Longest Track | Most Central | Largest Bbox |
|--------|---------------|--------------|--------------|
| **Selection Criteria** | Most frames | Closest to center | Largest area |
| **Pros** | Simple, robust | Intuitive, good for main subjects | Good for close-ups |
| **Cons** | May select bystanders | Requires good framing | Sensitive to distance |
| **Best For** | General use | Standard video framing | Close-up subjects |

## Validation

### Test Results
- All 18 tests pass successfully
- Comprehensive coverage of edge cases
- Integration tests verify complete workflow
- Error handling tests confirm robustness

### Manual Verification
- Heuristic class imports correctly
- Factory function works as expected
- CLI integration functional
- Documentation complete and accurate

## Future Enhancements

Potential improvements identified:
1. **Weighted Averaging**: Weight recent frames more heavily
2. **Confidence Integration**: Consider detection confidence in selection
3. **Temporal Consistency**: Ensure selected person remains consistent over time
4. **Multi-camera Support**: Handle multiple camera angles
5. **Adaptive Thresholds**: Adjust based on video characteristics

## Conclusion

The "most central" heuristic has been successfully implemented and integrated into the Toronto gait analysis system. The implementation provides:

- **Robust person selection** based on frame centrality
- **Full integration** with existing CLI and output formats
- **Comprehensive testing** and error handling
- **Complete documentation** for users and developers
- **Backward compatibility** with existing heuristics

The heuristic addresses the user's requirement to identify the dominant person in Toronto dataset videos and should provide more accurate results than the previous "longest track" approach for videos where the main subject is positioned centrally in the frame. 
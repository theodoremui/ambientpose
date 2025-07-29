# Most Central Person Selection Heuristic

## Overview

The "Most Central" heuristic is a person selection strategy for gait analysis that identifies the most salient person in a video by selecting the individual whose bounding box center is closest to the center of the frame on average.

## Rationale

In gait analysis videos, the main subject (typically the older adult being analyzed) is usually positioned more centrally in the frame compared to bystanders, caregivers, or other people in the scene. This heuristic leverages this observation to automatically identify the dominant person for analysis.

## Implementation

### Algorithm

1. **Bounding Box Center Calculation**: For each detected person in each frame, calculate the center of their bounding box:
   ```
   bbox_center_x = (bbox[0] + bbox[2]) / 2
   bbox_center_y = (bbox[1] + bbox[3]) / 2
   ```

2. **Frame Center Estimation**: Estimate the center of the frame based on the maximum bounding box coordinates:
   ```
   frame_center_x = max_bbox_x / 2
   frame_center_y = max_bbox_y / 2
   ```

3. **Distance Calculation**: Compute the Euclidean distance from each person's bounding box center to the frame center:
   ```
   distance = sqrt((bbox_center_x - frame_center_x)² + (bbox_center_y - frame_center_y)²)
   ```

4. **Average Distance**: Calculate the average distance for each person across all frames where they are detected.

5. **Selection**: Select the person with the smallest average distance to the frame center.

### Code Implementation

```python
class MostCentralHeuristic(PersonSelectionHeuristic):
    """
    Select the person whose bounding box center is closest to the frame center.
    
    This heuristic assumes that the main subject (e.g., the older adult walking)
    will be positioned more centrally in the frame compared to bystanders or
    other people in the scene.
    """
    
    def select_main_person(self, gait_data: Dict[int, Dict]) -> Optional[int]:
        # Calculate average distance to frame center for each person
        person_distances = {}
        for person_id, person_data in valid_people.items():
            distances = []
            for frame_data in person_data.values():
                if 'bbox' in frame_data:
                    bbox = frame_data['bbox']
                    if len(bbox) >= 4:
                        # Calculate bounding box center
                        bbox_center_x = (bbox[0] + bbox[2]) / 2
                        bbox_center_y = (bbox[1] + bbox[3]) / 2
                        
                        # Calculate frame center
                        frame_center_x = frame_width / 2
                        frame_center_y = frame_height / 2
                        
                        # Calculate Euclidean distance
                        distance = math.sqrt((bbox_center_x - frame_center_x)**2 + 
                                          (bbox_center_y - frame_center_y)**2)
                        distances.append(distance)
            
            if distances:
                person_distances[person_id] = sum(distances) / len(distances)
        
        # Select person with smallest average distance
        main_person_id = min(person_distances.keys(), key=lambda pid: person_distances[pid])
        return main_person_id
```

## Usage

### Command Line Interface

To use the most central heuristic with the CLI:

```bash
# Basic usage
python cli/detect.py --video input.mp4 --toronto-gait-format --person-selection most_central

# With custom minimum frames threshold
python cli/detect.py --video input.mp4 --toronto-gait-format --person-selection most_central --min-frames-for-selection 15

# Disable person selection (process all people)
python cli/detect.py --video input.mp4 --toronto-gait-format --person-selection none
```

### Programmatic Usage

```python
from cli.person_selection import create_person_selector

# Create a most central selector
selector = create_person_selector(heuristic="most_central", min_frames=10)

# Select main person from gait data
main_person_id = selector.select_main_person(gait_data)

# Filter data to include only the main person
from cli.person_selection import filter_gait_data_by_person
filtered_data = filter_gait_data_by_person(gait_data, main_person_id)
```

## Comparison with Other Heuristics

| Heuristic | Selection Criteria | Pros | Cons |
|-----------|-------------------|------|------|
| **Longest Track** | Most frames tracked | Simple, robust | May select bystanders |
| **Largest Bounding Box** | Largest average bbox area | Good for close-up subjects | Sensitive to camera distance |
| **Most Central** | Closest to frame center | Intuitive, good for main subjects | Requires good camera framing |

## Advantages

1. **Intuitive**: The heuristic aligns with human perception of "main subject"
2. **Robust**: Works well with standard video framing where the main subject is centered
3. **Automatic**: No manual intervention required
4. **Configurable**: Minimum frame threshold can be adjusted

## Limitations

1. **Frame Dependency**: Requires the main subject to be reasonably centered
2. **Camera Setup**: May not work well with unusual camera angles or framing
3. **Bounding Box Quality**: Depends on accurate bounding box detection
4. **Multiple Central Subjects**: May struggle when multiple people are similarly positioned

## Testing

The heuristic includes comprehensive tests covering:

- Selection with valid bounding box data
- Handling of missing bounding box data
- Insufficient frame scenarios
- Integration with the complete workflow

Run tests with:
```bash
python -m pytest tests/test_person_selection.py -v
```

## Integration with Toronto Gait Format

The most central heuristic is fully integrated with the Toronto gait analysis format:

1. **JSON Output**: Only the selected main person is included in `gait_analysis`
2. **CSV Output**: Only the selected main person's data is written to the CSV
3. **Summary Statistics**: Updated to reflect only the selected person

## Applying to Existing Outputs

To apply the most central heuristic to existing Toronto gait outputs:

```bash
python scripts/apply_most_central_to_existing_outputs.py
```

This script processes all outputs in `data/toronto-gait-outputs/` and updates them to include only the most central person.

## Configuration Options

- **`--person-selection most_central`**: Enable most central heuristic
- **`--min-frames-for-selection 10`**: Minimum frames required (default: 10)
- **`--person-selection none`**: Disable person selection (process all people)

## Error Handling

The heuristic gracefully handles various error conditions:

- **No bounding box data**: Returns `None` with warning
- **Insufficient frames**: Filters out people below threshold
- **No valid candidates**: Returns `None` with warning
- **Empty data**: Returns `None` with warning

## Performance Considerations

- **Time Complexity**: O(n × f) where n is number of people and f is number of frames
- **Memory Usage**: Minimal additional memory required
- **Accuracy**: Depends on bounding box detection quality

## Future Enhancements

Potential improvements to consider:

1. **Weighted Averaging**: Weight recent frames more heavily
2. **Confidence Integration**: Consider detection confidence in selection
3. **Temporal Consistency**: Ensure selected person remains consistent over time
4. **Multi-camera Support**: Handle multiple camera angles
5. **Adaptive Thresholds**: Adjust based on video characteristics 
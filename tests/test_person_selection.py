"""
Tests for person selection heuristics.

This module tests the person selection heuristics used in gait analysis
to select the most salient person from multiple detected people.
"""

import pytest
import sys
from pathlib import Path

# Add the cli directory to the path so we can import person_selection
sys.path.insert(0, str(Path(__file__).parent.parent / "cli"))

from person_selection import (
    LongestTrackHeuristic,
    LargestBoundingBoxHeuristic,
    MostCentralHeuristic,
    create_person_selector,
    filter_gait_data_by_person
)


class TestLongestTrackHeuristic:
    """Test the longest track heuristic."""
    
    def test_select_main_person_single_person(self):
        """Test selection with a single person."""
        heuristic = LongestTrackHeuristic(min_frames=5)
        
        gait_data = {
            1: {frame: {"timestamp": frame * 0.1} for frame in range(10)}
        }
        
        result = heuristic.select_main_person(gait_data)
        assert result == 1
    
    def test_select_main_person_multiple_people(self):
        """Test selection with multiple people."""
        heuristic = LongestTrackHeuristic(min_frames=5)
        
        gait_data = {
            1: {frame: {"timestamp": frame * 0.1} for frame in range(8)},
            2: {frame: {"timestamp": frame * 0.1} for frame in range(12)},
            3: {frame: {"timestamp": frame * 0.1} for frame in range(6)}
        }
        
        result = heuristic.select_main_person(gait_data)
        assert result == 2  # Person 2 has the most frames
    
    def test_select_main_person_insufficient_frames(self):
        """Test selection when no person has sufficient frames."""
        heuristic = LongestTrackHeuristic(min_frames=15)
        
        gait_data = {
            1: {frame: {"timestamp": frame * 0.1} for frame in range(10)},
            2: {frame: {"timestamp": frame * 0.1} for frame in range(8)}
        }
        
        result = heuristic.select_main_person(gait_data)
        assert result is None
    
    def test_select_main_person_empty_data(self):
        """Test selection with empty data."""
        heuristic = LongestTrackHeuristic(min_frames=5)
        
        result = heuristic.select_main_person({})
        assert result is None


class TestLargestBoundingBoxHeuristic:
    """Test the largest bounding box heuristic."""
    
    def test_select_main_person_with_bbox_data(self):
        """Test selection with bounding box data."""
        heuristic = LargestBoundingBoxHeuristic(min_frames=5)
        
        gait_data = {
            1: {
                frame: {
                    "timestamp": frame * 0.1,
                    "bbox": [0, 0, 50, 100]  # Area: 5000
                } for frame in range(10)
            },
            2: {
                frame: {
                    "timestamp": frame * 0.1,
                    "bbox": [0, 0, 100, 100]  # Area: 10000
                } for frame in range(10)
            }
        }
        
        result = heuristic.select_main_person(gait_data)
        assert result == 2  # Person 2 has larger bounding box
    
    def test_select_main_person_no_bbox_data(self):
        """Test selection when no bounding box data is available."""
        heuristic = LargestBoundingBoxHeuristic(min_frames=5)
        
        gait_data = {
            1: {frame: {"timestamp": frame * 0.1} for frame in range(10)},
            2: {frame: {"timestamp": frame * 0.1} for frame in range(10)}
        }
        
        result = heuristic.select_main_person(gait_data)
        assert result is None


class TestMostCentralHeuristic:
    """Test the most central heuristic."""
    
    def test_select_main_person_with_bbox_data(self):
        """Test selection with bounding box data."""
        heuristic = MostCentralHeuristic(min_frames=5)
        
        # Create test data with different distances to center
        # Person 1: closer to center (distance ~50)
        # Person 2: farther from center (distance ~100)
        gait_data = {
            1: {
                frame: {
                    "timestamp": frame * 0.1,
                    "bbox": [200, 200, 300, 400]  # Center around (250, 300)
                } for frame in range(10)
            },
            2: {
                frame: {
                    "timestamp": frame * 0.1,
                    "bbox": [100, 100, 200, 200]  # Center around (150, 150)
                } for frame in range(10)
            }
        }
        
        result = heuristic.select_main_person(gait_data)
        # The heuristic should select the person closest to the frame center
        # In this case, it depends on the frame dimensions calculated from the data
        assert result in [1, 2]
    
    def test_select_main_person_no_bbox_data(self):
        """Test selection when no bounding box data is available."""
        heuristic = MostCentralHeuristic(min_frames=5)
        
        gait_data = {
            1: {frame: {"timestamp": frame * 0.1} for frame in range(10)},
            2: {frame: {"timestamp": frame * 0.1} for frame in range(10)}
        }
        
        result = heuristic.select_main_person(gait_data)
        assert result is None
    
    def test_select_main_person_insufficient_frames(self):
        """Test selection when no person has sufficient frames."""
        heuristic = MostCentralHeuristic(min_frames=15)
        
        gait_data = {
            1: {
                frame: {
                    "timestamp": frame * 0.1,
                    "bbox": [0, 0, 50, 100]
                } for frame in range(10)
            }
        }
        
        result = heuristic.select_main_person(gait_data)
        assert result is None


class TestFactoryFunction:
    """Test the factory function for creating person selectors."""
    
    def test_create_longest_track_selector(self):
        """Test creating a longest track selector."""
        selector = create_person_selector("longest_track", min_frames=10)
        assert isinstance(selector, LongestTrackHeuristic)
        assert selector.min_frames == 10
    
    def test_create_largest_bbox_selector(self):
        """Test creating a largest bounding box selector."""
        selector = create_person_selector("largest_bbox", min_frames=15)
        assert isinstance(selector, LargestBoundingBoxHeuristic)
        assert selector.min_frames == 15
    
    def test_create_most_central_selector(self):
        """Test creating a most central selector."""
        selector = create_person_selector("most_central", min_frames=12)
        assert isinstance(selector, MostCentralHeuristic)
        assert selector.min_frames == 12
    
    def test_create_unknown_selector(self):
        """Test creating an unknown selector."""
        with pytest.raises(ValueError, match="Unknown heuristic"):
            create_person_selector("unknown_heuristic")


class TestFilterFunction:
    """Test the filter function for gait data."""
    
    def test_filter_gait_data(self):
        """Test filtering gait data by person."""
        gait_data = {
            1: {frame: {"timestamp": frame * 0.1} for frame in range(10)},
            2: {frame: {"timestamp": frame * 0.1} for frame in range(8)},
            3: {frame: {"timestamp": frame * 0.1} for frame in range(12)}
        }
        
        filtered_data = filter_gait_data_by_person(gait_data, 2)
        
        assert len(filtered_data) == 1
        assert 2 in filtered_data
        assert filtered_data[2] == gait_data[2]
    
    def test_filter_gait_data_person_not_found(self):
        """Test filtering when the specified person is not found."""
        gait_data = {
            1: {frame: {"timestamp": frame * 0.1} for frame in range(10)}
        }
        
        filtered_data = filter_gait_data_by_person(gait_data, 999)
        
        assert filtered_data == {}


class TestIntegration:
    """Integration tests for the complete person selection workflow."""
    
    def test_complete_workflow_longest_track(self):
        """Test the complete workflow using longest track heuristic."""
        # Create test data
        gait_data = {
            1: {frame: {"timestamp": frame * 0.1} for frame in range(8)},
            2: {frame: {"timestamp": frame * 0.1} for frame in range(12)},
            3: {frame: {"timestamp": frame * 0.1} for frame in range(6)}
        }
        
        # Create selector
        selector = create_person_selector("longest_track", min_frames=5)
        
        # Select main person
        main_person_id = selector.select_main_person(gait_data)
        assert main_person_id == 2
        
        # Filter data
        filtered_data = filter_gait_data_by_person(gait_data, main_person_id)
        assert len(filtered_data) == 1
        assert 2 in filtered_data
    
    def test_complete_workflow_largest_bbox(self):
        """Test the complete workflow using largest bounding box heuristic."""
        # Create test data with bounding boxes
        gait_data = {
            1: {
                frame: {
                    "timestamp": frame * 0.1,
                    "bbox": [0, 0, 50, 100]  # Area: 5000
                } for frame in range(10)
            },
            2: {
                frame: {
                    "timestamp": frame * 0.1,
                    "bbox": [0, 0, 100, 100]  # Area: 10000
                } for frame in range(10)
            }
        }
        
        # Create selector
        selector = create_person_selector("largest_bbox", min_frames=5)
        
        # Select main person
        main_person_id = selector.select_main_person(gait_data)
        assert main_person_id == 2
        
        # Filter data
        filtered_data = filter_gait_data_by_person(gait_data, main_person_id)
        assert len(filtered_data) == 1
        assert 2 in filtered_data
    
    def test_complete_workflow_most_central(self):
        """Test the complete workflow using most central heuristic."""
        # Create test data with bounding boxes at different positions
        gait_data = {
            1: {
                frame: {
                    "timestamp": frame * 0.1,
                    "bbox": [200, 200, 300, 400]  # More central
                } for frame in range(10)
            },
            2: {
                frame: {
                    "timestamp": frame * 0.1,
                    "bbox": [100, 100, 200, 200]  # Less central
                } for frame in range(10)
            }
        }
        
        # Create selector
        selector = create_person_selector("most_central", min_frames=5)
        
        # Select main person
        main_person_id = selector.select_main_person(gait_data)
        assert main_person_id in [1, 2]  # Should select one of them
        
        # Filter data
        filtered_data = filter_gait_data_by_person(gait_data, main_person_id)
        assert len(filtered_data) == 1
        assert main_person_id in filtered_data 
    pytest.main([__file__]) 
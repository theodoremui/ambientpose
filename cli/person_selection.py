"""
Person selection utilities for gait analysis.

This module provides heuristics for selecting the most salient person
from multiple detected people in gait analysis videos.
"""

from typing import Dict, List, Any, Optional
import logging
import math

logger = logging.getLogger(__name__)


from abc import ABC, abstractmethod

class PersonSelectionHeuristic(ABC):
    """Base class for person selection heuristics."""
    
    @abstractmethod
    def select_main_person(self, gait_data: Dict[int, Dict]) -> Optional[int]:
        """
        Select the main person from gait data.
        
        Args:
            gait_data: Dictionary mapping person_id to their frame data
            
        Returns:
            The person_id of the selected main person, or None if no valid selection
        """
        pass


class LongestTrackHeuristic(PersonSelectionHeuristic):
    """
    Select the person with the longest track (most frames).
    
    This heuristic assumes that the main subject (e.g., the older adult walking)
    will be tracked for the most frames, as they are the primary focus of the video.
    """
    
    def __init__(self, min_frames: int = 10):
        """
        Initialize the longest track heuristic.
        
        Args:
            min_frames: Minimum number of frames required for a person to be considered
        """
        self.min_frames = min_frames
    
    def select_main_person(self, gait_data: Dict[int, Dict]) -> Optional[int]:
        """
        Select the person with the most frames.
        
        Args:
            gait_data: Dictionary mapping person_id to their frame data
            
        Returns:
            The person_id with the most frames, or None if no valid candidates
        """
        if not gait_data:
            logger.warning("No gait data provided for person selection")
            return None
        
        # Filter out people with insufficient frames
        valid_people = {
            person_id: person_data 
            for person_id, person_data in gait_data.items() 
            if len(person_data) >= self.min_frames
        }
        
        if not valid_people:
            logger.warning(f"No people with at least {self.min_frames} frames found")
            return None
        
        # Select the person with the most frames
        main_person_id = max(valid_people.keys(), key=lambda pid: len(valid_people[pid]))
        frame_count = len(valid_people[main_person_id])
        
        logger.info(f"Selected person {main_person_id} with {frame_count} frames (longest track)")
        
        # Log information about other candidates
        if len(valid_people) > 1:
            other_people = [(pid, len(data)) for pid, data in valid_people.items() if pid != main_person_id]
            logger.info(f"Other candidates: {other_people}")
        
        return main_person_id


class LargestBoundingBoxHeuristic(PersonSelectionHeuristic):
    """
    Select the person with the largest average bounding box.
    
    This heuristic assumes that the main subject will be closer to the camera
    and thus have a larger bounding box.
    """
    
    def __init__(self, min_frames: int = 10):
        """
        Initialize the largest bounding box heuristic.
        
        Args:
            min_frames: Minimum number of frames required for a person to be considered
        """
        self.min_frames = min_frames
    
    def select_main_person(self, gait_data: Dict[int, Dict]) -> Optional[int]:
        """
        Select the person with the largest average bounding box.
        
        Args:
            gait_data: Dictionary mapping person_id to their frame data
            
        Returns:
            The person_id with the largest average bounding box, or None if no valid candidates
        """
        if not gait_data:
            logger.warning("No gait data provided for person selection")
            return None
        
        # Filter out people with insufficient frames
        valid_people = {
            person_id: person_data 
            for person_id, person_data in gait_data.items() 
            if len(person_data) >= self.min_frames
        }
        
        if not valid_people:
            logger.warning(f"No people with at least {self.min_frames} frames found")
            return None
        
        # Calculate average bounding box area for each person
        person_areas = {}
        for person_id, person_data in valid_people.items():
            areas = []
            for frame_data in person_data.values():
                # Extract bounding box from frame data if available
                # This is a simplified implementation - actual bbox data structure may vary
                if 'bbox' in frame_data:
                    bbox = frame_data['bbox']
                    if len(bbox) >= 4:
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                        areas.append(width * height)
            
            if areas:
                person_areas[person_id] = sum(areas) / len(areas)
            else:
                # Don't include people with no valid bbox data
                continue
        
        if not person_areas:
            logger.warning("No valid bounding box data found")
            return None
        
        # Select the person with the largest average bounding box
        main_person_id = max(person_areas.keys(), key=lambda pid: person_areas[pid])
        avg_area = person_areas[main_person_id]
        
        logger.info(f"Selected person {main_person_id} with average bbox area {avg_area:.2f}")
        
        return main_person_id


class MostCentralHeuristic(PersonSelectionHeuristic):
    """
    Select the person whose bounding box center is closest to the frame center.
    
    This heuristic assumes that the main subject (e.g., the older adult walking)
    will be positioned more centrally in the frame compared to bystanders or
    other people in the scene.
    """
    
    def __init__(self, min_frames: int = 10):
        """
        Initialize the most central heuristic.
        
        Args:
            min_frames: Minimum number of frames required for a person to be considered
        """
        self.min_frames = min_frames
    
    def select_main_person(self, gait_data: Dict[int, Dict]) -> Optional[int]:
        """
        Select the person whose bounding box center is closest to the frame center on average.
        
        Args:
            gait_data: Dictionary mapping person_id to their frame data
            
        Returns:
            The person_id with the most central average position, or None if no valid candidates
        """
        if not gait_data:
            logger.warning("No gait data provided for person selection")
            return None
        
        # Filter out people with insufficient frames
        valid_people = {
            person_id: person_data 
            for person_id, person_data in gait_data.items() 
            if len(person_data) >= self.min_frames
        }
        
        if not valid_people:
            logger.warning(f"No people with at least {self.min_frames} frames found")
            return None
        
        # Calculate average distance to frame center for each person
        person_distances = {}
        for person_id, person_data in valid_people.items():
            distances = []
            for frame_data in person_data.values():
                # Extract bounding box from frame data if available
                if 'bbox' in frame_data:
                    bbox = frame_data['bbox']
                    if len(bbox) >= 4:
                        # Calculate bounding box center
                        bbox_center_x = (bbox[0] + bbox[2]) / 2
                        bbox_center_y = (bbox[1] + bbox[3]) / 2
                        
                        # Calculate frame center (estimate from bounding box coordinates)
                        # We'll use the maximum coordinates as frame dimensions
                        frame_width = max(bbox[2] for frame_data in person_data.values() 
                                       if 'bbox' in frame_data and len(frame_data['bbox']) >= 4)
                        frame_height = max(bbox[3] for frame_data in person_data.values() 
                                        if 'bbox' in frame_data and len(frame_data['bbox']) >= 4)
                        
                        frame_center_x = frame_width / 2
                        frame_center_y = frame_height / 2
                        
                        # Calculate Euclidean distance to frame center
                        distance = math.sqrt((bbox_center_x - frame_center_x)**2 + 
                                          (bbox_center_y - frame_center_y)**2)
                        distances.append(distance)
            
            if distances:
                person_distances[person_id] = sum(distances) / len(distances)
            else:
                # Don't include people with no valid bbox data
                continue
        
        if not person_distances:
            logger.warning("No valid bounding box data found")
            return None
        
        # Select the person with the smallest average distance to frame center
        main_person_id = min(person_distances.keys(), key=lambda pid: person_distances[pid])
        avg_distance = person_distances[main_person_id]
        
        logger.info(f"Selected person {main_person_id} with average distance to center {avg_distance:.2f}")
        
        # Log information about other candidates
        if len(person_distances) > 1:
            other_people = [(pid, dist) for pid, dist in person_distances.items() if pid != main_person_id]
            logger.info(f"Other candidates: {other_people}")
        
        return main_person_id


def create_person_selector(heuristic: str = "longest_track", **kwargs) -> PersonSelectionHeuristic:
    """
    Factory function to create a person selector based on the specified heuristic.
    
    Args:
        heuristic: The heuristic to use ("longest_track", "largest_bbox", or "most_central")
        **kwargs: Additional arguments for the heuristic
        
    Returns:
        A PersonSelectionHeuristic instance
        
    Raises:
        ValueError: If an unknown heuristic is specified
    """
    if heuristic == "longest_track":
        return LongestTrackHeuristic(**kwargs)
    elif heuristic == "largest_bbox":
        return LargestBoundingBoxHeuristic(**kwargs)
    elif heuristic == "most_central":
        return MostCentralHeuristic(**kwargs)
    else:
        raise ValueError(f"Unknown heuristic: {heuristic}. Available: longest_track, largest_bbox, most_central")


def filter_gait_data_by_person(gait_data: Dict[int, Dict], main_person_id: int) -> Dict[int, Dict]:
    """
    Filter gait data to include only the selected main person.
    
    Args:
        gait_data: Original gait data with all people
        main_person_id: The person_id to keep
        
    Returns:
        Filtered gait data containing only the main person
    """
    if main_person_id not in gait_data:
        logger.warning(f"Main person {main_person_id} not found in gait data")
        return {}
    
    return {main_person_id: gait_data[main_person_id]} 
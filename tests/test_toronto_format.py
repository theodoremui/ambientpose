#!/usr/bin/env python3
"""
Test Toronto format functionality for OpenPose output.
"""

import unittest
import tempfile
import json
import csv
from pathlib import Path
import sys
import os

# Add the cli directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cli'))

from detect import get_coco_joint_names


class TestTorontoFormat(unittest.TestCase):
    """Test Toronto format functionality."""
    
    def test_coco_joint_names(self):
        """Test that COCO joint names match Toronto format."""
        joint_names = get_coco_joint_names()
        
        # Should have exactly 17 joints (standard COCO format)
        self.assertEqual(len(joint_names), 17)
        
        # Check that joint names match Toronto format
        expected_joints = [
            "Nose", "LEye", "REye", "LEar", "REar",
            "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
            "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle"
        ]
        
        self.assertEqual(joint_names, expected_joints)
        
        # Verify no "joint_17" or other non-standard joints
        for joint_name in joint_names:
            self.assertNotIn("joint_", joint_name)
            self.assertNotIn("_17", joint_name)
    
    def test_toronto_csv_header_format(self):
        """Test that Toronto CSV header format is correct."""
        joint_names = get_coco_joint_names()
        
        # Create header parts
        header_parts = ["time"]
        for joint_name in joint_names:
            header_parts.extend([f"{joint_name}_x", f"{joint_name}_y", f"{joint_name}_conf"])
        
        header = ",".join(header_parts)
        
        # Should have 1 + 3*17 = 52 columns
        self.assertEqual(len(header_parts), 52)
        
        # Check that header starts with "time"
        self.assertTrue(header.startswith("time"))
        
        # Check that all joint names are present with correct suffixes
        for joint_name in joint_names:
            self.assertIn(f"{joint_name}_x", header)
            self.assertIn(f"{joint_name}_y", header)
            self.assertIn(f"{joint_name}_conf", header)


if __name__ == "__main__":
    unittest.main() 
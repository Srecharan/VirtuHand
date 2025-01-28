"""
Gesture Classifier Module
------------------------
This module handles real-time classification of hand gestures based on detected landmarks.
It uses geometric relationships between finger joints to determine specific hand poses.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto

class GestureType(Enum):
    """Supported gesture types."""
    NONE = auto()
    OPEN_PALM = auto()
    PINCH = auto()
    GRAB = auto()
    POINT = auto()
    #VICTORY = auto()

@dataclass
class GestureResult:
    """Contains the result of gesture classification."""
    gesture_type: GestureType
    confidence: float
    hand_side: str
    palm_position: Tuple[float, float, float]

class GestureClassifier:
    def __init__(self):
        self.gesture_history = []
        self.history_size = 5  # Number of frames to consider for smoothing
        
        # Adjusted thresholds with tolerance ranges
        self.finger_straight_min = 160  # Minimum angle for "straight" finger
        self.finger_bent_max = 90      # Maximum angle for "bent" finger
        self.pinch_threshold = 0.1     # Increased for better pinch detection
        
    def _get_finger_state(self, angles: List[float]) -> List[bool]:
        """
        Determine if each finger is extended or bent.
        Returns list of booleans (True for extended, False for bent)
        """
        return [angle > self.finger_straight_min for angle in angles]
    
    def _calculate_finger_angles(self, landmarks: Dict) -> List[float]:
        """Calculate angles for each finger with improved robustness."""
        angles = []
        # Define finger joint triplets (base, middle, tip)
        finger_indices = [
            (5, 6, 8),   # Index
            (9, 10, 12), # Middle
            (13, 14, 16),# Ring
            (17, 18, 20) # Pinky
        ]
        
        for base, mid, tip in finger_indices:
            # Get the three points of the finger
            p1 = np.array([landmarks[base].x, landmarks[base].y])
            p2 = np.array([landmarks[mid].x, landmarks[mid].y])
            p3 = np.array([landmarks[tip].x, landmarks[tip].y])
            
            # Calculate vectors
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Calculate angle using arctan2 for more stable results
            angle = np.degrees(np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2)))
            angles.append(angle)
            
        return angles
    
    def _smooth_gesture(self, gesture: GestureType, confidence: float) -> Tuple[GestureType, float]:
        """
        Smooth gesture detection over multiple frames to reduce flickering.
        """
        self.gesture_history.append((gesture, confidence))
        if len(self.gesture_history) > self.history_size:
            self.gesture_history.pop(0)
        
        # Count occurrences of each gesture in history
        gesture_counts = {}
        for g, c in self.gesture_history:
            gesture_counts[g] = gesture_counts.get(g, 0) + 1
        
        # Get most common gesture and its average confidence
        most_common = max(gesture_counts.items(), key=lambda x: x[1])
        if most_common[1] >= self.history_size * 0.6:  # 60% threshold
            confidences = [c for g, c in self.gesture_history if g == most_common[0]]
            return most_common[0], sum(confidences) / len(confidences)
        
        return GestureType.NONE, 0.0

    def _check_open_palm(self, landmarks: Dict) -> float:
        """
        Strict open palm detection - requires ALL fingers to be clearly extended.
        """
        # Get wrist position as reference point
        wrist = np.array([landmarks[0].x, landmarks[0].y])
        
        # Check each fingertip's position relative to its base (MCP joint)
        fingertips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
        mcp_joints = [5, 9, 13, 17]   # Corresponding MCP joints
        
        # Check if each finger is extended
        extended_fingers = 0
        for tip_idx, mcp_idx in zip(fingertips, mcp_joints):
            tip = np.array([landmarks[tip_idx].x, landmarks[tip_idx].y])
            mcp = np.array([landmarks[mcp_idx].x, landmarks[mcp_idx].y])
            
            # Get distances
            tip_to_wrist = np.linalg.norm(tip - wrist)
            base_to_wrist = np.linalg.norm(mcp - wrist)
            
            # Finger must be significantly extended (20% longer than base distance)
            if tip_to_wrist > (base_to_wrist * 1.2):
                extended_fingers += 1
                
        # Check thumb separately - must also be extended
        thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
        thumb_base = np.array([landmarks[2].x, landmarks[2].y])
        thumb_extended = np.linalg.norm(thumb_tip - wrist) > np.linalg.norm(thumb_base - wrist) * 1.2
        
        # Return 1.0 ONLY if ALL 5 fingers are clearly extended
        if extended_fingers == 4 and thumb_extended:
            return 1.0
        return 0.0

    def _check_pinch(self, landmarks: Dict) -> float:
        """
        Enhanced pinch detection that checks for exactly one finger touching the thumb.
        Returns high confidence only when exactly one finger is close to thumb.
        """
        thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
        fingertips = {
            'index': np.array([landmarks[8].x, landmarks[8].y]),
            'middle': np.array([landmarks[12].x, landmarks[12].y]),
            'ring': np.array([landmarks[16].x, landmarks[16].y]),
            'pinky': np.array([landmarks[20].x, landmarks[20].y])
        }
        
        # Count how many fingers are close to thumb
        close_fingers = 0
        pinch_distance = float('inf')
        
        for finger_tip in fingertips.values():
            distance = np.linalg.norm(thumb_tip - finger_tip)
            if distance < self.pinch_threshold:
                close_fingers += 1
                pinch_distance = min(pinch_distance, distance)
        
        # Only return confidence if exactly one finger is close to thumb
        if close_fingers == 1:
            return 1.0 - (pinch_distance / self.pinch_threshold)
        return 0.0

    def _check_grab(self, landmarks: Dict) -> float:
        """
        Simplified grab detection that checks if fingers are curled toward palm.
        """
        # Get wrist position as reference
        wrist = np.array([landmarks[0].x, landmarks[0].y])
        
        # Check each fingertip
        fingertips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
        mcp_joints = [5, 9, 13, 17]   # Corresponding MCP joints
        
        # Count curled fingers
        curled_fingers = 0
        for tip_idx, mcp_idx in zip(fingertips, mcp_joints):
            tip = np.array([landmarks[tip_idx].x, landmarks[tip_idx].y])
            mcp = np.array([landmarks[mcp_idx].x, landmarks[mcp_idx].y])
            
            # Finger is curled if tip is closer to wrist than base
            if np.linalg.norm(tip - wrist) < np.linalg.norm(mcp - wrist):
                curled_fingers += 1
        
        # Check thumb curl
        thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
        thumb_base = np.array([landmarks[2].x, landmarks[2].y])
        if np.linalg.norm(thumb_tip - wrist) < np.linalg.norm(thumb_base - wrist):
            curled_fingers += 1
        
        # Return confidence based on number of curled fingers
        return 1.0 if curled_fingers >= 4 else 0.0  # Consider grab if at least 4 fingers are curled


    def _check_point(self, landmarks: Dict) -> float:
        """
        Check if hand is making a pointing gesture (index finger extended, others closed).
        Returns confidence score between 0.0 and 1.0.
        """
        # Get wrist position as reference
        wrist = np.array([landmarks[0].x, landmarks[0].y])
        
        # Check index finger extension
        index_tip = np.array([landmarks[8].x, landmarks[8].y])
        index_pip = np.array([landmarks[6].x, landmarks[6].y])  # Index PIP joint
        index_mcp = np.array([landmarks[5].x, landmarks[5].y])  # Index MCP joint
        
        # Check other fingertips are curled
        other_tips = [
            np.array([landmarks[12].x, landmarks[12].y]),  # Middle
            np.array([landmarks[16].x, landmarks[16].y]),  # Ring
            np.array([landmarks[20].x, landmarks[20].y])   # Pinky
        ]
        
        # Check if index finger is extended
        index_extended = np.linalg.norm(index_tip - wrist) > np.linalg.norm(index_mcp - wrist) * 1.2
        
        # Check if other fingers are curled
        others_curled = all(
            np.linalg.norm(tip - wrist) < np.linalg.norm(index_mcp - wrist)
            for tip in other_tips
        )
        
        # Check thumb is curled
        thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
        thumb_curled = np.linalg.norm(thumb_tip - wrist) < np.linalg.norm(landmarks[2].x - wrist)
        
        if index_extended and others_curled and thumb_curled:
            return 1.0
        return 0.0

    def _check_victory(self, finger_states: List[bool], angles: List[float]) -> float:
        """Check if hand is making a victory/peace gesture."""
        # First two fingers should be extended, others bent
        correct_finger_state = (
            finger_states[0] and  # Index extended
            finger_states[1] and  # Middle extended
            not any(finger_states[2:])  # Others bent
        )
        
        if correct_finger_state:
            return 1.0
        return 0.0
    
    def classify_gesture(self, hand) -> GestureResult:
        """Classify gesture with improved none detection."""
        # Check each gesture
        gesture_confidence = {
            GestureType.OPEN_PALM: self._check_open_palm(hand.landmarks),
            GestureType.PINCH: self._check_pinch(hand.landmarks),
            GestureType.GRAB: self._check_grab(hand.landmarks),
            GestureType.POINT: self._check_point(hand.landmarks)  # Add this line
        }
        
        # Print confidence scores for debugging
        print("\nConfidence scores:")
        for gesture, confidence in gesture_confidence.items():
            print(f"{gesture.name}: {confidence:.2f}")
        
        best_gesture = max(gesture_confidence.items(), key=lambda x: x[1])
        
        # Return NONE if confidence is too low
        if best_gesture[1] < 0.6:  # Increased threshold for more reliable detection
            return GestureResult(
                gesture_type=GestureType.NONE,
                confidence=0.0,
                hand_side=hand.handedness,
                palm_position=hand.palm_center
            )
        
        # Apply smoothing
        smoothed_gesture, smoothed_confidence = self._smooth_gesture(
            best_gesture[0], best_gesture[1]
        )
        
        return GestureResult(
            gesture_type=smoothed_gesture,
            confidence=smoothed_confidence,
            hand_side=hand.handedness,
            palm_position=hand.palm_center
        )
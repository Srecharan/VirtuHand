import numpy as np
import torch
import torch.nn as nn
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional

class DynamicGestureType(Enum):
    NONE = 0
    SWIPE_LEFT = 1
    SWIPE_RIGHT = 2
    CIRCLE = 3
    WAVE = 4

@dataclass
class DynamicGestureResult:
    gesture_type: DynamicGestureType
    confidence: float

class GRUGestureModel(nn.Module):
    def __init__(self, input_size=63, hidden_size=32, num_layers=2, num_classes=5):
        super(GRUGestureModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])  # Take only the last output
        return out

class DynamicGestureRecognizer:
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        self.landmark_buffer = []
        self.velocity_buffer = []
        self.last_position = None
        
        # Adjusted thresholds
        self.min_velocity_threshold = 0.01  # Reduced from 0.02
        self.min_active_frames = 10    # Reduced from 15
        self.motion_confidence_threshold = 0.6  # Reduced from 0.7
        
        # Gesture-specific parameters
        self.swipe_min_distance = 0.15  # Reduced from 0.3
        self.swipe_max_vertical = 0.1  # Maximum vertical movement for swipes
        self.circle_min_points = 15    # Reduced from 20
        
        # Add cooldown for gestures
        self.last_gesture = None
        self.gesture_cooldown = 0
        self.cooldown_frames = 15  # Frames to wait before detecting same gesture
        
        # Store previous gesture for continuity
        self.prev_gesture = DynamicGestureType.NONE
        self.gesture_count = 0
        self.min_consistent_frames = 5  # Minimum frames to maintain same gesture
        
    def _calculate_velocity_components(self, landmarks: dict) -> tuple:
        """Calculate both horizontal and vertical velocity components."""
        current_position = np.array([landmarks[0].x, landmarks[0].y])
        
        if self.last_position is None:
            self.last_position = current_position
            return 0.0, 0.0
            
        dx = abs(current_position[0] - self.last_position[0])
        dy = abs(current_position[1] - self.last_position[1])
        self.last_position = current_position
        return dx, dy
    
    def _detect_significant_motion(self) -> tuple:
        """Enhanced motion detection with better pattern classification."""
        if len(self.velocity_buffer) < self.sequence_length:
            return False, "none"
            
        # Calculate total motion and its components
        horizontal_motion = sum(v[0] for v in self.velocity_buffer)
        vertical_motion = sum(v[1] for v in self.velocity_buffer)
        total_motion = sum(v[0] + v[1] for v in self.velocity_buffer)
        
        # Calculate motion ratio between horizontal and vertical
        motion_ratio = abs(horizontal_motion - vertical_motion) / (total_motion + 1e-6)
        
        # Check motion patterns
        if motion_ratio < 0.3:  # Similar horizontal and vertical motion
            if total_motion > self.min_velocity_threshold * self.sequence_length:
                return True, "mixed"
        elif horizontal_motion > vertical_motion * 1.5:
            return True, "horizontal"
        elif vertical_motion > horizontal_motion * 1.5:
            return True, "vertical"
            
        return False, "none"
        
    def _check_swipe_pattern(self, positions: np.ndarray) -> Optional[DynamicGestureType]:
        """Enhanced swipe detection with direction validation."""
        if len(positions) < 10:  # Need minimum points for swipe
            return None
            
        # Calculate total horizontal and vertical movement
        total_movement = positions[-1] - positions[0]
        dx, dy = abs(total_movement[0]), abs(total_movement[1])
        
        # Check if movement is primarily horizontal
        if dx > self.swipe_min_distance and dx > dy * 2:
            # Determine direction
            direction = np.sign(total_movement[0])
            return (DynamicGestureType.SWIPE_RIGHT if direction > 0 
                   else DynamicGestureType.SWIPE_LEFT)
        return None
        
    def _check_circle_pattern(self, positions: np.ndarray) -> float:
        """Improved circle detection with better rotation and shape analysis."""
        if len(positions) < self.circle_min_points:
            return 0.0
            
        # Center the positions
        centered = positions - positions.mean(axis=0)
        
        # Calculate radius for each point
        radii = np.linalg.norm(centered, axis=1)
        mean_radius = np.mean(radii)
        
        # Check if points maintain consistent distance from center
        radius_variation = np.std(radii) / mean_radius
        if radius_variation > 0.4:  # Allow more variation in radius
            return 0.0
            
        # Calculate angles and their changes
        angles = np.arctan2(centered[:, 1], centered[:, 0])
        angle_diffs = np.diff(angles)
        
        # Wrap angle differences to [-pi, pi]
        angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
        
        # Calculate total angular movement
        total_angle = np.abs(np.sum(angle_diffs))
        
        # Check for minimum rotation (at least 270 degrees)
        if total_angle < 4.7:  # 4.7 radians ≈ 270 degrees
            return 0.0
            
        # Calculate direction consistency
        direction_changes = np.diff(np.sign(angle_diffs))
        direction_consistency = 1.0 - (np.count_nonzero(direction_changes) / len(direction_changes))
        
        # Final confidence score combines multiple factors
        shape_score = max(0.0, 1.0 - radius_variation * 2)
        rotation_score = min(1.0, total_angle / (2 * np.pi))
        consistency_score = direction_consistency
        
        confidence = (shape_score * 0.3 + 
                     rotation_score * 0.4 + 
                     consistency_score * 0.3)
        
        return confidence if confidence > 0.6 else 0.0
        
    def _check_wave_pattern(self, positions: np.ndarray) -> float:
        """Enhanced wave detection with stricter vertical movement check."""
        if len(positions) < 15:  # Need minimum points for wave
            return 0.0
            
        # Get vertical positions
        y_positions = positions[:, 1]
        
        # Find peaks and valleys
        peaks = []
        valleys = []
        
        for i in range(1, len(y_positions) - 1):
            if (y_positions[i] > y_positions[i-1] and 
                y_positions[i] > y_positions[i+1]):
                peaks.append(i)
            elif (y_positions[i] < y_positions[i-1] and 
                  y_positions[i] < y_positions[i+1]):
                valleys.append(i)
        
        # Need at least 2 peaks and 2 valleys
        if len(peaks) >= 2 and len(valleys) >= 2:
            # Calculate average peak-to-valley distance
            peak_valley_heights = []
            for p, v in zip(peaks, valleys):
                peak_valley_heights.append(abs(y_positions[p] - y_positions[v]))
            
            avg_height = np.mean(peak_valley_heights)
            
            # Check if horizontal movement is minimal
            x_movement = abs(positions[-1, 0] - positions[0, 0])
            if x_movement < avg_height * 0.5:
                return min(1.0, avg_height * 5)  # Scale confidence with wave amplitude
        
        return 0.0
    
    def update(self, landmarks: dict) -> DynamicGestureResult:
        """Update gesture recognition with improved gesture switching."""
        # Calculate and store velocities
        dx, dy = self._calculate_velocity_components(landmarks)
        self.velocity_buffer.append((dx, dy))
        if len(self.velocity_buffer) > self.sequence_length:
            self.velocity_buffer.pop(0)
            
        # Store landmarks
        self.landmark_buffer.append(landmarks)
        if len(self.landmark_buffer) > self.sequence_length:
            self.landmark_buffer.pop(0)
            
        # Decrease cooldown
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1
            
        # Get motion type
        has_motion, motion_type = self._detect_significant_motion()
        if not has_motion:
            self.prev_gesture = DynamicGestureType.NONE
            return DynamicGestureResult(DynamicGestureType.NONE, 0.0)
            
        # Extract positions for analysis
        positions = np.array([[landmarks[0].x, landmarks[0].y] 
                            for landmarks in self.landmark_buffer])
        
        # Check each gesture type with confidence
        confidences = {
            DynamicGestureType.NONE: 0.0
        }
        
        # Check swipe if motion is primarily horizontal
        if motion_type == "horizontal":
            swipe_type = self._check_swipe_pattern(positions)
            if swipe_type:
                confidences[swipe_type] = 0.8
        
        # Check circle if motion is mixed
        if motion_type == "mixed":
            circle_conf = self._check_circle_pattern(positions)
            if circle_conf > 0:
                confidences[DynamicGestureType.CIRCLE] = circle_conf
        
        # Check wave if motion is primarily vertical
        if motion_type == "vertical":
            wave_conf = self._check_wave_pattern(positions)
            if wave_conf > 0:
                confidences[DynamicGestureType.WAVE] = wave_conf
        
        # Get highest confidence gesture
        best_gesture = max(confidences.items(), key=lambda x: x[1])
        
        # Apply gesture continuity
        if best_gesture[0] == self.prev_gesture:
            self.gesture_count += 1
        else:
            self.gesture_count = 0
        
        # Only switch gestures if new gesture is significantly more confident
        if (best_gesture[1] > self.motion_confidence_threshold and 
            (best_gesture[0] == self.prev_gesture or 
             best_gesture[1] > confidences.get(self.prev_gesture, 0) + 0.2)):
            
            self.prev_gesture = best_gesture[0]
            return DynamicGestureResult(best_gesture[0], best_gesture[1])
        
        # Keep previous gesture if it was stable
        if self.gesture_count >= self.min_consistent_frames:
            prev_conf = confidences.get(self.prev_gesture, 0)
            if prev_conf > self.motion_confidence_threshold * 0.8:
                return DynamicGestureResult(self.prev_gesture, prev_conf)
        
        return DynamicGestureResult(DynamicGestureType.NONE, 0.0)
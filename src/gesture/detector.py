"""
Hand Detector Module
------------------
This module provides real-time hand detection and 3D tracking using MediaPipe and RealSense depth data.
It forms the foundation for gesture recognition by providing accurate hand landmark positions
and depth information.
"""

import mediapipe as mp
import numpy as np
import cv2
import os
import sys
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.gesture.classifier import GestureClassifier
# Rest of your imports...
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class HandLandmark:
    """Represents a single hand landmark with 3D coordinates."""
    x: float 
    y: float  
    z: float  
    pixel_x: int  
    pixel_y: int  
    visibility: float  

@dataclass
class Hand:
    """Represents a detected hand with all its landmarks and metadata."""
    landmarks: Dict[int, HandLandmark]  
    handedness: str  # 'Left' or 'Right'
    confidence: float  # Detection confidence score
    palm_center: Tuple[float, float, float]  # 3D position of palm center
    
class HandDetector:
    def __init__(self, 
                 max_hands: int = 2,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5):
        """Initialize the hand detector with MediaPipe Hands with GPU acceleration."""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.rotation_tracker = HandRotationTracker()

        # Enable GPU acceleration
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1,  
            static_image_mode=False  
        )
        
        self.previous_hands: List[Hand] = []
        
    def _calculate_palm_center(self, landmarks: Dict[int, HandLandmark]) -> Tuple[float, float, float]:
        """
        Calculate the center of the palm using specific landmark points.
        Uses landmarks 0 (wrist), 5, 9, 13, 17 (finger bases) for calculation.
        
        Args:
            landmarks: Dictionary of hand landmarks
        Returns:
            Tuple of (x, y, z) coordinates of palm center
        """
        palm_points = [0, 5, 9, 13, 17]  # Key landmark indices for palm
        coords = np.array([[landmarks[idx].x, landmarks[idx].y, landmarks[idx].z] 
                          for idx in palm_points])
        return tuple(np.mean(coords, axis=0))
        
    def _convert_landmarks_to_pixel(self, landmark, image_width: int, image_height: int) -> Tuple[int, int]:
        """Convert normalized landmark coordinates to pixel coordinates."""
        return (int(landmark.x * image_width), int(landmark.y * image_height))
        
    def _get_landmark_depth(self, depth_frame: np.ndarray, pixel_x: int, pixel_y: int) -> float:
        """Get depth value for a landmark from the depth frame."""
        if 0 <= pixel_y < depth_frame.shape[0] and 0 <= pixel_x < depth_frame.shape[1]:
            # Get depth in meters (assuming depth is in millimeters)
            return depth_frame[pixel_y, pixel_x] / 1000.0
        return 0.0
        
    def _determine_handedness(self, landmarks: Dict[int, HandLandmark]) -> str:
        """
        Determine hand side (left/right) based on thumb position relative to other fingers.
        When palm faces camera, thumb is on left for right hand and right for left hand.
        """
        # Get wrist and thumb positions
        wrist = np.array([landmarks[0].x, landmarks[0].y])
        thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
        index_mcp = np.array([landmarks[5].x, landmarks[5].y])
        
        # Calculate vector from wrist to index finger base
        palm_vector = index_mcp - wrist
        # Calculate vector from wrist to thumb
        thumb_vector = thumb_tip - wrist
        
        # Use cross product to determine thumb side
        cross_product = np.cross(palm_vector, thumb_vector)
        
        return "Left" if cross_product > 0 else "Right"

    def detect_hands(self, color_frame: np.ndarray, depth_frame: np.ndarray) -> List[Hand]:
        """
        Detect and track hands in the current frame.
        
        Args:
            color_frame: RGB image from RealSense
            depth_frame: Depth image from RealSense
            
        Returns:
            List of detected Hand objects with 3D landmarks
        """
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image_rgb.shape
        
        # Process the image and detect hands
        results = self.hands.process(image_rgb)
        
        # Initialize list for detected hands
        detected_hands: List[Hand] = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                                results.multi_handedness):
                landmarks_dict = {}
                
                # Process each landmark
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    pixel_x, pixel_y = self._convert_landmarks_to_pixel(
                        landmark, image_width, image_height
                    )
                    
                    # Get depth from RealSense depth frame
                    depth = self._get_landmark_depth(depth_frame, pixel_x, pixel_y)
                    
                    landmarks_dict[idx] = HandLandmark(
                        x=landmark.x,
                        y=landmark.y,
                        z=depth,
                        pixel_x=pixel_x,
                        pixel_y=pixel_y,
                        visibility=landmark.visibility
                    )
                
                # Create Hand object
                hand = Hand(
                    landmarks=landmarks_dict,
                    handedness=self._determine_handedness(landmarks_dict),  # Use our geometric calculation
                    confidence=handedness.classification[0].score,
                    palm_center=self._calculate_palm_center(landmarks_dict)
                )
                
                detected_hands.append(hand)
        
        self.previous_hands = detected_hands
        return detected_hands
        
    def draw_landmarks(self, image: np.ndarray, hands: List[Hand], classifier: GestureClassifier = None, dynamic_gesture_result = None) -> np.ndarray:
        """Draw hand landmarks, connections, and gesture information on the image."""
        annotated_image = image.copy()
        
        for hand in hands:
            # Draw basic landmarks
            for idx, landmark in hand.landmarks.items():
                point = (landmark.pixel_x, landmark.pixel_y)
                cv2.circle(annotated_image, point, 5, (0, 255, 0), -1)
            
            # Draw connections
            for connection in self.mp_hands.HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                
                if start_idx in hand.landmarks and end_idx in hand.landmarks:
                    start_point = (hand.landmarks[start_idx].pixel_x, 
                                hand.landmarks[start_idx].pixel_y)
                    end_point = (hand.landmarks[end_idx].pixel_x, 
                            hand.landmarks[end_idx].pixel_y)
                    cv2.line(annotated_image, start_point, end_point, (0, 255, 0), 2)
            
            if classifier:
                gesture = classifier.classify_gesture(hand)
                
                # Get wrist landmark for text positioning
                wrist = hand.landmarks[0]
                
                # Calculate text position
                text_x = wrist.pixel_x - 180
                text_y = wrist.pixel_y - 150
                
                # Draw static gesture
                self._draw_gesture_text(
                    annotated_image,
                    f"{gesture.gesture_type.name} ({gesture.confidence:.2f})",
                    (text_x, text_y)
                )
                
                # Draw dynamic gesture if available
                if dynamic_gesture_result and dynamic_gesture_result.confidence > 0.7:
                    self._draw_gesture_text(
                        annotated_image,
                        f"Dynamic: {dynamic_gesture_result.gesture_type.name}",
                        (text_x, text_y - 30)  
                    )
        
        return annotated_image
    
    def _draw_gesture_text(self, image, text, position):
        """Helper method to draw gesture text with background"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8
        thickness = 2

        (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
        
        cv2.rectangle(
            image,
            (position[0] - 5, position[1] - text_height - 5),
            (position[0] + text_width + 5, position[1] + 5),
            (0, 0, 0),
            -1
        )
        
        cv2.putText(
            image,
            text,
            position,
            font,
            scale,
            (0, 255, 0),
            thickness
        )

class HandRotationTracker:
    def __init__(self):
        self.prev_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.rotation_filter = RotationKalmanFilter()
        
    def calculate_hand_orientation(self, landmarks):
        """Calculate hand orientation using palm and finger vectors"""
        # Get key landmarks for better rotation calculation
        wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
        thumb_cmc = np.array([landmarks[1].x, landmarks[1].y, landmarks[1].z])
        index_mcp = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])
        middle_mcp = np.array([landmarks[9].x, landmarks[9].y, landmarks[9].z])
        pinky_mcp = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])
        
        # Calculate primary direction vectors
        palm_forward = middle_mcp - wrist  # Primary forward direction
        palm_forward = palm_forward / np.linalg.norm(palm_forward)
        
        palm_side = pinky_mcp - index_mcp  # Side vector across palm
        palm_side = palm_side / np.linalg.norm(palm_side)
        
        # Calculate palm normal (up vector) from cross product
        palm_normal = np.cross(palm_forward, palm_side)
        palm_normal = palm_normal / np.linalg.norm(palm_normal)
        
        # Recalculate side vector to ensure perfect orthogonality
        palm_side = np.cross(palm_normal, palm_forward)
        
        # Calculate wrist twist using thumb direction
        thumb_vector = thumb_cmc - wrist
        thumb_vector = thumb_vector / np.linalg.norm(thumb_vector)
        
        # Project thumb onto palm plane and calculate twist angle
        thumb_projected = thumb_vector - np.dot(thumb_vector, palm_normal) * palm_normal
        thumb_projected = thumb_projected / np.linalg.norm(thumb_projected)
        
        twist_angle = np.arctan2(
            np.dot(np.cross(palm_side, thumb_projected), palm_normal),
            np.dot(palm_side, thumb_projected)
        )
        
        # Apply twist rotation to the palm frame
        twist_matrix = self._axis_angle_to_matrix(palm_normal, twist_angle)
        rotated_side = twist_matrix @ palm_side
        
        # Create final rotation matrix
        rotation_matrix = np.column_stack((palm_forward, rotated_side, palm_normal))
        
        # Convert to quaternion
        quaternion = self._matrix_to_quaternion(rotation_matrix)
        
        # Ensure continuity with previous frame
        if np.dot(quaternion, self.prev_quaternion) < 0:
            quaternion = -quaternion
        
        # Apply Kalman filtering
        filtered_quaternion = self.rotation_filter.update(quaternion)
        self.prev_quaternion = filtered_quaternion
        
        return filtered_quaternion

    def _axis_angle_to_matrix(self, axis, angle):
        """Convert axis-angle rotation to matrix"""
        c = np.cos(angle)
        s = np.sin(angle)
        v = 1 - c
        x, y, z = axis
        
        return np.array([
            [x*x*v + c,   x*y*v - z*s, x*z*v + y*s],
            [x*y*v + z*s, y*y*v + c,   y*z*v - x*s],
            [x*z*v - y*s, y*z*v + x*s, z*z*v + c  ]
        ])
    
    def _matrix_to_quaternion(self, matrix):
        """Convert 3x3 rotation matrix to quaternion"""
        trace = np.trace(matrix)
        
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            w = 0.25 * S
            x = (matrix[2, 1] - matrix[1, 2]) / S
            y = (matrix[0, 2] - matrix[2, 0]) / S
            z = (matrix[1, 0] - matrix[0, 1]) / S
        else:
            if matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
                S = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
                w = (matrix[2, 1] - matrix[1, 2]) / S
                x = 0.25 * S
                y = (matrix[0, 1] + matrix[1, 0]) / S
                z = (matrix[0, 2] + matrix[2, 0]) / S
            elif matrix[1, 1] > matrix[2, 2]:
                S = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
                w = (matrix[0, 2] - matrix[2, 0]) / S
                x = (matrix[0, 1] + matrix[1, 0]) / S
                y = 0.25 * S
                z = (matrix[1, 2] + matrix[2, 1]) / S
            else:
                S = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
                w = (matrix[1, 0] - matrix[0, 1]) / S
                x = (matrix[0, 2] + matrix[2, 0]) / S
                y = (matrix[1, 2] + matrix[2, 1]) / S
                z = 0.25 * S
        
        return np.array([w, x, y, z])

class RotationKalmanFilter:
    def __init__(self):
        self.state = np.array([1.0, 0.0, 0.0, 0.0])  
        self.P = np.eye(4) * 0.1  
        self.Q = np.eye(4) * 0.01  
        self.R = np.eye(4) * 0.1   
        
    def update(self, measurement):
        # Predict
        # For quaternions, prediction is identity since we assume constant orientation
        
        y = measurement - self.state
        
        # Normalize quaternion difference
        y = y / np.linalg.norm(y)
        
        # Kalman gain
        K = self.P @ np.linalg.inv(self.P + self.R)
        
        # Update state
        self.state = self.state + K @ y
        
        # Normalize quaternion
        self.state = self.state / np.linalg.norm(self.state)
        
        # Update covariance
        self.P = (np.eye(4) - K) @ self.P
        
        return self.state


def test_hand_detector():
    """Test function to verify hand detection and gesture recognition."""
    from src.camera.realsense import RealSenseCamera
    from src.gesture.classifier import GestureClassifier
    
    detector = HandDetector()
    classifier = GestureClassifier()
    
    try:
        with RealSenseCamera() as camera:
            while True:
                frame_data = camera.get_frame()
                if frame_data is None:
                    continue
                
                try:
                    hands = detector.detect_hands(frame_data.color, frame_data.depth)
                    
                    # Classify gestures for each hand
                    for hand in hands:
                        gesture = classifier.classify_gesture(hand)
                        print(f"{gesture.hand_side} Hand: {gesture.gesture_type.name} "
                              f"({gesture.confidence:.2f})")
                    
                    # Draw landmarks - PASS THE CLASSIFIER HERE
                    annotated_image = detector.draw_landmarks(frame_data.color, hands, classifier)  # Added classifier here
                    
                    # Show results
                    cv2.imshow('Hand Tracking', annotated_image)
                    cv2.imshow('Depth', frame_data.depth_colormap)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue
                    
    except Exception as e:
        print(f"Error initializing camera: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_hand_detector()
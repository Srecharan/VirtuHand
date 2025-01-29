import asyncio
import websockets
import cv2
import json
import numpy as np
import pyrealsense2 as rs
from src.gesture.detector import HandDetector
from src.gesture.classifier import GestureClassifier, GestureType
from src.gesture.depth_utils import DepthKalmanFilter
from src.gesture.dynamic_gesture import DynamicGestureRecognizer, DynamicGestureType, DynamicGestureResult
from typing import Optional, Tuple
import numpy as np
from enum import Enum, auto

class DynamicGestureRecognizer:
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        self.landmark_buffer = []
        self.velocity_buffer = []
        self.last_position = None
        
        # Adjusted thresholds
        self.min_velocity_threshold = 0.01
        self.min_active_frames = 10
        self.motion_confidence_threshold = 0.6
        
        # Gesture-specific parameters
        self.swipe_min_distance = 0.15
        self.swipe_max_vertical = 0.1
        self.circle_min_points = 15
        
        # Gesture continuity
        self.prev_gesture = DynamicGestureType.NONE
        self.gesture_count = 0
        self.min_consistent_frames = 5
        
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

class EnhancedTestServer:
    def __init__(self):
        self.pipeline = None
        self.config = rs.config()
        
        # Initialize filters
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        
        # Configure spatial filter
        self.spatial_filter.set_option(rs.option.filter_magnitude, 2)
        self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
        
        # Initialize other components
        self.depth_kalman = DepthKalmanFilter()
        self.hand_detector = HandDetector(max_hands=1)
        self.gesture_classifier = GestureClassifier()
        self.dynamic_gesture_recognizer = DynamicGestureRecognizer()

        # Debug flags
        self.show_debug_info = True

        # Recording variables
        self.is_recording = False
        self.realsense_recorder = None
        
        # Get the project root directory (2 levels up from tests)
        import os
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.recording_path = os.path.join(self.project_root, "recordings")
        print(f"Recording path set to: {self.recording_path}")
        self.frame_count = 0

    def initialize_recorder(self):
        """Initialize video recorder"""
        import os
        from datetime import datetime
        
        try:
            # Create recordings directory if it doesn't exist
            if not os.path.exists(self.recording_path):
                os.makedirs(self.recording_path)
                
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(self.recording_path, f"realsense_{timestamp}.mp4")
            
            # Initialize video writer with MP4V codec - widely supported
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4V codec
            self.realsense_recorder = cv2.VideoWriter(
                video_path,
                fourcc,
                30.0,  # FPS
                (640, 480),  # Frame size
                isColor=True  # Color video
            )
            
            if not self.realsense_recorder.isOpened():
                raise Exception("Failed to create VideoWriter")
                
            print(f"Recording to: {video_path}")
            self.frame_count = 0
            return True
            
        except Exception as e:
            print(f"Error initializing recorder: {e}")
            self.realsense_recorder = None
            return False

    def handle_recording(self, color_image):
        """Handle recording of a frame"""
        if self.is_recording and self.realsense_recorder is not None:
            try:
                # Ensure image is in BGR format (OpenCV default)
                if len(color_image.shape) == 2:  # If grayscale
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_GRAY2BGR)
                    
                # Add recording indicator
                frame_to_save = color_image.copy()
                
                # Write the frame
                self.realsense_recorder.write(frame_to_save)
                self.frame_count += 1
                
                # Add recording indicator to display (not saved)
                cv2.circle(color_image, (30, 30), 15, (0, 0, 255), -1)
                cv2.putText(
                    color_image,
                    f"REC {self.frame_count//30}s",
                    (55, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
                
            except Exception as e:
                print(f"Error during recording: {e}")
                self.stop_recording()

    def stop_recording(self):
        """Stop recording and release resources"""
        if self.realsense_recorder is not None:
            print(f"Stopping recording... Total frames: {self.frame_count}")
            self.realsense_recorder.release()
            self.realsense_recorder = None
            self.is_recording = False
            print("Recording completed")
            
    def draw_debug_info(self, image, y_offset, text, color=(0, 255, 0)):
        """Helper function to draw debug text with background"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
        
        # Draw background rectangle
        cv2.rectangle(
            image,
            (10, y_offset - text_height - 5),
            (10 + text_width + 5, y_offset + 5),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            image,
            text,
            (10, y_offset),
            font,
            scale,
            color,
            thickness
        )
        
        return y_offset + 30

    def process_depth_frame(self, depth_frame, hand_landmarks):
        """Process depth frame with multiple filtering stages"""
        filtered_depth = self.spatial_filter.process(depth_frame)
        filtered_depth = self.temporal_filter.process(filtered_depth)
        depth_image = np.asanyarray(filtered_depth.get_data())
        
        palm_landmarks = [0, 5, 9, 13, 17]
        depths = []
        
        for idx in palm_landmarks:
            if idx in hand_landmarks:
                landmark = hand_landmarks[idx]
                pixel_x = int(landmark.x * 640)
                pixel_y = int(landmark.y * 480)
                
                if 0 <= pixel_x < 640 and 0 <= pixel_y < 480:
                    depth = depth_image[pixel_y, pixel_x] / 1000.0
                    if depth > 0:
                        depths.append(depth)
        
        if depths:
            median = np.median(depths)
            mad = np.median(np.abs(np.array(depths) - median))
            modified_z_scores = 0.6745 * (np.array(depths) - median) / mad
            filtered_depths = [d for d, z in zip(depths, modified_z_scores) if abs(z) < 3.5]
            
            if filtered_depths:
                palm_depth = np.mean(filtered_depths)
                return self.depth_kalman.update(palm_depth)
            
        return 0.0
    
    def initialize_camera(self):
        try:
            if self.pipeline is not None:
                self.pipeline.stop()
            
            self.pipeline = rs.pipeline()  # Create new pipeline
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            print("Starting RealSense pipeline...")
            self.pipeline.start(self.config)
            print("RealSense pipeline started successfully")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False

    def normalize_coordinates(self, x, y, z):
        """Normalize coordinates to a reasonable range for Unity"""
        normalized_x = (x - 0.5) * 2
        normalized_y = (y - 0.5) * 2
        
        min_depth = 0.2
        max_depth = 1.0
        normalized_z = 2.0 * ((z - min_depth) / (max_depth - min_depth)) - 1.0
        normalized_z = max(-1.0, min(1.0, normalized_z))
        
        return normalized_x, normalized_y, normalized_z

    async def handle_client(self, websocket):
        print("Client connected!")
        
        if not self.initialize_camera():
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Camera initialization failed"
            }))
            return

        # Start recording automatically
        print("Initializing recording...")
        if self.initialize_recorder():
            self.is_recording = True
            print("Recording started automatically")
        else:
            print("Failed to start recording")

        try:
            while True:
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    
                    if not color_frame or not depth_frame:
                        continue

                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())
                    
                    hands = self.hand_detector.detect_hands(color_image, depth_image)
                    
                    message = {
                        "type": "hand_tracking",
                        "hands_detected": False,
                        "hand_position": {"x": 0, "y": 0, "z": 0},
                        "gesture_type": "NONE",
                        "gesture_confidence": 0.0
                    }

                    if hands:
                        palm_center = hands[0].palm_center
                        gesture_result = self.gesture_classifier.classify_gesture(hands[0])
                        dynamic_gesture = self.dynamic_gesture_recognizer.update(hands[0].landmarks)
                        smoothed_depth = self.process_depth_frame(depth_frame, hands[0].landmarks)
                        
                        norm_x, norm_y, norm_z = self.normalize_coordinates(
                            palm_center[0], 
                            palm_center[1], 
                            smoothed_depth
                        )

                        # Calculate finger states
                        finger_states = self._calculate_finger_states(hands[0])
                        
                        message.update({
                            "hands_detected": True,
                            "hand_position": {
                                "x": norm_x,
                                "y": norm_y,
                                "z": norm_z
                            },
                            "gesture_type": gesture_result.gesture_type.name,
                            "gesture_confidence": gesture_result.confidence,
                            "fingerStates": finger_states,
                            "dynamic_gesture_type": dynamic_gesture.gesture_type.name,
                            "dynamic_gesture_confidence": dynamic_gesture.confidence    
                        })
                        
                        # Draw debug visualization
                        y_offset = 30
                        
                        if self.show_debug_info:
                            # Show gesture info
                            y_offset = self.draw_debug_info(
                                color_image,
                                y_offset,
                                f"Gesture: {gesture_result.gesture_type.name} ({gesture_result.confidence:.2f})"
                            )
                            
                            # Show dynamic gesture info
                            y_offset = self.draw_debug_info(
                                color_image,
                                y_offset,
                                f"Dynamic: {dynamic_gesture.gesture_type.name} ({dynamic_gesture.confidence:.2f})"
                            )
                            
                            # Show velocity info if available
                            if hasattr(self.dynamic_gesture_recognizer, 'velocity_buffer') and \
                               len(self.dynamic_gesture_recognizer.velocity_buffer) > 0:
                                dx, dy = self.dynamic_gesture_recognizer.velocity_buffer[-1]
                                y_offset = self.draw_debug_info(
                                    color_image,
                                    y_offset,
                                    f"Velocity - dx: {dx:.3f}, dy: {dy:.3f}"
                                )

                        # Draw hand landmarks
                        color_image = self.hand_detector.draw_landmarks(
                            color_image, 
                            hands,
                            self.gesture_classifier,
                            dynamic_gesture
                        )
                        
                        # Draw final prediction if confidence is high
                        if dynamic_gesture.confidence > 0.7:
                            self.draw_debug_info(
                                color_image,
                                color_image.shape[0] - 30,
                                f"Detected: {dynamic_gesture.gesture_type.name}",
                                color=(0, 255, 0)
                            )
                    
                    # Send data to Unity
                    await websocket.send(json.dumps(message))
                    self.handle_recording(color_image)

                    # Show video feed
                    cv2.imshow('Hand Tracking', color_image)
                    
                    # Simple quit with 'q'
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    await asyncio.sleep(0.033)  # ~30 FPS

                except Exception as e:
                    print(f"Error in processing loop: {e}")
                    continue

        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            print("Cleaning up...")
            if self.is_recording:
                self.stop_recording()
            if self.pipeline:
                self.pipeline.stop()
            cv2.destroyAllWindows()
            
    def _calculate_finger_states(self, hand):
        """Calculate finger states for Unity."""
        finger_states = []
        finger_indices = [
            [4, 3, 2],    # Thumb
            [8, 7, 6],    # Index
            [12, 11, 10], # Middle
            [16, 15, 14], # Ring
            [20, 19, 18]  # Pinky
        ]
        
        wrist = np.array([hand.landmarks[0].x, hand.landmarks[0].y])
        
        for finger_idx in finger_indices:
            tip_idx = finger_idx[0]
            mcp_idx = finger_idx[2]
            
            tip = np.array([hand.landmarks[tip_idx].x, hand.landmarks[tip_idx].y])
            mcp = np.array([hand.landmarks[mcp_idx].x, hand.landmarks[mcp_idx].y])
            
            tip_distance = np.linalg.norm(tip - wrist)
            mcp_distance = np.linalg.norm(mcp - wrist)
            
            is_extended = bool(tip_distance > mcp_distance * 1.2)
            bend_angle = float(max(0, min(1, 1 - (tip_distance / (mcp_distance * 1.5)))))
            
            finger_states.append({
                "isExtended": is_extended,
                "bendAngle": bend_angle
            })
            
        return finger_states

    async def start_server(self):
        async with websockets.serve(
            self.handle_client, 
            "127.0.0.1", 
            8765, 
            ping_interval=None,
            ping_timeout=None
        ):
            print("Server running on ws://127.0.0.1:8765")
            await asyncio.Future()

def main():
    server = EnhancedTestServer()
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
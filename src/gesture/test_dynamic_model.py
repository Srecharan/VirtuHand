import torch
import cv2
import numpy as np
import pyrealsense2 as rs
from detector import HandDetector
from dynamic_gesture import GRUGestureModel, DynamicGestureType, DynamicGestureRecognizer
from classifier import GestureClassifier

class DynamicGestureTester:
    def __init__(self):
        # Initialize RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # Initialize detectors and model
        self.hand_detector = HandDetector(max_hands=1)
        self.gesture_classifier = GestureClassifier()
        
        # Initialize both recognition systems
        self.model = GRUGestureModel()
        self.model.load_state_dict(torch.load('models/best_dynamic_gesture_model.pth'))
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        self.geometric_recognizer = DynamicGestureRecognizer()
        
        # Initialize sequence buffer for GRU model
        self.sequence_buffer = []
        self.sequence_length = 30
        
        # Debug flags
        self.show_debug_info = True
        
    def preprocess_landmarks(self, landmarks):
        """Convert landmarks to model input format"""
        landmark_array = []
        for i in range(21):  # 21 hand landmarks
            if i in landmarks:
                landmark_array.extend([
                    landmarks[i].x,
                    landmarks[i].y,
                    landmarks[i].z
                ])
            else:
                landmark_array.extend([0, 0, 0])
        return landmark_array
    
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
        
        return y_offset + 30  # Return next y_offset
        
    def run(self):
        try:
            self.pipeline.start(self.config)
            
            while True:
                # Get frames
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                    
                # Process frames
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Detect hands
                hands = self.hand_detector.detect_hands(color_image, depth_image)
                
                if hands:
                    # Process landmarks for GRU model
                    landmark_data = self.preprocess_landmarks(hands[0].landmarks)
                    self.sequence_buffer.append(landmark_data)
                    
                    # Keep only last sequence_length frames
                    if len(self.sequence_buffer) > self.sequence_length:
                        self.sequence_buffer.pop(0)
                    
                    # Get predictions from both systems
                    gru_prediction = None
                    geometric_prediction = self.geometric_recognizer.update(hands[0].landmarks)
                    
                    # Make GRU prediction if we have enough frames
                    if len(self.sequence_buffer) == self.sequence_length:
                        sequence = torch.FloatTensor(self.sequence_buffer).unsqueeze(0)
                        sequence = sequence.to(self.device)
                        
                        with torch.no_grad():
                            outputs = self.model(sequence)
                            probabilities = torch.softmax(outputs, dim=1)
                            pred_idx = torch.argmax(probabilities).item()
                            confidence = probabilities[0][pred_idx].item()
                            
                            if confidence > 0.7:  # Only consider high confidence predictions
                                gru_prediction = (DynamicGestureType(pred_idx), confidence)
                    
                    # Draw predictions and debug info
                    y_offset = 30
                    
                    if self.show_debug_info:
                        # Show geometric recognizer debug info
                        y_offset = self.draw_debug_info(
                            color_image,
                            y_offset,
                            f"Geometric: {geometric_prediction.gesture_type.name} ({geometric_prediction.confidence:.2f})"
                        )
                        
                        # Show GRU model prediction if available
                        if gru_prediction:
                            y_offset = self.draw_debug_info(
                                color_image,
                                y_offset,
                                f"GRU: {gru_prediction[0].name} ({gru_prediction[1]:.2f})"
                            )
                        
                        # Show motion info
                        if len(self.geometric_recognizer.velocity_buffer) > 0:
                            dx, dy = self.geometric_recognizer.velocity_buffer[-1]
                            y_offset = self.draw_debug_info(
                                color_image,
                                y_offset,
                                f"Velocity - dx: {dx:.3f}, dy: {dy:.3f}"
                            )
                    
                    # Draw final prediction (using geometric recognizer)
                    if geometric_prediction.confidence > 0.7:
                        self.draw_debug_info(
                            color_image,
                            color_image.shape[0] - 30,
                            f"Detected: {geometric_prediction.gesture_type.name}",
                            color=(0, 255, 0)
                        )
                    
                    # Draw landmarks and static gestures
                    color_image = self.hand_detector.draw_landmarks(
                        color_image,
                        hands,
                        self.gesture_classifier
                    )
                
                # Show image
                cv2.imshow('Dynamic Gesture Test', color_image)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):  # Toggle debug info
                    self.show_debug_info = not self.show_debug_info
                    
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

def main():
    tester = DynamicGestureTester()
    tester.run()

if __name__ == "__main__":
    main()
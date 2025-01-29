import os
import mediapipe as mp
import tensorflow as tf
import tf2onnx

def create_output_directory():
    """Create directory for output models if it doesn't exist"""
    output_dir = os.path.join(os.path.dirname(__file__), 'exported_models')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def export_hand_detection_model(output_dir):
    """Export the MediaPipe hand detection model to ONNX format"""
    print("Exporting hand detection model...")
    
    hands = mp.solutions.hands
    detection_model = hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7
    )._hand_detection_model
    
    output_path = os.path.join(output_dir, 'hand_detection.onnx')
    
    # Convert to ONNX
    try:
        model_proto, _ = tf2onnx.convert.from_keras(
            detection_model,
            input_signature=(tf.TensorSpec((None, 256, 256, 3), tf.float32, name="input"),),
            opset=13,
            output_path=output_path
        )
        print(f"Hand detection model exported to: {output_path}")
    except Exception as e:
        print(f"Error exporting hand detection model: {str(e)}")
    
def export_hand_landmark_model(output_dir):
    """Export the MediaPipe hand landmark model to ONNX format"""
    print("Exporting hand landmark model...")
    
    # Load MediaPipe hand landmark model
    hands = mp.solutions.hands
    landmark_model = hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7
    )._hand_landmark_model
    
    output_path = os.path.join(output_dir, 'hand_landmark.onnx')
    
    try:
        model_proto, _ = tf2onnx.convert.from_keras(
            landmark_model,
            input_signature=(tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),),
            opset=13,
            output_path=output_path
        )
        print(f"Hand landmark model exported to: {output_path}")
    except Exception as e:
        print(f"Error exporting hand landmark model: {str(e)}")

def main():
    output_dir = create_output_directory()
    print(f"Output directory created at: {output_dir}")
    
    export_hand_detection_model(output_dir)
    export_hand_landmark_model(output_dir)
    
    print("\nModel export complete!")
    print("Please copy the exported models from the 'exported_models' directory")
    print("to your Unity project's 'Assets/Resources' folder.")

if __name__ == "__main__":
    main()
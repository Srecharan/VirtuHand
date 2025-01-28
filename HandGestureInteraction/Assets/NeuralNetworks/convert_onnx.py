import onnx
from onnx import version_converter
import os

def convert_onnx_model(input_path, output_path):
    print(f"Converting {input_path} to {output_path}")
    try:
        # Load the model
        model = onnx.load(input_path)
        
        # Target a higher opset version
        target_opset = 12
        
        # Convert to target opset version
        converted_model = version_converter.convert_version(model, target_opset)
        
        # Print model info before saving
        print(f"Model inputs: {[i.name for i in converted_model.graph.input]}")
        print(f"Model outputs: {[o.name for o in converted_model.graph.output]}")
        
        # Save the converted model
        onnx.save(converted_model, output_path)
        print(f"Successfully converted model to ONNX opset {target_opset}")
    except Exception as e:
        print(f"Error converting model: {str(e)}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define input and output paths
    hand_landmark_input = os.path.join(current_dir, "hand_landmark.onnx")
    palm_detection_input = os.path.join(current_dir, "palm_detection.onnx")
    hand_landmark_output = os.path.join(current_dir, "hand_landmark_converted.onnx")
    palm_detection_output = os.path.join(current_dir, "palm_detection_converted.onnx")
    
    # Convert both models
    convert_onnx_model(hand_landmark_input, hand_landmark_output)
    convert_onnx_model(palm_detection_input, palm_detection_output)
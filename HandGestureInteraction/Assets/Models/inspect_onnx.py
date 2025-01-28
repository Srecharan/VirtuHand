import onnx
import os

def inspect_onnx_model(model_path):
    print(f"\nAnalyzing model: {model_path}")
    try:
        # Load the model
        model = onnx.load(model_path)
        
        print("Model Info:")
        print(f"IR Version: {model.ir_version}")
        print(f"Opset Version: {model.opset_import[0].version}")
        print("\nInputs:")
        for input in model.graph.input:
            print(f"  {input.name}: {[dim.dim_value for dim in input.type.tensor_type.shape.dim]}")
        print("\nOutputs:")
        for output in model.graph.output:
            print(f"  {output.name}: {[dim.dim_value for dim in output.type.tensor_type.shape.dim]}")
    except Exception as e:
        print(f"Error analyzing model: {str(e)}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Analyze original models
    hand_landmark_path = os.path.join(current_dir, "hand_landmark.onnx")
    palm_detection_path = os.path.join(current_dir, "palm_detection.onnx")
    
    inspect_onnx_model(hand_landmark_path)
    inspect_onnx_model(palm_detection_path)
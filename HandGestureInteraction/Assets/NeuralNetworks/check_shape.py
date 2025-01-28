import onnx

def inspect_onnx_model(model_path):
    model = onnx.load(model_path)
    print(f"\nAnalyzing model: {model_path}")
    print("Input shapes:")
    for input in model.graph.input:
        print(f"  {input.name}: {[dim.dim_value for dim in input.type.tensor_type.shape.dim]}")
    print("Output shapes:")
    for output in model.graph.output:
        print(f"  {output.name}: {[dim.dim_value for dim in output.type.tensor_type.shape.dim]}")

# Check both models
inspect_onnx_model('hand_landmark_converted.onnx')
inspect_onnx_model('palm_detection_converted.onnx')
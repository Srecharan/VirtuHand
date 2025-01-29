import mediapipe as mp
import tensorflow as tf
import tf2onnx
import numpy as np
import os

def convert_mediapipe_model():
    mp_hands = mp.solutions.hands
    
    base_path = mp_hands._ROOT
    model_path = os.path.join(base_path, 'hand_landmark_full.tflite')
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    class HandLandmarkModel(tf.Module):
        def __init__(self, interpreter):
            self.interpreter = interpreter
        
        @tf.function(input_signature=[tf.TensorSpec(shape=[1, 224, 224, 3], dtype=tf.float32)])
        def detect_hand(self, input_tensor):
            # Set the input tensor
            self.interpreter.set_tensor(input_details[0]['index'], input_tensor)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get the output tensor
            landmarks = self.interpreter.get_tensor(output_details[0]['index'])
            return landmarks
    
    model = HandLandmarkModel(interpreter)
    
    # Convert to ONNX
    spec = (tf.TensorSpec((1, 224, 224, 3), tf.float32, name="input_1"),)
    output_path = "HandLandmarkDetector.onnx"
    
    model_proto, _ = tf2onnx.convert.from_function(
        model.detect_hand,
        input_signature=spec,
        output_path=output_path,
        opset=13
    )
    
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    convert_mediapipe_model()
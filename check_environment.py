def check_dependencies():
    # Dictionary to store our test results
    status = {}
    
    print("Checking dependencies...")
    
    # Test numpy
    try:
        import numpy as np
        status['numpy'] = f"OK (version {np.__version__})"
    except Exception as e:
        status['numpy'] = f"Failed: {str(e)}"

    # Test OpenCV
    try:
        import cv2
        status['opencv'] = f"OK (version {cv2.__version__})"
    except Exception as e:
        status['opencv'] = f"Failed: {str(e)}"

    # Test MediaPipe
    try:
        import mediapipe as mp
        status['mediapipe'] = f"OK (version {mp.__version__})"
    except Exception as e:
        status['mediapipe'] = f"Failed: {str(e)}"

    # Test TensorFlow
    try:
        import tensorflow as tf
        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        status['tensorflow'] = f"OK (version {tf.__version__}, GPU {'available' if gpu_available else 'not available'})"
    except Exception as e:
        status['tensorflow'] = f"Failed: {str(e)}"

    # Test PyRealSense2
    try:
        import pyrealsense2 as rs
        status['pyrealsense2'] = "OK (version installed)"
    except Exception as e:
        status['pyrealsense2'] = f"Failed: {str(e)}"

    # Print results
    print("\nDependency Check Results:")
    print("-" * 50)
    for package, result in status.items():
        print(f"{package:15} : {result}")

if __name__ == "__main__":
    check_dependencies()
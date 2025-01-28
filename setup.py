from setuptools import setup, find_packages

setup(
    name="hand_gesture_recognition",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "mediapipe",
        "pyrealsense2",
        "tensorflow"  # Changed from tensorflow-gpu
    ]
)
"""
RealSense Camera Interface
-------------------------
This module provides a high-level interface for the Intel RealSense D435i camera,
optimized for hand gesture recognition tasks.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class FrameData:
    """Container for frame data from the RealSense camera."""
    color: np.ndarray
    depth: np.ndarray
    depth_colormap: np.ndarray
    timestamp: float

class RealSenseCamera:
    """Interface for the Intel RealSense D435i camera."""
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        """
        Initialize the RealSense camera with specified parameters.
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Frames per second
        """
        self.width = width
        self.height = height
        self.fps = fps
        
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable streams
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        
        # Align depth to color frame
        self.align = rs.align(rs.stream.color)
        
    def start(self) -> bool:
        """
        Start the camera stream.
        
        Returns:
            bool: True if started successfully
        """
        try:
            self.profile = self.pipeline.start(self.config)
            return True
        except RuntimeError as e:
            print(f"Failed to start camera: {e}")
            return False
            
    def get_frame(self) -> Optional[FrameData]:
        """
        Get the next frame from the camera.
        
        Returns:
            FrameData object containing color and depth frames
        """
        try:
            # Wait for a coherent pair of frames
            frames = self.pipeline.wait_for_frames()
            
            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)
            
            # Get aligned frames
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return None
                
            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Create colormap for visualization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            
            return FrameData(
                color=color_image,
                depth=depth_image,
                depth_colormap=depth_colormap,
                timestamp=frames.get_timestamp()
            )
            
        except RuntimeError as e:
            print(f"Error getting frame: {e}")
            return None
            
    def stop(self):
        """Stop the camera stream."""
        self.pipeline.stop()
        
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

def test_camera():
    """Test function to verify camera operation."""
    with RealSenseCamera() as camera:
        while True:
            frame_data = camera.get_frame()
            if frame_data is None:
                continue
                
            # Show color and depth frames
            cv2.imshow('Color Feed', frame_data.color)
            cv2.imshow('Depth Feed', frame_data.depth_colormap)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera()
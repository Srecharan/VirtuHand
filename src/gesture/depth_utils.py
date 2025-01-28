import numpy as np
import pyrealsense2 as rs

class DepthKalmanFilter:
    def __init__(self):
        # Initialize state (position and velocity)
        self.state = np.array([0.0, 0.0])
        self.P = np.eye(2) * 1000  # High initial uncertainty
        
        # Process noise
        self.Q = np.array([[0.1, 0.0],
                          [0.0, 0.1]])
        
        # Measurement noise
        self.R = np.array([[0.1]])
        
        # Time step
        self.dt = 1.0/30.0  # Assuming 30 FPS
        
    def update(self, measurement):
        # Predict
        F = np.array([[1, self.dt],
                     [0, 1]])
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q
        
        # Update
        H = np.array([[1.0, 0.0]])
        y = measurement - H @ self.state
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        self.P = (np.eye(2) - K @ H) @ self.P
        
        return self.state[0]  # Return filtered position
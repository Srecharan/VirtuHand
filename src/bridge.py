# src/bridge.py

import asyncio
import json
import websockets
import cv2
from src.camera.realsense import RealSenseCamera
from src.gesture.detector import HandDetector
from src.gesture.classifier import GestureClassifier, GestureType

class GestureBridge:
    def __init__(self, host="127.0.0.1", port=8765):  # Changed to explicit IP
        self.host = host
        self.port = port
        self.active_connections = set()
        self.detector = HandDetector()
        self.classifier = GestureClassifier()
        self.running = True
        print("Gesture Bridge initialized")

    # Modify the handle_client function in your bridge.py
    async def handle_client(self, websocket, path):
        """Handle a connected Unity client."""
        print(f"New client attempting to connect from {websocket.remote_address}")
        try:
            self.active_connections.add(websocket)
            print(f"Client successfully connected! Total clients: {len(self.active_connections)}")
            
            # Send an initial test message to confirm connection
            test_message = {"type": "connection_test", "status": "connected"}
            await websocket.send(json.dumps(test_message))
            print("Sent test message to client")
            
            while self.running:
                try:
                    message = await websocket.recv()
                    print(f"Received from Unity: {message}")
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed by client")
                    break
        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            self.active_connections.remove(websocket)
            print("Client disconnected")

    async def broadcast_gesture(self, gesture_result):
        """Send gesture data to Unity."""
        if not self.active_connections:
            return

        # Format gesture data for Unity
        message = {
            "gesture_type": gesture_result.gesture_type.name,
            "confidence": float(gesture_result.confidence),
            "hand_side": gesture_result.hand_side,
            "palm_position": {
                "x": float(gesture_result.palm_position[0]),
                "y": float(gesture_result.palm_position[1]),
                "z": float(gesture_result.palm_position[2])
            }
        }

        # Send to all connected clients
        disconnected = set()
        for websocket in self.active_connections:
            try:
                await websocket.send(json.dumps(message))
                print(f"Sent gesture: {gesture_result.gesture_type.name}")
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(websocket)
        
        # Remove disconnected clients
        self.active_connections -= disconnected

    async def process_frames(self):
        """Process camera frames and detect gestures."""
        try:
            with RealSenseCamera() as camera:
                print("Camera initialized")
                
                while self.running:
                    frame_data = camera.get_frame()
                    if frame_data is None:
                        continue

                    try:
                        # Detect and process hands
                        hands = self.detector.detect_hands(frame_data.color, frame_data.depth)
                        
                        for hand in hands:
                            gesture_result = self.classifier.classify_gesture(hand)
                            if gesture_result.gesture_type != GestureType.NONE:
                                await self.broadcast_gesture(gesture_result)
                        
                        # Show visualization
                        annotated_image = self.detector.draw_landmarks(
                            frame_data.color, hands, self.classifier
                        )
                        cv2.imshow('Hand Tracking', annotated_image)
                        
                    except Exception as e:
                        print(f"Frame processing error: {e}")
                        continue

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break
                        
        except Exception as e:
            print(f"Camera error: {e}")
        finally:
            cv2.destroyAllWindows()

    # In src/bridge.py, update the run method
    async def run(self):
        """Start the WebSocket server and process frames."""
        server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ping_interval=None  # Disable ping-pong messages
        )
        print(f"WebSocket server running on ws://{self.host}:{self.port}")
        
        # Instead of TaskGroup, we'll use asyncio.gather
        try:
            await asyncio.gather(
                server.wait_closed(),
                self.process_frames()
            )
        except Exception as e:
            print(f"Error in server: {e}")

# Also update the main function
def main():
    bridge = GestureBridge()
    try:
        # Create and run an event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(bridge.run())
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cv2.destroyAllWindows()
        if loop.is_running():
            loop.close()

if __name__ == "__main__":
    main()
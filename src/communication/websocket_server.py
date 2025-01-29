# src/communication/websocket_server.py

import asyncio
import json
import websockets
from dataclasses import asdict
from typing import Optional, Dict, Any

class GestureWebSocketServer:
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.server: Optional[websockets.serve] = None
        
    async def send_gesture_data(self, websocket, gesture_result):
        """Send gesture data to Unity client."""
        gesture_data = {
            "gesture_type": gesture_result.gesture_type.name,
            "confidence": gesture_result.confidence,
            "hand_side": gesture_result.hand_side,
            "palm_position": {
                "x": gesture_result.palm_position[0],
                "y": gesture_result.palm_position[1],
                "z": gesture_result.palm_position[2]
            }
        }
        
        try:
            await websocket.send(json.dumps(gesture_data))
        except websockets.exceptions.ConnectionClosed:
            print("Client connection closed")
            
    async def handle_client(self, websocket, path):
        """Handle incoming WebSocket connections."""
        print(f"New client connected from {websocket.remote_address}")
        try:
            while True:
                message = await websocket.recv()
                print(f"Received message from Unity: {message}")
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")
            
    async def start(self):
        """Start the WebSocket server."""
        self.server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port
        )
        print(f"WebSocket server started on ws://{self.host}:{self.port}")
        await self.server.wait_closed()
        
    def run(self):
        """Run the server in the event loop."""
        asyncio.get_event_loop().run_until_complete(self.start())
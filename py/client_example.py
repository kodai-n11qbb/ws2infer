#!/usr/bin/env python3
"""
Example WebSocket client for ws2infer server
"""

import asyncio
import json
import websockets
import time


async def test_client():
    """Test client for ws2infer server"""
    uri = "ws://localhost:8765"
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to {uri}")
            
            # Test offline mode
            offline_request = {
                "mode": "offline",
                "data": {
                    "text": "Hello from offline mode!"
                },
                "request_id": "offline_test_001"
            }
            
            print("Sending offline request...")
            await websocket.send(json.dumps(offline_request))
            response = await websocket.recv()
            print(f"Offline response: {json.dumps(json.loads(response), indent=2)}")
            
            # Test online mode
            online_request = {
                "mode": "online",
                "data": {
                    "text": "Hello from online mode!"
                },
                "request_id": "online_test_001"
            }
            
            print("\nSending online request...")
            await websocket.send(json.dumps(online_request))
            response = await websocket.recv()
            print(f"Online response: {json.dumps(json.loads(response), indent=2)}")
            
            # Test with image data (base64 encoded)
            import base64
            test_image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
            
            image_request = {
                "mode": "offline",
                "data": {
                    "image": test_image_data,
                    "format": "png"
                },
                "request_id": "image_test_001"
            }
            
            print("\nSending image request...")
            await websocket.send(json.dumps(image_request))
            response = await websocket.recv()
            print(f"Image response: {json.dumps(json.loads(response), indent=2)}")
            
    except ConnectionRefusedError:
        print("Connection refused. Make sure the server is running on ws://localhost:8765")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_client())

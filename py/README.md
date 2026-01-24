# ws2infer Python Implementation

WebSocket-based AI Inference Server supporting both online and offline modes.

## Architecture

Based on the system flowchart in `../mmds/system.mmd`:

```
Client -> WebSocket Handler -> Mode Selection -> Inference -> Response Delivery
```

## Features

- **WebSocket Server**: Handles ws:// and wss:// connections
- **Dual Mode Operation**:
  - **Online Mode**: Uses remote AI APIs (cloud services)
  - **Offline Mode**: Uses local models for inference
- **Flexible Data Handling**: Supports JSON, images, and other data formats
- **Async Processing**: Built on asyncio for high performance

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Start the server

```bash
python ws2infer_server.py --host localhost --port 8765
```

### Client Example

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8765');

// Send inference request (offline mode)
ws.send(JSON.stringify({
    mode: 'offline',
    data: {
        text: 'Hello, world!'
    },
    request_id: 'req_001'
}));

// Send inference request (online mode)
ws.send(JSON.stringify({
    mode: 'online',
    data: {
        text: 'Hello, world!'
    },
    request_id: 'req_002'
}));
```

## API Response Format

```json
{
    "request_id": "req_001",
    "success": true,
    "data": {
        "result": "processed result",
        "model": "local_model",
        "processing_time": 0.1
    },
    "error": null,
    "timestamp": 1640995200.0
}
```

## Configuration

### Local Models (Offline Mode)

The server supports loading local models in various formats:
- ONNX models
- PyTorch models
- TensorFlow models

### Remote APIs (Online Mode)

Configure remote API endpoints for cloud-based inference services.

## Development

This implementation is designed to be easily ported to other languages. The core logic is separated into:

- `WS2InferServer`: Main WebSocket server
- `LocalModelInference`: Offline mode handler
- `RemoteAPIInference`: Online mode handler

Future language implementations should follow the same architecture.

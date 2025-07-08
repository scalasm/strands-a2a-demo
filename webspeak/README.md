# WebSpeak - Web-based Voice Interface for A2A

A simplified web-based voice interface that provides the same functionality as the CLI speak mode through a web browser.

## Key Features

- **Same functionality as CLI speak mode** - Uses the existing `BedrockStreamManager` from `speak_interface.py`
- **WebRTC audio processing** - Client-side echo cancellation, noise suppression, auto-gain control
- **Real-time communication** - SocketIO for bidirectional communication
- **Tool use support** - Automatic `send_task` and `get_task_results` tools via Nova Sonic
- **Task completion notifications** - Audio notifications via Amazon Polly when tasks complete
- **Visual feedback** - Text output and tool usage displayed in the browser

## Architecture

```
Browser (WebRTC) ←→ SocketIO ←→ Python (BedrockStreamManager) ←→ Nova Sonic ←→ A2A Agents
```

### Audio Flow
1. **Input**: Browser microphone → WebRTC processing → Base64 → SocketIO → Python → Nova Sonic
2. **Output**: Nova Sonic → Python → Base64 → SocketIO → Browser → Audio playback

### Tool Integration
- Nova Sonic automatically gets `send_task` and `get_task_results` tools
- Task completion triggers Polly speech: "A task has completed. Get the results."
- Web interface shows tool usage and text output in real-time

## Usage

Start the WebSpeak server:
```bash
uv run a2a-client webspeak --agent calculator --port 3000
```

Then open your browser to `http://localhost:3000` and:
1. Select an agent
2. Click "Connect" 
3. Click "Start Conversation"
4. Speak to interact with the agent

## Implementation Details

### Reuses Existing Code
- **BedrockStreamManager**: Same class from `speak_interface.py` handles all Nova Sonic integration
- **A2A Client**: Same `CleanA2AClient` for agent communication
- **Webhook Manager**: Same push notification system for task completion
- **Tool Processing**: Same `send_a2a_task` and `get_all_task_results` methods

### Web-Specific Additions
- **SocketIO events**: `audio_chunk`, `audio_response`, `text_output`, `tool_use`
- **WebRTC audio**: Client-side audio processing instead of PyAudio/LiveKit
- **Real-time UI**: Shows conversation text and tool usage as it happens

## Files

- `templates/index.html` - Simple web interface
- `static/main.js` - WebRTC audio processing and SocketIO communication  
- `static/style.css` - Basic styling
- `../webspeak_interface.py` - Flask-SocketIO server that reuses `speak_interface.py`

## Benefits

- **No backend audio dependencies** - WebRTC handles all audio processing
- **Remote access** - Use from any modern web browser
- **Same reliability** - Reuses proven CLI speak interface code
- **Visual feedback** - See what's happening during conversations 
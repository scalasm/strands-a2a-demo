#!/usr/bin/env python3
"""
WebSpeak Interface - Simplified web-based voice interface for A2A clients

Simple web interface using WebRTC for client-side audio processing and 
SocketIO for real-time communication with Nova Sonic and A2A agents.
"""

import asyncio
import logging
import json
import secrets
from pathlib import Path
from typing import Optional, Dict, Any
import os
import base64

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import tomllib

# Import existing components
from speak_interface import BedrockStreamManager
from base_agent import configure_logging
from a2a_push_notification_manager import get_client_webhook_manager
from a2a_client import CleanA2AClient, discover_available_agents
from log_viewer import LogViewer, AgentLogInfo

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# Flask app and SocketIO
app = Flask(__name__, 
            template_folder='webspeak/templates',
            static_folder='webspeak/static')
app.config['SECRET_KEY'] = secrets.token_hex(16)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
sessions = {}
config = None

# Single shared event loop for all async operations
_shared_loop = None
_shared_thread = None

def get_shared_loop():
    """Get the single shared event loop."""
    global _shared_loop, _shared_thread
    
    if _shared_loop is None or _shared_loop.is_closed():
        import threading
        
        def run_shared_loop():
            global _shared_loop
            _shared_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(_shared_loop)
            _shared_loop.run_forever()
        
        _shared_thread = threading.Thread(target=run_shared_loop, daemon=True)
        _shared_thread.start()
        
        # Wait for loop to be ready
        import time
        while _shared_loop is None:
            time.sleep(0.001)
    
    return _shared_loop

def run_in_shared_loop(coro):
    """Run coroutine in the shared event loop."""
    loop = get_shared_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


def load_config():
    """Load configuration from config/config.toml."""
    global config
    config_path = Path(__file__).parent / "config" / "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    return config


class WebSpeakSession:
    """Manages a user session for the WebSpeak interface."""
    
    def __init__(self, sid: str):
        self.sid = sid
        self.a2a_client: Optional[CleanA2AClient] = None
        self.stream_manager: Optional[BedrockStreamManager] = None
        self.is_streaming = False
        self.response_handler_task: Optional[asyncio.Task] = None
        
    async def initialize(self, agent_name: str):
        """Initialize the session with a specific agent."""
        if not config:
            load_config()
            
        # Get agent URL from dynamic discovery
        agents_config = discover_available_agents()
        agent_url = agents_config.get(agent_name)
        if not agent_url:
            available = list(agents_config.keys())
            raise ValueError(f"Agent '{agent_name}' not found. Available: {', '.join(available)}")
            
        self.a2a_client = CleanA2AClient(agent_url, agent_name)
        await self.a2a_client.discover_agent_skills()
            
        self.stream_manager = BedrockStreamManager(a2a_client=self.a2a_client)
        await self.stream_manager.initialize_stream()
        
        self.response_handler_task = asyncio.create_task(self._handle_responses())
            
        return {
            'name': self.a2a_client.agent_card.name,
            'description': self.a2a_client.agent_card.description
        }

    async def process_audio_chunk(self, audio_data: str):
        """Process an incoming audio chunk from base64 string."""
        if self.is_streaming and self.stream_manager and audio_data:
            try:
                # Decode base64 audio data
                audio_bytes = base64.b64decode(audio_data)
                self.stream_manager.add_audio_chunk(audio_bytes)
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}")
    
    async def start_audio(self):
        """Start audio streaming."""
        if not self.stream_manager:
            raise Exception("No stream manager initialized")
        
        # Send audio content start event
        await self.stream_manager.send_audio_content_start_event()
        self.is_streaming = True
        
        # Notify client
        socketio.emit('audio_started', room=self.sid)
        
    async def stop_audio(self):
        """Stop audio streaming."""
        if self.stream_manager and self.is_streaming:
            await self.stream_manager.send_audio_content_end_event()
            self.is_streaming = False
            
            # Notify client
            socketio.emit('audio_stopped', room=self.sid)
    
    async def _handle_responses(self):
        """Handle all responses from the Bedrock stream."""
        last_barge_in_state = False  # Track barge-in state changes
        
        while self.stream_manager and self.stream_manager.is_active:
            try:
                # Check if barge-in state changed and notify frontend
                if self.stream_manager.barge_in != last_barge_in_state:
                    if self.stream_manager.barge_in:
                        # Notify frontend to clear audio buffers immediately
                        socketio.emit('barge_in', room=self.sid)
                        logger.debug("Barge-in detected - notified frontend to clear audio buffers")
                    last_barge_in_state = self.stream_manager.barge_in
                
                # Handle audio
                try:
                    audio_chunk = await asyncio.wait_for(self.stream_manager.audio_output_queue.get(), 0.01)
                    if audio_chunk and self.is_streaming and not self.stream_manager.barge_in:
                        socketio.emit('audio_response', {'audio': base64.b64encode(audio_chunk).decode()}, room=self.sid)
                except asyncio.TimeoutError:
                    pass

                # Handle text and other events
                try:
                    event_data = await asyncio.wait_for(self.stream_manager.output_queue.get(), 0.01)
                    
                    if event_data.get('type') == 'text':
                        # Handle text output
                        text_content = event_data.get('content', '')
                        role = event_data.get('role', 'assistant')
                        socketio.emit('text_output', {
                            'role': role,
                            'content': text_content
                        }, room=self.sid)
                    
                    elif event_data.get('type') == 'tool_use':
                        # Handle tool use
                        tool_name = event_data.get('name', 'Unknown Tool')
                        tool_input = event_data.get('input', {})
                        socketio.emit('tool_use', {
                            'message': f"Using tool: {tool_name} with input: {tool_input}"
                        }, room=self.sid)
                    
                    elif event_data.get('type') == 'content_end':
                        # Handle content end events
                        socketio.emit('content_end', event_data, room=self.sid)
                    
                    # Handle contentEnd events from Bedrock stream
                    elif 'event' in event_data and 'contentEnd' in event_data['event']:
                        content_end = event_data['event']['contentEnd']
                        content_end_data = {
                            'type': content_end.get('type', 'TEXT'),
                            'stopReason': content_end.get('stopReason', ''),
                            'promptName': content_end.get('promptName', ''),
                            'contentName': content_end.get('contentName', '')
                        }
                        socketio.emit('content_end', content_end_data, room=self.sid)
                        
                        # Additional barge-in trigger for interrupted content
                        if content_end.get('stopReason', '').upper() == 'INTERRUPTED':
                            logger.debug("ContentEnd INTERRUPTED detected - ensuring barge-in is triggered")
                            socketio.emit('barge_in', room=self.sid)
                        
                except asyncio.TimeoutError:
                    pass
                        
            except Exception as e:
                logger.error(f"Error in response handler for {self.sid}: {e}")
                break
    
    async def close(self):
        """Clean up the session."""
        if self.response_handler_task:
            self.response_handler_task.cancel()
        if self.stream_manager:
            await self.stream_manager.close()
        if self.a2a_client:
            await self.a2a_client.close()
        logger.info(f"Session {self.sid} cleaned up.")


# Flask routes
@app.route('/')
def index():
    """Serve the main interface."""
    return render_template('index.html')


@app.route('/api/agents')
def get_agents():
    """Get available agents."""
    agents_config = discover_available_agents()
    return jsonify({
        'agents': {name: {'url': url} for name, url in agents_config.items()}
    })


@app.route('/api/logs')
def get_agent_logs():
    """Get logs for all agents."""
    try:
        lines = request.args.get('lines', 20, type=int)
        log_viewer = LogViewer()
        agent_logs = log_viewer.get_all_agent_logs(lines_per_agent=lines)
        
        # Convert to JSON-serializable format
        result = []
        for log_info in agent_logs:
            result.append({
                'name': log_info.name,
                'log_file': log_info.log_file,
                'status': log_info.status,
                'pid': log_info.pid,
                'port': log_info.port,
                'recent_logs': [
                    {
                        'timestamp': entry.timestamp,
                        'level': entry.level,
                        'message': entry.message,
                        'raw_line': entry.raw_line
                    }
                    for entry in log_info.recent_logs
                ],
                'last_updated': log_info.last_updated
            })
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting agent logs: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/logs/<agent_name>')
def get_agent_log(agent_name):
    """Get logs for a specific agent."""
    try:
        lines = request.args.get('lines', 50, type=int)
        log_viewer = LogViewer()
        log_info = log_viewer.get_agent_log(agent_name, lines)
        
        if not log_info:
            return jsonify({'error': f'Agent {agent_name} not found'}), 404
        
        result = {
            'name': log_info.name,
            'log_file': log_info.log_file,
            'status': log_info.status,
            'pid': log_info.pid,
            'port': log_info.port,
            'recent_logs': [
                {
                    'timestamp': entry.timestamp,
                    'level': entry.level,
                    'message': entry.message,
                    'raw_line': entry.raw_line
                }
                for entry in log_info.recent_logs
            ],
            'last_updated': log_info.last_updated
        }
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting logs for agent {agent_name}: {e}")
        return jsonify({'error': str(e)}), 500


# SocketIO events
@socketio.on('connect')
def on_connect():
    sid = request.sid
    sessions[sid] = WebSpeakSession(sid)
    logger.info(f"Client connected: {sid}")


@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    if sid in sessions:
        run_in_shared_loop(sessions[sid].close())
        del sessions[sid]
    logger.info(f"Client disconnected: {sid}")


@socketio.on('initialize_agent')
def on_initialize_agent(data):
    sid = request.sid
    agent_name = data.get('agent_name')
    async def _init():
        try:
            agent_info = await sessions[sid].initialize(agent_name)
            emit('agent_initialized', agent_info, room=sid)
        except Exception as e:
            emit('initialization_error', {'error': str(e)}, room=sid)
    asyncio.run_coroutine_threadsafe(_init(), get_shared_loop())


@socketio.on('start_audio')
def handle_start_audio():
    """Handle start audio streaming."""
    session_id = request.sid
    
    if session_id not in sessions:
        emit('error', {'message': 'Session not found'})
        return
    
    def start_audio():
        try:
            session = sessions[session_id]
            run_in_shared_loop(session.start_audio())
        except Exception as e:
            socketio.emit('error', {'message': str(e)}, room=session_id)
    
    # Run in background thread
    import threading
    thread = threading.Thread(target=start_audio)
    thread.start()


@socketio.on('stop_audio')
def handle_stop_audio():
    """Handle stop audio streaming."""
    session_id = request.sid
    
    if session_id not in sessions:
        emit('error', {'message': 'Session not found'})
        return
    
    def stop_audio():
        try:
            session = sessions[session_id]
            run_in_shared_loop(session.stop_audio())
        except Exception as e:
            socketio.emit('error', {'message': str(e)}, room=session_id)
    
    # Run in background thread
    import threading
    thread = threading.Thread(target=stop_audio)
    thread.start()


@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """Handle incoming audio chunk."""
    session_id = request.sid
    audio_data = data.get('audio')
    
    if session_id not in sessions or not audio_data:
        return
    
    def process_audio():
        try:
            session = sessions[session_id]
            run_in_shared_loop(session.process_audio_chunk(audio_data))
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
    
    # Run in background thread
    import threading
    thread = threading.Thread(target=process_audio)
    thread.start()


def create_app():
    """Create and return the Flask app and SocketIO instance."""
    return app, socketio


# Async mode functions
async def webspeak_mode(agent_filter='all', port=None, host=None):
    """WebSpeak mode - Web-based voice interface."""
    try:
        # Load config to get default port and host
        config = load_config()
        if port is None:
            port = config.get("web_interface", {}).get("webspeak_port", 8081)
        if host is None:
            host = config.get("web_interface", {}).get("webspeak_host", "0.0.0.0")
        
        # Display user-friendly host for logging
        display_host = "localhost" if host in ["0.0.0.0", "127.0.0.1"] else host
        
        logger.info(f"🌐 Starting WebSpeak server on http://{display_host}:{port}")
        logger.info("🎤 WebSpeak provides web-based voice interface with WebRTC audio processing")
        logger.info("📱 Access the interface in your web browser")
        
        # Start server
        socketio.run(app, host=host, port=port, debug=False)
        
    except KeyboardInterrupt:
        logger.info("👋 WebSpeak server interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error running WebSpeak server: {e}")
    finally:
        # Cleanup all sessions
        for session in sessions.values():
            try:
                await session.close()
            except Exception as e:
                logger.debug(f"Error cleaning up session: {e}")


def main():
    """Main entry point for standalone webspeak command."""
    import argparse
    
    # Load config to get defaults
    try:
        config = load_config()
        web_config = config.get("web_interface", {})
        default_port = web_config.get("webspeak_port", 8081)
        default_host = web_config.get("webspeak_host", "0.0.0.0")
    except Exception:
        default_port = 8081
        default_host = "0.0.0.0"
    
    parser = argparse.ArgumentParser(description='WebSpeak - Web-based voice interface for A2A')
    parser.add_argument('--host', default=default_host, help=f'Host to bind to (default: {default_host})')
    parser.add_argument('--port', type=int, default=default_port, help=f'Port to bind to (default: {default_port})')
    
    args = parser.parse_args()
    
    asyncio.run(webspeak_mode(host=args.host, port=args.port))


if __name__ == '__main__':
    main()
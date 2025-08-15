#!/usr/bin/env python3
"""
Web Interface for A2A Client

A modern web interface that provides all the functionalities of the interactive mode
with real-time, non-blocking communication via WebSockets.
"""

import asyncio
import json
import logging
import tomllib
from pathlib import Path
from typing import Dict, List, Optional, Set
from uuid import uuid4
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from a2a.types import Task
from a2a_client import CleanA2AClient, discover_available_agents, get_agent_info_summary, setup_signal_handlers, _cleanup_in_progress
from a2a_push_notification_manager import get_client_webhook_manager
from model_config import configure_logging
from log_viewer import LogViewer, AgentLogInfo

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="A2A Web Interface",
    description="Modern web interface for A2A agent communication",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates")

# Global state
agents_cache: Dict[str, CleanA2AClient] = {}
webhook_manager = None
active_connections: Dict[str, WebSocket] = {}
_final_address = None


class AgentInfo(BaseModel):
    name: str
    url: str
    description: str = ""
    skills: List[Dict] = []
    connected: bool = False


class ChatMessage(BaseModel):
    type: str  # 'user', 'agent', 'system', 'task_status'
    agent: Optional[str] = None
    content: str
    task_id: Optional[str] = None
    timestamp: str


class TaskRequest(BaseModel):
    agent: str
    message: str
    use_tasks: bool = True


class LogEntryModel(BaseModel):
    timestamp: str
    level: str
    message: str
    raw_line: str


class AgentLogModel(BaseModel):
    name: str
    log_file: str
    status: str
    pid: Optional[int]
    port: int
    recent_logs: List[LogEntryModel]
    last_updated: Optional[str]


# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket connected: {client_id}")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"WebSocket disconnected: {client_id}")

    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected = []
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                disconnected.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected:
            self.disconnect(client_id)


manager = ConnectionManager()


async def initialize_agents():
    """Initialize all agents on startup."""
    global agents_cache, webhook_manager
    
    if agents_cache:
        return  # Already initialized
    
    logger.info("Initializing agents...")
    agents_config = discover_available_agents()
    
    for name, url in agents_config.items():
            try:
                client = CleanA2AClient(url, name)
                if await client.discover_agent_skills():
                    agents_cache[name] = client
                    logger.info(f"✅ Connected to {name}")
                else:
                    logger.warning(f"⚠️  {name}: Connected but skills discovery failed")
                    agents_cache[name] = client
            except Exception as e:
                logger.error(f"❌ Failed to connect to {name}: {e}")
        
    # Initialize webhook manager
    webhook_manager = get_client_webhook_manager()
    await webhook_manager.start()
    logger.info("🔔 Webhook manager started")
    
    # Show the final listening address when everything is ready
    global _final_address
    if _final_address:
        print()  # Add blank line for separation
        print("=" * 60)
        print(f"🌐 A2A Web Interface is ready and listening on {_final_address}")
        print("=" * 60)
    else:
        # Fallback: read from config
        try:
            import tomllib
            from pathlib import Path
            config_path = Path(__file__).parent / "config" / "config.toml"
            with open(config_path, "rb") as f:
                config = tomllib.load(f)
            web_config = config.get("web_interface", {})
            host = web_config.get("host", "0.0.0.0")
            port = web_config.get("port", 8080)
            print()  # Add blank line for separation
            print("=" * 60)
            print(f"🌐 A2A Web Interface is ready and listening on http://{host}:{port}")
            print("=" * 60)
        except Exception:
            print("🌐 A2A Web Interface is ready and listening")


async def cleanup_agents():
    """Clean up agents on shutdown."""
    global agents_cache, webhook_manager
    
    logger.info("Cleaning up agents...")
    for client in agents_cache.values():
        await client.close()
    agents_cache.clear()
    
    if webhook_manager:
        await webhook_manager.stop()
        webhook_manager = None
    
    logger.info("✅ Cleanup complete")


@app.on_event("startup")
async def startup_event():
    """Initialize agents on startup"""
    await initialize_agents()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await cleanup_agents()


@app.get("/", response_class=HTMLResponse)
async def get_web_interface(request: Request):
    """Serve the main web interface"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/agents", response_model=List[AgentInfo])
async def get_agents():
    """Get list of available agents with their information"""
    agent_list = []
    
    for name, client in agents_cache.items():
        agent_info = AgentInfo(
            name=name,
            url=client.agent_url,
            connected=True
        )
        
        if client.agent_card:
            agent_info.description = client.agent_card.description or ""
            if client.agent_card.skills:
                agent_info.skills = [
                    {
                        "name": skill.name,
                        "description": skill.description,
                        "examples": skill.examples[:2] if skill.examples else []
                    }
                    for skill in client.agent_card.skills
                ]
        
        agent_list.append(agent_info)
    
    return agent_list


@app.get("/api/logs", response_model=List[AgentLogModel])
async def get_agent_logs(lines: int = 20):
    """Get logs for all agents"""
    try:
        log_viewer = LogViewer()
        agent_logs = log_viewer.get_all_agent_logs(lines_per_agent=lines)
        
        # Convert to Pydantic models
        result = []
        for log_info in agent_logs:
            log_entries = [
                LogEntryModel(
                    timestamp=entry.timestamp,
                    level=entry.level,
                    message=entry.message,
                    raw_line=entry.raw_line
                )
                for entry in log_info.recent_logs
            ]
            
            result.append(AgentLogModel(
                name=log_info.name,
                log_file=log_info.log_file,
                status=log_info.status,
                pid=log_info.pid,
                port=log_info.port,
                recent_logs=log_entries,
                last_updated=log_info.last_updated
            ))
        
        return result
    except Exception as e:
        logger.error(f"Error getting agent logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/logs/{agent_name}", response_model=AgentLogModel)
async def get_agent_log(agent_name: str, lines: int = 50):
    """Get logs for a specific agent"""
    try:
        log_viewer = LogViewer()
        log_info = log_viewer.get_agent_log(agent_name, lines)
        
        if not log_info:
            raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
        
        log_entries = [
            LogEntryModel(
                timestamp=entry.timestamp,
                level=entry.level,
                message=entry.message,
                raw_line=entry.raw_line
            )
            for entry in log_info.recent_logs
        ]
        
        return AgentLogModel(
            name=log_info.name,
            log_file=log_info.log_file,
            status=log_info.status,
            pid=log_info.pid,
            port=log_info.port,
            recent_logs=log_entries,
            last_updated=log_info.last_updated
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting logs for agent {agent_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    client_id = str(uuid4())
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "get_agents":
                # Send agent list
                agents_data = await get_agents()
                await manager.send_personal_message({
                    "type": "agents_list",
                    "agents": [agent.dict() for agent in agents_data]
                }, client_id)
                
            elif message["type"] == "send_message":
                agent_name = message["agent"]
                user_message = message["message"]
                use_tasks = message.get("use_tasks", True)
                
                if agent_name not in agents_cache:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": f"Agent {agent_name} not available"
                    }, client_id)
                    continue
                
                client = agents_cache[agent_name]
                
                try:
                    if use_tasks:
                        # Use task mode with callback
                        def task_callback(task: Task):
                            result = client.extract_task_result(task)
                            # Send task completion via WebSocket
                            asyncio.create_task(manager.send_personal_message({
                                "type": "task_completed",
                                "agent": agent_name,
                                "result": result,
                                "task_id": task.id
                            }, client_id))
                        
                        task_id = await client.send_task(user_message, task_callback)
                        
                        if task_id == "immediate-response":
                            # Immediate response was already handled by callback
                            pass
                        else:
                            # Notify that task was created
                            await manager.send_personal_message({
                                "type": "task_created",
                                "task_id": task_id,
                                "agent": agent_name
                            }, client_id)
                    else:
                        # Use message mode
                        result = await client.send_message(user_message)
                        await manager.send_personal_message({
                            "type": "message",
                            "message_type": "agent",
                            "agent": agent_name,
                            "content": result,
                            "timestamp": str(uuid4())  # Will be replaced on frontend
                        }, client_id)
                        
                except Exception as e:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": str(e)
                    }, client_id)
                    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        manager.disconnect(client_id)


async def web_mode(port: Optional[int] = None, host: Optional[str] = None):
    """Start the web interface server."""
    if port is None or host is None:
        # Load web interface config
        try:
            import tomllib
            from pathlib import Path
            config_path = Path(__file__).parent / "config" / "config.toml"
            with open(config_path, "rb") as f:
                config = tomllib.load(f)
            web_config = config.get("web_interface", {})
            if port is None:
                port = web_config.get("port", 8080)
            if host is None:
                host = web_config.get("host", "0.0.0.0")
        except Exception:
            port = port or 8080
            host = host or "0.0.0.0"
    
    import uvicorn
    logger.info(f"🚀 Starting A2A Web Interface...")
    
    # Store the address for the startup message
    global _final_address
    _final_address = f"http://{host}:{port}"
    
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        access_log=False,  # Disable access logs for cleaner output
        log_level="warning"  # Reduce uvicorn startup noise
    )


if __name__ == "__main__":
    # Load web interface config
    try:
        import tomllib
        from pathlib import Path
        config_path = Path(__file__).parent / "config" / "config.toml"
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        web_config = config.get("web_interface", {})
        port = web_config.get("port", 8080)
        host = web_config.get("host", "0.0.0.0")
    except Exception:
        port = 8080
        host = "0.0.0.0"
    
    import uvicorn
    uvicorn.run(app, host=host, port=port) 
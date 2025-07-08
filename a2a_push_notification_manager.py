"""
A2A Push Notification Manager - Unified Implementation

This module provides a unified webhook implementation for both:
- Client-to-agent communication
- Agent-to-agent communication

Uses official A2A SDK patterns and can be used by any A2A participant.
"""
import asyncio
import logging
import threading
from typing import Callable, Optional, Dict, Any, Union
from contextlib import asynccontextmanager

import httpx
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.requests import Request
import uvicorn

from a2a.types import (
    Task,
    TaskState,
    PushNotificationConfig,
    TaskPushNotificationConfig,
    SetTaskPushNotificationConfigRequest,
    GetTaskPushNotificationConfigRequest,
)
from a2a.client import A2AClient


logger = logging.getLogger(__name__)


class A2AWebhookManager:
    """
    Unified A2A Webhook Manager for both clients and agents.
    
    This manager:
    - Handles webhook server for receiving Task updates
    - Works for both client-to-agent and agent-to-agent communication
    - Uses official A2A client methods for push notifications
    - Provides callback registration for task completion
    - Follows A2A specification patterns
    """
    
    # Class-level registry for different webhook instances
    _instances: Dict[int, 'A2AWebhookManager'] = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, webhook_port: int, webhook_host: str = "localhost", webhook_path: str = "/webhook/a2a-tasks") -> 'A2AWebhookManager':
        """Factory method to ensure one webhook manager per port."""
        with cls._lock:
            if webhook_port not in cls._instances:
                cls._instances[webhook_port] = cls(webhook_port, webhook_host, webhook_path)
            return cls._instances[webhook_port]
    
    def __init__(self, webhook_port: int, webhook_host: str = "localhost", webhook_path: str = "/webhook/a2a-tasks"):
        """
        Initialize the A2A Webhook Manager.
        
        Args:
            webhook_port: Port for the webhook server
            webhook_host: Host for the webhook server
            webhook_path: Path for the webhook endpoint
        """
        if hasattr(self, '_initialized'):
            return
            
        self.webhook_port = webhook_port
        self.webhook_host = webhook_host
        self.webhook_path = webhook_path
        self.webhook_url = f"http://{webhook_host}:{webhook_port}{webhook_path}"
        
        # Task tracking with callbacks
        self.task_callbacks: Dict[str, Callable[[Task], Any]] = {}
        self.task_timeouts: Dict[str, float] = {}
        self.task_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Server management
        self.server_task: Optional[asyncio.Task] = None
        self.server_started = asyncio.Event()
        self.server_instance: Optional[uvicorn.Server] = None
        
        self._initialized = True
        logger.info(f"A2A Webhook Manager initialized - webhook: {self.webhook_url}")
    
    async def start(self):
        """Start the webhook server if not already running."""
        if self.server_task and not self.server_task.done():
            logger.debug("Webhook server already running")
            await self.server_started.wait()
            return
            
        logger.info(f"Starting webhook server on {self.webhook_host}:{self.webhook_port}")
        
        # Create Starlette app with proper A2A webhook handling
        app = Starlette(
            routes=[
                Route(self.webhook_path, self._handle_task_update, methods=["POST"]),
                Route("/health", self._health_check, methods=["GET"]),
            ]
        )
        
        # Start server in background
        config = uvicorn.Config(
            app, 
            host=self.webhook_host, 
            port=self.webhook_port, 
            log_level="warning"
        )
        self.server_instance = uvicorn.Server(config)
        
        self.server_task = asyncio.create_task(self._run_server())
        
        # Wait for server to start
        try:
            await asyncio.wait_for(self.server_started.wait(), timeout=10.0)
            logger.info("Webhook server started successfully")
        except asyncio.TimeoutError:
            logger.error("Webhook server failed to start within timeout.")
            self.server_task.cancel()
            raise
    
    async def _run_server(self):
        """Run the uvicorn server."""
        try:
            # Create a task to signal when server is ready
            ready_task = asyncio.create_task(self._wait_for_server_ready())
            serve_task = asyncio.create_task(self.server_instance.serve())
            
            # Start both tasks
            done, pending = await asyncio.wait(
                [ready_task, serve_task], 
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # If ready task completed first, server is ready
            if ready_task in done:
                self.server_started.set()
                logger.debug("Server ready signal sent")
            
            # Wait for serve task to complete
            await serve_task
            
        except asyncio.CancelledError:
            logger.debug("Webhook server task cancelled")
            raise
        finally:
            # Ensure waiter is released on exit
            if not self.server_started.is_set():
                self.server_started.set()
    
    async def _wait_for_server_ready(self):
        """Wait for server to be ready by checking if it responds to requests."""
        max_attempts = 50
        for attempt in range(max_attempts):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"http://{self.webhook_host}:{self.webhook_port}/health", timeout=0.1)
                    if response.status_code == 200:
                        return True
            except (httpx.RequestError, httpx.TimeoutException):
                pass
            await asyncio.sleep(0.1)
        return False
    
    async def _handle_task_update(self, request: Request) -> JSONResponse:
        """
        Handle incoming A2A task updates.
        
        The A2A server sends the complete Task object as JSON,
        so we can directly process it.
        """
        try:
            # Parse the Task object sent by A2A server
            task_data = await request.json()
            task = Task.model_validate(task_data)
            
            logger.info(f"Received task update: {task.id} - state: {task.status.state}")
            
            # Find and execute callback
            callback = self.task_callbacks.get(task.id)
            if callback:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(task)
                    else:
                        callback(task)
                    
                    # Clean up completed/failed/canceled tasks
                    if task.status.state in [TaskState.completed, TaskState.failed, TaskState.canceled]:
                        await self._cleanup_task(task.id)
                        
                except Exception as e:
                    logger.error(f"Error in task callback for {task.id}: {e}")
            else:
                logger.warning(f"No callback registered for task {task.id}")
            
            return JSONResponse({"status": "received", "task_id": task.id})
            
        except Exception as e:
            logger.error(f"Error handling task update: {e}")
            return JSONResponse({"error": str(e)}, status_code=400)
    
    async def _health_check(self, request: Request) -> JSONResponse:
        """Health check endpoint."""
        return JSONResponse({
            "status": "healthy",
            "webhook_url": self.webhook_url,
            "active_tasks": len(self.task_callbacks),
            "webhook_port": self.webhook_port
        })
    
    async def register_task_callback(
        self,
        client: A2AClient,
        task_id: str,
        callback: Callable[[Task], Any],
        timeout_seconds: float = 300.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a push notification callback for a task using official A2A SDK.
        
        Args:
            client: A2A client instance
            task_id: The task ID to monitor
            callback: Function to call when task updates arrive
            timeout_seconds: Timeout for the callback registration
            metadata: Optional metadata to associate with the task
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Ensure webhook server is running
            await self.start()
            
            # Register callback locally
            self.task_callbacks[task_id] = callback
            self.task_timeouts[task_id] = timeout_seconds
            if metadata:
                self.task_metadata[task_id] = metadata
            
            # Use official A2A SDK to set push notification
            push_config = PushNotificationConfig(url=self.webhook_url)
            request = SetTaskPushNotificationConfigRequest(
                params=TaskPushNotificationConfig(
                    taskId=task_id,
                    pushNotificationConfig=push_config
                )
            )
            
            response = await client.set_task_callback(request)
            
            if hasattr(response, 'error'):
                logger.error(f"Failed to set task callback: {response.error}")
                await self._cleanup_task(task_id)
                return False
                
            logger.info(f"Successfully registered push notification for task {task_id}")
            
            # Set cleanup timeout
            asyncio.create_task(self._cleanup_after_timeout(task_id, timeout_seconds))
            
            return True
            
        except Exception as e:
            logger.error(f"Error registering task callback: {e}")
            await self._cleanup_task(task_id)
            return False
    
    async def get_task_callback_info(self, client: A2AClient, task_id: str) -> Optional[PushNotificationConfig]:
        """
        Get push notification configuration for a task using official A2A SDK.
        
        Args:
            client: A2A client instance
            task_id: The task ID to query
            
        Returns:
            PushNotificationConfig if found, None otherwise
        """
        try:
            request = GetTaskPushNotificationConfigRequest(
                params={"id": task_id}
            )
            
            response = await client.get_task_callback(request)
            
            if hasattr(response, 'error'):
                logger.error(f"Failed to get task callback info: {response.error}")
                return None
                
            if hasattr(response, 'result'):
                return response.result.pushNotificationConfig
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting task callback info: {e}")
            return None
    
    async def _cleanup_task(self, task_id: str):
        """Clean up task tracking."""
        self.task_callbacks.pop(task_id, None)
        self.task_timeouts.pop(task_id, None)
        self.task_metadata.pop(task_id, None)
        logger.debug(f"Cleaned up task {task_id}")
    
    async def _cleanup_after_timeout(self, task_id: str, timeout_seconds: float):
        """Clean up task after timeout."""
        await asyncio.sleep(timeout_seconds)
        if task_id in self.task_callbacks:
            logger.warning(f"Task {task_id} timed out after {timeout_seconds}s")
            await self._cleanup_task(task_id)
    
    async def stop(self):
        """Stop the webhook server and clean up."""
        if self.server_instance and self.server_task and not self.server_task.done():
            # First, try to shut down the server gracefully
            try:
                if hasattr(self.server_instance, 'shutdown'):
                    await self.server_instance.shutdown()
                    logger.debug("Server shutdown initiated")
                    
                # Wait for the server task to complete
                await asyncio.wait_for(self.server_task, timeout=5.0)
                
            except asyncio.TimeoutError:
                logger.warning("Server shutdown timed out, cancelling task")
                self.server_task.cancel()
                try:
                    await self.server_task
                except asyncio.CancelledError:
                    pass
            except Exception as e:
                logger.debug(f"Error during server shutdown: {e}")
                # If graceful shutdown fails, cancel the task
                self.server_task.cancel()
                try:
                    await self.server_task
                except asyncio.CancelledError:
                    pass
        
        # Clean up references
        self.server_instance = None
        self.server_task = None
        self.task_callbacks.clear()
        self.task_timeouts.clear()
        self.task_metadata.clear()
        logger.info("Webhook manager stopped")
    
    @property
    def is_running(self) -> bool:
        """Check if the webhook server is running."""
        return self.server_task is not None and not self.server_task.done()
    
    @property
    def active_task_count(self) -> int:
        """Get the number of active tasks being monitored."""
        return len(self.task_callbacks)


# Factory functions for different use cases

def get_client_webhook_manager(webhook_port: Optional[int] = None) -> A2AWebhookManager:
    """Get the webhook manager instance for a client."""
    if webhook_port is None:
        # Load webhook port from config
        try:
            import tomllib
            from pathlib import Path
            config_path = Path(__file__).parent / "config" / "config.toml"
            with open(config_path, "rb") as f:
                config = tomllib.load(f)
            webhook_port = config.get("webhook", {}).get("port", 8000)
        except Exception:
            webhook_port = 8000  # fallback default
    return A2AWebhookManager.get_instance(webhook_port)


def get_agent_webhook_manager(agent_port: int, webhook_path: str = "/webhook/a2a-tasks") -> A2AWebhookManager:
    """
    Get the webhook manager instance for an agent.
    
    To avoid port conflicts, the agent's webhook port is derived
    from its main application port (e.g., 8000 -> 8900).
    """
    webhook_port = 8900 + (agent_port % 100)
    return A2AWebhookManager.get_instance(webhook_port, webhook_path=webhook_path)


@asynccontextmanager
async def managed_webhook_server(webhook_port: int, webhook_path: str = "/webhook/a2a-tasks"):
    """Context manager for webhook server with automatic cleanup."""
    manager = A2AWebhookManager.get_instance(webhook_port, webhook_path=webhook_path)
    try:
        await manager.start()
        yield manager
    finally:
        await manager.stop()


# Backward compatibility aliases
A2APushNotificationManager = A2AWebhookManager
get_push_notification_manager = get_client_webhook_manager
managed_push_notifications = managed_webhook_server 
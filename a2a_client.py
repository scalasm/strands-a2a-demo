#!/usr/bin/env python3
"""
A2A Client - Clean Implementation with Official SDK

Main modes:
- interactive: Chat with agents (messages by default, --use-tasks option)
- speak: Voice interaction using Nova Sonic (tasks with push notifications only)
- web: Launch modern web interface with real-time communication
- test: Run comprehensive tests of all A2A configurations

All tasks use push notifications following official A2A SDK patterns.
"""

import asyncio
import copy
import logging
import argparse
import tomllib
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from uuid import uuid4
import os
import base64
import json
import threading
import time
import warnings
import uuid
import hashlib
import numpy as np
import boto3
import signal
import sys
from functools import wraps

import httpx
from agent_client import get_client_from_agent_card_url
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import (
    Task,
    TaskState,
    Message,
    MessageSendParams,
    SendMessageRequest,
    Part,
    TextPart,
    Role,
    AgentCard,
)

from a2a_push_notification_manager import get_client_webhook_manager, managed_webhook_server
from app_config import configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# Global cleanup flag for signal handling
_cleanup_in_progress = False

# Debug mode flag
DEBUG = False


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        global _cleanup_in_progress
        if _cleanup_in_progress:
            logger.info("Already cleaning up, forcing exit...")
            sys.exit(1)
        
        _cleanup_in_progress = True
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        
        # For asyncio programs, we'll raise KeyboardInterrupt to let the main loop handle cleanup
        raise KeyboardInterrupt("Signal received")
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)


def load_config():
    """Load configuration from config/config.toml."""
    config_path = Path(__file__).parent / "config" / "config.toml"
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def discover_available_agents():
    """
    Discover available agents from the configuration file.
    Returns a dictionary mapping agent names to their URLs.
    """
    config = load_config()
    agents = {}
    
    # Discover configurable agents
    configurable_agents = config.get("configurable_agent", {})
    for agent_name, agent_config in configurable_agents.items():
        # Validate that 'all' is not used as an agent name
        if agent_name.lower() == "all":
            logger.error(f"❌ Agent name 'all' is reserved and cannot be used in configuration: configurable_agent.{agent_name}")
            raise ValueError(f"Agent name 'all' is reserved and cannot be used. Found in: configurable_agent.{agent_name}")
        
        port = agent_config.get("port", 8003)
        agents[agent_name] = f"http://localhost:{port}"
    
    # Discover coordinator agents
    coordinator_agents = config.get("coordinator_agent", {})
    for coordinator_name, coordinator_config in coordinator_agents.items():
        # Validate that 'all' is not used as an agent name
        if coordinator_name.lower() == "all":
            logger.error(f"❌ Agent name 'all' is reserved and cannot be used in configuration: coordinator_agent.{coordinator_name}")
            raise ValueError(f"Agent name 'all' is reserved and cannot be used. Found in: coordinator_agent.{coordinator_name}")
        
        port = coordinator_config.get("port", 8000)
        agents[coordinator_name] = f"http://localhost:{port}"
    
    return agents


def get_agent_info_summary():
    """Get a formatted summary of all available agents and their capabilities."""
    config = load_config()
    summary_lines = []
    
    # Configurable agents
    configurable_agents = config.get("configurable_agent", {})
    if configurable_agents:
        summary_lines.append("📋 **Configurable Agents** (Individual Specialists):")
        for agent_name, agent_config in configurable_agents.items():
            port = agent_config.get("port", 8003)
            description = agent_config.get("description", "No description available")
            tools = agent_config.get("strands_tools", [])
            mcp_servers = list(agent_config.get("mcp_servers", {}).keys())
            
            capabilities = []
            if tools:
                capabilities.append(f"Tools: {', '.join(tools)}")
            if mcp_servers:
                capabilities.append(f"MCP: {', '.join(mcp_servers)}")
            
            capability_str = f" ({'; '.join(capabilities)})" if capabilities else ""
            summary_lines.append(f"  • {agent_name} (port {port}): {description}{capability_str}")
    
    # Coordinator agents
    coordinator_agents = config.get("coordinator_agent", {})
    if coordinator_agents:
        summary_lines.append("\n🌀 **Coordinator Agents** (Manage Groups):")
        for coordinator_name, coordinator_config in coordinator_agents.items():
            port = coordinator_config.get("port", 8000)
            description = coordinator_config.get("description", "No description available")
            coordinated = coordinator_config.get("coordinated_agents", [])
            
            coordinated_str = f" (manages: {', '.join(coordinated)})" if coordinated else ""
            summary_lines.append(f"  • {coordinator_name} (port {port}): {description}{coordinated_str}")
    
    return "\n".join(summary_lines) if summary_lines else "No agents configured"


def debug_print(message):
    """Print only if debug mode is enabled"""
    if DEBUG:
        import inspect
        functionName = inspect.stack()[1].function
        if functionName == 'time_it' or functionName == 'time_it_async':
            functionName = inspect.stack()[2].function
        logger.debug(f"{functionName}: {message}")


def time_it(label, methodToRun):
    """Time a synchronous operation"""
    start_time = time.perf_counter()
    result = methodToRun()
    end_time = time.perf_counter()
    logger.debug(f"Execution time for {label}: {end_time - start_time:.4f} seconds")
    return result


async def time_it_async(label, methodToRun):
    """Time an asynchronous operation"""
    start_time = time.perf_counter()
    result = await methodToRun()
    end_time = time.perf_counter()
    logger.debug(f"Execution time for {label}: {end_time - start_time:.4f} seconds")
    return result


class CleanA2AClient:
    """Clean A2A client using official SDK patterns with skill discovery."""
    
    def __init__(self, agent_url: str, agent_name: str):
        """Initialize the clean A2A client."""
        self.agent_url = agent_url
        self.agent_name = agent_name
        self.httpx_client = httpx.AsyncClient(timeout=None)
        self._a2a_client: Optional[A2AClient] = None
        
        # Agent card and skills information
        self.agent_card: Optional[AgentCard] = None
        self.skills_discovered = False
        
        logger.debug(f"Initialized {agent_name} client for {agent_url}")
    
    async def _get_client(self) -> A2AClient:
        """Get or create the A2A client instance."""
        if self._a2a_client is None:
            self._a2a_client = await get_client_from_agent_card_url(
                httpx_client=self.httpx_client,
                base_url=self.agent_url
            )
        return self._a2a_client
    
    async def discover_agent_skills(self) -> bool:
        """
        Discover agent skills by fetching the agent card.
        Returns True if successful, False otherwise.
        """
        if self.skills_discovered and self.agent_card:
            return True
            
        try:
            logger.info(f"🔍 Discovering skills for {self.agent_name}...")
            
            # Get agent card using A2ACardResolver
            resolver = A2ACardResolver(
                httpx_client=self.httpx_client,
                base_url=self.agent_url,
                agent_card_path='/.well-known/agent.json'
            )
            
            self.agent_card = await resolver.get_agent_card()
            self.skills_discovered = True
            
            # Log discovered information
            logger.info(f"✅ Discovered {self.agent_name}: {self.agent_card.name} v{self.agent_card.version}")
            logger.info(f"   📝 Description: {self.agent_card.description}")
            
            if self.agent_card.skills:
                skill_names = [skill.name for skill in self.agent_card.skills]
                logger.info(f"   🎯 Skills ({len(skill_names)}): {', '.join(skill_names)}")
                
                # Log detailed skill information in debug mode
                for skill in self.agent_card.skills:
                    logger.debug(f"   📋 Skill '{skill.name}': {skill.description}")
                    if skill.examples:
                        logger.debug(f"      Examples: {', '.join(skill.examples[:3])}")
            else:
                logger.info(f"   🎯 No skills defined")
            
            return True
            
        except Exception as e:
            logger.warning(f"❌ Failed to discover skills for {self.agent_name}: {e}")
            return False
    
    def get_skills_summary(self) -> str:
        """Get a formatted summary of the agent's skills."""
        if not self.agent_card or not self.agent_card.skills:
            return f"{self.agent_name}: No skills available"
        
        skills_list = []
        for skill in self.agent_card.skills:
            skill_desc = f"• {skill.name}: {skill.description}"
            if skill.examples:
                examples = ", ".join(skill.examples[:2])
                skill_desc += f" (e.g., {examples})"
            skills_list.append(skill_desc)
        
        header = f"{self.agent_name} - {self.agent_card.description}\n"
        header += f"Available Skills ({len(self.agent_card.skills)}):"
        
        return header + "\n" + "\n".join(skills_list)
    
    def get_capabilities_description(self) -> str:
        """Get a description of agent capabilities for tool/prompt descriptions."""
        if not self.agent_card or not self.agent_card.skills:
            return "Send a task to process requests"
        
        # Format: "Send a task to use these skills: <skills with descriptions>"
        skills_list = []
        for skill in self.agent_card.skills:
            skills_list.append(f"{skill.name}: {skill.description}")
        
        return f"Send a task to use these skills: {'; '.join(skills_list)}"
    
    async def ensure_healthy_connection(self) -> bool:
        """
        Centralized method to ensure the connection is healthy and refresh if needed.
        Returns True if connection is healthy, False if it couldn't be established.
        """
        if not self.httpx_client:
            logger.warning(f"httpx client not available for {self.agent_name}")
            return False
        
        # If A2A client doesn't exist yet, try to create it
        if not self._a2a_client:
            try:
                logger.debug(f"Initializing A2A client for {self.agent_name}")
                await self._refresh_connection()
                return True
            except Exception as e:
                logger.error(f"Failed to initialize A2A client for {self.agent_name}: {e}")
                return False
        
        # Check if connection is healthy
        if await self._check_connection_health():
            return True
        
        # Connection is unhealthy, attempt to refresh
        logger.warning(f"Connection to {self.agent_name} unhealthy, attempting refresh...")
        try:
            await self._refresh_connection()
            return True
        except Exception as e:
            logger.error(f"Failed to refresh connection to {self.agent_name}: {e}")
            return False
    
    async def _check_connection_health(self) -> bool:
        """Internal method to check if the connection to the agent is healthy."""
        if not self.httpx_client:
            return False
        
        try:
            # Check if httpx client is closed
            if self.httpx_client.is_closed:
                logger.debug(f"httpx client for {self.agent_name} is closed")
                return False
            
            # Simple HTTP GET to check if the agent is reachable
            response = await self.httpx_client.get(f"{self.agent_url}/.well-known/agent.json")
            return response.status_code == 200
        except Exception as e:
            # Any exception means connection is not healthy
            logger.debug(f"Connection health check failed for {self.agent_name}: {e}")
            return False
    
    async def _refresh_connection(self):
        """Internal method to refresh the connection by recreating the httpx client and A2A client."""
        try:
            # Only close existing httpx client if it exists and we're refreshing (not initial creation)
            if self._a2a_client and self.httpx_client and not self.httpx_client.is_closed:
                logger.debug(f"Closing existing httpx client for {self.agent_name}")
                await self.httpx_client.aclose()
            
            # Create new httpx client with no timeout (only if we don't have one or it was closed)
            if not self.httpx_client or self.httpx_client.is_closed:
                limits = httpx.Limits(
                    max_keepalive_connections=5,
                    max_connections=10,
                    keepalive_expiry=300.0  # Keep connections alive for 5 minutes
                )
                
                self.httpx_client = httpx.AsyncClient(
                    timeout=None,
                    limits=limits,
                    http2=False  # Use HTTP/1.1 for better compatibility
                )
                logger.debug(f"Created new httpx client for {self.agent_name}")
            
            # Initialize/reinitialize the A2A client
            self._a2a_client = await get_client_from_agent_card_url(
                httpx_client=self.httpx_client,
                base_url=self.agent_url
            )
            
            logger.info(f"✅ Connection to {self.agent_name} established successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to refresh connection to {self.agent_name}: {e}")
            self._a2a_client = None
            raise
    
    def _create_message(self, content: str, context_id: Optional[str] = None) -> Message:
        """Create an A2A Message using official SDK patterns."""
        text_part = TextPart(text=content)
        part = Part(root=text_part)
        
        return Message(
            message_id=str(uuid4()),
            context_id=context_id or str(uuid4()),
            role=Role.user,
            parts=[part]
        )
    
    def _extract_message_text(self, message: Message) -> str:
        """Extract text content from a Message object."""
        if message.parts:
            text_parts = []
            for part in message.parts:
                if hasattr(part.root, 'text'):
                    text_parts.append(part.root.text)
            if text_parts:
                return ''.join(text_parts).strip()
        return str(message)
    
    async def send_message(self, prompt: str) -> str:
        """Send a message and wait for immediate response."""
        # Ensure connection is healthy before proceeding
        if not await self.ensure_healthy_connection():
            raise Exception(f"Unable to establish connection to {self.agent_name}")
        
        client = await self._get_client()
        message = self._create_message(prompt)
        
        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(message=message)
        )
        
        logger.debug(f"Sending message to {self.agent_name}: {prompt[:100]}...")
        
        response = await client.send_message(request)
        
        if hasattr(response.root, 'error'):
            raise Exception(f"Agent error: {response.root.error}")
        
        if hasattr(response.root, 'result'):
            result = response.root.result
            if isinstance(result, Message):
                return self._extract_message_text(result)
            elif isinstance(result, Task):
                # Task completed immediately
                return self.extract_task_result(result)
            else:
                return str(result)
        else:
            raise Exception("No result in response")
    
    async def send_task(self, prompt: str, callback: Optional[Callable[[Task], Any]] = None) -> str:
        """
        Send a task with push notifications (always non-blocking).
        
        Returns task ID immediately, callback is executed when task completes.
        """
        # Ensure connection is healthy before proceeding
        if not await self.ensure_healthy_connection():
            raise Exception(f"Unable to establish connection to {self.agent_name}")
        
        client = await self._get_client()
        message = self._create_message(prompt)
        
        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(message=message)
        )
        
        logger.debug(f"Sending task to {self.agent_name}: {prompt[:100]}...")
        
        # Send task and get task ID immediately
        response = await client.send_message(request)
        
        if hasattr(response.root, 'error'):
            raise Exception(f"Agent error: {response.root.error}")
        
        if hasattr(response.root, 'result'):
            result = response.root.result
            if isinstance(result, Task):
                task = result
                task_id = task.id
                
                logger.debug(f"Task {task_id} created, setting up push notifications...")
                
                # Register push notification using official A2A SDK
                if callback:
                    webhook_manager = get_client_webhook_manager()
                    success = await webhook_manager.register_task_callback(
                        client=client,
                        task_id=task_id,
                        callback=callback,
                        timeout_seconds=300.0
                    )
                    
                    if success:
                        logger.debug(f"Push notification registered for task {task_id}")
                    else:
                        logger.warning(f"Failed to register push notification for task {task_id}")
                
                return task_id
            elif isinstance(result, Message):
                # Agent responded immediately - call callback immediately if provided
                logger.debug(f"Immediate response received from {self.agent_name}")
                
                if callback:
                    # Create a mock completed task for the callback
                    from a2a.types import TaskStatus, TaskState
                    mock_task = Task(
                        id="immediate-" + str(uuid4()),
                        context_id=result.context_id or str(uuid4()),
                        status=TaskStatus(state=TaskState.completed, message=result),
                        history=[result]
                    )
                    # Call the callback immediately
                    if asyncio.iscoroutinefunction(callback):
                        await callback(mock_task)
                    else:
                        callback(mock_task)
                
                return "immediate-response"
            else:
                raise Exception(f"Unexpected response type: {type(result)}")
        else:
            raise Exception("No result in response")
    
    def extract_task_result(self, task: Task) -> str:
        """Extract the text result from a completed task."""
        if task.status.state != TaskState.completed:
            return f"Task not completed. Current state: {task.status.state}"
        
        # Extract from status message first
        if task.status.message and task.status.message.parts:
            for part in task.status.message.parts:
                if hasattr(part.root, 'text'):
                    return part.root.text
        
        # Extract from artifacts
        if task.artifacts:
            for artifact in task.artifacts:
                for part in artifact.parts:
                    if hasattr(part.root, 'text'):
                        return part.root.text
        
        # Extract from history (last agent message)
        if task.history:
            for message in reversed(task.history):
                if message.role == Role.agent and message.parts:
                    for part in message.parts:
                        if hasattr(part.root, 'text'):
                            return part.root.text
        
        return "No text result found in task"
    
    async def close(self):
        """Clean up resources."""
        try:
            if self.httpx_client and not self.httpx_client.is_closed:
                await self.httpx_client.aclose()
                logger.debug(f"Closed httpx client for {self.agent_name}")
        except Exception as e:
            logger.debug(f"Error closing httpx client for {self.agent_name}: {e}")


# =============================================================================
# MODE IMPORTS
# =============================================================================

def get_interactive_mode():
    """Import and return interactive mode function."""
    from interactive_interface import interactive_mode
    return interactive_mode

def get_speak_mode():
    """Import and return speak mode function."""
    from speak_interface import speak_mode
    return speak_mode

def get_test_mode():
    """Import and return test mode function."""
    from test_interface import test_mode
    return test_mode

def get_web_mode():
    """Import and return web mode function."""
    from web_interface import web_mode
    return web_mode

def get_start_agents_mode():
    """Import and return start agents mode function."""
    from agent_process_manager import AgentProcessManager, setup_signal_handlers, start_single_agent, stop_single_agent, restart_single_agent, restart_all_agents
    return AgentProcessManager, setup_signal_handlers, start_single_agent, stop_single_agent, restart_single_agent, restart_all_agents

def get_stop_agents_mode():
    """Import and return stop agents mode function."""
    from agent_process_manager import AgentProcessManager
    return AgentProcessManager

def get_status_agents_mode():
    """Import and return status agents mode function."""
    from agent_process_manager import AgentProcessManager
    return AgentProcessManager


# =============================================================================
# MAIN CLI
# =============================================================================

def main():
    """Main function to run the A2A client in different modes."""
    parser = argparse.ArgumentParser(
        description="A2A Client for interacting with agents via multiple interfaces.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug-level logs",
    )
    
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Available modes")

    # Get mode entry points
    interactive_mode_func = get_interactive_mode()
    speak_mode_func = get_speak_mode()
    test_mode_func = get_test_mode()
    web_mode_func = get_web_mode()
    AgentProcessManager, setup_signal_handlers, start_single_agent, stop_single_agent, restart_single_agent, restart_all_agents = get_start_agents_mode()
    AgentProcessManagerStop = get_stop_agents_mode()
    AgentProcessManagerStatus = get_status_agents_mode()
    
    # Interactive mode sub-command
    interactive_parser = subparsers.add_parser(
        "interactive",
        help="Interactive chat mode - select agent then chat",
        description="Run the client in interactive mode with a specified agent or all agents.",
    )
    interactive_parser.add_argument(
        "--use-tasks",
        action="store_true",
        help="Use tasks instead of messages (enables push notifications)",
    )
    available_agents = list(discover_available_agents().keys())
    interactive_parser.add_argument(
        "--agent",
        default="all",
        choices=["all"] + available_agents,
        help=f"Filter to specific agent (default: all, choices: {', '.join(['all'] + available_agents)})",
    )
    interactive_parser.add_argument(
        "--send",
        type=str,
        help="Send a single message to the specified agent and exit (requires --agent to be specified, not 'all')",
    )
    
    # Speak mode sub-command
    speak_parser = subparsers.add_parser(
        "speak",
        help="Voice interaction mode",
        description="Run the client in voice interaction mode with a specified agent.",
    )
    speak_parser.add_argument(
        "--agent",
        required=True,
        choices=available_agents,
        help=f"Agent for voice interaction (required, choices: {', '.join(available_agents)})",
    )
    speak_parser.add_argument(
        "--disable-echo-cancellation",
        action="store_true",
        help="Disable echo cancellation for speak mode",
    )
    
    # WebSpeak mode sub-command
    webspeak_parser = subparsers.add_parser(
        "webspeak",
        help="Launch a web-based voice interface using Nova Sonic",
        description="Run a Flask-SocketIO server for real-time voice chat in the browser.",
    )
    # Load config to get default webspeak port
    try:
        config = load_config()
        default_webspeak_port = config.get("web_interface", {}).get("webspeak_port", 8081)
    except Exception:
        default_webspeak_port = 8081
    
    webspeak_parser.add_argument(
        "--port",
        type=int,
        default=default_webspeak_port,
        help=f"Port to run the WebSpeak server on (default: {default_webspeak_port})",
    )
    webspeak_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the WebSpeak server to (default: 0.0.0.0)",
    )
    
    # Test mode sub-command
    test_parser = subparsers.add_parser(
        "test",
        help="Run the client as a comprehensive test suite for all agents.",
        description="Run the client in test mode to perform comprehensive tests of all A2A configurations.",
    )
    test_parser.add_argument(
        "--agent",
        default="all",
        choices=["all"] + available_agents,
        help=f"Filter to specific agent (default: all, choices: {', '.join(['all'] + available_agents)})",
    )
    
    # Web mode sub-command
    web_parser = subparsers.add_parser(
        "web",
        help="Launch a modern web interface for real-time agent chat",
        description="Run a FastAPI web server with a WebSocket-based chat interface.",
    )
    # Load config to get default web interface port
    try:
        config = load_config()
        default_web_port = config.get("web_interface", {}).get("port", 8080)
    except Exception:
        default_web_port = 8080
    
    web_parser.add_argument(
        "--port",
        type=int,
        default=default_web_port,
        help=f"Port to run the web server on (default: {default_web_port})",
    )
    web_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the web server to (default: 0.0.0.0)",
    )
    
    # Agents management sub-command
    agents_parser = subparsers.add_parser(
        "agents",
        help="Manage agent processes (start, stop, restart, status)",
        description="Manage agent processes with various operations.",
    )
    agents_subparsers = agents_parser.add_subparsers(
        dest="agent_command",
        help="Agent management commands",
        required=True
    )
    
    # Start command
    start_parser = agents_subparsers.add_parser(
        "start",
        help="Start agent processes",
        description="Start specified agents or all agents as background processes with logging.",
    )
    start_parser.add_argument(
        "target",
        nargs="?",
        default="all",
        choices=["all"] + available_agents,
        help=f"Agent to start (choices: all, {', '.join(available_agents)}, default: all)",
    )
    start_parser.add_argument(
        "--force",
        action="store_true",
        help="Kill existing processes on agent ports before starting",
    )
    start_parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for agents to be ready before exiting",
    )
    start_parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run in daemon mode (keep process manager running)",
    )
    
    # Stop command
    stop_parser = agents_subparsers.add_parser(
        "stop",
        help="Stop agent processes",
        description="Stop specified agents or all running agent processes.",
    )
    stop_parser.add_argument(
        "target",
        nargs="?",
        default="all",
        choices=["all"] + available_agents,
        help=f"Agent to stop (choices: all, {', '.join(available_agents)}, default: all)",
    )
    stop_parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Also kill any processes on configured ports (not just managed ones)",
    )
    
    # Cleanup command for killing all processes on configured ports
    cleanup_parser = agents_subparsers.add_parser(
        "cleanup",
        help="Kill any processes running on configured agent ports",
        description="Force kill any processes running on ports configured for agents. Useful for cleaning up orphaned processes.",
    )
    
    # Restart command
    restart_parser = agents_subparsers.add_parser(
        "restart",
        help="Restart agent processes",
        description="Restart specified agents or all agent processes.",
    )
    restart_parser.add_argument(
        "target",
        nargs="?",
        default="all",
        choices=["all"] + available_agents,
        help=f"Agent to restart (choices: all, {', '.join(available_agents)}, default: all)",
    )
    restart_parser.add_argument(
        "--force",
        action="store_true",
        help="Kill existing processes on agent ports before starting",
    )
    restart_parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for agents to be ready before exiting",
    )
    restart_parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run in daemon mode (keep process manager running)",
    )
    
    # Status command
    status_parser = agents_subparsers.add_parser(
        "status",
        help="Show status of agent processes",
        description="Display the current status of specified agents or all configured agents.",
    )
    status_parser.add_argument(
        "target",
        nargs="?",
        default="all",
        choices=["all"] + available_agents,
        help=f"Agent to check status (choices: all, {', '.join(available_agents)}, default: all)",
    )
    status_parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed information including PIDs and log files",
    )
    
    args = parser.parse_args()
    
    global DEBUG
    if args.debug:
        DEBUG = True
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    try:
        if args.mode == "interactive":
            # Validate --send requires specific agent
            if hasattr(args, 'send') and args.send and args.agent == "all":
                parser.error("--send requires --agent to be specified (not 'all')")
            asyncio.run(interactive_mode_func(use_tasks=args.use_tasks, agent_filter=args.agent, send_message=getattr(args, 'send', None)))
        
        elif args.mode == "speak":
            if not args.agent:
                parser.error("speak mode requires --agent")
            asyncio.run(speak_mode_func(agent_name=args.agent, disable_echo_cancellation=args.disable_echo_cancellation))
            
        elif args.mode == "test":
            asyncio.run(test_mode_func(agent_filter=args.agent))
            
        elif args.mode == "web":
            # Running uvicorn programmatically
            import uvicorn
            from web_interface import app as web_app
            uvicorn.run(web_app, host=args.host, port=args.port)

        elif args.mode == "webspeak":
            # Running Flask-SocketIO programmatically
            from webspeak_interface import create_app
            app, socketio = create_app()
            print(f"Starting WebSpeak server on http://{args.host}:{args.port}")
            socketio.run(app, host=args.host, port=args.port, debug=args.debug)
            
        elif args.mode == "agents":
            # Handle agents management commands
            manager = AgentProcessManager()
            
            if args.agent_command == "start":
                if args.target == "all":
                    success = manager.start_all_agents(force=args.force)
                    
                    if success and args.wait:
                        ready = manager.wait_for_agents_ready()
                        if not ready:
                            logger.warning("Some agents may not be fully ready")
                    
                    if args.daemon and success:
                        logger.info("Running in daemon mode. Press Ctrl+C to stop all agents.")
                        setup_signal_handlers(manager)
                        try:
                            while True:
                                time.sleep(1)
                        except KeyboardInterrupt:
                            logger.info("Shutting down...")
                            manager.stop_all_agents()
                    elif success:
                        logger.info("All agents started successfully. Use 'uv run a2a-client agents all stop' to stop them.")
                    else:
                        logger.error("Failed to start some agents. Check logs for details.")
                        sys.exit(1)
                else:
                    # Start specific agent
                    success = start_single_agent(manager, args.target)
                    
                    if success and args.wait:
                        ready = manager.wait_for_agent_ready(args.target)
                        if not ready:
                            logger.warning(f"Agent {args.target} may not be fully ready")
                    
                    if args.daemon and success:
                        logger.info(f"Running agent {args.target} in daemon mode. Press Ctrl+C to stop.")
                        setup_signal_handlers(manager)
                        try:
                            while True:
                                time.sleep(1)
                        except KeyboardInterrupt:
                            logger.info("Shutting down...")
                            stop_single_agent(manager, args.target)
                    elif success:
                        logger.info(f"Agent {args.target} started successfully.")
                    else:
                        logger.error(f"Failed to start agent {args.target}. Check logs for details.")
                        sys.exit(1)
                        
            elif args.agent_command == "stop":
                if args.target == "all":
                    if args.cleanup:
                        # Kill any processes on configured ports
                        manager.kill_all_agents_on_configured_ports()
                    else:
                        # Only stop managed processes
                        success = manager.stop_all_agents()
                        if not success:
                            logger.warning("Some agents may not have stopped cleanly")
                else:
                    # Stop specific agent
                    success = stop_single_agent(manager, args.target)
                    if not success:
                        logger.warning(f"Agent {args.target} may not have stopped cleanly")
                        
            elif args.agent_command == "cleanup":
                # Kill all processes on configured agent ports
                logger.info("🧹 Force killing any processes on configured agent ports...")
                success = manager.kill_all_agents_on_configured_ports()
                if success:
                    logger.info("✅ Cleanup completed successfully")
                else:
                    logger.error("❌ Some cleanup operations failed")
                    sys.exit(1)
                    
            elif args.agent_command == "restart":
                if args.target == "all":
                    # Stop all first
                    logger.info("Stopping all agents...")
                    manager.stop_all_agents()
                    time.sleep(2)  # Give processes time to stop
                    
                    # Start all
                    success = manager.start_all_agents(force=args.force)
                    
                    if success and args.wait:
                        ready = manager.wait_for_agents_ready()
                        if not ready:
                            logger.warning("Some agents may not be fully ready")
                    
                    if args.daemon and success:
                        logger.info("Running in daemon mode. Press Ctrl+C to stop all agents.")
                        setup_signal_handlers(manager)
                        try:
                            while True:
                                time.sleep(1)
                        except KeyboardInterrupt:
                            logger.info("Shutting down...")
                            manager.stop_all_agents()
                    elif success:
                        logger.info("All agents restarted successfully.")
                    else:
                        logger.error("Failed to restart some agents. Check logs for details.")
                        sys.exit(1)
                else:
                    # Restart specific agent
                    logger.info(f"Stopping agent {args.target}...")
                    stop_single_agent(manager, args.target)
                    time.sleep(2)  # Give process time to stop
                    
                    success = start_single_agent(manager, args.target)
                    
                    if success and args.wait:
                        ready = manager.wait_for_agent_ready(args.target)
                        if not ready:
                            logger.warning(f"Agent {args.target} may not be fully ready")
                    
                    if args.daemon and success:
                        logger.info(f"Running agent {args.target} in daemon mode. Press Ctrl+C to stop.")
                        setup_signal_handlers(manager)
                        try:
                            while True:
                                time.sleep(1)
                        except KeyboardInterrupt:
                            logger.info("Shutting down...")
                            stop_single_agent(manager, args.target)
                    elif success:
                        logger.info(f"Agent {args.target} restarted successfully.")
                    else:
                        logger.error(f"Failed to restart agent {args.target}. Check logs for details.")
                        sys.exit(1)
                        
            elif args.agent_command == "status":
                # Show agent status
                if args.target == "all":
                    status = manager.get_agent_status()
                    
                    print("\n🤖 Agent Status")
                    print("=" * 50)
                    
                    for agent_name, info in status.items():
                        status_icon = "✅" if info["running"] else "❌"
                        managed_icon = "🔧" if info["managed"] else "🔍"
                        
                        print(f"{status_icon} {agent_name} (port {info['port']})")
                        
                        if args.detailed:
                            print(f"   {managed_icon} Managed: {info['managed']}")
                            if info["pid"]:
                                print(f"   🆔 PID: {info['pid']}")
                            print(f"   📄 Log: {info['log_file']}")
                            print(f"   ⚙️  Command: {' '.join(info['command'])}")
                            print()
                    
                    running_count = sum(1 for info in status.values() if info["running"])
                    total_count = len(status)
                    print(f"\n📊 Summary: {running_count}/{total_count} agents running")
                else:
                    # Show status for specific agent
                    status = manager.get_agent_status(args.target)
                    if status:
                        agent_name = args.target
                        info = status[agent_name]
                        status_icon = "✅" if info["running"] else "❌"
                        managed_icon = "🔧" if info["managed"] else "🔍"
                        
                        print(f"\n🤖 Agent Status: {agent_name}")
                        print("=" * 50)
                        print(f"{status_icon} {agent_name} (port {info['port']})")
                        
                        if args.detailed:
                            print(f"   {managed_icon} Managed: {info['managed']}")
                            if info["pid"]:
                                print(f"   🆔 PID: {info['pid']}")
                            print(f"   📄 Log: {info['log_file']}")
                            print(f"   ⚙️  Command: {' '.join(info['command'])}")
                    else:
                        print(f"❌ Agent {args.target} not found in configuration")

    except KeyboardInterrupt:
        logger.info("Exiting gracefully.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=DEBUG)

if __name__ == "__main__":
    main() 
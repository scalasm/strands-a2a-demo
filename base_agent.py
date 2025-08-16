#!/usr/bin/env python3
"""
Base Agent - Shared functionality for A2A agents
"""

import logging
import os
import sys
import argparse
import tomllib
from pathlib import Path
from typing import Dict, Any, Optional

from strands import Agent
from a2a.types import AgentCard, AgentSkill, AgentCapabilities, Task, TaskState, Message
from a2a.server.agent_execution import AgentExecutor
from a2a.server.events import EventQueue
from a2a.server.agent_execution.context import RequestContext
from a2a.utils import new_agent_text_message
from agent_client import AgentCommunicationClient

from app_config import ModelConfig, get_or_create_ai_model, configure_logging


configure_logging()
logger = logging.getLogger(__name__)

os.environ["BYPASS_TOOL_CONSENT"] = "true"


class BaseAgent:
    """Base class for Strands-based A2A agents."""
    
    def __init__(self, system_prompt: str, tools: list, agent_name: str = "Unknown Agent"):
        """Initialize the base agent."""
        model = get_or_create_ai_model(ModelConfig.from_config())
        logger.info(f"Using AI model: {model.model_id} with type {model.model_type}")
        
        self.strands_agent = Agent(system_prompt=system_prompt, tools=tools, model=model)
        self.agent_name = agent_name
        self.httpx_client = None
        self.communication_clients: Dict[str, AgentCommunicationClient] = {}
    
    def set_httpx_client(self, httpx_client):
        """Set the httpx client for agent-to-agent communication."""
        self.httpx_client = httpx_client
    
    async def add_agent_connection(self, agent_name: str, agent_url: str):
        """Add and initialize a connection to another agent."""
        if agent_name not in self.communication_clients:
            client = AgentCommunicationClient(agent_url, agent_name, self.httpx_client)
            await client.initialize()
            self.communication_clients[agent_name] = client
    
    async def call_agent(self, agent_name: str, query: str, use_tasks: bool = True) -> str:
        """Call another agent using message or task-based communication."""
        if agent_name not in self.communication_clients:
            return f"Agent {agent_name} not connected."
        
        client = self.communication_clients[agent_name]
        if use_tasks:
            return await client.send_task(query)
        else:
            return await client.send_message(query)
    
    async def invoke(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Invoke the Strands agent."""
        import asyncio
        
        # Strands Agent uses __call__ method, not invoke
        # Convert context to kwargs if provided
        kwargs = context if context else {}
        
        # Run the synchronous Strands agent call in a thread executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: self.strands_agent(query, **kwargs))
        
        # Extract text from AgentResult
        if hasattr(result, 'message') and hasattr(result.message, 'content'):
            # Extract text from content blocks
            for content_block in result.message.content:
                if 'text' in content_block:
                    return content_block['text']
        
        # Fallback to string representation
        return str(result)


class BaseAgentExecutor(AgentExecutor):
    """Base A2A AgentExecutor implementation."""
    
    def __init__(self, agent: BaseAgent, task_store, task_description: str = "task"):
        self.agent = agent
        self.task_store = task_store
        self.task_description = task_description
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the agent task."""
        query = self._extract_query(context.message)
        if not query:
            await event_queue.enqueue_event(new_agent_text_message(self.get_empty_query_message()))
            return
            
        is_task_mode = "[TASK MODE]" in query
        if is_task_mode:
            await self._process_as_task(context, query.replace("[TASK MODE]", "").strip(), event_queue)
        else:
            result = await self.agent.invoke(query)
            await event_queue.enqueue_event(new_agent_text_message(result))
            
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the agent task."""
        # For simple agents, we don't have complex cancellation logic
        # Just send a cancellation message
        await event_queue.enqueue_event(new_agent_text_message("Task cancelled."))
    
    def _extract_query(self, message: Message) -> str:
        """Extract the text query from a message."""
        if not message or not message.parts:
            return ""
        return " ".join(part.root.text for part in message.parts if part.root.kind == "text").strip()
    
    async def _process_as_task(self, context: RequestContext, query: str, event_queue: EventQueue) -> None:
        """Process the request as an asynchronous task."""
        from a2a.utils import create_task_obj
        
        task = create_task_obj(
            context_id=context.context_id,
            request_message=context.message,
            initial_state=TaskState.working,
            initial_message=self.get_working_message(),
        )
        task.id = context.task_id
        await event_queue.enqueue_event(task)
        await self.task_store.save(task)
                
        result = await self.agent.invoke(query)
        
        await self._update_task_status(task.id, TaskState.completed, result)
    
    async def _update_task_status(self, task_id: str, state: TaskState, message_text: str) -> None:
        """Update the status of a task."""
        task = await self.task_store.get(task_id)
        if task:
            task.status.state = state
            task.status.message = new_agent_text_message(message_text)
            await self.task_store.save(task)
    
    def get_empty_query_message(self) -> str:
        """Get the message to return when the query is empty."""
        raise NotImplementedError
    
    def get_working_message(self) -> str:
        """Get the message to show when the task is working."""
        raise NotImplementedError


def create_base_agent_card(name: str, description: str, url: str, skills: list[AgentSkill], version: str = "1.0.0") -> AgentCard:
    """Create a base agent card with common capabilities."""
    capabilities = AgentCapabilities(streaming=False, pushNotifications=True, stateTransitionHistory=False)
    
    return AgentCard(
        name=name,
        description=description,
        url=url,
        version=version,
        capabilities=capabilities,
        defaultInputModes=["text", "text/plain"],
        defaultOutputModes=["text", "text/plain"],
        skills=skills
    )
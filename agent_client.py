#!/usr/bin/env python3
"""
A2A Agent Communication Client
"""

import logging
import httpx
from typing import Any
from uuid import uuid4

from a2a.client import A2AClient, A2ACardResolver
from a2a.types import (
    MessageSendParams,
    SendMessageRequest,
    AgentCard,
)

logger = logging.getLogger(__name__)

# Added this for retro-compatibility with A2A older versions
async def get_client_from_agent_card_url(
    httpx_client: httpx.AsyncClient,
    base_url: str,
    agent_card_path: str = '/.well-known/agent.json',
    http_kwargs: dict[str, Any] | None = None,
) -> 'A2AClient':
    """Fetches the public AgentCard and initializes an A2A client.

    This method will always fetch the public agent card. If an authenticated
    or extended agent card is required, the A2ACardResolver should be used
    directly to fetch the specific card, and then the A2AClient should be
    instantiated with it.

    Args:
        httpx_client: An async HTTP client instance (e.g., httpx.AsyncClient).
        base_url: The base URL of the agent's host.
        agent_card_path: The path to the agent card endpoint, relative to the base URL.
        http_kwargs: Optional dictionary of keyword arguments to pass to the
            underlying httpx.get request when fetching the agent card.

    Returns:
        An initialized `A2AClient` instance.

    Raises:
        A2AClientHTTPError: If an HTTP error occurs fetching the agent card.
        A2AClientJSONError: If the agent card response is invalid.
    """
    agent_card: AgentCard = await A2ACardResolver(
        httpx_client, base_url=base_url, agent_card_path=agent_card_path
    ).get_agent_card(
        http_kwargs=http_kwargs
    )  # Fetches public card by default
    return A2AClient(httpx_client=httpx_client, agent_card=agent_card)



class AgentCommunicationClient:
    """Client for both message and task-based communication with other A2A agents."""
    
    def __init__(self, base_url: str, agent_name: str, httpx_client=None):
        self.base_url = base_url
        self.agent_name = agent_name
        self.httpx_client = httpx_client
        self.client: A2AClient = None
        
    async def initialize(self):
        """Initialize the A2A client."""
        if not self.httpx_client:
            logger.warning(f"No httpx client provided for {self.agent_name}")
            return
            
        try:
            self.client = await get_client_from_agent_card_url(
                httpx_client=self.httpx_client,
                base_url=self.base_url
            )
            logger.info(f"✅ Communication client for {self.agent_name} initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize communication client for {self.agent_name}: {e}")
            raise
    
    async def send_message(self, query: str) -> str:
        """Send a message to the agent (immediate response)."""
        try:
            request = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(
                    message={
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': query}],
                        'message_id': uuid4().hex,
                    }
                )
            )
            
            response = await self.client.send_message(request)
            
            if hasattr(response, 'root') and hasattr(response.root, 'result'):
                result = response.root.result
                if result.parts:
                    return result.parts[0].text
            
            return "No response from agent."
            
        except Exception as e:
            logger.error(f"❌ Error sending message to {self.agent_name}: {e}")
            return f"Error contacting {self.agent_name}: {str(e)}"

    async def send_task(self, query: str) -> str:
        """Send a task to the agent and poll for completion."""
        try:
            query_with_hint = f"[TASK MODE] {query}"
            
            request = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(
                    message={
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': query_with_hint}],
                        'message_id': uuid4().hex,
                    }
                )
            )

            response = await self.client.send_message(request)
            
            if hasattr(response, 'root') and hasattr(response.root, 'result'):
                task = response.root.result
                if hasattr(task, 'id'):
                    return await self._poll_task_completion(task.id)

            return "Failed to create task."

        except Exception as e:
            logger.error(f"❌ Error sending task to {self.agent_name}: {e}")
            return f"Error creating task with {self.agent_name}: {str(e)}"

    async def _poll_task_completion(self, task_id: str, max_attempts: int = 180) -> str:
        """Poll for task completion."""
        import asyncio
        for _ in range(max_attempts):
            try:
                task = await self.client.get_task(task_id)
                if task.status.state in ["completed", "done", "finished", "success", "successful"]:
                    if task.status.message and task.status.message.parts:
                        return task.status.message.parts[0].text
                    return "Task completed without a message."
                elif task.status.state in ["failed", "error", "cancelled", "aborted", "rejected"]:
                    return f"Task failed: {task.status.message.parts[0].text if task.status.message else 'No reason provided'}"
                
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Polling failed for task {task_id}: {e}")
                return "Failed to get task status."
        
        return "Task timed out." 
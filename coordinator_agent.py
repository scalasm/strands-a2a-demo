#!/usr/bin/env python3
"""
Configurable Coordinator Agent - Dynamically connects to and coordinates multiple configurable agents
Following official A2A Python samples patterns from github.com/google-a2a/a2a-samples

Usage: python coordinator_agent_configurable.py <coordinator_key>
Example: python coordinator_agent_configurable.py coordinator
"""

import logging
import uvicorn
import sys
import asyncio
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional

from strands import Agent, tool
from a2a.types import (
    AgentSkill, AgentCard, Message, MessageSendParams, SendMessageRequest,
    Part, TextPart, Role
)
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.client import A2AClient
from uuid import uuid4

from base_agent import BaseAgent, BaseAgentExecutor, create_base_agent_card
from app_config import ModelConfig, load_config, parse_agent_key, load_agent_config, configure_logging, get_or_create_ai_model

# Use centralized logging configuration
configure_logging()
logger = logging.getLogger(__name__)


def configure_coordinator_logging(coordinator_key: str):
    """Configure logging to include coordinator name in all log messages."""
    # Create a custom logger for this specific coordinator
    coordinator_logger = logging.getLogger(f"coordinator_agent.{coordinator_key}")
    return coordinator_logger


def get_coordination_examples(agent_name: str, agent_config: Dict[str, Any]) -> List[str]:
    """Generate specific, helpful examples for coordination with different agent types."""
    # Get the tools this agent uses to understand its capabilities
    strands_tools = agent_config.get("strands_tools", [])
    
    coordination_examples = {
        # Mathematical agents
        "calculator": [
            "What is 15 + 27 * 3?",
            "Calculate the square root of 144",
            "What is 25% of 200?"
        ],
        
        # Time-related agents
        "time": [
            "What time is it in Tokyo?",
            "Current time in UTC and local time",
            "Tell me the time in London and New York"
        ],
        
        # File operation agents
        "fileeditor": [
            "Read the config file and edit line 5",
            "Create a new Python script with a hello world function",
            "Modify the README file to add installation instructions"
        ],
        
        # Web browsing agents
        "webbrowser": [
            "Fetch the latest news from a tech website",
            "Get data from a REST API endpoint",
            "Download content from a specific URL"
        ],
        
        # Combination coordinators
        "time-and-calculator": [
            "What time is it in Paris and calculate 45 * 7?",
            "Current time in Sydney plus the square root of 256",
            "Tell me the UTC time and compute 15% of 400"
        ],
        
        "edit-and-browse": [
            "Fetch data from a URL and save it to a file",
            "Download content and create a summary document",
            "Read a config file and look up documentation online"
        ]
    }
    
    # Try to match by agent name first
    if agent_name in coordination_examples:
        return coordination_examples[agent_name]
    
    # If no specific match, create examples based on tools
    examples = []
    if "calculator" in strands_tools:
        examples.extend(["Calculate 25 + 17", "What is the square of 12?"])
    if "current_time" in strands_tools:
        examples.extend(["What time is it?", "Current time in London"])
    if "editor" in strands_tools or "file_read" in strands_tools or "file_write" in strands_tools:
        examples.extend(["Edit a configuration file", "Read and modify a document"])
    if "fetch" in strands_tools:
        examples.extend(["Fetch data from a website", "Get information from an API"])
    
    # Fallback to generic examples if nothing specific found
    if not examples:
        examples = [
            f"Send a task to the {agent_name} agent",
            f"Use {agent_name} for specialized operations",
            f"Ask {agent_name} to help with tasks"
        ]
    
    return examples[:3]  # Limit to 3 examples for cleaner display


class ConfigurableCoordinatorAgent:
    """Configurable coordinator agent that delegates tasks to other configurable agents."""
    
    def __init__(self, coordinator_key: str, httpx_client=None):
        """Initialize the configurable coordinator agent."""
        self.coordinator_key = coordinator_key
        self.logger = configure_coordinator_logging(coordinator_key)
        self.httpx_client = httpx_client
        config = load_config()
        
        # Get the specific coordinator configuration
        coordinator_agents = config.get("coordinator_agent", {})
        if coordinator_key not in coordinator_agents:
            available_keys = list(coordinator_agents.keys())
            raise ValueError(f"Coordinator '{coordinator_key}' not found in configuration. Available coordinators: {available_keys}")
        
        self.coordinator_config = coordinator_agents[coordinator_key]
        self.configurable_agents_config = config.get("configurable_agent", {})
        
        # Get list of agents this coordinator manages
        self.coordinated_agents = self.coordinator_config.get("coordinated_agents", [])
        self.logger.info(f"Coordinator '{coordinator_key}' will manage agents: {self.coordinated_agents}")
        
        # Create delegation tools for each managed agent
        self.delegation_tools = self._create_delegation_tools()
        
        model_config = self.coordinator_config.get("model", config["agents"]["model"])
        model = get_or_create_ai_model(ModelConfig.from_config(model_config))

        # Create the Strands agent instance
        self.agent = Agent(
            system_prompt=self._build_system_prompt(),
            tools=self.delegation_tools,
            model=model
        )
        
        # Keep track of agent clients
        self.agent_clients: Dict[str, A2AClient] = {}
        
        self.logger.info(f"✅ Configurable coordinator '{coordinator_key}' created with {len(self.delegation_tools)} delegation tools.")
    
    def _create_delegation_tools(self) -> List:
        """Create delegation tools for each managed agent."""
        # Build a dictionary mapping agent names to their delegation functions
        self.agent_tool_map = {}
        
        for agent_name in self.coordinated_agents:
            if agent_name not in self.configurable_agents_config:
                self.logger.warning(f"Coordinated agent '{agent_name}' not found in configurable_agent configuration")
                continue
            
            agent_config = self.configurable_agents_config[agent_name]
            agent_port = agent_config.get("port", 8003)
            agent_url = f"http://localhost:{agent_port}"
            
            # Store the delegation function in our mapping
            self.agent_tool_map[f"{agent_name}_agent_tool"] = {
                'agent_name': agent_name,
                'agent_url': agent_url,
                'description': agent_config.get('description', f'{agent_name} agent')
            }
            
            self.logger.info(f"📋 Mapped delegation tool: {agent_name}_agent_tool -> {agent_url}")
        
        # Create individual tools for each agent using the mapping
        tools = []
        for tool_name, tool_info in self.agent_tool_map.items():
            delegation_tool = self._create_individual_delegation_tool(tool_name, tool_info)
            tools.append(delegation_tool)
        
        return tools
    
    def _create_individual_delegation_tool(self, tool_name: str, tool_info: dict):
        """Create a properly named delegation tool for a specific agent."""
        agent_name = tool_info['agent_name']
        agent_url = tool_info['agent_url']
        description = tool_info['description']
        
        # Create a closure that captures the specific agent details
        def create_tool_function():
            def delegation_function(query: str) -> str:
                f"""Send a query to the {agent_name} agent: {description}
                
                Args:
                    query: The query to send to the {agent_name} agent
                    
                Returns:
                    The response from the {agent_name} agent
                """
                self.logger.info(f"🔄 Delegating to {agent_name}: {query}")
                self.logger.debug(f"🔍 DEBUG - Query type: {type(query)}, Query value: {repr(query)}")
                
                # Ensure query is a string
                if not isinstance(query, str):
                    self.logger.warning(f"⚠️ Query is not a string: {type(query)}, converting to string")
                    query = str(query)
                
                try:
                    # Run async task delegation by creating a new event loop
                    # This is needed because we're in a thread pool executor thread
                    try:
                        # Try to get the current event loop
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        # No event loop in current thread, create a new one
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    # Run the async operation
                    result = loop.run_until_complete(
                        self._delegate_task(agent_name, agent_url, query)
                    )
                    self.logger.info(f"✅ {agent_name} completed task successfully")
                    return result
                except Exception as e:
                    self.logger.error(f"❌ Error delegating to {agent_name}: {e}")
                    return f"Error communicating with {agent_name} agent: {str(e)}"
            
            # Set the function name and docstring properly
            delegation_function.__name__ = tool_name
            delegation_function.__doc__ = f"""Send a query to the {agent_name} agent: {description}
                
Args:
    query: The query to send to the {agent_name} agent
    
Returns:
    The response from the {agent_name} agent
"""
            return delegation_function
        
        # Create the function and apply the @tool decorator
        tool_func = create_tool_function()
        return tool(tool_func)
    
    async def _delegate_task(self, agent_name: str, agent_url: str, query: str) -> str:
        """Coroutine to delegate a task to another agent."""
        try:
            # Get or create A2A client for the target agent
            if agent_name not in self.agent_clients:
                self.agent_clients[agent_name] = await A2AClient.get_client_from_agent_card_url(
                    httpx_client=self.httpx_client,
                    base_url=agent_url
                )
                self.logger.info(f"✅ Created A2A client for agent: {agent_name}")
            
            client = self.agent_clients[agent_name]
            
            # Create proper A2A message request
            text_part = TextPart(text=query)
            part = Part(root=text_part)
            message = Message(
                message_id=str(uuid4()),
                context_id=str(uuid4()),
                role=Role.user,
                parts=[part]
            )
            
            request = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(message=message)
            )
            
            # Send the message and get a response
            response = await client.send_message(request)
            
            # Extract text from the A2A response 
            if hasattr(response, 'root') and hasattr(response.root, 'result'):
                result = response.root.result
                if isinstance(result, Message):
                    # Extract text from message parts
                    if result.parts:
                        text_parts = []
                        for part in result.parts:
                            if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                text_parts.append(part.root.text)
                            elif hasattr(part, 'text'):
                                text_parts.append(part.text)
                        if text_parts:
                            final_result = ''.join(text_parts).strip()
                            self.logger.info(f"✅ Received response from {agent_name}: {final_result[:100]}...")
                            return final_result
                    # Fallback to string representation
                    final_result = str(result)
                    self.logger.info(f"✅ Received response from {agent_name}: {final_result[:100]}...")
                    return final_result
                else:
                    # Handle other result types (like Task)
                    final_result = str(result)
                    self.logger.info(f"✅ Received response from {agent_name}: {final_result[:100]}...")
                    return final_result
            elif hasattr(response, 'root') and hasattr(response.root, 'error'):
                error_msg = f"Agent {agent_name} returned error: {response.root.error}"
                self.logger.error(error_msg)
                return error_msg
            else:
                # Fallback
                final_result = str(response)
                self.logger.info(f"✅ Received response from {agent_name}: {final_result[:100]}...")
                return final_result
            
        except Exception as e:
            self.logger.error(f"Failed to delegate task to {agent_name}: {e}")
            return f"Failed to get response from {agent_name} agent: {str(e)}"
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the coordinator agent."""
        base_prompt = self.coordinator_config.get(
            "system_prompt", 
            f"""You are {self.coordinator_key}, a coordinator agent that delegates tasks to specialist agents.
            Always ask to the specialist agents if they have skills the can help with the task."""
        )
        
        # Add information about available delegation tools
        if self.coordinated_agents:
            tools_info = []
            for agent_name in self.coordinated_agents:
                if agent_name in self.configurable_agents_config:
                    agent_config = self.configurable_agents_config[agent_name]
                    description = agent_config.get("description", f"{agent_name} agent")
                    tools_info.append(f"- `{agent_name}_agent_tool`: {description}")
            
            if tools_info:
                tools_list = "\n".join(tools_info)
                return f"""
{base_prompt}

You have the following delegation tools available:
{tools_list}

Based on the user's request, analyze the query and use the most appropriate delegation tool to forward the task to the specialist agent.
                """.strip()
        
        return base_prompt
    
    async def invoke(self, query: str) -> str:
        """Execute coordinator query."""
        # Strands Agent uses __call__ method, not invoke
        # Run the synchronous Strands agent call in a thread executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: self.agent(query))
        
        # Extract text from AgentResult
        if hasattr(result, 'message') and hasattr(result.message, 'content'):
            # Extract text from content blocks
            for content_block in result.message.content:
                if 'text' in content_block:
                    return content_block['text']
        
        # Fallback to string representation
        return str(result)
    
    async def close(self):
        """Close all agent clients."""
        for agent_name, client in self.agent_clients.items():
            try:
                await client.httpx_client.aclose()
                self.logger.info(f"✅ Closed A2A client for agent: {agent_name}")
            except Exception as e:
                self.logger.error(f"Error closing client for {agent_name}: {e}")


class ConfigurableCoordinatorAgentExecutor(BaseAgentExecutor):
    """A2A AgentExecutor for the configurable coordinator agent."""
    
    def __init__(self, coordinator_key: str, task_store: InMemoryTaskStore, httpx_client=None):
        """Initialize the executor."""
        agent = ConfigurableCoordinatorAgent(coordinator_key, httpx_client=httpx_client)
        super().__init__(agent, task_store, f"coordination request")
        self.coordinator_key = coordinator_key
    
    def get_empty_query_message(self) -> str:
        """Get the message to return when the query is empty."""
        return f"Please provide a task to coordinate with the {self.coordinator_key} coordinator."
    
    def get_working_message(self) -> str:
        """Get the message to show when the task is working."""
        return f"Coordinating with agents via {self.coordinator_key}..."
    
    async def close(self):
        """Cleanup resources used by the executor."""
        if isinstance(self.agent, ConfigurableCoordinatorAgent):
            await self.agent.close()


def create_coordinator_agent_card(coordinator_key: str) -> AgentCard:
    """Create the agent card for the configurable coordinator agent."""
    config = load_config()
    coordinator_agents = config.get("coordinator_agent", {})
    
    if coordinator_key not in coordinator_agents:
        raise ValueError(f"Coordinator '{coordinator_key}' not found in configuration.")
    
    coordinator_config = coordinator_agents[coordinator_key]
    configurable_agents_config = config.get("configurable_agent", {})
    coordinator_logger = configure_coordinator_logging(coordinator_key)
    
    # Create skills based on the coordinated agents
    skills = []
    coordinated_agents = coordinator_config.get("coordinated_agents", [])
    
    for agent_name in coordinated_agents:
        if agent_name in configurable_agents_config:
            agent_config = configurable_agents_config[agent_name]
            
            # Create a coordination skill for this agent with specific examples
            examples = get_coordination_examples(agent_name, agent_config)
            
            skill = AgentSkill(
                id=f"{agent_name}_coordination",
                name=f"{agent_name.title()} Coordination",
                description=f"Coordinates with the {agent_name} agent: {agent_config.get('description', 'Specialist agent')}",
                tags=["coordination", "delegation", agent_name],
                examples=examples
            )
            skills.append(skill)
            coordinator_logger.info(f"📋 Added coordination skill for agent: {agent_name}")
    
    total_skills = len(skills)
    coordinator_logger.info(f"📊 Coordinator '{coordinator_key}' configured with {total_skills} coordination skills")
    
    return create_base_agent_card(
        name=f"{coordinator_key}_coordinator",
        description=coordinator_config.get("description", f"A configurable {coordinator_key} coordinator agent"),
        url=f"http://localhost:{coordinator_config.get('port', 8000)}/",
        skills=skills
    )


def main():
    """Main entry point for the configurable coordinator agent."""
    try:
        # Parse command line arguments using shared helper function
        coordinator_key = parse_agent_key(
            "coordinator",
            "Start a configurable coordinator A2A agent"
        )
        
        # Configure coordinator-specific logging
        coordinator_logger = configure_coordinator_logging(coordinator_key)
        
        # Load and validate coordinator configuration using shared helper function
        coordinator_config = load_agent_config(coordinator_key, "coordinator_agent", fallback_port=8889)
        port = coordinator_config.get("port", 8889)  # Double fallback just in case
        
        coordinator_card = create_coordinator_agent_card(coordinator_key)
        task_store = InMemoryTaskStore()
        executor = ConfigurableCoordinatorAgentExecutor(coordinator_key, task_store)
        
        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=task_store
        )
        
        app = A2AStarletteApplication(
            agent_card=coordinator_card,
            http_handler=request_handler
        ).build()
        
        # Add startup and shutdown events to manage the httpx client
        @app.on_event("startup")
        async def startup_event():
            """Create and assign the httpx client on startup."""
            import httpx
            executor.agent.httpx_client = httpx.AsyncClient(timeout=None)
            coordinator_logger.info("Coordinator's httpx client initialized.")
        
        @app.on_event("shutdown")
        async def shutdown_event():
            """Close the httpx client on shutdown."""
            await executor.close()
            coordinator_logger.info("Coordinator's httpx client closed.")

        coordinator_logger.info(f"🌀 Configurable Coordinator Agent '{coordinator_key}' starting...")
        coordinator_logger.info(f"📋 Agent Card: {coordinator_card.name} v{coordinator_card.version}")
        coordinator_logger.info(f"🎯 Skills: {[skill.name for skill in coordinator_card.skills]}")
        coordinator_logger.info(f"🚀 Server starting on http://localhost:{port}")

        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
        
    except Exception as e:
        # Use generic logger for initialization errors before coordinator_key is known
        if 'coordinator_logger' in locals():
            coordinator_logger.error(f"Failed to start configurable coordinator agent: {e}", exc_info=True)
        else:
            logger.error(f"Failed to start configurable coordinator agent: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main() 
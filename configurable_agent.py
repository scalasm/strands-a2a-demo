#!/usr/bin/env python3
"""
Configurable Agent - Dynamically loads Strands tools and MCP servers
Following official A2A Python samples patterns from github.com/google-a2a/a2a-samples
"""

import logging
import uvicorn
from pathlib import Path
from typing import Dict, List, Any, Optional
from mcp import stdio_client, StdioServerParameters

from strands import Agent

# Import available Strands tools directly (like the working agents)
_AVAILABLE_TOOLS = {}

# Import tools directly with error handling
def _import_tool_direct(tool_name):
    """Try to import a tool directly and add it to available tools."""
    try:
        if tool_name == "calculator":
            from strands_tools import calculator
            _AVAILABLE_TOOLS[tool_name] = calculator
        elif tool_name == "current_time":
            from strands_tools import current_time
            _AVAILABLE_TOOLS[tool_name] = current_time
        elif tool_name == "editor":
            from strands_tools import editor
            _AVAILABLE_TOOLS[tool_name] = editor
        elif tool_name == "file_read":
            from strands_tools import file_read
            _AVAILABLE_TOOLS[tool_name] = file_read
        elif tool_name == "file_write":
            from strands_tools import file_write
            _AVAILABLE_TOOLS[tool_name] = file_write
        elif tool_name == "load_tool":
            from strands_tools import load_tool
            _AVAILABLE_TOOLS[tool_name] = load_tool
        elif tool_name == "python_repl":
            from strands_tools import python_repl
            _AVAILABLE_TOOLS[tool_name] = python_repl
        elif tool_name == "http_request":
            from strands_tools import http_request
            _AVAILABLE_TOOLS[tool_name] = http_request
        elif tool_name == "shell":
            from strands_tools import shell
            _AVAILABLE_TOOLS[tool_name] = shell
        elif tool_name == "environment":
            from strands_tools import environment
            _AVAILABLE_TOOLS[tool_name] = environment
        elif tool_name == "use_aws":
            from strands_tools import use_aws
            _AVAILABLE_TOOLS[tool_name] = use_aws
        elif tool_name == "slack":
            from strands_tools import slack
            _AVAILABLE_TOOLS[tool_name] = slack
        elif tool_name == "cron":
            from strands_tools import cron
            _AVAILABLE_TOOLS[tool_name] = cron
        elif tool_name == "image_reader":
            from strands_tools import image_reader
            _AVAILABLE_TOOLS[tool_name] = image_reader
        elif tool_name == "generate_image":
            from strands_tools import generate_image
            _AVAILABLE_TOOLS[tool_name] = generate_image
        elif tool_name == "nova_reels":
            from strands_tools import nova_reels
            _AVAILABLE_TOOLS[tool_name] = nova_reels
        elif tool_name == "speak":
            from strands_tools import speak
            _AVAILABLE_TOOLS[tool_name] = speak
        elif tool_name == "retrieve":
            from strands_tools import retrieve
            _AVAILABLE_TOOLS[tool_name] = retrieve
        elif tool_name == "memory":
            from strands_tools import memory
            _AVAILABLE_TOOLS[tool_name] = memory
        elif tool_name == "mem0_memory":
            from strands_tools import mem0_memory
            _AVAILABLE_TOOLS[tool_name] = mem0_memory
        elif tool_name == "agent_graph":
            from strands_tools import agent_graph
            _AVAILABLE_TOOLS[tool_name] = agent_graph
        elif tool_name == "journal":
            from strands_tools import journal
            _AVAILABLE_TOOLS[tool_name] = journal
        elif tool_name == "swarm":
            from strands_tools import swarm
            _AVAILABLE_TOOLS[tool_name] = swarm
        elif tool_name == "stop":
            from strands_tools import stop
            _AVAILABLE_TOOLS[tool_name] = stop
        elif tool_name == "think":
            from strands_tools import think
            _AVAILABLE_TOOLS[tool_name] = think
        elif tool_name == "use_llm":
            from strands_tools import use_llm
            _AVAILABLE_TOOLS[tool_name] = use_llm
        elif tool_name == "workflow":
            from strands_tools import workflow
            _AVAILABLE_TOOLS[tool_name] = workflow
        else:
            return False
        return True
    except ImportError:
        return False

# Try to import all known Strands tools
_KNOWN_TOOLS = [
    # File Operations
    "editor", "file_read", "file_write",
    # Utilities
    "calculator", "current_time", "load_tool",
    # Shell & System
    "environment", "shell", "cron",
    # Code Interpretation
    "python_repl",
    # Web & Network
    "http_request", "slack",
    # Multi-modal
    "image_reader", "generate_image", "nova_reels", "speak",
    # AWS Services
    "use_aws",
    # RAG & Memory
    "retrieve", "memory", "mem0_memory",
    # Agents & Workflows
    "agent_graph", "journal", "swarm", "stop", "think", "use_llm", "workflow"
]

# Import all available tools at module load time
for tool in _KNOWN_TOOLS:
    _import_tool_direct(tool)
from strands.tools.mcp import MCPClient
from a2a.types import AgentSkill, AgentCard
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

from base_agent import BaseAgent, BaseAgentExecutor, create_base_agent_card
from model_config import load_config, parse_agent_key, load_agent_config, configure_logging, get_or_create_ai_model, ModelConfig


# Use centralized logging configuration
configure_logging()
logger = logging.getLogger(__name__)


class ConfigurableAgent:
    """Configurable agent that dynamically loads Strands tools and MCP servers."""
    
    def __init__(self, agent_key: str):
        """Initialize the configurable agent with a specific agent key."""
        self.agent_key = agent_key
        
        # Load the specific agent configuration using the helper function
        self.agent_config = load_agent_config(agent_key, "configurable_agent")
        
        # Initialize storage for tools
        self.loaded_tools = []
        self.mcp_clients = []
        self.agent: Optional[Agent] = None
        
        # Load Strands tools
        self._load_strands_tools()
        
        logger.info(f"✅ Loaded {len(self.loaded_tools)} Strands tools for agent '{agent_key}' (agent creation delayed)")
    
    async def initialize_with_mcp(self):
        """Initialize MCP servers and create the agent with all tools."""
        mcp_servers_config = self.agent_config.get("mcp_servers", {})
        
        if not mcp_servers_config:
            # No MCP servers, create agent with just Strands tools
            self._create_agent()
            logger.info("✅ Agent created with Strands tools only")
            return
        
        # Initialize MCP servers and collect all tools
        all_tools = self.loaded_tools.copy()
        
        for server_name, server_config in mcp_servers_config.items():
            try:
                logger.info(f"🔗 Initializing MCP server: {server_name}")
                
                # Create MCP client using Strands built-in support
                mcp_client = MCPClient(lambda: stdio_client(
                    StdioServerParameters(
                        command=server_config["command"],
                        args=server_config["args"]
                    )
                ))
                
                # Start the MCP client (this keeps the session running)
                mcp_client.start()
                
                # Get tools from the running MCP client
                mcp_tools = mcp_client.list_tools_sync()
                all_tools.extend(mcp_tools)
                self.mcp_clients.append(mcp_client)
                
                logger.info(f"✅ MCP server {server_name} added {len(mcp_tools)} tools")
            
            except Exception as e:
                # Clean up the error message for common issues
                error_msg = str(e).lower()
                
                if "not found in the package registry" in error_msg:
                    logger.warning(f"❌ MCP server {server_name}: Package not found. Try installing it first or check the command.")
                elif "connection closed" in error_msg or "initialization failed" in error_msg:
                    logger.warning(f"❌ MCP server {server_name}: Failed to connect. Server may not be available.")
                elif "permission denied" in error_msg:
                    logger.warning(f"❌ MCP server {server_name}: Permission denied. Check file permissions.")
                elif "command not found" in error_msg or "no such file" in error_msg:
                    logger.warning(f"❌ MCP server {server_name}: Command '{server_config['command']}' not found.")
                else:
                    logger.warning(f"❌ MCP server {server_name}: {str(e)[:100]}...")
                
                logger.debug(f"Full MCP error for {server_name}: {e}")  # Full error only in debug mode
        
        # Update loaded tools and create agent
        self.loaded_tools = all_tools
        self._create_agent()
        
        logger.info(f"🎉 Agent created with {len(self.loaded_tools)} total tools ({len(self.mcp_clients)} MCP servers)")
    
    async def stop_mcp_clients(self):
        """Stop all MCP clients when shutting down."""
        for mcp_client in self.mcp_clients:
            try:
                mcp_client.stop()
                logger.info("🔒 MCP client stopped")
            except Exception as e:
                logger.warning(f"Error stopping MCP client: {e}")
    
    def _create_agent(self):
        """Create the Strands agent with all loaded tools."""
        # Don't call super().__init__ here since we're not using the BaseAgent pattern normally
        model_config = self.agent_config.get("model")
        model = get_or_create_ai_model(ModelConfig.from_config(model_config))
        self.agent = Agent(
            system_prompt=self._build_system_prompt(),
            tools=self.loaded_tools,
            model=model
        )
    
    async def invoke(self, query: str) -> str:
        """
        Execute agent query with proper thread safety using isolated agent instances.
        
        This method creates a new agent instance for each request to avoid
        shared state issues that cause the AWS Bedrock validation errors.
        """
        if not self.agent:
            return "Agent not initialized yet"
        
        try:
            import asyncio
            # Create an isolated agent instance for this request to avoid shared state
            isolated_agent = self._create_isolated_agent()
            response = await asyncio.to_thread(isolated_agent, query)
            return str(response)
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return f"Agent error: {str(e)}"
    
    def _create_isolated_agent(self) -> Agent:
        """
        Create an isolated agent instance for thread-safe execution.
        
        This creates a completely new Agent instance with fresh conversation
        history to prevent the AWS Bedrock validation errors caused by
        shared state between concurrent requests.
        """
        if not self.agent:
            raise RuntimeError("Base agent not initialized yet")
            
        # Get the same configuration used for the original agent
        model = self.agent_config.get("model")
        model_config = self.agent_config.get("model")
        model = get_or_create_ai_model(ModelConfig.from_config(model_config))
        
        # Create a new isolated agent instance with the same tools and prompt
        isolated_agent = Agent(
            system_prompt=self._build_system_prompt(),
            tools=self.loaded_tools.copy(),  # Copy tools list to avoid shared references
            model=model
        )
        
        return isolated_agent
    
    def _load_strands_tools(self):
        """Load Strands tools from configuration."""
        strands_tools_config = self.agent_config.get("strands_tools", [])
        
        logger.info(f"📋 Discovered {len(_AVAILABLE_TOOLS)} available Strands tools: {list(_AVAILABLE_TOOLS.keys())}")
        
        # Load requested tools
        for tool_name in strands_tools_config:
            if tool_name in _AVAILABLE_TOOLS:
                self.loaded_tools.append(_AVAILABLE_TOOLS[tool_name])
                logger.info(f"✅ Loaded Strands tool: {tool_name}")
            else:
                logger.warning(f"⚠️ Unknown Strands tool: {tool_name}. Available: {list(_AVAILABLE_TOOLS.keys())}")
    
    def _build_system_prompt(self) -> str:
        """Build system prompt based on available tools."""
        tool_descriptions = []
        
        # Strands tools
        strands_tools = self.agent_config.get("strands_tools", [])
        if strands_tools:
            tool_descriptions.append("Strands Tools Available:")
            
            # Group tools by category for better organization
            categories = {
                "File Operations": ["editor", "file_read", "file_write"],
                "Utilities": ["calculator", "current_time", "load_tool"],
                "Shell & System": ["environment", "shell", "cron"],
                "Code Interpretation": ["python_repl"],
                "Web & Network": ["http_request", "slack"],
                "Multi-modal": ["image_reader", "generate_image", "nova_reels", "speak"],
                "AWS Services": ["use_aws"],
                "RAG & Memory": ["retrieve", "memory", "mem0_memory"],
                "Agents & Workflows": ["agent_graph", "journal", "swarm", "stop", "think", "use_llm", "workflow"]
            }
            
            for category, tools in categories.items():
                loaded_in_category = [tool for tool in tools if tool in strands_tools]
                if loaded_in_category:
                    tool_descriptions.append(f"  {category}: {', '.join(loaded_in_category)}")
        
        # MCP tools 
        mcp_servers = self.agent_config.get("mcp_servers", {})
        if mcp_servers:
            tool_descriptions.append("\nMCP Server Tools:")
            for server_name in mcp_servers.keys():
                tool_descriptions.append(f"  - {server_name}: MCP server for external capabilities")
        
        tools_section = "\n".join(tool_descriptions)
        
        return f"""You are a configurable agent with access to multiple types of tools.

Available Tools:
{tools_section}

Instructions:
1. Use the appropriate tools based on user requests and available capabilities
2. File Operations: Use editor, file_read, file_write for document management
3. Code & Development: Use python_repl for code execution, shell for system commands
4. Web & Network: Use http_request for API calls, web data fetching
5. Multi-modal: Use image tools for visual content, speak for audio output
6. AWS Services: Use use_aws for cloud service interactions
7. Calculations & Time: Use calculator for math, current_time for temporal queries
8. Agent Coordination: Use workflow, swarm, agent_graph for complex orchestration
9. Memory & RAG: Use memory, retrieve for persistent knowledge and retrieval
10. Combine multiple tools for comprehensive solutions
11. Provide clear explanations of your actions and tool usage
12. Handle tool failures gracefully with helpful error messages

You are a versatile agent capable of file operations, code execution, web interactions, multi-modal processing, cloud services, and complex agent workflows.
"""


class ConfigurableAgentExecutor(BaseAgentExecutor):
    """A2A AgentExecutor implementation for configurable agent."""
    
    def __init__(self, agent_key: str, task_store: InMemoryTaskStore):
        # Create configurable agent instance with the specific agent key
        self.configurable_agent = ConfigurableAgent(agent_key)
        
        # Initialize base executor with the configurable agent
        super().__init__(self.configurable_agent, task_store, "configurable task")
        
        self.agent_key = agent_key
    
    async def initialize_with_all_tools(self):
        """Initialize MCP servers and create the agent with all tools."""
        await self.configurable_agent.initialize_with_mcp()
        
        # Now the configurable_agent.agent is properly initialized
        # The BaseAgentExecutor will use self.agent which points to configurable_agent
    
    def get_empty_query_message(self) -> str:
        """Get the message to return when query is empty."""
        return f"Please provide a task for the {self.agent_key} agent using available Strands tools or MCP services."
    
    def get_working_message(self) -> str:
        """Get the message to show when task is working."""
        return f"Processing request with {self.agent_key} agent tools..."


def create_agent_card(agent_key: str):
    """Create agent card dynamically based on configuration for a specific agent."""
    agent_config = load_agent_config(agent_key, "configurable_agent", fallback_port=8888)
    
    skills = []
    
    # Create skills for Strands tools
    strands_tools = agent_config.get("strands_tools", [])
    if strands_tools:
        strands_skill = AgentSkill(
            id="strands_tools",
            name="Strands Framework Tools",
            description=f"Specialized tools for {agent_key}: {', '.join(strands_tools)}",
            tags=["strands", "tools", "framework", agent_key] + strands_tools,
            examples=[
                f"Use {agent_key} for specialized tasks",
                f"Perform {agent_key} operations",
                f"Execute {agent_key} functions"
            ]
        )
        skills.append(strands_skill)
    
    # Create skills for MCP servers
    mcp_servers = agent_config.get("mcp_servers", {})
    for server_name, server_config in mcp_servers.items():
        mcp_skill = AgentSkill(
            id=f"mcp_{server_name}",
            name=f"MCP {server_name.title()} Server", 
            description=f"External capabilities via {server_name} MCP server for {agent_key}",
            tags=["mcp", server_name, "external", agent_key],
            examples=[
                f"Use {server_name} services for {agent_key}",
                f"Access {server_name} capabilities"
            ]
        )
        skills.append(mcp_skill)
    
    # If no specific tools configured, create a general skill
    if not skills:
        general_skill = AgentSkill(
            id="general_assistance",
            name="General Assistance",
            description=f"Configurable {agent_key} agent ready to be equipped with tools",
            tags=["configurable", "flexible", "general", agent_key],
            examples=[
                f"Configure {agent_key} with tools",
                f"What can {agent_key} help you with?"
            ]
        )
        skills.append(general_skill)
    
    return create_base_agent_card(
        name=agent_config.get("name", f"{agent_key}_agent"),
        description=agent_config.get("description", f"A flexible {agent_key} agent configurable with Strands tools and MCP servers"),
        url=f"http://localhost:{agent_config.get('port', 8888)}/",  # Use fallback port if needed
        skills=skills
    )


def main():
    """Main entry point for the configurable agent."""
    try:
        # Parse command line arguments to get the agent key
        agent_key = parse_agent_key(
            "configurable",
            "Start a configurable A2A agent with specific tools and capabilities"
        )
        
        # Load the specific agent configuration with 8888 as fallback port
        agent_config = load_agent_config(agent_key, "configurable_agent", fallback_port=8888)
        
        # Extract port from the specific agent configuration (now guaranteed to have a port)
        port = agent_config.get("port", 8888)  # Double fallback just in case
        
        # Create agent card for the specific agent
        agent_card = create_agent_card(agent_key)
        
        # Create task store
        task_store = InMemoryTaskStore()
        
        # Create executor with the specific agent key
        executor = ConfigurableAgentExecutor(agent_key, task_store)
        
        # Create request handler
        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=task_store
        )
        
        # Create Starlette application
        app = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler
        )
        
        # Build the Starlette app
        starlette_app = app.build()
        
        # Add startup event to initialize all tools
        @starlette_app.on_event("startup")
        async def startup_event():
            await executor.initialize_with_all_tools()
            logger.info(f"✅ Configurable agent '{agent_key}' startup complete")
        
        @starlette_app.on_event("shutdown")
        async def shutdown_event():
            await executor.configurable_agent.stop_mcp_clients()
            logger.info(f"🔒 Configurable agent '{agent_key}' shutdown complete")
        
        # Log startup
        logger.info(f"🔧 Configurable Agent '{agent_key}' starting...")
        logger.info(f"📋 Agent Card: {agent_card.name} v{agent_card.version}")
        logger.info(f"🎯 Skills: {[skill.name for skill in agent_card.skills]}")
        logger.info(f"🚀 Server starting on http://localhost:{port}")
        
        # Start server
        uvicorn.run(
            starlette_app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"Failed to start configurable agent: {e}")
        raise


if __name__ == "__main__":
    main() 
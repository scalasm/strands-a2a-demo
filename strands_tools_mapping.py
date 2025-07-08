#!/usr/bin/env python3
"""
Strands Tools Static Mapping

This file contains static imports for all available strands tools using the correct
import pattern: from strands_tools import module1, module2, ...

This ensures each imported module has the proper TOOL_SPEC attribute attached,
and extracts the properly decorated functions for the Agent framework.
"""

import logging
from typing import Dict, Any, Callable

# Set up logging
logger = logging.getLogger(__name__)

# Single import for all strands tools modules
from strands_tools import (
    agent_graph,
    batch, 
    calculator,
    cron,
    current_time,
    editor,
    environment,
    file_read,
    file_write,
    generate_image,
    http_request,
    image_reader,
    journal,
    load_tool,
    mem0_memory,
    memory,
    nova_reels,
    python_repl,
    retrieve,
    shell,
    slack,
    sleep,
    speak,
    stop,
    swarm,
    think,
    use_aws,
    use_llm,
    workflow,
)

def _get_tool_function(module, tool_name: str):
    """Get the properly decorated tool function from a module."""
    # First try: function with same name as module
    if hasattr(module, tool_name):
        func = getattr(module, tool_name)
        if callable(func) and hasattr(func, 'TOOL_SPEC'):
            return func
    
    # Second try: any function with TOOL_SPEC
    for attr_name in dir(module):
        if not attr_name.startswith('_'):
            attr = getattr(module, attr_name)
            if callable(attr) and hasattr(attr, 'TOOL_SPEC'):
                return attr
    
    # If no function found, return None (this is expected for some modules)
    return None

# Create the tools mapping 
STRANDS_TOOLS_MAPPING = {
    'agent_graph': agent_graph,
    'batch': batch,
    'calculator': calculator,
    'cron': cron,
    'current_time': current_time,
    'editor': editor,
    'environment': environment,
    'file_read': file_read,
    'file_write': file_write,
    'generate_image': generate_image,
    'http_request': http_request,
    'image_reader': image_reader,
    'journal': journal,
    'load_tool': load_tool,
    'mem0_memory': mem0_memory,
    'memory': memory,
    'nova_reels': nova_reels,
    'python_repl': python_repl,
    'retrieve': retrieve,
    'shell': shell,
    'slack': slack,
    'sleep': sleep,
    'speak': speak,
    'stop': stop,
    'swarm': swarm,
    'think': think,
    'use_aws': use_aws,
    'use_llm': use_llm,
    'workflow': workflow,
}

def get_available_tool_names():
    """Get list of successfully loaded tool names."""
    return list(STRANDS_TOOLS_MAPPING.keys())

def get_tool_by_name(tool_name: str) -> Callable:
    """
    Get a specific tool by name.
    
    Args:
        tool_name: Name of the tool to retrieve
        
    Returns:
        The tool function if found, None otherwise
    """
    return STRANDS_TOOLS_MAPPING.get(tool_name)

def is_tool_available(tool_name: str) -> bool:
    """Check if a tool is available."""
    return tool_name in STRANDS_TOOLS_MAPPING

def get_tools_list(tool_names: list) -> list:
    """
    Get a list of tool functions for the specified tool names.
    
    Args:
        tool_names: List of tool names to retrieve
        
    Returns:
        List of tool functions
    """
    return [STRANDS_TOOLS_MAPPING[name] for name in tool_names if name in STRANDS_TOOLS_MAPPING]

# Log summary on import
logger.info(f"📦 Strands Tools Mapping: {len(STRANDS_TOOLS_MAPPING)} tools available")
logger.info(f"🔧 Available tools: {', '.join(sorted(STRANDS_TOOLS_MAPPING.keys()))}") 
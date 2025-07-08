#!/usr/bin/env python3
"""
Log Viewer Utility - Read agent logs and status for web interfaces

This module provides utilities to read agent log files and status from state files
for display in web interfaces.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """Represents a single log entry."""
    timestamp: str
    level: str
    message: str
    raw_line: str


@dataclass
class AgentLogInfo:
    """Information about an agent's logs."""
    name: str
    log_file: str
    status: str  # running, stopped, error
    pid: Optional[int]
    port: int
    recent_logs: List[LogEntry]
    last_updated: Optional[str]


class LogViewer:
    """Utility class for reading agent logs and status."""
    
    def __init__(self, logs_dir: str = "./logs", state_file: str = "./state/agents.json"):
        self.logs_dir = Path(logs_dir)
        self.state_file = Path(state_file)
    
    def get_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get agent status from state file."""
        try:
            if not self.state_file.exists():
                return {}
            
            with open(self.state_file, 'r') as f:
                state_data = json.load(f)
            
            return state_data.get("agents", {})
        except Exception as e:
            logger.error(f"Error reading state file: {e}")
            return {}
    
    def get_configured_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get configured agents from config file."""
        try:
            from base_agent import load_config
            config = load_config()
            
            agents = {}
            
            # Add configurable agents
            configurable_agents = config.get("configurable_agent", {})
            for agent_name, agent_config in configurable_agents.items():
                agents[agent_name] = {
                    "port": agent_config.get("port"),
                    "type": "configurable",
                    "description": agent_config.get("description", "")
                }
            
            # Add coordinator agents
            coordinator_agents = config.get("coordinator_agent", {})
            for coordinator_name, coordinator_config in coordinator_agents.items():
                agents[coordinator_name] = {
                    "port": coordinator_config.get("port"),
                    "type": "coordinator", 
                    "description": coordinator_config.get("description", ""),
                    "coordinated_agents": coordinator_config.get("coordinated_agents", [])
                }
            
            return agents
        except Exception as e:
            logger.error(f"Error reading config: {e}")
            return {}
    
    def tail_log_file(self, log_file: Path, lines: int = 50) -> List[LogEntry]:
        """Read the last N lines from a log file."""
        if not log_file.exists():
            return []
        
        try:
            # Read last N lines efficiently
            with open(log_file, 'r', encoding='utf-8') as f:
                # Go to end of file
                f.seek(0, 2)
                file_size = f.tell()
                
                # If file is empty
                if file_size == 0:
                    return []
                
                # Read chunks from end until we have enough lines
                buffer_size = min(8192, file_size)
                lines_found = []
                pos = file_size
                
                while len(lines_found) < lines and pos > 0:
                    # Move position back
                    pos = max(0, pos - buffer_size)
                    f.seek(pos)
                    
                    # Read chunk
                    chunk = f.read(min(buffer_size, file_size - pos))
                    
                    # Split into lines and prepend to our list
                    chunk_lines = chunk.split('\n')
                    if pos > 0:
                        # Remove first partial line (it's from previous chunk)
                        chunk_lines = chunk_lines[1:]
                    
                    lines_found = chunk_lines + lines_found
                
                # Take only the last N lines and remove empty ones
                recent_lines = [line.strip() for line in lines_found[-lines:] if line.strip()]
                
                # Parse log entries
                log_entries = []
                for line in recent_lines:
                    entry = self._parse_log_line(line)
                    if entry:
                        log_entries.append(entry)
                
                return log_entries
                
        except Exception as e:
            logger.error(f"Error reading log file {log_file}: {e}")
            return []
    
    def _parse_log_line(self, line: str) -> Optional[LogEntry]:
        """Parse a log line into structured data."""
        try:
            # Try to parse standard Python logging format: timestamp - logger - level - message
            if ' - ' in line:
                parts = line.split(' - ', 3)
                if len(parts) >= 3:
                    timestamp = parts[0]
                    level = parts[2] if len(parts) > 2 else "INFO"
                    message = parts[3] if len(parts) > 3 else parts[2]
                    
                    return LogEntry(
                        timestamp=timestamp,
                        level=level.strip(),
                        message=message.strip(),
                        raw_line=line
                    )
            
            # Fallback for non-standard format
            return LogEntry(
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                level="INFO",
                message=line,
                raw_line=line
            )
            
        except Exception:
            return None
    
    def get_all_agent_logs(self, lines_per_agent: int = 20) -> List[AgentLogInfo]:
        """Get logs and status for all configured agents."""
        configured_agents = self.get_configured_agents()
        agent_status = self.get_agent_status()
        
        agent_logs = []
        
        for agent_name, config_info in configured_agents.items():
            # Get log file path
            log_file = self.logs_dir / f"{agent_name}.log"
            
            # Get status from state file
            status_info = agent_status.get(agent_name, {})
            
            # Determine status
            if status_info.get("pid"):
                # Check if PID is still valid
                try:
                    import psutil
                    if psutil.pid_exists(status_info["pid"]):
                        status = "running"
                    else:
                        status = "stopped"
                except:
                    status = "unknown"
            else:
                status = "stopped"
            
            # Get recent logs
            recent_logs = self.tail_log_file(log_file, lines_per_agent)
            
            # Get log file modification time for accurate last_updated
            log_last_updated = None
            if log_file.exists():
                try:
                    import os
                    mtime = os.path.getmtime(log_file)
                    log_last_updated = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
                except Exception as e:
                    logger.debug(f"Could not get modification time for {log_file}: {e}")
                    log_last_updated = status_info.get("started_at")
            else:
                log_last_updated = status_info.get("started_at")
            
            agent_logs.append(AgentLogInfo(
                name=agent_name,
                log_file=str(log_file),
                status=status,
                pid=status_info.get("pid"),
                port=config_info.get("port", 0),
                recent_logs=recent_logs,
                last_updated=log_last_updated
            ))
        
        return agent_logs
    
    def get_agent_log(self, agent_name: str, lines: int = 50) -> Optional[AgentLogInfo]:
        """Get logs for a specific agent."""
        all_logs = self.get_all_agent_logs(lines)
        for agent_log in all_logs:
            if agent_log.name == agent_name:
                return agent_log
        return None 
#!/usr/bin/env python3
"""
Agent Process Manager - Manages starting and stopping A2A agents as processes

This module provides functionality to:
- Start all configured agents as separate processes
- Stop all running agent processes
- Set up automatic log rotation for each agent
- Monitor agent process health
"""

import os
import sys
import signal
import subprocess
import time
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import psutil
import argparse

from model_config import load_config

logger = logging.getLogger(__name__)


@dataclass
class AgentProcess:
    """Information about a running agent process."""
    name: str
    port: int
    process: subprocess.Popen
    log_file: Path
    command: List[str]


class AgentProcessManager:
    """Manages A2A agent processes with automatic logging."""
    
    def __init__(self):
        """Initialize the process manager."""
        self.config = load_config()
        self.processes: Dict[str, AgentProcess] = {}
        self.log_directory = Path(self.config.get("logging", {}).get("log_directory", "./logs"))
        
        # Set up state directory and file for persistent PID tracking
        self.state_directory = Path("./state")
        self.state_file = self.state_directory / "agents.json"
        
        # Ensure directories exist
        self.log_directory.mkdir(exist_ok=True)
        self.state_directory.mkdir(exist_ok=True)
        
        logger.info(f"Agent logs will be stored in: {self.log_directory.absolute()}")
        logger.info(f"Agent state will be stored in: {self.state_file.absolute()}")
        
        # Load existing state and cleanup stale PIDs
        self._load_agent_state()
        self._cleanup_stale_pids()
    
    def _write_log_entry(self, log_file: Path, entry_type: str, agent_name: str, details: Dict[str, Any]) -> None:
        """Write a standardized log entry to the agent log file."""
        with open(log_file, 'a') as log_handle:
            log_handle.write(f"\n{'='*60}\n")
            log_handle.write(f"{entry_type}: {agent_name}\n")
            log_handle.write(f"Date/Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Write all details in a consistent order
            for key, value in details.items():
                if value is not None:
                    log_handle.write(f"{key}: {value}\n")
            
            log_handle.write(f"{'='*60}\n\n")
            log_handle.flush()
    
    def _load_agent_state(self) -> None:
        """Load agent state from persistent storage."""
        if not self.state_file.exists():
            logger.debug("No existing agent state file found")
            return
        
        try:
            with open(self.state_file, 'r') as f:
                state_data = json.load(f)
            
            agents_data = state_data.get("agents", {})
            logger.info(f"Loading state for {len(agents_data)} agents from {self.state_file}")
            
            for agent_name, agent_data in agents_data.items():
                pid = agent_data.get("pid")
                port = agent_data.get("port")
                command = agent_data.get("command", [])
                log_file_path = agent_data.get("log_file")
                
                if pid and port and command and log_file_path:
                    try:
                        # Check if the process still exists
                        process = psutil.Process(pid)
                        if process.is_running():
                            # Create a mock subprocess.Popen object for compatibility
                            class MockProcess:
                                def __init__(self, pid, real_process):
                                    self.pid = pid
                                    self._real_process = real_process
                                
                                def poll(self):
                                    return None  # Always return None (running)
                                
                                def terminate(self):
                                    return self._real_process.terminate()
                                
                                def kill(self):
                                    return self._real_process.kill()
                                
                                def wait(self, timeout=None):
                                    return self._real_process.wait(timeout)
                            
                            mock_process = MockProcess(pid, process)
                            
                            agent_process = AgentProcess(
                                name=agent_name,
                                port=port,
                                process=mock_process,
                                log_file=Path(log_file_path),
                                command=command
                            )
                            
                            self.processes[agent_name] = agent_process
                            logger.debug(f"Restored agent {agent_name} (PID: {pid})")
                        else:
                            logger.debug(f"Agent {agent_name} PID {pid} no longer running")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        logger.debug(f"Agent {agent_name} PID {pid} not found or inaccessible")
                else:
                    logger.warning(f"Invalid state data for agent {agent_name}")
                    
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load agent state: {e}")
    
    def _save_agent_state(self) -> None:
        """Save current agent state to persistent storage."""
        try:
            agents_data = {}
            
            for agent_name, agent_process in self.processes.items():
                # Only save if process is still running
                if agent_process.process.poll() is None:
                    agents_data[agent_name] = {
                        "pid": agent_process.process.pid,
                        "port": agent_process.port,
                        "command": agent_process.command,
                        "log_file": str(agent_process.log_file),
                        "started_at": time.strftime('%Y-%m-%d %H:%M:%S')
                    }
            
            state_data = {
                "agents": agents_data,
                "last_updated": time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Write atomically by writing to a temp file first
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            # Atomically replace the old file
            temp_file.replace(self.state_file)
            logger.debug(f"Saved state for {len(agents_data)} agents")
            
        except Exception as e:
            logger.error(f"Failed to save agent state: {e}")
    
    def _cleanup_stale_pids(self) -> None:
        """Remove agents from memory whose PIDs are no longer valid."""
        stale_agents = []
        
        for agent_name, agent_process in self.processes.items():
            try:
                # Check if process still exists and is running
                if not psutil.pid_exists(agent_process.process.pid):
                    stale_agents.append(agent_name)
                    logger.info(f"🧹 Removing stale agent {agent_name} (PID {agent_process.process.pid} no longer exists)")
                elif agent_process.process.poll() is not None:
                    stale_agents.append(agent_name)
                    logger.info(f"🧹 Removing terminated agent {agent_name} (PID {agent_process.process.pid})")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                stale_agents.append(agent_name)
                logger.info(f"🧹 Removing inaccessible agent {agent_name} (PID {agent_process.process.pid})")
        
        # Remove stale agents
        for agent_name in stale_agents:
            del self.processes[agent_name]
        
        if stale_agents:
            logger.info(f"🎯 Cleaned up {len(stale_agents)} stale agent references")
            # Save the updated state
            self._save_agent_state()
    
    def _stop_agent_by_pid(self, agent_name: str, pid: int, port: int, command: List[str]) -> bool:
        """Stop an agent by PID when we don't have a subprocess.Popen object."""
        try:
            # Check if PID exists
            if not psutil.pid_exists(pid):
                logger.warning(f"⚠️  Agent {agent_name} PID {pid} no longer exists")
                self._write_log_entry(
                    self._setup_agent_logging(agent_name),
                    "AGENT STOP ATTEMPT",
                    agent_name,
                    {
                        "Command": ' '.join(command),
                        "Port": port,
                        "PID": pid,
                        "Status": "Failed - PID no longer exists"
                    }
                )
                return True  # Consider this success since the process is already gone
            
            process = psutil.Process(pid)
            if not process.is_running():
                logger.warning(f"⚠️  Agent {agent_name} PID {pid} is not running")
                self._write_log_entry(
                    self._setup_agent_logging(agent_name),
                    "AGENT STOP ATTEMPT", 
                    agent_name,
                    {
                        "Command": ' '.join(command),
                        "Port": port,
                        "PID": pid,
                        "Status": "Failed - Process not running"
                    }
                )
                return True  # Process already stopped
            
            # Try graceful termination first
            process.terminate()
            logger.info(f"🛑 Sent SIGTERM to agent {agent_name} (PID: {pid})")
            
            # Wait up to 5 seconds for graceful shutdown
            try:
                process.wait(timeout=5)
                logger.info(f"✅ Agent {agent_name} terminated gracefully")
            except psutil.TimeoutExpired:
                # Force kill if graceful shutdown failed
                logger.warning(f"⚠️  Force killing agent {agent_name} (PID: {pid})")
                process.kill()
                process.wait()
                logger.info(f"✅ Agent {agent_name} force killed")
            
            # Log successful stop
            self._write_log_entry(
                self._setup_agent_logging(agent_name),
                "AGENT STOP",
                agent_name,
                {
                    "Command": ' '.join(command),
                    "Port": port, 
                    "PID": pid,
                    "Status": "Stopped successfully"
                }
            )
            
            return True
            
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning(f"⚠️  Could not stop agent {agent_name} PID {pid}: {e}")
            self._write_log_entry(
                self._setup_agent_logging(agent_name),
                "AGENT STOP ATTEMPT",
                agent_name,
                {
                    "Command": ' '.join(command),
                    "Port": port,
                    "PID": pid,
                    "Status": f"Failed - {str(e)}"
                }
            )
            return False
        except Exception as e:
            logger.error(f"❌ Error stopping agent {agent_name} PID {pid}: {e}")
            return False
    
    def _get_agent_commands(self) -> List[Tuple[str, int, List[str]]]:
        """Get commands to start all configured agents."""
        commands = []
        
        # Add configurable agents
        configurable_agents = self.config.get("configurable_agent", {})
        for agent_name, agent_config in configurable_agents.items():
            port = agent_config.get("port")
            if port:
                command = ["uv", "run", "configurable-agent", agent_name]
                commands.append((agent_name, port, command))
        
        # Add coordinator agents
        coordinator_agents = self.config.get("coordinator_agent", {})
        for coordinator_name, coordinator_config in coordinator_agents.items():
            port = coordinator_config.get("port")
            if port:
                command = ["uv", "run", "coordinator-agent", coordinator_name]
                commands.append((coordinator_name, port, command))
        
        return commands
    
    def _get_agent_dependencies(self) -> Dict[str, List[str]]:
        """Get agent dependencies from configuration."""
        dependencies = {}
        
        # Configurable agents have no dependencies
        configurable_agents = self.config.get("configurable_agent", {})
        for agent_name in configurable_agents.keys():
            dependencies[agent_name] = []
        
        # Coordinator agents may have dependencies
        coordinator_agents = self.config.get("coordinator_agent", {})
        for coordinator_name, coordinator_config in coordinator_agents.items():
            coordinated = coordinator_config.get("coordinated_agents", [])
            dependencies[coordinator_name] = coordinated
        
        return dependencies
    
    def _get_start_order(self) -> List[List[str]]:
        """Get agents grouped by dependency level for ordered starting."""
        dependencies = self._get_agent_dependencies()
        
        # Group agents by dependency level
        level_0 = []  # No dependencies (regular agents)
        level_1 = []  # Depend on level 0 (coordinators)
        
        for agent_name, deps in dependencies.items():
            if not deps:  # No dependencies
                level_0.append(agent_name)
            else:  # Has dependencies
                level_1.append(agent_name)
        
        # Return as list of levels (could be extended for deeper dependencies)
        levels = []
        if level_0:
            levels.append(level_0)
        if level_1:
            levels.append(level_1)
        
        return levels
    
    def _setup_agent_logging(self, agent_name: str) -> Path:
        """Set up log file for an agent with rotation."""
        log_file = self.log_directory / f"{agent_name}.log"
        
        # Create log file if it doesn't exist
        if not log_file.exists():
            log_file.touch()
            logger.info(f"Created log file for {agent_name}: {log_file}")
        
        return log_file
    
    def _is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use."""
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                    return True
            return False
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            # Fallback: try to bind to the port
            import socket
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return False
            except OSError:
                return True
    
    def start_all_agents(self, force: bool = False) -> bool:
        """
        Start all configured agents with dependency-aware ordering.
        
        Args:
            force: If True, kill existing processes on the same ports first
            
        Returns:
            True if all agents started successfully, False otherwise
        """
        logger.info("🚀 Starting all A2A agents with dependency-aware ordering...")
        
        commands_dict = {name: (port, command) for name, port, command in self._get_agent_commands()}
        if not commands_dict:
            logger.warning("No agents configured to start")
            return True
        
        start_levels = self._get_start_order()
        total_count = len(commands_dict)
        success_count = 0
        
        for level_index, level_agents in enumerate(start_levels):
            if level_index == 0:
                logger.info(f"📋 Level {level_index + 1}: Starting independent agents: {', '.join(level_agents)}")
            else:
                logger.info(f"📋 Level {level_index + 1}: Starting coordinator agents: {', '.join(level_agents)}")
            
            # Start all agents in this level
            level_success_count = 0
            for agent_name in level_agents:
                if agent_name not in commands_dict:
                    logger.warning(f"⚠️  Agent {agent_name} not found in commands")
                    continue
                
                port, command = commands_dict[agent_name]
                
                if self._is_port_in_use(port) and not force:
                    logger.warning(f"⚠️  Port {port} is already in use for {agent_name}, skipping")
                    continue
                elif self._is_port_in_use(port) and force:
                    logger.info(f"🔄 Port {port} in use, killing existing process for {agent_name}")
                    self._kill_process_on_port(port)
                    time.sleep(1)  # Give time for the port to be released
                
                success = self._start_agent(agent_name, port, command)
                if success:
                    level_success_count += 1
                    success_count += 1
                    logger.info(f"✅ Started {agent_name} on port {port}")
                else:
                    logger.error(f"❌ Failed to start {agent_name} on port {port}")
            
            # Wait for this level to be ready before starting the next level
            if level_success_count > 0 and level_index < len(start_levels) - 1:
                logger.info(f"⏳ Waiting for level {level_index + 1} agents to be ready before starting coordinators...")
                
                # Wait for agents in this level to be ready
                ready_agents = []
                for agent_name in level_agents:
                    if agent_name in self.processes:  # Only check agents we successfully started
                        ready_agents.append(agent_name)
                
                if ready_agents:
                    agents_ready = self.wait_for_specific_agents_ready(ready_agents, timeout=30)
                    if agents_ready:
                        logger.info(f"✅ Level {level_index + 1} agents are ready!")
                    else:
                        logger.warning(f"⚠️  Some level {level_index + 1} agents may not be fully ready, continuing...")
        
        logger.info(f"🎯 Started {success_count}/{total_count} agents successfully")
        return success_count == total_count
    
    def _kill_process_on_port(self, port: int):
        """Kill any process listening on the specified port."""
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                    try:
                        process = psutil.Process(conn.pid)
                        process.terminate()
                        logger.info(f"Terminated process {conn.pid} on port {port}")
                        time.sleep(0.5)
                        if process.is_running():
                            process.kill()
                            logger.info(f"Killed process {conn.pid} on port {port}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                        logger.debug(f"Could not kill process on port {port}: {e}")
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            logger.debug(f"Could not check connections for port {port}")
    
    def _start_agent(self, agent_name: str, port: int, command: List[str]) -> bool:
        """Start a single agent process."""
        try:
            # Set up logging
            log_file = self._setup_agent_logging(agent_name)
            
            # Write start log entry
            self._write_log_entry(log_file, "AGENT START", agent_name, {
                "Command": ' '.join(command),
                "Port": port
            })
            
            # Start the process with output redirected to log file
            with open(log_file, 'a') as log_handle:
                process = subprocess.Popen(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,  # Line buffered
                    cwd=os.getcwd()
                )
            
            # Store process information
            agent_process = AgentProcess(
                name=agent_name,
                port=port,
                process=process,
                log_file=log_file,
                command=command
            )
            self.processes[agent_name] = agent_process
            
            # Give the process a moment to start
            time.sleep(0.5)
            
            # Check if process is still running
            if process.poll() is None:
                # Log success with PID
                self._write_log_entry(log_file, "AGENT START SUCCESS", agent_name, {
                    "PID": process.pid,
                    "Status": "Started successfully"
                })
                
                # Save the updated state to persistent storage
                self._save_agent_state()
                
                logger.debug(f"Process for {agent_name} started successfully (PID: {process.pid})")
                return True
            else:
                # Log failure 
                self._write_log_entry(log_file, "AGENT START FAILURE", agent_name, {
                    "Status": f"Failed to start (exit code: {process.returncode})"
                })
                
                logger.error(f"Process for {agent_name} exited immediately with code {process.returncode}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start {agent_name}: {e}")
            return False
    
    def stop_all_agents(self) -> bool:
        """
        Stop all running agent processes, including those tracked in state file.
        
        Returns:
            True if all agents stopped successfully, False otherwise
        """
        # Get all agents from both memory and state file
        status = self.get_agent_status()
        agents_to_stop = [(name, info) for name, info in status.items() if info["running"]]
        
        if not agents_to_stop:
            logger.info("No running agents to stop")
            return True
        
        logger.info(f"🛑 Stopping {len(agents_to_stop)} running agent processes...")
        
        success_count = 0
        total_count = len(agents_to_stop)
        
        for agent_name, agent_info in agents_to_stop:
            if agent_name in self.processes:
                # Stop managed agent
                success = self._stop_agent(self.processes[agent_name])
            elif agent_info["pid"]:
                # Stop agent by PID from state file
                success = self._stop_agent_by_pid(
                    agent_name, 
                    agent_info["pid"], 
                    agent_info["port"], 
                    agent_info["command"]
                )
            else:
                # Fallback: kill process on port
                self._kill_process_on_port(agent_info["port"])
                success = True
            
            if success:
                success_count += 1
                logger.info(f"✅ Stopped {agent_name}")
            else:
                logger.error(f"❌ Failed to stop {agent_name}")
        
        # Clear memory and save updated state
        self.processes.clear()
        self._save_agent_state()
        
        logger.info(f"🎯 Stopped {success_count}/{total_count} agents successfully")
        return success_count == total_count
    
    def _stop_agent(self, agent_process: AgentProcess) -> bool:
        """Stop a single agent process."""
        try:
            process = agent_process.process
            
            if process.poll() is not None:
                logger.debug(f"Process for {agent_process.name} already terminated")
                return True
            
            # Try graceful termination first
            process.terminate()
            
            # Wait up to 5 seconds for graceful shutdown
            try:
                process.wait(timeout=5)
                logger.debug(f"Process for {agent_process.name} terminated gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown failed
                logger.warning(f"Force killing {agent_process.name} process")
                process.kill()
                process.wait()
            
            # Log shutdown using standardized format
            self._write_log_entry(
                agent_process.log_file,
                "AGENT STOP",
                agent_process.name,
                {
                    "Command": ' '.join(agent_process.command),
                    "Port": agent_process.port,
                    "PID": agent_process.process.pid,
                    "Status": "Stopped successfully"
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop {agent_process.name}: {e}")
            self._write_log_entry(
                agent_process.log_file,
                "AGENT STOP FAILURE",
                agent_process.name,
                {
                    "Command": ' '.join(agent_process.command),
                    "Port": agent_process.port,
                    "PID": agent_process.process.pid,
                    "Status": f"Failed to stop: {str(e)}"
                }
            )
            return False
    
    def kill_all_agents_on_configured_ports(self) -> bool:
        """
        Kill any processes running on ports configured for agents.
        This is useful for cleanup when processes weren't started by this manager.
        
        Returns:
            True if cleanup was successful, False otherwise
        """
        logger.info("🧹 Cleaning up any existing agent processes...")
        
        commands = self._get_agent_commands()
        killed_count = 0
        
        for agent_name, port, _ in commands:
            if self._is_port_in_use(port):
                logger.info(f"Killing process on port {port} ({agent_name})")
                self._kill_process_on_port(port)
                killed_count += 1
                time.sleep(0.5)  # Give time for cleanup
        
        if killed_count > 0:
            logger.info(f"🎯 Cleaned up {killed_count} existing processes")
        else:
            logger.info("✅ No existing processes found to clean up")
        
        return True
    
    def get_agent_status(self, check_state_file: bool = True) -> Dict[str, Dict[str, any]]:
        """Get status of all configured agents, checking against state file."""
        status = {}
        commands = self._get_agent_commands()
        stale_agents = []
        
        # Load current state from file if requested
        if check_state_file:
            self._cleanup_stale_pids()
        
        for agent_name, port, command in commands:
            is_managed = agent_name in self.processes
            is_running = False
            pid = None
            from_state_file = False
            
            if is_managed:
                agent_process = self.processes[agent_name]
                try:
                    # Double-check that the PID is still valid
                    if psutil.pid_exists(agent_process.process.pid):
                        process = psutil.Process(agent_process.process.pid)
                        is_running = process.is_running()
                        pid = agent_process.process.pid if is_running else None
                        
                        if not is_running:
                            logger.info(f"🧹 Agent {agent_name} PID {agent_process.process.pid} is no longer running")
                            stale_agents.append(agent_name)
                    else:
                        logger.info(f"🧹 Agent {agent_name} PID {agent_process.process.pid} no longer exists")
                        stale_agents.append(agent_name)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    logger.info(f"🧹 Agent {agent_name} PID {agent_process.process.pid} is inaccessible")
                    stale_agents.append(agent_name)
            else:
                # Check state file for this agent
                if check_state_file and self.state_file.exists():
                    try:
                        with open(self.state_file, 'r') as f:
                            state_data = json.load(f)
                        
                        agents_data = state_data.get("agents", {})
                        if agent_name in agents_data:
                            agent_data = agents_data[agent_name]
                            state_pid = agent_data.get("pid")
                            
                            if state_pid:
                                try:
                                    if psutil.pid_exists(state_pid):
                                        process = psutil.Process(state_pid)
                                        if process.is_running():
                                            is_running = True
                                            pid = state_pid
                                            from_state_file = True
                                        else:
                                            logger.info(f"🧹 State file agent {agent_name} PID {state_pid} not running, removing from state")
                                            stale_agents.append(agent_name)
                                    else:
                                        logger.info(f"🧹 State file agent {agent_name} PID {state_pid} no longer exists, removing from state")
                                        stale_agents.append(agent_name)
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    logger.info(f"🧹 State file agent {agent_name} PID {state_pid} inaccessible, removing from state")
                                    stale_agents.append(agent_name)
                    except (json.JSONDecodeError, OSError):
                        pass  # Ignore state file errors for status checking
                
                # Fallback: check if something else is running on the port
                if not is_running:
                    is_running = self._is_port_in_use(port)
            
            status[agent_name] = {
                "port": port,
                "running": is_running,
                "managed": is_managed,
                "pid": pid,
                "command": command,
                "log_file": str(self.log_directory / f"{agent_name}.log"),
                "from_state_file": from_state_file
            }
        
        # Clean up stale agents
        if stale_agents:
            self._remove_stale_agents_from_state(stale_agents)
        
        return status
    
    def _remove_stale_agents_from_state(self, stale_agent_names: List[str]) -> None:
        """Remove stale agents from both memory and state file."""
        # Remove from memory
        for agent_name in stale_agent_names:
            if agent_name in self.processes:
                agent_process = self.processes[agent_name]
                # Log the stale removal
                self._write_log_entry(
                    agent_process.log_file,
                    "AGENT STALE CLEANUP",
                    agent_name,
                    {
                        "PID": agent_process.process.pid,
                        "Status": "Removed from tracking - process no longer exists"
                    }
                )
                del self.processes[agent_name]
        
        # Remove from state file
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state_data = json.load(f)
                
                agents_data = state_data.get("agents", {})
                removed_count = 0
                
                for agent_name in stale_agent_names:
                    if agent_name in agents_data:
                        # Log the removal to the agent's log file
                        agent_data = agents_data[agent_name]
                        self._write_log_entry(
                            self._setup_agent_logging(agent_name),
                            "AGENT STALE CLEANUP",
                            agent_name,
                            {
                                "PID": agent_data.get("pid", "unknown"),
                                "Status": "Removed from state file - process no longer exists"
                            }
                        )
                        del agents_data[agent_name]
                        removed_count += 1
                
                if removed_count > 0:
                    # Update state file
                    state_data["agents"] = agents_data
                    state_data["last_updated"] = time.strftime('%Y-%m-%d %H:%M:%S')
                    
                    temp_file = self.state_file.with_suffix('.tmp')
                    with open(temp_file, 'w') as f:
                        json.dump(state_data, f, indent=2)
                    temp_file.replace(self.state_file)
                    
                    logger.info(f"🎯 Removed {removed_count} stale agents from state file")
                    
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to clean up state file: {e}")
    
    def wait_for_agents_ready(self, timeout: int = 30) -> bool:
        """
        Wait for all agents to be ready (responding on their ports).
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if all agents are ready, False if timeout
        """
        import requests
        
        commands = self._get_agent_commands()
        start_time = time.time()
        
        logger.info(f"⏳ Waiting for {len(commands)} agents to be ready...")
        
        while time.time() - start_time < timeout:
            ready_count = 0
            
            for agent_name, port, _ in commands:
                try:
                    response = requests.get(f"http://localhost:{port}/.well-known/agent.json", timeout=1)
                    if response.status_code == 200:
                        ready_count += 1
                except requests.RequestException:
                    pass
            
            if ready_count == len(commands):
                logger.info(f"✅ All {len(commands)} agents are ready!")
                return True
            
            logger.debug(f"⏳ {ready_count}/{len(commands)} agents ready, waiting...")
            time.sleep(1)
        
        logger.warning(f"⚠️  Timeout waiting for agents to be ready ({ready_count}/{len(commands)} ready)")
        return False
    
    def wait_for_specific_agents_ready(self, agent_names: List[str], timeout: int = 30) -> bool:
        """
        Wait for specific agents to be ready (responding on their ports).
        
        Args:
            agent_names: List of agent names to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if all specified agents are ready, False if timeout
        """
        import requests
        
        # Get port mapping for the specified agents
        commands_dict = {name: (port, command) for name, port, command in self._get_agent_commands()}
        agents_to_check = []
        
        for agent_name in agent_names:
            if agent_name in commands_dict:
                port, _ = commands_dict[agent_name]
                agents_to_check.append((agent_name, port))
            else:
                logger.warning(f"⚠️  Agent {agent_name} not found in configuration")
        
        if not agents_to_check:
            return True
        
        start_time = time.time()
        logger.debug(f"⏳ Waiting for {len(agents_to_check)} specific agents to be ready...")
        
        while time.time() - start_time < timeout:
            ready_count = 0
            
            for agent_name, port in agents_to_check:
                try:
                    response = requests.get(f"http://localhost:{port}/.well-known/agent.json", timeout=1)
                    if response.status_code == 200:
                        ready_count += 1
                except requests.RequestException:
                    pass
            
            if ready_count == len(agents_to_check):
                logger.debug(f"✅ All {len(agents_to_check)} specified agents are ready!")
                return True
            
            logger.debug(f"⏳ {ready_count}/{len(agents_to_check)} specified agents ready, waiting...")
            time.sleep(1)
        
        logger.warning(f"⚠️  Timeout waiting for specified agents to be ready ({ready_count}/{len(agents_to_check)} ready)")
        return False
    
    def wait_for_agent_ready(self, agent_name: str, timeout: int = 30) -> bool:
        """
        Wait for a single agent to be ready.
        
        Args:
            agent_name: Name of the agent to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if agent is ready, False if timeout
        """
        return self.wait_for_specific_agents_ready([agent_name], timeout)


def setup_signal_handlers(manager: AgentProcessManager):
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down agents...")
        manager.stop_all_agents()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main entry point for the agents command."""
    parser = argparse.ArgumentParser(
        description="Manage A2A agent processes",
        prog="agents"
    )
    
    parser.add_argument(
        "action",
        choices=["start", "stop", "restart", "status"],
        help="Action to perform on agents"
    )
    
    parser.add_argument(
        "agent_name",
        nargs="?",
        default=None,
        help="Name of specific agent to manage (if not specified, applies to all agents)"
    )
    
    args = parser.parse_args()
    
    # Initialize the manager
    manager = AgentProcessManager()
    
    # Set up signal handlers for graceful shutdown
    setup_signal_handlers(manager)
    
    try:
        if args.action == "start":
            if args.agent_name:
                success = start_single_agent(manager, args.agent_name)
            else:
                success = manager.start_all_agents()
            sys.exit(0 if success else 1)
            
        elif args.action == "stop":
            if args.agent_name:
                success = stop_single_agent(manager, args.agent_name)
            else:
                success = manager.stop_all_agents()
            sys.exit(0 if success else 1)
            
        elif args.action == "restart":
            if args.agent_name:
                success = restart_single_agent(manager, args.agent_name)
            else:
                success = restart_all_agents(manager)
            sys.exit(0 if success else 1)
            
        elif args.action == "status":
            if args.agent_name:
                show_single_agent_status(manager, args.agent_name)
            else:
                show_all_agents_status(manager)
            
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


def start_single_agent(manager: AgentProcessManager, agent_name: str) -> bool:
    """Start a single agent by name."""
    commands = manager._get_agent_commands()
    
    # Find the agent
    agent_command = None
    for name, port, command in commands:
        if name == agent_name:
            agent_command = (name, port, command)
            break
    
    if not agent_command:
        available_agents = [name for name, _, _ in commands]
        logger.error(f"Agent '{agent_name}' not found. Available agents: {', '.join(available_agents)}")
        return False
    
    name, port, command = agent_command
    
    # Check if already running
    if manager._is_port_in_use(port):
        logger.warning(f"⚠️  Agent '{agent_name}' is already running on port {port}")
        return True
    
    logger.info(f"🚀 Starting agent '{agent_name}'...")
    success = manager._start_agent(name, port, command)
    
    if success:
        logger.info(f"✅ Agent '{agent_name}' started successfully on port {port}")
    else:
        logger.error(f"❌ Failed to start agent '{agent_name}'")
    
    return success


def stop_single_agent(manager: AgentProcessManager, agent_name: str) -> bool:
    """Stop a single agent by name."""
    # Check if we're managing this agent in memory
    if agent_name in manager.processes:
        agent_process = manager.processes[agent_name]
        success = manager._stop_agent(agent_process)
        if success:
            logger.info(f"✅ Agent '{agent_name}' stopped successfully")
            del manager.processes[agent_name]
            manager._save_agent_state()  # Update state file
        else:
            logger.error(f"❌ Failed to stop agent '{agent_name}'")
        return success
    
    # Check if agent exists in state file
    status = manager.get_agent_status()
    if agent_name in status:
        agent_info = status[agent_name]
        
        if agent_info["running"]:
            if agent_info["pid"]:
                # Stop using PID from state file
                logger.info(f"🛑 Stopping agent '{agent_name}' using PID {agent_info['pid']} from state file")
                success = manager._stop_agent_by_pid(
                    agent_name,
                    agent_info["pid"],
                    agent_info["port"],
                    agent_info["command"]
                )
                if success:
                    logger.info(f"✅ Agent '{agent_name}' stopped successfully")
                    manager._save_agent_state()  # Update state file
                else:
                    logger.error(f"❌ Failed to stop agent '{agent_name}'")
                return success
            else:
                # Fallback: kill process on port
                logger.info(f"🛑 Stopping unmanaged agent '{agent_name}' on port {agent_info['port']}")
                manager._kill_process_on_port(agent_info["port"])
                logger.info(f"✅ Agent '{agent_name}' stopped")
                return True
        else:
            logger.info(f"ℹ️  Agent '{agent_name}' is not running")
            return True
    else:
        # Agent not found in configuration
        commands = manager._get_agent_commands()
        available_agents = [name for name, _, _ in commands]
        logger.error(f"Agent '{agent_name}' not found. Available agents: {', '.join(available_agents)}")
        return False


def restart_single_agent(manager: AgentProcessManager, agent_name: str) -> bool:
    """Restart a single agent by name."""
    logger.info(f"🔄 Restarting agent '{agent_name}'...")
    
    # Stop first
    stop_success = stop_single_agent(manager, agent_name)
    if not stop_success:
        return False
    
    # Wait a moment for cleanup
    time.sleep(1)
    
    # Start again
    start_success = start_single_agent(manager, agent_name)
    
    if start_success:
        logger.info(f"✅ Agent '{agent_name}' restarted successfully")
    else:
        logger.error(f"❌ Failed to restart agent '{agent_name}'")
    
    return start_success


def restart_all_agents(manager: AgentProcessManager) -> bool:
    """Restart all agents."""
    logger.info("🔄 Restarting all agents...")
    
    # Stop all first
    stop_success = manager.stop_all_agents()
    if not stop_success:
        logger.warning("⚠️  Some agents failed to stop, continuing with start...")
    
    # Wait a moment for cleanup
    time.sleep(2)
    
    # Start all again
    start_success = manager.start_all_agents()
    
    if start_success:
        logger.info("✅ All agents restarted successfully")
    else:
        logger.error("❌ Some agents failed to restart")
    
    return start_success


def show_single_agent_status(manager: AgentProcessManager, agent_name: str):
    """Show status of a single agent."""
    status = manager.get_agent_status()
    
    if agent_name not in status:
        available_agents = list(status.keys())
        logger.error(f"Agent '{agent_name}' not found. Available agents: {', '.join(available_agents)}")
        return
    
    agent_status = status[agent_name]
    
    print(f"📊 Status for agent '{agent_name}':")
    print(f"   Port: {agent_status['port']}")
    print(f"   Running: {'✅ Yes' if agent_status['running'] else '❌ No'}")
    print(f"   Managed: {'✅ Yes' if agent_status['managed'] else '❌ No'}")
    
    if agent_status['pid']:
        print(f"   PID: {agent_status['pid']}")
    
    print(f"   Command: {' '.join(agent_status['command'])}")
    print(f"   Log file: {agent_status['log_file']}")


def show_all_agents_status(manager: AgentProcessManager):
    """Show status of all agents."""
    status = manager.get_agent_status()
    
    if not status:
        print("📊 No agents configured")
        return
    
    print("📊 Agent Status Summary:")
    print("=" * 80)
    
    for agent_name, agent_status in status.items():
        running_status = "✅ Running" if agent_status['running'] else "❌ Stopped"
        
        # Determine the source of tracking
        if agent_status['managed']:
            source_status = "(in-memory)"
        elif agent_status.get('from_state_file', False):
            source_status = "(state-file)"
        else:
            source_status = "(port-check)"
        
        pid_info = f"PID:{agent_status['pid']}" if agent_status['pid'] else "No PID"
        
        print(f"  {agent_name:20} | Port {agent_status['port']:5} | {running_status:11} | {pid_info:10} | {source_status}")
    
    # Summary
    total_agents = len(status)
    running_agents = sum(1 for s in status.values() if s['running'])
    managed_agents = sum(1 for s in status.values() if s['managed'])
    state_file_agents = sum(1 for s in status.values() if s.get('from_state_file', False))
    
    print("=" * 80)
    print(f"📈 Summary: {running_agents}/{total_agents} running")
    print(f"   • {managed_agents} tracked in memory")
    print(f"   • {state_file_agents} tracked in state file") 
    print(f"   • State file location: {manager.state_file}")


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Interactive Interface for A2A Client

Provides interactive chat mode with agents - one agent at a time.
"""

import asyncio
import logging
from typing import Dict

from a2a.types import Task
from a2a_client import CleanA2AClient, discover_available_agents, setup_signal_handlers, _cleanup_in_progress
from a2a_push_notification_manager import get_client_webhook_manager
from base_agent import configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


async def _load_agents(agent_filter: str = "all") -> Dict[str, CleanA2AClient]:
    """Load and initialize A2A clients based on the filter."""
    agents_config = discover_available_agents()
    
    if agent_filter != "all" and agent_filter in agents_config:
        agents_config = {agent_filter: agents_config[agent_filter]}
    
    clients = {}
    for name, url in agents_config.items():
        try:
            client = CleanA2AClient(url, name)
            if await client.discover_agent_skills():
                clients[name] = client
            else:
                print(f"⚠️  {name}: Connected but skills discovery failed")
                clients[name] = client
        except Exception as e:
            print(f"❌ Failed to connect to {name}: {e}")
    
    return clients


async def single_send_mode(message: str, agent_filter: str, use_tasks: bool):
    """Send a single message to the specified agent and exit."""
    global _cleanup_in_progress
    
    print(f"🚀 Sending single message to {agent_filter}...")
    
    clients = await _load_agents(agent_filter)
    if not clients:
        print("❌ No agents available")
        return
    
    if len(clients) != 1:
        print(f"❌ Expected exactly one agent, found {len(clients)}")
        return
    
    agent_name = list(clients.keys())[0]
    client = clients[agent_name]
    
    webhook_manager = None
    if use_tasks:
        webhook_manager = get_client_webhook_manager()
        await webhook_manager.start()
    
    try:
        print(f"🤖 Connecting to {agent_name}...")
        if client.agent_card and client.agent_card.skills:
            print(f"📋 {client.agent_card.name} - {client.agent_card.description}")
        
        print(f"📤 Sending: {message}")
        
        # Send message and wait for response
        if use_tasks:
            # For tasks, we need to wait for the callback
            response_received = asyncio.Event()
            response_result = []
            
            def callback(task: Task):
                result = client.extract_task_result(task)
                response_result.append(result)
                response_received.set()
            
            task_id = await client.send_task(message, callback)
            if task_id.startswith("immediate-"):
                # Immediate response, result should be in response_result
                if response_result:
                    print(f"🤖 {agent_name}: {response_result[0]}")
                else:
                    print(f"🤖 {agent_name}: (immediate response - no content)")
            else:
                print(f"⏳ Task {task_id} created, waiting for response...")
                # Wait for the callback to be triggered
                try:
                    await asyncio.wait_for(response_received.wait(), timeout=120.0)
                    print(f"🤖 {agent_name}: {response_result[0]}")
                except asyncio.TimeoutError:
                    print(f"⏰ Timeout waiting for response from {agent_name}")
        else:
            # For messages, response is immediate
            result = await client.send_message(message)
            print(f"🤖 {agent_name}: {result}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Cleanup
        _cleanup_in_progress = True
        for client_to_close in clients.values():
            await client_to_close.close()
        if webhook_manager:
            await webhook_manager.stop()


async def interactive_mode(use_tasks: bool = False, agent_filter: str = "all", send_message: str = None):
    """Interactive chat mode with agents."""
    global _cleanup_in_progress
    setup_signal_handlers()
    
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("a2a.client.client").setLevel(logging.WARNING)
    
    # Handle single message send-and-exit mode
    if send_message:
        await single_send_mode(send_message, agent_filter, use_tasks)
        return
    
    print(f"🎭 A2A Interactive Mode ({'Tasks' if use_tasks else 'Messages'})")
    print("=" * 50)
    
    clients = await _load_agents(agent_filter)
    if not clients:
        print("❌ No agents available")
        return
    
    webhook_manager = None
    if use_tasks:
        webhook_manager = get_client_webhook_manager()
        await webhook_manager.start()
        print("🔔 Push notifications enabled")
    
    try:
        if agent_filter != "all" and len(clients) == 1:
            agent_name = list(clients.keys())[0]
            if await chat_with_agent(clients[agent_name], use_tasks):
                clients = await _load_agents("all") # Reload all agents
        
        while not _cleanup_in_progress and clients:
            if not await _agent_selection_loop(clients, use_tasks):
                break
                
    finally:
        print("🧹 Cleaning up resources...")
        _cleanup_in_progress = True
        for client in clients.values():
            await client.close()
        if webhook_manager:
            await webhook_manager.stop()
        print("✅ Cleanup complete")


async def _agent_selection_loop(clients: Dict[str, CleanA2AClient], use_tasks: bool) -> bool:
    """The main loop for agent selection and interaction."""
    print("\n📋 Available agents:")
    agent_list = list(clients.keys())
    for i, name in enumerate(agent_list, 1):
        client = clients[name]
        skills_preview = "No skills defined"
        if client.agent_card and client.agent_card.skills:
            skills_count = len(client.agent_card.skills)
            skill_names = [skill.name for skill in client.agent_card.skills[:2]]
            skills_preview = ", ".join(skill_names)
            if skills_count > 2:
                skills_preview += f" (+{skills_count - 2} more)"
            print(f"  {i}. {name} - {skills_preview}")
            
    print("\n💬 Commands: /skills, /quit, /exit")
            
    try:
        selection = input("\n🎯 Select agent (number or name): ").strip()
                
        if selection.lower() in ['/quit', '/exit']:
            print("👋 Goodbye!")
            return False
        elif selection.lower() == '/skills':
            print("\n🎯 Detailed Agent Skills:\n" + "=" * 50)
            for client in clients.values():
                print(f"\n{client.get_skills_summary()}")
            return True
        elif not selection:
            return True
        
        selected_agent_name = None
        if selection.isdigit() and 0 <= int(selection) - 1 < len(agent_list):
            selected_agent_name = agent_list[int(selection) - 1]
        elif selection in clients:
            selected_agent_name = selection
        
        if selected_agent_name:
            return await chat_with_agent(clients[selected_agent_name], use_tasks)
        else:
            print("❌ Invalid selection.")
            return True
            
    except (KeyboardInterrupt, EOFError):
        print("\n👋 Interrupted, exiting...")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return True


async def chat_with_agent(client: CleanA2AClient, use_tasks: bool) -> bool:
    """Chat with a single agent. Returns True to continue to agent selection, False to exit."""
    global _cleanup_in_progress
    
    print(f"\n🤖 Connected to {client.agent_name}")
    if client.agent_card and client.agent_card.skills:
        print(f"📋 {client.agent_card.name} - {client.agent_card.description}")
        print(f"🎯 Skills: {', '.join([s.name for s in client.agent_card.skills])}")
    
    print("💬 Commands: /skills, /bye (back to menu), /quit, /exit")
    print("-" * 40)
    
    while not _cleanup_in_progress:
        try:
            user_input = input(f"👤 You → {client.agent_name}: ").strip()
            
            if user_input.lower() == '/bye':
                return True
            elif user_input.lower() in ['/quit', '/exit']:
                return False
            elif user_input.lower() == '/skills':
                print(f"\n{client.get_skills_summary()}")
            elif user_input:
                await send_to_agent(client, user_input, use_tasks)
            
        except (KeyboardInterrupt, EOFError):
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return False


async def send_to_agent(client: CleanA2AClient, message: str, use_tasks: bool):
    """Send message to a single agent."""
    try:
        if use_tasks:
            def callback(task: Task):
                result = client.extract_task_result(task)
                print(f"🤖 {client.agent_name}: {result}")
            
            task_id = await client.send_task(message, callback)
            if task_id != "immediate-response":
                print(f"🚀 Task {task_id} created, waiting for response...")
        else:
            result = await client.send_message(message)
            print(f"🤖 {client.agent_name}: {result}")
    except Exception as e:
        print(f"❌ Error: {e}") 
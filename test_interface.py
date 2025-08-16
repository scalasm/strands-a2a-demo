#!/usr/bin/env python3
"""
Test Interface for A2A Client

Comprehensive test suite for all A2A agents and configurations.
"""

import asyncio
import logging
from typing import Dict, List
from dataclasses import dataclass

from a2a.types import Task
from a2a_client import CleanA2AClient, discover_available_agents, get_agent_info_summary, setup_signal_handlers, _cleanup_in_progress
from a2a_push_notification_manager import get_client_webhook_manager, managed_webhook_server
from app_config import get_application_config, configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    agent_name: str
    test_name: str
    success: bool
    response_time: float
    response: str
    error: str = ""


async def _load_all_agents() -> Dict[str, CleanA2AClient]:
    """Load and initialize all available A2A clients."""
    agents_config = discover_available_agents()
    
    clients = {}
    for name, url in agents_config.items():
        try:
            client = CleanA2AClient(url, name)
            if await client.discover_agent_skills():
                clients[name] = client
                logger.info(f"✅ Connected to {name}")
            else:
                logger.warning(f"⚠️  {name}: Connected but skills discovery failed")
                clients[name] = client
        except Exception as e:
            logger.error(f"❌ Failed to connect to {name}: {e}")
    
    return clients


async def generate_test_questions_for_agent(client: CleanA2AClient) -> List[str]:
    """Generate test questions using agent card examples as the primary source."""
    questions = []
    
    # If agent has no card or no skills, return basic greeting
    if not client.agent_card or not client.agent_card.skills:
        return ["Hello! Can you tell me what you can help me with?"]
    
    # Priority 1: Use examples directly from the agent card skills
    for skill in client.agent_card.skills:
        if skill.examples:
            # Add all examples from this skill (these are now high-quality, specific examples)
            questions.extend(skill.examples)
    
    # Priority 2: Only add fallback questions if no examples were found
    if not questions:
        # Generic greeting that works for all agents
        questions.append("Hello! Can you tell me what you can help me with?")
        
        # Generate skill-specific questions as fallback
        for skill in client.agent_card.skills:
            skill_name = skill.name.lower()
            skill_description = skill.description.lower()
            
            # Math/Calculator skills
            if any(word in skill_name or word in skill_description for word in ['math', 'calculator', 'calculation', 'arithmetic']):
                questions.extend([
                    "What is 15 + 27?",
                    "Calculate the square root of 144",
                    "What is 8 * 7?"
                ])
            
            # Time skills
            elif any(word in skill_name or word in skill_description for word in ['time', 'timezone', 'date', 'clock']):
                questions.extend([
                    "What time is it?",
                    "What is the current date?",
                    "Tell me the time in UTC"
                ])
            
            # File operations
            elif any(word in skill_name or word in skill_description for word in ['file', 'editor', 'read', 'write']):
                questions.extend([
                    "Can you help me with file operations?",
                    "What file editing capabilities do you have?",
                    "How can you help with reading files?"
                ])
            
            # Web/Fetch capabilities
            elif any(word in skill_name or word in skill_description for word in ['web', 'fetch', 'browser', 'http', 'url']):
                questions.extend([
                    "Can you help me browse the web?",
                    "What web content can you fetch?",
                    "How do you handle web requests?"
                ])
            
            # Coordination skills
            elif any(word in skill_name or word in skill_description for word in ['coordinat', 'delegat', 'manage']):
                questions.extend([
                    "What agents do you coordinate?",
                    "How do you delegate tasks?",
                    "Can you help me with a calculation and time query?"
                ])
            
            # MCP servers
            elif 'mcp' in skill_name or 'mcp' in skill_description:
                questions.extend([
                    f"Can you use your {skill.name} capability?",
                    "What external services can you access?"
                ])
    
    # Remove duplicates while preserving order
    unique_questions = list(dict.fromkeys(questions))
    
    # Always include a basic greeting as the first question if not already present
    greeting = "Hello! Can you tell me what you can help me with?"
    if greeting not in unique_questions:
        unique_questions.insert(0, greeting)
    
    # Return up to 6 questions (greeting + 5 examples)
    return unique_questions[:6]


async def test_mode(agent_filter: str = "all"):
    """Run comprehensive tests of all A2A configurations."""
    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()
    
    print("🧪 A2A Dynamic Test Suite")
    print("=" * 40)
    
    if agent_filter != "all":
        print(f"🎯 Testing only agent: {agent_filter}")
    
    # Print agent info summary
    print("\n📋 Available Agents:")
    print(get_agent_info_summary())
    
    agents_config = discover_available_agents()
    
    # Filter agents if specific agent requested
    if agent_filter != "all":
        if agent_filter in agents_config:
            agents_config = {agent_filter: agents_config[agent_filter]}
            print(f"🔍 Filtered to agent: {agent_filter}")
        else:
            available_agents = list(agents_config.keys())
            print(f"❌ Agent '{agent_filter}' not found. Available agents: {', '.join(available_agents)}")
            return False
    
    # Test results
    results = {
        "agent_discovery": {},
        "skill_based_tests": {},
        "task_tests": {},
        "push_notification_tests": {}
    }
    
    # Test 1: Agent Discovery and Skills Analysis
    print("\n📡 Testing agent discovery and skills analysis...")
    agent_questions = {}
    
    for name, url in agents_config.items():
        try:
            client = CleanA2AClient(url, name)
            
            # Test skill discovery
            if await client.discover_agent_skills():
                if client.agent_card and client.agent_card.skills:
                    skills_count = len(client.agent_card.skills)
                    results["agent_discovery"][name] = f"✅ Success ({skills_count} skills)"
                    print(f"✅ {name}: Agent card retrieved with {skills_count} skills")
                    
                    # Generate dynamic test questions
                    questions = await generate_test_questions_for_agent(client)
                    agent_questions[name] = questions
                    print(f"   📝 Generated {len(questions)} skill-based test questions")
                else:
                    results["agent_discovery"][name] = "✅ Success (no skills)"
                    print(f"✅ {name}: Agent card retrieved (no skills defined)")
                    agent_questions[name] = ["Hello! What can you help me with?"]
            else:
                results["agent_discovery"][name] = "⚠️  Connected but skills discovery failed"
                print(f"⚠️  {name}: Connected but skills discovery failed")
                agent_questions[name] = ["Hello! What can you help me with?"]
                
            await client.close()
        except Exception as e:
            results["agent_discovery"][name] = f"❌ {e}"
            print(f"❌ {name}: {e}")
            agent_questions[name] = ["Hello! What can you help me with?"]
    
    # Test 2: Skill-Based Dynamic Communication
    print("\n💬 Testing skill-based dynamic communication...")
    
    for name, url in agents_config.items():
        if name not in agent_questions:
            continue
            
        try:
            client = CleanA2AClient(url, name)
            questions = agent_questions[name]
            
            # Test with the first skill-appropriate question
            test_question = questions[0] if questions else "Hello! What can you help me with?"
            
            result = await client.send_message(test_question)
            results["skill_based_tests"][name] = f"✅ Response to '{test_question[:30]}...': {result[:50]}..."
            print(f"✅ {name}: Responded appropriately to skill-based question")
            await client.close()
        except Exception as e:
            results["skill_based_tests"][name] = f"❌ {e}"
            print(f"❌ {name}: {e}")
    
    # Test 3: Task Communication with Push Notifications
    print("\n🚀 Testing task communication with skill-appropriate queries...")
    
    # Load webhook port from config
    try:
        webhook_port = get_application_config().get("webhook", {}).get("port", 8000)
    except Exception:
        webhook_port = 8000  # fallback default
    
    async with managed_webhook_server(webhook_port) as webhook_manager:
        for name, url in agents_config.items():
            if name not in agent_questions:
                continue
        
            try:
                client = CleanA2AClient(url, name)
                questions = agent_questions[name]
                
                # Use a skill-appropriate question for task testing
                test_task = questions[1] if len(questions) > 1 else questions[0]
                
                # Create callback to capture result
                task_results = {}
                
                def callback(task: Task):
                    task_results[task.id] = client.extract_task_result(task)
                
                task_id = await client.send_task(test_task, callback)
                
                # Wait a bit for callback
                await asyncio.sleep(2)
                
                if task_id in task_results:
                    result = task_results[task_id]
                    results["task_tests"][name] = f"✅ Task completed: {result[:50]}..."
                    results["push_notification_tests"][name] = "✅ Push notification received"
                    print(f"✅ {name}: Task completed with push notification")
                elif task_id == "immediate-response":
                    results["task_tests"][name] = "✅ Immediate response"
                    results["push_notification_tests"][name] = "✅ Immediate callback"
                    print(f"✅ {name}: Immediate response (no task needed)")
                else:
                    results["task_tests"][name] = f"⏳ Task {task_id} created (pending)"
                    results["push_notification_tests"][name] = "⏳ Waiting for callback"
                    print(f"⏳ {name}: Task created, waiting for completion")
                
                await client.close()
                
            except Exception as e:
                results["task_tests"][name] = f"❌ {e}"
                results["push_notification_tests"][name] = f"❌ {e}"
                print(f"❌ {name}: {e}")
    
    # Test 4: Multi-Agent Workflow (if coordinators are available and not filtering to single agent)
    coordinators = [name for name in agents_config.keys() if 'coordinator' in name.lower()]
    if coordinators and (agent_filter == "all" or agent_filter in coordinators):
        print(f"\n🔄 Testing multi-agent workflows with {len(coordinators)} coordinators...")
        
        for coordinator_name in coordinators:
            try:
                client = CleanA2AClient(agents_config[coordinator_name], coordinator_name)
                
                # Test a complex query that requires coordination
                complex_query = "I need to know what time it is and also calculate 25 * 4"
                result = await client.send_message(complex_query)
                
                print(f"✅ {coordinator_name}: Successfully handled multi-agent query")
                print(f"   📄 Response: {result[:100]}...")
                
                await client.close()
            except Exception as e:
                print(f"❌ {coordinator_name}: Failed multi-agent test: {e}")
    
    # Print comprehensive summary
    print("\n📊 Dynamic Test Results Summary")
    print("=" * 50)
    
    for test_type, test_results in results.items():
        print(f"\n{test_type.replace('_', ' ').title()}:")
        for agent, result in test_results.items():
            print(f"  {agent}: {result}")
    
    # Calculate overall stats
    total_tests = sum(len(test_results) for test_results in results.values())
    successful_tests = sum(
        1 for test_results in results.values() 
        for result in test_results.values() 
        if result.startswith("✅")
    )
    
    print(f"\n🎯 Overall: {successful_tests}/{total_tests} tests passed")
    print(f"📈 Success rate: {successful_tests/total_tests*100:.1f}%" if total_tests > 0 else "No tests run") 
    
    # Print discovered capabilities summary
    print(f"\n🔍 Agent Capabilities Summary:")
    print(f"   📊 Total agents discovered: {len(agents_config)}")
    print(f"   🤖 Individual agents: {len([n for n in agents_config.keys() if 'coordinator' not in n.lower()])}")
    print(f"   🌀 Coordinator agents: {len(coordinators)}")
    print(f"   📝 Skill-based questions generated: {sum(len(q) for q in agent_questions.values())}")
    
    return successful_tests == total_tests 
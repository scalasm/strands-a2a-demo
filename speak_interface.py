#!/usr/bin/env python3
"""
Speak Interface for A2A Client

Voice interaction mode using Nova Sonic with A2A agent integration.
"""

import asyncio
import base64
import json
import logging
import time
import uuid
import hashlib
import numpy as np
import boto3
from typing import Optional

# Nova Sonic imports for speak mode
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient, InvokeModelWithBidirectionalStreamOperationInput
    from aws_sdk_bedrock_runtime.models import InvokeModelWithBidirectionalStreamInputChunk, BidirectionalInputPayloadPart
    from aws_sdk_bedrock_runtime.config import Config, HTTPAuthSchemeResolver, SigV4AuthScheme
    from smithy_aws_core.credentials_resolvers.environment import EnvironmentCredentialsResolver
    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False

# Import LiveKit's AudioProcessingModule for echo cancellation
try:
    from livekit import rtc
    APM_AVAILABLE = True
except ImportError:
    APM_AVAILABLE = False

from a2a.types import Task
from a2a_client import CleanA2AClient, load_config, setup_signal_handlers, _cleanup_in_progress, time_it_async, DEBUG, discover_available_agents
from a2a_push_notification_manager import get_client_webhook_manager
from aws_credentials_helper import ensure_aws_credentials
from base_agent import configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# Audio configuration
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
CHANNELS = 1
FORMAT = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None
CHUNK_SIZE = 1024
TASK_RESULT_SIZE = 10

def debug_print(message):
    """Print only if debug mode is enabled"""
    if DEBUG:
        print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {message}')

class BedrockStreamManager:
    """Manages bidirectional streaming with AWS Bedrock using asyncio for speak mode"""
    
    # Event templates (same as in nova_sonic_tool_use_mcp_task.py)
    START_SESSION_EVENT = '''{
        "event": {
            "sessionStart": {
            "inferenceConfiguration": {
                "maxTokens": 1024,
                "topP": 0.9,
                "temperature": 0.7
                }
            }
        }
    }'''

    CONTENT_START_EVENT = '''{
        "event": {
            "contentStart": {
            "promptName": "%s",
            "contentName": "%s",
            "type": "AUDIO",
            "interactive": true,
            "role": "USER",
            "audioInputConfiguration": {
                "mediaType": "audio/lpcm",
                "sampleRateHertz": 16000,
                "sampleSizeBits": 16,
                "channelCount": 1,
                "audioType": "SPEECH",
                "encoding": "base64"
                }
            }
        }
    }'''

    AUDIO_EVENT_TEMPLATE = '''{
        "event": {
            "audioInput": {
            "promptName": "%s",
            "contentName": "%s",
            "content": "%s"
            }
        }
    }'''

    TEXT_CONTENT_START_EVENT = '''{
        "event": {
            "contentStart": {
            "promptName": "%s",
            "contentName": "%s",
            "type": "TEXT",
            "role": "%s",
            "interactive": true,
            "textInputConfiguration": {
                "mediaType": "text/plain"
            }
            }
        }
    }'''

    TEXT_INPUT_EVENT = '''{
        "event": {
            "textInput": {
            "promptName": "%s",
            "contentName": "%s",
            "content": "%s"
            }
        }
    }'''

    TOOL_CONTENT_START_EVENT = '''{
        "event": {
            "contentStart": {
                "promptName": "%s",
                "contentName": "%s",
                "interactive": false,
                "type": "TOOL",
                "role": "TOOL",
                "toolResultInputConfiguration": {
                    "toolUseId": "%s",
                    "type": "TEXT",
                    "textInputConfiguration": {
                        "mediaType": "text/plain"
                    }
                }
            }
        }
    }'''

    CONTENT_END_EVENT = '''{
        "event": {
            "contentEnd": {
            "promptName": "%s",
            "contentName": "%s"
            }
        }
    }'''

    PROMPT_END_EVENT = '''{
        "event": {
            "promptEnd": {
            "promptName": "%s"
            }
        }
    }'''

    SESSION_END_EVENT = '''{
        "event": {
            "sessionEnd": {}
        }
    }'''
    
    def __init__(self, model_id='amazon.nova-sonic-v1:0', region='us-east-1', a2a_client=None):
        """Initialize the stream manager."""
        self.model_id = model_id
        self.region = region
        self.a2a_client = a2a_client
        
        # Asyncio queues for audio processing
        self.audio_input_queue = asyncio.Queue()
        self.audio_output_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        
        self.response_task = None
        self.stream_response = None
        self.is_active = False
        self.barge_in = False
        self.bedrock_client = None
        
        # Text response components
        self.display_assistant_text = False
        self.role = None

        # Session information
        self.prompt_name = str(uuid.uuid4())
        self.content_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())
        self.toolUseContent = ""
        self.toolUseId = ""
        self.toolName = ""
        
        # Task results storage
        self.task_results = []
        
        # Audio setup and cache for Polly
        self.polly_client = boto3.client('polly')
        self._audio_cache = {}
        
        # Flag to track when we're injecting Polly audio (to prevent barge-in)
        self.injecting_polly_audio = False

    def _initialize_client(self):
        """Initialize the Bedrock client."""
        # Ensure AWS credentials are available in environment variables
        if not ensure_aws_credentials():
            raise Exception("AWS credentials not available for Bedrock client")
        
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            http_auth_scheme_resolver=HTTPAuthSchemeResolver(),
            http_auth_schemes={"aws.auth#sigv4": SigV4AuthScheme()}
        )
        self.bedrock_client = BedrockRuntimeClient(config=config)
    
    def start_prompt(self):
        """Create a promptStart event with send_task and get_results tools."""
        send_task_schema = json.dumps({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to send to the A2A agent"
                }
            },
            "required": ["query"]
        })
        
        get_results_schema = json.dumps({
            "type": "object",
            "properties": {},
            "required": []
        })
        
        # Generate dynamic description based on discovered agent skills
        tool_description = "Send a task to the A2A agent to process"
        if self.a2a_client and hasattr(self.a2a_client, 'agent_card') and self.a2a_client.agent_card:
            agent_name = self.a2a_client.agent_card.name
            agent_description = self.a2a_client.agent_card.description
            
            tool_description = f"Send a task to the {agent_name} agent. {agent_description}."
            
            if self.a2a_client.agent_card.skills:
                skills_list = []
                for skill in self.a2a_client.agent_card.skills:
                    skills_list.append(f"{skill.name}: {skill.description}")
                
                tool_description += f" Available capabilities: {'; '.join(skills_list[:3])}"
                tool_description += " Do not use send_task to get results. Do not send the same task twice."
                
                # Add examples from the agent's skills
                examples = []
                for skill in self.a2a_client.agent_card.skills[:2]:  # Use first 2 skills for examplesç
                    if skill.examples:
                        examples.extend(skill.examples[:1])  # Take 1 example per skill
                
                if examples:
                    tool_description += f" Examples: {', '.join(examples[:3])}"
        
        tools = [
            {
                "toolSpec": {
                    "name": "send_task",
                    "description": tool_description,
                    "inputSchema": {"json": send_task_schema}
                }
            },
            {
                "toolSpec": {
                    "name": "get_results", 
                    "description": "ONLY use this tool to retrieve results after you hear 'the task is complete' or 'get the results'.",
                    "inputSchema": {"json": get_results_schema}
                }
            }
        ]
        
        logger.info(f"🔧 Configuring {len(tools)} tools for Nova Sonic")
        logger.info(f"🔧 Tool 1: {tools[0]['toolSpec']['name']} - {tools[0]['toolSpec']['description'][:100]}...")
        logger.info(f"🔧 Tool 2: {tools[1]['toolSpec']['name']} - {tools[1]['toolSpec']['description'][:100]}...")
        
        prompt_start_event = {
            "event": {
                "promptStart": {
                    "promptName": self.prompt_name,
                    "textOutputConfiguration": {"mediaType": "text/plain"},
                    "audioOutputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": 24000,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "voiceId": "tiffany",
                        "encoding": "base64",
                        "audioType": "SPEECH"
                    },
                    "toolUseOutputConfiguration": {"mediaType": "application/json"},
                    "toolConfiguration": {"tools": tools}
                }
            }
        }
        
        return json.dumps(prompt_start_event)
    
    def tool_result_event(self, content_name, content, role):
        """Create a tool result event (matching reference implementation)"""
        # Handle dict content like the reference implementation
        if isinstance(content, dict):
            content_json_string = json.dumps(content)
        else:
            content_json_string = content
            
        tool_result_event = {
            "event": {
                "toolResult": {
                    "promptName": self.prompt_name,
                    "contentName": content_name,
                    "content": content_json_string
                }
            }
        }
        return json.dumps(tool_result_event)
    
    async def initialize_stream(self):
        """Initialize the bidirectional stream with Bedrock."""
        if not self.bedrock_client:
            self._initialize_client()
        
        try:
            input_params = InvokeModelWithBidirectionalStreamOperationInput(
                model_id=self.model_id
            )
            
            self.stream_response = await self.bedrock_client.invoke_model_with_bidirectional_stream(input=input_params)
            self.is_active = True
            
            # Build informative system prompt based on connected agent
            if (self.a2a_client and 
                hasattr(self.a2a_client, 'agent_card') and 
                self.a2a_client.agent_card):
                
                agent_name = self.a2a_client.agent_card.name
                agent_description = self.a2a_client.agent_card.description
                
                default_system_prompt = f"You are a friendly voice assistant. The user and you will engage in a spoken dialog exchanging the transcripts of a natural real-time conversation. " \
                                       f"You have access to the {agent_name} agent which can help with: {agent_description}. " \
                                       f"You have two tools: 1) send_task - use this to send new requests to the {agent_name} agent, " \
                                       f"2) get_results - use this ONLY when you hear 'get the results'. " \
                                       f"Never use send_task when you need to get results - always use get_results for that. " \
                                       f"Be helpful and explain what you're doing when you send tasks."
                
                if self.a2a_client.agent_card.skills:
                    skills_list = []
                    for skill in self.a2a_client.agent_card.skills[:3]:  # Limit to first 3 skills
                        skills_list.append(skill.name)
                    
                    skills_text = ", ".join(skills_list)
                    default_system_prompt += f" The {agent_name} agent has these capabilities: {skills_text}."
            else:
                # Fallback system prompt
                agent_name = self.a2a_client.agent_name if self.a2a_client else "the connected agent"
                default_system_prompt = f"You are a friendly voice assistant. The user and you will engage in a spoken dialog exchanging the transcripts of a natural real-time conversation. " \
                                       f"You have access to {agent_name} through two tools: 1) send_task - use this to send new requests to {agent_name}, " \
                                       f"2) get_results - use this ONLY when you hear 'get the results'. " \
                                       f"Never use send_task when you need to get results - always use get_results for that. " \
                                       f"Be helpful and explain what you're doing when you send tasks."
            
            # Send initialization events with proper sequence like reference
            prompt_event = self.start_prompt()
            text_content_start = self.TEXT_CONTENT_START_EVENT % (self.prompt_name, self.content_name, "SYSTEM")
            text_content = self.TEXT_INPUT_EVENT % (self.prompt_name, self.content_name, default_system_prompt)
            text_content_end = self.CONTENT_END_EVENT % (self.prompt_name, self.content_name)
            
            init_events = [self.START_SESSION_EVENT, prompt_event, text_content_start, text_content, text_content_end]
            
            for event in init_events:
                await self.send_raw_event(event)
                # Small delay between init events like reference
                await asyncio.sleep(0.1)
            
            # Start response processing
            self.response_task = asyncio.create_task(self._process_responses())
            
            # Start processing audio input like reference
            asyncio.create_task(self._process_audio_input())
            
            # Wait a bit to ensure everything is set up
            await asyncio.sleep(0.1)
            
            return self
        except Exception as e:
            self.is_active = False
            print(f"Failed to initialize stream: {str(e)}")
            raise
    
    async def send_raw_event(self, event_json):
        """Send a raw JSON event to the stream."""
        if not self.is_active or not self.stream_response:
            return
        
        chunk = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode('utf-8'))
        )
        
        try:
            await self.stream_response.input_stream.send(chunk)
            # Only log non-audio events to reduce spam
            if DEBUG:
                try:
                    event_data = json.loads(event_json)
                    event_type = list(event_data.get("event", {}).keys())
                    # Skip logging frequent audio input events
                    if event_type != ['audioInput'] and len(event_json) <= 200:
                        debug_print(f"Sent event: {event_json}")
                    elif event_type != ['audioInput'] and len(event_json) > 200:
                        debug_print(f"Sent event type: {event_type}")
                except:
                    pass  # Skip logging if JSON parsing fails
        except Exception as e:
            debug_print(f"Error sending event: {str(e)}")
            if DEBUG:
                import traceback
                traceback.print_exc()
    
    async def send_audio_content_start_event(self):
        """Send audio content start event."""
        event = self.CONTENT_START_EVENT % (self.prompt_name, self.audio_content_name)
        await self.send_raw_event(event)
    
    async def send_audio_content_end_event(self):
        """Send audio content end event."""
        event = self.CONTENT_END_EVENT % (self.prompt_name, self.audio_content_name)
        await self.send_raw_event(event)
    
    async def send_prompt_end_event(self):
        """Send prompt end event."""
        event = self.PROMPT_END_EVENT % self.prompt_name
        await self.send_raw_event(event)
    
    async def send_session_end_event(self):
        """Send session end event."""
        await self.send_raw_event(self.SESSION_END_EVENT)
        self.is_active = False
    
    async def _process_audio_input(self):
        """Process audio input from the queue and send to Bedrock."""
        while self.is_active:
            try:
                # Get audio data from the queue
                data = await self.audio_input_queue.get()
                
                audio_bytes = data.get('audio_bytes')
                if not audio_bytes:
                    continue
                
                # Base64 encode the audio data
                blob = base64.b64encode(audio_bytes)
                audio_event = self.AUDIO_EVENT_TEMPLATE % (
                    self.prompt_name, 
                    self.audio_content_name, 
                    blob.decode('utf-8')
                )
                
                # Send the event
                await self.send_raw_event(audio_event)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                debug_print(f"Error processing audio: {e}")
                if DEBUG:
                    import traceback
                    traceback.print_exc()
    
    def add_audio_chunk(self, audio_bytes):
        """Add an audio chunk to the queue."""
        if self.is_active and not self.barge_in:
            try:
                self.audio_input_queue.put_nowait({
                    'audio_bytes': audio_bytes,
                    'prompt_name': self.prompt_name,
                    'content_name': self.audio_content_name
                })
            except asyncio.QueueFull:
                logger.warning("Audio input queue full, dropping chunk")
    
    async def _clear_audio_output_queue(self):
        """Clear the audio output queue to stop audio playback during barge-in."""
        cleared_count = 0
        while not self.audio_output_queue.empty():
            try:
                self.audio_output_queue.get_nowait()
                cleared_count += 1
            except asyncio.QueueEmpty:
                break
        if cleared_count > 0:
            logger.debug(f"Cleared {cleared_count} audio chunks from output queue due to barge-in")
    
    async def _process_responses(self):
        """Process responses from the Bedrock stream."""
        try:            
            while self.is_active:
                try:
                    output = await self.stream_response.await_output()
                    result = await output[1].receive()
                    if result.value and result.value.bytes_:
                        try:
                            response_data = result.value.bytes_.decode('utf-8')
                            event_data = json.loads(response_data)
                            
                            # Handle different response types
                            if 'event' in event_data:
                                if 'contentStart' in event_data['event']:
                                    content_start = event_data['event']['contentStart']
                                    # set role
                                    self.role = content_start.get('role')
                                    # Check for speculative content
                                    if 'additionalModelFields' in content_start:
                                        try:
                                            additional_fields = json.loads(content_start['additionalModelFields'])
                                            if additional_fields.get('generationStage') == 'SPECULATIVE':
                                                self.display_assistant_text = True
                                            else:
                                                self.display_assistant_text = False
                                        except json.JSONDecodeError:
                                            pass
                                elif 'textOutput' in event_data['event']:
                                    text_content = event_data['event']['textOutput'].get('content', '')
                                    role = event_data['event']['textOutput'].get('role', '')
                                    # Check if there is a barge-in, but only trigger it if we're not injecting Polly audio
                                    if '{ "interrupted" : true }' in text_content:
                                        if not self.injecting_polly_audio:
                                            logger.info("Barge-in detected. Stopping audio output.")
                                            self.barge_in = True
                                            # Clear audio output queue immediately to stop audio playback
                                            await self._clear_audio_output_queue()
                                        else:
                                            logger.info("Barge-in detected but ignoring (Polly audio injection in progress).")

                                    if (self.role == "ASSISTANT" and self.display_assistant_text):
                                        print(f"🤖 Assistant: {text_content}")
                                        # Send formatted text event for webspeak interface
                                        await self.output_queue.put({
                                            'type': 'text',
                                            'content': text_content,
                                            'role': 'assistant'
                                        })
                                    elif (self.role == "USER"):
                                        print(f"👤 User: {text_content}")
                                        # Send formatted text event for webspeak interface  
                                        await self.output_queue.put({
                                            'type': 'text',
                                            'content': text_content,
                                            'role': 'user'
                                        })

                                elif 'audioOutput' in event_data['event']:
                                    audio_content = event_data['event']['audioOutput'].get('content', '')
                                    if audio_content:
                                        audio_bytes = base64.b64decode(audio_content)
                                        await self.audio_output_queue.put(audio_bytes)
                                        
                                elif 'toolUse' in event_data['event']:
                                    self.toolUseContent = event_data['event']['toolUse']
                                    self.toolName = event_data['event']['toolUse']['toolName']
                                    self.toolUseId = event_data['event']['toolUse']['toolUseId']
                                    
                                elif 'contentEnd' in event_data['event'] and event_data['event'].get('contentEnd', {}).get('type') == 'TOOL':
                                    toolResult = await self.processToolUse(self.toolName, self.toolUseContent)
                                    toolContent = str(uuid.uuid4())
                                    await self.send_tool_start_event(toolContent)
                                    await self.send_tool_result_event(toolContent, toolResult)
                                    await self.send_tool_content_end_event(toolContent)
                                    
                                elif 'contentEnd' in event_data['event']:
                                    content_end = event_data['event']['contentEnd']
                                    
                                    # Check for interruption in contentEnd events
                                    stop_reason = content_end.get('stopReason', '').upper()
                                    if stop_reason == 'INTERRUPTED' and not self.injecting_polly_audio:
                                        logger.info("Content interrupted by user - triggering barge-in")
                                        self.barge_in = True
                                        # Clear audio output queue immediately
                                        await self._clear_audio_output_queue()
                                    
                                    # Reset barge-in flag when content ends (but not for interruptions)
                                    elif self.barge_in and stop_reason != 'INTERRUPTED':
                                        self.barge_in = False
                                
                                elif 'completionEnd' in event_data['event']:
                                    # Handle end of conversation, no more response will be generated
                                    pass
                                
                                elif 'error' in event_data['event']:
                                    logger.error(f"Bedrock error: {event_data['event']['error']}")
                            
                            # Put the response in the output queue for other components
                            await self.output_queue.put(event_data)
                        except json.JSONDecodeError as e:
                            debug_print(f"Error decoding JSON: {e}")
                            await self.output_queue.put({"raw_data": response_data})
                except StopAsyncIteration:
                    # Stream has ended
                    break
                except Exception as e:
                    # Handle ValidationException properly
                    if "ValidationException" in str(e):
                        error_message = str(e)
                        print(f"Validation error: {error_message}")
                    else:
                        print(f"Error receiving response: {e}")
                    break
                    
        except Exception as e:
            print(f"Response processing error: {e}")
        finally:
            self.is_active = False
    
    async def processToolUse(self, toolName, toolUseContent):
        """Process tool use requests using hardcoded logic (matching reference implementation)."""
        logger.debug(f"Processing tool use: {toolName}")
        logger.debug(f"Tool use content structure: {toolUseContent}")
        
        try:
            # Process send_task tool 
            if toolName.lower() == "send_task":
                # Extract input content from toolUseContent (match reference pattern)
                content = toolUseContent.get("content", "{}")
                try:
                    content_data = json.loads(content)
                except json.JSONDecodeError:
                    content_data = {}
                
                # Extract query from the input
                query = content_data.get('query', '')
                logger.debug(f"Extracted query: {query}")
                
                if not query:
                    result = {"error": "No query provided"}
                else:
                    # Call the send_task function directly
                    logger.info(f"🔧 Calling send_task with query: {query}")
                    result = await self.send_a2a_task(query)
                    logger.debug(f"Tool function result: {result}")
                
                # Return result in the same format as reference implementation
                return result
                
            elif toolName.lower() == "get_results":
                logger.info(f"🔧 Calling get_results")
                result = await self.get_all_results()
                return result
            else:
                # Unknown tool (matching reference error handling)
                logger.warning(f"Unknown tool: {toolName}")
                result = {"error": f"Unknown tool: {toolName}"}
                return result
            
        except Exception as e:
            logger.error(f"Error processing tool use: {e}")
            if DEBUG:
                import traceback
                traceback.print_exc()
            
            # Return error result (matching reference format)
            result = {"error": str(e)}
            return result
    
    async def send_a2a_task(self, query: str) -> dict:
        """Send a task to the A2A agent (non-blocking) and handle completion in background."""
        try:
            logger.debug(f"send_a2a_task called with query: {query}")
            
            # Use the same clean pattern as interactive mode
            def callback(task: Task):
                """Webhook callback - process result with Polly when task completes"""
                result = self.a2a_client.extract_task_result(task)
                logger.info(f"🔔 Task {task.id} completed: {result[:100]}...")

                # Store result and announce completion
                self.task_results.append({"task_id": task.id, "result": result})

                task_completed_message = f"The task has completed. Get the results."
                asyncio.run_coroutine_threadsafe(
                    self.speak_text_via_polly(task_completed_message),
                    asyncio.get_event_loop()
                )
            
            # Direct call to send_task (same as interactive mode) - returns immediately
            task_id = await self.a2a_client.send_task(query, callback)
            
            if task_id == "immediate-response":
                # Handle immediate responses (same as interactive mode)
                agent_name = self.a2a_client.agent_name
                if self.a2a_client.agent_card:
                    agent_name = self.a2a_client.agent_card.name
                logger.info(f"📨 Immediate response from {agent_name}")
                return {"status": "completed", "result": "Response received"}
            else:
                # Task created successfully (same as interactive mode)
                agent_name = self.a2a_client.agent_name
                if self.a2a_client.agent_card:
                    agent_name = self.a2a_client.agent_card.name
                logger.info(f"🚀 Task {task_id} created for {agent_name}, waiting for webhook...")
                return {"status": "task_sent", "task_id": task_id, "message": "Processing, tell the user you're working on it..."}
                
        except Exception as e:
            error_msg = f"Error sending A2A task: {str(e)}"
            logger.error(error_msg)
            
            # Determine user-friendly error message for audio
            if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                user_error_msg = "The request timed out. Please try again."
            elif "connection" in str(e).lower():
                user_error_msg = "Trouble connecting to the agent. Please try again."
            elif "credential" in str(e).lower() or "auth" in str(e).lower():
                user_error_msg = "Authentication issues. Please check the setup."
            else:
                user_error_msg = "Something went wrong. Please try again."
            
            # Speak user-friendly error message (not the technical details)
            asyncio.run_coroutine_threadsafe(
                self.speak_text_via_polly(user_error_msg),
                asyncio.get_event_loop()
            )
            return {"status": "error", "error": "Audio error provided"}
    
    def generate_audio_stream(self, text: str):
        """Generate audio stream using Amazon Polly"""
        # Synthesize speech using Polly
        response = self.polly_client.synthesize_speech(
            Text=text,
            OutputFormat='pcm',
            VoiceId='Joanna',
            SampleRate=str(INPUT_SAMPLE_RATE)
        )
        
        # Get the audio stream and inject as input chunks
        return response['AudioStream'].read()

    async def speak_text_via_polly(self, text: str):
        """Reusable method to speak text via Amazon Polly and inject as audio input."""
        try:
            logger.info(f"🔊 Speaking with Polly: {text[:100]}...")
            
            # Set flag to prevent barge-in during Polly injection
            self.injecting_polly_audio = True
            
            if self._audio_cache.get(text):
                logger.info(f"🎵 Using cached audio for: {text[:100]}...")
            else:
                logger.info(f"🎤 Generating new audio with Polly for: {text[:100]}...")
                self._audio_cache[text] = self.generate_audio_stream(text)

            audio_stream = self._audio_cache[text]
                
            chunk_size = 1024
            
            for i in range(0, len(audio_stream), chunk_size):
                chunk = audio_stream[i:i + chunk_size]
                self.add_audio_chunk(chunk)
                await asyncio.sleep(0.01)  # Small delay between chunks
            
            # Wait for processing, then re-enable barge-in
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error speaking with Polly: {e}")
        finally:
            self.injecting_polly_audio = False
    
    async def get_all_results(self):
        """Retrieve all results and return them as tool output."""
        try:
            if not self.task_results:
                # No results available
                logger.info("📋 No results available")
                return {"message": "No results available yet."}
                        
            logger.info(f"📋 Returning {len(self.task_results)} results")
            
            # Clear the results list after returning them
            import copy
            results = copy.deepcopy(self.task_results)
            # Keep the last TASK_RESULT_SIZE results
            self.task_results = self.task_results[-TASK_RESULT_SIZE:]
            
            logger.info(f"📋 Results: {results}")

            return {
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error getting results: {e}")
            return {"status": "error", "error": str(e)}
    
    async def send_tool_start_event(self, content_name):
        """Send tool content start event."""
        event = self.TOOL_CONTENT_START_EVENT % (self.prompt_name, content_name, self.toolUseId)
        await self.send_raw_event(event)
    
    async def send_tool_result_event(self, content_name, tool_result):
        """Send tool result event."""
        tool_result_event = self.tool_result_event(content_name=content_name, content=tool_result, role="TOOL")
        await self.send_raw_event(tool_result_event)
    
    async def send_tool_content_end_event(self, content_name):
        """Send tool content end event."""
        event = self.CONTENT_END_EVENT % (self.prompt_name, content_name)
        await self.send_raw_event(event)
    
    async def close(self):
        """Close the stream and cleanup."""
        self.is_active = False
        
        # Cancel response processing task first
        if self.response_task and not self.response_task.done():
            self.response_task.cancel()
            try:
                await asyncio.wait_for(self.response_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception as e:
                debug_print(f"Error cancelling response task: {e}")
        
        # Send session end event and close stream
        try:
            if self.stream_response:
                await self.send_audio_content_end_event()
                await self.send_prompt_end_event()
                await self.send_session_end_event()
                await self.stream_response.input_stream.close()
                self.stream_response = None
        except Exception as e:
            debug_print(f"Error closing stream response: {e}")
        
        # Clear queues
        while not self.audio_input_queue.empty():
            try:
                self.audio_input_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        while not self.audio_output_queue.empty():
            try:
                self.audio_output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break


class AudioStreamer:
    """Handles audio input/output streaming for speak mode."""
    
    def __init__(self, stream_manager, disable_echo_cancellation=False):
        self.stream_manager = stream_manager
        self.is_streaming = False
        self.loop = asyncio.get_event_loop()

        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        # Echo cancellation setup
        self.apm = None
        self.echo_cancellation_enabled = APM_AVAILABLE and not disable_echo_cancellation
        self.frame_duration_ms = 10  # WebRTC APM requires 10ms frames
        self.input_frames_per_10ms = int(INPUT_SAMPLE_RATE * self.frame_duration_ms / 1000)
        self.output_frames_per_10ms = int(OUTPUT_SAMPLE_RATE * self.frame_duration_ms / 1000)
        
        # Audio buffers for 10ms frame processing
        self.input_buffer = np.array([], dtype=np.int16)
        self.output_buffer = np.array([], dtype=np.int16)
        
        # Timing for delay estimation
        self.last_output_time = 0
        self.last_input_time = 0
        
        if self.echo_cancellation_enabled:
            try:
                debug_print("Initializing WebRTC Audio Processing Module for echo cancellation...")
                self.apm = rtc.AudioProcessingModule(
                    echo_cancellation=True,
                    noise_suppression=True,
                    high_pass_filter=True,
                    auto_gain_control=True
                )
                # Set initial stream delay (can be adjusted dynamically)
                self.apm.set_stream_delay_ms(50)  # 50ms initial delay estimate
                logger.info("✅ Audio Processing Module initialized for echo cancellation")
            except Exception as e:
                logger.warning(f"Failed to initialize WebRTC APM: {e}")
                self.echo_cancellation_enabled = False
                self.apm = None

        # Initialize separate streams for input and output
        # Input stream with callback for microphone
        self.input_stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=INPUT_SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=self.input_callback
        )

        # Output stream for direct writing (no callback)
        self.output_stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=OUTPUT_SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SIZE
        )
    
    def input_callback(self, in_data, frame_count, time_info, status):
        """Callback function that schedules audio processing in the asyncio event loop"""
        if self.is_streaming and in_data:
            self.last_input_time = time.time()
            # Schedule the task in the event loop
            asyncio.run_coroutine_threadsafe(
                self.process_input_audio(in_data), 
                self.loop
            )
        return (None, pyaudio.paContinue)
    
    async def process_input_audio(self, audio_data):
        """Process input audio with echo cancellation before sending to Bedrock"""
        try:
            if self.echo_cancellation_enabled and self.apm:
                # Convert audio data to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Add to buffer
                self.input_buffer = np.concatenate([self.input_buffer, audio_array])
                
                # Process in 10ms chunks
                processed_audio = b''
                while len(self.input_buffer) >= self.input_frames_per_10ms:
                    # Extract 10ms frame
                    frame_data = self.input_buffer[:self.input_frames_per_10ms]
                    self.input_buffer = self.input_buffer[self.input_frames_per_10ms:]
                    
                    # Create AudioFrame for APM processing
                    audio_frame = rtc.AudioFrame(
                        data=frame_data.tobytes(),
                        sample_rate=INPUT_SAMPLE_RATE,
                        num_channels=CHANNELS,
                        samples_per_channel=self.input_frames_per_10ms
                    )
                    
                    # Process the frame through APM (near-end processing)
                    self.apm.process_stream(audio_frame)
                    
                    # Append processed audio
                    processed_audio += audio_frame.data.tobytes()
                
                # Send processed audio if any
                if processed_audio:
                    self.stream_manager.add_audio_chunk(processed_audio)
            else:
                # Send audio directly if echo cancellation is not available
                self.stream_manager.add_audio_chunk(audio_data)
                
        except Exception as e:
            if self.is_streaming:
                print(f"Error processing input audio: {e}")
    
    def process_output_audio(self, audio_data):
        """Process output audio through APM for echo cancellation reference"""
        if not self.echo_cancellation_enabled or not self.apm:
            return audio_data
            
        try:
            # Convert to numpy array and resample if needed
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Add to buffer
            self.output_buffer = np.concatenate([self.output_buffer, audio_array])
            
            processed_audio = b''
            while len(self.output_buffer) >= self.output_frames_per_10ms:
                # Extract 10ms frame
                frame_data = self.output_buffer[:self.output_frames_per_10ms]
                self.output_buffer = self.output_buffer[self.output_frames_per_10ms:]
                
                # Create AudioFrame for APM processing
                audio_frame = rtc.AudioFrame(
                    data=frame_data.tobytes(),
                    sample_rate=OUTPUT_SAMPLE_RATE,
                    num_channels=CHANNELS,
                    samples_per_channel=self.output_frames_per_10ms
                )
                
                # Process the reverse stream (far-end processing for echo cancellation)
                self.apm.process_reverse_stream(audio_frame)
                
                # Append processed audio
                processed_audio += audio_frame.data.tobytes()
            
            return processed_audio if processed_audio else audio_data
            
        except Exception as e:
            if DEBUG:
                debug_print(f"Error processing output audio for echo cancellation: {e}")
            return audio_data
    
    def update_stream_delay(self):
        """Update the stream delay for better echo cancellation"""
        if self.echo_cancellation_enabled and self.apm and self.last_output_time > 0 and self.last_input_time > 0:
            try:
                # Calculate delay between output and input processing
                delay_ms = int(abs(self.last_input_time - self.last_output_time) * 1000)
                # Clamp delay to reasonable bounds
                delay_ms = max(10, min(delay_ms, 500))
                self.apm.set_stream_delay_ms(delay_ms)
            except Exception as e:
                debug_print(f"Error updating stream delay: {e}")
    
    async def play_output_audio(self):
        """Play audio from the output queue."""
        while self.is_streaming:
            try:
                # Check for barge-in
                if self.stream_manager.barge_in:
                    while not self.stream_manager.audio_output_queue.empty():
                        try:
                            self.stream_manager.audio_output_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    self.stream_manager.barge_in = False
                    await asyncio.sleep(0.05)
                    continue
                
                audio_data = await asyncio.wait_for(
                    self.stream_manager.audio_output_queue.get(),
                    timeout=0.1
                )
                
                if audio_data and self.is_streaming:
                    self.last_output_time = time.time()
                    processed_audio = self.process_output_audio(audio_data)
                    self.update_stream_delay()
                    
                    chunk_size = CHUNK_SIZE * 2
                    
                    for i in range(0, len(processed_audio), chunk_size):
                        if not self.is_streaming:
                            break
                        
                        end = min(i + chunk_size, len(processed_audio))
                        chunk = processed_audio[i:end]
                        
                        def write_chunk(data):
                            return self.output_stream.write(data)
                        
                        await asyncio.get_event_loop().run_in_executor(None, write_chunk, chunk)
                        await asyncio.sleep(0.001)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if self.is_streaming:
                    print(f"Error playing output audio: {str(e)}")
                    import traceback
                    traceback.print_exc()
                await asyncio.sleep(0.05)
    
    async def start_streaming(self):
        """Start streaming audio."""
        if self.is_streaming:
            return
        
        print("Starting audio streaming. Speak into your microphone...")
        print("Press Enter to stop streaming...")
        
        # Send audio content start event
        await time_it_async("send_audio_content_start_event", 
                           self.stream_manager.send_audio_content_start_event)
        
        self.is_streaming = True
        
        # Start the input stream if not already started
        if not self.input_stream.is_active():
            self.input_stream.start_stream()
        
        # Start processing tasks
        self.output_task = asyncio.create_task(self.play_output_audio())
        
        # Wait for user to press Enter to stop
        await asyncio.get_event_loop().run_in_executor(None, input)
        
        # Once input() returns, stop streaming
        await self.stop_streaming()
    
    async def stop_streaming(self):
        """Stop streaming audio."""
        if not self.is_streaming:
            return
            
        self.is_streaming = False

        # Cancel the tasks
        tasks = []
        if hasattr(self, 'output_task') and not self.output_task.done():
            tasks.append(self.output_task)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Stop and close the streams
        if self.input_stream:
            if self.input_stream.is_active():
                self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            if self.output_stream.is_active():
                self.output_stream.stop_stream()
            self.output_stream.close()
        if self.p:
            self.p.terminate()
        
        await self.stream_manager.close()


async def voice_chat_loop(client: CleanA2AClient, disable_echo_cancellation: bool):
    """
    Voice chat loop placeholder.
    
    Note: This is a simplified version. The full voice implementation would require:
    - Audio input/output handling (PyAudio)
    - Speech-to-text conversion (AWS Bedrock Nova Sonic)
    - Text-to-speech for responses
    - Echo cancellation
    """
    print("🎤 Voice chat mode is not fully implemented in this version.")
    print("💡 This would require audio processing dependencies and AWS Bedrock Nova Sonic.")
    print("🔄 Falling back to text input mode...")
    
    # Simple text-based interaction as fallback
    while True:
        try:
            user_input = input(f"\n👤 You (text) → {client.agent_name}: ").strip()
            
            if user_input.lower() in ['/quit', '/exit', '/bye']:
                break
            elif user_input:
                result = await client.send_message(user_input)
                print(f"🤖 {client.agent_name}: {result}")
                
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            print(f"❌ Error: {e}")


async def speak_mode(agent_name: str, disable_echo_cancellation: bool = False):
    """Voice interaction mode with a specific agent."""
    global _cleanup_in_progress
    setup_signal_handlers()
    
    # Configure logging for cleaner output
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("a2a.client.client").setLevel(logging.WARNING)
    
    print(f"🗣️  A2A Voice Mode with {agent_name}")
    print("=" * 40)
    
    # Get agent URL from configuration
    agents_config = discover_available_agents()
    if agent_name not in agents_config:
        print(f"❌ Agent '{agent_name}' not found in configuration")
        available = list(agents_config.keys())
        print(f"Available agents: {', '.join(available)}")
        return
    
    agent_url = agents_config[agent_name]
    
    # Initialize client
    client = CleanA2AClient(agent_url, agent_name)
    
    try:
        # Test connection and discover skills
        if not await client.discover_agent_skills():
            print(f"⚠️  Connected to {agent_name} but skills discovery failed")
        else:
            print(f"✅ Connected to {agent_name}")
            if client.agent_card and client.agent_card.skills:
                skills = [skill.name for skill in client.agent_card.skills]
                print(f"🎯 Available skills: {', '.join(skills)}")
        
        # Start voice interaction
        await voice_chat_loop(client, disable_echo_cancellation)
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await client.close()
        print("✅ Voice mode ended") 
// A2A WebSpeak Interface - Updated with proper AudioPlayer integration for barge-in

console.log('WebSpeak Interface loading...');

// Import AudioPlayer for proper barge-in support
import { AudioPlayer } from './lib/play/AudioPlayer.js';
// Import shared components
import { LogsManager } from './lib/shared-logs-manager.js';
import { ChatManager } from './lib/shared-chat-manager.js';
import { ConnectionManager } from './lib/shared-connection-manager.js';

class WebSpeakInterface {
    constructor() {
        this.socket = io();
        this.isConnected = false;
        this.isStreaming = false;
        this.selectedAgent = null;
        
        this.audioContext = null;
        this.audioStream = null;
        this.processor = null;
        
        // Use proper AudioPlayer for barge-in support (same as Node.js reference)
        this.audioPlayer = new AudioPlayer();
        this.audioPlayerInitialized = false;
        
        this.initializeElements();
        
        // Initialize shared components
        this.chatManager = new ChatManager(this.chatContainer);
        this.connectionManager = new ConnectionManager(this.connectionStatus);
        this.logsManager = new LogsManager(this.agentLogs);
        
        this.setupSocketEvents();
        this.loadAgents();
        this.logsManager.startLogsUpdates(); // Always start logs updates
    }
    
    initializeElements() {
        this.agentSelect = document.getElementById('agent-select');
        this.connectBtn = document.getElementById('connect-btn');
        this.status = document.getElementById('status');
        this.chatContainer = document.getElementById('chat-container');
        this.startBtn = document.getElementById('start-btn');
        this.stopBtn = document.getElementById('stop-btn');
        this.connectionStatus = document.getElementById('connection-status');
        
        // Logs elements
        this.agentLogs = document.getElementById('agent-logs');
        
        // Event listeners
        this.connectBtn.addEventListener('click', () => this.connectToAgent());
        this.startBtn.addEventListener('click', () => this.startConversation());
        this.stopBtn.addEventListener('click', () => this.stopConversation());
        
        this.agentSelect.addEventListener('change', () => {
            this.connectBtn.disabled = !this.agentSelect.value;
        });
    }
    
    setupSocketEvents() {
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.updateStatus('Connected to server');
            this.connectionManager.handleConnect();
        });
        
        this.socket.on('connected', (data) => {
            console.log('Session created:', data);
        });
        
        this.socket.on('agent_initialized', (data) => {
            console.log('Agent initialized:', data);
            this.isConnected = true;
            this.startBtn.disabled = false;
            this.updateStatus(`Connected to ${data.name} - Ready to start conversation`);
            this.chatManager.addMessage('system', `Connected to ${data.name}`);
        });
        
        this.socket.on('audio_started', () => {
            console.log('Audio streaming started');
            this.isStreaming = true;
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.updateStatus('Recording... Speak now');
        });
        
        this.socket.on('audio_stopped', () => {
            console.log('Audio streaming stopped');
            this.isStreaming = false;
            this.startBtn.disabled = false;
            this.stopBtn.disabled = true;
            this.updateStatus('Ready to start conversation');
        });
        
        this.socket.on('audio_response', (data) => {
            this.playAudio(data.audio);
        });
        
        this.socket.on('barge_in', () => {
            console.log('Barge-in detected - clearing audio buffer');
            this.bargeIn();
        });
        
        // Handle content end events (same as Node.js reference)
        this.socket.on('content_end', (data) => {
            console.log('Content end received:', data);
            
            if (data.type === 'TEXT') {
                // Handle stop reasons (same as Node.js reference)
                if (data.stopReason && data.stopReason.toUpperCase() === 'INTERRUPTED') {
                    console.log("Interrupted by user - triggering barge-in");
                    this.bargeIn();
                }
            }
        });
        
        this.socket.on('text_output', (data) => {
            console.log('Text output:', data);
            const role = data.role || 'assistant';
            const content = data.content || '';
            this.chatManager.addMessage(role, content);
        });
        
        this.socket.on('tool_use', (data) => {
            console.log('Tool use:', data);
            const message = data.message || 'Tool used';
            this.chatManager.addMessage('system', `🔧 ${message}`);
        });
        
        this.socket.on('error', (data) => {
            console.error('Error:', data);
            this.updateStatus(`Error: ${data.message || 'Unknown error'}`);
            this.chatManager.addError(data.message || 'Unknown error occurred');
        });
    }
    
    // updateConnectionStatus now handled by ConnectionManager
    
    async loadAgents() {
        try {
            const response = await fetch('/api/agents');
            const data = await response.json();
            
            this.agentSelect.innerHTML = '<option value="">Select an agent...</option>';
            
            Object.entries(data.agents).forEach(([name, info]) => {
                const option = document.createElement('option');
                option.value = name;
                option.textContent = name;
                this.agentSelect.appendChild(option);
            });
            
        } catch (error) {
            console.error('Error loading agents:', error);
            this.updateStatus('Failed to load agents');
        }
    }
    
    connectToAgent() {
        const agentName = this.agentSelect.value;
        if (!agentName) return;
        
        this.selectedAgent = agentName;
        this.updateStatus('Connecting to agent...');
        this.socket.emit('initialize_agent', { agent_name: agentName });
    }
    
    async startConversation() {
        if (!this.isConnected) return;
        
        try {
            await this.initializeAudio();
            await this.initializeAudioPlayer();
            this.socket.emit('start_audio');
            this.updateStatus('Starting audio...');
        } catch (error) {
            console.error('Error starting conversation:', error);
            this.updateStatus(`Error: ${error.message}`);
        }
    }
    
    stopConversation() {
        this.socket.emit('stop_audio');
        this.stopAudioProcessing();
    }
    
    async initializeAudio() {
        if (this.audioContext) return;
        
        try {
            // Get microphone access
            this.audioStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    channelCount: 1,
                    sampleRate: 16000
                }
            });
            
            // Create audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000
            });
            
            // Create audio processing
            const source = this.audioContext.createMediaStreamSource(this.audioStream);
            this.processor = this.audioContext.createScriptProcessor(1024, 1, 1);
            
            this.processor.onaudioprocess = (e) => {
                if (!this.isStreaming) return;
                
                const inputData = e.inputBuffer.getChannelData(0);
                const pcmData = new Int16Array(inputData.length);
                
                // Convert to 16-bit PCM
                for (let i = 0; i < inputData.length; i++) {
                    pcmData[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7FFF;
                }
                
                // Send as base64
                const base64Data = this.arrayBufferToBase64(pcmData.buffer);
                this.socket.emit('audio_chunk', { audio: base64Data });
            };
            
            source.connect(this.processor);
            this.processor.connect(this.audioContext.destination);
            
        } catch (error) {
            throw new Error(`Failed to initialize audio: ${error.message}`);
        }
    }
    
    stopAudioProcessing() {
        this.isStreaming = false;
        
        if (this.processor) {
            this.processor.disconnect();
        }
        
        if (this.audioStream) {
            this.audioStream.getTracks().forEach(track => track.stop());
        }
    }
    
    // Initialize AudioPlayer (same pattern as Node.js reference)
    async initializeAudioPlayer() {
        if (!this.audioPlayerInitialized) {
            try {
                await this.audioPlayer.start();
                this.audioPlayerInitialized = true;
                console.log('AudioPlayer initialized with proper barge-in support');
            } catch (error) {
                console.error('Failed to initialize AudioPlayer:', error);
                throw error;
            }
        }
    }
    
    // Updated audio playback using AudioPlayer (same as Node.js reference)
    async playAudio(base64Audio) {
        try {
            await this.initializeAudioPlayer();
            
            // Decode base64 to Float32Array (same as Node.js reference)
            const binaryString = atob(base64Audio);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            
            const int16Array = new Int16Array(bytes.buffer);
            const float32Array = new Float32Array(int16Array.length);
            for (let i = 0; i < int16Array.length; i++) {
                float32Array[i] = int16Array[i] / 32768.0;
            }
            
            // Use AudioPlayer for proper buffering and barge-in support
            this.audioPlayer.playAudio(float32Array);
            
        } catch (error) {
            console.error('Error playing audio:', error);
        }
    }
    
    // Proper barge-in implementation (same as Node.js reference)
    bargeIn() {
        if (this.audioPlayerInitialized && this.audioPlayer) {
            console.log('Triggering barge-in via AudioPlayer');
            this.audioPlayer.bargeIn();
        } else {
            console.warn('AudioPlayer not initialized - cannot perform barge-in');
        }
    }
    
    // addMessage now handled by ChatManager
    
    // Log management now handled by LogsManager
    
    updateStatus(message) {
        this.status.textContent = message;
        console.log('Status:', message);
    }
    
    arrayBufferToBase64(buffer) {
        const binary = [];
        const bytes = new Uint8Array(buffer);
        for (let i = 0; i < bytes.byteLength; i++) {
            binary.push(String.fromCharCode(bytes[i]));
        }
        return btoa(binary.join(''));
    }
    
    // updateLogsDynamicHeight now handled by LogsManager
}

// Initialize when DOM loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing WebSpeak...');
    window.webSpeakInterface = new WebSpeakInterface();
});
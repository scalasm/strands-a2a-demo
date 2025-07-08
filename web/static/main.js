class A2AWebInterface {
    constructor() {
        this.ws = null;
        this.selectedAgent = null;
        this.useTasks = true;
        
        this.initializeElements();
        this.setupEventListeners();
        this.connect();
        
        // Initialize shared components asynchronously
        this.initializeSharedComponents();
    }

    async initializeSharedComponents() {
        // Dynamic imports to avoid module issues
        const { LogsManager } = await import('./lib/shared-logs-manager.js');
        const { ChatManager } = await import('./lib/shared-chat-manager.js');
        const { ConnectionManager } = await import('./lib/shared-connection-manager.js');
        
        this.chatManager = new ChatManager(this.chatMessages);
        this.connectionManager = new ConnectionManager(this.connectionStatus);
        this.logsManager = new LogsManager(this.agentLogs);
        
        this.logsManager.startLogsUpdates();
    }
    
    initializeElements() {
        this.agentList = document.getElementById('agent-list');
        this.chatMessages = document.getElementById('chat-messages');
        this.messageInput = document.getElementById('message-input');
        this.sendButton = document.getElementById('send-button');
        this.loading = document.getElementById('loading');
        this.connectionStatus = document.getElementById('connection-status');
        this.modeButtons = document.querySelectorAll('.mode-button');
        this.agentLogs = document.getElementById('agent-logs');
        
        // Add auto-scroll class to chat messages
        this.chatMessages.classList.add('auto-scroll-chat');
    }
    
    setupEventListeners() {
        // Send message on button click
        this.sendButton.addEventListener('click', () => this.sendMessage());
        
        // Send message on Enter key
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Mode selection
        this.modeButtons.forEach(button => {
            button.addEventListener('click', () => this.selectMode(button.dataset.mode));
        });
        
        // Scroll tracking now handled by ChatManager
    }
    
    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            if (this.connectionManager) {
                this.connectionManager.handleConnect();
            }
            this.requestAgentList();
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            if (this.connectionManager) {
                this.connectionManager.handleDisconnect();
            }
            this.attemptReconnect();
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            if (this.connectionManager) {
                this.connectionManager.handleError();
            }
        };
    }
    
    attemptReconnect() {
        if (this.connectionManager && this.connectionManager.shouldReconnect()) {
            this.connectionManager.incrementReconnectAttempts();
            this.connectionManager.updateStatus(this.connectionManager.getReconnectStatus(), false);
            setTimeout(() => this.connect(), this.connectionManager.getReconnectDelay());
        } else if (this.connectionManager) {
            this.connectionManager.setConnectionFailed();
        }
    }
    
    requestAgentList() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'get_agents' }));
        }
    }
    
    handleMessage(data) {
        switch (data.type) {
            case 'agents_list':
                this.populateAgentList(data.agents);
                break;
            case 'message':
                if (this.chatManager) {
                    this.chatManager.addMessage(data.message_type, data.content, data.timestamp, data.agent);
                }
                this.hideLoading();
                break;
            case 'task_created':
                if (this.chatManager) {
                    this.chatManager.addMessage('system', `Task ${data.task_id} created, waiting for response...`);
                }
                break;
            case 'task_completed':
                if (this.chatManager) {
                    this.chatManager.addMessage('agent', data.result, null, data.agent);
                }
                this.hideLoading();
                break;
            case 'error':
                if (this.chatManager) {
                    this.chatManager.addError(data.message);
                }
                this.hideLoading();
                break;
        }
    }
    
    populateAgentList(agents) {
        this.agentList.innerHTML = '';
        
        agents.forEach(agent => {
            const agentElement = document.createElement('div');
            agentElement.className = 'agent-item';
            agentElement.dataset.agent = agent.name;
            
            const skillTags = agent.skills.map(skill => 
                `<span class="skill-tag">${skill.name}</span>`
            ).join('');
            
            agentElement.innerHTML = `
                <div class="agent-name">${agent.name}</div>
                <div class="agent-description">${agent.description}</div>
                <div class="agent-skills">${skillTags}</div>
            `;
            
            agentElement.addEventListener('click', () => this.selectAgent(agent.name));
            this.agentList.appendChild(agentElement);
        });
    }
    
    selectAgent(agentName) {
        // Update UI
        document.querySelectorAll('.agent-item').forEach(item => {
            item.classList.remove('selected');
        });
        document.querySelector(`[data-agent="${agentName}"]`).classList.add('selected');
        
        // Update state
        this.selectedAgent = agentName;
        this.messageInput.disabled = false;
        this.sendButton.disabled = false;
        this.messageInput.placeholder = `Type your message to ${agentName}...`;
        
        // Add system message
        if (this.chatManager) {
            this.chatManager.addMessage('system', `Connected to ${agentName}`);
        }
        
        this.messageInput.focus();
    }
    
    selectMode(mode) {
        this.modeButtons.forEach(btn => btn.classList.remove('active'));
        document.querySelector(`[data-mode="${mode}"]`).classList.add('active');
        this.useTasks = mode === 'tasks';
        
        if (this.chatManager) {
            this.chatManager.addMessage('system', `Switched to ${mode} mode`);
        }
    }
    
    sendMessage() {
        if (!this.selectedAgent || !this.messageInput.value.trim()) {
            return;
        }
        
        const message = this.messageInput.value.trim();
        this.messageInput.value = '';
        
        // Display user message
        if (this.chatManager) {
            this.chatManager.addMessage('user', message);
        }
        
        // Show loading
        this.showLoading();
        
        // Send to WebSocket
        this.ws.send(JSON.stringify({
            type: 'send_message',
            agent: this.selectedAgent,
            message: message,
            use_tasks: this.useTasks
        }));
    }
    
    // Message display now handled by ChatManager
    
    showLoading() {
        this.loading.classList.add('show');
        this.sendButton.disabled = true;
    }
    
    hideLoading() {
        this.loading.classList.remove('show');
        this.sendButton.disabled = false;
    }
    
    // Chat scrolling and log management now handled by shared components
    
    // Cleanup when page unloads
    destroy() {
        if (this.logsManager) {
            this.logsManager.stopLogsUpdates();
        }
        if (this.ws) {
            this.ws.close();
        }
    }
}

// Initialize the interface when the page loads
document.addEventListener('DOMContentLoaded', () => {
    const interface = new A2AWebInterface();
    
    // Cleanup on page unload
    window.addEventListener('beforeunload', () => {
        interface.destroy();
    });
}); 
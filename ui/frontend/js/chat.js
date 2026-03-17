class ChatManager {
    constructor() {
        this.currentSessionId = null;
        this.messages = [];
        this.isStreaming = false;
    }
    
    async init() {
        this.bindEvents();
        this.updateConfigDisplay();
    }
    
    bindEvents() {
        const sendBtn = document.getElementById('sendBtn');
        const messageInput = document.getElementById('messageInput');
        const saveConfigBtn = document.getElementById('saveConfigBtn');
        const newChatBtn = document.getElementById('newChatBtn');
        
        sendBtn.addEventListener('click', () => this.sendMessage());
        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        saveConfigBtn.addEventListener('click', () => this.saveConfig());
        newChatBtn.addEventListener('click', () => this.createNewSession());
    }
    
    updateConfigDisplay() {
        document.getElementById('apiKey').value = API.apiKey;
        document.getElementById('apiBaseUrl').value = API.baseUrl;
        document.getElementById('modelName').value = API.model;
    }
    
    saveConfig() {
        const apiKey = document.getElementById('apiKey').value;
        const apiBaseUrl = document.getElementById('apiBaseUrl').value;
        const modelName = document.getElementById('modelName').value;
        
        if (!apiKey) {
            alert('请输入 API Key');
            return;
        }
        
        API.saveConfig(apiBaseUrl, apiKey, modelName);
        alert('配置已保存');
    }
    
    async createNewSession() {
        try {
            const response = await fetch(`${API.baseUrl}/sessions`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${API.apiKey}`
                }
            });
            
            if (!response.ok) {
                throw new Error('创建会话失败');
            }
            
            const session = await response.json();
            
            let sessions = JSON.parse(localStorage.getItem('sessions') || '[]');
            sessions.unshift(session);
            localStorage.setItem('sessions', JSON.stringify(sessions));
            
            this.currentSessionId = session.id;
            this.messages = [];
            
            window.workflowManager.clearWorkflow();
            this.renderSessions();
            this.clearChat();
            
            window.workflowManager.updateWorkflow([
                { id: 'start', name: '开始', status: 'completed', time: '0s' }
            ]);
        } catch (error) {
            console.error('Failed to create session:', error);
            alert('创建会话失败: ' + error.message);
        }
    }
    
    renderSessions() {
        const sessionList = document.getElementById('sessionList');
        const sessions = JSON.parse(localStorage.getItem('sessions') || '[]');
        
        if (sessions.length === 0) {
            sessionList.innerHTML = '<div class="text-gray-500 text-center p-4 text-sm">暂无会话</div>';
            return;
        }
        
        sessionList.innerHTML = sessions.map(session => `
            <div class="session-item ${session.id === this.currentSessionId ? 'active' : ''}" data-id="${session.id}">
                <span class="truncate">${session.title}</span>
                <button class="delete-btn" data-id="${session.id}">×</button>
            </div>
        `).join('');
        
        sessionList.querySelectorAll('.session-item').forEach(item => {
            item.addEventListener('click', (e) => {
                if (!e.target.classList.contains('delete-btn')) {
                    this.loadSession(item.dataset.id);
                }
            });
        });
        
        sessionList.querySelectorAll('.delete-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.deleteSession(btn.dataset.id);
            });
        });
    }
    
    async loadSession(sessionId) {
        this.currentSessionId = sessionId;
        this.messages = JSON.parse(localStorage.getItem(`messages_${sessionId}`) || '[]');
        
        this.renderSessions();
        this.renderMessages();
        
        const sessions = JSON.parse(localStorage.getItem('sessions') || '[]');
        const session = sessions.find(s => s.id === sessionId);
        if (session) {
            window.workflowManager.updateWorkflow([
                { id: 'start', name: '开始', status: 'completed', time: '0s' }
            ]);
        }
    }
    
    async deleteSession(sessionId) {
        if (!confirm('确定删除此会话？')) return;
        
        try {
            await fetch(`${API.baseUrl}/sessions/${sessionId}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${API.apiKey}`
                }
            });
        } catch (e) {
            console.error('Delete session error:', e);
        }
        
        let sessions = JSON.parse(localStorage.getItem('sessions') || '[]');
        sessions = sessions.filter(s => s.id !== sessionId);
        localStorage.setItem('sessions', JSON.stringify(sessions));
        localStorage.removeItem(`messages_${sessionId}`);
        
        if (this.currentSessionId === sessionId) {
            this.currentSessionId = null;
            this.messages = [];
            this.clearChat();
        }
        
        this.renderSessions();
    }
    
    clearChat() {
        const chatContainer = document.getElementById('chatContainer');
        chatContainer.innerHTML = `
            <div class="flex gap-3">
                <div class="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center flex-shrink-0">
                    <span class="text-white text-sm">🤖</span>
                </div>
                <div class="bg-gray-800 rounded-lg px-4 py-3 max-w-[80%]">
                    <p class="text-gray-300">欢迎使用 AutoML 对话式建模系统！</p>
                    <p class="text-gray-400 text-sm mt-2">请先在左侧新建一个会话，然后可以：</p>
                    <ul class="text-gray-400 text-sm mt-1 list-disc list-inside">
                        <li>上传数据文件</li>
                        <li>描述建模需求</li>
                        <li>查看工作流执行状态</li>
                    </ul>
                </div>
            </div>
        `;
    }
    
    async sendMessage() {
        if (this.isStreaming) return;
        
        const input = document.getElementById('messageInput');
        const message = input.value.trim();
        
        if (!message) return;
        
        if (!this.currentSessionId) {
            alert('请先创建或选择一个会话');
            return;
        }
        
        if (!API.apiKey) {
            alert('请先配置 API Key');
            return;
        }
        
        this.messages.push({ role: 'user', content: message });
        this.renderMessages();
        input.value = '';
        
        this.isStreaming = true;
        this.showTypingIndicator();
        
        try {
            await this.streamResponse(message);
        } catch (error) {
            console.error('Send message error:', error);
            this.addMessage({
                role: 'assistant',
                content: `抱歉，发生错误: ${error.message}`
            });
        } finally {
            this.hideTypingIndicator();
            this.isStreaming = false;
        }
    }
    
    async streamResponse(message) {
        const response = await fetch(`${API.baseUrl}/chat/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${API.apiKey}`
            },
            body: JSON.stringify({
                session_id: this.currentSessionId,
                message: message,
                model: API.model
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let fullContent = '';
        let currentStep = '';
        
        const assistantMessage = { role: 'assistant', content: '' };
        this.messages.push(assistantMessage);
        this.renderMessages();
        
        // 使用节流来优化渲染性能
        let lastUpdateTime = 0;
        const updateInterval = 100; // 每100ms更新一次
        let pendingContent = '';
        
        const throttledUpdate = () => {
            const now = Date.now();
            if (now - lastUpdateTime >= updateInterval && pendingContent !== fullContent) {
                pendingContent = fullContent;
                this.updateLastMessage(fullContent, true); // true 表示是流式更新
                lastUpdateTime = now;
            }
        };
        
        // 启动一个定时器来处理最后的更新
        const updateTimer = setInterval(throttledUpdate, updateInterval);
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    
                    if (data === '[DONE]') continue;
                    
                    try {
                        const event = JSON.parse(data);
                        
                        if (event.type === 'step') {
                            currentStep = event.name;
                            window.workflowManager.addStep(event.name, event.status);
                        } else if (event.type === 'content') {
                            fullContent += event.delta;
                            assistantMessage.content = fullContent;
                            // 使用节流更新，避免频繁渲染
                            throttledUpdate();
                        } else if (event.type === 'error') {
                            throw new Error(event.message);
                        }
                    } catch (e) {
                        if (data.trim()) {
                            fullContent += data;
                            assistantMessage.content = fullContent;
                            throttledUpdate();
                        }
                    }
                }
            }
        }
        
        // 清理定时器并确保最终更新
        clearInterval(updateTimer);
        this.updateLastMessage(fullContent, false); // false 表示最终更新，完整渲染
        this.saveMessages();
    }
    
    addMessage(message) {
        // 调试：检查 message 内容
        console.log('addMessage called with:', message);
        console.log('message.content type:', typeof message.content);
        console.log('message.content value:', message.content);
        
        // 确保 message.content 是字符串
        if (typeof message.content !== 'string') {
            console.warn('message.content is not string, converting...');
            if (message.content === null || message.content === undefined) {
                message.content = '';
            } else if (typeof message.content === 'object') {
                if (Array.isArray(message.content)) {
                    message.content = message.content.join(', ');
                } else {
                    message.content = JSON.stringify(message.content, null, 2);
                }
            } else {
                message.content = String(message.content);
            }
        }
        
        this.messages.push(message);
        this.renderMessages();
        this.saveMessages();
    }
    
    renderMessages() {
        const chatContainer = document.getElementById('chatContainer');
        
        chatContainer.innerHTML = this.messages.map(msg => {
            // 确保 content 是字符串
            let content = msg.content;
            if (typeof content !== 'string') {
                if (content === null || content === undefined) {
                    content = '';
                } else if (typeof content === 'object') {
                    // 如果是数组，直接转换为字符串
                    if (Array.isArray(content)) {
                        content = content.join(', ');
                    } else {
                        content = JSON.stringify(content, null, 2);
                    }
                } else {
                    content = String(content);
                }
            }
            
            return `
            <div class="message ${msg.role}">
                <div class="message-avatar">
                    ${msg.role === 'assistant' ? '🤖' : '👤'}
                </div>
                <div class="message-content ${msg.role === 'user' ? 'bg-blue-600' : 'bg-gray-800'}">
                    ${this.formatContent(content)}
                </div>
            </div>
        `}).join('');
        
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    updateLastMessage(content, isStreaming = false) {
        const chatContainer = document.getElementById('chatContainer');
        const messages = chatContainer.querySelectorAll('.message');
        const lastMessage = messages[messages.length - 1];
        
        if (lastMessage) {
            const contentDiv = lastMessage.querySelector('.message-content');
            
            if (isStreaming) {
                // 流式更新：只显示纯文本，不渲染 Markdown 和图片
                // 这样可以避免频繁渲染大段内容导致的卡顿
                const plainText = content.slice(0, 500) + (content.length > 500 ? '...' : '');
                contentDiv.innerHTML = `<div class="text-gray-300">${plainText.replace(/\n/g, '<br>')}</div><div class="text-blue-400 text-sm mt-2">⏳ 正在生成内容...</div>`;
            } else {
                // 最终更新：完整渲染 Markdown
                contentDiv.innerHTML = this.formatContent(content);
            }
            
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }
    
    formatContent(content) {
        // 确保 content 是字符串
        if (typeof content !== 'string') {
            if (content === null || content === undefined) {
                return '';
            } else if (typeof content === 'object') {
                // 如果是数组，直接转换为字符串
                if (Array.isArray(content)) {
                    content = content.join(', ');
                } else {
                    content = JSON.stringify(content, null, 2);
                }
            } else {
                content = String(content);
            }
        }
        
        if (!content || !content.trim()) return '';
        
        // 检查 marked 是否可用
        if (typeof marked === 'undefined') {
            console.error('marked is not defined, returning plain text');
            return `<pre class="text-gray-300">${content.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</pre>`;
        }
        
        // 配置 marked 选项
        marked.setOptions({
            breaks: true,
            gfm: true,
            headerIds: false,
            mangle: false,
            sanitize: false
        });
        
        // 自定义渲染器，支持 base64 图片
        const renderer = new marked.Renderer();
        
        // 自定义图片渲染，支持 base64
        renderer.image = (href, title, text) => {
            // 处理 base64 图片
            if (href && href.startsWith('data:image')) {
                return `<img src="${href}" alt="${text || ''}" title="${title || ''}" class="max-w-full h-auto rounded-lg shadow-lg my-4" style="max-height: 400px;" />`;
            }
            // 处理普通图片链接
            return `<img src="${href}" alt="${text || ''}" title="${title || ''}" class="max-w-full h-auto rounded-lg shadow-lg my-4" style="max-height: 400px;" />`;
        };
        
        // 自定义代码块渲染
        renderer.code = (code, language) => {
            const validLanguage = language && hljs.getLanguage(language) ? language : 'plaintext';
            const highlighted = hljs.highlight(code, { language: validLanguage }).value;
            return `
                <div class="code-block-wrapper relative my-4">
                    <div class="code-header bg-gray-700 px-4 py-2 rounded-t-lg flex justify-between items-center">
                        <span class="text-gray-300 text-sm">${validLanguage}</span>
                        <button class="copy-code-btn bg-gray-600 hover:bg-gray-500 text-white px-3 py-1 rounded text-xs transition-colors" onclick="navigator.clipboard.writeText(this.closest('.code-block-wrapper').querySelector('code').textContent)">复制</button>
                    </div>
                    <pre class="bg-gray-800 p-4 rounded-b-lg overflow-x-auto"><code class="language-${validLanguage}">${highlighted}</code></pre>
                </div>
            `;
        };
        
        // 自定义表格渲染
        renderer.table = (header, body) => {
            return `
                <div class="overflow-x-auto my-4">
                    <table class="min-w-full border-collapse border border-gray-600">
                        <thead class="bg-gray-700">${header}</thead>
                        <tbody class="bg-gray-800">${body}</tbody>
                    </table>
                </div>
            `;
        };
        
        renderer.tablerow = (content) => {
            return `<tr class="border-b border-gray-600">${content}</tr>`;
        };
        
        renderer.tablecell = (content, flags) => {
            const tag = flags.header ? 'th' : 'td';
            const className = flags.header ? 'px-4 py-2 text-left font-semibold text-gray-200' : 'px-4 py-2 text-gray-300';
            return `<${tag} class="${className} border border-gray-600">${content}</${tag}>`;
        };
        
        // 自定义标题渲染
        renderer.heading = (text, level) => {
            const sizes = {
                1: 'text-2xl',
                2: 'text-xl',
                3: 'text-lg',
                4: 'text-base',
                5: 'text-sm',
                6: 'text-xs'
            };
            return `<h${level} class="${sizes[level]} font-bold mt-6 mb-3 text-white">${text}</h${level}>`;
        };
        
        // 自定义段落渲染
        renderer.paragraph = (text) => {
            return `<p class="my-3 text-gray-300 leading-relaxed">${text}</p>`;
        };
        
        // 自定义列表渲染
        renderer.list = (body, ordered) => {
            const tag = ordered ? 'ol' : 'ul';
            const className = ordered ? 'list-decimal' : 'list-disc';
            return `<${tag} class="${className} ml-6 my-3 text-gray-300">${body}</${tag}>`;
        };
        
        renderer.listitem = (text) => {
            return `<li class="my-1">${text}</li>`;
        };
        
        // 自定义链接渲染
        renderer.link = (href, title, text) => {
            return `<a href="${href}" title="${title || ''}" target="_blank" rel="noopener noreferrer" class="text-blue-400 hover:text-blue-300 underline">${text}</a>`;
        };
        
        // 自定义强调渲染
        renderer.strong = (text) => {
            return `<strong class="font-bold text-white">${text}</strong>`;
        };
        
        renderer.em = (text) => {
            return `<em class="italic text-gray-300">${text}</em>`;
        };
        
        // 使用自定义渲染器解析 Markdown
        try {
            const html = marked.parse(content, { renderer });
            return html;
        } catch (e) {
            console.error('Markdown parsing error:', e);
            // 如果解析失败，返回原始内容
            return `<pre class="text-gray-300">${content}</pre>`;
        }
    }
    
    showTypingIndicator() {
        const chatContainer = document.getElementById('chatContainer');
        const indicator = document.createElement('div');
        indicator.id = 'typingIndicator';
        indicator.className = 'message assistant';
        indicator.innerHTML = `
            <div class="message-avatar">🤖</div>
            <div class="message-content bg-gray-800">
                <div class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
        chatContainer.appendChild(indicator);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    hideTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.remove();
        }
    }
    
    saveMessages() {
        if (this.currentSessionId) {
            localStorage.setItem(`messages_${this.currentSessionId}`, JSON.stringify(this.messages));
            
            const firstUserMessage = this.messages.find(m => m.role === 'user');
            if (firstUserMessage) {
                const sessions = JSON.parse(localStorage.getItem('sessions') || '[]');
                const sessionIndex = sessions.findIndex(s => s.id === this.currentSessionId);
                if (sessionIndex !== -1) {
                    const title = firstUserMessage.content.slice(0, 30) + (firstUserMessage.content.length > 30 ? '...' : '');
                    sessions[sessionIndex].title = title;
                    localStorage.setItem('sessions', JSON.stringify(sessions));
                    this.renderSessions();
                }
            }
        }
    }
}

window.ChatManager = ChatManager;

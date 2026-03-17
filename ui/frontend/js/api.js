const API = {
    baseUrl: '',
    model: '',
    apiKey: '',
    
    init() {
        const stored = localStorage.getItem('apiBaseUrl');
        this.baseUrl = stored && stored.startsWith('http') ? stored : 'http://localhost:8001';
        this.model = localStorage.getItem('modelName') || 'kimi-k2-0905-preview';
        this.apiKey = localStorage.getItem('apiKey') || '';
    },
    
    saveConfig(baseUrl, apiKey, model) {
        this.baseUrl = baseUrl;
        this.apiKey = apiKey;
        this.model = model;
        localStorage.setItem('apiBaseUrl', baseUrl);
        localStorage.setItem('apiKey', apiKey);
        localStorage.setItem('modelName', model);
    },
    
    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const headers = {
            'Content-Type': 'application/json',
            ...options.headers
        };
        
        if (this.apiKey) {
            headers['Authorization'] = `Bearer ${this.apiKey}`;
        }
        
        try {
            const response = await fetch(url, {
                ...options,
                headers
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return response;
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    },
    
    async createSession() {
        const response = await this.request('/sessions', {
            method: 'POST'
        });
        return response.json();
    },
    
    async getSessions() {
        const response = await this.request('/sessions');
        return response.json();
    },
    
    async deleteSession(sessionId) {
        await this.request(`/sessions/${sessionId}`, {
            method: 'DELETE'
        });
    },
    
    async sendMessage(sessionId, message) {
        const response = await this.request('/chat/stream', {
            method: 'POST',
            body: JSON.stringify({
                session_id: sessionId,
                message: message
            })
        });
        return response;
    },
    
    async loadData(sessionId, file, targetColumn) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('target_column', targetColumn);
        formData.append('session_id', sessionId);
        
        const response = await fetch(`${this.baseUrl}/data/load`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${this.apiKey}`
            },
            body: formData
        });
        
        return response.json();
    }
};

window.API = API;

class App {
    constructor() {
        this.chatManager = null;
        this.workflowManager = null;
    }
    
    async init() {
        API.init();
        
        this.workflowManager = new WorkflowManager();
        this.chatManager = new ChatManager();
        
        window.workflowManager = this.workflowManager;
        
        await this.chatManager.init();
        
        this.bindDataUpload();
        
        this.chatManager.renderSessions();
    }
    
    bindDataUpload() {
        const dataFile = document.getElementById('dataFile');
        const targetColumn = document.getElementById('targetColumn');
        const loadDataBtn = document.getElementById('loadDataBtn');
        
        dataFile.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            
            try {
                const df = await this.parseCSV(file);
                const columns = df.columns;
                
                targetColumn.innerHTML = '<option value="">选择目标列</option>' +
                    columns.map(col => `<option value="${col}">${col}</option>`).join('');
            } catch (error) {
                console.error('Failed to read file:', error);
                alert('读取文件失败');
            }
        });
        
        loadDataBtn.addEventListener('click', async () => {
            const file = dataFile.files[0];
            const target = targetColumn.value;
            
            if (!file) {
                alert('请选择数据文件');
                return;
            }
            
            if (!target) {
                alert('请选择目标列');
                return;
            }
            
            if (!this.chatManager.currentSessionId) {
                alert('请先创建会话');
                return;
            }
            
            await this.loadData(file, target);
        });
    }
    
    async parseCSV(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const text = e.target.result;
                    const lines = text.split('\n').filter(line => line.trim());
                    const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
                    
                    const data = {};
                    headers.forEach(h => data[h] = []);
                    
                    for (let i = 1; i < lines.length; i++) {
                        const values = this.parseCSVLine(lines[i]);
                        headers.forEach((h, idx) => {
                            if (values[idx] !== undefined) {
                                const val = values[idx].trim().replace(/^"|"$/g, '');
                                data[h].push(isNaN(val) ? val : parseFloat(val));
                            }
                        });
                    }
                    
                    resolve({ columns: headers, data });
                } catch (error) {
                    reject(error);
                }
            };
            reader.onerror = reject;
            reader.readAsText(file);
        });
    }
    
    parseCSVLine(line) {
        const result = [];
        let current = '';
        let inQuotes = false;
        
        for (let i = 0; i < line.length; i++) {
            const char = line[i];
            
            if (char === '"') {
                inQuotes = !inQuotes;
            } else if (char === ',' && !inQuotes) {
                result.push(current);
                current = '';
            } else {
                current += char;
            }
        }
        result.push(current);
        
        return result;
    }
    
    async loadData(file, targetColumn) {
        try {
            const chatContainer = document.getElementById('chatContainer');
            
            chatContainer.innerHTML += `
                <div class="message assistant">
                    <div class="message-avatar">🤖</div>
                    <div class="message-content bg-gray-800">
                        <div class="flex items-center gap-2">
                            <span class="loading-spinner"></span>
                            <span>正在加载数据...</span>
                        </div>
                    </div>
                </div>
            `;
            chatContainer.scrollTop = chatContainer.scrollHeight;
            
            const formData = new FormData();
            formData.append('file', file);
            formData.append('target_column', targetColumn);
            formData.append('session_id', this.chatManager.currentSessionId);
            
            const response = await fetch(`${API.baseUrl}/data/upload`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${API.apiKey}`
                },
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`上传失败: ${response.status}`);
            }
            
            const result = await response.json();
            console.log('Upload result:', result);
            
            if (!result.success) {
                throw new Error(result.detail || '加载失败');
            }
            
            const profile = result.profile;
            console.log('Profile:', profile);
            console.log('Shape:', profile.shape);
            console.log('Shape type:', typeof profile.shape);
            
            this.workflowManager.clearWorkflow();
            this.workflowManager.addStep('数据加载', 'completed');
            this.workflowManager.addStep('数据探索', 'running');
            
            // 确保数据正确解析
            const shape = Array.isArray(profile.shape) ? profile.shape : [0, 0];
            const numericCount = Array.isArray(profile.numeric_columns) ? profile.numeric_columns.length : 0;
            const categoricalCount = Array.isArray(profile.categorical_columns) ? profile.categorical_columns.length : 0;
            
            const profileInfo = `**数据加载完成！**

- 数据形状: ${shape[0] || 0} 行 × ${shape[1] || 0} 列
- 数值特征: ${numericCount} 个
- 类别特征: ${categoricalCount} 个
- 目标列: ${profile.target_column || '未指定'}

请选择下一步操作：
- "分析数据" - 进行数据分析（推荐）
- "清洗数据" - 进行数据清洗
- "特征工程" - 创建新特征
- "训练模型" - 训练模型`;
            
            this.chatManager.addMessage({
                role: 'assistant',
                content: profileInfo
            });
            
            this.workflowManager.completeStep('数据探索', `数值: ${profile.numeric_columns.length}, 类别: ${profile.categorical_columns.length}`);
            
        } catch (error) {
            console.error('Load data error:', error);
            this.chatManager.addMessage({
                role: 'assistant',
                content: `加载数据失败: ${error.message}`
            });
            this.workflowManager.failStep('数据加载', error.message);
        }
    }
}

const app = new App();
document.addEventListener('DOMContentLoaded', () => app.init());

window.app = app;

class WorkflowManager {
    constructor() {
        this.steps = [];
    }
    
    updateWorkflow(steps) {
        this.steps = steps;
        this.render();
    }
    
    addStep(name, status = 'running') {
        const existingIndex = this.steps.findIndex(s => s.name === name);
        
        if (existingIndex !== -1) {
            this.steps[existingIndex].status = status;
            if (status === 'completed') {
                this.steps[existingIndex].time = this.calculateTime();
            }
        } else {
            this.steps.push({
                id: `step_${Date.now()}`,
                name: name,
                status: status,
                time: status === 'completed' ? this.calculateTime() : ''
            });
        }
        
        this.render();
    }
    
    completeStep(name, detail = '') {
        const step = this.steps.find(s => s.name === name);
        if (step) {
            step.status = 'completed';
            step.time = this.calculateTime();
            step.detail = detail;
        }
        this.render();
    }
    
    failStep(name, error = '') {
        const step = this.steps.find(s => s.name === name);
        if (step) {
            step.status = 'failed';
            step.error = error;
        }
        this.render();
    }
    
    calculateTime() {
        if (this.startTime) {
            const elapsed = Date.now() - this.startTime;
            if (elapsed < 1000) return `${elapsed}ms`;
            return `${(elapsed / 1000).toFixed(1)}s`;
        }
        return '0s';
    }
    
    clearWorkflow() {
        this.steps = [];
        this.startTime = Date.now();
        this.render();
    }
    
    render() {
        const container = document.getElementById('workflowNodes');
        
        if (this.steps.length === 0) {
            container.innerHTML = '<div class="text-gray-500 text-center p-4 text-sm">暂无工作流</div>';
            return;
        }
        
        container.innerHTML = this.steps.map((step, index) => {
            const iconMap = {
                pending: '⏳',
                running: '🔄',
                completed: '✓',
                failed: '✗'
            };
            
            return `
                <div class="workflow-node ${step.status}" data-index="${index}">
                    <div class="workflow-node-header">
                        <div class="workflow-node-icon">${iconMap[step.status]}</div>
                        <div class="workflow-node-title">${step.name}</div>
                        <div class="workflow-node-time">${step.time || ''}</div>
                    </div>
                    ${step.detail || step.error ? `
                        <div class="workflow-node-detail">
                            ${step.error ? `<span class="text-red-400">${step.error}</span>` : step.detail}
                        </div>
                    ` : ''}
                </div>
            `;
        }).join('');
        
        container.querySelectorAll('.workflow-node').forEach(node => {
            node.addEventListener('click', () => {
                node.classList.toggle('expanded');
            });
        });
    }
}

window.WorkflowManager = WorkflowManager;

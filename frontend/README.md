# AutoML ReAct Frontend

交互式建模平台前端界面。

## 特点

- **单页应用**: 独立的 HTML 文件，无需构建工具
- **Markdown 渲染**: 使用 marked.js 渲染 LLM 生成的内容
- **代码高亮**: 使用 highlight.js 高亮代码块
- **响应式设计**: 适配桌面和移动设备
- **流式输出**: 支持 SSE 流式响应展示

## 文件说明

- `index.html` - 完整的单页应用（包含 CSS 和 JavaScript）

## 使用方式

### 方式 1: 直接用浏览器打开

```bash
open index.html
```

### 方式 2: 使用 HTTP 服务器

```bash
python -m http.server 8080
```

访问 http://localhost:8080

## 功能

1. **工作流可视化**: 显示当前建模进度
2. **方案展示**: Markdown 格式渲染清洗/特征/建模方案
3. **代码高亮**: Python 代码语法高亮
4. **用户确认**: 确认/修改/跳过操作
5. **Skills 参考**: 显示当前步骤参考的 Skills
6. **资产下载**: 下载生成的数据和模型文件
7. **聊天界面**: 与 Agent 进行对话

## 依赖

- marked.js (CDN) - Markdown 渲染
- highlight.js (CDN) - 代码高亮

## 配置

编辑 `index.html` 中的 API 地址：

```javascript
const API_BASE = 'http://localhost:8000';  // 后端 API 地址
```

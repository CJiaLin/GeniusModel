# 启动说明

## 方式一：使用 Python 内置 HTTP 服务器（推荐开发调试）

在项目根目录运行：

```bash
cd /home/chenjl/AutoMLByLLM
python -m http.server 8080 --directory ui/frontend
```

然后访问：http://localhost:8080

## 方式二：启动后端 API 服务

```bash
cd /home/chenjl/AutoMLByLLM
python -m api.main
```

后端会在 http://localhost:8000 启动。

## 方式三：同时启动前端和后端

终端1（后端）：
```bash
cd /home/chenjl/AutoMLByLLM
python -m api.main
```

终端2（前端）：
```bash
cd /home/chenjl/AutoMLByLLM
python -m http.server 8080 --directory ui/frontend
```

访问：http://localhost:8080

## 配置说明

1. 在浏览器中打开前端页面
2. 在顶部输入 API Key（必须）
3. 点击"保存配置"
4. 点击左侧"新建会话"开始对话

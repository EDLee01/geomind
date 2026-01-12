# GeoMind - 部署指南

## 🚀 快速部署到 Zeabur

### 方法 1：网页部署（推荐新手）

1. **推送到 GitHub**
```bash
cd geomind
git init
git add .
git commit -m "GeoMind multi-agent system"
git remote add origin https://github.com/YOUR_USERNAME/geomind.git
git push -u origin main
```

2. **Zeabur 网页部署**
   - 访问 [dash.zeabur.com](https://dash.zeabur.com)
   - New Project → Deploy from GitHub
   - 选择你的仓库

3. **配置环境变量**（在 Variables 页面）
```
QDRANT_URL=https://xxx.qdrant.io:6333
QDRANT_API_KEY=your-key
DEEPSEEK_API_KEY=sk-xxx
```

4. **配置端口**（在 Networking 页面）
   - 添加端口 8700
   - 生成域名

---

### 方法 2：Zeabur CLI 部署

**安装 CLI**
```bash
npm install -g zeabur
# 或
brew install zeabur/tap/cli
```

**登录**
```bash
zeabur auth login
```

**部署**
```bash
cd geomind
zeabur deploy
```

**设置环境变量**
```bash
# 方法 A：交互式设置
zeabur variable set QDRANT_URL
zeabur variable set QDRANT_API_KEY
zeabur variable set DEEPSEEK_API_KEY

# 方法 B：直接设置
zeabur variable set QDRANT_URL=https://xxx.qdrant.io:6333
zeabur variable set QDRANT_API_KEY=your-key
zeabur variable set DEEPSEEK_API_KEY=sk-xxx
```

**查看日志**
```bash
zeabur log
```

**重新部署**
```bash
zeabur redeploy
```

---

## 📋 必需的环境变量

| 变量 | 说明 | 示例 |
|------|------|------|
| `QDRANT_URL` | Qdrant Cloud 地址 | `https://xxx.qdrant.io:6333` |
| `QDRANT_API_KEY` | Qdrant API Key | `your-api-key` |
| `DEEPSEEK_API_KEY` | DeepSeek API Key | `sk-xxx` |

---

## 📁 项目结构

```
geomind/
├── start.py              # 启动入口
├── config.py             # 配置管理
├── shared_tools.py       # LLM + Qdrant + Embedding
├── requirements.txt      # Python 依赖
├── Dockerfile            # Docker 构建
├── .env.example          # 环境变量模板
├── .gitignore            # Git 忽略
└── agents/
    ├── __init__.py
    ├── planner_agent.py  # 规划 Agent
    ├── research_agent.py # 检索 Agent
    ├── critic_agent.py   # 审核 Agent
    └── writer_agent.py   # 撰写 Agent
```

---

## ⚠️ 重要提醒

### Embedding 模型
- **不要**把模型上传到 Git（太大）
- FastEmbed 会自动下载 `BAAI/bge-base-en-v1.5`
- 首次启动需要 1-2 分钟下载模型

### 内存配置
- 推荐 **1GB+** 内存
- 如果 Zeabur 默认配置不够，在设置里调大

### OpenAgents 网络
- `start.py` 会自动启动本地 OpenAgents Network
- 网络运行在容器内部的 8700 端口
- 4 个 Agent 连接到这个内部网络

---

## 🔍 常见问题

**Q: 部署后访问不了？**
- 检查 Networking 是否暴露了 8700 端口
- 查看日志是否有错误

**Q: Agent 连接失败？**
- 确认 OpenAgents Network 已启动
- 检查日志中是否有 "Network started" 字样

**Q: 搜索结果不对？**
- 确认 `QDRANT_COLLECTION` 名称正确
- 确保使用相同的 Embedding 模型

**Q: 首次启动很慢？**
- 正常！FastEmbed 在下载 400MB 的模型
- 后续启动会快很多（模型已缓存）

---

## 📞 调试命令

```bash
# 查看实时日志
zeabur log -f

# 查看环境变量
zeabur variable list

# 重启服务
zeabur restart

# 查看服务状态
zeabur status
```

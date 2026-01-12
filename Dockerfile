# GeoMind Dockerfile for Zeabur
# 包含 OpenAgents Network + 4 Agents

FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建 workspace 目录
RUN mkdir -p /app/workspace

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV NETWORK_HOST=localhost
ENV NETWORK_PORT=8700

# 暴露端口（OpenAgents Studio）
EXPOSE 8700

# 启动命令
CMD ["python", "start.py"]

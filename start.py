"""
GeoMind - Zeabur 部署启动脚本
同时启动 OpenAgents Network + 4 个 Agent
"""
import asyncio
import subprocess
import sys
import os
import time
import signal

import nest_asyncio
nest_asyncio.apply()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()


def start_network():
    """启动 OpenAgents Network"""
    print("🌐 Starting OpenAgents Network...")
    
    # 创建 workspace 目录
    workspace_dir = "/app/workspace"
    os.makedirs(workspace_dir, exist_ok=True)
    
    # 启动 network（后台运行）
    process = subprocess.Popen(
        ["openagents", "network", "start", workspace_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    
    # 等待网络启动
    time.sleep(5)
    
    if process.poll() is None:
        print("✅ OpenAgents Network started on port 8700")
        return process
    else:
        print("❌ Failed to start OpenAgents Network")
        stdout, _ = process.communicate()
        print(stdout)
        return None


async def start_agents():
    """启动所有 Agent"""
    from config import NETWORK_HOST, NETWORK_PORT, NETWORK_ID, AGENT_IDS
    from agents import PlannerAgent, ResearchAgent, CriticAgent, WriterAgent
    
    print("📦 Creating agents...")
    
    agents = [
        PlannerAgent(),
        ResearchAgent(),
        CriticAgent(),
        WriterAgent(),
    ]
    
    print(f"🔗 Connecting to OpenAgents ({NETWORK_HOST}:{NETWORK_PORT})...")
    
    for agent in agents:
        agent.start(
            network_host=NETWORK_HOST,
            network_port=NETWORK_PORT,
            network_id=NETWORK_ID
        )
        print(f"  ✓ {agent.default_agent_id}")
        await asyncio.sleep(0.5)
    
    print("✅ All agents connected!")
    return agents


async def main():
    print("=" * 60)
    print("  🌍 GeoMind - Multi-Agent Literature Review System")
    print("  Zeabur Cloud Deployment")
    print("=" * 60)
    print()
    
    # 检查环境变量
    from config import QDRANT_URL, QDRANT_API_KEY, DEEPSEEK_API_KEY
    
    missing = []
    if not QDRANT_API_KEY:
        missing.append("QDRANT_API_KEY")
    if not DEEPSEEK_API_KEY:
        missing.append("DEEPSEEK_API_KEY")
    
    if missing:
        print(f"❌ Missing environment variables: {missing}")
        print("Please set them in Zeabur dashboard")
        sys.exit(1)
    
    print(f"✅ Qdrant: {QDRANT_URL[:40]}...")
    print(f"✅ DeepSeek API: configured")
    print()
    
    # 启动 OpenAgents Network
    network_process = start_network()
    if not network_process:
        sys.exit(1)
    
    # 启动 Agents
    try:
        agents = await start_agents()
        
        print()
        print("=" * 60)
        print("  🚀 GeoMind is LIVE!")
        print("=" * 60)
        print()
        print(f"  📱 Access: https://your-app.zeabur.app:8700/")
        print(f"  💬 DM @geomind-planner to start")
        print()
        
        # Keep running
        while True:
            await asyncio.sleep(10)
            
    except KeyboardInterrupt:
        print("\n⏹️ Shutting down...")
    finally:
        # 停止 agents
        for agent in agents:
            try:
                agent.stop()
            except:
                pass
        
        # 停止 network
        if network_process:
            network_process.terminate()
            network_process.wait()
        
        print("✅ Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())

"""
GeoMind - Zeabur 部署启动脚本 v2
"""
import asyncio
import subprocess
import sys
import os
import time
import threading
import socket

import nest_asyncio
nest_asyncio.apply()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()


def print_output(pipe, prefix):
    """实时打印进程输出"""
    for line in iter(pipe.readline, ''):
        if line:
            print(f"{prefix} {line.strip()}")


def start_network():
    """启动 OpenAgents Network"""
    print("🌐 Starting OpenAgents Network...")
    
    workspace_dir = "/app/workspace"
    config_path = os.path.join(workspace_dir, "network.yaml")
    
    if not os.path.exists(config_path):
        print(f"❌ network.yaml not found at {config_path}")
        return None
    
    print(f"✅ Found network.yaml")
    
    process = subprocess.Popen(
        ["openagents", "network", "start", workspace_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    
    # 打印 network 输出
    stdout_thread = threading.Thread(
        target=print_output, 
        args=(process.stdout, "[Network]"),
        daemon=True
    )
    stderr_thread = threading.Thread(
        target=print_output,
        args=(process.stderr, "[Network ERR]"),
        daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()
    
    # 等待并检查端口
    for i in range(30):
        time.sleep(2)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', 8700))
        sock.close()
        
        if result == 0:
            print(f"✅ Port 8700 is listening! (after {(i+1)*2}s)")
            return process
        else:
            print(f"⏳ Waiting for port 8700... ({(i+1)*2}s)")
    
    print("❌ Port 8700 never started listening")
    return process


async def start_agents_with_retry(max_retries=5, retry_delay=10):
    from config import NETWORK_HOST, NETWORK_PORT, NETWORK_ID
    from agents import PlannerAgent, ResearchAgent, CriticAgent, WriterAgent
    
    print("📦 Creating agents...")
    
    agents_config = [
        ("Planner", PlannerAgent),
        ("Research", ResearchAgent),
        ("Critic", CriticAgent),
        ("Writer", WriterAgent),
    ]
    
    connected_agents = []
    
    for name, AgentClass in agents_config:
        for attempt in range(max_retries):
            try:
                print(f"🔗 Connecting {name}Agent... (attempt {attempt + 1}/{max_retries})")
                
                agent = AgentClass()
                agent.start(
                    network_host=NETWORK_HOST,
                    network_port=NETWORK_PORT,
                    network_id=NETWORK_ID
                )
                
                print(f"  ✓ {name}Agent connected")
                connected_agents.append(agent)
                await asyncio.sleep(2)
                break
                
            except Exception as e:
                print(f"  ⚠️ {name}Agent failed: {e}")
                if attempt < max_retries - 1:
                    print(f"  ⏳ Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
    
    if len(connected_agents) == 0:
        raise Exception("No agents could connect to the network")
    
    return connected_agents


async def main():
    print("=" * 60)
    print("  🌍 GeoMind v2 - Debug Mode")
    print("=" * 60)
    
    from config import QDRANT_URL, QDRANT_API_KEY, DEEPSEEK_API_KEY
    
    if not QDRANT_API_KEY or not DEEPSEEK_API_KEY:
        print("❌ Missing API keys")
        sys.exit(1)
    
    print(f"✅ Config OK")
    
    network_process = start_network()
    if not network_process:
        sys.exit(1)
    
    agents = []
    
    try:
        agents = await start_agents_with_retry()
        print("🚀 GeoMind is LIVE!")
        
        while True:
            await asyncio.sleep(10)
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        for agent in agents:
            try:
                agent.stop()
            except:
                pass
        if network_process:
            network_process.terminate()


if __name__ == "__main__":
    asyncio.run(main())

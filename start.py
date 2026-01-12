"""
GeoMind - Zeabur 部署启动脚本
同时启动 OpenAgents Network + 4 个 Agent
"""
import asyncio
import subprocess
import sys
import os
import time

import nest_asyncio
nest_asyncio.apply()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()


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
        stderr=subprocess.STDOUT,
        text=True,
    )
    
    print("⏳ Waiting for network to initialize (15s)...")
    time.sleep(15)
    
    if process.poll() is None:
        print("✅ OpenAgents Network started on port 8700")
        return process
    else:
        print("❌ Failed to start OpenAgents Network")
        stdout = process.stdout.read() if process.stdout else ""
        print(stdout)
        return None


async def start_agents_with_retry(max_retries=5, retry_delay=5):
    """启动所有 Agent（带重试机制）"""
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
                await asyncio.sleep(1)
                break
                
            except Exception as e:
                print(f"  ⚠️ {name}Agent failed: {e}")
                if attempt < max_retries - 1:
                    print(f"  ⏳ Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                else:
                    print(f"  ❌ {name}Agent failed after {max_retries} attempts")
    
    if len(connected_agents) == 0:
        raise Exception("No agents could connect to the network")
    
    print(f"✅ {len(connected_agents)}/{len(agents_config)} agents connected!")
    return connected_agents


async def main():
    print("=" * 60)
    print("  🌍 GeoMind - Multi-Agent Literature Review System")
    print("  Zeabur Cloud Deployment")
    print("=" * 60)
    print()
    
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
    
    network_process = start_network()
    if not network_process:
        print("❌ Cannot start without network, exiting...")
        sys.exit(1)
    
    print("⏳ Additional wait for network stability (10s)...")
    await asyncio.sleep(10)
    
    agents = []
    
    try:
        agents = await start_agents_with_retry(max_retries=5, retry_delay=5)
        
        print()
        print("=" * 60)
        print("  🚀 GeoMind is LIVE!")
        print("=" * 60)
        print()
        
        while True:
            await asyncio.sleep(10)
            
    except KeyboardInterrupt:
        print("\n⏹️ Shutting down...")
    except Exception as e:
        print(f"❌ Startup error: {e}")
    finally:
        for agent in agents:
            try:
                agent.stop()
            except:
                pass
        
        if network_process:
            network_process.terminate()
            try:
                network_process.wait(timeout=5)
            except:
                network_process.kill()
        
        print("✅ Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())

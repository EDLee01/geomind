"""
Research Agent - 文献检索
接收 Planner 的任务，搜索 Qdrant，返回结果
"""
import json
import os
from typing import Dict, List

from openagents.agents.worker_agent import WorkerAgent

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AGENT_IDS
from shared_tools import get_vector_store, extract_message


class ResearchAgent(WorkerAgent):
    """
    Research Agent - 文献检索
    从 Qdrant 向量库检索相关论文
    """
    
    default_agent_id = AGENT_IDS["research"]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vector_store = get_vector_store()
        print(f"--- ResearchAgent initialized [{self.default_agent_id}] ---")
    
    async def on_direct(self, ctx):
        """Handle direct messages from Planner"""
        text, sender = extract_message(ctx)
        
        if not text:
            return
        
        try:
            msg = json.loads(text)
            if msg.get('type') == 'search_task':
                await self.execute_search(msg)
        except json.JSONDecodeError:
            pass
    
    async def execute_search(self, task: Dict):
        """执行文献检索"""
        ws = self.workspace()
        
        session_id = task.get('session_id', '')
        query = task.get('query', '')
        keywords = task.get('keywords', [])
        year_from = task.get('year_from', 2021)
        year_to = task.get('year_to', 2025)
        
        print(f"[Research] 开始检索: {query}")
        print(f"[Research] 关键词: {keywords}")
        
        try:
            # 执行多关键词检索
            all_papers = []
            search_count = 0
            
            for keyword in keywords[:5]:
                print(f"[Research] 搜索: {keyword}")
                papers = await self.vector_store.search(
                    query=keyword,
                    n_results=25,
                    year_from=year_from,
                    year_to=year_to
                )
                all_papers.extend(papers)
                search_count += 1
                print(f"[Research] 找到 {len(papers)} 篇")
            
            # 去重
            seen_titles = set()
            unique_papers = []
            for p in all_papers:
                title = p.get('title', '')
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    unique_papers.append(p)
            
            print(f"[Research] 去重后: {len(unique_papers)} 篇")
            
            # 按相似度排序
            unique_papers.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            
            # 返回结果给 Planner
            result = json.dumps({
                "type": "search_complete",
                "session_id": session_id,
                "papers": unique_papers[:60],  # 最多60篇给Critic筛选
                "search_count": search_count,
                "total_found": len(unique_papers),
            }, ensure_ascii=False)
            
            await ws.agent(AGENT_IDS['planner']).send(result)
            print(f"[Research] Result sent to Planner successfully")
            
        except Exception as e:
            print(f"[Research] ERROR: {e}")
            import traceback
            traceback.print_exc()
            
            # 发送空结果
            error_result = json.dumps({
                "type": "search_complete",
                "session_id": session_id,
                "papers": [],
                "search_count": 0,
                "total_found": 0,
                "error": str(e),
            }, ensure_ascii=False)
            await ws.agent(AGENT_IDS['planner']).send(error_result)

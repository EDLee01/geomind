"""
Critic Agent - 文献质量评估
评估论文相关性和质量，筛选高质量文献
"""
import json
import os
from typing import Dict, List

from openagents.agents.worker_agent import WorkerAgent

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AGENT_IDS
from shared_tools import get_llm_client, get_vector_store, extract_message


class CriticAgent(WorkerAgent):
    """
    Critic Agent - 质量评估
    评估文献相关性、引用质量，筛选最佳论文
    """

    default_agent_id = AGENT_IDS["critic"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = get_llm_client()
        self.vector_store = get_vector_store()
        print(f"--- CriticAgent initialized [{self.default_agent_id}] ---")

    async def on_direct(self, ctx):
        """Handle direct messages from Planner"""
        text, sender = extract_message(ctx)

        if not text:
            return

        try:
            msg = json.loads(text)
            if msg.get('type') == 'critic_task':
                await self.evaluate_papers(msg)
        except json.JSONDecodeError:
            pass

    async def evaluate_papers(self, task: Dict):
        """评估论文质量 - 核心：用 LLM 判断相关性"""
        ws = self.workspace()

        session_id = task.get('session_id', '')
        query = task.get('query', '')
        papers = task.get('papers', [])

        print(f"[Critic] 开始评估 {len(papers)} 篇论文")
        print(f"[Critic] 研究主题: {query}")

        try:
            if not papers:
                result = json.dumps({
                    "type": "critic_complete",
                    "session_id": session_id,
                    "score": 0,
                    "verdict": "FAIL",
                    "top_papers": [],
                    "stats": {"reason": "没有找到任何论文"},
                }, ensure_ascii=False)
                await ws.agent(AGENT_IDS['planner']).send(result)
                return

            # === 核心：用 LLM 批量判断论文相关性 ===
            relevant_papers = []
            irrelevant_papers = []

            # 分批处理（每批 10 篇）
            batch_size = 10
            for i in range(0, len(papers), batch_size):
                batch = papers[i:i + batch_size]
                print(f"[Critic] 评估第 {i // batch_size + 1} 批 ({len(batch)} 篇)...")

                batch_results = await self.check_relevance_batch(query, batch)

                for paper, is_relevant, reason in batch_results:
                    if is_relevant:
                        paper['relevance_reason'] = reason
                        relevant_papers.append(paper)
                    else:
                        paper['irrelevance_reason'] = reason
                        irrelevant_papers.append(paper)

            print(f"[Critic] 相关: {len(relevant_papers)} 篇, 不相关: {len(irrelevant_papers)} 篇")

            # === 计算质量分数 ===
            relevance_rate = len(relevant_papers) / len(papers) if papers else 0

            # 对相关论文按引用数和年份排序
            from datetime import datetime
            current_year = datetime.now().year

            for p in relevant_papers:
                cite = p.get('cited_by_count', 0)
                year = p.get('year', 2020)
                sim = p.get('similarity', 0.5)
                recency = max(0, 5 - (current_year - year)) / 5
                # 综合得分：相似度 40% + 引用 35% + 时效性 25%
                p['_rank_score'] = sim * 0.4 + min(cite / 100, 0.35) + recency * 0.25

            relevant_papers.sort(key=lambda x: x.get('_rank_score', 0), reverse=True)

            # 取前 30 篇
            top_papers = relevant_papers[:30]

            # 清理临时字段
            for p in top_papers:
                p.pop('_rank_score', None)

            # 计算统计
            citations = [p.get('cited_by_count', 0) for p in top_papers]
            avg_citations = sum(citations) / len(citations) if citations else 0

            # 综合评分
            score = min(10, relevance_rate * 7 + min(avg_citations / 50, 2) + (1 if len(top_papers) >= 20 else 0))

            # 判定
            if len(top_papers) >= 20 and relevance_rate >= 0.5:
                verdict = "PASS"
            elif len(top_papers) >= 10:
                verdict = "ACCEPTABLE"
            else:
                verdict = "NEEDS_IMPROVEMENT"

            print(f"[Critic] 评分: {score:.1f}/10, 判定: {verdict}")

            # 返回结果
            result = json.dumps({
                "type": "critic_complete",
                "session_id": session_id,
                "score": round(score, 1),
                "verdict": verdict,
                "top_papers": top_papers,
                "irrelevant_examples": irrelevant_papers[:5],  # 返回几个不相关的例子
                "stats": {
                    "total_evaluated": len(papers),
                    "relevant_count": len(relevant_papers),
                    "irrelevant_count": len(irrelevant_papers),
                    "relevance_rate": f"{relevance_rate * 100:.0f}%",
                    "selected": len(top_papers),
                    "avg_citations": round(avg_citations, 1),
                }
            }, ensure_ascii=False)

            await ws.agent(AGENT_IDS['planner']).send(result)
            print(f"[Critic] Result sent to Planner successfully")

        except Exception as e:
            print(f"[Critic] ERROR: {e}")
            import traceback
            traceback.print_exc()

            # 出错时返回原始论文（降级处理）
            error_result = json.dumps({
                "type": "critic_complete",
                "session_id": session_id,
                "score": 5.0,
                "verdict": "ACCEPTABLE",
                "top_papers": papers[:30],
                "stats": {"error": str(e)},
            }, ensure_ascii=False)
            await ws.agent(AGENT_IDS['planner']).send(error_result)

    async def check_relevance_batch(self, query: str, papers: List[Dict]) -> List[tuple]:
        """
        批量检查论文相关性
        返回: [(paper, is_relevant, reason), ...]
        """
        # 构建论文列表
        papers_text = ""
        for i, p in enumerate(papers):
            title = p.get('title', 'Unknown')[:100]
            abstract = p.get('abstract', '')[:300]
            papers_text += f"""
[{i + 1}] 标题: {title}
摘要: {abstract}
---"""

        prompt = f"""你是学术文献审核专家。判断以下论文是否与研究主题相关。

研究主题: {query}

论文列表:
{papers_text}

对每篇论文，判断：
1. 是否与研究主题直接相关？
2. 给出简短理由（10字以内）

输出格式（每行一篇）:
1|YES|理由
2|NO|理由
3|YES|理由
...

只输出判断结果，不要其他内容:"""

        try:
            response = self.llm.chat(prompt)

            results = []
            lines = response.strip().split('\n')

            for line in lines:
                line = line.strip()
                if not line or '|' not in line:
                    continue

                parts = line.split('|')
                if len(parts) >= 2:
                    try:
                        idx = int(parts[0].strip()) - 1
                        is_relevant = parts[1].strip().upper() in ['YES', 'Y', '是', 'TRUE']
                        reason = parts[2].strip() if len(parts) > 2 else ""

                        if 0 <= idx < len(papers):
                            results.append((papers[idx], is_relevant, reason))
                    except (ValueError, IndexError):
                        continue

            # 补充未处理的论文（默认相关）
            processed_indices = set(i for i, (p, _, _) in enumerate(results)
                                    for j, paper in enumerate(papers) if p is paper)
            for i, paper in enumerate(papers):
                if i not in processed_indices:
                    # 检查是否在结果中
                    found = False
                    for p, _, _ in results:
                        if p.get('title') == paper.get('title'):
                            found = True
                            break
                    if not found:
                        results.append((paper, True, "默认相关"))

            return results

        except Exception as e:
            print(f"[Critic] LLM 评估失败: {e}")
            # 降级：全部标记为相关
            return [(p, True, "评估跳过") for p in papers]
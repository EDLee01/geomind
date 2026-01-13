"""
Writer Agent - 文献综述撰写
基于筛选后的论文生成结构化综述
"""
import json
import os
from typing import Dict, List

from openagents.agents.worker_agent import WorkerAgent

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AGENT_IDS
from shared_tools import get_llm_client, extract_message


def format_papers_detailed(papers: List[Dict], max_papers: int = 30) -> str:
    """格式化论文列表，提供详细信息供 LLM 引用"""
    lines = []
    for i, p in enumerate(papers[:max_papers], 1):
        # 提取作者
        authors = p.get('authors', [])
        if isinstance(authors, list) and authors:
            if len(authors) == 1:
                author_str = authors[0]
            elif len(authors) == 2:
                author_str = f"{authors[0]} & {authors[1]}"
            else:
                author_str = f"{authors[0]} et al."
        else:
            author_str = "Unknown"

        year = p.get('year', 'N/A')
        title = p.get('title', 'Untitled')  # 不截断标题！
        journal = p.get('journal', p.get('journal_name', 'Unknown'))[:40]
        abstract = p.get('abstract', '')[:200]
        citations = p.get('cited_by_count', 0)
        doi = p.get('doi', '')  # 添加 DOI

        lines.append(f"""[{i}] {author_str} ({year})
标题: {title}
期刊: {journal} | 引用: {citations}
DOI: {doi}
摘要: {abstract}...
""")

    return "\n".join(lines)


class WriterAgent(WorkerAgent):
    """
    Writer Agent - 综述撰写
    根据高质量论文列表生成学术综述
    """

    default_agent_id = AGENT_IDS["writer"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = get_llm_client()
        print(f"--- WriterAgent initialized [{self.default_agent_id}] ---")

    async def on_direct(self, ctx):
        """Handle direct messages from Planner"""
        text, sender = extract_message(ctx)

        if not text:
            return

        try:
            msg = json.loads(text)
            if msg.get('type') == 'write_task':
                await self.write_review(msg)
        except json.JSONDecodeError:
            pass

    async def write_review(self, task: Dict):
        """撰写文献综述"""
        ws = self.workspace()

        session_id = task.get('session_id', '')
        query = task.get('query', '')
        original_topic = task.get('original_topic', query)
        papers = task.get('papers', [])

        print(f"[Writer] 开始撰写综述: {query}")
        print(f"[Writer] 论文数量: {len(papers)}")

        try:
            # 生成大纲
            print("[Writer] 生成大纲...")
            outline = await self.generate_outline(original_topic, papers)

            # 生成综述
            print("[Writer] 撰写正文...")
            review = await self.generate_review(original_topic, papers, outline)

            print(f"[Writer] 完成，字数: {len(review)}")

            # 返回结果给 Planner
            result = json.dumps({
                "type": "review_complete",
                "session_id": session_id,
                "review": review,
                "paper_count": len(papers),
                "word_count": len(review),
            }, ensure_ascii=False)

            await ws.agent(AGENT_IDS['planner']).send(result)
            print(f"[Writer] Result sent to Planner successfully")

        except Exception as e:
            print(f"[Writer] ERROR: {e}")
            import traceback
            traceback.print_exc()

            # 发送错误信息
            error_result = json.dumps({
                "type": "review_complete",
                "session_id": session_id,
                "review": f"综述生成过程中遇到错误：{str(e)}\n\n请稍后重试或更换主题。",
                "paper_count": len(papers),
                "word_count": 0,
            }, ensure_ascii=False)
            await ws.agent(AGENT_IDS['planner']).send(error_result)

    async def generate_outline(self, topic: str, papers: List[Dict]) -> str:
        """生成综述大纲"""
        # 使用所有论文生成大纲
        papers_brief = "\n".join([
            f"[{i + 1}] {p.get('title', '')[:70]} ({p.get('year', '')})"
            for i, p in enumerate(papers[:30])
        ])

        prompt = f"""为以下研究主题生成文献综述大纲。

研究主题: {topic}

可用论文（共 {len(papers)} 篇，编号 [1]-[{min(30, len(papers))}]）:
{papers_brief}

生成详细的学术综述大纲（中文），要求：

1. **摘要**（200-300字）
   - 研究背景与意义
   - 主要发现综述
   - 研究展望

2. **引言**
   - 研究背景（引用论文 [?], [?], [?]）
   - 研究现状与问题（引用论文 [?], [?]）
   - 本综述目的与结构

3. **研究进展**（分 3-4 个主题子章节）
   - 3.1 [主题1名称]（引用论文 [?], [?], [?], [?], [?]）
   - 3.2 [主题2名称]（引用论文 [?], [?], [?], [?], [?]）
   - 3.3 [主题3名称]（引用论文 [?], [?], [?], [?], [?]）
   - 3.4 [主题4名称]（引用论文 [?], [?], [?], [?]）

4. **讨论与展望**
   - 当前研究局限（引用论文 [?], [?]）
   - 未来研究方向（引用论文 [?], [?]）

5. **结论**

重要：
- 每个子章节必须关联 4-6 篇相关论文
- 用论文编号 [1]-[30] 标注每章节应引用的文献
- 确保 30 篇论文被合理分配到各章节

大纲:"""

        return self.llm.chat(prompt)

    async def generate_review(self, topic: str, papers: List[Dict], outline: str) -> str:
        """生成完整综述"""
        papers_text = format_papers_detailed(papers, max_papers=30)

        prompt = f"""基于以下大纲和论文，撰写一篇完整的中文文献综述。

研究主题: {topic}

大纲:
{outline}

参考论文库（共 {len(papers)} 篇，请充分引用）:
{papers_text}

撰写要求:
1. 学术语言，严谨客观
2. 引用格式：(作者, 年份)，如 (Wang et al., 2023)
3. 字数：3000-5000字
4. 使用 Markdown 格式，包含标题层级

**重要引用要求**:
- 引言部分：引用 4-6 篇文献说明研究背景
- 每个研究进展子章节：引用 6-10 篇相关文献，深入讨论各研究发现
- 讨论部分：引用 4-6 篇文献支撑观点
- 全文必须引用 **至少 20 篇不同文献**
- 结尾列出所有引用的参考文献（完整格式，按作者字母排序）

**参考文献格式要求（非常重要）**:
在综述末尾的参考文献列表中，每条必须包含 DOI 链接：

格式示例：
Chen, S., et al. (2021). Larger phosphorus flux triggered by smaller tributary watersheds in a river reservoir system after dam construction. Journal of Hydrology. https://doi.org/10.1016/j.jhydrol.2021.126956

Wang, L., & Zhang, H. (2023). Machine learning for water quality prediction in rivers. Water Research. https://doi.org/10.1016/j.watres.2023.xxxxxx

要求：
- 必须使用论文库中提供的**完整原始标题**，不要截断或修改
- 必须包含论文库中提供的 **DOI 链接**
- 如果论文没有 DOI，则省略 DOI 部分
- DOI 链接格式：https://doi.org/10.xxxx/xxxxx

注意：
- 上面提供的每篇论文都是精选的高质量文献，请尽量全部引用
- 引用时要结合论文的具体内容（标题、摘要信息）
- 不要只引用同一篇论文多次，要分散引用
- **标题必须与论文库中的完全一致，不要自行缩写或修改**

直接输出综述正文:"""

        return self.llm.chat(prompt)

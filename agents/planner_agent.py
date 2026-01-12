"""
Planner Agent - 研究规划与任务协调
私信模式：用户只和 Planner 对话，看到所有 agent 协作过程
"""
import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List

from openagents.agents.worker_agent import WorkerAgent

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AGENT_IDS
from shared_tools import (
    get_llm_client, get_chinese_processor, extract_message, SessionManager
)


class PlannerAgent(WorkerAgent):
    """
    Planner Agent - 主控协调
    用户只和 Planner 对话，Planner 展示所有 agent 的协作过程
    """

    default_agent_id = AGENT_IDS["planner"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = get_llm_client()
        self.chinese_processor = get_chinese_processor()
        self.sessions: Dict[str, SessionManager] = {}
        self.user_map: Dict[str, str] = {}  # session_id -> user_id
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id (当前活跃会话)
        self.pending_choice: Dict[str, dict] = {}  # session_id -> {type, data}
        print(f"--- PlannerAgent initialized [{self.default_agent_id}] ---")

    async def reply_to_user(self, session_id: str, message: str):
        """回复给用户"""
        ws = self.workspace()
        user_id = self.user_map.get(session_id)
        if user_id:
            await ws.agent(user_id).send(message)

    async def show_thinking(self, session_id: str, agent_name: str, message: str):
        """展示某个 agent 的思考过程"""
        await self.reply_to_user(session_id, f"💭 [{agent_name}] {message}")

    async def on_direct(self, ctx):
        """Handle direct messages"""
        text, sender = extract_message(ctx)

        if not text or not sender:
            return

        # 尝试解析 JSON（来自其他 agent）
        try:
            msg = json.loads(text)
            if isinstance(msg, dict):
                msg_type = msg.get('type', '')

                if msg_type == 'search_complete':
                    await self.handle_search_complete(msg)
                    return
                elif msg_type == 'critic_complete':
                    await self.handle_critic_complete(msg)
                    return
                elif msg_type == 'review_complete':
                    await self.handle_review_complete(msg)
                    return
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

        # 检查是否有等待用户选择的会话
        session_id = self.user_sessions.get(sender)
        if session_id and session_id in self.pending_choice:
            await self.handle_user_choice(sender, session_id, text.strip())
            return

        # 普通文本 = 用户的新请求
        await self.start_review(text, sender)

    async def start_review(self, topic: str, user_id: str):
        """开始文献综述流程"""
        ws = self.workspace()

        # 清理命令前缀
        topic = topic.replace('/review ', '').replace('综述 ', '').strip()

        if not topic:
            await ws.agent(user_id).send("请输入研究主题，例如：河流溶解氧浓度变化的影响因素")
            return

        # 创建会话
        session_id = str(uuid.uuid4())[:8]
        session = SessionManager(session_id)
        self.sessions[session_id] = session
        self.user_map[session_id] = user_id
        self.user_sessions[user_id] = session_id  # 记录用户当前会话

        # === Planner 思考过程 ===
        await self.show_thinking(session_id, "Planner", f"收到研究主题：{topic}")
        await self.show_thinking(session_id, "Planner", "正在分析主题...")

        # 翻译中文
        search_query = topic
        if self.chinese_processor.is_chinese(topic):
            await self.show_thinking(session_id, "Planner", "检测到中文，翻译为英文检索词...")
            search_query = self.chinese_processor.translate_to_english(topic)
            await self.show_thinking(session_id, "Planner", f"英文检索词：{search_query}")

        session.set_query(topic, search_query)

        # 生成关键词
        await self.show_thinking(session_id, "Planner", "生成多角度检索策略...")
        keywords = await self.generate_keywords(search_query)
        await self.show_thinking(session_id, "Planner", f"生成 {len(keywords)} 个检索关键词")

        # === 派发给 Research ===
        await self.show_thinking(session_id, "Planner", "→ 派发检索任务给 ResearchAgent")

        task = json.dumps({
            "type": "search_task",
            "session_id": session_id,
            "query": search_query,
            "keywords": keywords,
            "year_from": 2021,
            "year_to": 2025,
        }, ensure_ascii=False)

        await ws.agent(AGENT_IDS['research']).send(task)
        session.set_status('searching')

    async def generate_keywords(self, query: str) -> List[str]:
        """生成搜索关键词"""
        prompt = f"""Generate 6 English academic search keywords for this topic.

Topic: {query}

Output only keywords, one per line:"""

        result = self.llm.chat(prompt)
        keywords = [line.strip() for line in result.split('\n') if line.strip()]
        keywords = [k for k in keywords if 2 < len(k) < 50]

        if query not in keywords:
            keywords.insert(0, query)

        return keywords[:6]

    async def handle_search_complete(self, msg: Dict):
        """处理 Research 的搜索结果"""
        ws = self.workspace()

        session_id = msg.get('session_id', '')
        papers = msg.get('papers', [])
        search_count = msg.get('search_count', 0)

        print(f"[Planner] handle_search_complete: session={session_id}, papers={len(papers)}")

        if session_id not in self.sessions:
            print(f"[Planner] ERROR: session {session_id} not found!")
            return

        session = self.sessions[session_id]
        session.add_papers(papers)

        try:
            # 展示 Research 的反馈
            await self.show_thinking(session_id, "Research", f"检索完成，执行了 {search_count} 次搜索")
            await self.show_thinking(session_id, "Research", f"找到 {len(papers)} 篇候选论文")
            await self.show_thinking(session_id, "Research", "→ 将结果发送给 CriticAgent")

            # === 派发给 Critic ===
            await self.show_thinking(session_id, "Planner", "→ 派发质量评估任务给 CriticAgent")

            critic_task = json.dumps({
                "type": "critic_task",
                "session_id": session_id,
                "query": session.data.get('query', ''),
                "papers": papers[:50],
            }, ensure_ascii=False)

            await ws.agent(AGENT_IDS['critic']).send(critic_task)
            session.set_status('reviewing')
            print(f"[Planner] Task sent to Critic successfully")

        except Exception as e:
            print(f"[Planner] ERROR in handle_search_complete: {e}")
            import traceback
            traceback.print_exc()

    async def handle_critic_complete(self, msg: Dict):
        """处理 Critic 的评估结果"""
        ws = self.workspace()

        session_id = msg.get('session_id', '')
        score = msg.get('score', 0)
        verdict = msg.get('verdict', 'PASS')
        top_papers = msg.get('top_papers', [])
        irrelevant_examples = msg.get('irrelevant_examples', [])
        stats = msg.get('stats', {})

        print(f"[Planner] handle_critic_complete: session={session_id}, score={score}")

        if session_id not in self.sessions:
            print(f"[Planner] ERROR: session {session_id} not found!")
            return

        session = self.sessions[session_id]

        try:
            # 保存论文供后续使用
            session.data['top_papers'] = top_papers
            session.data['critic_stats'] = stats
            session.data['critic_score'] = score

            # 展示 Critic 的反馈
            await self.show_thinking(session_id, "Critic", "文献质量评估完成")

            # 显示相关性分析
            relevant_count = stats.get('relevant_count', len(top_papers))
            irrelevant_count = stats.get('irrelevant_count', 0)
            relevance_rate = stats.get('relevance_rate', 'N/A')

            await self.show_thinking(session_id, "Critic",
                                     f"相关性筛选：{relevant_count} 篇相关 / {irrelevant_count} 篇不相关 ({relevance_rate})")

            # 显示不相关论文例子
            if irrelevant_examples:
                examples_text = "发现不相关文献：\n"
                for p in irrelevant_examples[:3]:
                    title = p.get('title', '')[:40]
                    reason = p.get('irrelevance_reason', '主题不符')
                    examples_text += f"  ❌ {title}... - {reason}\n"
                await self.show_thinking(session_id, "Critic", examples_text.strip())

            await self.show_thinking(session_id, "Critic",
                                     f"最终筛选：{len(top_papers)} 篇高质量文献")
            await self.show_thinking(session_id, "Critic",
                                     f"平均引用：{stats.get('avg_citations', 0):.0f} 次")
            await self.show_thinking(session_id, "Critic",
                                     f"质量评分：{score}/10 {'✓' if score >= 6 else '⚠'}")

            # 高分或及格分（>=6）都自动继续
            if score >= 6:
                await self.show_thinking(session_id, "Critic", "→ 文献质量达标，发送给 WriterAgent")
                await self.dispatch_to_writer(session_id)
            else:
                # 低分：让用户选择
                await self.show_thinking(session_id, "Planner", "⚠ 检测到文献质量问题")

                # 分析问题
                problems = []
                if irrelevant_count > relevant_count:
                    problems.append(f"大部分论文与主题不相关 ({relevance_rate})")
                if len(top_papers) < 15:
                    problems.append(f"相关文献数量不足 ({len(top_papers)} 篇)")
                if stats.get('avg_citations', 0) < 10:
                    problems.append(f"文献影响力较低 (平均引用 {stats.get('avg_citations', 0):.0f})")

                if problems:
                    await self.reply_to_user(session_id, f"📊 **问题分析**：\n" + "\n".join(f"  • {p}" for p in problems))

                # 显示被排除的论文例子
                if irrelevant_examples:
                    excluded_text = "**被排除的论文示例**：\n"
                    for p in irrelevant_examples[:3]:
                        title = p.get('title', '')[:50]
                        reason = p.get('irrelevance_reason', '')
                        excluded_text += f"  • {title}...\n    原因：{reason}\n"
                    await self.reply_to_user(session_id, excluded_text)

                # 提供选择
                choice_msg = """
🤔 **请选择下一步操作**：

**1️⃣ 调整关键词重新搜索**
   → 您可以提供新的关键词，我会重新检索

**2️⃣ 接受当前文献，继续生成综述**
   → 使用现有 {paper_count} 篇相关文献撰写

**3️⃣ 换个研究主题**
   → 重新开始新的文献综述

请回复 **1**、**2** 或 **3**：""".format(paper_count=len(top_papers))

                await self.reply_to_user(session_id, choice_msg)

                # 设置等待状态
                self.pending_choice[session_id] = {
                    "type": "critic_review",
                    "score": score,
                    "papers": top_papers,
                }
                session.set_status('waiting_choice')

        except Exception as e:
            print(f"[Planner] ERROR in handle_critic_complete: {e}")
            import traceback
            traceback.print_exc()
            # 出错也尝试继续
            if top_papers:
                await self.show_thinking(session_id, "Planner", "处理过程中遇到问题，尝试继续...")
                await self.dispatch_to_writer(session_id)

    async def handle_user_choice(self, user_id: str, session_id: str, choice: str):
        """处理用户的选择"""
        ws = self.workspace()

        if session_id not in self.pending_choice:
            await ws.agent(user_id).send("没有待处理的选择，请输入新的研究主题开始。")
            return

        pending = self.pending_choice[session_id]
        session = self.sessions.get(session_id)

        if not session:
            return

        # 解析选择
        choice = choice.strip()

        if choice in ['1', '1️⃣', '一']:
            # 调整关键词重新搜索
            del self.pending_choice[session_id]
            await self.show_thinking(session_id, "Planner", "好的，请提供新的搜索关键词")
            await self.reply_to_user(session_id,
                                     "请输入新的搜索关键词（可以是中文或英文）：\n"
                                     "例如：`river nitrogen phosphorus agricultural runoff`"
                                     )
            self.pending_choice[session_id] = {
                "type": "new_keywords",
            }

        elif choice in ['2', '2️⃣', '二']:
            # 接受当前文献，继续生成
            del self.pending_choice[session_id]
            await self.show_thinking(session_id, "Planner", "收到，使用现有文献继续生成综述")
            await self.dispatch_to_writer(session_id)

        elif choice in ['3', '3️⃣', '三']:
            # 换主题
            del self.pending_choice[session_id]
            if session_id in self.sessions:
                del self.sessions[session_id]
            if user_id in self.user_sessions:
                del self.user_sessions[user_id]
            await self.reply_to_user(session_id, "好的，请输入新的研究主题：")

        elif pending.get("type") == "new_keywords":
            # 用户输入了新关键词
            del self.pending_choice[session_id]
            new_keywords = choice.split()
            if len(new_keywords) < 2:
                new_keywords = [choice]  # 整体作为一个关键词

            await self.show_thinking(session_id, "Planner", f"使用新关键词重新搜索：{new_keywords}")
            await self.show_thinking(session_id, "Planner", "→ 派发新检索任务给 ResearchAgent")

            # 重新搜索
            task = json.dumps({
                "type": "search_task",
                "session_id": session_id,
                "query": session.data.get('search_query', ''),
                "keywords": new_keywords,
                "year_from": 2021,
                "year_to": 2025,
            }, ensure_ascii=False)

            await ws.agent(AGENT_IDS['research']).send(task)
            session.set_status('searching')

        else:
            await self.reply_to_user(session_id, "请回复 **1**、**2** 或 **3** 选择操作：")

    async def dispatch_to_writer(self, session_id: str):
        """派发任务给 Writer"""
        ws = self.workspace()
        session = self.sessions.get(session_id)

        if not session:
            print(f"[Planner] dispatch_to_writer: session {session_id} not found!")
            return

        top_papers = session.data.get('top_papers', [])
        print(f"[Planner] dispatch_to_writer: {len(top_papers)} papers")

        try:
            await self.show_thinking(session_id, "Planner", "→ 派发撰写任务给 WriterAgent")

            writer_task = json.dumps({
                "type": "write_task",
                "session_id": session_id,
                "query": session.data.get('query', ''),
                "original_topic": session.data.get('query', ''),
                "papers": top_papers[:30],
            }, ensure_ascii=False)

            await ws.agent(AGENT_IDS['writer']).send(writer_task)
            session.set_status('writing')
            print(f"[Planner] Task sent to Writer successfully")

        except Exception as e:
            print(f"[Planner] ERROR in dispatch_to_writer: {e}")
            import traceback
            traceback.print_exc()

    async def handle_review_complete(self, msg: Dict):
        """处理 Writer 的综述结果"""
        session_id = msg.get('session_id', '')
        review = msg.get('review', '')
        paper_count = msg.get('paper_count', 0)
        word_count = msg.get('word_count', 0)
        pdf_base64 = msg.get('pdf_base64')

        print(f"[Planner] handle_review_complete: session={session_id}, words={word_count}")

        if session_id not in self.sessions:
            print(f"[Planner] ERROR: session {session_id} not found!")
            return

        session = self.sessions[session_id]

        try:
            # 展示 Writer 的反馈
            await self.show_thinking(session_id, "Writer", "综述撰写完成")
            await self.show_thinking(session_id, "Writer", f"引用文献：{paper_count} 篇")
            await self.show_thinking(session_id, "Writer", f"综述字数：{word_count} 字")

            # 处理 PDF
            pdf_saved = False
            pdf_path = None
            if pdf_base64:
                try:
                    import base64
                    import os

                    # 保存 PDF 到文件
                    pdf_dir = os.path.expanduser("~/geomind_outputs")
                    os.makedirs(pdf_dir, exist_ok=True)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    pdf_filename = f"review_{session_id}_{timestamp}.pdf"
                    pdf_path = os.path.join(pdf_dir, pdf_filename)

                    with open(pdf_path, 'wb') as f:
                        f.write(base64.b64decode(pdf_base64))

                    pdf_saved = True
                    await self.show_thinking(session_id, "Writer", f"📄 PDF 已生成：{pdf_filename}")
                    print(f"[Planner] PDF saved to: {pdf_path}")

                except Exception as e:
                    print(f"[Planner] PDF save error: {e}")

            # 展示完成
            await self.show_thinking(session_id, "Planner", "✅ 多智能体协作完成！")

            # 发送最终综述
            header = f"""
---
📚 **GeoMind 文献综述**
🔍 主题：{session.data.get('query', '')}
📄 引用文献：{paper_count} 篇
🤖 协作智能体：Planner → Research → Critic → Writer
📅 生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M")}
"""
            if pdf_saved:
                header += f"💾 PDF 已保存：`{pdf_path}`\n"

            header += "\n---\n\n"

            full_review = header + review

            # 分段发送
            if len(full_review) > 3500:
                chunks = self.split_text(full_review, 3000)
                for i, chunk in enumerate(chunks):
                    await self.reply_to_user(session_id, chunk if i == 0 else f"*(续)*\n\n{chunk}")
                    await asyncio.sleep(0.3)
            else:
                await self.reply_to_user(session_id, full_review)

            # 如果有 PDF，提示用户
            if pdf_saved:
                await self.reply_to_user(session_id,
                                         f"📥 **PDF 下载**\n\n"
                                         f"综述 PDF 已保存到服务器：\n"
                                         f"`{pdf_path}`\n\n"
                                         f"您也可以复制上面的 Markdown 内容，使用在线工具转换：\n"
                                         f"• https://md2pdf.netlify.app\n"
                                         f"• https://www.markdowntopdf.com"
                                         )

            session.set_status('completed')
            print(f"[Planner] Review sent to user successfully")

        except Exception as e:
            print(f"[Planner] ERROR in handle_review_complete: {e}")
            import traceback
            traceback.print_exc()

    def split_text(self, text: str, max_length: int = 3000) -> List[str]:
        """分割长文本"""
        chunks = []
        sections = text.split('\n## ')

        current = ""
        for i, section in enumerate(sections):
            if i > 0:
                section = '## ' + section

            if len(current) + len(section) < max_length:
                current += section + "\n"
            else:
                if current:
                    chunks.append(current.strip())
                current = section + "\n"

        if current:
            chunks.append(current.strip())

        return chunks if chunks else [text]
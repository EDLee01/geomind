"""
GeoMind OpenAgents - Shared Tools
Qdrant 向量搜索 + DeepSeek LLM + FastEmbed 本地 Embedding
"""
import json
import math
from datetime import datetime
from typing import List, Dict

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range
from fastembed import TextEmbedding

from config import (
    QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    EMBEDDING_MODEL, TOP_K_CANDIDATES, RERANK_WEIGHTS,
    MAX_TOKENS, TEMPERATURE
)


def extract_message(ctx):
    """
    Extract message text and sender from EventContext
    """
    try:
        text = getattr(ctx, 'text', '') or ''
        sender_id = getattr(ctx, 'source_id', 'unknown') or 'unknown'
        
        if not text and hasattr(ctx, 'payload'):
            payload = ctx.payload
            if isinstance(payload, dict):
                content = payload.get('content', {})
                if isinstance(content, dict):
                    text = content.get('text', '')
                sender_id = payload.get('sender_id', sender_id)
        
        return text.strip() if text else '', sender_id
    except Exception as e:
        print(f"[extract_message error]: {e}")
        return '', 'unknown'


class LLMClient:
    """DeepSeek LLM Client"""
    
    def __init__(self):
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
        print(f"[LLMClient] Connected to DeepSeek API")
    
    def chat(self, prompt: str, system_prompt: str = None) -> str:
        """Sync chat request"""
        messages = []
        
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[LLM Error]: {e}")
            return f"[LLM Error: {str(e)}]"
    
    async def achat(self, prompt: str, system_prompt: str = None) -> str:
        """Async chat request (wrapper for sync)"""
        return self.chat(prompt, system_prompt)


class VectorStore:
    """Qdrant Vector Store with FastEmbed"""
    
    def __init__(self):
        # Qdrant Client
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=60
        )
        self.collection = QDRANT_COLLECTION
        
        # FastEmbed for local embeddings
        self.embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
        
        # Check connection
        try:
            info = self.client.get_collection(self.collection)
            print(f"[VectorStore] Connected to Qdrant: {info.points_count} papers")
        except Exception as e:
            print(f"[VectorStore] Connection error: {e}")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding using FastEmbed"""
        embeddings = list(self.embedder.embed([text]))
        return embeddings[0].tolist() if embeddings else []

    async def search(self, query: str, n_results: int = TOP_K_CANDIDATES,
                     year_from: int = None, year_to: int = None) -> List[Dict]:
        """Semantic search with optional year filter"""

        query_vector = self._get_embedding(query)
        if not query_vector:
            print(f"[VectorStore] Failed to get embedding")
            return []

        filter_conditions = []
        if year_from:
            filter_conditions.append(
                FieldCondition(key="year", range=Range(gte=year_from))
            )
        if year_to:
            filter_conditions.append(
                FieldCondition(key="year", range=Range(lte=year_to))
            )

        query_filter = Filter(must=filter_conditions) if filter_conditions else None

        try:
            results = self.client.query_points(
                collection_name=self.collection,  # ✅ 修复
                query=query_vector,
                query_filter=query_filter,  # ✅ 修复
                limit=n_results,
                with_payload=True
            )

            papers = []
            points = results.points if hasattr(results, 'points') else results

            for hit in points:
                payload = hit.payload or {}
                papers.append({
                    'id': str(hit.id),
                    'similarity': hit.score,
                    'title': payload.get('title', ''),
                    'abstract': payload.get('abstract', ''),
                    'journal': payload.get('journal_name', payload.get('journal', '')),
                    'year': payload.get('year', 0),
                    'authors': payload.get('authors', ''),
                    'doi': payload.get('doi', ''),
                    'cited_by_count': payload.get('cited_by_count', 0),
                    'cas_zone': payload.get('cas_zone', ''),
                })

            print(f"[VectorStore] Found {len(papers)} papers")
            return papers

        except Exception as e:
            print(f"[VectorStore] Search error: {e}")
            return []
    def rerank_papers(self, papers: List[Dict], current_year: int = None) -> List[Dict]:
        """Multi-factor reranking"""
        if not papers:
            return []
        
        current_year = current_year or datetime.now().year
        
        for paper in papers:
            # Similarity score
            sim_score = paper.get('similarity', 0) * RERANK_WEIGHTS['similarity']
            
            # Citation score (log scale)
            citations = paper.get('cited_by_count', 0)
            cite_score = (math.log(citations + 1) / 10) * RERANK_WEIGHTS['citation']
            
            # Recency score
            year = paper.get('year', 0)
            if year and (current_year - year) <= 3:
                recency_score = 0.15
            elif year and (current_year - year) <= 5:
                recency_score = 0.10
            else:
                recency_score = 0.05
            recency_score *= RERANK_WEIGHTS['recency'] / 0.15
            
            # Quality score (CAS Zone 1 = highest)
            cas_zone = paper.get('cas_zone', '')
            if cas_zone == '1' or cas_zone == 1:
                quality_score = 0.10
            else:
                quality_score = 0.05
            quality_score *= RERANK_WEIGHTS['quality'] / 0.10
            
            paper['rerank_score'] = sim_score + cite_score + recency_score + quality_score
        
        return sorted(papers, key=lambda x: x.get('rerank_score', 0), reverse=True)


class ChineseProcessor:
    """Chinese text processing with DeepSeek"""
    
    def __init__(self):
        self.llm = LLMClient()
    
    @staticmethod
    def is_chinese(text: str) -> bool:
        """Detect if text is primarily Chinese"""
        if not text:
            return False
        chinese_count = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        return chinese_count / max(len(text), 1) > 0.3
    
    def translate_to_english(self, chinese_text: str) -> str:
        """Translate Chinese to English for search"""
        prompt = f"""Translate the following Chinese text to English for academic literature search.
Only output the translation, nothing else. Keep technical terms accurate.

Chinese: {chinese_text}

English:"""
        
        result = self.llm.chat(prompt)
        return result.strip()


class SessionManager:
    """Session manager for review workflow"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.data = {
            'query': '',
            'translated_query': '',
            'plan': {},
            'papers': {},
            'analysis': {},
            'outline': '',
            'review': '',
            'status': 'initialized',
        }
    
    def set_query(self, query: str, translated: str = None):
        self.data['query'] = query
        self.data['translated_query'] = translated or query
    
    def add_papers(self, papers: List[Dict]):
        """Add papers with deduplication"""
        for paper in papers:
            doi = paper.get('doi', '')
            paper_id = paper.get('id', '')
            key = doi or paper_id
            if key and key not in self.data['papers']:
                self.data['papers'][key] = paper
    
    def get_all_papers(self) -> List[Dict]:
        return list(self.data['papers'].values())
    
    def get_paper_count(self) -> int:
        return len(self.data['papers'])
    
    def set_outline(self, outline: str):
        self.data['outline'] = outline
    
    def set_review(self, review: str):
        self.data['review'] = review
    
    def set_status(self, status: str):
        self.data['status'] = status
    
    def to_json(self) -> str:
        return json.dumps(self.data, ensure_ascii=False, indent=2)


def format_paper_brief(paper: Dict) -> str:
    """Format paper brief info"""
    title = paper.get('title', 'Unknown')[:80]
    journal = paper.get('journal', '')
    year = paper.get('year', '')
    citations = paper.get('cited_by_count', 0)
    
    return f"- {title}... ({journal}, {year}, cited:{citations})"


def format_paper_full(paper: Dict, index: int = 1) -> str:
    """Format paper full info"""
    return f"""
{index}. {paper.get('title', 'Unknown Title')}
   Journal: {paper.get('journal', 'N/A')}
   Year: {paper.get('year', 'N/A')}
   Authors: {str(paper.get('authors', 'N/A'))[:50]}...
   Citations: {paper.get('cited_by_count', 0)} | CAS Zone: {paper.get('cas_zone', 'N/A')}
   DOI: {paper.get('doi', 'N/A')}
"""


def format_papers_for_llm(papers: List[Dict], max_papers: int = 30) -> str:
    """Format papers for LLM context"""
    result = []
    for i, p in enumerate(papers[:max_papers], 1):
        result.append(f"""
[{i}] {p.get('title', 'Unknown')}
Authors: {str(p.get('authors', ''))[:100]}
Journal: {p.get('journal', '')} ({p.get('year', '')})
Citations: {p.get('cited_by_count', 0)}
Abstract: {p.get('abstract', '')[:500]}...
""")
    return "\n".join(result)


# ========================
# Global Instances
# ========================
_llm_client = None
_vector_store = None
_chinese_processor = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def get_chinese_processor() -> ChineseProcessor:
    global _chinese_processor
    if _chinese_processor is None:
        _chinese_processor = ChineseProcessor()
    return _chinese_processor

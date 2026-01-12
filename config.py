"""
Deep Review Agents - Configuration
整合 GeoMind 的 Qdrant + DeepSeek 后端
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ========================
# Qdrant Cloud Configuration
# ========================
# Qdrant (从环境变量读取，不要硬编码!)
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "geomind_papers")


# DeepSeek (从环境变量读取，不要硬编码!)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


# ========================
# FastEmbed Configuration (本地 Embedding)
# ========================
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
EMBEDDING_DIM = 768

# ========================
# Path Configuration
# ========================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================
# OpenAgents Network Configuration
# ========================
NETWORK_HOST = os.getenv("NETWORK_HOST", "127.0.0.1")
NETWORK_PORT = int(os.getenv("NETWORK_PORT", "8700"))
NETWORK_ID = os.getenv("NETWORK_ID", "main")

# ========================
# Search Parameters
# ========================
TOP_K_CANDIDATES = 50  # 每次搜索候选数
MAX_SEARCH_ROUNDS = 1  # 搜索轮数
MIN_PAPERS_TARGET = 50  # 最少论文数

# Agent IDs
AGENT_IDS = {
    "planner": os.getenv("PLANNER_AGENT_ID", "geomind-planner"),
    "research": os.getenv("RESEARCH_AGENT_ID", "geomind-research"),
    "critic": os.getenv("CRITIC_AGENT_ID", "geomind-critic"),
    "writer": os.getenv("WRITER_AGENT_ID", "geomind-writer"),
}

# ========================
# Channel Configuration
# ========================
REVIEW_CHANNEL = "#geomind-review"

# ========================
# Reranking Weights
# ========================
RERANK_WEIGHTS = {
    "similarity": 0.50,
    "citation": 0.25,
    "recency": 0.15,
    "quality": 0.10,
}

# ========================
# LLM Configuration
# ========================
MAX_TOKENS = 8192
TEMPERATURE = 0.7

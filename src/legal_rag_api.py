"""
法律 RAG 检索 API 服务
- 封装现有的 DenseRetriever + BM25Retriever + Reranker
- 提供 FastAPI REST 接口，供 DeerFlow Agent 调用
"""

import sys
from pathlib import Path

# 确保 src 目录在 Python 路径中
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel

from retriever import DenseRetriever, BM25Retriever
from merger import weighted_fusion
from reranker import Reranker

# ── 全局实例 ──
app = FastAPI(title="Legal RAG API", version="1.0")

dense_retriever: Optional[DenseRetriever] = None
bm25_retriever: Optional[BM25Retriever] = None
reranker: Optional[Reranker] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    use_reranker: bool = True


class SearchResult(BaseModel):
    text: str
    source: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total: int


@app.on_event("startup")
def load_models():
    """服务启动时加载所有检索模块"""
    global dense_retriever, bm25_retriever, reranker

    print("[API] 加载向量检索器...")
    dense_retriever = DenseRetriever(model_name="BAAI/bge-small-zh-v1.5")
    dense_retriever.load()

    print("[API] 加载 BM25 检索器...")
    bm25_retriever = BM25Retriever()
    bm25_retriever.load()

    # Reranker 需要约 1.2GB 内存，内存不足时跳过
    try:
        print("[API] 加载 Reranker...")
        reranker = Reranker(model_name="BAAI/bge-reranker-v2-m3")
        reranker.load()
    except Exception as e:
        print(f"[API] Reranker 加载失败 ({e})，将使用 RRF 融合排序代替")
        reranker = None

    print("[API] 所有模块加载完成！")


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """
    法律检索接口
    1. 向量检索 Top20
    2. BM25 检索 Top20
    3. RRF 融合
    4. Reranker 重排 (可选)
    5. 返回 Top-K 结果
    """
    # 双路召回
    dense_results = dense_retriever.search(req.query, top_k=20)
    bm25_results = bm25_retriever.search(req.query, top_k=20)

    # RRF 融合
    merged = weighted_fusion(dense_results, bm25_results, top_k=20)

    # Reranker 重排
    if req.use_reranker and reranker is not None:
        ranked = reranker.rerank(req.query, merged, top_k=req.top_k)
        results = [
            SearchResult(
                text=r["text"],
                source=r.get("source", ""),
                score=r.get("rerank_score", 0.0),
            )
            for r in ranked
        ]
    else:
        results = [
            SearchResult(
                text=r["text"],
                source=r.get("source", ""),
                score=r.get("score", 0.0),
            )
            for r in merged[:req.top_k]
        ]

    return SearchResponse(
        query=req.query,
        results=results,
        total=len(results),
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "dense_loaded": dense_retriever is not None,
        "bm25_loaded": bm25_retriever is not None,
        "reranker_loaded": reranker is not None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

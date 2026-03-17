"""
重排序模块
- 基于 bge-reranker 系列模型对检索结果进行精排
- 支持多模型对比
- 统一接口: rerank(query, candidates, top_k) -> List[Dict]
"""

import time
from typing import List, Dict, Optional
from sentence_transformers import CrossEncoder


# 可选的 reranker 模型
RERANKER_MODELS = {
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    "bge-reranker-large": "BAAI/bge-reranker-large",
    "bge-reranker-base": "BAAI/bge-reranker-base",
}


class Reranker:
    """基于 BGE-Reranker 的重排序器"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self.model: Optional[CrossEncoder] = None

    def load(self):
        """加载 reranker 模型"""
        print(f"[Reranker] 加载模型: {self.model_name}")
        t0 = time.time()
        self.model = CrossEncoder(self.model_name, max_length=512)
        print(f"[Reranker] 加载完成: {time.time() - t0:.1f}s")

    def rerank(self, query: str, candidates: List[Dict],
               top_k: int = 10) -> List[Dict]:
        """
        对候选文档重排序

        参数:
            query: 用户查询
            candidates: 检索返回的候选文档 [{text, source, ...}, ...]
            top_k: 重排后返回的文档数

        返回:
            按 reranker 分数降序排列的 top_k 个文档
        """
        if not candidates:
            return []

        # 构造 query-document pairs
        pairs = [(query, c["text"]) for c in candidates]

        # 计算 reranker 分数
        scores = self.model.predict(pairs)

        # 给每个候选附上 reranker 分数
        for i, score in enumerate(scores):
            candidates[i] = candidates[i].copy()
            candidates[i]["rerank_score"] = float(score)

        # 按 reranker 分数降序排列
        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        return ranked[:top_k]


# ── CLI: 测试 reranker ──

if __name__ == "__main__":
    import sys
    from retriever import DenseRetriever, BM25Retriever
    from merger import weighted_fusion

    model_name = sys.argv[1] if len(sys.argv) > 1 else "BAAI/bge-reranker-v2-m3"

    # 加载检索器
    print("加载检索器...")
    dense = DenseRetriever()
    dense.load()
    bm25 = BM25Retriever()
    bm25.load()

    # 加载 reranker
    reranker = Reranker(model_name=model_name)
    reranker.load()

    # 测试
    test_queries = [
        "离婚财产怎么分割",
        "盗窃罪怎么量刑",
        "合同违约要承担什么责任",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"查询: {q}")

        # 双路召回 Top20
        dr = dense.search(q, top_k=20)
        br = bm25.search(q, top_k=20)
        candidates = weighted_fusion(dr, br, top_k=20)

        # Rerank 取 Top5
        reranked = reranker.rerank(q, candidates, top_k=5)

        print(f"\n[Rerank 前 Top3]")
        for i, r in enumerate(candidates[:3]):
            print(f"  [{i+1}] rrf={r.get('score',0):.4f} | {r['text'][:60]}")

        print(f"\n[Rerank 后 Top3]")
        for i, r in enumerate(reranked[:3]):
            print(f"  [{i+1}] rerank={r['rerank_score']:.4f} | {r['text'][:60]}")

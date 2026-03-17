"""
检索模块
- DenseRetriever: 向量检索（基于 FAISS）
- BM25Retriever:  稀疏检索（基于 BM25 + jieba 分词）
- 统一接口: search(query, top_k) -> List[Dict]
"""

import json
import time
import pickle
from pathlib import Path
from typing import List, Dict, Optional

import jieba
from rank_bm25 import BM25Okapi

from vectorstore import FaissVectorStore, load_chunks, INDEX_DIR, PROCESSED_DIR


class DenseRetriever:
    """向量检索：封装 FaissVectorStore"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.model_name = model_name
        self.store = FaissVectorStore(model_name=model_name)

    def load(self):
        """加载已构建的 FAISS 索引"""
        self.store.load()
        self.store._load_model()
        print(f"[DenseRetriever] 已加载 {self.store.index.ntotal} 个向量")

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        向量检索
        返回: [{text, source, title, section, chunk_id, score}, ...]
        """
        return self.store.search(query, top_k=top_k)


class BM25Retriever:
    """稀疏检索：BM25 + jieba 中文分词"""

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[Dict] = []
        self.tokenized_corpus: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        """jieba 分词，过滤单字符和纯空白"""
        return [w for w in jieba.lcut(text) if len(w.strip()) > 1]

    def build(self, chunks: List[Dict]):
        """
        构建 BM25 索引
        对每个 chunk 的 text 做分词，然后构建 BM25Okapi
        """
        self.chunks = chunks
        print(f"[BM25] 开始分词 {len(chunks)} 个 chunks...")
        start = time.time()
        self.tokenized_corpus = [self._tokenize(c["text"]) for c in chunks]
        elapsed = time.time() - start
        print(f"[BM25] 分词完成: {elapsed:.1f}s")

        print("[BM25] 构建 BM25 索引...")
        start = time.time()
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        elapsed = time.time() - start
        print(f"[BM25] 索引构建完成: {elapsed:.1f}s")

    def save(self, name: str = "bm25"):
        """保存 BM25 索引到磁盘"""
        save_dir = INDEX_DIR / name
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "bm25.pkl", "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "tokenized_corpus": self.tokenized_corpus,
            }, f)
        with open(save_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        print(f"[BM25] 已保存: {save_dir}")

    def load(self, name: str = "bm25"):
        """从磁盘加载 BM25 索引"""
        save_dir = INDEX_DIR / name

        with open(save_dir / "bm25.pkl", "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.tokenized_corpus = data["tokenized_corpus"]
        with open(save_dir / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        print(f"[BM25] 已加载: {len(self.chunks)} 个 chunks")

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        BM25 检索
        返回: [{text, source, title, section, chunk_id, score}, ...]
        """
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # 取 top_k 个最高分
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            idx = int(idx)
            if scores[idx] <= 0:
                break
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(scores[idx])
            results.append(chunk)

        return results


# ── CLI: 构建 BM25 索引 + 测试 ──

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "build"

    if mode == "build":
        # 构建并保存 BM25 索引
        chunks = load_chunks()
        bm25_retriever = BM25Retriever()
        bm25_retriever.build(chunks)
        bm25_retriever.save()

    elif mode == "test":
        # 加载并测试两种检索
        print("=" * 60)
        print("加载检索器...")
        print("=" * 60)

        dense = DenseRetriever()
        dense.load()

        bm25 = BM25Retriever()
        bm25.load()

        test_queries = [
            "盗窃罪怎么量刑",
            "正当防卫的条件",
            "劳动合同解除的条件",
        ]

        for q in test_queries:
            print(f"\n{'='*60}")
            print(f"查询: {q}")
            print(f"{'='*60}")

            print("\n[向量检索 Top5]")
            for i, r in enumerate(dense.search(q, top_k=5)):
                print(f"  [{i+1}] score={r['score']:.4f} | {r['source']} | {r['text'][:60]}")

            print("\n[BM25 检索 Top5]")
            for i, r in enumerate(bm25.search(q, top_k=5)):
                print(f"  [{i+1}] score={r['score']:.4f} | {r['source']} | {r['text'][:60]}")

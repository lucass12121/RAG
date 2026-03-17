"""
向量库构建模块
- 加载 chunks.jsonl
- 用 embedding 模型编码
- 构建 FAISS 索引
- 支持保存/加载/检索
"""

import json
import time
import pickle
from pathlib import Path
from typing import List, Dict, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
INDEX_DIR = ROOT_DIR / "data" / "index"

# 默认模型：BGE-small-zh，轻量级中文 embedding，适合 CPU 测试
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"


def load_chunks(filepath: Path = None) -> List[Dict]:
    """加载 chunks.jsonl"""
    if filepath is None:
        filepath = PROCESSED_DIR / "chunks.jsonl"
    chunks = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"加载 {len(chunks)} 个 chunks")
    return chunks


class FaissVectorStore:
    """FAISS 向量库"""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[Dict] = []
        self.dimension: int = 0

    def _load_model(self):
        """懒加载 embedding 模型"""
        if self.model is None:
            print(f"加载模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            print(f"模型维度: {self.dimension}")

    def build(self, chunks: List[Dict], batch_size: int = 32,
              checkpoint_every: int = 5000):
        """
        构建 FAISS 索引（支持断点续存）
        1. 分组编码，每 checkpoint_every 个 chunk 保存一次中间结果
        2. 归一化（用于余弦相似度）
        3. 存入 IndexFlatIP（内积 = 余弦相似度，因为已归一化）
        """
        self._load_model()
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        total = len(texts)

        # 检查是否有断点可恢复
        ckpt_dir = INDEX_DIR / "_checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_meta = ckpt_dir / "meta.json"

        encoded_offset = 0
        all_embeddings = []

        if ckpt_meta.exists():
            with open(ckpt_meta, "r") as f:
                meta = json.load(f)
            if meta.get("model_name") == self.model_name and meta.get("total") == total:
                encoded_offset = meta["encoded"]
                # 加载已保存的 embeddings
                for i in range(0, encoded_offset, checkpoint_every):
                    part = np.load(ckpt_dir / f"emb_{i}.npy")
                    all_embeddings.append(part)
                print(f"从断点恢复: 已编码 {encoded_offset}/{total}")

        print(f"\n开始编码 {total} 个 chunks (batch_size={batch_size}, "
              f"checkpoint_every={checkpoint_every})...")
        start = time.time()

        # 分组编码
        for group_start in range(encoded_offset, total, checkpoint_every):
            group_end = min(group_start + checkpoint_every, total)
            group_texts = texts[group_start:group_end]

            group_emb = self.model.encode(
                group_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            group_emb = np.array(group_emb, dtype=np.float32)
            all_embeddings.append(group_emb)

            # 保存 checkpoint
            np.save(ckpt_dir / f"emb_{group_start}.npy", group_emb)
            with open(ckpt_meta, "w") as f:
                json.dump({
                    "model_name": self.model_name,
                    "total": total,
                    "encoded": group_end,
                }, f)

            elapsed = time.time() - start
            speed = (group_end - encoded_offset) / elapsed if elapsed > 0 else 0
            eta = (total - group_end) / speed if speed > 0 else 0
            print(f"  [{group_end}/{total}] "
                  f"{speed:.0f} chunks/s, ETA {eta/60:.1f}min")

        elapsed = time.time() - start
        actually_encoded = total - encoded_offset
        if actually_encoded > 0:
            print(f"编码完成: {elapsed:.1f}s "
                  f"({actually_encoded/elapsed:.0f} chunks/s)")

        # 合并所有 embeddings，构建 FAISS 索引
        embeddings = np.concatenate(all_embeddings, axis=0)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        print(f"FAISS 索引构建完成: {self.index.ntotal} 向量, 维度 {self.dimension}")

        # 清理 checkpoint
        import shutil
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    def save(self, name: str = None):
        """保存索引和元数据到磁盘"""
        if name is None:
            # 用模型名做目录名，替换斜杠
            name = self.model_name.replace("/", "_")
        save_dir = INDEX_DIR / name
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存 FAISS 索引
        faiss.write_index(self.index, str(save_dir / "index.faiss"))
        # 保存 chunks 元数据
        with open(save_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        # 保存配置
        with open(save_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump({
                "model_name": self.model_name,
                "dimension": self.dimension,
                "num_chunks": len(self.chunks),
            }, f, ensure_ascii=False, indent=2)

        print(f"索引已保存: {save_dir}")

    def load(self, name: str = None):
        """从磁盘加载索引和元数据"""
        if name is None:
            name = self.model_name.replace("/", "_")
        save_dir = INDEX_DIR / name

        # 加载配置
        with open(save_dir / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        self.model_name = config["model_name"]
        self.dimension = config["dimension"]

        # 加载 FAISS 索引
        self.index = faiss.read_index(str(save_dir / "index.faiss"))
        # 加载 chunks 元数据
        with open(save_dir / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        print(f"索引已加载: {self.index.ntotal} 向量, 模型: {self.model_name}")

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        检索最相关的 chunks
        返回: [{text, source, title, section, score}, ...]
        """
        self._load_model()

        # 编码查询
        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        query_vec = np.array(query_vec, dtype=np.float32)

        # FAISS 检索
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS 返回 -1 表示无效
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)

        return results


def build_index(model_name: str = DEFAULT_MODEL, batch_size: int = 32):
    """一键构建索引的便捷函数"""
    chunks = load_chunks()
    store = FaissVectorStore(model_name=model_name)
    store.build(chunks, batch_size=batch_size)
    store.save()
    return store


if __name__ == "__main__":
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 32

    store = build_index(model_name=model, batch_size=batch)

    # 测试检索
    print("\n" + "=" * 50)
    print("测试检索")
    print("=" * 50)
    test_queries = [
        "盗窃罪怎么量刑",
        "离婚财产怎么分割",
        "合同违约要承担什么责任",
    ]
    for q in test_queries:
        print(f"\n查询: {q}")
        results = store.search(q, top_k=3)
        for i, r in enumerate(results):
            print(f"  [{i+1}] score={r['score']:.4f} | {r['source']} | {r.get('section','')}")
            print(f"      {r['text'][:80]}...")

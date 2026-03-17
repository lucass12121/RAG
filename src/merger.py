"""
合并策略模块
- union_merge:      向量 TopK + BM25 TopK 取并集去重
- weighted_fusion:  RRF（Reciprocal Rank Fusion）加权融合排序
"""

from typing import List, Dict


def union_merge(
    dense_results: List[Dict],
    sparse_results: List[Dict],
    top_k: int = 20,
) -> List[Dict]:
    """
    策略一：取并集
    1. 合并两路结果
    2. 按 chunk_id 去重（保留分数更高的那条，分数归一化后比较）
    3. 按原始排名排序（先 dense 后 sparse 的出现顺序）
    4. 截取 top_k
    """
    seen = {}  # chunk_id -> result dict (带 rank 信息)

    # dense 结果先加入，记录排名
    for rank, r in enumerate(dense_results):
        cid = r["chunk_id"]
        entry = r.copy()
        entry["retriever"] = "dense"
        entry["rank"] = rank
        seen[cid] = entry

    # sparse 结果，如果 chunk_id 未出现则加入
    for rank, r in enumerate(sparse_results):
        cid = r["chunk_id"]
        if cid not in seen:
            entry = r.copy()
            entry["retriever"] = "sparse"
            entry["rank"] = len(dense_results) + rank  # 排在 dense 后面
            seen[cid] = entry

    # 按加入顺序（rank）排序
    merged = sorted(seen.values(), key=lambda x: x["rank"])
    return merged[:top_k]


def weighted_fusion(
    dense_results: List[Dict],
    sparse_results: List[Dict],
    top_k: int = 20,
    k: int = 60,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
) -> List[Dict]:
    """
    策略二：RRF（Reciprocal Rank Fusion）加权融合排序

    RRF 公式: score(d) = Σ weight_i / (k + rank_i(d))
    - k=60 是标准 RRF 常数，用于平滑排名差异
    - dense_weight / sparse_weight 控制两路权重

    优点：不依赖原始分数的尺度（Dense 是 0~1, BM25 是 0~∞），
    只用排名信息，天然归一化。
    """
    rrf_scores = {}   # chunk_id -> rrf_score
    chunk_data = {}   # chunk_id -> chunk dict

    # Dense 路 RRF 分数
    for rank, r in enumerate(dense_results):
        cid = r["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + dense_weight / (k + rank + 1)
        if cid not in chunk_data:
            chunk_data[cid] = r.copy()

    # Sparse 路 RRF 分数
    for rank, r in enumerate(sparse_results):
        cid = r["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + sparse_weight / (k + rank + 1)
        if cid not in chunk_data:
            chunk_data[cid] = r.copy()

    # 按 RRF 分数降序排序
    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    results = []
    for cid in sorted_ids[:top_k]:
        entry = chunk_data[cid].copy()
        entry["score"] = rrf_scores[cid]  # 替换为 RRF 分数
        entry["retriever"] = "fusion"
        results.append(entry)

    return results

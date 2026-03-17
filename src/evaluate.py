"""
RAG 检索评测模块
- 自动构造评测数据集（从 law_qa 抽样）
- 评测指标: Hit@K, MRR, Recall@K
- 对比: Dense / BM25 / Union / RRF
"""

import json
import random
import time
from pathlib import Path
from typing import List, Dict, Set

from retriever import DenseRetriever, BM25Retriever
from merger import union_merge, weighted_fusion
from vectorstore import load_chunks, PROCESSED_DIR

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
TEST_QUERIES_PATH = PROCESSED_DIR / "test_queries.jsonl"


# ── 评测数据集构造 ──

def build_test_set(n_samples: int = 200, seed: int = 42) -> List[Dict]:
    """
    从 law_qa 中抽样构造评测集

    每条测试数据:
    {
        "query": "用户问题",
        "relevant_ids": [chunk_id1, chunk_id2, ...],  # 所有包含此问题的 chunk
    }

    策略：
    - 用 title（问题）做 query
    - 找到所有 text 中包含该 title 的 chunk 作为 ground truth
    - 过滤太短的问题（<10字符）
    """
    chunks = load_chunks()

    # 收集 law_qa chunks，按 title 分组
    qa_by_title = {}
    for c in chunks:
        if c["source"] != "law_qa":
            continue
        title = c.get("title", "")
        if len(title) < 10:
            continue
        if title not in qa_by_title:
            qa_by_title[title] = []
        qa_by_title[title].append(c["chunk_id"])

    # 抽样
    random.seed(seed)
    titles = list(qa_by_title.keys())
    if len(titles) > n_samples:
        titles = random.sample(titles, n_samples)

    test_set = []
    for title in titles:
        test_set.append({
            "query": title,
            "relevant_ids": qa_by_title[title],
        })

    print(f"构造评测集: {len(test_set)} 条查询")
    avg_rel = sum(len(t["relevant_ids"]) for t in test_set) / len(test_set)
    print(f"平均每条 ground truth 数: {avg_rel:.1f}")

    return test_set


def save_test_set(test_set: List[Dict], path: Path = None):
    """保存评测集"""
    if path is None:
        path = TEST_QUERIES_PATH
    with open(path, "w", encoding="utf-8") as f:
        for item in test_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"评测集已保存: {path}")


def load_test_set(path: Path = None) -> List[Dict]:
    """加载评测集"""
    if path is None:
        path = TEST_QUERIES_PATH
    test_set = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_set.append(json.loads(line))
    print(f"加载评测集: {len(test_set)} 条")
    return test_set


# ── 评测指标 ──

def hit_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """Hit@K: top-K 中是否至少命中一个 relevant (0 或 1)"""
    for rid in retrieved_ids[:k]:
        if rid in relevant_ids:
            return 1.0
    return 0.0


def recall_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """Recall@K: top-K 中命中了多少比例的 relevant"""
    if not relevant_ids:
        return 0.0
    hits = sum(1 for rid in retrieved_ids[:k] if rid in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: List[int], relevant_ids: Set[int]) -> float:
    """MRR 的单条: 1 / (第一个命中的排名)"""
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


# ── 评测流程 ──

def evaluate_retriever(
    name: str,
    search_fn,
    test_set: List[Dict],
    ks: List[int] = [1, 5, 10, 20],
) -> Dict:
    """
    评测单个检索方法

    参数:
        name: 方法名称
        search_fn: 接受 (query, top_k) 返回 List[Dict]
        test_set: 评测数据
        ks: 要评测的 K 值列表
    """
    max_k = max(ks)
    results = {f"Hit@{k}": 0.0 for k in ks}
    results.update({f"Recall@{k}": 0.0 for k in ks})
    results["MRR"] = 0.0

    start = time.time()
    for item in test_set:
        query = item["query"]
        relevant = set(item["relevant_ids"])

        retrieved = search_fn(query, max_k)
        retrieved_ids = [r["chunk_id"] for r in retrieved]

        for k in ks:
            results[f"Hit@{k}"] += hit_at_k(retrieved_ids, relevant, k)
            results[f"Recall@{k}"] += recall_at_k(retrieved_ids, relevant, k)
        results["MRR"] += reciprocal_rank(retrieved_ids, relevant)

    elapsed = time.time() - start
    n = len(test_set)

    # 取平均
    for key in results:
        results[key] /= n

    results["name"] = name
    results["time"] = f"{elapsed:.1f}s"
    results["n_queries"] = n

    return results


def run_full_evaluation(test_set: List[Dict]) -> List[Dict]:
    """运行全部配置的评测"""

    # 加载检索器
    print("加载 Dense 检索器...")
    dense = DenseRetriever()
    dense.load()

    print("加载 BM25 检索器...")
    bm25 = BM25Retriever()
    bm25.load()

    all_results = []

    # 1. Dense only
    print("\n[1/4] 评测: Dense Only")
    r = evaluate_retriever(
        "Dense (BGE-small)",
        lambda q, k: dense.search(q, top_k=k),
        test_set,
    )
    all_results.append(r)

    # 2. BM25 only
    print("[2/4] 评测: BM25 Only")
    r = evaluate_retriever(
        "BM25",
        lambda q, k: bm25.search(q, top_k=k),
        test_set,
    )
    all_results.append(r)

    # 3. Union merge
    print("[3/4] 评测: Union Merge")
    def union_search(q, k):
        dr = dense.search(q, top_k=k)
        br = bm25.search(q, top_k=k)
        return union_merge(dr, br, top_k=k)

    r = evaluate_retriever("Union (Dense+BM25)", union_search, test_set)
    all_results.append(r)

    # 4. RRF fusion
    print("[4/4] 评测: RRF Fusion")
    def rrf_search(q, k):
        dr = dense.search(q, top_k=k)
        br = bm25.search(q, top_k=k)
        return weighted_fusion(dr, br, top_k=k)

    r = evaluate_retriever("RRF (Dense+BM25)", rrf_search, test_set)
    all_results.append(r)

    return all_results


def print_results(all_results: List[Dict]):
    """打印对比表格"""
    metrics = ["Hit@1", "Hit@5", "Hit@10", "Hit@20",
               "Recall@1", "Recall@5", "Recall@10", "Recall@20",
               "MRR", "time"]

    # 表头
    name_width = max(len(r["name"]) for r in all_results) + 2
    header = f"{'Method':<{name_width}}"
    for m in metrics:
        header += f" {m:>10}"
    print(header)
    print("-" * len(header))

    # 每行
    for r in all_results:
        row = f"{r['name']:<{name_width}}"
        for m in metrics:
            val = r.get(m, "")
            if isinstance(val, float):
                row += f" {val:>10.4f}"
            else:
                row += f" {val:>10}"
        print(row)


def save_results(all_results: List[Dict]):
    """保存评测结果"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "retrieval_eval.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {path}")


# ── CLI ──

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode == "build":
        # 仅构造评测集
        test_set = build_test_set(n_samples=200)
        save_test_set(test_set)

    elif mode == "eval":
        # 仅运行评测（需已有评测集）
        test_set = load_test_set()
        all_results = run_full_evaluation(test_set)
        print_results(all_results)
        save_results(all_results)

    elif mode == "all":
        # 构造评测集 + 运行评测
        test_set = build_test_set(n_samples=200)
        save_test_set(test_set)
        all_results = run_full_evaluation(test_set)
        print_results(all_results)
        save_results(all_results)

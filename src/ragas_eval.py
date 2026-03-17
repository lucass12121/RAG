"""
RAGAS 检索评测模块
- 使用 law_qa_cleaned.csv (519条) 作为测试集
- 评测指标: Context Precision, Context Recall (LLM-as-Judge)
- 支持: 多模型对比 / 单双路召回 / 合并策略 / Reranker
- 每种方法单独子进程运行，避免 event loop 问题
"""

import csv
import json
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

RESULTS_DIR = ROOT_DIR / "results"
RAW_DIR = ROOT_DIR / "data" / "raw"
TEST_CSV = RAW_DIR / "law_qa_cleaned.csv"


# ── 加载测试集 ──

def load_test_csv(path: Path = None) -> List[Dict]:
    if path is None:
        path = TEST_CSV
    test_set = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("question", "").strip()
            a = row.get("reply", "").strip()
            if q and a:
                test_set.append({"question": q, "ground_truth": a})
    print(f"加载测试集: {len(test_set)} 条")
    return test_set


# ── 单方法评测 (子进程入口) ──

def run_single_method(method: str, sample_n: int, top_k: int,
                      model_name: str = "", reranker_name: str = ""):
    """
    评测单个检索方法 — 作为独立进程运行

    method 格式:
        "Dense"          — 单路向量检索
        "BM25"           — 单路稀疏检索
        "Union"          — 双路取并集
        "RRF"            — 双路 RRF 融合
        "RRF+Rerank"     — 双路 RRF + Reranker

    model_name: embedding 模型名 (默认 BGE-small)
    reranker_name: reranker 模型名 (仅 method 含 Rerank 时有效)
    """
    import asyncio
    import nest_asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    nest_asyncio.apply(loop)

    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import context_precision, context_recall
    from retriever import DenseRetriever, BM25Retriever
    from merger import union_merge, weighted_fusion

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 默认模型
    if not model_name:
        model_name = "BAAI/bge-small-zh-v1.5"

    # 结果文件名：包含模型信息
    model_short = model_name.split("/")[-1]
    result_name = f"{method}_{model_short}"
    if reranker_name:
        reranker_short = reranker_name.split("/")[-1]
        result_name += f"_{reranker_short}"

    # 加载测试集
    test_set = load_test_csv()
    if sample_n > 0 and sample_n < len(test_set):
        import random
        random.seed(42)
        test_set = random.sample(test_set, sample_n)
        print(f"[{result_name}] 抽样 {sample_n} 条")
    else:
        print(f"[{result_name}] 全量 {len(test_set)} 条")

    # 加载检索器
    dense = None
    bm25 = None
    reranker = None

    use_dense = method in ("Dense", "Union", "RRF", "RRF+Rerank")
    use_bm25 = method in ("BM25", "Union", "RRF", "RRF+Rerank")
    use_rerank = "Rerank" in method

    if use_dense:
        print(f"加载 Dense 检索器 ({model_name})...")
        dense = DenseRetriever(model_name=model_name)
        dense.load()
    if use_bm25:
        print("加载 BM25 检索器...")
        bm25 = BM25Retriever()
        bm25.load()
    if use_rerank and reranker_name:
        from reranker import Reranker
        reranker = Reranker(model_name=reranker_name)
        reranker.load()

    # 定义搜索函数
    def search_fn(query, k):
        if method == "Dense":
            return dense.search(query, top_k=k)
        elif method == "BM25":
            return bm25.search(query, top_k=k)
        elif method == "Union":
            dr = dense.search(query, top_k=k)
            br = bm25.search(query, top_k=k)
            return union_merge(dr, br, top_k=k)
        elif method == "RRF":
            dr = dense.search(query, top_k=k)
            br = bm25.search(query, top_k=k)
            return weighted_fusion(dr, br, top_k=k)
        elif method == "RRF+Rerank":
            # 双路召回 Top20 → Rerank → Top K
            dr = dense.search(query, top_k=20)
            br = bm25.search(query, top_k=20)
            candidates = weighted_fusion(dr, br, top_k=20)
            return reranker.rerank(query, candidates, top_k=k)

    # 检索
    questions, contexts_list, answers, ground_truths = [], [], [], []
    print(f"检索中 ({len(test_set)} 条, top_k={top_k})...")
    t0 = time.time()

    for i, item in enumerate(test_set):
        results = search_fn(item["question"], top_k)
        questions.append(item["question"])
        contexts_list.append([r["text"] for r in results])
        answers.append("暂无生成回答")
        ground_truths.append(item["ground_truth"])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(test_set)}")

    print(f"检索完成: {time.time() - t0:.1f}s")

    ds = Dataset.from_dict({
        "question": questions,
        "contexts": contexts_list,
        "answer": answers,
        "ground_truth": ground_truths,
    })

    # RAGAS 评测
    print(f"RAGAS 评测中 (调用 OpenAI API)...")
    t0 = time.time()
    result = ragas_evaluate(ds, metrics=[context_precision, context_recall])
    elapsed = time.time() - t0
    print(f"RAGAS 评测完成: {elapsed:.1f}s")

    scores = {
        "name": result_name,
        "method": method,
        "model": model_name,
        "reranker": reranker_name,
        "n_queries": len(test_set),
        "top_k": top_k,
        "context_precision": result["context_precision"],
        "context_recall": result["context_recall"],
        "time": f"{elapsed:.1f}s",
    }
    print(f"Context Precision: {scores['context_precision']:.4f}")
    print(f"Context Recall:    {scores['context_recall']:.4f}")

    # 保存详细结果
    detail_df = result.to_pandas()
    detail_path = RESULTS_DIR / f"ragas_detail_{result_name}.csv"
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    # 保存汇总
    score_path = RESULTS_DIR / f"ragas_{result_name}.json"
    with open(score_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {score_path}")

    return scores


# ── 汇总结果 ──

def collect_results(pattern: str = "ragas_*.json") -> List[Dict]:
    """收集所有匹配的结果文件"""
    import glob
    all_scores = []
    for path in sorted(RESULTS_DIR.glob(pattern)):
        if "detail" in path.name or "summary" in path.name:
            continue
        with open(path, "r", encoding="utf-8") as f:
            all_scores.append(json.load(f))
    return all_scores


def print_results(all_scores: List[Dict]):
    if not all_scores:
        print("没有找到任何结果")
        return

    print("\n" + "=" * 80)
    print("RAGAS 检索评测汇总")
    print("=" * 80)
    print(f"{'Name':<40} {'CtxPrecision':>14} {'CtxRecall':>12} {'Time':>10}")
    print("-" * 80)
    for s in all_scores:
        print(f"{s['name']:<40} {s['context_precision']:>14.4f} {s['context_recall']:>12.4f} {s['time']:>10}")

    summary_path = RESULTS_DIR / "ragas_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_scores, f, ensure_ascii=False, indent=2)
    print(f"\n汇总结果: {summary_path}")


# ── CLI ──

if __name__ == "__main__":
    """
    用法:
      # 单个方法评测
      python ragas_eval.py single <method> [sample_n] [top_k] [--model MODEL] [--reranker RERANKER]

      # 批量评测预设方案
      python ragas_eval.py batch <plan> [sample_n] [top_k]

      # 汇总已有结果
      python ragas_eval.py summary

    Plans:
      base       — BGE-small: Dense/BM25/Union/RRF (已在本地跑过)
      models     — 多模型对比: BGE-small/BGE-large/Qwen3-Embedding Dense
      reranker   — Reranker 对比: RRF vs RRF+Rerank (多个 reranker 模型)
      full       — 全部实验

    示例:
      python ragas_eval.py batch base 50 10
      python ragas_eval.py batch models 50 10
      python ragas_eval.py batch reranker 50 10
      python ragas_eval.py single Dense 50 10 --model BAAI/bge-large-zh-v1.5
      python ragas_eval.py single RRF+Rerank 50 10 --reranker BAAI/bge-reranker-v2-m3
    """

    def parse_args():
        args = sys.argv[1:]
        model_name = ""
        reranker_name = ""
        cleaned = []
        i = 0
        while i < len(args):
            if args[i] == "--model" and i + 1 < len(args):
                model_name = args[i + 1]
                i += 2
            elif args[i] == "--reranker" and i + 1 < len(args):
                reranker_name = args[i + 1]
                i += 2
            else:
                cleaned.append(args[i])
                i += 1
        return cleaned, model_name, reranker_name

    def run_subprocess(method, sample_n, top_k, model_name="", reranker_name=""):
        cmd = [
            sys.executable, __file__,
            "single", method, str(sample_n), str(top_k),
        ]
        if model_name:
            cmd += ["--model", model_name]
        if reranker_name:
            cmd += ["--reranker", reranker_name]
        proc = subprocess.run(cmd, cwd=str(ROOT_DIR / "src"))
        if proc.returncode != 0:
            print(f"[错误] {method} 评测失败 (returncode={proc.returncode})")

    args, model_name, reranker_name = parse_args()
    action = args[0] if args else "summary"

    if action == "single":
        method = args[1]
        sample_n = int(args[2]) if len(args) > 2 else 50
        top_k = int(args[3]) if len(args) > 3 else 10
        run_single_method(method, sample_n, top_k,
                          model_name=model_name, reranker_name=reranker_name)

    elif action == "batch":
        plan = args[1] if len(args) > 1 else "base"
        sample_n = int(args[2]) if len(args) > 2 else 50
        top_k = int(args[3]) if len(args) > 3 else 10

        print(f"配置: plan={plan}, sample_n={sample_n}, top_k={top_k}")
        print(f"OPENAI_API_KEY: {'已设置' if os.getenv('OPENAI_API_KEY') else '未设置!'}")
        print()

        # 定义实验方案
        experiments = []

        if plan in ("base", "full"):
            # 实验1-3: BGE-small 的 Dense/BM25/Union/RRF
            for m in ["Dense", "BM25", "Union", "RRF"]:
                experiments.append({
                    "method": m, "model": "BAAI/bge-small-zh-v1.5", "reranker": ""
                })

        if plan in ("models", "full"):
            # 实验4: 多模型对比 (Dense 单路)
            for model in [
                "BAAI/bge-small-zh-v1.5",
                "BAAI/bge-large-zh-v1.5",
                "Qwen/Qwen3-Embedding-0.6B",
            ]:
                experiments.append({
                    "method": "Dense", "model": model, "reranker": ""
                })

        if plan in ("reranker", "full"):
            # 实验5: Reranker 对比
            # 无 reranker 基线
            experiments.append({
                "method": "RRF", "model": "BAAI/bge-small-zh-v1.5", "reranker": ""
            })
            # 不同 reranker
            for rr in [
                "BAAI/bge-reranker-v2-m3",
                "BAAI/bge-reranker-large",
            ]:
                experiments.append({
                    "method": "RRF+Rerank",
                    "model": "BAAI/bge-small-zh-v1.5",
                    "reranker": rr,
                })

        # 去重
        seen = set()
        unique_experiments = []
        for exp in experiments:
            key = (exp["method"], exp["model"], exp["reranker"])
            if key not in seen:
                seen.add(key)
                unique_experiments.append(exp)

        print(f"共 {len(unique_experiments)} 组实验:")
        for i, exp in enumerate(unique_experiments):
            desc = f"{exp['method']} | {exp['model'].split('/')[-1]}"
            if exp['reranker']:
                desc += f" | reranker={exp['reranker'].split('/')[-1]}"
            print(f"  [{i+1}] {desc}")
        print()

        # 依次运行
        for i, exp in enumerate(unique_experiments):
            print(f"{'='*60}")
            print(f"[{i+1}/{len(unique_experiments)}] {exp['method']} | {exp['model'].split('/')[-1]}")
            print(f"{'='*60}")
            run_subprocess(
                exp["method"], sample_n, top_k,
                model_name=exp["model"], reranker_name=exp["reranker"],
            )
            print()

        # 汇总
        all_scores = collect_results()
        print_results(all_scores)

    elif action == "summary":
        all_scores = collect_results()
        print_results(all_scores)

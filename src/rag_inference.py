"""
RAG 推理模块 (Step 3-6)
- 基于检索结果 + Qwen2.5 3B/7B 生成回答
- 评测: RAGAS answer_relevancy / 人工对比
- 支持: 有RAG vs 无RAG 对比
"""

import csv
import json
import time
import sys
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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


# ── 检索 ──

def retrieve(query: str, top_k: int = 10, method: str = "RRF+Rerank",
             dense=None, bm25=None, reranker=None):
    """使用指定方法检索"""
    from merger import weighted_fusion

    if method == "Dense":
        return dense.search(query, top_k=top_k)
    elif method == "BM25":
        return bm25.search(query, top_k=top_k)
    elif method == "RRF":
        dr = dense.search(query, top_k=top_k)
        br = bm25.search(query, top_k=top_k)
        return weighted_fusion(dr, br, top_k=top_k)
    elif method == "RRF+Rerank":
        dr = dense.search(query, top_k=20)
        br = bm25.search(query, top_k=20)
        candidates = weighted_fusion(dr, br, top_k=20)
        return reranker.rerank(query, candidates, top_k=top_k)
    else:
        raise ValueError(f"未知方法: {method}")


# ── 构造 Prompt ──

def build_prompt(question: str, contexts: List[str] = None) -> str:
    """构造 RAG prompt"""
    if contexts:
        context_text = "\n\n".join([f"[参考文档{i+1}] {c}" for i, c in enumerate(contexts)])
        prompt = (
            f"你是一个法律问答助手。请根据以下参考文档回答用户的问题。"
            f"如果参考文档中没有相关信息，请基于你的知识回答。\n\n"
            f"参考文档:\n{context_text}\n\n"
            f"问题: {question}\n\n"
            f"回答:"
        )
    else:
        # 无 RAG，直接回答
        prompt = (
            f"你是一个法律问答助手。请回答以下法律问题。\n\n"
            f"问题: {question}\n\n"
            f"回答:"
        )
    return prompt


# ── 加载 Qwen2.5 模型 ──

def load_qwen_model(model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
    """加载 Qwen2.5 模型"""
    print(f"[LLM] 加载模型: {model_name}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    print(f"[LLM] 加载完成: {time.time() - t0:.1f}s")
    return tokenizer, model


def generate_answer(tokenizer, model, prompt: str, max_new_tokens: int = 512) -> str:
    """使用 Qwen2.5 生成回答"""
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )

    # 只取生成的部分
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return answer


# ── 主流程 ──

def run_inference(
    llm_model: str = "Qwen/Qwen2.5-3B-Instruct",
    retrieval_method: str = "RRF+Rerank",
    reranker_name: str = "BAAI/bge-reranker-v2-m3",
    embedding_model: str = "BAAI/bge-small-zh-v1.5",
    use_rag: bool = True,
    sample_n: int = 50,
    top_k: int = 5,
):
    """RAG 推理主流程"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    llm_short = llm_model.split("/")[-1]
    tag = f"{llm_short}_{'RAG_' + retrieval_method if use_rag else 'noRAG'}"
    print(f"\n{'='*60}")
    print(f"实验: {tag}")
    print(f"{'='*60}")

    # 加载测试集
    test_set = load_test_csv()
    if 0 < sample_n < len(test_set):
        import random
        random.seed(42)
        test_set = random.sample(test_set, sample_n)
        print(f"抽样 {sample_n} 条")

    # 加载检索器 (仅 RAG 模式)
    dense, bm25, reranker_obj = None, None, None
    if use_rag:
        from retriever import DenseRetriever, BM25Retriever

        dense = DenseRetriever(model_name=embedding_model)
        dense.load()
        bm25 = BM25Retriever()
        bm25.load()

        if "Rerank" in retrieval_method:
            from reranker import Reranker
            reranker_obj = Reranker(model_name=reranker_name)
            reranker_obj.load()

    # 加载 LLM
    tokenizer, model = load_qwen_model(llm_model)

    # 推理
    results = []
    print(f"\n推理中 ({len(test_set)} 条)...")
    t0 = time.time()

    for i, item in enumerate(test_set):
        q = item["question"]

        # 检索
        contexts = []
        if use_rag:
            docs = retrieve(q, top_k=top_k, method=retrieval_method,
                          dense=dense, bm25=bm25, reranker=reranker_obj)
            contexts = [d["text"] for d in docs]

        # 生成
        prompt = build_prompt(q, contexts if use_rag else None)
        answer = generate_answer(tokenizer, model, prompt)

        results.append({
            "question": q,
            "ground_truth": item["ground_truth"],
            "generated_answer": answer,
            "contexts": contexts,
            "use_rag": use_rag,
        })

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(test_set)} | 最近回答: {answer[:50]}...")

    elapsed = time.time() - t0
    print(f"推理完成: {elapsed:.1f}s")

    # 保存结果
    output_path = RESULTS_DIR / f"inference_{tag}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "llm_model": llm_model,
                "retrieval_method": retrieval_method,
                "use_rag": use_rag,
                "sample_n": len(test_set),
                "top_k": top_k,
            },
            "time": f"{elapsed:.1f}s",
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {output_path}")

    # 打印几个样例
    print(f"\n{'='*60}")
    print("样例展示 (前3条)")
    print(f"{'='*60}")
    for r in results[:3]:
        print(f"\n问题: {r['question']}")
        print(f"生成回答: {r['generated_answer'][:200]}")
        print(f"标准答案: {r['ground_truth'][:200]}")
        print("-" * 40)

    return results


# ── CLI ──

if __name__ == "__main__":
    """
    用法:
      # Qwen2.5-3B 有RAG
      python rag_inference.py --llm Qwen/Qwen2.5-3B-Instruct --rag --sample 50

      # Qwen2.5-7B 有RAG
      python rag_inference.py --llm Qwen/Qwen2.5-7B-Instruct --rag --sample 50

      # Qwen2.5-3B 无RAG (baseline)
      python rag_inference.py --llm Qwen/Qwen2.5-3B-Instruct --no-rag --sample 50

      # 批量: 3B+7B, 有RAG+无RAG
      python rag_inference.py --batch --sample 50
    """
    import argparse

    parser = argparse.ArgumentParser(description="RAG 推理")
    parser.add_argument("--llm", default="Qwen/Qwen2.5-3B-Instruct",
                       help="LLM 模型名")
    parser.add_argument("--rag", action="store_true", default=False,
                       help="启用 RAG")
    parser.add_argument("--no-rag", action="store_true", default=False,
                       help="禁用 RAG (baseline)")
    parser.add_argument("--method", default="RRF+Rerank",
                       help="检索方法")
    parser.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3",
                       help="Reranker 模型")
    parser.add_argument("--embedding", default="BAAI/bge-small-zh-v1.5",
                       help="Embedding 模型")
    parser.add_argument("--sample", type=int, default=50,
                       help="抽样数量")
    parser.add_argument("--top-k", type=int, default=5,
                       help="检索 Top K")
    parser.add_argument("--batch", action="store_true",
                       help="批量运行: 3B/7B × 有RAG/无RAG")
    args = parser.parse_args()

    if args.batch:
        # 批量: 4组实验
        experiments = [
            {"llm": "Qwen/Qwen2.5-3B-Instruct", "use_rag": False},
            {"llm": "Qwen/Qwen2.5-3B-Instruct", "use_rag": True},
            {"llm": "Qwen/Qwen2.5-7B-Instruct", "use_rag": False},
            {"llm": "Qwen/Qwen2.5-7B-Instruct", "use_rag": True},
        ]
        print(f"批量模式: {len(experiments)} 组实验")
        for i, exp in enumerate(experiments):
            print(f"\n[{i+1}/{len(experiments)}]")
            run_inference(
                llm_model=exp["llm"],
                retrieval_method=args.method,
                reranker_name=args.reranker,
                embedding_model=args.embedding,
                use_rag=exp["use_rag"],
                sample_n=args.sample,
                top_k=args.top_k,
            )
    else:
        use_rag = args.rag or (not args.no_rag)
        run_inference(
            llm_model=args.llm,
            retrieval_method=args.method,
            reranker_name=args.reranker,
            embedding_model=args.embedding,
            use_rag=use_rag,
            sample_n=args.sample,
            top_k=args.top_k,
        )

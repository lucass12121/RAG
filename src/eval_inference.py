"""
推理结果评测
- 读取 inference_*.json，评测生成回答质量
- 指标:
    - precision (语义一致性准确率): GPT 二元判断回答方向是否一致
    - answer_similarity (语义相似度): RAGAS embedding 余弦相似度
    - answer_relevancy (回答相关性): RAGAS LLM 评分
- 对比: 有RAG vs 无RAG, 3B vs 7B
"""

import json
import os
import sys
import time
from pathlib import Path

import nest_asyncio
nest_asyncio.apply()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, answer_similarity
from openai import OpenAI
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
RESULTS_DIR = ROOT_DIR / "results"


# ── Precision: GPT 语义一致性判断 ──

EVAL_PROMPT = """你是法律问答评测员。判断模型回答与标准答案在法律层面是否基本一致。

判为"一致"：
- 核心结论方向相同（如都说"可以"、都说"违法"、都说"需要满足条件"）
- 涉及的法律领域和处理方向正确，即使细节详略不同
- 引用不同法条但结论相同

判为"不一致"：
- 结论方向相反（"可以"vs"不可以"、"合法"vs"违法"）
- 答非所问，回答了不同的问题
- 给出了错误的责任主体或管辖机关（如标准答案说"卫生部门"，模型说"公安机关"）
- 法律定性错误（如标准答案说"民事责任"，模型说"刑事责任"）
- 模型回答完全是泛泛而谈的套话，没有针对问题给出任何具体的法律信息

注意：如果模型回答方向正确但不够详细（如只提到了部分条件或部分步骤），仍应判为"一致"。

【问题】：{question}
【标准答案】：{reference}
【模型回答】：{answer}

只输出"一致"或"不一致"。"""


def eval_precision(results: list, batch_size: int = 5) -> float:
    """用 GPT 判断每条回答与标准答案的语义一致性，返回准确率"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    correct = 0
    total = 0

    for i, r in enumerate(results):
        question = r["question"]
        reference = r["ground_truth"]
        answer = r["generated_answer"]

        if not answer or answer in ("TIMEOUT", "FAIL"):
            continue

        prompt = EVAL_PROMPT.format(
            question=question, reference=reference, answer=answer
        )

        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=10,
                )
                text = resp.choices[0].message.content.strip()
                if "不一致" in text:
                    total += 1
                    break
                elif "一致" in text:
                    correct += 1
                    total += 1
                    break
                else:
                    if attempt < 2:
                        time.sleep(1)
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    total += 1  # 失败算不一致

        if (i + 1) % 50 == 0:
            acc = correct / total if total > 0 else 0
            print(f"  Precision 进度: {i+1}/{len(results)}, 当前准确率={acc:.4f}")

    precision = correct / total if total > 0 else 0
    print(f"  Precision 完成: {correct}/{total} = {precision:.4f}")
    return precision


# ── 主评测函数 ──

def eval_one(json_path: Path):
    """评测单个推理结果文件"""
    print(f"\n{'='*60}")
    print(f"评测: {json_path.name}")
    print(f"{'='*60}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    config = data["config"]
    results = data["results"]
    use_rag = config["use_rag"]

    # 构造 RAGAS 数据集
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    for r in results:
        questions.append(r["question"])
        answers.append(r["generated_answer"])
        ctx = r.get("contexts", [])
        if not ctx:
            ctx = ["无参考文档"]
        contexts_list.append(ctx)
        ground_truths.append(r["ground_truth"])

    ds = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

    # ── 1. Precision (GPT 语义一致性) ──
    print(f"\n[1/2] Precision 评测 (GPT 语义一致性判断)...")
    precision = eval_precision(results)

    # ── 2. RAGAS 指标 (Similarity + Relevancy) ──
    print(f"\n[2/2] RAGAS 评测 (Similarity + Relevancy)...")
    if use_rag:
        metrics = [answer_relevancy, faithfulness, answer_similarity]
    else:
        metrics = [answer_relevancy, answer_similarity]

    print(f"样本数: {len(questions)}")
    print(f"指标: {[m.name for m in metrics]}")

    ragas_result = evaluate(ds, metrics=metrics)

    # 汇总所有分数
    scores = {"precision": round(precision, 4)}
    for m in metrics:
        val = ragas_result[m.name]
        scores[m.name] = round(val, 4)

    print(f"\n结果:")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # 保存
    output = {
        "file": json_path.name,
        "config": config,
        "scores": scores,
    }
    out_path = RESULTS_DIR / f"eval_{json_path.stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, ensure_ascii=False, indent=2, fp=f)
    print(f"已保存: {out_path}")

    return output


def eval_all():
    """评测所有推理结果并汇总"""
    files = sorted(RESULTS_DIR.glob("inference_*.json"))
    if not files:
        print("未找到推理结果文件")
        return

    print(f"找到 {len(files)} 个推理结果文件")
    all_results = []

    for f in files:
        r = eval_one(f)
        all_results.append(r)

    # 汇总表
    print(f"\n{'='*85}")
    print("汇总")
    print(f"{'='*85}")
    print(f"{'配置':<40} {'Precision':>10} {'Relevancy':>10} {'Similarity':>11} {'Faithful':>10}")
    print("-" * 85)
    for r in all_results:
        llm = r["config"]["llm_model"].split("/")[-1]
        rag = "RAG" if r["config"]["use_rag"] else "noRAG"
        name = f"{llm} + {rag}"
        prec = r["scores"].get("precision", "-")
        rel = r["scores"].get("answer_relevancy", "-")
        sim = r["scores"].get("answer_similarity", "-")
        faith = r["scores"].get("faithfulness", "-")
        prec_s = f"{prec:.4f}" if isinstance(prec, float) else prec
        rel_s = f"{rel:.4f}" if isinstance(rel, float) else rel
        sim_s = f"{sim:.4f}" if isinstance(sim, float) else sim
        faith_s = f"{faith:.4f}" if isinstance(faith, float) else faith
        print(f"{name:<40} {prec_s:>10} {rel_s:>10} {sim_s:>11} {faith_s:>10}")

    # 保存汇总
    summary_path = RESULTS_DIR / "eval_inference_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, ensure_ascii=False, indent=2, fp=f)
    print(f"\n汇总已保存: {summary_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if not p.exists():
            p = RESULTS_DIR / sys.argv[1]
        eval_one(p)
    else:
        eval_all()

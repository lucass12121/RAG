# -*- coding: utf-8 -*-
"""DPO 1-10分评测 (GPT-4o-mini)"""
import json, os, time, sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SCORING_PROMPT = """你是法律问答评测助手。请对模型回答进行1-10分评分。

评分标准：
10分：结论完全正确，覆盖了标准答案中的全部关键要点
9分：结论正确，覆盖了大部分关键要点，仅有少量遗漏
8分：结论正确，覆盖了标准答案中的核心要点
7分：结论方向正确，但有部分要点遗漏或表述不够准确
6分：结论大致正确，但遗漏较多或有细节偏差
5分：回答部分正确，但存在明显不足
4分：回答涉及相关领域但结论有偏差
3分：回答基本不正确或严重偏题
2分：回答与问题几乎无关
1分：完全错误或拒绝回答

注意：法条编号引用不同不扣分，重点评判结论和要点覆盖度。

【问题】：{question}
【标准答案】：{reference}
【模型回答】：{answer}

只输出一个1-10的整数分数。"""


def score_one(q, ref, ans):
    prompt = SCORING_PROMPT.format(question=q, reference=ref, answer=ans)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=5,
            )
            text = resp.choices[0].message.content.strip()
            for c in text:
                if c.isdigit():
                    score = int(c)
                    if text.startswith("10"):
                        return 10
                    return max(1, min(10, score))
            return -1
        except Exception:
            time.sleep(2 ** attempt)
    return -1


def run_scoring(inference_path, references, label):
    with open(inference_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = len(data)
    scores = [None] * n

    def process(i):
        r = data[i]
        ref = references.get(r["question"], "")
        return (i, score_one(r["question"], ref, r["answer"]))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=15) as ex:
        futs = {ex.submit(process, i): i for i in range(n)}
        done = 0
        for fut in as_completed(futs):
            idx, s = fut.result()
            scores[idx] = s
            done += 1
            if done % 100 == 0:
                valid = [x for x in scores if x is not None and x > 0]
                avg = sum(valid) / len(valid) if valid else 0
                print(f"  {label} [{done}/{n}] avg={avg:.4f}", flush=True)

    valid_scores = [x for x in scores if x is not None and x > 0]
    avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    elapsed = time.time() - t0
    print(f"  {label} 完成: avg={avg:.4f} ({elapsed:.0f}s)", flush=True)
    return {"avg_score": round(avg, 4), "precision_pct": round(avg * 10, 2), "n": len(valid_scores), "scores": scores}


if __name__ == "__main__":
    results_dir = ROOT / "results"

    # Load references
    references = {}
    for p in [ROOT / "data" / "test_qa_519.json", ROOT / "data" / "test_queries.json"]:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            for obj in data:
                q = obj.get("question", "")
                a = obj.get("reply", obj.get("answer", ""))
                if q and a:
                    references[q] = a
            break

    if not references:
        test_path = ROOT / "data" / "processed" / "test_queries.jsonl"
        if test_path.exists():
            with open(test_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        q = obj.get("question", obj.get("query", ""))
                        a = obj.get("reply", obj.get("answer", ""))
                        if q and a:
                            references[q] = a

    print(f"Loaded {len(references)} references")

    # Eval DPO
    target = sys.argv[1] if len(sys.argv) > 1 else "inference_Qwen2.5-7B-DPO-2ep_noRAG.json"
    target_path = results_dir / target
    label = target.replace("inference_", "").replace("_noRAG.json", "").replace(".json", "")

    print(f"\n{'='*50}")
    print(f"1-10 Scoring: {label}")
    print(f"{'='*50}")

    result = run_scoring(target_path, references, label)

    output = {
        "eval_model": "gpt-4o-mini",
        "method": "scoring_1to10",
        label: result,
    }
    out_path = results_dir / f"precision_scoring_dpo_{label}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, ensure_ascii=False, indent=2, fp=f)
    print(f"\nSaved: {out_path}")
    print(f"Precision: {result['precision_pct']}%")

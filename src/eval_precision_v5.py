# -*- coding: utf-8 -*-
"""Precision 评测 V5: GPT-4o-mini, 宽松核心结论判断"""
import json, os, time, sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROMPT = """你是法律问答评测助手。请判断模型的回答是否正确回答了用户的法律问题。

评判标准：
- 核心结论方向与标准答案一致即可判正确（如法律定性、权利义务归属、处理方向等一致）
- 细节详略不同、表述方式不同不影响判断
- 法条编号引用不同或有偏差属于瑕疵，不影响正确性判断
- 只有当核心结论方向完全相反，或完全答非所问时，才判错误

【问题】：{question}
【标准答案】：{reference}
【模型回答】：{answer}

只输出"正确"或"错误"两个字。"""


def eval_one(q, ref, ans):
    prompt = PROMPT.format(question=q, reference=ref, answer=ans)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            time.sleep(2 ** attempt)
    return "ERROR"


def run_eval(json_path, label):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data["results"]
    n = len(results)
    judgments = [None] * n

    def process(i):
        r = results[i]
        return (i, eval_one(r["question"], r["ground_truth"], r["generated_answer"]))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=15) as ex:
        futs = {ex.submit(process, i): i for i in range(n)}
        done = 0
        for fut in as_completed(futs):
            idx, j = fut.result()
            judgments[idx] = j
            done += 1
            if done % 100 == 0:
                c = sum(1 for x in judgments if x and "正确" in x and "错误" not in x)
                valid = sum(1 for x in judgments if x is not None)
                print(f"  {label} [{done}/{n}] 当前: {c}/{valid} = {c/valid:.4f}")

    correct = sum(1 for x in judgments if x and "正确" in x and "错误" not in x)
    total = sum(1 for x in judgments if x and x != "ERROR")
    elapsed = time.time() - t0
    print(f"  {label} 完成: {correct}/{total} = {correct/total:.4f} ({elapsed:.0f}s)")
    return {"correct": correct, "total": total, "precision": round(correct / total, 4), "judgments": judgments}


if __name__ == "__main__":
    results_dir = ROOT / "results"

    sft_path = results_dir / "inference_Qwen2.5-7B-SFTv2_noRAG.json"
    base_path = results_dir / "inference_Qwen2.5-7B-Instruct_noRAG.json"

    print("=" * 50)
    print("Precision V5 评测 (GPT-4o-mini)")
    print("=" * 50)

    print("\n[1/2] SFT v2:")
    sft_res = run_eval(sft_path, "SFT v2")

    print("\n[2/2] Base:")
    base_res = run_eval(base_path, "Base")

    # Compare
    sft_j = sft_res["judgments"]
    base_j = base_res["judgments"]
    diff = sw = bw = 0
    for i in range(len(sft_j)):
        s_ok = sft_j[i] and "正确" in sft_j[i] and "错误" not in sft_j[i]
        b_ok = base_j[i] and "正确" in base_j[i] and "错误" not in base_j[i]
        if s_ok != b_ok:
            diff += 1
            if s_ok:
                sw += 1
            else:
                bw += 1

    print(f"\n{'='*50}")
    print(f"SFT v2: {sft_res['precision']:.4f}")
    print(f"Base:   {base_res['precision']:.4f}")
    print(f"判断不同: {diff}条 (SFT赢{sw}, Base赢{bw})")

    output = {
        "eval_model": "gpt-4o-mini",
        "prompt_version": "v5",
        "sft_v2": sft_res,
        "base": base_res,
    }
    out_path = results_dir / "precision_mini_v5.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, ensure_ascii=False, indent=2, fp=f)
    print(f"已保存: {out_path}")

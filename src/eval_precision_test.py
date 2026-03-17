"""
测试新 Precision prompt - 只跑 3B noRAG 一组
"""
import json
import os
import sys
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
RESULTS_DIR = ROOT_DIR / "results"

EVAL_PROMPT_STRICT = """你是法律问答评测员。判断模型回答与标准答案在法律层面是否基本一致。

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


def eval_precision_strict(results: list) -> float:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    correct = 0
    total = 0

    for i, r in enumerate(results):
        question = r["question"]
        reference = r["ground_truth"]
        answer = r["generated_answer"]

        if not answer or answer in ("TIMEOUT", "FAIL"):
            continue

        prompt = EVAL_PROMPT_STRICT.format(
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
                    total += 1

        if (i + 1) % 50 == 0:
            acc = correct / total if total > 0 else 0
            print(f"  进度: {i+1}/{len(results)}, 当前准确率={acc:.4f}")

    precision = correct / total if total > 0 else 0
    print(f"\n  最终: {correct}/{total} = {precision:.4f}")
    return precision


if __name__ == "__main__":
    json_path = RESULTS_DIR / "inference_Qwen2.5-3B-Instruct_noRAG.json"
    print(f"测试严格 Precision prompt")
    print(f"文件: {json_path.name}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    print(f"样本数: {len(results)}\n")

    precision = eval_precision_strict(results)
    print(f"\n旧 prompt (宽松): 0.9056")
    print(f"上一版 (过严): 0.4143")
    print(f"本版 prompt: {precision:.4f}")

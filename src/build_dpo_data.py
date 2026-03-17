# -*- coding: utf-8 -*-
"""构建 DPO 偏好数据集: GPT-4o (chosen) vs SFT数据 (rejected)"""
import json
import os
import time
import sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_PATH = ROOT / "data" / "dpo_questions_4k.json"
OUTPUT_PATH = ROOT / "data" / "dpo_legal_4k.json"
CHECKPOINT_PATH = ROOT / "data" / "dpo_checkpoint.json"

SYSTEM_PROMPT = "你是一个专业的法律问答助手。请根据中国现行法律法规，给出准确、详细、专业的法律回答。回答应包含相关法律依据、适用条件和实际操作建议。"

MAX_WORKERS = 20


def generate_chosen(question):
    """用 GPT-4o 生成高质量回答作为 chosen"""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=0.3,
                max_tokens=1500,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return None
    return None


def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(done_map):
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(done_map, ensure_ascii=False, fp=f)


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"总问题数: {len(questions)}")

    # Load checkpoint
    done_map = load_checkpoint()
    print(f"已完成: {len(done_map)}")

    # Find remaining
    remaining = [(i, q) for i, q in enumerate(questions) if str(i) not in done_map]
    print(f"待处理: {len(remaining)}")

    if not remaining:
        print("全部完成，生成最终文件")
    else:
        t0 = time.time()
        completed = 0

        def process(idx, q):
            chosen = generate_chosen(q["instruction"])
            return (idx, chosen)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(process, i, q): (i, q) for i, q in remaining}
            for fut in as_completed(futs):
                idx, chosen = fut.result()
                if chosen:
                    done_map[str(idx)] = chosen
                completed += 1

                if completed % 100 == 0:
                    elapsed = time.time() - t0
                    speed = completed / elapsed
                    eta = (len(remaining) - completed) / speed
                    print(f"  [{completed}/{len(remaining)}] {speed:.1f}/s, ETA {eta/60:.1f}min")
                    save_checkpoint(done_map)

        save_checkpoint(done_map)
        print(f"GPT-4o 生成完成: {len(done_map)}/{len(questions)}")

    # Build final DPO dataset
    dpo_data = []
    for i, q in enumerate(questions):
        if str(i) in done_map:
            dpo_data.append({
                "instruction": q["instruction"],
                "input": q.get("input", ""),
                "chosen": done_map[str(i)],
                "rejected": q["output"],
            })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dpo_data, ensure_ascii=False, indent=2, fp=f)

    print(f"\nDPO 数据集已保存: {OUTPUT_PATH}")
    print(f"总条数: {len(dpo_data)}")
    chosen_lens = [len(d["chosen"]) for d in dpo_data]
    rejected_lens = [len(d["rejected"]) for d in dpo_data]
    print(f"chosen 平均长度: {sum(chosen_lens)/len(chosen_lens):.0f}")
    print(f"rejected 平均长度: {sum(rejected_lens)/len(rejected_lens):.0f}")


if __name__ == "__main__":
    main()

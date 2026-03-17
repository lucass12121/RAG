"""
SFT 数据集扩写 — 对法律相关短回答进行扩写
- 筛选: 法律相关 + output < 500 字
- 扩写: GPT-4o-mini, ThreadPool 10 并发
- 质量保障: 扩写后必须 >= 原长度 * 1.3 且 >= 200 字, 否则重试
- 断点续跑: 已扩写的跳过
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DATA_PATH = ROOT_DIR / "data" / "sft_all_90k.json"
OUTPUT_PATH = ROOT_DIR / "data" / "sft_all_90k_expanded.json"
PROGRESS_PATH = ROOT_DIR / "data" / "expand_progress.json"

LEGAL_KW = ['法', '条', '规定', '合同', '责任', '权利', '义务', '诉讼',
            '赔偿', '刑', '民', '侵权', '犯罪', '法院', '法律']

EXPAND_PROMPT = """你是法律领域的专业写作助手。请对以下法律问答中的回答进行扩写，使其更加专业、完整、有参考价值。

扩写要求：
1. 保持原回答的核心结论和方向不变
2. 补充相关的法律依据（如具体法律名称、条款编号）
3. 补充适用条件、例外情形或实际操作要点
4. 扩写后长度控制在 400-800 字之间
5. 语言专业但通俗易懂，适合法律咨询场景
6. 不要编造不存在的法律条文

【问题】：{instruction}
{input_text}
【原始回答】：{output}

请直接输出扩写后的回答，不要输出任何前缀或解释。"""

MAX_WORKERS = 50


def is_legal(entry: dict) -> bool:
    text = entry['instruction'] + entry['output']
    return any(kw in text for kw in LEGAL_KW)


def expand_one(client: OpenAI, entry: dict, idx: int,
               max_retries: int = 3) -> tuple:
    """扩写单条，带重试和质量校验。返回 (idx, expanded_text or None)"""
    input_text = f"【补充信息】：{entry['input']}" if entry.get('input') else ""
    prompt = EXPAND_PROMPT.format(
        instruction=entry['instruction'],
        input_text=input_text,
        output=entry['output'],
    )
    orig_len = len(entry['output'])
    expanded = None

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=1200,
            )
            expanded = resp.choices[0].message.content.strip()

            # 质量校验: 扩写后至少是原长度的 1.3 倍，且不少于 200 字
            if len(expanded) >= max(orig_len * 1.3, 200):
                return (idx, expanded)
            # 不合格则重试
            if attempt < max_retries - 1:
                time.sleep(0.5)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [FAIL] idx={idx}: {e}", flush=True)
                return (idx, None)

    # 重试耗尽，返回最后一次结果
    return (idx, expanded)


def main():
    # 加载数据
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"总数据: {len(data)} 条", flush=True)

    # 筛选需要扩写的条目
    targets = []
    for i, d in enumerate(data):
        if is_legal(d) and len(d['output']) < 500:
            targets.append(i)
    print(f"需要扩写: {len(targets)} 条 (法律相关 + output < 500 字)", flush=True)

    # 加载断点进度
    progress = {}
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            progress = json.load(f)
        print(f"已有进度: {len(progress)} 条已完成", flush=True)

    # 过滤掉已完成的
    todo = [i for i in targets if str(i) not in progress]
    print(f"本次需处理: {len(todo)} 条", flush=True)

    if not todo:
        print("全部已完成，直接生成最终文件", flush=True)
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        t0 = time.time()
        completed = 0
        total_todo = len(todo)

        # 分批处理，每批 200 条，用线程池并发
        batch_size = 200
        for batch_start in range(0, total_todo, batch_size):
            batch = todo[batch_start:batch_start + batch_size]

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(expand_one, client, data[idx], idx): idx
                    for idx in batch
                }
                for future in as_completed(futures):
                    idx, result = future.result()
                    if result is not None:
                        progress[str(idx)] = result
                    completed += 1

            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total_todo - completed) / rate if rate > 0 else 0
            print(f"  进度: {completed}/{total_todo}, "
                  f"耗时 {elapsed:.0f}s, ETA {eta:.0f}s, "
                  f"成功 {len(progress)}/{len(targets)}",
                  flush=True)

            # 每批保存进度
            with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
                json.dump(progress, f, ensure_ascii=False)

    # 生成最终数据集
    print("\n生成最终数据集...", flush=True)
    expanded_count = 0
    kept_count = 0
    result_data = []

    for i, d in enumerate(data):
        new_d = dict(d)
        if str(i) in progress:
            new_d['output'] = progress[str(i)]
            expanded_count += 1
        else:
            kept_count += 1
        result_data.append(new_d)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    # 统计
    old_lens = [len(d['output']) for d in data]
    new_lens = [len(d['output']) for d in result_data]
    print(f"\n完成:", flush=True)
    print(f"  扩写: {expanded_count} 条", flush=True)
    print(f"  保留原样: {kept_count} 条", flush=True)
    print(f"  原始平均长度: {sum(old_lens)/len(old_lens):.0f} 字", flush=True)
    print(f"  新平均长度: {sum(new_lens)/len(new_lens):.0f} 字", flush=True)
    print(f"  保存: {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()

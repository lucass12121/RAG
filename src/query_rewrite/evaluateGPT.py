#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPT-4 语义一致性评测 - 判断模型回答与标准答案是否语义一致
"""

# ============================================================
# 👇 在这里填写配置
# ============================================================
API_KEY = os.getenv("OPENAI_API_KEY", "")  # 设置环境变量 OPENAI_API_KEY
INFILE = "../output/baseline_3b.csv"
OUTFILE = "../output/eval_3b.csv"
MODEL = "gpt-4o"
# ============================================================

import time
import re
import pandas as pd
from openai import OpenAI


EVAL_PROMPT = """你是法律问答评测员。判断模型回答和标准答案对同一问题的核心回答方向是否一致。

⚠️ 请注意：你只需要判断"回答方向"是否一致，不要纠结细节差异。

以下情况都算"一致"：
- 核心结论相同（如都说"可以"、都说"违法"、都说"需要XX条件"）
- 引用不同法条但结论相同
- 一方更详细一方更简略
- 列举要点有多有少但方向相同
- 具体金额/期限不同但处罚类型相同（如都说罚款+停业）
- 表述角度不同但意思一样

只有以下情况才算"不一致"：
- 结论完全相反（"可以"vs"不可以"、"有罪"vs"无罪"）
- 答非所问，回答了完全不同的问题
- 模型回答的法律定性与标准答案根本矛盾

【问题】：{question}
【标准答案】：{reference}
【模型回答】：{answer}

只输出"一致"或"不一致"。"""

def call_gpt4(client, question, reference, answer):
    prompt = EVAL_PROMPT.format(question=question, reference=reference, answer=answer)

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )
            text = response.choices[0].message.content.strip()

            if "不一致" in text:
                return 0
            if "一致" in text:
                return 1

            print(f"    ⚠️ 无法解析: [{text}]")
            if attempt < 2:
                time.sleep(1)

        except Exception as e:
            print(f"    ❌ API错误: {e}")
            if "rate_limit" in str(e).lower():
                time.sleep(10 * (attempt + 1))
            elif attempt < 2:
                time.sleep(3)
            else:
                return -1

    return -1


def main():
    if not API_KEY:
        print("❌ 请先在脚本顶部填写 API_KEY")
        return

    df = pd.read_csv(INFILE, encoding="utf-8")
    print(f"加载 {len(df)} 条数据\n")

    client = OpenAI(api_key=API_KEY)
    print("测试API连接...")
    try:
        test = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "输出：一致"}],
            max_tokens=5,
        )
        print(f"✅ API连接成功，返回: [{test.choices[0].message.content.strip()}]\n")
    except Exception as e:
        print(f"❌ API连接失败: {e}")
        return

    results = []
    total = len(df)
    correct = 0
    wrong = 0
    fail = 0
    start_time = time.time()

    for i, row in df.iterrows():
        q = str(row["question"]).strip()
        ref = str(row["reference_answer"]).strip()
        ans = str(row["model_answer"]).strip()

        if row.get("success") == False or ans in ("TIMEOUT", "FAIL", ""):
            results.append({**row.to_dict(), "consistent": -1})
            fail += 1
            print(f"[{i+1}/{total}] SKIP")
            continue

        result = call_gpt4(client, q, ref, ans)
        results.append({**row.to_dict(), "consistent": result})

        if result == 1:
            correct += 1
            tag = "✅"
        elif result == 0:
            wrong += 1
            tag = "❌"
        else:
            fail += 1
            tag = "⚠️"

        done = correct + wrong
        acc = correct / done * 100 if done > 0 else 0
        elapsed = time.time() - start_time
        eta = (elapsed / (i + 1)) * (total - i - 1)
        print(f"[{i+1}/{total}] {tag} 准确率={acc:.1f}% ({correct}/{done}) ETA={eta/60:.1f}min")

        if (i + 1) % 100 == 0:
            pd.DataFrame(results).to_csv(OUTFILE, index=False, encoding="utf-8-sig")
            print(f"  💾 已保存 ({i+1}/{total})")

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTFILE, index=False, encoding="utf-8-sig")

    # 汇总
    done = correct + wrong
    print(f"\n{'='*60}")
    print(f"评测完成")
    print(f"{'='*60}")
    print(f"总计: {total}")
    print(f"一致: {correct}  不一致: {wrong}  失败: {fail}")
    if done > 0:
        print(f"准确率: {correct/done*100:.1f}% ({correct}/{done})")
    print(f"耗时: {(time.time()-start_time)/60:.1f}分钟")
    print(f"已保存: {OUTFILE}")


if __name__ == "__main__":
    main()
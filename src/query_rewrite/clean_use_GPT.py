"""
法律QA数据清洗脚本
使用 GPT API 对每条QA进行质量评分，保留高质量数据（至少500条）

使用方法:
1. pip install openai pandas tqdm
2. 设置环境变量: export OPENAI_API_KEY="sk-xxx"
   或直接在下方 OPENAI_API_KEY 变量中填入你的key
3. python clean_law_qa.py
"""

import os
import json
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ==================== 配置区 ====================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # 设置环境变量 OPENAI_API_KEY
MODEL = "gpt-4o-mini"  # 便宜且够用，可换成 gpt-4o
INPUT_FILE = "../data/law_qa_mix_question_reply_clean2.csv"
OUTPUT_FILE = "../output/law_qa_cleaned.csv"
REMOVED_FILE = "../output/law_qa_removed.csv"  # 被移除的数据（方便检查）
MIN_KEEP = 500  # 至少保留条数
BATCH_SIZE = 5  # 每次发给GPT评分的条数（节省API调用次数）
MAX_RETRIES = 3  # API调用失败重试次数
# ================================================

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """你是一个法律QA数据集质量审核员。你需要对每条法律问答数据进行质量评分。

评分标准（1-10分）：
- 问题清晰度：问题是否表述清楚、有实际意义（2分）
- 回答准确性：回答是否准确、专业、有法律依据（3分）
- 回答完整性：回答是否充分回应了问题（2分）
- 回答实用性：回答对提问者是否有实际帮助（2分）
- 格式规范性：无明显乱码、重复、格式错误（1分）

低质量特征（应给低分）：
1. 回答过于笼统/敷衍（如"建议咨询律师"而无实质内容）
2. 回答与问题不相关或答非所问
3. 回答有明显重复段落
4. 问题或回答有乱码/格式严重错误
5. 回答照搬法条却未结合问题分析
6. 问题过于模糊无法形成有效QA对

请严格按JSON格式返回，不要有其他内容。"""


def build_batch_prompt(batch: list[dict]) -> str:
    """构建批量评分的prompt"""
    items = []
    for i, row in enumerate(batch):
        items.append(f"[{i}] 问题: {row['question']}\n回答: {row['reply']}")

    text = "\n\n---\n\n".join(items)

    return f"""请对以下{len(batch)}条法律QA数据逐条评分。

{text}

请严格返回如下JSON格式（不要有markdown代码块标记）：
[
  {{"index": 0, "score": 7, "reason": "简短理由"}},
  {{"index": 1, "score": 3, "reason": "简短理由"}},
  ...
]"""


def score_batch(batch: list[dict]) -> list[dict]:
    """调用GPT对一批数据评分"""
    prompt = build_batch_prompt(batch)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2000,
            )

            content = response.choices[0].message.content.strip()
            # 清理可能的markdown代码块标记
            content = content.replace("```json", "").replace("```", "").strip()
            results = json.loads(content)

            # 验证返回数量
            if len(results) == len(batch):
                return results
            else:
                print(f"  ⚠️ 返回{len(results)}条，期望{len(batch)}条，重试...")

        except json.JSONDecodeError as e:
            print(f"  ⚠️ JSON解析失败 (attempt {attempt + 1}): {e}")
        except Exception as e:
            print(f"  ⚠️ API调用失败 (attempt {attempt + 1}): {e}")
            time.sleep(2 ** attempt)  # 指数退避

    # 全部重试失败，给默认中等分（保守保留）
    print(f"  ❌ 批次评分失败，给默认分6分保留")
    return [{"index": i, "score": 6, "reason": "API评分失败，默认保留"} for i in range(len(batch))]


def main():
    # 1. 读取数据
    print(f"📂 读取文件: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    print(f"   共 {len(df)} 条数据\n")

    # 2. 基础预清洗（不需要API的规则）
    print("🧹 基础预清洗...")
    df = df.dropna(subset=["question", "reply"])
    df = df.drop_duplicates(subset=["question", "reply"])
    df["question"] = df["question"].str.strip()
    df["reply"] = df["reply"].str.strip()
    df = df[df["question"].str.len() >= 4]  # 问题至少4个字
    df = df[df["reply"].str.len() >= 10]  # 回答至少10个字
    df = df.reset_index(drop=True)
    print(f"   预清洗后剩余 {len(df)} 条\n")

    # 3. 批量调用GPT评分
    print(f"🤖 开始GPT评分 (模型: {MODEL}, 批次大小: {BATCH_SIZE})...")
    all_scores = []
    records = df.to_dict("records")

    for start in tqdm(range(0, len(records), BATCH_SIZE), desc="评分进度"):
        batch = records[start:start + BATCH_SIZE]
        results = score_batch(batch)

        for r in results:
            idx = start + r["index"]
            all_scores.append({
                "idx": idx,
                "score": r["score"],
                "reason": r.get("reason", ""),
            })

        time.sleep(0.5)  # 控制调用频率，避免rate limit

    # 4. 合并评分结果
    score_df = pd.DataFrame(all_scores)
    df["score"] = score_df["score"]
    df["reason"] = score_df["reason"]

    # 5. 按分数排序，确定阈值
    df_sorted = df.sort_values("score", ascending=False).reset_index(drop=True)

    print(f"\n📊 评分分布:")
    print(df["score"].value_counts().sort_index())
    print(f"   平均分: {df['score'].mean():.1f}")

    # 动态阈值：确保至少保留 MIN_KEEP 条
    if len(df_sorted) <= MIN_KEEP:
        threshold = 0  # 数据本身不够，全部保留
    else:
        # 取第 MIN_KEEP 条的分数作为阈值
        threshold = df_sorted.iloc[MIN_KEEP - 1]["score"]
        # 保留所有 >= threshold 的数据

    df_keep = df_sorted[df_sorted["score"] >= threshold].copy()
    df_remove = df_sorted[df_sorted["score"] < threshold].copy()

    # 如果保留太多（同分数的都被保留了），就精确截取
    # 确保至少 MIN_KEEP 条，但不会为了凑数而保留过多低分数据
    print(f"\n✅ 阈值分数: {threshold}")
    print(f"   保留: {len(df_keep)} 条")
    print(f"   移除: {len(df_remove)} 条")

    # 6. 保存结果
    df_keep[["question", "reply"]].to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    df_remove[["question", "reply", "score", "reason"]].to_csv(
        REMOVED_FILE, index=False, encoding="utf-8-sig"
    )

    print(f"\n💾 已保存:")
    print(f"   保留数据 → {OUTPUT_FILE}")
    print(f"   移除数据 → {REMOVED_FILE}（含评分和理由，方便审查）")
    print(f"\n🎉 完成！")


if __name__ == "__main__":
    main()
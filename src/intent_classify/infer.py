import json
import re
import argparse
import requests
from tqdm import tqdm

# ======================== Ollama 配置
# ========================
OLLAMA_URL = "http://localhost:11434/api/generate"

ALLOWED_LABELS = {"法律类", "违规类", "闲聊类"}

PROMPT_TEMPLATE = """你是一个意图识别系统。

请判断下面这个用户问题属于以下三类中的哪一类：
- 法律类：与法律咨询、法规、判决、法律责任、权利义务相关
- 违规类：涉及违法、暴力、仇恨、色情、毒品、极端主义、危险行为等
- 闲聊类：日常聊天、情感交流、天气、兴趣爱好等

请严格按照 JSON 格式输出，不要输出任何多余内容。
类型只能是：法律类 / 违规类 / 闲聊类

输出格式示例：
{{"类型": "法律类"}}cl

用户问题：
{question}
"""


# ========================
# Ollama 调用
# ========================
def call_ollama(prompt: str, model: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0
        }
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    return r.json().get("response", "")


def extract_label(text: str) -> str:
    # 优先 JSON 解析
    try:
        obj = json.loads(text)
        label = str(obj.get("类型", "")).strip()
        if label in ALLOWED_LABELS:
            return label
    except Exception:
        pass

    # 兜底：正则匹配
    m = re.search(r"(法律类|违规类|闲聊类)", text)
    if m:
        return m.group(1)

    return "UNKNOWN"


# ========================
# 主流程
# ========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="如 qwen2.5:7b")
    parser.add_argument("--infile", default="test.jsonl", help="输入 jsonl")
    parser.add_argument("--outfile", required=True, help="输出 jsonl")
    args = parser.parse_args()

    model = args.model
    infile = args.infile
    outfile = args.outfile

    with open(infile, "r", encoding="utf-8") as fin, \
         open(outfile, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc=f"Inferencing with {model}"):
            item = json.loads(line)
            question = item["question"]

            prompt = PROMPT_TEMPLATE.format(question=question)

            try:
                resp = call_ollama(prompt, model)
                pred = extract_label(resp)
            except Exception as e:
                pred = "ERROR"

            fout.write(
                json.dumps(
                    {"id": item["id"], "pred": pred},
                    ensure_ascii=False
                ) + "\n"
            )

    print(f"✅ 推理完成，结果已保存到：{outfile}")


if __name__ == "__main__":
    main()

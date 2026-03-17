# -*- coding: utf-8 -*-
import json, re, argparse, requests
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"
ALLOWED = {"法律类", "违规类", "闲聊类"}

def call_ollama(prompt: str, model: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 64, "stop": ["\n\n", "```"]}
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    return (r.json().get("response", "") or "").strip()

def normalize(s: str) -> str:
    if not s:
        return ""
    return (s.replace("“", '"').replace("”", '"')
             .replace("：", ":").strip())

def extract_label(text: str):
    t = normalize(text)

    # 1) 尝试从文本中抓 JSON 段（即便前后有废话）
    # 找到第一个 { 到最后一个 } 之间
    if "{" in t and "}" in t:
        try:
            j = t[t.find("{"): t.rfind("}") + 1]
            obj = json.loads(j)
            # 支持多种 key
            for k in ["label", "类型", "pred", "category"]:
                if k in obj:
                    v = str(obj[k]).strip()
                    # 支持 0/1/2
                    if v in ["0","1","2"]:
                        return {"0":"法律类", "1":"违规类", "2":"闲聊类"}[v]
                    # 支持 A/B/C
                    if v.upper() in ["A","B","C"]:
                        return {"A":"法律类","B":"违规类","C":"闲聊类"}[v.upper()]
                    # 支持 L/R/C
                    if v.upper() in ["L","R","C"]:
                        return {"L":"法律类","R":"违规类","C":"闲聊类"}[v.upper()]
                    # 支持中文标签
                    if v in ALLOWED:
                        return v
        except Exception:
            pass

    # 2) 直接搜中文标签
    m = re.search(r"(法律类|违规类|闲聊类)", t)
    if m:
        return m.group(1)

    # 3) 搜单字母（A/B/C 或 L/R/C）
    m = re.search(r"\b([ABC])\b", t.upper())
    if m:
        return {"A":"法律类","B":"违规类","C":"闲聊类"}[m.group(1)]
    m = re.search(r"\b([LRC])\b", t.upper())
    if m:
        return {"L":"法律类","R":"违规类","C":"闲聊类"}[m.group(1)]

    # 4) 搜 0/1/2
    m = re.search(r"\b([0-2])\b", t)
    if m:
        return {"0":"法律类", "1":"违规类", "2":"闲聊类"}[m.group(1)]

    return None

def predict_one(q: str, base_prompt: str, model: str):
    p = base_prompt.replace("{{question}}", q)
    r1 = call_ollama(p, model)
    lab = extract_label(r1)
    if lab:
        return lab

    # 重试：要求只输出 JSON（续写式）
    retry = p + '\n只输出一行 JSON，例如 {"label":"法律类"}，不要解释。\n输出：'
    r2 = call_ollama(retry, model)
    lab2 = extract_label(r2)
    if lab2:
        return lab2

    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--fallback", default="闲聊类", choices=list(ALLOWED))
    ap.add_argument("--dump_raw", action="store_true")
    args = ap.parse_args()

    base_prompt = open(args.prompt, "r", encoding="utf-8").read()

    with open(args.infile, "r", encoding="utf-8") as fin, open(args.outfile, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="infer_strict"):
            item = json.loads(line)
            q = item["question"]
            try:
                pred = predict_one(q, base_prompt, args.model) or args.fallback
                out = {"id": item["id"], "pred": pred}
                if args.dump_raw:
                    out["question"] = q
            except Exception:
                out = {"id": item["id"], "pred": "ERROR"}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"完成：{args.outfile}")

if __name__ == "__main__":
    main()

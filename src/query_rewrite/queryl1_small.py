#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lightweight rewrite pipeline for small models (0.5b/1.5b).

Key differences from queryl1.py:
  - Does NOT require JSON output from model — accepts plain text
  - Falls back to JSON parsing if model happens to output JSON
  - Simpler prompt (promptl1_small.txt)
  - Same quality checks (looks_bad) as main pipeline

Usage:
  python queryl1_small.py infile.csv outfile.csv model_name [prompt_file]

Default prompt: promptl1_small.txt
"""

import sys
import json
import re
import subprocess
from typing import Optional, Tuple
import pandas as pd

QUESTION_COL = "question"
DEFAULT_PROMPT_PATH = "promptl1_small.txt"

CODEBLOCK_RE = re.compile(r"```(?:json|jsonc)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
EN3_RE = re.compile(r"[A-Za-z]{3,}")
SLUG_RE = re.compile(r"[_]|[A-Za-z0-9]+-[A-Za-z0-9]+")
HAS_CN_RE = re.compile(r"[\u4e00-\u9fff]")
TRIM_QUOTES_RE = re.compile(r'^[\s"""\']+|[\s"""\']+$')
# 繁体字常见样本
TRADITIONAL_RE = re.compile(
    r'[\u570b\u7121\u8207\u9019\u958b\u5c0d\u6a23\u8acb\u554f\u9818'
    r'\u8b93\u689d\u4ef6\u7db2\u7d61\u8655\u5247\u8a72\u8b49\u9a57'
    r'\u6b0a\u5229\u7fa9\u52d9\u8996\u5bdf\u5831\u544a\u8abf\u67e5'
    r'\u8a55\u4f30\u689d\u6b3e\u898f\u5b9a\u5354\u8b70\u7d93\u6fdf'
    r'\u8ca1\u7522\u8cc7\u7522]'
)


def call_ollama(model: str, prompt: str, timeout: int = 120) -> Tuple[str, bool]:
    try:
        r = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "OLLAMA_TIMEOUT", False
    except Exception as e:
        return f"OLLAMA_ERROR: {e}", False

    out = (r.stdout or "").strip()
    err = (r.stderr or "").strip()

    if r.returncode != 0:
        msg = out if out else ""
        if err:
            msg = (msg + "\n" + err).strip()
        return f"OLLAMA_RC_{r.returncode}: {msg}".strip(), False

    if (not out) and err:
        return f"OLLAMA_EMPTY_STDOUT: {err}", False

    return out, True


def _json_load(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


def _extract_balanced_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def extract_from_json(text: str) -> Optional[str]:
    """Try to extract rewritten_question from JSON in text."""
    if not text:
        return None

    # codeblock
    for m in CODEBLOCK_RE.finditer(text):
        obj = _json_load(m.group(1))
        if isinstance(obj, dict):
            v = obj.get("rewritten_question")
            if isinstance(v, str) and v.strip():
                return v.strip()

    # whole text
    obj = _json_load(text)
    if isinstance(obj, dict):
        v = obj.get("rewritten_question")
        if isinstance(v, str) and v.strip():
            return v.strip()

    # balanced
    bal = _extract_balanced_json_object(text)
    if bal:
        obj2 = _json_load(bal)
        if isinstance(obj2, dict):
            v = obj2.get("rewritten_question")
            if isinstance(v, str) and v.strip():
                return v.strip()

    return None


def extract_plain_text(text: str, orig: str) -> Optional[str]:
    """
    Extract rewritten question from plain text output.
    Strategy:
      1. Take first non-empty line that looks like a Chinese question
      2. Strip common prefixes like "改写：" "答：" etc.
      3. Ignore lines that are too similar to prompt fragments
    """
    if not text:
        return None

    lines = text.strip().split("\n")
    skip_prefixes = ["问题", "原问题", "用户问题", "输出", "json", "```"]

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # skip lines that look like prompt echo or metadata
        low = line.lower()
        if any(low.startswith(p) for p in skip_prefixes):
            continue

        # strip common answer prefixes
        line = re.sub(r'^(改写[：:]|答[：:]|回答[：:])\s*', '', line).strip()
        line = TRIM_QUOTES_RE.sub('', line).strip()

        # must contain Chinese
        if not HAS_CN_RE.search(line):
            continue

        # skip if it's just the original question echoed back
        if line == orig.strip():
            continue

        # skip very short lines (likely fragments)
        if len(line) < 4:
            continue

        return line

    return None


def extract_rewritten(text: str, orig: str) -> Tuple[Optional[str], str]:
    """
    Try JSON first, then plain text.
    Returns: (rewritten_question_or_None, method)
    method is one of: "json", "plain", "none"
    """
    # 1. Try JSON
    rq = extract_from_json(text)
    if rq:
        return rq, "json"

    # 2. Try plain text
    rq = extract_plain_text(text, orig)
    if rq:
        return rq, "plain"

    return None, "none"


def norm(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").strip())


def clean_sentence(s: str) -> str:
    s = (s or "").strip()
    s = TRIM_QUOTES_RE.sub("", s).strip()
    return s


def looks_bad(rq: str, orig: str) -> Optional[str]:
    if not rq:
        return "EMPTY_RQ"
    if not HAS_CN_RE.search(rq):
        return "NO_CHINESE"
    if EN3_RE.search(rq):
        return "ENGLISH_MIXED"
    if SLUG_RE.search(rq):
        return "SLUG_OR_UNDERSCORE"

    # 长度异常
    if orig and len(rq) > len(orig) * 3:
        return "TOO_LONG"

    # 太短（丢失太多信息）
    if orig and len(orig) > 10 and len(rq) < len(orig) * 0.3:
        return "TOO_SHORT"

    # 主体篡改
    corp_keywords = ["公司", "我司", "我等", "本公司", "本司"]
    for kw in corp_keywords:
        if kw not in orig and kw in rq:
            return f"SUBJECT_CHANGED({kw})"

    # 繁体字
    if TRADITIONAL_RE.search(rq):
        return "TRADITIONAL_CHINESE"

    # 问号检测（包含问号即可，不要求结尾）
    if not re.search(r'[？?]', rq):
        return "NO_QUESTION_MARK"

    return None


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def render_prompt(tpl: str, q: str) -> str:
    try:
        return tpl.format(q=q)
    except KeyError as e:
        raise KeyError(f"{e}\nprompt 文件中有未转义的花括号")


def rewrite_one(model: str, q: str, tpl: str) -> Tuple[str, bool, bool, str, str]:
    q = (q or "").strip()
    if not q or q.lower() == "nan":
        return q, False, True, "EMPTY", "EMPTY"

    prompt = render_prompt(tpl, q)

    # 1st call
    out1, ok1 = call_ollama(model, prompt)
    if not ok1:
        return q, False, False, "CALL_FAIL", out1

    rq1, method1 = extract_rewritten(out1, q)
    if not rq1:
        return q, False, True, "PARSE_FAIL", out1

    rq1 = clean_sentence(rq1)
    bad1 = looks_bad(rq1, q)

    if not bad1:
        changed = norm(rq1) != norm(q)
        return rq1, changed, True, f"OK_{method1}", out1

    # 2nd call (retry)
    retry_prompt = (
        f"请把下面的中文问题改写得更清晰，以问号结尾，只输出改写后的问句，不要解释。\n\n"
        f"问题：{q}\n"
        f"改写："
    )
    out2, ok2 = call_ollama(model, retry_prompt)
    if ok2:
        rq2, method2 = extract_rewritten(out2, q)
        if rq2:
            rq2 = clean_sentence(rq2)
            bad2 = looks_bad(rq2, q)
            if not bad2:
                changed = norm(rq2) != norm(q)
                return rq2, changed, True, f"OK_{method2}_retry", out2

        return q, False, True, f"BAD_RQ_ROLLBACK:{bad1}", out2

    return q, False, False, "RETRY_CALL_FAIL", out2


def run(infile: str, outfile: str, model: str, prompt_path: str) -> None:
    df = pd.read_csv(infile, encoding="utf-8")
    if QUESTION_COL not in df.columns:
        raise ValueError(f"CSV 中未找到列 {QUESTION_COL}，实际列：{list(df.columns)}")

    tpl = load_prompt(prompt_path)
    rows = []
    total = len(df)

    for i, row in df.iterrows():
        q = "" if pd.isna(row[QUESTION_COL]) else str(row[QUESTION_COL]).strip()
        rq, changed, ok, status, raw = rewrite_one(model, q, tpl)

        rows.append({
            "id": i + 1,
            "model": model,
            "question": q,
            "rewritten_question": rq,
            "changed": changed,
            "ok": ok,
            "status": status,
            "raw_output": raw,
        })

        flag = ""
        if "ROLLBACK" in status:
            flag = " ⚠️"
        elif "FAIL" in status and "OK" not in status:
            flag = " ❌"
        print(f"[{i+1}/{total}] {status}{flag}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(outfile, index=False, encoding="utf-8-sig")

    # 汇总
    total_n = len(out_df)
    ok_json = len(out_df[out_df["status"].str.startswith("OK_json")])
    ok_plain = len(out_df[out_df["status"].str.startswith("OK_plain")])
    rollback_n = len(out_df[out_df["status"].str.contains("ROLLBACK")])
    fail_n = total_n - ok_json - ok_plain - rollback_n
    print(f"\n{'='*50}")
    print(f"总计: {total_n}")
    print(f"  成功(JSON): {ok_json}")
    print(f"  成功(纯文本): {ok_plain}")
    print(f"  回滚: {rollback_n}")
    print(f"  失败: {fail_n}")
    print(f"Saved to {outfile}")


def main() -> int:
    if len(sys.argv) < 4:
        print("Usage: python queryl1_small.py infile.csv outfile.csv model_name [prompt_file]")
        return 1
    infile = sys.argv[1]
    outfile = sys.argv[2]
    model = sys.argv[3]
    prompt_path = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_PROMPT_PATH
    run(infile, outfile, model, prompt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

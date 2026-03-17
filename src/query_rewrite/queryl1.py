#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Query rewrite pipeline v4 (CN legal, with single keyword extraction)

Usage:
  python queryl1.py infile.csv outfile.csv model_name

Input CSV must contain column: question
Output CSV:
  详细版(outfile): id, model, 原始问题, 改写问题, 关键词, changed, ok, status, raw_output
  干净版(outfile_clean): 原始问题, 改写问题, 关键词
"""

import sys
import json
import re
import subprocess
from typing import Optional, Tuple
import pandas as pd

QUESTION_COL = "question"
PROMPT_PATH = "promptl1.txt"

CODEBLOCK_RE = re.compile(r"```(?:json|jsonc)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

# 英文/slug 检测
EN3_RE = re.compile(r"[A-Za-z]{3,}")
SLUG_RE = re.compile(r"[_]|[A-Za-z0-9]+-[A-Za-z0-9]+")

# 至少包含一个中文字符
HAS_CN_RE = re.compile(r"[\u4e00-\u9fff]")

# 清理首尾引号
TRIM_QUOTES_RE = re.compile(r'^[\s"""]+|[\s"""]+$')


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


def extract_result(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract 改写问题 and 关键词 from model output.
    Supports both Chinese keys and English keys as fallback.
    Returns: (rewritten_question, keyword)
    """
    if not text:
        return None, None

    def _get_fields(obj):
        if not isinstance(obj, dict):
            return None, None
        # 优先中文 key，兼容英文 key
        rq = obj.get("改写问题") or obj.get("rewritten_question")
        kw = obj.get("关键词") or obj.get("keywords") or obj.get("keyword", "")
        if isinstance(rq, str) and rq.strip():
            kw_str = kw.strip() if isinstance(kw, str) else ""
            return rq.strip(), kw_str
        return None, None

    # 1) codeblock
    for m in CODEBLOCK_RE.finditer(text):
        obj = _json_load(m.group(1))
        rq, kw = _get_fields(obj)
        if rq:
            return rq, kw

    # 2) whole text
    obj = _json_load(text)
    rq, kw = _get_fields(obj)
    if rq:
        return rq, kw

    # 3) balanced object
    bal = _extract_balanced_json_object(text)
    if bal:
        obj2 = _json_load(bal)
        rq, kw = _get_fields(obj2)
        if rq:
            return rq, kw

    return None, None


def norm(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").strip())


def clean_sentence(s: str) -> str:
    """清理改写结果：去首尾引号、修复双问号"""
    s = (s or "").strip()
    s = TRIM_QUOTES_RE.sub("", s).strip()
    # 修复重复问号
    s = re.sub(r'[？?]{2,}', '？', s)
    return s


def auto_fix_question_mark(rq: str) -> str:
    """对缺少问号但内容合理的改写，自动补问号。"""
    rq = rq.strip()
    if rq and not re.search(r'[？?]$', rq):
        rq = rq + '？'
    return rq


def clean_keyword(kw: str) -> str:
    """清理关键词：去引号、只保留第一个关键词"""
    if not kw:
        return ""
    kw = kw.strip()
    kw = re.sub(r'["""\'\']+', '', kw)
    # 如果模型输出了多个，只取第一个
    parts = re.split(r'[,，、\s]+', kw)
    for p in parts:
        p = p.strip()
        if p and HAS_CN_RE.search(p):
            return p
    return kw.strip()


def looks_bad(rq: str, orig: str) -> Optional[str]:
    """
    Return reason string if the rewritten question looks problematic, else None.
    """
    if not rq:
        return "EMPTY_RQ"
    if not HAS_CN_RE.search(rq):
        return "NO_CHINESE"
    if EN3_RE.search(rq):
        return "ENGLISH_MIXED"
    if SLUG_RE.search(rq):
        return "SLUG_OR_UNDERSCORE"

    # 长度异常：改写后不应超过原文 1.5 倍
    if orig and len(rq) > len(orig) * 1.5:
        return "TOO_LONG"

    # 主体篡改检测
    corp_keywords = ["公司", "我司", "我等", "本公司", "本司"]
    for kw in corp_keywords:
        if kw not in orig and kw in rq:
            return f"SUBJECT_CHANGED({kw})"

    # 虚构人名检测
    added_text = rq
    for ch in orig:
        added_text = added_text.replace(ch, "", 1)
    if re.search(r"小[明红刚丽]|张[三某]|李[四某]|王[五某]", added_text):
        return "FABRICATED_NAME"

    # 句子完整性：应以问号结尾
    if not re.search(r"[？?]$", rq.strip()):
        return "NO_QUESTION_MARK"

    # prompt泄漏检测
    leak_patterns = ["解析", "解释：", "根据法律规定", "核心问题", "改写为", "这个问题"]
    for pat in leak_patterns:
        if pat in rq and pat not in orig:
            return f"PROMPT_LEAK({pat})"

    return None


def load_prompt() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def render_prompt(tpl: str, q: str) -> str:
    try:
        return tpl.format(q=q)
    except KeyError as e:
        raise KeyError(
            f"{e}\n"
            f"promptl1.txt 里可能有未转义的花括号。\n"
            f"除了 {{q}} 以外，JSON 示例必须写成：{{{{\"改写问题\":\"...\"}}}}"
        )


def rewrite_one(model: str, q: str, tpl: str) -> Tuple[str, str, bool, bool, str, str]:
    """
    Returns: (rewritten_question, keyword, changed, ok, status, raw_output)
    """
    q = (q or "").strip()
    if not q or q.lower() == "nan":
        return q, "", False, True, "EMPTY", "EMPTY"

    prompt = render_prompt(tpl, q)

    # 1st call
    out1, ok1 = call_ollama(model, prompt)
    if not ok1:
        return q, "", False, False, "CALL_FAIL", out1

    rq1, kw1 = extract_result(out1)
    if not rq1:
        return q, "", False, True, "JSON_FAIL", out1

    rq1 = clean_sentence(rq1)
    rq1 = auto_fix_question_mark(rq1)
    kw1 = clean_keyword(kw1 or "")
    bad1 = looks_bad(rq1, q)

    if not bad1:
        changed = norm(rq1) != norm(q)
        return rq1, kw1, changed, True, "OK", out1

    # 2nd call (retry)
    suffix = (
        "\n\n【再次强调】\n"
        "1) 必须只输出一行 JSON，不能有任何解释/多余文字/代码块/注释。\n"
        "2) 改写问题必须是简洁的中文问句，比原文更短，以问号结尾，"
        "使用法律术语，不新增事实，不输出英文、拼音、下划线、连字符 slug。\n"
        "3) 关键词是1个中文法律关键词。\n"
        "4) 严禁改变提问主体和法律关系。严禁篡改数字金额。\n"
        '5) 格式：{"改写问题":"...","关键词":"..."}\n'
    )
    out2, ok2 = call_ollama(model, prompt + suffix)
    if ok2:
        rq2, kw2 = extract_result(out2)
        if rq2:
            rq2 = clean_sentence(rq2)
            rq2 = auto_fix_question_mark(rq2)
            kw2 = clean_keyword(kw2 or "")
            bad2 = looks_bad(rq2, q)
            if not bad2:
                changed = norm(rq2) != norm(q)
                return rq2, kw2, changed, True, "OK", out2

        return q, "", False, True, f"BAD_RQ_ROLLBACK:{bad1}", out2

    return q, "", False, False, "RETRY_CALL_FAIL", out2


def run(infile: str, outfile: str, model: str) -> None:
    df = pd.read_csv(infile, encoding="utf-8")
    if QUESTION_COL not in df.columns:
        raise ValueError(f"CSV 中未找到列 {QUESTION_COL}，实际列：{list(df.columns)}")

    tpl = load_prompt()
    rows = []
    total = len(df)

    for i, row in df.iterrows():
        q = "" if pd.isna(row[QUESTION_COL]) else str(row[QUESTION_COL]).strip()
        rq, kw, changed, ok, status, raw = rewrite_one(model, q, tpl)

        rows.append({
            "id": i + 1,
            "model": model,
            "原始问题": q,
            "改写问题": rq,
            "关键词": kw,
            "changed": changed,
            "ok": ok,
            "status": status,
            "raw_output": raw,
        })

        flag = ""
        if status.startswith("BAD_RQ_ROLLBACK"):
            flag = " ⚠️ ROLLBACK"
        elif status in ("CALL_FAIL", "RETRY_CALL_FAIL"):
            flag = " ❌ FAIL"
        print(f"[{i+1}/{total}] changed={changed} ok={ok} kw=[{kw}] status={status}{flag}")

    out_df = pd.DataFrame(rows)

    # 保存完整debug版本
    out_df.to_csv(outfile, index=False, encoding="utf-8-sig")

    # 同时保存老师要求的干净版本（只有三列）
    clean_name = outfile.replace(".csv", "_clean.csv")
    clean_df = out_df[["原始问题", "改写问题", "关键词"]].copy()
    clean_df.to_csv(clean_name, index=False, encoding="utf-8-sig")

    # 打印汇总
    total_n = len(out_df)
    ok_n = len(out_df[out_df["status"] == "OK"])
    rollback_n = len(out_df[out_df["status"].str.startswith("BAD_RQ_ROLLBACK")])
    fail_n = total_n - ok_n - rollback_n
    changed_n = len(out_df[out_df["changed"] == True])
    kw_n = len(out_df[out_df["关键词"].str.len() > 0])
    print(f"\n{'='*50}")
    print(f"总计: {total_n}  成功: {ok_n}  回滚: {rollback_n}  失败: {fail_n}")
    print(f"有效改写: {changed_n}/{ok_n}  有关键词: {kw_n}/{ok_n}")
    print(f"详细结果: {outfile}")
    print(f"干净结果: {clean_name}")


def main() -> int:
    if len(sys.argv) != 4:
        print("Usage: python queryl1.py infile.csv outfile.csv model_name")
        return 1
    _, infile, outfile, model = sys.argv
    run(infile, outfile, model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
数据预处理模块
1. 将 4 个原始数据源统一转换为纯文本格式
2. 对纯文本进行递归分块，输出 JSONL
"""
import csv
import json
import re
from pathlib import Path
from typing import List


# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def clean_markdown(text: str) -> str:
    """去掉 Markdown 标记，保留纯文本"""
    # 去掉 HTML 注释 <!-- ... -->
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # 去掉标题标记 # ## ### 等，保留文字
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # 去掉加粗 **text** 或 __text__
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    # 去掉斜体 *text* 或 _text_
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    # 去掉链接 [text](url) -> text
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    # 去掉多余空行（连续3个以上换行合并为2个）
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def process_law_book():
    """处理 Law-Book 目录下的 .md 法律文件 -> 纯文本"""
    law_book_dir = RAW_DIR / "Law-Book"
    if not law_book_dir.exists():
        print("[跳过] Law-Book 目录不存在")
        return

    output_file = PROCESSED_DIR / "law_book_statutes.txt"
    count = 0

    with open(output_file, "w", encoding="utf-8") as fout:
        for md_file in sorted(law_book_dir.rglob("*.md")):
            text = md_file.read_text(encoding="utf-8")
            clean_text = clean_markdown(text)
            if clean_text:
                fout.write(clean_text)
                fout.write("\n\n" + "=" * 80 + "\n\n")
                count += 1

    print(f"[Law-Book] 处理完成: {count} 个法律文件 -> {output_file}")


def process_legal_article():
    """处理 legal_article/article.txt (JSON Lines) -> 纯文本"""
    article_file = RAW_DIR / "legal_article" / "article.txt"
    if not article_file.exists():
        print("[跳过] legal_article/article.txt 不存在")
        return

    output_file = PROCESSED_DIR / "legal_articles.txt"
    count = 0

    with open(article_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            input_text = item.get("input", "").strip()
            answer_text = item.get("answer", "").strip()
            if input_text and answer_text:
                fout.write(f"{input_text}\n{answer_text}\n\n")
                count += 1

    print(f"[legal_article] 处理完成: {count} 条法条记录 -> {output_file}")


def process_legal_book():
    """处理 legal_book 目录下的 .txt 法律教材 -> 直接复制合并"""
    book_dir = RAW_DIR / "legal_book" / "legal_book"
    if not book_dir.exists():
        print("[跳过] legal_book 目录不存在")
        return

    output_file = PROCESSED_DIR / "legal_books.txt"
    count = 0

    with open(output_file, "w", encoding="utf-8") as fout:
        for txt_file in sorted(book_dir.glob("*.txt")):
            # 读取文件内容，尝试多种编码
            text = None
            for enc in ("utf-8", "gbk", "gb18030"):
                try:
                    text = txt_file.read_text(encoding=enc)
                    break
                except (UnicodeDecodeError, ValueError):
                    continue
            if text is None:
                text = txt_file.read_text(encoding="utf-8", errors="ignore")

            # 文件名也可能是非 UTF-8 编码，尝试修复
            try:
                title = txt_file.stem
                # 检测是否是 mojibake（乱码），尝试 latin1->gbk 解码
                title.encode("ascii")
            except UnicodeEncodeError:
                title = txt_file.stem  # 本身就是正常中文
            else:
                try:
                    title = txt_file.stem.encode("latin1").decode("gbk")
                except (UnicodeDecodeError, UnicodeEncodeError):
                    title = txt_file.stem

            text = text.strip()
            if text:
                fout.write(f"【{title}】\n\n")
                fout.write(text)
                fout.write("\n\n" + "=" * 80 + "\n\n")
                count += 1

    print(f"[legal_book] 处理完成: {count} 本法律教材 -> {output_file}")


def process_lawzhidao():
    """处理 lawzhidao_filter.csv 法律问答数据 -> 纯文本"""
    csv_file = RAW_DIR / "lawzhidao_filter.csv"
    if not csv_file.exists():
        print("[跳过] lawzhidao_filter.csv 不存在")
        return

    output_file = PROCESSED_DIR / "law_qa.txt"
    count = 0

    # 尝试不同编码
    for encoding in ("utf-8", "gbk", "gb18030"):
        try:
            with open(csv_file, "r", encoding=encoding) as f:
                f.readline()  # 测试读取
            break
        except UnicodeDecodeError:
            continue
    else:
        encoding = "utf-8"

    with open(csv_file, "r", encoding=encoding, errors="ignore") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        for row in reader:
            title = row.get("title", "").strip()
            question = row.get("question", "").strip()
            reply = row.get("reply", "").strip()

            # 至少有标题和回复才保留
            if not (title and reply):
                continue

            parts = []
            parts.append(f"问：{title}")
            if question:
                parts.append(f"补充：{question}")
            parts.append(f"答：{reply}")

            fout.write("\n".join(parts))
            fout.write("\n\n")
            count += 1

    print(f"[lawzhidao] 处理完成: {count} 条问答记录 -> {output_file}")


## ========== 第二阶段：递归分块 ==========

DEFAULT_CHUNK_SIZE = 512

# 中文数字字符集
_CN_NUM = r"[一二三四五六七八九十百千零\d]+"

# 法律结构层级正则（从粗到细）
RE_BIAN = re.compile(rf"^第{_CN_NUM}编[\s 　]", re.MULTILINE)   # 编
RE_ZHANG = re.compile(rf"^第{_CN_NUM}章[\s 　]", re.MULTILINE)  # 章
RE_JIE = re.compile(rf"^第{_CN_NUM}节[\s 　]", re.MULTILINE)    # 节
RE_TIAO = re.compile(rf"^第{_CN_NUM}条[\s 　]", re.MULTILINE)   # 条


def split_by_regex(text: str, pattern: re.Pattern) -> List[str]:
    """
    用正则按匹配位置切分文本。
    每个匹配的行作为新段的开头，匹配之前的内容作为前一段。
    """
    positions = [m.start() for m in pattern.finditer(text)]
    if not positions:
        return [text] if text.strip() else []

    parts = []
    # 匹配之前的内容（前言、序言等）
    if positions[0] > 0:
        preamble = text[:positions[0]].strip()
        if preamble:
            parts.append(preamble)
    # 每个匹配到下一个匹配之间
    for i, pos in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        part = text[pos:end].strip()
        if part:
            parts.append(part)
    return parts


def split_by_sentences(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    """按句子切分超长文本（最细粒度的兜底）"""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    sentences = re.split(r"(?<=[。；])", text)
    chunks = []
    current = ""
    for sent in sentences:
        if not sent:
            continue
        if len(current) + len(sent) <= chunk_size:
            current += sent
        else:
            if current.strip():
                chunks.append(current.strip())
            if len(sent) > chunk_size:
                for i in range(0, len(sent), chunk_size):
                    piece = sent[i:i + chunk_size].strip()
                    if piece:
                        chunks.append(piece)
                current = ""
            else:
                current = sent
    if current.strip():
        chunks.append(current.strip())
    return chunks


def merge_paragraphs(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    """按段落合并短文本，超长段落按句子切"""
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = current + "\n\n" + para if current else para
        else:
            if current.strip():
                chunks.append(current.strip())
            if len(para) > chunk_size:
                chunks.extend(split_by_sentences(para, chunk_size))
                current = ""
            else:
                current = para
    if current.strip():
        chunks.append(current.strip())
    return chunks


def detect_structure_context(text: str) -> str:
    """从文本中提取结构上下文（编/章/节），用于 metadata"""
    context_parts = []
    for pattern, label in [(RE_BIAN, ""), (RE_ZHANG, ""), (RE_JIE, "")]:
        match = pattern.search(text)
        if match:
            # 提取整行
            line_end = text.find("\n", match.start())
            line = text[match.start():line_end].strip() if line_end > 0 else text[match.start():].strip()
            context_parts.append(line)
    return " > ".join(context_parts)


def _extract_header(text: str, pattern: re.Pattern) -> str:
    """从片段开头提取结构标题（只取第一行中匹配的部分，不含正文）"""
    match = pattern.match(text)
    if not match:
        return ""
    line_end = text.find("\n")
    return text[:line_end].strip() if line_end > 0 else text.strip()


def _is_pure_header(text: str) -> bool:
    """判断文本是否只是一个结构标题（编/章/节），没有实质内容"""
    text = text.strip()
    # 只有一行，且是"第X编/章/节 xxx"格式
    if "\n" in text:
        return False
    return bool(re.match(rf"^第{_CN_NUM}[编章节][\s 　]", text))


def recursive_split_law(text: str, law_name: str,
                        chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[dict]:
    """
    对单部法律进行真正的多层递归分块：
    第1层：按"编"切
    第2层：按"章"切
    第3层：按"节"切
    第4层：按"条"切 → 每条是独立语义单元
    第5层：单条太长 → 按句子切

    修复规则：
    - section 只保留编/章/节路径，不含条级内容
    - 纯标题 chunk（如"第一章 总纲"）不入库，标题信息已在 section 中
    - 前言（第一条之前的内容）标记 section 为"序言"
    """
    chunks = []

    # 结构层级：编/章/节用于 section metadata
    structure_levels = [
        (RE_BIAN, "编"),
        (RE_ZHANG, "章"),
        (RE_JIE, "节"),
    ]

    def _add_chunk(text_content: str, section: str):
        """添加 chunk（跳过纯标题）"""
        text_content = text_content.strip()
        if not text_content:
            return
        # 跳过纯结构标题（如"第一章 总纲"），这些信息已在 section 中
        if _is_pure_header(text_content):
            return
        chunks.append({
            "text": text_content,
            "source": "law_book_statutes",
            "title": law_name,
            "section": section,
        })

    def _split_into_articles(fragment: str, section: str):
        """将片段按"条"切分，每条独立入库"""
        parts = split_by_regex(fragment, RE_TIAO)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # 判断是否是条文前面的前言/序言内容
            is_preamble = not RE_TIAO.match(part)
            part_section = section
            if is_preamble and not section:
                part_section = "序言"

            if len(part) <= chunk_size:
                _add_chunk(part, part_section)
            else:
                # 单条太长，按句子切
                for sc in merge_paragraphs(part, chunk_size):
                    _add_chunk(sc, part_section)

    def _recurse(fragment: str, level_idx: int, context: str):
        """递归拆分结构层级（编/章/节），最后进入条级处理"""
        fragment = fragment.strip()
        if not fragment:
            return

        # 所有结构层级都用完了，进入条级处理
        if level_idx >= len(structure_levels):
            _split_into_articles(fragment, context)
            return

        pattern, level_name = structure_levels[level_idx]
        parts = split_by_regex(fragment, pattern)

        if len(parts) > 1:
            # 切分成功：对每个子片段递归
            for part in parts:
                new_context = context
                header = _extract_header(part, pattern)
                if header:
                    new_context = f"{context} > {header}" if context else header
                _recurse(part, level_idx + 1, new_context)
        else:
            # 当前层级切不动，跳到下一层
            _recurse(fragment, level_idx + 1, context)

    _recurse(text, 0, "")
    return chunks


def chunk_law_book_statutes() -> List[dict]:
    """对法律条文进行分块：每部法律 → 编 → 章 → 节 → 条 → 句子"""
    filepath = PROCESSED_DIR / "law_book_statutes.txt"
    if not filepath.exists():
        return []

    text = filepath.read_text(encoding="utf-8")
    docs = text.split("=" * 80)
    chunks = []

    for doc in docs:
        doc = doc.strip()
        if not doc:
            continue
        law_name = doc.split("\n")[0].strip()
        law_chunks = recursive_split_law(doc, law_name)
        chunks.extend(law_chunks)

    print(f"[分块] law_book_statutes: {len(chunks)} 个 chunks")
    return chunks


def chunk_legal_articles() -> List[dict]:
    """对法条数据进行分块（每条法条本身就是自然单元）"""
    filepath = PROCESSED_DIR / "legal_articles.txt"
    if not filepath.exists():
        return []

    text = filepath.read_text(encoding="utf-8")
    entries = text.split("\n\n")
    chunks = []

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        title = entry.split("\n")[0][:50]
        if len(entry) <= DEFAULT_CHUNK_SIZE:
            chunks.append({
                "text": entry,
                "source": "legal_articles",
                "title": title,
            })
        else:
            for sc in split_by_sentences(entry):
                chunks.append({
                    "text": sc,
                    "source": "legal_articles",
                    "title": title,
                })

    print(f"[分块] legal_articles: {len(chunks)} 个 chunks")
    return chunks


def recursive_split_book(text: str, book_name: str,
                         chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[dict]:
    """
    对单本教材进行递归分块：
    第1层：按"编"切
    第2层：按"章"切
    第3层：按"节"切
    第4层：按段落合并
    第5层：单段太长 → 按句子切

    纯标题 chunk 不入库，标题信息已在 section 中。
    """
    chunks = []
    levels = [
        (RE_BIAN, "编"),
        (RE_ZHANG, "章"),
        (RE_JIE, "节"),
    ]

    def _add_chunk(text_content: str, section: str):
        text_content = text_content.strip()
        if not text_content:
            return
        if _is_pure_header(text_content):
            return
        # 跳过只有书名标记的 chunk（如 "【book】"）
        if re.match(r"^【.+?】$", text_content):
            return
        chunks.append({
            "text": text_content,
            "source": "legal_books",
            "title": book_name,
            "section": section,
        })

    def _recurse(fragment: str, level_idx: int, context: str):
        fragment = fragment.strip()
        if not fragment:
            return

        if level_idx < len(levels):
            pattern, level_name = levels[level_idx]
            parts = split_by_regex(fragment, pattern)

            if len(parts) > 1:
                for part in parts:
                    new_context = context
                    header = _extract_header(part, pattern)
                    if header:
                        new_context = f"{context} > {header}" if context else header
                    _recurse(part, level_idx + 1, new_context)
                return
            else:
                _recurse(fragment, level_idx + 1, context)
                return

        # 所有结构层级用完，按段落合并
        for sc in merge_paragraphs(fragment, chunk_size):
            _add_chunk(sc, context)

    _recurse(text, 0, "")
    return chunks


def chunk_legal_books() -> List[dict]:
    """对法律教材进行分块：编 → 章 → 节 → 段落 → 句子"""
    filepath = PROCESSED_DIR / "legal_books.txt"
    if not filepath.exists():
        return []

    text = filepath.read_text(encoding="utf-8")
    books = text.split("=" * 80)
    chunks = []

    for book in books:
        book = book.strip()
        if not book:
            continue
        title_match = re.match(r"【(.+?)】", book)
        title = title_match.group(1) if title_match else "未知教材"
        book_chunks = recursive_split_book(book, title)
        chunks.extend(book_chunks)

    print(f"[分块] legal_books: {len(chunks)} 个 chunks")
    return chunks


def chunk_law_qa() -> List[dict]:
    """对法律问答数据进行分块（每条问答是自然单元）"""
    filepath = PROCESSED_DIR / "law_qa.txt"
    if not filepath.exists():
        return []

    text = filepath.read_text(encoding="utf-8")
    entries = text.split("\n\n")
    chunks = []

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        title = ""
        for line in entry.split("\n"):
            if line.startswith("问："):
                title = line[2:].strip()[:50]
                break

        # 过滤无意义的短回答（回答部分 < 10 字）
        answer_part = ""
        for line in entry.split("\n"):
            if line.startswith("答："):
                answer_part = line[2:].strip()
                break
        if len(answer_part) < 10:
            continue

        if len(entry) <= DEFAULT_CHUNK_SIZE:
            chunks.append({
                "text": entry,
                "source": "law_qa",
                "title": title,
            })
        else:
            for sc in split_by_sentences(entry):
                chunks.append({
                    "text": sc,
                    "source": "law_qa",
                    "title": title,
                })

    print(f"[分块] law_qa: {len(chunks)} 个 chunks")
    return chunks


def run_chunking():
    """执行全部分块，输出 chunks.jsonl"""
    output_file = PROCESSED_DIR / "chunks.jsonl"

    all_chunks = []
    all_chunks.extend(chunk_law_book_statutes())
    all_chunks.extend(chunk_legal_articles())
    all_chunks.extend(chunk_legal_books())
    all_chunks.extend(chunk_law_qa())

    # 分配全局 chunk_id 并写入 JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(all_chunks):
            chunk["chunk_id"] = i
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\n总计: {len(all_chunks)} 个 chunks -> {output_file}")

    # 统计信息
    lengths = [len(c["text"]) for c in all_chunks]
    print(f"  chunk 长度: 最短={min(lengths)}, 最长={max(lengths)}, "
          f"平均={sum(lengths) / len(lengths):.0f}")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("第一阶段：格式转换")
    print("=" * 50)
    process_law_book()
    process_legal_article()
    process_legal_book()
    process_lawzhidao()

    print(f"\n{'=' * 50}")
    print("第二阶段：递归分块")
    print("=" * 50)
    run_chunking()

    print("\n全部预处理完成！")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--chunk-only":
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        run_chunking()
    else:
        main()

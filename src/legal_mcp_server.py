"""
法律 RAG MCP Server
- 将法律检索能力封装为 MCP (Model Context Protocol) 工具
- DeerFlow 通过 MCP 协议调用本服务
- 启动方式: python legal_mcp_server.py
"""

import sys
import json
from pathlib import Path

# 确保 src 目录在 Python 路径中
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from mcp.server.fastmcp import FastMCP

from retriever import DenseRetriever, BM25Retriever
from merger import weighted_fusion
from reranker import Reranker

# ── 创建 MCP Server ──
mcp = FastMCP(
    "Legal RAG Server",
    description="中国法律法规检索服务，基于向量检索+BM25+Reranker的高精度法条检索",
)

# ── 全局检索器（延迟加载）──
_dense = None
_bm25 = None
_reranker = None


def _ensure_loaded():
    """确保检索模块已加载"""
    global _dense, _bm25, _reranker
    if _dense is None:
        print("[MCP] 加载向量检索器...")
        _dense = DenseRetriever(model_name="BAAI/bge-small-zh-v1.5")
        _dense.load()

        print("[MCP] 加载 BM25 检索器...")
        _bm25 = BM25Retriever()
        _bm25.load()

        print("[MCP] 加载 Reranker...")
        _reranker = Reranker(model_name="BAAI/bge-reranker-v2-m3")
        _reranker.load()

        print("[MCP] 所有模块加载完成！")


@mcp.tool()
def legal_search(query: str, top_k: int = 5) -> str:
    """从中国法律法规数据库中检索相关法条。

    使用向量检索+BM25双路召回+Reranker重排序，从11万+法律条文中
    精准检索最相关的法律规定。适用于：
    - 查找特定法律问题对应的法条
    - 审查合同条款是否合规
    - 法律咨询中的法规依据检索

    Args:
        query: 法律检索查询，例如"试用期最长多久"、"盗窃罪量刑标准"
        top_k: 返回结果数量，默认5条

    Returns:
        检索到的法律条文，包含来源和相关度评分
    """
    _ensure_loaded()

    # 双路召回
    dense_results = _dense.search(query, top_k=20)
    bm25_results = _bm25.search(query, top_k=20)

    # RRF 融合
    merged = weighted_fusion(dense_results, bm25_results, top_k=20)

    # Reranker 重排
    ranked = _reranker.rerank(query, merged, top_k=top_k)

    # 格式化输出
    output_parts = []
    for i, r in enumerate(ranked):
        source = r.get("source", "未知来源")
        section = r.get("section", "")
        score = r.get("rerank_score", 0.0)
        text = r["text"]

        part = f"【法条 {i+1}】(相关度: {score:.4f})\n"
        part += f"来源: {source}"
        if section:
            part += f" > {section}"
        part += f"\n内容: {text}\n"
        output_parts.append(part)

    if not output_parts:
        return f"未找到与「{query}」相关的法律条文。"

    header = f"检索到 {len(output_parts)} 条相关法律规定：\n\n"
    return header + "\n".join(output_parts)


@mcp.tool()
def legal_multi_search(queries: list[str], top_k: int = 3) -> str:
    """批量检索多个法律问题的相关法条。

    适用于合同审查等需要同时检索多个法律要点的场景。
    例如审查劳动合同时，同时检索试用期、工资、竞业限制等条款。

    Args:
        queries: 多个检索查询的列表
        top_k: 每个查询返回的结果数量，默认3条

    Returns:
        每个查询对应的法律条文
    """
    _ensure_loaded()

    all_results = []
    for q in queries:
        dense_results = _dense.search(q, top_k=20)
        bm25_results = _bm25.search(q, top_k=20)
        merged = weighted_fusion(dense_results, bm25_results, top_k=20)
        ranked = _reranker.rerank(q, merged, top_k=top_k)

        section = f"## 查询: {q}\n\n"
        for i, r in enumerate(ranked):
            source = r.get("source", "未知来源")
            text = r["text"]
            score = r.get("rerank_score", 0.0)
            section += f"  [{i+1}] ({score:.4f}) {source}: {text}\n\n"

        if not ranked:
            section += "  未找到相关法条。\n\n"

        all_results.append(section)

    return "\n".join(all_results)


if __name__ == "__main__":
    print("[MCP] 启动法律 RAG MCP Server...")
    mcp.run(transport="sse")

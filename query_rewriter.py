"""
Query改写/扩展模块

将用户问题改写成多个相关查询，提升检索召回率

原理：
- 用户问题可能表述不清晰或简短
- LLM根据原始问题生成多个相关查询
- 每个查询检索一部分相关文档
- 最终合并去重，获得更全面的召回结果

支持的策略：
1. Multi-Query Expansion：生成多个同义/相关查询
2. HyDE (Hypothetical Document Embeddings)：让LLM生成假设性回答，再用回答去检索
"""
import os
import requests
from typing import List, Dict, Any, Optional

# 配置
API_KEY = os.getenv("API_KEY")
API_URL = "https://api.minimaxi.com/anthropic/v1/messages"


class QueryRewriter:
    """Query改写器"""

    # Multi-Query的提示模板
    MULTI_QUERY_TEMPLATE = """你是一个专业的搜索查询改写专家。你的任务是将用户的问题改写成3-5个不同的、相关的搜索查询。

要求：
1. 生成的查询应该围绕同一个主题，但用不同的表述方式
2. 可以包括同义词扩展、问题分解、角度变换等
3. 每个查询应该是独立的、完整的问题
4. 只输出查询，不要其他解释

用户问题：{original_query}

改写后的查询（每行一个）："""

    # HyDE的提示模板
    HYDE_TEMPLATE = """你是一个知识库助手。请根据用户的问题，生成一段假设性的回答。

这段回答应该是对用户问题的一个合理、但可能是简化版的答案。
生成回答时，假设你知道知识库中可能包含的相关内容。

用户问题：{question}

假设性回答："""

    def __init__(self):
        self.api_key = API_KEY
        self.api_url = API_URL

    def _call_llm(self, prompt: str) -> str:
        """调用LLM API"""
        messages = [{"role": "user", "content": prompt}]
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "model": "minimax-m2",
            "messages": messages
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            result = response.json()

            for item in result.get("content", []):
                if item.get("type") == "text":
                    return item["text"]
            return ""
        except Exception as e:
            print(f"[QueryRewriter] LLM调用失败: {e}")
            return ""

    def rewrite_multi_query(self, query: str, num_queries: int = 4) -> List[str]:
        """
        Multi-Query策略：生成多个相关查询

        Args:
            query: 原始用户问题
            num_queries: 生成的查询数量

        Returns:
            改写后的查询列表
        """
        prompt = self.MULTI_QUERY_TEMPLATE.format(original_query=query)

        response = self._call_llm(prompt)

        if not response:
            return [query]  # 失败时返回原始查询

        # 解析结果，每行一个查询
        lines = response.strip().split("\n")
        queries = []

        for line in lines:
            # 清理每行
            line = line.strip()
            # 去除可能的序号（如"1. "）
            if line and line[0].isdigit():
                line = line.split(".", 1)[-1].strip()
            if line and (line.startswith("-") or line.startswith("•")):
                line = line[1:].strip()
            if line:
                queries.append(line)

        if not queries:
            return [query]

        # 确保原始查询在列表中
        if query not in queries:
            queries.insert(0, query)

        return queries[:num_queries]

    def rewrite_hyde(self, query: str) -> str:
        """
        HyDE策略：生成假设性回答

        Args:
            query: 原始用户问题

        Returns:
            假设性回答（用于后续检索）
        """
        prompt = self.HYDE_TEMPLATE.format(question=query)

        response = self._call_llm(prompt)

        if not response:
            return query  # 失败时返回原始查询

        return response.strip()

    def rewrite(self, query: str, strategy: str = "multi_query") -> List[str]:
        """
        通用改写接口

        Args:
            query: 原始查询
            strategy: 策略选择
                - "multi_query": 生成多个相关查询
                - "hyde": 生成假设性回答

        Returns:
            改写后的查询列表
        """
        if strategy == "hyde":
            hyde_result = self.rewrite_hyde(query)
            return [hyde_result, query]  # 同时保留原始查询
        else:
            return self.rewrite_multi_query(query)


# ============ 多查询并行检索 ============

def multi_query_search(
    query: str,
    search_func,
    num_queries: int = 4,
    strategy: str = "multi_query"
) -> List[Dict[str, Any]]:
    """
    使用Query改写进行多查询检索

    1. 改写用户问题为多个查询
    2. 并行执行多个检索
    3. 合并结果并去重

    Args:
        query: 原始用户问题
        search_func: 检索函数，签名为 (query: str, top_k: int) -> List[Dict]
        num_queries: 生成的查询数量
        strategy: 改写策略

    Returns:
        合并后的检索结果
    """
    rewriter = QueryRewriter()

    # 1. 改写查询
    queries = rewriter.rewrite(query, strategy=strategy)
    print(f"[MultiQuerySearch] 原始查询: {query}")
    print(f"[MultiQuerySearch] 改写后查询数: {len(queries)}")

    # 2. 并行检索（简化处理，顺序执行）
    all_results = []
    seen_chunks = set()

    for q in queries:
        results = search_func(q, top_k=3)

        for r in results:
            # 去重（基于chunk内容）
            chunk_key = r.get("chunk", "")[:100]  # 用前100字符作为key
            if chunk_key not in seen_chunks:
                seen_chunks.add(chunk_key)
                r["query"] = q  # 标记来源查询
                all_results.append(r)

    print(f"[MultiQuerySearch] 合并后结果数: {len(all_results)}")

    # 3. 按分数排序
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

    return all_results


# ============ 全局实例 ============

_query_rewriter = None


def get_query_rewriter() -> QueryRewriter:
    """获取Query改写器实例"""
    global _query_rewriter
    if _query_rewriter is None:
        _query_rewriter = QueryRewriter()
    return _query_rewriter


def rewrite_query(query: str, strategy: str = "multi_query") -> List[str]:
    """
    便捷函数：改写查询

    Args:
        query: 原始查询
        strategy: 改写策略

    Returns:
        改写后的查询列表
    """
    rewriter = get_query_rewriter()
    return rewriter.rewrite(query, strategy=strategy)
"""
混合检索模块

结合向量检索 + BM25关键词检索，通过RRF(Reciprocal Rank Fusion)融合两个排名结果

原理：
- 向量检索：语义相似度匹配（认识同义词）
- BM25检索：关键词精确匹配（专业术语）
- RRF融合：综合两种检索方式的结果，提升召回率
"""
import os
from typing import List, Dict, Any, Tuple
import jieba
from rank_bm25 import BM25Okapi

import vector_store

# BM25参数
BM25_K1 = 1.5  # 词频饱和度参数
BM25_B = 0.75  # 文档长度归一化参数


class HybridRetriever:
    """混合检索器"""

    def __init__(self):
        self.chunks = []  # 存储所有chunks
        self.chunk_ids = []  # 存储chunk ID到索引的映射
        self.bm25 = None  # BM25索引
        self._initialized = False

    def _initialize_bm25(self):
        """初始化BM25索引"""
        if not self.chunks:
            return

        # 使用jieba分词
        tokenized_chunks = [list(jieba.cut(chunk)) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks, k1=BM25_K1, b=BM25_B)
        self._initialized = True
        print(f"[HybridRetriever] BM25索引已建立，共 {len(self.chunks)} 个文档")

    def load_chunks(self, chunks: List[str], chunk_ids: List[str]):
        """
        加载文档块

        Args:
            chunks: 文档块内容列表
            chunk_ids: 每个chunk的唯一ID
        """
        self.chunks = chunks
        self.chunk_ids = chunk_ids
        self._initialized = False
        self._initialize_bm25()

    def _get_all_chunks_from_chromadb(self) -> Tuple[List[str], List[str]]:
        """从ChromaDB获取所有chunks"""
        client = vector_store.get_chroma_client()
        try:
            collection = client.get_collection(name=vector_store.COLLECTION_NAME)
            all_data = collection.get()

            chunks = []
            chunk_ids = []

            for i, doc in enumerate(all_data.get("documents", [])):
                chunks.append(doc)
                chunk_ids.append(all_data["ids"][i])

            return chunks, chunk_ids
        except Exception as e:
            print(f"[HybridRetriever] 获取ChromaDB数据失败: {e}")
            return [], []

    def search_bm25(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        BM25关键词检索

        Returns:
            List of (chunk_id, score) tuples
        """
        if not self._initialized or self.bm25 is None:
            self._initialize_bm25()

        if not self.chunks:
            return []

        # 分词查询
        query_tokens = list(jieba.cut(query))

        # 获取BM25分数
        scores = self.bm25.get_scores(query_tokens)

        # 排序并返回top_k
        scored_chunks = list(zip(self.chunk_ids, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        return scored_chunks[:top_k]

    def search_vector(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        向量检索

        Returns:
            List of (chunk_id, score) tuples
        """
        results = vector_store.search_vector_store(query, top_k=top_k)

        # 将distance转换为类似BM25的分数（distance越小越相似，转换为负数作为"分数"）
        # 或者直接使用向量检索的rank
        return [(r["metadata"].get("doc_id", 0), -r["distance"]) for r in results]

    @staticmethod
    def reciprocal_rank_fusion(
        results_list: List[List[Tuple[Any, float]]],
        k: int = 60
    ) -> List[Tuple[Any, float]]:
        """
        倒数排名融合（RRF）算法

        RRF公式: score(d) = Σ 1/(k + rank(d))

        Args:
            results_list: 多个检索结果列表，每个列表是 (id, score) 的元组列表
            k: RRF参数，通常设为60

        Returns:
            融合后的排序结果
        """
        rrf_scores = {}

        for results in results_list:
            for rank, (doc_id, _) in enumerate(results):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0
                rrf_scores[doc_id] += 1.0 / (k + rank + 1)  # rank从0开始，所以+1

        # 排序
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_docs

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        混合检索主入口

        1. 并行执行向量检索和BM25检索
        2. 使用RRF融合两个结果
        3. 返回最终排序结果

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            包含chunk内容和融合分数的列表
        """
        # 确保BM25已初始化
        if not self._initialized:
            chunks, chunk_ids = self._get_all_chunks_from_chromadb()
            if chunks:
                self.load_chunks(chunks, chunk_ids)

        # 1. 向量检索
        vector_results = self.search_vector(query, top_k=top_k * 2)

        # 2. BM25检索
        bm25_results = self.search_bm25(query, top_k=top_k * 2)

        # 3. RRF融合
        fused_results = self.reciprocal_rank_fusion(
            [vector_results, bm25_results],
            k=60
        )

        # 4. 构建最终结果
        # 创建chunk_id到内容的映射
        chunk_id_to_content = {cid: chunk for cid, chunk in zip(self.chunk_ids, self.chunks)}

        final_results = []
        for doc_id, rrf_score in fused_results[:top_k]:
            if doc_id in chunk_id_to_content:
                final_results.append({
                    "chunk": chunk_id_to_content[doc_id],
                    "score": rrf_score,
                    "doc_id": doc_id
                })

        return final_results


# ============ 全局混合检索器实例 ============

_hybrid_retriever = None


def get_hybrid_retriever() -> HybridRetriever:
    """获取或初始化全局混合检索器"""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
        # 从ChromaDB加载现有chunks
        client = vector_store.get_chroma_client()
        try:
            collection = client.get_collection(name=vector_store.COLLECTION_NAME)
            all_data = collection.get()

            if all_data.get("documents"):
                chunks = all_data["documents"]
                chunk_ids = all_data["ids"]
                _hybrid_retriever.load_chunks(chunks, chunk_ids)
                print(f"[HybridRetriever] 已加载 {len(chunks)} 个chunks")
        except Exception as e:
            print(f"[HybridRetriever] 初始化失败: {e}")

    return _hybrid_retriever


def hybrid_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    便捷函数：使用混合检索搜索文档

    Args:
        query: 查询文本
        top_k: 返回结果数量

    Returns:
        检索结果列表
    """
    retriever = get_hybrid_retriever()
    return retriever.search(query, top_k=top_k)


# ============ 初始化函数 ============

def init_hybrid_retriever():
    """初始化混合检索器（从ChromaDB加载数据）"""
    retriever = get_hybrid_retriever()
    return retriever is not None
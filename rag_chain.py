"""
LangChain RAG链实现

提供基于LangChain的RAG问答能力，支持：
- LangChain Document格式
- RetrievalQA链
- 带源引用的回答
"""
# 必须最先设置环境变量（在任何导入之前）
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from typing import List, Dict, Any, Optional, Tuple

# LangChain核心
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# LangChain Community - ChromaDB封装
from langchain_community.vectorstores import Chroma

# LangChain Text Splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 自定义模块
import vector_store

# ============ 配置 ============
API_KEY = os.getenv("API_KEY")
API_URL = "https://api.minimaxi.com/anthropic/v1/messages"
HF_ENDPOINT = "https://hf-mirror.com"

# LangChain嵌入模型配置
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


class SentenceTransformerEmbeddings:
    """
    自定义的句子嵌入类，包装我们已有的vector_store模型

    LangChain的Chroma需要传入一个embedding_function参数，
    我们使用已有的vector_store的预加载模型，避免重复下载
    """

    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档"""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        embedding = self.model.encode([text])
        return embedding[0].tolist()


def get_embedding_function():
    """获取LangChain兼容的嵌入函数

    每次调用时确保模型已加载（延迟初始化）
    解决uvicorn多worker进程中模型不共享的问题
    """
    # 尝试获取已有模型，如果没有则加载
    model = vector_store.get_model()
    if model is None:
        print("[LangChain] Worker进程中加载模型...")
        vector_store.preload_model()
        model = vector_store.get_model()

    if model is None:
        raise RuntimeError("模型加载失败")

    return SentenceTransformerEmbeddings(model)


def get_vectorstore() -> Chroma:
    """
    获取LangChain的Chroma向量存储

    复用vector_store.py中的ChromaDB配置
    """
    embedding_function = get_embedding_function()

    # 复用现有的chroma_db目录
    vectorstore = Chroma(
        persist_directory=vector_store.CHROMA_PATH,
        embedding_function=embedding_function,
        collection_name=vector_store.COLLECTION_NAME
    )

    return vectorstore


def create_rag_chain():
    """
    创建LangChain RAG链

    返回一个可调用的RAG链：
    输入: question (str)
    输出: {"answer": str, "sources": List[Dict]}
    """
    # 获取向量存储
    vectorstore = get_vectorstore()

    # 创建检索器
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # 定义提示模板
    template = """你是一个专业的知识库问答助手。请基于以下检索到的文档内容回答用户的问题。

如果检索到的文档中没有相关信息，请如实说明"未找到相关信息"，不要编造答案。

检索到的文档：
{context}

用户问题：{question}

请给出准确、简洁的回答。如果提供了文档来源，请注明。"""

    prompt = ChatPromptTemplate.from_template(template)

    # 定义回答生成函数
    def generate_answer(question: str, context: str) -> str:
        """调用MiniMax API生成回答"""
        full_prompt = f"{context}\n\n用户问题：{question}\n\n请给出回答："

        import requests

        messages = [{"role": "user", "content": full_prompt}]
        headers = {
            "x-api-key": API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "model": "minimax-m2",
            "messages": messages
        }

        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            result = response.json()

            for item in result.get("content", []):
                if item.get("type") == "text":
                    return item["text"]
            return "抱歉，无法生成回答。"
        except Exception as e:
            return f"调用AI服务时出错：{str(e)}"

    # 构建RAG链
    def rag_chain_inner(question: str) -> Dict[str, Any]:
        """RAG链的核心逻辑"""
        # 1. 检索相关文档（使用新的invoke API）
        try:
            docs = retriever.invoke(question)
        except Exception as e:
            print(f"[RAG Chain] 检索失败: {e}")
            docs = []

        if not docs:
            return {
                "answer": "抱歉，知识库中没有找到与您问题相关的内容。",
                "sources": []
            }

        # 2. 构建上下文
        context_parts = []
        sources = []

        for i, doc in enumerate(docs):
            source = doc.metadata.get("filename", "未知来源")
            context_parts.append(f"[文档{i+1}] {doc.page_content}")
            sources.append({
                "content": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                "source": source,
                "metadata": doc.metadata
            })

        context = "\n\n".join(context_parts)

        # 3. 生成回答
        answer = generate_answer(question, context)

        return {
            "answer": answer,
            "sources": sources
        }

    return rag_chain_inner


def add_document_with_langchain(chunks: List[str], doc_id: int, filename: str):
    """
    使用LangChain添加文档到向量存储

    Args:
        chunks: 分块后的文本列表
        doc_id: 文档ID
        filename: 文件名
    """
    if not chunks:
        return 0

    # 创建LangChain Document对象
    documents = [
        Document(
            page_content=chunk,
            metadata={"doc_id": doc_id, "filename": filename, "chunk_index": i}
        )
        for i, chunk in enumerate(chunks)
    ]

    # 获取向量存储
    vectorstore = get_vectorstore()

    # 添加文档
    vectorstore.add_documents(documents)

    print(f"[LangChain] 已添加 {len(documents)} 个文档块到向量存储")
    return len(documents)


def search_with_langchain(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    使用LangChain进行语义检索

    Args:
        query: 查询文本
        top_k: 返回结果数量

    Returns:
        包含chunk内容和元数据的列表
    """
    vectorstore = get_vectorstore()

    # 执行相似度搜索
    docs = vectorstore.similarity_search_with_score(query, k=top_k)

    results = []
    for doc, score in docs:
        results.append({
            "chunk": doc.page_content,
            "score": score,
            "metadata": doc.metadata
        })

    return results


# ============ LangChain RAG Chain 实例 ============

# 全局RAG链（延迟初始化）
_rag_chain = None


def get_rag_chain():
    """获取或初始化RAG链"""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = create_rag_chain()
    return _rag_chain


def chat_with_rag_chain(question: str) -> Dict[str, Any]:
    """
    使用LangChain RAG链进行问答

    Args:
        question: 用户问题

    Returns:
        包含回答和来源的字典
    """
    rag_chain = get_rag_chain()
    return rag_chain(question)
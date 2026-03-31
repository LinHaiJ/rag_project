import chromadb
from chromadb.config import Settings
import os

# 使用清华镜像源（用于国内访问HuggingFace）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ChromaDB 配置
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "rag_chunks"

# 嵌入模型配置
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
model = None
_model_loading_error = None


def get_model():
    """获取或加载嵌入模型"""
    global model, _model_loading_error

    if model is not None:
        return model

    try:
        from sentence_transformers import SentenceTransformer
        print(f"加载嵌入模型: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)
        print("模型加载成功")
        return model
    except Exception as e:
        _model_loading_error = str(e)
        print(f"模型加载失败: {e}")
        return None


def preload_model():
    """预加载模型（同步阻塞方式）"""
    global model, _model_loading_error

    if model is not None:
        print("模型已预加载")
        return

    try:
        from sentence_transformers import SentenceTransformer
        print(f"同步加载嵌入模型: {MODEL_NAME}...")
        model = SentenceTransformer(MODEL_NAME)
        _model_loading_error = None
        print("模型预加载完成")
    except Exception as e:
        _model_loading_error = str(e)
        print(f"模型预加载失败: {e}")
        raise


def get_chroma_client():
    """获取 ChromaDB 客户端"""
    return chromadb.PersistentClient(path=CHROMA_PATH)


def init_vector_store():
    """初始化向量数据库"""
    client = get_chroma_client()
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"向量数据库已存在，共 {collection.count()} 个块")
    except:
        collection = client.create_collection(name=COLLECTION_NAME)
        print("创建新的向量数据库集合")


def add_chunks_to_vector_store(chunks: list, doc_id: int, filename: str):
    """将文档块添加到向量数据库"""
    if not chunks:
        return 0

    embed_model = get_model()
    if embed_model is None:
        print("错误: 模型未加载，无法进行向量存储")
        return 0

    client = get_chroma_client()

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except:
        collection = client.create_collection(name=COLLECTION_NAME)

    # 批量生成嵌入
    embeddings = embed_model.encode(chunks, show_progress_bar=True).tolist()

    # 生成每个chunk的唯一ID
    chunk_ids = [f"doc_{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"doc_id": doc_id, "filename": filename} for _ in chunks]

    collection.add(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )

    print(f"已将 {len(chunks)} 个块添加到向量数据库")
    return len(chunks)


def search_vector_store(query: str, top_k: int = 3):
    """在向量数据库中搜索最相关的块"""
    embed_model = get_model()
    if embed_model is None:
        return []

    client = get_chroma_client()

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except:
        return []

    # 生成查询的嵌入
    query_embedding = embed_model.encode([query]).tolist()[0]

    # 执行向量检索
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # 整理结果
    chunks_with_scores = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            distance = results["distances"][0][i] if results["distances"] else 0
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            chunks_with_scores.append({
                "chunk": doc,
                "distance": distance,
                "metadata": metadata
            })

    return chunks_with_scores


def delete_chunks_by_doc_id(doc_id: int):
    """删除指定文档的所有块"""
    client = get_chroma_client()
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        all_data = collection.get()

        ids_to_delete = []
        for i, metadata in enumerate(all_data.get("metadatas", [])):
            if metadata and metadata.get("doc_id") == doc_id:
                ids_to_delete.append(all_data["ids"][i])

        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            print(f"从向量数据库删除 {len(ids_to_delete)} 个块")
    except Exception as e:
        print(f"删除向量数据时出错: {e}")
import os
import re
import uuid
import aiosqlite
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

import database
import vector_store

# 新增模块
import rag_chain
import hybrid_retriever
import query_rewriter

# ============ 配置 ============
API_KEY = os.getenv("API_KEY")
API_URL = "https://api.minimaxi.com/anthropic/v1/messages"
UPLOAD_FOLDER = "uploaded_docs"

app = FastAPI(title="RAG知识库API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def split_chunks(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    智能分块策略：按句子/段落分块，保留上下文重叠

    Args:
        text: 原始文本
        chunk_size: 每个块的目标字符数
        overlap: 相邻块之间的重叠字符数

    Returns:
        分块后的文本列表
    """
    # 清理文本
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 先按段落分割（双换行）
    paragraphs = re.split(r"\n\n+", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = ""
    current_size = 0

    for para in paragraphs:
        para_size = len(para)

        # 如果段落本身就超过chunk_size，需要进一步拆分
        if para_size > chunk_size:
            # 保存当前chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
                # 保留重叠部分
                current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
                current_size = len(current_chunk)

            # 按句子拆分大段落
            sentences = re.split(r"(?<=[。！？.!?])", para)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                sentence_size = len(sentence)
                if sentence_size > chunk_size:
                    # 极长的句子，直接截断
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                        current_size = 0
                    chunks.append(sentence[:chunk_size])
                elif current_size + sentence_size <= chunk_size:
                    current_chunk += sentence
                    current_size += sentence_size
                else:
                    # 当前chunk满了，保存并开始新的
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                    current_size = sentence_size
        else:
            # 普通段落
            if current_size + para_size <= chunk_size:
                current_chunk += "\n" + para
                current_size += para_size + 1
            else:
                # 当前chunk满了
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
                current_size = para_size

    # 处理最后一个chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return [c for c in chunks if c]


async def chat_with_ai(messages):
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "model": "minimax-m2",
        "messages": messages
    }
    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    result = response.json()

    for item in result.get("content", []):
        if item.get("type") == "text":
            return item["text"]
    return ""


@app.on_event("startup")
async def startup():
    await database.init_db()
    vector_store.init_vector_store()
    # 同步预加载模型，确保在处理请求前模型已就绪
    vector_store.preload_model()
    # 初始化混合检索器
    hybrid_retriever.init_hybrid_retriever()
    print("[API] 启动完成，所有组件已就绪")


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    content = await file.read()

    with open(filepath, "wb") as f:
        f.write(content)

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # 使用智能分块
    chunks = split_chunks(text)

    # 保存到SQLite
    async with aiosqlite.connect(database.DATABASE_PATH) as db:
        await db.execute(
            "INSERT INTO documents (filename, filepath, content, chunk_count) VALUES (?, ?, ?, ?)",
            (file.filename, filepath, text, len(chunks))
        )
        await db.commit()
        cursor = await db.execute("SELECT last_insert_rowid()")
        doc_id = (await cursor.fetchone())[0]

        # 保存每个chunk到数据库
        for i, chunk_content in enumerate(chunks):
            await db.execute(
                "INSERT INTO chunks (doc_id, chunk_index, content) VALUES (?, ?, ?)",
                (doc_id, i, chunk_content)
            )
        await db.commit()

    # 添加到向量数据库
    vector_store.add_chunks_to_vector_store(chunks, doc_id, file.filename)

    return {"success": True, "document_id": doc_id, "filename": file.filename, "chunk_count": len(chunks)}


@app.get("/api/documents")
async def list_documents():
    async with aiosqlite.connect(database.DATABASE_PATH) as db:
        cursor = await db.execute(
            "SELECT id, filename, chunk_count, created_at FROM documents ORDER BY created_at DESC"
        )
        rows = await cursor.fetchall()

    documents = [{"id": row[0], "filename": row[1], "chunk_count": row[2], "created_at": row[3]} for row in rows]
    return {"documents": documents}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: int):
    async with aiosqlite.connect(database.DATABASE_PATH) as db:
        cursor = await db.execute("SELECT filepath FROM documents WHERE id = ?", (doc_id,))
        row = await cursor.fetchone()

        if row:
            filepath = row[0]
            if os.path.exists(filepath):
                os.remove(filepath)

            # 删除chunks
            await db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            await db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            await db.commit()

            # 从向量数据库删除
            vector_store.delete_chunks_by_doc_id(doc_id)

            return {"success": True}

    return {"success": False, "error": "文档不存在"}


class ChatRequest(BaseModel):
    session_id: str
    message: str
    use_rag: bool = True


class ChatRequestV2(BaseModel):
    """增强版聊天请求，支持高级RAG功能"""
    session_id: str
    message: str
    use_rag: bool = True
    use_hybrid: bool = False  # 使用混合检索
    use_langchain: bool = False  # 使用LangChain RAG
    use_query_rewrite: bool = False  # 使用Query改写


@app.post("/api/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id
    message = request.message

    async with aiosqlite.connect(database.DATABASE_PATH) as db:
        cursor = await db.execute(
            "SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY created_at",
            (session_id,)
        )
        rows = await cursor.fetchall()

    messages = [{"role": row[0], "content": row[1]} for row in rows]

    if request.use_rag:
        # 使用向量检索
        results = vector_store.search_vector_store(message, top_k=3)

        if results:
            # 构建上下文
            context = "\n\n".join([r["chunk"] for r in results])
            sources = [r["metadata"].get("filename", "未知") for r in results]
            source_str = "、".join(set(sources))

            system_prompt = f"""基于以下文档内容回答问题。如果文档中没有相关信息，请如实说明。

文档来源：{source_str}

文档内容：
{context}

问题：{message}

回答："""
            messages.append({"role": "user", "content": system_prompt})
        else:
            return {"reply": "抱歉，知识库为空或没有找到相关内容。请先上传文档后再进行RAG检索。"}

    else:
        # RAG关闭，直接回答
        messages.append({"role": "user", "content": message})

    ai_reply = await chat_with_ai(messages)

    async with aiosqlite.connect(database.DATABASE_PATH) as db:
        await db.execute(
            "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, "user", message)
        )
        await db.execute(
            "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, "assistant", ai_reply)
        )
        await db.commit()

    return {"reply": ai_reply}


@app.get("/api/chat/{session_id}")
async def get_chat_history(session_id: str):
    async with aiosqlite.connect(database.DATABASE_PATH) as db:
        cursor = await db.execute(
            "SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY created_at",
            (session_id,)
        )
        rows = await cursor.fetchall()

    messages = [{"role": row[0], "content": row[1]} for row in rows]
    return {"messages": messages}


@app.post("/api/chat-v2")
async def chat_v2(request: ChatRequestV2):
    """
    增强版RAG问答接口

    支持的功能组合：
    - 基础RAG：纯向量检索
    - 混合检索：向量 + BM25
    - LangChain RAG：使用LangChain框架
    - Query改写：多查询扩展
    """
    session_id = request.session_id
    message = request.message

    # 获取对话历史
    async with aiosqlite.connect(database.DATABASE_PATH) as db:
        cursor = await db.execute(
            "SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY created_at",
            (session_id,)
        )
        rows = await cursor.fetchall()

    messages = [{"role": row[0], "content": row[1]} for row in rows]

    if not request.use_rag:
        # 关闭RAG，直接回答
        messages.append({"role": "user", "content": message})
        ai_reply = await chat_with_ai(messages)
    else:
        # ========== RAG检索 ==========
        retrieval_results = []

        if request.use_langchain:
            # 使用LangChain RAG
            # 注意：LangChain模式目前为实验性功能，如遇问题会使用向量检索替代
            print(f"[ChatV2] 使用LangChain RAG")
            langchain_success = False
            try:
                rag_result = rag_chain.chat_with_rag_chain(message)
                ai_reply = rag_result["answer"]
                retrieval_results = [
                    {"chunk": s["content"], "metadata": {"filename": s.get("source", "未知")}}
                    for s in rag_result.get("sources", [])
                ]
                langchain_success = True
                mode = "langchain"
            except Exception as e:
                print(f"[ChatV2] LangChain RAG出错，回退到向量检索: {e}")
                mode = "langchain_fallback"

            if not langchain_success:
                # 回退到普通向量检索
                retrieval_results = vector_store.search_vector_store(message, top_k=3)
                if retrieval_results:
                    context = "\n\n".join([r["chunk"] for r in retrieval_results])
                    sources = [r["metadata"].get("filename", "未知") for r in retrieval_results]
                    system_prompt = f"""基于以下文档内容回答问题。如果文档中没有相关信息，请如实说明。

文档来源：{', '.join(set(sources))}

文档内容：
{context}

问题：{message}

回答："""
                    messages.append({"role": "user", "content": system_prompt})
                    ai_reply = await chat_with_ai(messages)
                else:
                    ai_reply = "抱歉，知识库为空或没有找到相关内容。"

            # 保存对话历史
            async with aiosqlite.connect(database.DATABASE_PATH) as db:
                await db.execute(
                    "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
                    (session_id, "user", message)
                )
                await db.execute(
                    "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
                    (session_id, "assistant", ai_reply)
                )
                await db.commit()

            return {
                "reply": ai_reply,
                "sources": [{"chunk": r["chunk"], "metadata": r.get("metadata", {})} for r in retrieval_results],
                "mode": mode
            }

        else:
            # 使用原生检索
            if request.use_query_rewrite:
                # Query改写 + 多查询检索
                print(f"[ChatV2] 使用Query改写")

                def search_func(query, top_k):
                    return vector_store.search_vector_store(query, top_k=top_k)

                # 先改写查询
                queries = query_rewriter.rewrite_query(message, strategy="multi_query")
                print(f"[ChatV2] 改写后的查询: {queries}")

                # 多查询检索并合并
                all_results = []
                seen_chunks = set()
                for q in queries:
                    results = vector_store.search_vector_store(q, top_k=3)
                    for r in results:
                        chunk_key = r["chunk"][:100]
                        if chunk_key not in seen_chunks:
                            seen_chunks.add(chunk_key)
                            all_results.append(r)

                # 按相似度排序
                all_results.sort(key=lambda x: x.get("distance", 0))
                retrieval_results = all_results[:3]

            elif request.use_hybrid:
                # 混合检索
                print(f"[ChatV2] 使用混合检索")
                retrieval_results = hybrid_retriever.hybrid_search(message, top_k=5)
                print(f"[ChatV2] 混合检索返回 {len(retrieval_results)} 个结果")

            else:
                # 基础向量检索
                print(f"[ChatV2] 使用基础向量检索")
                retrieval_results = vector_store.search_vector_store(message, top_k=3)

            if retrieval_results:
                # 构建上下文
                context = "\n\n".join([r["chunk"] for r in retrieval_results])
                sources = [r.get("metadata", {}).get("filename", "未知") for r in retrieval_results]
                source_str = "、".join(set(sources))

                system_prompt = f"""基于以下文档内容回答问题。如果文档中没有相关信息，请如实说明。

文档来源：{source_str}

文档内容：
{context}

问题：{message}

回答："""
                messages.append({"role": "user", "content": system_prompt})
                ai_reply = await chat_with_ai(messages)
            else:
                return {
                    "reply": "抱歉，知识库为空或没有找到相关内容。请先上传文档后再进行RAG检索。",
                    "sources": [],
                    "mode": "hybrid" if request.use_hybrid else ("langchain" if request.use_langchain else "vector")
                }

    # 保存对话历史
    async with aiosqlite.connect(database.DATABASE_PATH) as db:
        await db.execute(
            "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, "user", message)
        )
        await db.execute(
            "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, "assistant", ai_reply)
        )
        await db.commit()

    # 构建返回
    response = {
        "reply": ai_reply,
        "sources": [{"chunk": r["chunk"], "metadata": r.get("metadata", {})} for r in retrieval_results],
        "mode": "hybrid" if request.use_hybrid else ("langchain" if request.use_langchain else ("query_rewrite" if request.use_query_rewrite else "vector"))
    }

    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
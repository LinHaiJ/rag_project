# import requests
# import json
# import chromadb

# API_KEY = "sk-cp--UAHVBzvskROZc4uYgrtMZyWIjc4-ShYHwu3vEMsGNqNktcEn6krLldGnyZ2ByEiDjmsoA20KZ94jmMho3DfohZPmXXulIdN82cuzOGoXfHqikEhjJSzzh8"
# API_URL = "https://api.minimaxi.com/anthropic/v1/messages"

# # 使用 ChromaDB 内置嵌入（离线可用）
# print("初始化向量数据库...")
# client = chromadb.Client()

# def chat(messages):
#     headers = {
#         "x-api-key": API_KEY,
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "model": "minimax-m2.7",
#         "messages": messages
#     }
#     response = requests.post(API_URL, headers=headers, json=payload)
#     result = response.json()

#     for item in result["content"]:
#         if item["type"] == "text":
#             return item["text"]
#     return ""

# def load_documents(folder_path):
#     import os
#     docs = []

#     for filename in os.listdir(folder_path):
#         if filename.endswith(".txt"):
#             filepath = os.path.join(folder_path, filename)
#             with open(filepath, "r", encoding="utf-8") as f:
#                 content = f.read()
#                 docs.append({"name": filename, "content": content})

#     return docs

# def split_chunks(text, chunk_size=200):
#     chunks = []
#     for i in range(0, len(text), chunk_size):
#         chunk = text[i:i+chunk_size]
#         if chunk.strip():
#             chunks.append(chunk)
#     return chunks

# def create_vector_db(docs, chunk_size=200):
#     """创建向量数据库（使用ChromaDB内置嵌入）"""
#     # 删除已有的集合（如果存在）
#     try:
#         client.delete_collection("documents")
#     except:
#         pass

#     # 创建集合，ChromaDB会自动使用内置嵌入函数
#     collection = client.create_collection("documents")

#     # 存储所有块
#     all_chunks = []
#     for doc in docs:
#         chunks = split_chunks(doc["content"], chunk_size)
#         for i, chunk in enumerate(chunks):
#             all_chunks.append({
#                 "id": f"{doc['name']}_{i}",
#                 "text": chunk,
#                 "metadata": {"source": doc["name"]}
#             })

#     # 添加到集合（ChromaDB会自动生成向量）
#     collection.add(
#         ids=[c["id"] for c in all_chunks],
#         documents=[c["text"] for c in all_chunks],
#         metadatas=[c["metadata"] for c in all_chunks]
#     )

#     return collection, all_chunks

# def retrieve_relevant_chunks(query, collection, top_k=3):
#     """向量检索：找出最相关的块"""
#     # 在向量数据库中搜索（ChromaDB会自动把query转成向量）
#     results = collection.query(
#         query_texts=[query],
#         n_results=top_k
#     )

#     # 返回找到的文档块
#     return results["documents"][0]

# def generate_answer(query, relevant_chunks):
#     """基于相关文档片段生成回答"""
#     context = "\n\n".join(relevant_chunks)

#     prompt = f"""基于以下文档内容回答问题。如果文档中没有相关信息，请回答"文档中没有相关内容"。

# 文档内容：
# {context}

# 问题：{query}

# 回答："""

#     return chat([{"role": "user", "content": prompt}])

# # 主程序
# print("=" * 30)
# print("RAG知识库问答程序（向量检索版 - ChromaDB内置嵌入）")
# print("=" * 30)

# # 加载文档
# docs = load_documents("docs")

# if not docs:
#     print("请先在 docs 文件夹中放入 .txt 文档")
# else:
#     print(f"已加载 {len(docs)} 个文档")

#     # 创建向量数据库
#     print("正在创建向量数据库...")
#     collection, all_chunks = create_vector_db(docs)
#     print(f"共处理 {len(all_chunks)} 个文本块\n")

# # 问答循环
# while True:
#     query = input("你：")
#     if query == "quit":
#         break

#     relevant = retrieve_relevant_chunks(query, collection)

#     if not relevant:
#         print("AI：抱歉，没有找到相关文档内容\n")
#     else:
#         print(f"找到 {len(relevant)} 个相关片段：")
#         for i, chunk in enumerate(relevant):
#             print(f"  [{i+1}] {chunk[:100]}...")
#         print()

#         answer = generate_answer(query, relevant)
#         print("AI：", answer)


# =============================================================================
# 以下是使用 sentence-transformers 的旧版本（需要联网下载模型）
# =============================================================================
import requests
import json
import chromadb
from sentence_transformers import SentenceTransformer

API_KEY = os.getenv("API_KEY")
API_URL = "https://api.minimaxi.com/anthropic/v1/messages"

# 加载嵌入模型
print("加载嵌入模型...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("模型加载完成！")

def chat(messages):
    headers = {
        "x-api-key":API_KEY,
        "Content-Type":"application/json"
    }
    payload = {
        "model":"minimax-m2.7",
        "messages":messages
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    result = response.json()

    for item in result["content"]:
        if item["type"] == "text":
            return item["text"]
    return ""

def load_documents(folder_path):
    import os
    docs = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                docs.append({"name": filename, "content": content})

    return docs

def split_chunks(text, chunk_size=200):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def create_vector_db(docs, chunk_size=200):
    """创建向量数据库"""
    client = chromadb.Client()

    try:
        client.delete_collection("documents")
    except:
        pass

    collection = client.create_collection("documents")

    all_chunks = []
    for doc in docs:
        chunks = split_chunks(doc["content"], chunk_size)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"{doc['name']}_{i}",
                "text": chunk,
                "metadata": {"source": doc["name"]}
            })

    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts).tolist()

    collection.add(
        ids=[c["id"] for c in all_chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[c["metadata"] for c in all_chunks]
    )

    return collection, all_chunks

def retrieve_relevant_chunks(query, collection, top_k=3):
    """向量检索：找出最相关的块"""
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    return results["documents"][0]

def generate_answer(query, relevant_chunks):
    """基于相关文档片段生成回答"""
    context = "\n\n".join(relevant_chunks)

    prompt = f"""基于以下文档内容回答问题。如果文档中没有相关信息，请回答"文档中没有相关内容"。

文档内容：
{context}

问题：{query}

回答："""

    return chat([{"role": "user", "content": prompt}])

print("=" * 30)
print("RAG知识库问答程序（向量检索版）")
print("=" * 30)

docs = load_documents("docs")

if not docs:
    print("请先在 docs 文件夹中放入 .txt 文档")
else:
    print(f"已加载 {len(docs)} 个文档")
    print("正在创建向量数据库...")
    collection, all_chunks = create_vector_db(docs)
    print(f"共处理 {len(all_chunks)} 个文本块\n")

while True:
    query = input("你：")
    if query == "quit":
        break

    relevant = retrieve_relevant_chunks(query, collection)

    if not relevant:
        print("AI：抱歉，没有找到相关文档内容\n")
    else:
        print(f"找到 {len(relevant)} 个相关片段：")
        for i, chunk in enumerate(relevant):
            print(f"  [{i+1}] {chunk[:100]}...")
        print()

        answer = generate_answer(query, relevant)
        print("AI：", answer)

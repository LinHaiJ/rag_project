import os
import requests
import json

API_KEY = os.getenv("API_KEY")
API_URL = "https://api.minimaxi.com/anthropic/v1/messages"

def chat(messages):
    headers = {#请求头
        "x-api-key":API_KEY,
        "content-Type":"application/json"#格式
    }
    payload = {#请求体
        "model":"minimax-m2.7",
        "messages": messages
    }
    response = requests.post(API_URL, headers=headers, json=payload)#发请求、发给谁、把payload 自动变成 JSON 发给服务器
    result = response.json()#把 AI 返回的 JSON 答案 → 变成 Python 字典。
    for item in result["content"]:
        if item["type"] == "text":
            return item["text"]
    return ""

def load_documents(folder_path):
    """加载文档"""
    import os
    docs = []
    
    for filename in os.listdir(folder_path):#列出文件列表
        if filename.endswith(".txt"):#根据文件后缀招
            filepath = os.path.join(folder_path, filename)#拼接路径
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()#读取文本内容
                docs.append({"name": filename, "content": content})#添加到docs

    return docs
def split_chunks(text, chunk_size=200):#切分文本
    chunks = []
    for i in range(0, len(text), chunk_size):#用chunk_size的步长切
        chunk = text[i:i+chunk_size]#从 i 开始，到 i+chunk_size 结束
        chunks.append(chunk)
    return chunks

def retrieve_relevant_chunks(query, chunks, top_k=2):
    """简单检索：找出包含查询关键词的块"""
    query_words = set(query)#将字符串拆成一个一个，并去重
    relevant = []
    
    for chunk in chunks:
        # 简单的关键词匹配
        chunk_words = set(chunk)
        overlap = len(query_words & chunk_words)#求交集数量
        if overlap > 0:#有匹配的就加入字符串
            relevant.append((overlap, chunk))
    
    # 按相关度排序
    relevant.sort(reverse=True)
    return [chunk for _, chunk in relevant[:top_k]]#取前top_k个最相关的
    #只取 chunk，不要前面的分数
def generate_answer(query, relevant_chunks):
    """基于相关文档片段生成回答"""
    context = "\n\n".join(relevant_chunks)#让文本空一格
    
    prompt = f"""基于以下文档内容回答问题。如果文档中没有相关信息，请回答"文档中没有相关内容"。

文档内容：
{context}

问题：{query}

回答："""
    
    return chat([{"role": "user", "content": prompt}])
# 主程序
print("RAG知识库问答程序")
print("=" * 30)

# 加载文档
docs = load_documents("docs")

if not docs:
    print("请先在 docs 文件夹中放入 .txt 文档")
else:
    print(f"已加载 {len(docs)} 个文档")
    
    # 将所有文档内容切块
    all_chunks = []
    for doc in docs:
        chunks = split_chunks(doc["content"])
        all_chunks.extend(chunks)
    
    print(f"共切分成 {len(all_chunks)} 个文本块\n")

# 问答循环
while True:
    query = input("你：")
    if query == "quit":
        break
    
    relevant = retrieve_relevant_chunks(query, all_chunks)
    
    if not relevant:
        print("AI：抱歉，没有找到相关文档内容\n")
    else:
        answer = generate_answer(query, relevant)
        print("AI：", answer)
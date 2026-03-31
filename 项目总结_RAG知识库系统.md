# RAG知识库系统 - 项目总结

## 一、项目概述

这是一个基于检索增强生成（RAG）的知识库问答系统，支持上传文档、向量检索、智能问答。

**核心功能**：
- 文档上传与分块存储
- 多种检索模式（向量/混合/Query改写/LangChain）
- 多轮对话上下文支持
- Web界面交互

---

## 二、系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      用户浏览器                             │
│                   http://localhost:8501                     │
└─────────────────────┬─────────────────────────────────────┘
                      │ HTTP请求
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit 前端                           │
│                   frontend.py (端口8501)                     │
│  - 文档上传/管理    - 聊天界面    - RAG开关                │
└─────────────────────┬─────────────────────────────────────┘
                      │ HTTP JSON API
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI 后端                             │
│                   api.py (端口8000)                          │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ upload_doc   │  │ chat         │  │ chat-v2      │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                 │                  │             │
│         ▼                 ▼                  ▼             │
│  ┌──────────────┐  ┌──────────────────────────────┐        │
│  │ 智能分块      │  │ 多种检索模式                  │        │
│  │ split_chunks │  │ - 向量检索 (vector_store)    │        │
│  └──────────────┘  │ - 混合检索 (hybrid_retriever)│        │
│                    │ - Query改写 (query_rewriter) │        │
│                    │ - LangChain RAG (rag_chain)  │        │
│                    └──────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| 后端框架 | FastAPI | RESTful API服务 |
| 数据库 | SQLite + aiosqlite | 文档和对话存储 |
| 向量数据库 | ChromaDB | 语义检索 |
| 嵌入模型 | sentence-transformers | 文本向量化 |
| RAG框架 | LangChain | RAG链编排 |
| 关键词检索 | BM25 (rank-bm25) | 关键词精确匹配 |
| Query处理 | Multi-Query Expansion | 查询改写扩展 |
| 前端框架 | Streamlit | Web交互界面 |
| 容器化 | Docker | 部署 |
| AI接口 | MiniMax API | LLM对话 |

---

## 四、核心模块详解

### 4.1 database.py - 数据库模块

```python
# 三个核心表
documents  # 文档表
chunks     # 文档块表
chat_history  # 对话历史表
```

**设计思路**：
- `documents`: 存储文件元信息
- `chunks`: 存储分块后的文本，与documents一对多关系
- `chat_history`: 存储对话历史，支持多会话

---

### 4.2 vector_store.py - 向量存储模块

**核心功能**：

```python
# 1. 预加载模型（启动时调用）
preload_model()

# 2. 初始化向量数据库
init_vector_store()

# 3. 添加文档块到向量库
add_chunks_to_vector_store(chunks, doc_id, filename)

# 4. 语义检索
search_vector_store(query, top_k=3)
```

**向量检索流程**：
```
用户问题 → 生成问题向量 → ChromaDB相似度搜索 → 返回top_k相关块
```

---

### 4.3 hybrid_retriever.py - 混合检索模块

**原理**：结合向量检索 + BM25关键词检索，通过RRF(Reciprocal Rank Fusion)融合两个排名结果

```
用户问题
    │
    ├── 向量检索 (语义相似) ──→ top_k vectors
    │
    └── BM25检索 (关键词匹配) ──→ top_k keywords
    │
    └── RRF融合 ──→ 最终top_k
```

**RRF公式**：
```
score(d) = Σ 1/(k + rank(d))
```

| 检索方式 | 原理 | 适合场景 |
|---------|------|---------|
| **向量检索** | 语义理解，认识同义词 | 同义词、语义相似 |
| **BM25** | 关键词精确匹配，必须包含关键词 | 专业术语、缩写 |
| **混合检索** | 综合两者排名 | 兼顾精确和语义 |

**代码示例**：
```python
# 向量检索结果
vec_results = vector_store.search_vector_store(query, top_k=5)

# BM25检索结果
bm25_results = hr.search_bm25(query, top_k=5)

# RRF融合
fused = HybridRetriever.reciprocal_rank_fusion(
    [vec_results, bm25_results], k=60
)
```

---

### 4.4 query_rewriter.py - Query改写模块

**原理**：将用户问题改写成多个相关查询，提升检索召回率

```
原始: "这玩意儿咋学"
改写后:
  → 如何高效学习新技能
  → 学习编程的方法有哪些
  → 怎样快速掌握一门知识
  → 如何开始学习编程
```

**适用场景**：
- 口语化问题（"这玩意儿"、"啥"、"咋整"）
- 简短模糊问题
- 多义词问题

**代码示例**：
```python
rewriter = QueryRewriter()
queries = rewriter.rewrite_multi_query("Python能干啥", num_queries=4)

# 并行检索多个改写后的查询，合并去重
for q in queries:
    results = vector_store.search_vector_store(q, top_k=3)
    # 合并去重...
```

---

### 4.5 rag_chain.py - LangChain RAG模块

**功能**：基于LangChain框架的RAG链实现

```python
# LangChain RAG流程
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke(question)

# docs是LangChain Document对象列表
for doc in docs:
    print(doc.page_content)  # 文档内容
    print(doc.metadata)     # 元数据
```

**自定义嵌入函数**：
```python
# 复用vector_store的预加载模型，避免重复下载
class SentenceTransformerEmbeddings:
    def embed_query(self, text):
        return model.encode([text])[0].tolist()

    def embed_documents(self, texts):
        return model.encode(texts).tolist()
```

---

### 4.6 api.py - API服务模块

**核心接口**：

| 接口 | 方法 | 功能 |
|------|------|------|
| `/api/documents` | POST | 上传文档 |
| `/api/documents` | GET | 列出所有文档 |
| `/api/documents/{id}` | DELETE | 删除文档 |
| `/api/chat` | POST | 问答对话（基础版） |
| `/api/chat-v2` | POST | 增强版问答（支持多种检索模式） |
| `/api/chat/{session_id}` | GET | 获取对话历史 |

**chat-v2 请求格式**：
```json
{
  "session_id": "test",
  "message": "Python有什么特点",
  "use_rag": true,
  "use_hybrid": true,        // 混合检索
  "use_query_rewrite": true,  // Query改写
  "use_langchain": true       // LangChain RAG
}
```

**chat-v2 返回格式**：
```json
{
  "reply": "Python是一种...",
  "sources": [
    {"chunk": "文档内容...", "metadata": {"filename": "test.txt"}}
  ],
  "mode": "hybrid"  // vector | hybrid | query_rewrite | langchain
}
```

---

### 4.7 frontend.py - 前端界面模块

**会话状态管理**：
```python
st.session_state.session_id    # 唯一会话ID
st.session_state.messages      # 对话历史
st.session_state.file_uploaded  # 上传状态（防重复）
```

---

## 五、知识点详解

### 5.1 什么是RAG？

**RAG = Retrieval-Augmented Generation（检索增强生成）**

```
传统LLM：
用户问题 → LLM直接回答（可能胡编）

RAG：
用户问题 → 检索相关文档 → 构建上下文 → LLM生成回答
```

**为什么用RAG**：
1. 解决LLM知识过时问题
2. 减少"幻觉"（胡编乱造）
3. 可以使用自己的私有数据
4. 回答更准确、可溯源

### 5.2 向量检索 vs 关键词匹配

**传统关键词匹配**：
```
查询"Python特点" → 精确匹配"Python"和"特点"
缺点：不认识同义词（如"特性"就匹配不到）
```

**向量语义检索**：
```
查询"Python特点"
    ↓
模型理解语义："Python是一种编程语言，其特性是..."
    ↓
向量数据库找语义相似的内容
优点：认识同义词、理解语义
```

### 5.3 什么是RRF融合？

**Reciprocal Rank Fusion（倒数排名融合）**

当同一个文档在多个检索结果中出现时，综合计算其排名得分：

```
RRF公式: score(d) = Σ 1/(k + rank(d))

假设：
- 向量检索排名: doc_A排第1, doc_B排第2
- BM25检索排名: doc_B排第1, doc_A排第3

doc_A的RRF分数 = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
doc_B的RRF分数 = 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325

最终排名: doc_B > doc_A（两个检索都认可）
```

### 5.4 什么是Query改写？

**Multi-Query Expansion**：将一个查询改写成多个不同角度的查询

```
原始问题: "这玩意儿咋学"
    ↓
LLM改写（Few-shot prompt）:
  → 如何高效学习新技能
  → 学习编程的方法有哪些
  → 怎样快速掌握一门知识
  → 如何开始学习编程
    ↓
并行检索 → 合并去重 → 更全面的召回
```

### 5.5 嵌入模型如何工作？

```python
# 文本 → 数字向量
text = "Python是一种编程语言"
vector = model.encode([text])  # → [0.123, -0.456, 0.789, ...] (384维)

# 相似度计算：余弦相似度，越接近1越相似
```

### 5.6 为什么要分块？

```
假设文档10000字：
- 直接向量化：丢失太多细节
- 分块后：每块300字，可以精确定位相关段落
```

---

## 六、迭代过程

### 版本1.0 - 基础版本
- 简单文本存储
- jieba关键词匹配
- 基础API接口

### 版本2.0 - 向量检索版本
- 智能分块（按段落/句子）
- ChromaDB向量存储
- 语义检索
- 启动脚本优化

### 版本3.0 - 高级RAG版本（当前）
- LangChain RAG框架集成
- 混合检索（向量+BM25）
- Query改写/扩展
- RRF倒数排名融合

---

## 七、遇到的问题与解决

### 问题1：模型加载失败（SSL证书）

**错误信息**：
```
[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed
```

**原因**：直接访问HuggingFace下载模型，SSL验证失败

**解决**：配置清华镜像源
```python
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

---

### 问题2：向量检索一直返回空

**原因**：多进程状态不共享

```
start_server.py (主进程)
    ↓
vector_store.preload_model()  ← 模型加载到主进程
    ↓
uvicorn.run()  ← 启动worker进程
    ↓
Worker进程里 model = None  ← 进程不共享内存！
```

**解决**：修改 `start_server.py`，先加载模型再启动uvicorn

```python
vector_store.preload_model()  # 先加载
from api import app
uvicorn.run(app)  # 再启动，共享同一个进程
```

---

### 问题3：LangChain + FastAPI多进程问题

**现象**：LangChain模式返回500错误

**原因**：
```
@ app.on_event("startup") 在主进程执行
uvicorn worker是独立进程，不共享主进程的内存
worker进程里全局变量 model = None
```

**排查过程**：
```python
# 直接调用 - 成功
vector_store.preload_model()
rag_chain.chat_with_rag_chain('test')  # 成功

# API调用 - 失败
# worker进程里 model = None
```

**解决**：延迟初始化（Lazy Initialization）

```python
def get_embedding_function():
    """每次调用时确保模型已加载"""
    model = vector_store.get_model()
    if model is None:
        vector_store.preload_model()  # worker进程内加载
        model = vector_store.get_model()
    return SentenceTransformerEmbeddings(model)
```

---

### 问题4：前端重复上传文件

**原因**：Streamlit的`file_uploader`在每次`rerun`时都会重新渲染

**解决**：使用session_state标记上传状态
```python
if uploaded_file and not st.session_state.file_uploaded:
    st.session_state.file_uploaded = True  # 标记已上传
```

---

### 问题5：Docker部署时前端无法访问API

**原因**：前端容器里用`localhost:8000`访问后端，但后端在另一个容器

**解决**：使用Docker Compose的服务名
```yaml
environment:
  - API_BASE=http://api:8000  # api是服务名
```

---

## 八、调试方法

### 8.1 查看服务器日志

```bash
python start_server.py
# 观察输出：
# [1/4] 预加载嵌入模型...
# 模型预加载完成
# [2/4] 初始化数据库...
# 数据库就绪
# [3/4] 初始化向量数据库...
# [4/4] 启动API服务器...
```

### 8.2 API调试

```bash
# 查看文档列表
curl http://localhost:8000/api/documents

# 上传文档
curl -X POST -F "file=@test.txt" http://localhost:8000/api/documents

# 测试增强版问答（混合检索）
curl -X POST http://localhost:8000/api/chat-v2 \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test","message":"Python有什么特点","use_rag":true,"use_hybrid":true}'
```

### 8.3 测试各模式差异

```python
import requests

modes = [
    ('基础向量检索', {'use_rag': True}),
    ('混合检索', {'use_rag': True, 'use_hybrid': True}),
    ('Query改写', {'use_rag': True, 'use_query_rewrite': True}),
    ('LangChain', {'use_rag': True, 'use_langchain': True}),
]

for name, payload in modes:
    payload['session_id'] = 'test'
    payload['message'] = 'Python有什么特点'
    response = requests.post('http://localhost:8000/api/chat-v2', json=payload)
    result = response.json()
    print(f"模式: {name}, 返回: {result.get('mode')}")
```

### 8.4 测试Query改写效果

```python
import query_rewriter

queries = query_rewriter.rewrite_query('这玩意儿咋学')
print(f'原始: 这玩意儿咋学')
print('改写后:')
for q in queries:
    print(f'  → {q}')
```

---

## 九、运行指南

### 9.1 启动后端
```bash
cd d:/maybe_code/rag_project
python start_server.py
```

### 9.2 启动前端（新开终端）
```bash
cd d:/maybe_code/rag_project
streamlit run frontend.py --server.port 8501
```

### 9.3 Docker部署
```bash
cd d:/maybe_code/rag_project
docker-compose up --build
```

---

## 十、关键代码位置

| 功能 | 文件 | 说明 |
|------|------|------|
| 智能分块 | `api.py` | `split_chunks`函数 |
| 向量存储 | `vector_store.py` | ChromaDB操作 |
| 混合检索 | `hybrid_retriever.py` | BM25+向量+RRF |
| Query改写 | `query_rewriter.py` | LLM改写 |
| LangChain RAG | `rag_chain.py` | LangChain封装 |
| API接口 | `api.py` | chat-v2端点 |
| 前端界面 | `frontend.py` | Streamlit |
| 数据库表 | `database.py` | SQLite |
| 启动脚本 | `start_server.py` | 模型预加载 |

---

## 十一、面试加分点

### 1. LangChain集成
- 使用LangChain框架实现RAG
- 自定义嵌入函数，复用已有模型
- 了解LangChain的Retriever接口

### 2. 混合检索
- 向量检索 + BM25关键词检索
- RRF倒数排名融合算法
- 兼顾语义理解和精确匹配

### 3. Query改写
- Multi-Query Expansion策略
- 用LLM进行查询扩展
- 提升模糊问题的召回率

### 4. 多进程状态管理
- FastAPI + uvicorn多worker模型
- 全局变量不共享问题
- 延迟初始化解决方案

### 5. 工程问题解决
- 清华镜像解决HuggingFace下载问题
- session_state解决Streamlit重复上传
- 环境变量解决Docker网络问题

---

如果有任何知识点不清楚的地方，可以问我！
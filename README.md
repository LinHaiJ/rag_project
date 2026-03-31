# rag_project
个人对rag的学习项目QAQ
基于 RAG（检索增强生成）的知识库问答系统，支持多种检索策略和对话管理。

## 功能特性

- 📄 **文档上传管理** - 上传 TXT 文档，自动分块存储
- 🔍 **多种检索模式**
  - 向量检索（Sentence Transformers 嵌入）
  - 混合检索（向量 + BM25）
  - LangChain RAG
  - Query 改写（多查询扩展）
- 💬 **对话管理** - 支持多会话、对话历史持久化
- 🌐 **RESTful API** - 基于 FastAPI，提供完整的 API 接口
- 🔧 **可扩展** - 模块化设计，方便集成新功能

## 技术栈

| 组件 | 技术 |
|------|------|
| 后端框架 | FastAPI |
| 向量数据库 | ChromaDB |
| 嵌入模型 | paraphrase-multilingual-MiniLM-L12-v2 |
| 数据库 | SQLite |
| RAG 框架 | LangChain |
| 混合检索 | BM25 + 向量检索 |

## 快速开始

### 1. 安装依赖

```bash
pip install fastapi uvicorn aiosqlite chromadb sentence-transformers langchain langchain-community langchain-core langchain-text-splitters requests
```

### 2. 配置环境变量

创建 `.env` 文件：

```env
API_KEY=你的API密钥
```

### 3. 启动服务

```bash
python start_server.py
```

服务启动后访问 `http://localhost:8000`

### 4. API 文档

启动后打开 `http://localhost:8000/docs` 查看 Swagger API 文档。

## API 接口

### 文档管理

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/documents` | 上传文档 |
| GET | `/api/documents` | 获取文档列表 |
| DELETE | `/api/documents/{doc_id}` | 删除文档 |

### 问答

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/chat` | 基础问答（RAG 模式） |
| POST | `/api/chat-v2` | 增强问答（支持混合检索/LangChain/Query改写） |
| GET | `/api/chat/{session_id}` | 获取对话历史 |

### 请求示例

```bash
# 上传文档
curl -X POST "http://localhost:8000/api/documents" -F "file=@文档.txt"

# 问答
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "message": "你的问题", "use_rag": true}'
```

## 项目结构

```
rag_project/
├── api.py              # FastAPI 主应用
├── start_server.py     # 服务启动脚本
├── database.py         # SQLite 数据库管理
├── vector_store.py     # 向量存储管理
├── rag_chain.py        # LangChain RAG 链
├── hybrid_retriever.py # 混合检索器
├── query_rewriter.py   # Query 改写模块
├── rag_app.py          # RAG 主应用（命令行版）
├── rag_vector.py       # 向量检索演示
├── frontend.py         # 前端界面
└── uploaded_docs/      # 上传文档存储目录
```

## 检索模式说明

### 基础向量检索
传统的向量相似度匹配，返回最相关的文档片段。

### 混合检索
结合向量检索和 BM25 关键词检索，提升召回效果。

### LangChain RAG
使用 LangChain 框架的 RAG 链，支持更复杂的检索和生成流程。

### Query 改写
将用户问题改写为多个相关查询，提升检索召回率。

## 环境要求

- Python 3.8+
- 支持 Linux/macOS/Windows

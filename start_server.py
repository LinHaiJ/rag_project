#!/usr/bin/env python
"""
RAG知识库系统启动脚本

使用方法:
    python start_server.py

启动顺序:
1. 预加载嵌入模型（避免请求时延迟）
2. 初始化数据库
3. 初始化向量数据库
4. 启动API服务器
"""
import vector_store
import database
import asyncio


def main():
    print("=" * 50)
    print("RAG知识库系统启动")
    print("=" * 50)

    # 1. 预加载模型（必须在导入api模块之前完成）
    print("\n[1/4] 预加载嵌入模型...")
    vector_store.preload_model()
    print("模型就绪")

    # 2. 初始化数据库
    print("\n[2/4] 初始化数据库...")
    asyncio.run(database.init_db())
    print("数据库就绪")

    # 3. 初始化向量数据库
    print("\n[3/4] 初始化向量数据库...")
    vector_store.init_vector_store()
    print("向量数据库就绪")

    # 4. 启动API服务器（必须在导入api之前确保模型已加载）
    print("\n[4/4] 启动API服务器...")
    print("=" * 50)
    print("所有组件就绪!")
    print("=" * 50)

    import uvicorn
    from api import app

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()
import aiosqlite
import os

DATABASE_PATH = "rag_system.db"#RAG 问答系统用来存数据的本地数据库文件

async def init_db():#异步函数，非阻塞
    """初始化数据库"""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        # 文档表，sql命令，告诉数据库建表（id、名字、路径、内容、切块、时间）
        await db.execute("""                    
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                content TEXT,
                chunk_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 文档块表
        await db.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents(id)
            )
        """)

        # 聊天记录表，sql命令，await：等待数据库完成再操作
        await db.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await db.commit()   #保存到磁盘
    print("数据库初始化完成！")

if __name__ == "__main__":
    import asyncio
    asyncio.run(init_db())

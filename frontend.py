import streamlit as st
import requests
import uuid
import time

# ============ 配置 ============
import os
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="RAG知识库", page_icon="📚")
st.title("📚 RAG知识库问答系统")

# ============ 初始化会话状态 ============
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

# ============ 侧边栏：文档管理 ============
with st.sidebar:
    st.header("📁 文档管理")
    
    uploaded_file = st.file_uploader("上传文档", type=["txt"])

    # 只有在文件已选择 且 未上传过 时才上传
    if uploaded_file and not st.session_state.file_uploaded:
        with st.spinner("上传中..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/plain")}
            response = requests.post(f"{API_BASE}/api/documents", files=files)

            if response.status_code == 200:
                result = response.json()
                st.success(f"上传成功！文档ID: {result['document_id']}")
                # 标记已上传，防止重复
                st.session_state.file_uploaded = True
                time.sleep(1)
                st.rerun()
            else:
                st.error("上传失败")
    
    st.divider()
    st.subheader("已上传文档")

    # 清空所有文档按钮
    if st.button("🗑️ 清空所有文档", use_container_width=True):
        try:
            response = requests.get(f"{API_BASE}/api/documents")
            if response.status_code == 200:
                docs = response.json().get("documents", [])
                for doc in docs:
                    requests.delete(f"{API_BASE}/api/documents/{doc['id']}")
                st.session_state.file_uploaded = False  # 重置上传状态
                st.rerun()
        except:
            st.error("清空失败")

    # 获取文档列表
    try:
        response = requests.get(f"{API_BASE}/api/documents")
        if response.status_code == 200:
            docs = response.json().get("documents", [])
            if docs:
                for doc in docs:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(doc["filename"])
                    with col2:
                        if st.button("🗑️", key=f"del_{doc['id']}"):
                            requests.delete(f"{API_BASE}/api/documents/{doc['id']}")
                            st.session_state.file_uploaded = False  # 重置上传状态
                            st.rerun()
                    st.caption(f"分块数: {doc['chunk_count']}")
            else:
                st.info("暂无文档")
    except:
        st.error("无法连接API服务")

# ============ 主界面：聊天 ============
use_rag = st.checkbox("🔍 开启RAG检索", value=True)

# 显示聊天历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
if prompt := st.chat_input("输入问题..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 调用API
    with st.spinner("思考中..."):
        try:
            response = requests.post(
                f"{API_BASE}/api/chat",
                json={
                    "session_id": st.session_state.session_id,
                    "message": prompt,
                    "use_rag": use_rag
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_reply = result["reply"]
                
                st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                with st.chat_message("assistant"):
                    st.markdown(ai_reply)
            else:
                st.error(f"请求失败: {response.status_code}")
        except Exception as e:
            st.error(f"连接失败: {e}")

import os
import sys
import time
import logging
import streamlit as st
from dotenv import load_dotenv

# --- 1. 基础配置与初始化 ---
st.set_page_config(
    page_title="尝尝咸淡 AI",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

if not os.getenv("DASHSCOPE_API_KEY"):
    st.error("🚨 错误: 未检测到 DASHSCOPE_API_KEY，请检查 .env 文件。")
    st.stop()

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from rag_modules.data_preparation import DataPreparationModule
    from rag_modules.index_construction import IndexConstructionModule
    from rag_modules.retrieval_optimization import RetrievalOptimizationModule
    from rag_modules.generation_integration import GenerationIntegrationModule
    from config import DEFAULT_CONFIG
except ImportError as e:
    st.error(f"❌ 核心模块导入失败: {e}")
    st.stop()

logging.getLogger("langchain_core").setLevel(logging.WARNING)

# --- 2. 核心系统加载 (带缓存) ---
@st.cache_resource(show_spinner=False)
def load_rag_system():
    """初始化并缓存RAG系统"""
    try:
        data_module = DataPreparationModule(data_path=DEFAULT_CONFIG.data_path)
        data_module.load_and_process_documents()
        
        index_module = IndexConstructionModule(
            model_name=DEFAULT_CONFIG.embedding_model,
            index_save_path=DEFAULT_CONFIG.index_save_path
        )
        vectorstore = index_module.load_or_build_index(data_module.chunks)
        
        retrieval_module = RetrievalOptimizationModule(vectorstore, data_module.chunks)
        
        generation_module = GenerationIntegrationModule(
            model_name=DEFAULT_CONFIG.llm_model,
            temperature=DEFAULT_CONFIG.temperature,
            max_tokens=DEFAULT_CONFIG.max_tokens
        )
        
        return {
            "data": data_module,
            "retrieval": retrieval_module,
            "generation": generation_module,
            "status": "ready"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def format_references(docs):
    """辅助函数：将文档对象转换为易于展示和存储的字典列表"""
    if not docs:
        return []
    refs = []
    for doc in docs:
        refs.append({
            "dish": doc.metadata.get("dish_name", "未知菜品"),
            "category": doc.metadata.get("category", "其他"),
            "difficulty": doc.metadata.get("difficulty", "未知"),
            "source": os.path.basename(doc.metadata.get("source", ""))
        })
    return refs

# --- 3. 侧边栏 ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/cooking-pot.png", width=80)
    st.title("尝尝咸淡 AI")
    st.caption("您的私人智能膳食顾问")
    st.divider()
    
    if "rag_system" not in st.session_state:
        with st.status("🚀 系统正在启动...", expanded=True) as status:
            st.write("正在加载菜谱数据...")
            rag = load_rag_system()
            if rag["status"] == "ready":
                st.session_state.rag_system = rag
                st.write("索引构建完成...")
                status.update(label="✅ 系统就绪", state="complete", expanded=False)
            else:
                st.error(f"系统初始化失败: {rag['message']}")
                st.stop()
    
    rag = st.session_state.rag_system
    st.metric(label="已收录菜谱", value=f"{len(rag['data'].documents)} 道")
    st.divider()
    
    if st.button("🗑️ 清空对话历史", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- 4. 主聊天界面 ---

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是小当家。今天想吃点什么？"}
    ]

# 渲染历史消息 (关键修改：增加引用渲染)
for msg in st.session_state.messages:
    avatar = "🍳" if msg["role"] == "assistant" else "🧑‍💻"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        
        # 如果该消息包含引用信息，则渲染折叠框
        if "references" in msg and msg["references"]:
            with st.expander("📚 参考食谱 / 来源"):
                for i, ref in enumerate(msg["references"]):
                    st.markdown(f"**{i+1}. {ref['dish']}**")
                    st.caption(f"分类: {ref['category']} | 难度: {ref['difficulty']} | 文件: `{ref['source']}`")

# 处理用户输入
if prompt := st.chat_input("输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🍳"):
        message_placeholder = st.empty()
        full_response = ""
        relevant_docs = [] # 初始化
        
        with st.status("🍳 正在思考中...", expanded=False) as status:
            try:
                rag = st.session_state.rag_system
                
                st.write("🤔 分析用户意图...")
                route_type = rag["generation"].query_router(prompt)
                
                st.write("✍️ 优化查询关键词...")
                rewritten_query = rag["generation"].query_rewrite(prompt)
                
                st.write("🔍 分析筛选条件...")
                filters = rag["generation"].extract_filters(prompt, rag["data"])
                
                st.write("📚 检索知识库...")
                relevant_chunks = []
                if filters:
                    relevant_chunks = rag["retrieval"].metadata_filtered_search(
                        rewritten_query, filters, top_k=5
                    )
                    if not relevant_chunks:
                        st.write("⚠️ 过滤检索无结果，降级为混合检索...")
                        relevant_chunks = rag["retrieval"].hybrid_search(rewritten_query, top_k=5)
                else:
                    relevant_chunks = rag["retrieval"].hybrid_search(rewritten_query, top_k=5)
                
                relevant_docs = rag["data"].get_parent_documents(relevant_chunks)
                status.update(label="✨ 思考完成", state="complete", expanded=False)
                
            except Exception as e:
                status.update(label="❌ 发生错误", state="error")
                st.error(f"处理流程异常: {e}")
                st.stop()

        try:
            if not relevant_docs:
                full_response = "抱歉，我的菜谱库里暂时没有找到相关内容。"
                message_placeholder.markdown(full_response)
                # 即使没有找到，也保存一条空引用的消息
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "references": []
                })
            else:
                response_generator = rag["generation"].generate_answer(
                    prompt, relevant_docs, route_type
                )
                
                for chunk in response_generator:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # --- 核心修改：处理并展示引用 ---
                # 1. 格式化引用数据
                refs_data = format_references(relevant_docs)
                
                # 2. 在当前回答下方立即展示
                if refs_data:
                    with st.expander("📚 参考食谱 / 来源"):
                        for i, ref in enumerate(refs_data):
                            st.markdown(f"**{i+1}. {ref['dish']}**")
                            st.caption(f"分类: {ref['category']} | 难度: {ref['difficulty']} | 文件: `{ref['source']}`")

                # 3. 保存到历史记录 (包含引用数据)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "references": refs_data
                })
        
        except Exception as e:
            st.error(f"生成回答时出错: {e}")
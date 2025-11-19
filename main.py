# main.py

import os
import logging
from dotenv import load_dotenv

from config import DEFAULT_CONFIG, RAGConfig
from rag_modules.data_preparation import DataPreparationModule
from rag_modules.index_construction import IndexConstructionModule
from rag_modules.retrieval_optimization import RetrievalOptimizationModule
from rag_modules.generation_integration import GenerationIntegrationModule

# --- 初始化 ---
# 加载.env文件中的环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# 将 'langchain_core' 的日志级别设置为 WARNING，屏蔽掉底层繁琐的日志
logging.getLogger("langchain_core").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class RecipeRAGSystem:
    """食谱RAG系统主类，负责协调所有模块。"""

    def __init__(self, config: RAGConfig = None):
        self.config = config or DEFAULT_CONFIG
        
        # 检查API密钥 (通义千问)
        if not os.getenv("DASHSCOPE_API_KEY"):
            raise ValueError("请在.env文件中设置 DASHSCOPE_API_KEY 环境变量")
        
        # 初始化所有模块占位符
        self.data_module: DataPreparationModule = None
        self.index_module: IndexConstructionModule = None
        self.retrieval_module: RetrievalOptimizationModule = None
        self.generation_module: GenerationIntegrationModule = None

    def initialize_system(self):
        """初始化所有RAG模块。"""
        logger.info("🚀 正在初始化RAG系统...")
        
        # 1. 数据准备模块
        self.data_module = DataPreparationModule(data_path=self.config.data_path)
        
        # 2. 索引构建模块
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path
        )
        
        # 3. 生成集成模块
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        logger.info("✅ 系统模块初始化完成！")

    def build_knowledge_base(self):
        """构建或加载知识库。"""
        logger.info("\n构建知识库...")
        
        # 加载并切分数据
        self.data_module.load_and_process_documents()
        chunks = self.data_module.chunks
        
        # 加载或构建向量索引
        vectorstore = self.index_module.load_or_build_index(chunks)
        
        # 初始化检索优化模块 (传入向量库和文档块)
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)
        logger.info("✅ 知识库准备就绪！")

    def ask_question(self, question: str):
        """
        核心的问答流程：意图识别 -> 过滤提取 -> 智能检索 -> 生成回答
        """
        logger.info(f"\n❓ 开始处理新问题: {question}")

        # --- 步骤 1: 意图识别与查询优化 ---
        route_type = self.generation_module.query_router(question)
        

        rewritten_query = self.generation_module.query_rewrite(question)

        # --- 步骤 2: 提取过滤器 (元数据分析) ---
        # 传入 data_module 以获取动态分类列表
        filters = self.generation_module.extract_filters(question, self.data_module)
        if filters:
            logger.info(f"提取到的过滤器: {filters}")

        # --- 步骤 3: 智能检索 (分支逻辑) ---
        if filters:
            # A. 如果有过滤条件，使用前过滤检索 (Pre-filtering)
            # 注意：这里调用的是 metadata_filtered_search
            relevant_chunks = self.retrieval_module.metadata_filtered_search(
                rewritten_query,
                filters=filters,
                top_k=self.config.top_k
            )

            # 新增：降级重试逻辑 
            if not relevant_chunks:
                logger.warning(f"过滤器 {filters} 导致零结果。正在降级为无过滤混合检索...")
                relevant_chunks = self.retrieval_module.hybrid_search(
                    rewritten_query, 
                    top_k=self.config.top_k
                )
        else:
            # B. 如果没有过滤条件，使用混合检索 (Hybrid Search)
            relevant_chunks = self.retrieval_module.hybrid_search(
                rewritten_query, 
                top_k=self.config.top_k
            )
        
        # --- 步骤 4: 上下文处理 (父子文档去重) ---
        relevant_docs = self.data_module.get_parent_documents(relevant_chunks)
        
        # 边界情况处理：如果没有检索到任何文档
        if not relevant_docs:
            msg = "抱歉，没有找到相关的食谱信息来回答您的问题。"
            if filters:
                msg = f"抱歉，在满足条件 {filters} 的情况下，没有找到相关菜谱。建议您放宽筛选条件试试。"
            # 返回一个迭代器，保证 run_interactive 中的循环不报错
            return iter([msg])

        # --- 步骤 5: 生成回答 (流式) ---
        # 将处理好的上下文交给生成模块
        return self.generation_module.generate_answer(
            question, 
            relevant_docs, 
            route_type
        )

    def run_interactive(self):
        """运行交互式命令行界面。"""
        print("\n" + "="*60)
        print("🍽️  欢迎使用'尝尝咸淡'智能菜谱问答系统  🍽️")
        print("="*60)
        print("💡 您可以问我任何关于烹饪的问题，例如：")
        print("   - '推荐几道简单的素菜'")
        print("   - '宫保鸡丁怎么做？'")
        print("   - '红烧肉需要什么食材？'")
        print("   - (输入 'quit' 或 'exit' 退出)")
        
        # 启动初始化流程
        self.initialize_system()
        self.build_knowledge_base()
        
        while True:
            try:
                user_input = input("\n🤔 请输入您的问题: ").strip()
                if user_input.lower() in ['quit', 'exit']:
                    print("👋 感谢使用，下次再见！")
                    break
                if not user_input:
                    continue

                print("\n🍳 小当家正在思考中...")
                
                # 获取生成器
                response_generator = self.ask_question(user_input)
                
                # --- 流式输出打印 ---
                full_response = ""
                for chunk in response_generator:
                    print(chunk, end="", flush=True)
                    full_response += chunk
                
                print("\n") # 回答结束后换行

            except KeyboardInterrupt:
                print("\n👋 感谢使用，下次再见！")
                break
            except Exception as e:
                logger.error(f"处理问题时发生错误: {e}", exc_info=True)
                print(f"\n😥 抱歉，处理您的问题时遇到了一个错误。请稍后再试。")

def main():
    """主函数入口。"""
    try:
        rag_system = RecipeRAGSystem()
        rag_system.run_interactive()
    except Exception as e:
        logger.error(f"系统启动失败: {e}", exc_info=True)
        print(f"❌ 系统启动失败: {e}")

if __name__ == "__main__":
    main()
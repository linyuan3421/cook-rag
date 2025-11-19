# config.py

from dataclasses import dataclass

@dataclass
class RAGConfig:
    """RAG系统配置类"""
    # 数据路径
    data_path: str = "./data"
    
    # 索引配置
    index_save_path: str = "./vector_index"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    
    # LLM配置
    llm_model: str = "qwen-plus-latest"  
    temperature: float = 0.1
    max_tokens: int = 4096
    
    # 检索配置
    top_k: int = 8 # 混合检索时，每路召回的数量

# 创建一个默认配置实例，方便直接导入使用
DEFAULT_CONFIG = RAGConfig()


# rag_modules/index_construction.py

import logging
from typing import List
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 使用与data_preparation模块相同的日志记录器，保持一致性
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IndexConstructionModule:
    """
    索引构建模块，核心职责包括：
    1.  初始化Embedding模型。
    2.  根据文档块（chunks）构建FAISS向量索引。
    3.  提供索引的保存和加载（缓存）功能，以提升效率。
    4.  支持向现有索引增量添加文档。
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", index_save_path: str = "./vector_index"):
        """
        初始化索引构建模块。

        Args:
            model_name (str): Hugging Face上的嵌入模型名称。
            index_save_path (str): 本地FAISS索引的保存路径。
        """
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.embeddings: HuggingFaceEmbeddings = None # 明确类型注解
        self.vectorstore: FAISS = None # 明确类型注解
        
        # 在初始化时就设置好模型，后续方法可以直接使用
        self._setup_embeddings()
    
    def _setup_embeddings(self):
        """私有方法，用于初始化嵌入模型。"""
        if self.embeddings is None:
            logger.info(f"正在初始化嵌入模型: {self.model_name}")
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={'device': 'cpu'}, # 明确在CPU上运行
                    encode_kwargs={'normalize_embeddings': True} # 归一化嵌入，对于IP（内积）/Cosine相似度很重要
                )
                logger.info("嵌入模型初始化完成。")
            except Exception as e:
                logger.error(f"初始化嵌入模型失败: {e}")
                raise

    def build_vector_index(self, chunks: List[Document]) -> FAISS:
        """
        从头构建一个全新的FAISS向量索引。

        Args:
            chunks (List[Document]): 从数据准备模块获得的文档块列表。

        Returns:
            FAISS: 构建完成的向量存储对象。
        """
        logger.info(f"开始从 {len(chunks)} 个文档块构建新的FAISS向量索引...")
        
        if not chunks:
            raise ValueError("文档块列表不能为空，无法构建索引。")
        
        try:
            self.vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
            logger.info(f"向量索引构建完成，包含了 {self.vectorstore.index.ntotal} 个向量。")
            return self.vectorstore
        except Exception as e:
            logger.error(f"构建FAISS索引时发生错误: {e}")
            raise

    def add_documents(self, new_chunks: List[Document]):
        """
        向现有的FAISS索引中增量添加新的文档块。

        Args:
            new_chunks (List[Document]): 新的文档块列表。
        """
        if not self.vectorstore:
            raise ValueError("向量存储未初始化，无法添加文档。请先构建或加载索引。")
        if not new_chunks:
            logger.warning("没有新的文档块需要添加。")
            return
            
        logger.info(f"正在向现有索引中添加 {len(new_chunks)} 个新文档块...")
        # FAISS的add_documents方法是在现有索引上进行增量添加
        self.vectorstore.add_documents(new_chunks)
        logger.info(f"新文档添加完成。索引现在总共有 {self.vectorstore.index.ntotal} 个向量。")

    def save_index(self):
        """将当前的向量索引持久化保存到本地文件。"""
        if not self.vectorstore:
            raise ValueError("没有可保存的向量索引。")

        save_path = Path(self.index_save_path)
        save_path.mkdir(parents=True, exist_ok=True) # 确保目录存在

        logger.info(f"正在将向量索引保存到: {save_path.resolve()}...")
        self.vectorstore.save_local(str(save_path))
        logger.info("向量索引已成功保存。")
    
    def load_or_build_index(self, chunks: List[Document]):
        """
        核心流程方法：尝试加载索引，如果失败或不存在，则从头构建。
        这是推荐在主流程中调用的方法。
        
        Args:
            chunks (List[Document]): 如果需要新建索引时使用的文档块。
        """
        index_path = Path(self.index_save_path)
        if index_path.exists() and any(index_path.iterdir()):
            logger.info(f"发现已存在的索引，正在从 {index_path.resolve()} 加载...")
            try:
                self.vectorstore = FAISS.load_local(
                    str(index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"向量索引加载成功，包含 {self.vectorstore.index.ntotal} 个向量。")
            except Exception as e:
                logger.warning(f"加载索引失败: {e}。将重新构建新索引。")
                self.build_vector_index(chunks)
                self.save_index() # 构建后别忘了保存
        else:
            logger.info("未发现本地索引，开始构建新索引...")
            self.build_vector_index(chunks)
            self.save_index() # 构建后别忘了保存

        return self.vectorstore
# rag_modules/data_preparation.py

import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

# 配置一个基础的日志记录器，方便我们观察模块的运行状态
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreparationModule:
    """
    数据准备模块，负责以下核心任务：
    1.  从指定路径加载Markdown文档。
    2.  为每个文档增强元数据（分类、名称、难度）。
    3.  将文档按Markdown标题结构切分为子块（Chunks）。
    4.  建立父文档与子块之间的关联。
    5.  提供根据子块检索父文档（含去重和排序）的功能。
    """
    
    def __init__(self, data_path: str):
        """
        初始化数据准备模块。

        Args:
            data_path (str): 存放菜谱Markdown文件的数据文件夹路径。
        """
        self.data_path = data_path
        self.documents: List[Document] = []  # 存储所有父文档（完整菜谱）
        self.chunks: List[Document] = []     # 存储所有子块（按标题分割的小块）
        
        # 优化点：创建一个从 parent_id 到 parent_doc 的快速查找字典，避免在 get_parent_documents 中重复循环
        self._parent_doc_map: Dict[str, Document] = {}
        
        # --- 将分类映射公开为类属性，供外部调用 ---
        self.available_categories: Dict[str, str] = {
            'meat_dish': '荤菜', 
            'vegetable_dish': '素菜', 
            'soup': '汤品',
            'aquatic': '水产', 
            'dessert': '甜品', 
            'breakfast': '早餐', 
            'staple': '主食',
            'condiment': '调料', 
            'drink': '饮品',
            'semi-finished': '半成品'
        }

    def load_and_process_documents(self):
        """
        执行完整的数据加载和处理流程：加载 -> 增强 -> 切分。
        这是一个便捷方法，按顺序调用了模块的核心功能。
        """
        self.load_documents()
        self.chunk_documents()
        logger.info("数据加载和处理流程完成。")
        logger.info(f"总共加载了 {len(self.documents)} 个父文档。")
        logger.info(f"总共切分出 {len(self.chunks)} 个子文档块。")

    def load_documents(self):
        """
        从指定路径递归加载所有Markdown文件作为父文档。
        为每个文档分配一个确定性的唯一ID，并进行元数据增强。
        """
        logger.info(f"开始从 '{self.data_path}' 加载文档...")
        documents = []
        data_path_obj = Path(self.data_path)

        for md_file in data_path_obj.rglob("*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 使用文件相对路径的MD5哈希作为确定性的ID
                # 这确保了每次运行时，同一个文件的ID都是相同的，对于缓存和复现至关重要。
                relative_path = md_file.relative_to(data_path_obj).as_posix()
                parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()

                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(md_file),
                        "parent_id": parent_id,
                        "doc_type": "parent"
                    }
                )
                documents.append(doc)
            except Exception as e:
                logger.warning(f"读取文件 {md_file} 失败: {e}")
        
        # 对加载的所有文档进行元数据增强
        for doc in documents:
            self._enhance_metadata(doc)
        
        self.documents = documents
        
        # 关键优化：构建 parent_id 到 Document 对象的映射，用于快速查找
        self._parent_doc_map = {doc.metadata["parent_id"]: doc for doc in self.documents}
        
        logger.info(f"成功加载并预处理了 {len(self.documents)} 个文档。")

    def _enhance_metadata(self, doc: Document):
        """
        为单个文档提取并添加额外的元数据。
        从文件路径提取'category'和'dish_name'，从内容中提取'difficulty'。
        """
        file_path = Path(doc.metadata.get('source', ''))
        
        # 1. 提取菜品分类
        # 这里的映射关系基于HowToCook项目的目录结构
        category_mapping = {
            'meat_dish': '荤菜', 'vegetable_dish': '素菜', 'soup': '汤品',
            'dessert': '甜品', 'breakfast': '早餐', 'staple': '主食',
            'aquatic': '水产', 'condiment': '调料', 'drink': '饮品'
        }
        doc.metadata['category'] = '其他'
        for key, value in category_mapping.items():
            if key in file_path.parts:
                doc.metadata['category'] = value
                break
        
        # 2. 提取菜品名称 (从文件名，不含扩展名)
        doc.metadata['dish_name'] = file_path.stem

        # 3. 分析难度等级 (从文件内容)
        content = doc.page_content
        if '★★★★★' in content: doc.metadata['difficulty'] = '非常困难'
        elif '★★★★' in content: doc.metadata['difficulty'] = '困难'
        elif '★★★' in content: doc.metadata['difficulty'] = '中等'
        elif '★★' in content: doc.metadata['difficulty'] = '简单'
        elif '★' in content: doc.metadata['difficulty'] = '非常简单'
        else: doc.metadata['difficulty'] = '未知'

    def chunk_documents(self):
        """
        使用Markdown标题分割器，将已加载的父文档切分为子块。
        同时，为每个子块继承父文档的元数据并建立父子关联。
        """
        if not self.documents:
            raise ValueError("没有已加载的文档可供切分，请先调用 load_documents()")

        # 定义按哪几级标题进行分割
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False  # 保留标题文本在内容中，有助于理解上下文
        )

        all_chunks = []
        for doc in self.documents:
            # 对每个父文档进行切分
            md_chunks = markdown_splitter.split_text(doc.page_content)
            
            # 遍历切分出的所有子块
            for i, chunk in enumerate(md_chunks):
                # 关键步骤：为子块构建丰富的元数据
                # 1. 复制父文档的所有元数据
                chunk.metadata.update(doc.metadata)
                
                # 2. 添加子块特有的元数据
                chunk.metadata.update({
                    "chunk_id": str(uuid.uuid4()), # 为每个子块生成一个唯一的ID
                    "doc_type": "child",            # 标记这是一个子文档
                    "chunk_index": i                # 标记这是父文档的第几个子块
                })
            all_chunks.extend(md_chunks)

        self.chunks = all_chunks
    
    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        """
        根据检索到的子块列表，获取它们对应的、去重且按相关性排序的父文档列表。

        Args:
            child_chunks: 检索模块返回的子块文档列表。

        Returns:
            一个去重且按相关性（即被命中子块的数量）降序排列的父文档列表。
        """
        parent_relevance = {} # 用于存储每个父文档被命中的次数

        # 遍历所有命中的子块
        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if parent_id:
                # 累加计数
                parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1

        # 根据命中次数对parent_id进行降序排序
        sorted_parent_ids = sorted(parent_relevance.keys(), key=lambda pid: parent_relevance[pid], reverse=True)

        # 使用预先构建的映射快速获取对应的父文档对象
        sorted_parent_docs = [self._parent_doc_map[pid] for pid in sorted_parent_ids]

        logger.info(f"从 {len(child_chunks)} 个子块中，智能去重并排序后得到 {len(sorted_parent_docs)} 个父文档:")
        # 打印前几个结果用于调试
        for i, doc in enumerate(sorted_parent_docs[:5]):
            logger.info(f"  {i+1}. {doc.metadata.get('dish_name')} ({parent_relevance[doc.metadata['parent_id']]}次命中)")
        
        return sorted_parent_docs
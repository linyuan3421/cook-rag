# rag_modules/retrieval_optimization.py

import logging
import hashlib
from typing import List, Dict, Any
import torch

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
# 引入重排序模型
from sentence_transformers import CrossEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RetrievalOptimizationModule:
    """
    检索优化模块 V2.0 (引入 Rerank 精排)
    
    核心升级：
    1. 扩大初排召回量 (Recall): 先捞回更多数据 (Top 30-50)，避免漏掉正确答案。
    2. 引入重排序 (Rerank): 使用 Cross-Encoder 模型对召回结果进行精准打分。
    3. 阈值过滤 (Threshold): 只有得分超过阈值的文档才会被送给 LLM，彻底解决“不相关”问题。
    """

    def __init__(self, vectorstore: FAISS, chunks: List[Document], rerank_model_name: str = "BAAI/bge-reranker-base"):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.vector_retriever = None
        self.bm25_retriever = None
        
        # 初始化检索器
        self._setup_retrievers()
        
        # 初始化重排序模型 (自动检测GPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"正在加载重排序模型: {rerank_model_name} (Device: {self.device})...")
        try:
            self.reranker = CrossEncoder(rerank_model_name, device=self.device, automodel_args={"torch_dtype": torch.float16} if self.device=="cuda" else {})
            logger.info("重排序模型加载完成。")
        except Exception as e:
            logger.error(f"重排序模型加载失败: {e}")
            self.reranker = None

    def _setup_retrievers(self):
        """初始化向量和BM25检索器，注意这里大幅扩大了初始召回数量 """
        # 向量检索器：扩大召回范围，防止漏网之鱼
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 15} 
        )

        # BM25检索器
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=15
        )
        logger.info("基础检索器初始化完成 (Initial k=15)。")

    def hybrid_search(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Document]:
        """
        执行两阶段检索：混合召回 -> Rerank精排
        """
        logger.info(f"开始检索: '{query}'")
        
        # --- 第一阶段：广撒网 (Recall) ---
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        
        # 使用RRF进行初步融合去重
        candidate_docs = self._rrf_merge(vector_docs, bm25_docs, top_n=20)
        logger.info(f"粗排召回 {len(candidate_docs)} 个候选文档。")

        if not candidate_docs:
            return []

        # --- 第二阶段：精挑选 (Precision) ---
        # 如果重排序模型未加载，降级使用RRF结果
        if not self.reranker:
            logger.warning("重排序模型不可用，仅返回粗排结果。")
            return candidate_docs[:top_k]

        # 构造 (Query, Doc) 对进行打分
        rerank_pairs = [[query, doc.page_content] for doc in candidate_docs]
        scores = self.reranker.predict(rerank_pairs, show_progress_bar=False)

        # 将分数绑定回文档，并排序
        scored_docs = []
        for doc, score in zip(candidate_docs, scores):
            doc.metadata['rerank_score'] = float(score) # 记录分数用于调试
            if score > score_threshold: # 核心：过滤掉低分文档
                scored_docs.append((doc, score))
        
        # 按分数降序排列
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # 截取最终的 Top-K
        final_results = [doc for doc, score in scored_docs[:top_k]]
        
        # 日志打印 Top 3 文档及其分数，方便观察效果
        for i, doc in enumerate(final_results[:3]):
            dish = doc.metadata.get('dish_name', '未知')
            score = doc.metadata.get('rerank_score')
            logger.info(f"精排 Top{i+1}: {dish} (Score: {score:.4f})")

        return final_results

    def _rrf_merge(self, list1: List[Document], list2: List[Document], top_n: int = 50) -> List[Document]:
        """RRF 融合算法 (仅用于合并去重，不作为最终排序依据)"""
        rrf_scores = {}
        doc_map = {}
        k = 60

        for rank, doc in enumerate(list1):
            # 使用 chunk_id 或 内容哈希 作为唯一键
            doc_id = doc.metadata.get("chunk_id") or hashlib.md5(doc.page_content.encode()).hexdigest()
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        for rank, doc in enumerate(list2):
            doc_id = doc.metadata.get("chunk_id") or hashlib.md5(doc.page_content.encode()).hexdigest()
            if doc_id not in doc_map: doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return [doc_map[uid] for uid in sorted_ids[:top_n]]
    
    def metadata_filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Document]:
        """带元数据过滤的检索 (同样引入Rerank)"""
        logger.info(f"执行过滤检索: {filters}")
        
        # 1. 前过滤 (Pre-filtering) + 向量召回
        filtered_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 30, "filter": filters} # 同样扩大召回
        )
        candidate_docs = filtered_retriever.invoke(query)
        
        if not candidate_docs or not self.reranker:
             return candidate_docs[:top_k]
        
        # 2. Rerank 精排
        rerank_pairs = [[query, doc.page_content] for doc in candidate_docs]
        scores = self.reranker.predict(rerank_pairs, show_progress_bar=False)
        
        scored_docs = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
        final_results = [doc for doc, _ in scored_docs[:top_k]]
        
        return final_results
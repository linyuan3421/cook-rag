# rag_modules/generation_integration.py

import os
import json
import logging
from typing import List, Dict, Any, Iterator

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
# 导入 DataPreparationModule 以进行类型注解
from .data_preparation import DataPreparationModule 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GenerationIntegrationModule:
    """
    生成集成模块，用于处理用户查询并生成答案。
    核心职责：
    1.  初始化大语言模型 (LLM)。
    2.  智能路由：根据用户问题分类意图。
    3.  查询重写：对模糊问题进行优化。
    4.  多模式生成：根据意图选择不同的Prompt和生成策略。
    5.  支持流式输出，提升用户体验。
    """

    def __init__(self, model_name: str = "qwen-plus-latest", temperature: float = 0.1, max_tokens: int = 4096):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm: BaseChatModel = None
        self._setup_llm()

    def _setup_llm(self):
        """私有方法，初始化大语言模型。"""
        logger.info(f"正在初始化LLM: {self.model_name}")
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("环境变量 DASHSCOPE_API_KEY 未设置，请在.env文件中配置。")
        self.llm = ChatTongyi(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            dashscope_api_key=api_key
        )
        logger.info("LLM初始化完成。")

    def query_router(self, query: str) -> str:
        """使用LLM对用户查询进行意图分类。"""
        prompt = ChatPromptTemplate.from_template("""根据用户的菜谱查询问题，将其分类为以下三种类型之一：
        1. 'list': 当用户想要获取一个菜品列表或推荐时。例如："推荐几个素菜"、"有什么简单的早餐"。
        2. 'detail': 当用户询问特定菜品的制作方法、食材或步骤时。例如："宫保鸡丁怎么做"、"番茄炒蛋需要什么原料"。
        3. 'general': 当问题不属于以上两类，可能是一般性知识、烹饪技巧或模糊不清时。例如："什么是川菜"、"如何让鸡肉更嫩"。

        请只返回'list'、'detail'或'general'这三个单词中的一个。

        用户问题: "{query}"
        分类结果:""")
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"query": query}).strip().lower()
        
        logger.info(f"查询 '{query}' 的路由类型判定为: {result}")
        return result if result in ['list', 'detail', 'general'] else 'general'

    def query_rewrite(self, query: str) -> str:
        """对模糊查询进行重写，使其更适合检索。"""
        prompt = ChatPromptTemplate.from_template(
        """你是一个查询优化助手。你的目标是将用户的【模糊意图】转化为数据库能听懂的【具体食材或菜名】。

        我们的数据库是一个【美食菜谱库】，包含具体的制作步骤。它**不包含**营养学标签（如"减肥"、"健身"）或场景标签（如"宴客"）。

        ### 重写策略:
        1.  **场景转食材:** 如果用户问场景（健身、减肥、生病），请重写为适合该场景的**具体食材**或**烹饪方式**。
        2.  **模糊转具体:** 如果用户问"好吃的"，重写为"热门家常菜"。
        3.  **保持原意:** 如果用户已经问了具体的菜（"宫保鸡丁"），不要改。

        ### 示例:
        - 原: "健身期间吃什么" -> 新: "鸡胸肉 牛肉 虾 鱼 清淡做法" (翻译成高蛋白食材)
        - 原: "减肥餐" -> 新: "凉拌菜 蔬菜沙拉 低脂 鸡肉"
        - 原: "适合老人的菜" -> 新: "炖菜 粥 软烂 易消化"
        - 原: "有什么下饭菜" -> 新: "回锅肉 麻婆豆腐 红烧肉 辣"
        - 原: "做菜" -> 新: "简单易做的美食菜谱"

        原始查询: "{query}"
        优化后的查询:""")

        chain = prompt | self.llm | StrOutputParser()
        rewritten_query = chain.invoke({"query": query}).strip()
        
        logger.info(f"查询重写: '{query}' → '{rewritten_query}'")
        return rewritten_query
    
    def generate_list_answer(self, context_docs: List[Document]) -> str:
        """对于'list'类型的查询，直接从元数据生成简洁的菜品列表。"""
        if not context_docs:
            return "抱歉，根据您的描述，我没有找到相关的菜品推荐。"

        dish_names = [doc.metadata.get('dish_name', '未知菜品') for doc in context_docs]
        # 简单去重
        dish_names = list(dict.fromkeys(dish_names))
        
        if not dish_names:
             return "抱歉，未能从相关信息中提取出菜品名称。"
        response = "为您推荐以下菜品：\n" + "\n".join([f"  - {name}" for name in dish_names])
        return response

    def _build_context(self, docs: List[Document], max_length: int = 3500) -> str:
        """构建用于生成答案的上下文字符串，包含元数据和长度控制。"""
        context_parts = []
        current_length = 0
        
        for doc in docs:
            metadata_header = f"--- 食谱: {doc.metadata.get('dish_name', 'N/A')} | 分类: {doc.metadata.get('category', 'N/A')} | 难度: {doc.metadata.get('difficulty', 'N/A')} ---\n"
            doc_text = metadata_header + doc.page_content
            
            if current_length + len(doc_text) > max_length:
                remaining_len = max_length - current_length
                context_parts.append(doc_text[:remaining_len] + "...")
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n\n".join(context_parts)

    def get_prompt_template(self, route_type: str) -> str:
        """根据路由类型，返回对应的Prompt模板字符串。"""
        if route_type == 'detail':
            return """你是一位专业的烹饪导师。请根据下面提供的食谱信息，精准回答用户的问题。

        ### 回答原则（至关重要）：
        1. **直击痛点**：如果用户询问的是**具体细节**（如“比例是多少”、“需要焯水吗”、“煮多久”），请**直接、正面地回答该问题**，并引用上下文中的关键数据佐证。**绝对不要**输出无关的完整食谱结构或废话。
        2. **完整教学**：只有当用户明确询问**整体做法**（如“怎么做”、“制作步骤”、“教我做这个”）时，才使用以下结构化格式：
           ### 🥘 菜品介绍
           ### 🛒 所需食材
           ### 👨‍🍳 制作步骤
           ### 💡 制作技巧

        请严格基于提供的上下文作答。

        ---
        上下文食谱信息:
        {context}
        ---
        用户问题: "{question}"

        你的回答:
        """
        else:  # general
            return """你是一位友善的烹饪助手。请根据下面提供的相关食谱信息，简洁、直接地回答用户的问题。如果信息不足，请诚实告知。

        ---
        相关食谱信息:
        {context}
        ---
        用户问题: "{question}"

        你的回答:
        """

    def _get_generation_chain(self, prompt_template: str) -> Any:
        """辅助函数，根据Prompt模板构建一个标准的LCEL生成链。"""
        return (
            {
                "context": lambda x: self._build_context(x["context_docs"]),
                "question": lambda x: x["query"]
            }
            | ChatPromptTemplate.from_template(prompt_template)
            | self.llm
            | StrOutputParser()
        )

    def generate_answer(self, query: str, context_docs: List[Document], route_type: str) -> Iterator[str]:
        """
        统一的生成入口，根据路由类型选择不同的生成策略，并支持流式输出。
        """
        if route_type == 'list':
            # 对于list类型，需要逐字符yield以确保流式输出正常工作
            response = self.generate_list_answer(context_docs)
            for char in response:
                yield char
            return

        prompt_template = self.get_prompt_template(route_type)
        chain = self._get_generation_chain(prompt_template)
        
        # 使用.stream()方法确保流式输出正常工作
        for chunk in chain.stream({"query": query, "context_docs": context_docs}):
            yield chunk


    
    def extract_filters(self, query: str, data_module: DataPreparationModule) -> dict:
        # 1. 获取动态分类列表
        dynamic_categories = list(data_module.available_categories.values())
        
        # 2. 在Python层面构建描述字符串
        metadata_description = f"""
        你可以根据以下字段进行过滤：
        
        - `category`: 菜品分类。可选值与定义如下：
            - '早餐': **包含鸡蛋、玉米、红薯、粥、馒头、吐司等早晨常吃的食物。** (注意：煮玉米、茶叶蛋属于此列，而非主食或荤菜)
            - '主食': 指正餐的主食，如米饭、面条、饺子、炒饭、炒面、饼。
            - '荤菜': 以肉类（猪牛羊鸡鸭）为主要食材的菜肴。
            - '素菜': 以蔬菜、豆制品、菌菇为主要食材的菜肴。
            - '水产': 鱼、虾、蟹、贝类。
            - '汤与粥': 各种汤类和正餐粥品。
            - '甜品': 蛋糕、饼干、糖水。
            - '饮料': 饮品、酒水。
            - '半成品加工': 速冻食品、空气炸锅半成品。
            
        - `difficulty`: 烹饪难度。可选值：['非常简单', '简单', '中等', '困难', '非常困难', '未知']。
        """
        
        # 3. 修改 Prompt：直接使用 {metadata_description}
        prompt = ChatPromptTemplate.from_template("""你是一个查询解析专家。你的任务是从用户的查询中，提取**明确的**元数据过滤条件。

        ### 可用的元数据字段及其说明:
        {metadata_description}

        ### ⚠️ 重要原则 (必须严格遵守):
        1.  **不要推断！不要推断！** 只有当用户**显式**提到了上述可选值中的词汇（或其精确同义词）时，才提取该条件。
        2.  **场景与食材不是分类：** 
            - 如果用户说 "健身"、"减肥"、"宴客"，**不要**提取 category。
            - 如果用户说 "土豆"、"牛肉"、"鸡蛋"，**不要**提取 category。
        3.  **宁缺毋滥：** 如果你不确定，或者用户只是在描述一种模糊的感觉，请返回空字典 `{{}}`。让检索系统通过语义去匹配，比错误的过滤更好。

        ### 示例:
        - "推荐一道简单的荤菜汤" -> {{"category": "汤品", "difficulty": "简单"}}
        - "家里只有鸡蛋和西红柿" -> {{}}
        - "健身期间吃什么" -> {{}}
        - "有什么素菜" -> {{"category": "素菜"}}

        ### 用户查询:
        "{query}"

        JSON输出:
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            # 4. 现在这里传 metadata_description 
            response_str = chain.invoke({
                "query": query,
                "metadata_description": metadata_description 
            })
            
            # 5. 解析JSON
            if "```json" in response_str:
                response_str = response_str.split("```json")[1].split("```")[0].strip()
            filters = json.loads(response_str)
            if not isinstance(filters, dict): return {}
            return filters
        except Exception as e:
            logger.error(f"解析过滤器JSON时失败: {e}")
            return {}
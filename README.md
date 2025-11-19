# 🍳 Cook-RAG | 智能菜谱问答助手

> **"今天吃什么？" —— 解决世纪难题的智能方案。**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![RAG](https://img.shields.io/badge/RAG-Advanced-green)](https://github.com/datawhalechina/all-in-rag)

**Cook-RAG** 是一个基于 **Advanced RAG (高级检索增强生成)** 架构构建的垂直领域问答系统。它不仅仅是一个简单的搜索工具，更是一个能够理解烹饪意图、提供结构化指导的智能 AI 厨房助手。

本项目数据源自 [Anduin2017/HowToCook](https://github.com/Anduin2017/HowToCook)。

---

## ✨ 核心亮点

本项目实践了 RAG 领域的一系列进阶技术：

*   🧠 **智能意图识别 (Router)**: 自动识别用户意图（推荐模式 / 详细指导模式 / 通用问答模式）。
*   🔄 **查询重写与优化 (Query Rewrite)**: 将模糊场景（如"健身吃什么"）转化为具体的食材检索词。
*   🔍 **混合检索 (Hybrid Search)**: 结合 **Vector (语义)** 和 **BM25 (关键词)**，使用 **RRF** 算法融合。
*   ⚖️ **重排序 (Rerank)**: 引入 **BGE-Reranker** 模型对召回结果进行精排，大幅提升准确率。
*   🏷️ **元数据前过滤 (Pre-filtering)**: 基于 LLM 提取查询中的分类/难度条件，精准过滤。
*   📚 **来源追踪**: 回答附带可折叠的参考食谱来源，拒绝 AI 幻觉。

## 🏗️ 技术架构

| 模块 | 技术选型 | 说明 |
| :--- | :--- | :--- |
| **LLM** | **Qwen-Plus (通义千问)** | 意图识别、重写、生成与总结 |
| **Embedding** | **BGE-Small-zh-v1.5** | 中文语义向量化 |
| **Rerank** | **BGE-Reranker-Base** | Cross-Encoder 重排序模型 |
| **Vector DB** | **FAISS** | 本地高性能向量索引 |
| **Framework** | **LangChain** | RAG 流程编排 |
| **UI** | **Streamlit** | 交互式 Web 界面 |

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/你的用户名/cook-rag.git
cd cook-rag

# 创建环境
conda create -n cook-rag python=3.12
conda activate cook-rag

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API 密钥

在项目根目录创建 `.env` 文件，填入阿里云 DashScope 密钥：

```env
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. 数据准备 

本项目已内置一份 `HowToCook` 的菜谱数据。

**🔄 如何获取最新菜谱？**
由于源项目 [HowToCook](https://github.com/Anduin2017/HowToCook) 仍在持续更新，您可以手动同步最新数据：
1.  下载源项目的最新 ZIP 包。
2.  将解压后的 `dishes/` 下的分类文件夹（如 `meat_dish`, `soup` 等）覆盖到本项目的 `data/` 目录。
3.  **⚠️ 重要：** 删除项目根目录下的 `vector_index/` 文件夹。
4.  重新运行程序，系统会自动检测并构建新的向量索引。

### 4. 启动应用

方式一：Web 界面 (推荐)
```bash
streamlit run cook-rag/app.py
```

方式二：命令行交互
```bash
python main.py
```

首次运行会自动下载 Embedding 和 Rerank 模型，并构建本地向量索引，请耐心等待。

## 📂 项目结构

```
cook-rag/
├── cook-rag/               # 前端应用源码
│   ├── app.py              # Streamlit Web 入口
│   └── style.css           # (可选) 界面样式文件
├── main.py                 # 命令行入口
├── config.py               # 全局配置
├── rag_modules/            # RAG 核心组件
│   ├── data_preparation.py     # 数据加载、清洗、父子文档切分
│   ├── index_construction.py   # 向量索引构建与缓存
│   ├── retrieval_optimization.py # 混合检索、RRF、重排序实现
│   └── generation_integration.py # LLM 路由、重写、生成逻辑
├── tests/                  # 测试代码
│   └── test_e2e.py         # 端到端系统测试
├── data/                   # 菜谱数据源
└── requirements.txt        # 依赖清单
```

## 📢 引用与致谢

数据来源: [HowToCook (程序员做饭指南)](https://github.com/Anduin2017/HowToCook)

License: CC BY-SA 4.0

技术参考: [Datawhale AI 夏令营](https://github.com/datawhalechina/all-in-rag)

## 📝 License

代码部分遵循 MIT License。

菜谱数据部分遵循 CC BY-SA 4.0。
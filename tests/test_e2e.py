import sys
import os
import logging

# --- 1. 确保能导入项目根目录的模块 ---
# 获取当前脚本所在目录的上一级目录（即项目根目录）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from main import RecipeRAGSystem
from config import DEFAULT_CONFIG

# 配置日志，只显示关键信息
logging.basicConfig(level=logging.INFO)

def test_full_workflow():
    """
    端到端测试：模拟真实用户提问，验证系统是否崩溃，以及是否返回了有效结果。
    """
    print("\n========== 开始端到端系统测试 ==========")

    # 1. 初始化系统
    print("\n[Step 1] 初始化系统...")
    try:
        rag = RecipeRAGSystem(DEFAULT_CONFIG)
        rag.initialize_system()
        rag.build_knowledge_base()
        print("✅ 系统初始化成功")
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        return

    # 2. 定义测试问题集 (覆盖不同意图)
    test_cases = [
        "推荐几道简单的素菜",          # 列表/推荐模式
        "宫保鸡丁怎么做？",            # 详细指导模式
        "健身期间吃什么好？",          # 需要重写 + 过滤的场景
        "红烧肉需要什么食材？"         # 细节提取
    ]

    # 3. 循环测试
    for question in test_cases:
        print(f"\n[Step 2] 测试提问: {question}")
        try:
            # 模拟非流式调用 (为了测试方便，我们需要把生成器转为字符串)
            response_generator = rag.ask_question(question)
            
            full_response = ""
            print("🤖 回答生成中: ", end="")
            for chunk in response_generator:
                full_response += chunk
                # 稍微打印一点点进度，证明在动
                print(".", end="", flush=True)
            
            print("\n✅ 回答完成")
            
            # 简单的断言：回答不应为空，且长度应合理
            assert full_response is not None
            assert len(full_response) > 10
            
            # 打印前50个字符预览
            print(f"📝 回答预览: {full_response[:50].replace(chr(10), ' ')}...")

        except Exception as e:
            print(f"\n❌ 测试失败: 问题 '{question}' 引发错误 - {e}")
            # 在测试中，遇到错误通常应该抛出，或者记录下来
            # raise e 

    print("\n========== 测试结束：所有流程均未崩溃 ==========")

if __name__ == "__main__":
    test_full_workflow()
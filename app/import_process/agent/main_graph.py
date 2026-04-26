# 加载环境变量：从 .env 文件读取配置（如Milvus地址、KG服务地址、BGE模型路径等）
from dotenv import load_dotenv
# 导入LangGraph核心依赖：StateGraph(状态图)、START/END(内置起始/结束节点常量)
from langgraph.graph import StateGraph, END, START
import json
from app.core.logger import logger
# 导入自定义状态类：统一管理工作流全程的所有数据（各节点共享/修改）
from app.import_process.agent.state import ImportGraphState, create_default_state
# # 导入所有自定义业务节点：每个节点对应知识库导入的一个具体步骤
from app.import_process.agent.nodes.node_entry import node_entry  # 入口节点：初始化参数、校验输入
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md  # PDF转MD：解析PDF文件为markdown格式
from app.import_process.agent.nodes.node_md_img import node_md_img  # MD图片处理：提取/下载markdown中的图片、修复图片路径
from app.import_process.agent.nodes.node_document_split import node_document_split  # 文档分块：将长文档切分为符合模型要求的小片段
from app.import_process.agent.nodes.node_item_name_recognition import node_item_name_recognition  # 项目名识别：从分块中提取核心项目名称（业务定制化）
from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding  # BGE向量化：将文本分块转换为向量表示（适配Milvus向量库）
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus  # 导入Milvus：将向量数据写入Milvus向量数据库


# 初始化环境变量：必须在配置读取前执行，确保后续节点能获取到环境变量中的配置信息
load_dotenv()

workflow = StateGraph(ImportGraphState)
workflow.add_node("node_entry", node_entry)  # 流程入口：参数初始化、输入校验
workflow.add_node("node_pdf_to_md", node_pdf_to_md)  # PDF转MD：非MD格式文件的前置处理
workflow.add_node("node_md_img", node_md_img)  # MD图片处理：保证文档中图片的可访问性
workflow.add_node("node_document_split", node_document_split)  # 文档分块：解决大文本无法向量化/推理的问题
workflow.add_node("node_item_name_recognition", node_item_name_recognition)  # 项目名识别：业务定制化步骤，提取核心业务标识
workflow.add_node("node_bge_embedding", node_bge_embedding)  # BGE向量化：文本→向量，为Milvus存储做准备
workflow.add_node("node_import_milvus", node_import_milvus)  # 向量入库：将向量数据持久化到Milvus

workflow.set_entry_point("node_entry")  # 设置流程入口

#条件边的路由函数
def route_after_entry(state: ImportGraphState) -> str:
    """
    入口节点后的条件路由逻辑
    :param state: 工作流全量状态对象，包含所有配置项和中间结果
    :return: 目标节点标识/END，LangGraph会自动跳转到对应节点
    """
    # 分支1：开启MD直接导入 → 跳过PDF转MD，直接执行MD图片处理
    if state.get("is_md_read_enabled"):
        return "node_md_img"
    # 分支2：开启PDF导入 → 执行PDF转MD，再走后续流程
    elif state.get("is_pdf_read_enabled"):
        return "node_pdf_to_md"
    # 分支3：未开启任何导入配置 → 直接终止工作流（END是LangGraph内置结束常量）
    else:
        return END

#添加条件边
workflow.add_conditional_edges(
    "node_entry",
    route_after_entry,
    {
        "node_md_img": "node_md_img",
        "node_pdf_to_md": "node_pdf_to_md",
        END: END
    })

#静态边
workflow.add_edge("node_pdf_to_md", "node_md_img")  # PDF转MD完成 → MD图片处理
workflow.add_edge("node_md_img", "node_document_split")  # MD处理完成 → 文档分块
workflow.add_edge("node_document_split", "node_item_name_recognition")  # 分块完成 → 项目名识别
workflow.add_edge("node_item_name_recognition", "node_bge_embedding")  # 项目名识别完成 → BGE向量化
workflow.add_edge("node_bge_embedding", "node_import_milvus")  # 向量化完成 → 导入Milvus向量库
workflow.add_edge("node_import_milvus", END)  # Milvus入库完成 → 工作流执行结束（END是内置结束节点）

kb_import_app = workflow.compile()

if __name__ == "__main__":
    # 测试用例：正常流程测试
    logger.info("===== 开始测试 =====")

    initial_state = create_default_state(local_file_path="万用表RS-12的使用.pdf")
    final_state = None

    # 只输出更最终的状态值（字典形式），不包含节点名称、执行日志、元数据等额外信息
    for event in kb_import_app.stream(initial_state):
        for key, value in event.items():
            logger.info(f"节点: {key}")
            final_state = value

    # 格式化输出最终状态
    logger.info(f"最终状态: {json.dumps(final_state, indent=4, ensure_ascii=False)}")

    logger.info("图结构:")
    # uv add grandalf
    kb_import_app.get_graph().print_ascii()

    logger.info("===== 测试结束 =====")

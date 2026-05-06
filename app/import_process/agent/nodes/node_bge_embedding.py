import sys
import os
from typing import Any, List, Dict

from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
from app.utils.task_utils import add_running_task,add_done_task
from app.core.logger import logger

def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    LangGraph核心节点：BGE-M3文本向量化处理
    主流程（串行执行，全流程异常隔离）：
        1. 输入校验：验证chunks有效性，核心数据缺失则终止当前节点
        2. 模型初始化：获取BGE-M3单例模型实例，避免重复加载
        3. 批量向量化：分批拼接文本、生成双向量，为切片绑定向量字段
        4. 状态更新：将带向量的chunks更新回全局状态，供下游Milvus入库节点使用
    参数：
        state: ImportGraphState - 流程全局状态对象，包含上游传入的chunks、task_id等数据
    返回：
        ImportGraphState - 更新后的状态对象，chunks字段新增dense_vector/sparse_vector
    异常处理：
        节点内所有异常均捕获，不终止整体LangGraph流程，仅记录错误日志
    """
    # 获取当前节点名称，用于日志和任务状态记录
    current_node = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行LangGraph节点：{current_node}")

    # 标记任务运行状态，用于任务监控/前端进度展示
    add_running_task(state.get("task_id", ""), current_node)
    logger.info("--- BGE-M3 文本向量化处理启动 ---")

    try:
        #获取要生成向量的chunks
        chunks=state['chunks']
        if not chunks:
            logger.error("输入的chunks为空，无法进行向量化处理")
            raise ValueError("输入的chunks为空，无法进行向量化处理")
        #给每个chunk生成向量
        #01 获取嵌入式模型的客户端 但是embedding里面已经封装好了 不需要手动获取
        #02 批量向量化
        '''
        1.什么内容生成向量？
            chunks里面的每个chunk的content字段 
            用户 问题 （）  向量混合检索-》向量 
            eg 华为怎么开机  华为是item_name   华为手机---》content：充电器怎么开机xxxx等等content没有主语
            所以光有content不合适 主语可能不匹配  再加上item_name可能更合适
        '''
        result = generate_embeddings(chunks)
        #完善chunk的属性添加稠密和稀疏向量

        
        
        # state['chunks'] = output_data
        # logger.info(f"--- BGE-M3 向量化处理完成，共处理 {len(output_data)} 条文本切片 ---")
        add_done_task(state.get("task_id", ""), current_node)
    except Exception as e:
        # 捕获节点所有异常，记录错误堆栈，不中断整体流程
        logger.error(f"BGE-M3向量化节点执行失败：{str(e)}", exc_info=True)

    # 返回更新后的状态对象，传递至下游节点
    return state
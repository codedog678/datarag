import os
import sys
from os.path import splitext

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.format_utils import format_state
from app.utils.task_utils import add_running_task, add_done_task


def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    未来要实现:
    0. 进入节点的日志输出 节点+核心参数
        记录任务状态  哪个开始了  埋点
    1. 接收文件路径。
        参数校验 local_file_path 有没有传入文件-》end local_dir 有没有输出文件-》创建一个临时
    2. 判断文件类型 (PDF/MD) 
        修改 state 中的路由标记 (is_pdf_read_enabled / is_md_read_enabled)
        md_path  或者是  pdf_path   赋值
        file_title 提取文件名
    3. 结束节点的日志输出 节点+核心参数
        记录任务状态  哪个结束了  埋点
    """
    #0. 进入节点的日志输出 节点+核心参数
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] 开始执行！现在的状态是: {state}")
    #add running task 在内存中记录当前正在运行的任务节点 用于前端的sse推送
    add_running_task(state['task_id'],function_name)
    
    #1. 接收文件路径。
    local_file_path = state['local_file_path']
    if not local_file_path:
        logger.error(f"[{function_name}] 未传入文件路径，无法继续执行！") #因为后续路由的条件边自动有end
        return state

    #2. 判断文件类型 (PDF/MD) 
    if local_file_path.endswith('.pdf'):
        state['is_pdf_read_enabled'] = True
        state['pdf_path'] = local_file_path
    elif local_file_path.endswith('.md'):
        state['is_md_read_enabled'] = True
        state['md_path'] = local_file_path
    else:
        logger.error(f"[{function_name}] 文件类型不支持，无法继续执行！") #因为后续路由的条件边自动有end
        return state
    #提取file_title 如果大模型没有识别出来当前文件对应的item_name，则使用文件名作为item_name
    # 从 xxxx/bbbb/a.pdf 获取a.pdf  
    # 可能有多个点，有隐患：file_title = os.path.basename(local_file_path).split('.')[0]
    file_title = os.path.basename(local_file_path).rsplit('.', 1)[0]
    state['file_title'] = file_title
    #3. 结束节点的日志输出 节点+核心参数
    logger.info(f">>> [{function_name}] 执行完毕！现在的状态是:{format_state(state)}")
    add_done_task(state['task_id'],function_name) 

    return state
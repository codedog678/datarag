import os
import sys
from typing import List, Dict, Any
# 导入Milvus相关依赖
from pymilvus import DataType
# 导入自定义模块
from app.import_process.agent.state import ImportGraphState
from app.clients.milvus_utils import get_milvus_client
from app.utils.task_utils import add_running_task
from app.core.logger import logger
from app.conf.milvus_config import milvus_config
from app.utils.escape_milvus_string_utils import escape_milvus_string

#http://milvus-standalone:19530 不是本机电脑单独配置了
#因为在docker-compose.yml中配置了milvus-standalone的别名 所以这里可以直接使用milvus-standalone:19530

def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入向量库 (node_import_milvus)
    将处理好的向量数据写入 Milvus 数据库。
    未来要实现:
    1. 连接 Milvus。
    2. 根据 item_name 删除旧数据 (幂等性)。
    3. 批量插入新的向量数据。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] 开始执行！现在的状态是: {state}")
    #add running task 在内存中记录当前正在运行的任务节点 用于前端的sse推送
    add_running_task(state['task_id'],function_name)

from app.query_process.agent.state import ImportGraphState
import sys
import sys
import os
from app.utils.task_utils import add_running_task,add_done_task
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import create_hybrid_search_requests,hybrid_search,get_milvus_client
from app.core.logger import logger
from app.conf.milvus_config import milvus_config
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())


def node_search_embedding(state):
    """
    核心节点函数：基于已确认商品名+改写后的用户问题，执行Milvus向量数据库混合检索
    流程：用户问题向量化 → 构造带商品名过滤的混合搜索请求 → 执行稠密+稀疏混合检索 → 返回检索结果
    :param state: Dict - 会话状态字典，包含上游传递的核心信息，关键字段：
                  {
                      "session_id": str,        # 会话唯一标识
                      "rewritten_query": str,   # step3改写后的完整用户问题（含商品名）
                      "item_names": list[str],  # step6已确认的标准化商品名列表
                      "is_stream": bool/None    # 是否为流式响应，可选
                  }
    :return: Dict - 检索结果字典，仅包含embedding_chunks字段，供下游节点使用：
             {
                 "embedding_chunks": List[Dict]  # Milvus检索结果列表，无结果则为空列表
                                                 # 每个元素为一条匹配的向量数据，含业务字段
             }
    """
    logger.info("---search_milvus 开始处理---")
    add_running_task(state["session_id"],sys._getframe().f_code.co_name,state["is_stream"])

    #1.获取数据
    rewritten_query = state["rewritten_query"]
    item_names = state["item_names"]

    #2.用户问题向量化 稀疏+稠密
    embeddings = generate_embeddings(rewritten_query)
    dense_vector, sparse_vector = embeddings['dense'][0], embeddings['sparse'][0]
    #3.向量数据库混合查询
    #3.1创建混合查询请求  item_names过滤查询+稠密+稀疏混合查询
    #expr是混合查询的混合条件
    search_requests = create_hybrid_search_requests(
        dense_vector=dense_vector, 
        sparse_vector=sparse_vector, 
        expr=f'item_name in {[f"'{name}'" for name in item_names]}')
    
    #3.2执行混合查询请求
    milvus_client = get_milvus_client()
    results = hybrid_search(
        milvus_client=milvus_client,
        reqs=search_requests,  
        collection_name=milvus_config.chunks_collection,
        ranker_weights=(0.8,0.2),
        norm_score=True,
        limit=5,
        output_fields=["item_name","parent_title","file_title","title","chunk_id","content"]
        )
    
    #4.处理查询结果及赋值 embedding_chunks字段
    embedding_chunks = results[0] if results else []
    #5.返回结果
    add_done_task(state["session_id"],sys._getframe().f_code.co_name,embedding_chunks)
    return {"embedding_chunks": embedding_chunks}

import sys
from typing import List, Dict, Any, Tuple
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger


def reciprocal_rank_fusion(source_with_weight: List[Tuple[List[Dict[str, Any]], float]],top_k:int=5) :
    """
    倒排列表融合算法
    :param source_with_weight: 每个元素为 (文档列表, 权重) 的元组列表
    :return: 融合后的文档列表
    """
    #准备两个容器 一个记录历史得分 一个记录chunk片段
    score_dict={}   #key:chunk_id value:score
    #一个记录chunk片段
    chunk_dict={} 
    #循环处理每个集合中的数据
    for source,weight in source_with_weight:
        #循环处理每个文档
        for rank,doc in enumerate(source,start=1):
            #获取chunk_id
            chunk_id=doc['entity']['chunk_id']
            #计算得分
            score_dict[chunk_id]=score_dict.get(chunk_id,0)+(1.0/(60+rank))*weight
            chunk_dict[chunk_id]=doc 
    #融合和排序
    merged=[]
    for chunk_id,score in score_dict.items():
        merged.append({
            'chunk':chunk_dict[chunk_id],
            'score':score,
        })
    #排序
    merged.sort(key=lambda x:x['score'],reverse=True)
    #截断
    merged=merged[:top_k]
    #获取chunk的排名数据
    rank_chunks=[item['chunk'] for item in merged]
    logger.info(f"RRF 融合后结果：{rank_chunks}")
    return rank_chunks

            
def node_rrf(state):
    """
    RRF (Reciprocal Rank Fusion) 倒数排名融合节点
    
    功能：
    将来自不同检索源（如 Embedding 检索、HyDE 检索、知识图谱检索等）的结果进行融合排序。
    RRF 是一种无需训练的算法，仅根据文档在不同列表中的排名来计算最终得分。
    
    步骤：
    1. 提取各路检索结果：从 state 中获取 embedding_chunks 和 hyde_embedding_chunks。
    2. 结果标准化：将不同格式的检索结果统一转换为包含 chunk_id 的实体列表。
    3. 设置权重：为不同来源分配权重（当前配置：Embedding=1.0, HyDE=1.0）。
    4. 执行 RRF：计算融合分数并重新排序。
    5. 结果截断：保留 Top K 个结果。
    6. 更新状态：将融合后的结果存入 state["rrf_chunks"]。
    """
    logger.info("---RRF (倒数排名融合) 开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 第一步：获取上游检索节点返回的文档
    # 上游检索节点（Milvus hybrid_search）返回的通常是 hit 列表：
    #  {"entity": {...fields...}, "distance": ...}
    # RRF 需要使用 chunk_id 做去重与计分，因此这里必须保留 entity（而不是仅抽取 content 字符串）。
    embedding_chunks = state.get("embedding_chunks", [])
    hyde_embedding_chunks = state.get("hyde_embedding_chunks", [])
    
    #2.数据进行整合
    source_with_weight=[
        (embedding_chunks,1.0),
        (hyde_embedding_chunks,1.0),
    ]

    #3.rrf融合
    rrf_response=reciprocal_rank_fusion(source_with_weight)

    #4.将排序后的数据添加到state中
    state["rrf_chunks"]=rrf_response
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state
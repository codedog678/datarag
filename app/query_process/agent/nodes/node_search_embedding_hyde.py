import time
import sys

from langchain.messages import HumanMessage
from app.utils.task_utils import  add_done_task,add_running_task
from app.lm.lm_utils import *
from app.lm.embedding_utils import *
from app.clients.milvus_utils import *
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

def generate_hyde_doc(query):
    """
    生成假设文档（HyDE文档）
    :param query: 用户问题（改写后的查询）
    :return: 假设文档（LLM生成）
    """
    llm_client = get_llm_client()
    hyde_prompt=load_prompt("hyde_prompt", rewritten_query=query)
    messages=[
        HumanMessage(content=hyde_prompt)
    ]
    response = llm_client.invoke(messages)
    logger.info(f"LLM生成假设文档完成，内容：{response.content}")
    return response.content

def search_embedding_hybrid(query, hyde_doc, item_names):
    """
    执行混合检索（BGE-M3 + Milvus）
    :param query: 用户问题（改写后的查询）
    :param hyde_doc: 假设文档（LLM生成）
    :param item_names: 已确认的商品名列表
    :return: [[]->结果 id 分数 实体列表]
    """
    #拼接问题+假设文档
    query_hyde = query + " " + hyde_doc
    #生成BGE-M3稠密+稀疏向量
    query_embedding = generate_embeddings([query_hyde])
    #生成混合检索ANNSearchRequest
    reqs=create_hybrid_search_requests(
        query_embedding["dense"][0],
        query_embedding["sparse"][0],
        expr=f'item_name in [{", ".join(chr(34) + name + chr(34) for name in item_names)}]')
    #进行混合查询处理
    milvus_client = get_milvus_client()
    results = hybrid_search(
        client=milvus_client,
        reqs=reqs,  
        collection_name=milvus_config.chunks_collection,
        ranker_weights=(0.8,0.2),
        norm_score=True,
        limit=5,
        output_fields=["item_name","parent_title","file_title","title","chunk_id","content"]
        )
    #处理返回结果
    return_chunks = results[0] if results else []
    logger.info(f"检索结果：{return_chunks}")
    return return_chunks

def node_search_embedding_hyde(state):
    """
    HyDE (Hypothetical Document Embedding) 检索节点
    核心思想：通过LLM生成假设性答案（HyDE文档），将其向量化后用于检索，以解决短查询语义稀疏问题。

    执行步骤：
    1. 参数提取：从会话状态中获取改写后的查询（rewritten_query）和已确认的商品名（item_names）。
    2. 生成假设文档 (Step 1)：调用LLM，基于用户问题生成一段假设性的理想回答（即HyDE文档）。
    3. 混合检索 (Step 2)：
       - 将“用户问题 + 假设文档”合并，生成BGE-M3稠密+稀疏向量。
       - 在Milvus中执行混合检索（带商品名过滤），召回最相似的知识切片。
    4. 结果封装：返回检索到的切片列表和生成的假设文档，更新会话状态。

    :param state: 会话状态字典，包含 session_id, rewritten_query, item_names 等
    :return: 包含 hyde_embedding_chunks (检索结果) 和 hyde_doc (假设文档) 的字典
    """
    logger.info("---HyDE (假设文档检索) 节点开始处理---")
    # 记录任务开始状态
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    #1.获取数据
    rewritten_query = state["rewritten_query"]
    item_names = state["item_names"]

    #2.生成假设文档
    hyde_doc = generate_hyde_doc(rewritten_query)

    #3.问题+答案 混合检索
    result = search_embedding_hybrid(rewritten_query, hyde_doc, item_names)

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    logger.info("---HyDE (假设文档检索) 节点处理完成---")
    return {"hyde_embedding_chunks": result}

import sys
import os
import json
import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from mpmath import limit

from app.core.load_prompt import load_prompt
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.clients.mongo_history_utils import get_recent_messages, save_chat_message, update_message_item_names
from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from dotenv import load_dotenv,find_dotenv
from app.core.logger import logger

load_dotenv(find_dotenv())

def extract_item_names_and_rewrite_query(original_query, history_chat):
    """
    从历史对话记录中提取商品名称和重写用户问题。
    输入：original_query 原始用户问题
          history_chat 历史对话记录
    输出：item_names 商品名称列表 []
          rewritten_query 重写后用户问题 str
    """
    #1.准备提示词
    #？？？？why  直接塞入不行吗？
    history_text=''
    for message in history_chat:
        #history_text+=f"聊天角色：{message['role']}\n聊天内容：{message['text']}\n重写问题：{message.get('rewritten_query', '')}\n关联主体：{message.get('item_names', [])}\n\n"
        #感觉有点过度设计了 没有必要
        history_text += f"{message['role']}: {message['text']}\n"
    prompt=load_prompt("rewritten_query_and_itemnames", history_text=history_text, query=original_query)

    #2.模型调用
    llm_client = get_llm_client(json_mode=True)
    messages=[HumanMessage(content=prompt)]
    response = llm_client.invoke(messages)
    
    #3.结果解析
    content=response[0].content
    if content.startswith("```json"):
        content=content.replace("```json","").replace("```","")
    dict_data=json.loads(content)
    if 'item_names' not in dict_data:
        dict_data['item_names']=[]
    if 'rewritten_query' not in dict_data:
        dict_data['rewritten_query']=original_query
    #4.封装返回
    return dict_data


def node_item_name_confirm(state):
    """
    节点功能：确认用户问题中的核心商品名称。
    核心目标：1.提取 item_name 大模型从历史对话+本次提问 提取 然后向量库搜索 打分得到ABC...
            2.利用模型重写用户问题 确保后续查询召回率更高
    过程：
    1.获取历史对话记录
    2.保存当前次的聊天记录
    3.提取 item_name 重写问题
    4.向量库查询
    5.对分 分类处理 A确认集合  B可选集合
    6.处理确认和可选集合  有确认 则直接返回确认结果  无确认有可选 则返回可选集合  无确认无可选 则answer赋值（配和图结构）
    7.补充state状态  item_name  rewrite_query  history
    输入：state['original_query'] 原始用户问题
    输出：更新 state['item_names']
    """
    # 记录任务开始
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])
    
    #1.获取历史对话记录
    history_chat=get_recent_messages(state["session_id"])
    #2.保存当前次的聊天记录
    message_id=save_chat_message(state["session_id"], 
                      role="system", 
                      text=state["original_query"],
                      rewritten_query=state.get("rewrite_query", ''),
                      item_names=state.get("item_names", []),
                      image_urls=state.get("image_urls", []),)
    #3.提取 item_name  利用LLM模型 重写问题
    #参数：original_query, history_chat
    itemnames_and_requery=extract_item_names_and_rewrite_query(state["original_query"], history_chat)

    
    # 记录任务结束
    add_done_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])

    print(f"---node_item_name_confirm---处理结束")

    return {"item_names": ["示例商品"]}
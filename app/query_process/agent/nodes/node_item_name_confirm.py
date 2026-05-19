import sys
import os
import json

from langchain_core.messages import SystemMessage, HumanMessage
from app.conf.milvus_config import milvus_config
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
    '''
    怎么确保返回的是json格式的数据？：1.在提示词中指定json格式（给出示例） 2.在模型调用时指定json_mode=True（json格式化可以配置）
    3.返回格式的校验 （解析器）
    '''
    #3.结果解析
    content = response.content
    if content.startswith("```json"):
        content=content.replace("```json","").replace("```","")
    #json.loads(content)  把字符串转换为字典
    #json.dumps(dict_data)  把字典转换为字符串
    dict_data=json.loads(content)
    if 'item_names' not in dict_data:
        dict_data['item_names']=[]
    if 'rewritten_query' not in dict_data:
        dict_data['rewritten_query']=original_query
    #4.封装返回  字典
    return dict_data

def query_milvus_item_names(item_names):
    """
    向量库查询商品名称 混合查询：稠密（语义）+稀释（关键字）
    输入：item_names 商品名称列表 []可能不准确
    输出：query_milvus_results 查询结果列表 
    [{extracted:(模型提取的item_name),matches:[{item_name:名字（数据库搜到的），score:分数（模型给出的）},{..}]}]
    """
    final_results=[]
    
    #1.获取Milvus客户端  官网查询都看看
    milvus_client=get_milvus_client()
    
    #2.item_name 先转成向量 已经封装好的函数
    embeddings=generate_embeddings(item_names)
    #3.混合查询   权重重排进行混合查询
    for index,item_name in enumerate(item_names):
        #获取对应的向量
        dense_vector=embeddings['dense'][index]
        sparse_vector=embeddings['sparse'][index]
        #拼接对应的ANNSearchRequest 已经封装好了
        reqs=create_hybrid_search_requests(dense_vector, sparse_vector)
        #定义权重 混合查询
        response=hybrid_search(
            client=milvus_client, 
            collection_name=milvus_config.item_name_collection, 
            reqs=reqs, 
            ranker_weights=(0.5, 0.5), 
            norm_score=True,  #归一化 
            )
        ''''
        看官网的返回形式是什么样 因为输出字段output_fields 定位为item_name 所以返回的结果是
        [[
           {id:xx  , distance:xx  , entity:{item_name:xx}},
           {id:xx  , distance:xx  , entity:{item_name:xx}}...
        ]]

        实际要的返回形式
        [{extracted:(模型提取的item_name),matches:[{item_name:名字（数据库搜到的），score:分数（模型给出的）},{..}]}]
        '''

        #结果解析
        matches=[]
        if response and len(response)>0:
            for hit in response[0]:
                entity=hit.get('entity', {})
                hit_item_name=entity.get('item_name', '')
                score=hit.get('distance', 0.0)
                if hit_item_name:   
                    matches.append({'item_name':hit_item_name,'score':score})
        final_results.append({'extracted':item_name,'matches':matches})

    #4.提取查询结果封装返回的数据   
    return final_results

def classify_item_names(query_milvus_results):
    """
    通过向量数据库查询出的item_name  根据分数归纳出两个列表
    输入：query_milvus_results 查询结果列表 
    [{extracted:(模型提取的item_name),matches:[{item_name:名字（数据库搜到的），score:分数（模型给出的）},{..}]}]
    输出：{确定的item_name:[x,x,x],可选的item_name:[x,x,x]}
    {
    confirmed_item_names:[],
    optional_item_names:[]}
    }
    评分规则：最高分>=0.85确认，其余>=0.6可选（根据实际需求修改）
    """
    confirmed_item_names=[]
    optional_item_names=[]
    CONFIRM_THRESHOLD = 0.85  # 确认阈值
    OPTIONAL_THRESHOLD = 0.6  # 可选阈值

    for item_name in query_milvus_results:
        matches=item_name.get('matches', [])
        #排序 搜到的结果根据分数排序 从高到低
        matches.sort(key=lambda x: x['score'], reverse=True)
        logger.info(f"[分类商品名] 提取='{item_name.get('extracted', '')}' 匹配结果：{[(m['item_name'],round(m['score'],3)) for m in matches]}")

        if not matches:
            continue

        #最高分够高 → 确认最高分，剩余>=0.6的作为可选
        if matches[0]['score'] >= CONFIRM_THRESHOLD:
            confirmed_item_names.append(matches[0]['item_name'])
            for item in matches[1:]:
                if item['score'] >= OPTIONAL_THRESHOLD:
                    optional_item_names.append(item['item_name'])
        else:
            #没有高分 → 全部>=0.6的作为可选
            for item in matches:
                if item['score'] >= OPTIONAL_THRESHOLD:
                    optional_item_names.append(item['item_name'])

    return {'confirmed_item_names':list(set(confirmed_item_names)),
            'optional_item_names':list(set(optional_item_names))[:5]}   

def deal_list(item_results,history_chat,state,rewritten_query):
    '''
    处理确认和可选集合  有确认 则直接返回确认结果  无确认有可选 则返回可选集合  无确认无可选 则answer赋值（配和图结构）
    输入：item_results 确认和可选集合
    history_chat 历史对话记录
    输出：历史对话记录
    '''
    #1.获取两个集合
    confirmed_item_names=item_results.get('confirmed_item_names', [])
    optional_item_names=item_results.get('optional_item_names', [])
    #2.确认集合有数据进行处理
    if len(confirmed_item_names)>0:
        #2.1 更新聊天记录 item_names-》confirmed_item_names
        #2.2 更新state状态
        state['item_names']=confirmed_item_names
        state['rewritten_query']=rewritten_query
        state['history_chat']=history_chat
        if 'answer' in state:
            del state['answer']
        logger.info(f"有确认商品，名称为：{confirmed_item_names}")
        return state
    #3.确认集合没数据处理可选集合 下一轮重写问题
    if len(optional_item_names)>0:
        option_names='、'.join(optional_item_names)
        answer=f"您好，您是想咨询以下哪个商品：{option_names}？请下次提问明确商品名称"
        state['answer']=answer
        logger.info(f"无确认商品，可选商品名称为：{optional_item_names}")
        return state
    #4.都没数据
    answer=f"抱歉，暂时没有找到您想要的商品，请您重新描述您的问题。"
    logger.info(f"无确认商品，无可选商品")
    state['answer']=answer
    return state



def node_item_name_confirm(state):
    """
    节点功能：确认用户问题中的核心商品名称。
    核心目标：1.提取 item_name 大模型从历史对话+本次提问 提取 然后向量库搜索 打分得到ABC...
            2.利用模型重写用户问题 确保后续查询召回率更高
    过程：
    1.获取历史对话记录
    3.提取 item_name 重写问题
    4.向量库查询
    5.对分 分类处理 A确认集合  B可选集合
    6.处理确认和可选集合  有确认 则直接返回确认结果  无确认有可选 则返回可选集合  无确认无可选 则answer赋值（配和图结构）
    7.补充state状态  item_name  rewritten_query  history
    输入：state['original_query'] 原始用户问题
    输出：更新 state['item_names']
    """
    # 记录任务开始
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])
    
    #1.获取历史对话记录 第一次访问chat.html前端用随机字符串+时间戳生成一个session_id 存到了localStorage
    history_chat=get_recent_messages(state["session_id"])
    
    #3.提取 item_name  利用LLM模型 重写问题
    #参数：original_query, history_chat
    itemnames_and_requery=extract_item_names_and_rewrite_query(state["original_query"], history_chat)
    item_names=itemnames_and_requery.get('item_names', [])
    rewritten_query=itemnames_and_requery.get('rewritten_query', state["original_query"])
    #4.向量库查询
    #参数：item_names 模型给出的 但不一定与向量数据库完全相同
    item_results={'confirmed_item_names':[],'optional_item_names':[]}
    if len(item_names)>0:
        #查询 item_names是一个列表 提取可能不止一个[1,2...] 那么查询也就不止查一次
        #返回：[extracte:(模型提取的item_name),matches:[{item_name:名字（数据库搜到的），score:分数（模型给出的）},{..}]]
        query_milvus_results=query_milvus_item_names(item_names)
        #5.打分 分类处理 A确认集合  B可选集合
        #参数：query_milvus_results
        #返回：{确定的item_name:[x,x,x],可选的item_name:[x,x,x]}
        item_results=classify_item_names(query_milvus_results)
        #6.处理确认和可选集合  有确认 则直接返回确认结果  无确认有可选 则返回可选集合  无确认无可选 则answer赋值（配和图结构）
    state=deal_list(item_results,history_chat,state,rewritten_query)
    # #7.记录本次聊天对话（看answer)  换位置到输出节点node_answer_output
    # if state.get('answer'):
    #     save_chat_message(state["session_id"], 
    #                       role="assistant",
    #                       text=state['answer'],
    #                       rewritten_query=state.get("rewritten_query"),
    #                       )
    #2.保存当前次的聊天记录
    message_id=save_chat_message(state["session_id"],
                      role="user",
                      text=state["original_query"],
                      rewritten_query=state.get("rewritten_query", ''),
                      item_names=state.get("item_names", []),
                      image_urls=state.get("image_urls", []),)

    # 记录任务结束
    add_done_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])

    print(f"---node_item_name_confirm---处理结束")

    return state
if __name__ == "__main__":
    # 模拟输入状态
    mock_state = {
        "session_id": "test_session_001",
        "original_query": "它好用吗？",
        "is_stream": False
    }

    print(">>> 开始测试 node_item_name_confirm...")
    try:
        # 运行节点
        result_state = node_item_name_confirm(mock_state)

        print("\n>>> 测试完成！最终状态:")
        print(json.dumps(result_state, indent=2, ensure_ascii=False,default=str))

        # 简单验证
        if result_state.get("item_names"):
            print(f"\n[PASS] 成功提取并确认商品名: {result_state['item_names']}")
        else:
            print(f"\n[WARN] 未确认到商品名 (可能是向量库无匹配或LLM未提取)")

    except Exception as e:
        print(f"\n[FAIL] 测试运行出错: {e}")
import sys
from app.core.logger import logger
from app.utils.task_utils import *

from dotenv import load_dotenv
import sys
from app.lm.reranker_utils import get_reranker_model
from app.utils.task_utils import add_running_task

load_dotenv()

# -----------------------------
# Rerank / TopK 全局常量（不从 state 读取）
# -----------------------------
# 动态 TopK 硬上限：最多取前 N 条（<=10）
RERANK_MAX_TOPK: int = 10
# 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
RERANK_MIN_TOPK: int = 1
# 断崖阈值（相对）
RERANK_GAP_RATIO: float = 0.25
# 断崖阈值（绝对） 最大间断分值
RERANK_GAP_ABS: float = 0.5

def merge_rrf_mcp(state):
    #1.获取不同路的数据
    rrf_list=state['rrf_chunks']
    mcp_list=state['web_search_docs']
    
    #2.准备一个列表容器
    chunks_list=[]
    
    #3.循环进行数据添加
    #3.1 rrf
    for chunk in rrf_list:
        entity=chunk.get('entity')
        chunk_id=entity.get('chunk_id')
        content=entity.get('content')
        title=entity.get('title')
        chunks_list.append({'chunk_id':chunk_id,'text':content,'title':title,'source':"local","url":''})

    #3.2 mcp
    for chunk in mcp_list:
       text=chunk.get('snippet')
       url=chunk.get('url')
       title=chunk.get('title')
       chunks_list.append({'chunk_id':'','text':text,'title':title,'source':"web","url":url})

    logger.info(f"多路数据融合，最终结果：{chunks_list}")
    return chunks_list

def rerank_doc_list(doc_list,state):
    #1.获取原有的问题
    rewritten_query=state['rewritten_query'] or state['original_query'] #兜底使用原始问题
    #2.获取问题所有的答案
    text_list=[doc['text'] for doc in doc_list]
    #3.加载reranker模型 已经封装好的直接调用
    rerank=get_reranker_model()
    #4.处理数据 设置成：问题+答案 装到列表中  调用打分方法
    questions_pairs=[[rewritten_query,text] for text in text_list]
    #normalize=True 归一化处理  默认是False  归一化处理后  分数范围在[0,1]之间
    rerank_scores=rerank.compute_score(questions_pairs,normalize=True)
    #5.将原有数据添加对应的分数
    #返回值[xxx,xxx,xxx,xx....]
    doc_list_with_scores=[{'doc':doc,'score':score} for doc,score in zip(doc_list,rerank_scores)]
    #6.根据分数进行排序
    doc_list_with_scores.sort(key=lambda x: x['score'], reverse=True)
    logger.info(f"rerank结果：{doc_list_with_scores}")
    return doc_list_with_scores


def topk_doc_list(reranker_list):
    '''
    动态 TopK 处理：
    之前完成的：
    问题稠密和稀疏-》问题进行混合检索
                                                   -》rrf 同源 
    问题+假设性答案稠密-》问题+假设性答案进行混合检索                       ->rerank  -》topk
                                                     mcp
    '''
    max_topk=RERANK_MAX_TOPK  #最多取前 N 条（<=10）
    min_topk=RERANK_MIN_TOPK  #至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
    gap_ratio=RERANK_GAP_RATIO   #断崖阈值（相对）  (1-2)/1  
    gap_abs=RERANK_GAP_ABS  #断崖阈值（绝对） 最大间断分值 0.9 0.74 ->0.16

    topk=min(max_topk,len(reranker_list))

    if topk>min_topk:
        #目的是实现 相邻两个元素的比较 （双指针）  min_topK是个数 现在要找下标
        #index 最后一个值必须是 倒数第二个元素的索引 ，否则 index+1 会越界。
        for index in range(min_topk-1,topk-1):
            #双指针 前后
            item1=reranker_list[index].get('score',0.0)
            item2=reranker_list[index+1].get('score',0.0)
            #分母不能为0
            gap=item1-item2
            rel=gap/(abs(item1)+1e-6)
            if gap>=gap_abs or rel>=gap_ratio:
                #断崖
                logger.info(f"数据集合{index}和{index+1}发生了断崖，结束循环")
                topk=index+1
                break
    topk_doc_list_list=reranker_list[:topk]
    logger.info(f"topk截取长度：{topk}")
    logger.info(f"topk结果：{topk_doc_list_list}")
    return topk_doc_list_list

    


def node_rerank(state):
    """
    非同源排序
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    rrf+mcp 精排序  chunk打分  topk重排
    """

    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    #1.非同源路的结果合并  
    '''
    rrf={id:chunk_id,distance:0.x,entity:{chunk_id,content,title}}
    mcp={snippet,title,url}

    {\"snippet\":\"《哈哈哈哈哈 第六季》综艺在线观看- 全集大陆综艺- 口袋影视 哈哈哈哈哈第六季\",
            \"hostname\":\"美食\",\"hostlogo\":\"\",
            \"title\":\"《哈哈哈哈哈 第六季》综艺在线观看- 全集大陆综艺- 口袋影视\",
            \"url\":\"http://koudaxiang.com/news/1439037.html\"},{\"snippet\":\"《哈哈哈哈哈第四季》综艺全集-免费在线观看高清完整版-片库网 哈哈哈哈哈第四季\",\"hostname\":\"\",\"hostlogo\":\"https://pkwdy.com/template/pkwdy/images/favicon.ico\",\"title\":\"《哈哈哈哈哈第四季》综艺全集-免费在线观看高清完整版-片库网\",\"url\":\"http://pkwdy.com/pplayk/1510-1-1.html\"}],\"request_id\":\"f6469ac9-211a-4588-ba92-e5ca447887da\",\"tools\":[],\"status\":0}",
            "type": "text"
            }
    '''
    doc_list=merge_rrf_mcp(state)
    #2.启用reranker模型进行打分
    reranker_list=rerank_doc_list(doc_list,state)
    #3.启动算法进行防断崖处理 以及 topk处理
    final_doc_list=topk_doc_list(reranker_list)
    #4.结果装到state中
    #展平嵌套结构：将 {doc:{text,source,title,...}, score:x} 扁平化为 {text,source,title,...,score:x}
    flat_list=[]
    for item in final_doc_list:
        flat_doc=item.get('doc',{})
        flat_doc['score']=item.get('score',0.0)
        flat_list.append(flat_doc)
    state['reranked_docs']=flat_list 

    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state


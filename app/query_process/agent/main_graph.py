from langgraph.graph import StateGraph, END
from app.query_process.agent.state import QueryGraphState
# 导入所有节点函数
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp

builder = StateGraph(QueryGraphState)

builder.add_node('node_item_name_confirm', node_item_name_confirm)
builder.add_node('node_answer_output', node_answer_output)
builder.add_node('node_rerank', node_rerank)
builder.add_node('node_rrf', node_rrf)
builder.add_node('node_search_embedding', node_search_embedding)
builder.add_node('node_search_embedding_hyde', node_search_embedding_hyde)
builder.add_node('node_web_search_mcp', node_web_search_mcp)

builder.set_entry_point('node_item_name_confirm')

#node_item_name_confirm 可能出现没有明确主体item_name的情况 会提前结束返回用户提示，提醒明确内容
#条件边 node_item_name_confirm ->多路召回 | 答案生成返回前端 


def router_after_item_name_confirm(state: QueryGraphState):
    if state['answer']:
        return 'node_answer_output'
    else:
        return "node_search_embedding", "node_search_embedding_hyde", "node_web_search_mcp"
builder.add_conditional_edges('node_item_name_confirm', 
                              router_after_item_name_confirm,
                              {
                                  'node_answer_output': 'node_answer_output',
                                  'node_search_embedding': 'node_search_embedding',
                                  'node_search_embedding_hyde': 'node_search_embedding_hyde',
                                  'node_web_search_mcp': 'node_web_search_mcp'
                              })

#切记：并发边不能直接返回state 而是改了什么返回什么
builder.add_edge('node_search_embedding','node_rrf')#实际上向量搜索和假设检索是同源的，有分可以rrf
builder.add_edge('node_search_embedding_hyde','node_rrf')
builder.add_edge('node_web_search_mcp','node_rrf')#网络搜索没有打分 是精排的 但是在rrf这里期待一个同步开始和结束 所以在rrf这里其实没有处理 其实是一个虚拟合并节点
builder.add_edge('node_rrf','node_rerank')
builder.add_edge('node_rerank','node_answer_output')
builder.add_edge('node_answer_output',END)

query_graph = builder.compile()

from pathlib import Path
import uuid
import uvicorn
from app.core.logger import logger
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from app.clients.mongo_history_utils import get_recent_history
from app.query_process.agent.state import create_default_state
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.clients.mongo_history_utils import *
from app.query_process.agent.main_graph import query_app

# 后续导入启动图对象
#from app.query_process.main_graph import query_app


# 定义fastapi对象
app = FastAPI(title="query service",description="掌柜智库查询服务！")
# 跨域问题解决
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


#6个接口：健康状态  返回页面  发起提问  sse长连接  获取历史记录  清空历史记录

@app.get("/health")
async def health():
    return {"status": "ok"}


#返回chat.html
@app.get("/chat.html")
async def chat_html():
    #查找chat.html地址
    chat_html_path = PROJECT_ROOT /'app'/ "query_process"/'page' / "chat.html"
    if not chat_html_path.exists():
        raise HTTPException(status_code=404, detail="chat.html不存在")
    return FileResponse(chat_html_path)

#检索查询接口（聊天接口)
#定义一个接受参数的类型
class QueryRequest(BaseModel):
    session_id: str = Field(None, description="会话ID")
    query: str = Field(..., description="查询内容")
    is_stream: bool = Field(False, description="是否开启SSE流式返回")

def run_graph(session_id: str, query: str, is_stream: bool):
    #调用main_graph执行
    update_task_status(session_id,"processing",is_stream)
    state = create_default_state(session_id=session_id,
                                  original_query=query,
                                  is_stream=is_stream)
    try:
        query_app.invoke(state)  #为什么用的invoke 不是 stream？
        update_task_status(session_id,"completed",is_stream)
    except Exception as e:
        logger.exception(f"session_id={session_id} 查询 {query} 失败 ")
        update_task_status(session_id,"failed",is_stream)



@app.post("/query")
async def query(request: QueryRequest, background_tasks: BackgroundTasks):
    '''
    发起查询
    :param request: 查询参数
    :param background_tasks: 背景任务 异步执行函数：is_stream=True时，开启SSE流式返回
    :return:
    '''
    query = request.query
    session_id = request.session_id or str(uuid.uuid4()) #没有的话，默认生成一个uuid
    is_stream = request.is_stream

    #判断是不是流式处理 异步-》先返回一个结果  在开始真的处理
    #异步/同步 指的是 FastAPI 接口的响应方式，异步是指接口立即返回，而同步是指接口等待处理完成后才返回。
    if is_stream:#异步执行
        #创建一个SSE队列  用于存储流式数据 {session_id,queue[update_task_state,add_running_task,add_done_task]}
        create_sse_queue(session_id)
    
        background_tasks.add_task(run_graph, session_id, query, is_stream)
        logger.info(f"异步执行查询 {query}进行流式输出")
        #返回一个结果  提示查询处理中...
        return{
            "session_id": session_id,
            "message": "查询处理中...",
                }

    else:#同步执行
        run_graph(session_id, query, is_stream)
        #获取最后一个节点插入的结果 key是answer
        answer=get_task_result(session_id, "answer")
        logger.info(f"同步执行查询 {query} 结果 {answer}")
        return {
            "answer": answer,
            "session_id": session_id,
            "message": "查询完成",
            "done_list":[]}

@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    '''
    开启SSE流式输出
    :param session_id: 会话ID
    :param request: 请求对象 前端原生请求对象 可以判断是否断开连接
    :return:
    '''
    logger.info(f"开始SSE流式输出 session_id={session_id}")
    return StreamingResponse(
        sse_generator(session_id,request),
        media_type="text/event-stream")


@app.get("/history/{session_id}")
async def history(session_id: str,limit: int = 10):
    chats=get_recent_history(session_id,limit)
    # items=[]
    # for chat in chats:
    #     items.append(chat)
    return {
        'session_id':session_id,
        'items':chats
    }
    

@app.delete("/history/{session_id}")
async def delete_history(session_id: str):
    delete_count=clear_history(session_id)
    return {
        'delete_count':delete_count,
        'message':f'{session_id}历史记录已清空'
    }

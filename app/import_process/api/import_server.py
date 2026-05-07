import os
import shutil
import uuid
from typing import List, Dict, Any
from datetime import datetime
import uvicorn
# 第三方库
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
# 项目内部工具/配置/客户端
from app.clients.minio_utils import get_minio_client
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import (
    add_running_task,
    add_done_task,
    get_done_task_list,
    get_running_task_list,
    update_task_status,
    get_task_status,
)
from app.import_process.agent.state import get_default_state
from app.import_process.agent.main_graph import kb_import_app  # LangGraph全流程编译实例
from app.core.logger import logger  # 项目统一日志工具

# 初始化FastAPI应用实例
# 标题和描述会在Swagger文档(http://ip:port/docs)中展示
app = FastAPI(
    title="File Import Service",
    description="Web service for uploading files to Knowledge Base (PDF/MD → 解析 → 切分 → 向量化 → Milvus入库)"
)

# 跨域中间件配置：解决前端调用后端接口的跨域限制
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有前端域名访问（生产环境建议指定具体域名）
    allow_credentials=True,  # 允许携带Cookie等认证信息
    allow_methods=["*"],  # 允许所有HTTP方法（GET/POST/PUT/DELETE等）
    allow_headers=["*"],  # 允许所有请求头
)

# --------------------------
# 静态页面路由：返回文件导入前端页面import.html
# 访问地址：http://localhost:8000/import.html
# --------------------------
@app.get("/import",response_class=FileResponse)
async def import_page():
    # 构建导入页面路径 从path_util.py获取项目根目录
    # 当用户访问 http://localhost:8000/import 时,后端会直接返回文件导入页面的 HTML 内容
    import_html_path = PROJECT_ROOT / "app" / "import_process"/'page'/ 'import.html'
    #然后通过该界面调用 /upload 和 /status/{taskId} 接口进行文件上传和状态查询
    if not import_html_path.exists():
        raise HTTPException(status_code=404, detail="导入页面不存在")
    return FileResponse(import_html_path,media_type="text/html")

#  --------------------------
# 核心接口：文件上传接口
# 文件上传+开启导入流程
# 支持多文件上传，核心流程：接收文件 → 本地保存 → MinIO上传 → 启动后台任务
# 访问地址：http://localhost:8000/upload （POST请求，form-data格式传参）
# --------------------------
'''
1.接受文件存储到output文件夹  /output/当天日期/uuid（task_id）/文件名
2.异步开启 import_graph 任务：01整个任务的状态：开始和结束  02每个节点的状态：开始和结束 03每个节点的结果
'''
#定义调用import_graph任务 和测试用例一样执行图
#必须有的参数：task_id  local_file_path  local_dir  （输出目录可选）
def run_import_graph(task_id: str, local_file_path: str, local_dir: str=None):
    '''
    调用import_graph任务
    :param task_id: 任务ID
    :param local_file_path: 本地文件路径
    :param local_dir: 本地输出目录（可选）
    :return: 
    '''
    try:
        '''
        对比：
        key:task_id 
        value:list[node_name] 正在进行的节点 已经完成的节点
        #_tasks_running_list:Dict[str,List[str]]={}   节点状态
        #_tasks_done_list:Dict[str,List[str]]={}
        add_running_task(task_id, node_name="upload_file")
        add_done_task(task_id, node_name="upload_file")

        #_tasks_status:Dict[str,str]={} 总状态
        key:task_id 
        value:str task_id的任务状态
        
        update_task_status(task_id, 'processing' )
        update_task_status(task_id, 'completed' )
        '''
        update_task_status(task_id, 'processing' )
        # 调用图执行器  并监听每个节点的状态
        init_state = get_default_state()  # 获取默认状态
        init_state["task_id"] = task_id  # 设置任务ID
        init_state["local_file_path"] = local_file_path  # 设置本地文件路径
        init_state["local_dir"] = local_dir  # 设置本地输出目录（可选）
        
        for event in kb_import_app.stream(init_state):
            for node_name,result in event.items():
                logger.info(f'节点{node_name}已经完成执行，执行结果为{result}')
                
        update_task_status(task_id, 'completed' )
        logger.info(f'{task_id}图执行完成！')

    except Exception as e:
        logger.exception('====图执行失败！发生异常====')
        update_task_status(task_id, 'failed' )

#前端上传的是文件列表 files  每个文件都有一个文件名 ...表示一定要上传文件
@app.post("/upload")
async def upload_file(files: List[UploadFile] = File(...), 
                      background_tasks: BackgroundTasks = BackgroundTasks):

    #1.整理输出位置
    today_str=datetime.now().strftime('%Y%m%d')
    base_output_path=PROJECT_ROOT / "output" / today_str
    #2.记录每个文件上传任务的id
    #存储taskid的列表
    task_ids=[]

    #3.循环处理每个上传的文件：存储到本地+异步图任务调用
    for file in files:
        #每个file 都是 UploadFile 对象，包含文件名、内容等信息
        #.file 上传文件的输入流  .filename 文件名  .content_type 文件类型   .read() 读取文件内容
        task_id=str(uuid.uuid4())
        task_ids.append(task_id)
        #记录下进行文件上传了
        add_running_task(task_id, node_name="upload_file")
        #文件的dir_path 在原来日期后面加一个uuid（task_id）
        dir_path=base_output_path / task_id
        #创建文件夹
        dir_path.mkdir(parents=True, exist_ok=True)
        #文件的local_file_path/output/当天日期/uuid（task_id）/文件名
        local_file_path=dir_path / file.filename
        #上传的文件写入到local_file_path
        with open(local_file_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)
            # 分块复制，内存友好，避免一次读取大文件导致内存不足
        
        # 异步开启图任务：将 run_import_graph 函数注册为后台任务 立即返回响应给前端 ，不等待任务完成 
        # 在响应发送后自动执行 run_import_graph 函数
        # 参1：func 要执行的函数
        # 参2：*args 要传递给 func 的参数 位置参数
        # 参3：**kwargs 要传递给 func 的参数 关键字参数
        background_tasks.add_task(run_import_graph, 
                                  task_id, 
                                  str(local_file_path), 
                                  str(dir_path))
        logger.info(f'异步开启图任务：{task_id}')
        add_done_task(task_id, node_name="upload_file")

    #4.最终返回结果
    return {
        "code": 200,
        "message": f"完成文件上传，上传文件数: {len(files)}",
        "task_ids": task_ids
    }


# --------------------------
# 核心接口：任务状态查询接口
# 前端轮询此接口获取单个任务的处理进度和状态
# 访问地址：http://localhost:8000/status/{task_id} （GET请求）
# --------------------------
@app.get("/status/{task_id}", summary="任务状态查询", description="根据TaskID查询单个文件的处理进度和全局状态")
async def get_task_progress(task_id: str):
    """
    任务状态查询接口
    前端轮询此接口（如每秒1次），获取任务的实时处理进度
    返回数据均来自内存中的任务管理字典（task_utils.py），高性能无IO

    :param task_id: 全局唯一任务ID（由/upload接口返回）
    :return: 包含任务全局状态、已完成节点、运行中节点的JSON响应
    """
    # 构造任务状态返回体
    task_status_info: Dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),  # 任务全局状态：pending/processing/completed/failed
        "done_list": get_done_task_list(task_id),  # 已完成的节点/阶段列表
        "running_list": get_running_task_list(task_id)  # 正在运行的节点/阶段列表
    }
    # 记录状态查询日志，方便追踪前端轮询情况
    logger.info(
        f"[{task_id}] 任务状态查询，当前状态：{task_status_info['status']}，已完成节点：{task_status_info['done_list']}")
    return task_status_info

# --------------------------
# 服务启动入口
# 直接运行此脚本即可启动FastAPI服务，无需额外执行uvicorn命令
# --------------------------
if __name__ == "__main__":
    """服务启动入口：本地开发环境直接运行"""
    logger.info("File Import Service 服务启动中...")
    # 启动uvicorn服务，绑定本地IP和8000端口，关闭自动重载（生产环境建议用workers多进程）
    uvicorn.run(
        app=app,
        host="127.0.0.1",  # 仅本地访问，生产环境改为0.0.0.0（允许所有IP访问）
        port=8000  # 服务端口
    )


    
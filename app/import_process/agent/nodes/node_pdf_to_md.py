import os
import sys
from pathlib import Path
import time
from typing import Tuple
import zipfile

import requests  # 推荐添加，用于类型提示
# 项目内部库
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.format_utils import format_state
from app.utils.task_utils import add_running_task, add_done_task
from app.utils.path_util import PROJECT_ROOT    # 项目根目录
from app.conf.mineru_config import mineru_config
from app.core.logger import logger  # 统一日志工具

'''
 node_pdf_to_md
    参数：state [is_pdf_read_enable=True|pdf_path, local_dir]
    返回：state [md_path=地址|md_content=内容]
    1.日志和任务状态
    2.校验路径是否存在
    3.上传pdf至MinerU并轮询解析任务状态
    4.下载解析后的zip压缩包并解压到指定目录
    5.日志和任务状态更新   return state
 step1:validate_state(state) 校验路径是否存在
    参数：state[pdf_path | loacl_dir=output]
    返回：pdf_path_obj, local_dir_obj  
    1.非空校验
    2.文件校验 一个异常 一个默认地址
    3.返回path对象
 step2:upload(pdf_path_obj) 上传pdf至MinerU并轮询解析任务状态
    参数：pdf_path_obj
    返回：zip_url
    1.申请上传解析的地址
    2.上传文件 session|千万不能是session.put  而是requests.put   代理影响
    3.轮询解析任务状态 直到解析完成/失败/超时 
    4.返回解析后的zip压缩包下载地址
 step3:download(zip_url, local_dir_obj) 下载解析后的zip压缩包并解压到指定目录
 优化方向：加uuid防止覆盖
    参数：zip_url, local_dir_obj 原文件名file_name
    返回：解压后的.md文件路径
    1.下载zip压缩包 output/file_name_result.zip
    2.检查解压的文件地址 output/file_name(文件目录)
    3.检查文件夹防止重复（处理），解压，返回解压后的.md文件路径 extractall
    4.优先级考虑.md文件 
    5.重命名 
    6.路径转成字符串 获取绝对路径返回 
    '''


def validate_state(state: ImportGraphState) -> Tuple[Path, Path]:
    '''
    路径校验方法  校验PDF路径和输出目录  返回校验完毕后可以直接使用的路径对象
    校验pdf_path 是否存在 失效则抛出异常（输入文件地址）
    校验local_dir 是否存在 失效则创建目录（输出文件地址）
    '''
    logger.debug(f">>> 在md转pdf时的路径校验")
    #常规非空校验
    pdf_path = state["pdf_path"]
    local_dir = state["local_dir"]
    if not pdf_path:
        logger.error(f">>> 校验发现pdf_path为空")
        raise ValueError("pdf_path 不能为空")
    if not local_dir:
        #创建一个默认输出目录
        local_dir=PROJECT_ROOT / "output"
        logger.info(f">>> 校验发现local_dir为空，创建默认输出目录: {local_dir}")
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)
    if not pdf_path_obj.exists():
        logger.error(f">>> 校验发现pdf_path不存在: {pdf_path}")#路径确实存在 但是文件不存在
        raise ValueError("pdf_path 路径不存在")
    if not local_dir_obj.exists():
        local_dir_obj.mkdir(parents=True, exist_ok=True)
        logger.info(f">>> 校验发现local_dir不存在，创建目录: {local_dir_obj}")
    return pdf_path_obj, local_dir_obj

def upload(pdf_path_obj: Path) -> str:
    """
    步骤2：上传PDF至MinerU并轮询解析任务状态
    核心流程：配置校验 → 获取上传链接 → 文件上传（含重试） → 任务轮询（直至完成/失败/超时）
    :param pdf_path_obj: 上传解析 PDF 文件路径对象
    :return: 解析后的zip压缩包下载地址
    异常：ValueError(配置缺失)、RuntimeError(请求/上传失败)、TimeoutError(任务超时)
    """ 
    #1.申请上传解析的地址
    #前置的准备：url api |token|固定格式的请求头（mineru官网实例）
    token = mineru_config.api_key
    url = f"{mineru_config.base_url}/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {"name":f"{pdf_path_obj.name}"}
        ],
        "model_version":"vlm",  
    }
    response = requests.post(url,headers=header,json=data)
    #结果处理
    if response.status_code != 200 or response.json()['code']!=0:
        logger.error(f">>> MinerU上传失败！错误信息: {response.json()}")
        raise RuntimeError("MinerU上传失败！")
    #获取 url
    upload_url = response.json()['data']['file_urls'][0]
    batch_id = response.json()['data']['batch_id']
    
    #2.上传文件
    #不能直接加put请求 电脑开了代理 put的请求头会添加额外的参数头 将文件真的转存到第三方的文件存储服务器 但是检查严格
    #get post 宽进宽出  put 严进严出
    http_session=requests.Session()
    http_session.trust_env=False  #禁止使用代理
    try:
        with open (pdf_path_obj,"rb") as f:
            file_data=f.read()
        upload_response=http_session.put(upload_url,data=file_data)
        if upload_response.status_code != 200 :
            logger.error(f">>> 上传文件失败！错误信息: {upload_response.json()}")
            raise RuntimeError("上传文件失败！")
    except Exception as e:
        logger.error(f">>> 上传文件失败！错误信息: {e}")
        raise RuntimeError("上传文件失败！")
    finally:
        http_session.close()

    #3.轮询获取结果
    #确保获取到结果才行 循环获取  3/5s获取一次 最多等待10分钟-官方1s 1页pdf
    url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
    start_time = time.time()
    timeout_seconds = 600  # 最大超时时间10分钟（适配600页内PDF）
    poll_interval = 3      # 轮询间隔3秒（平衡查询频率和服务端压力）
    logger.info(f"开始任务轮询，batch_id：{batch_id}，最大超时：{timeout_seconds}s")
    while True:
        #1.超时检查：超过最大时间直接终止轮询
        if time.time() - start_time > timeout_seconds:
            logger.error(f">>> 任务轮询超时！最大超时时间：{timeout_seconds}s")
            raise TimeoutError("任务轮询超时！")
        
        #2.发起轮询请求
        res = requests.get(url, headers=header)
        #3.解析轮询结果 和 获取zip下载地址
        if res.status_code != 200 :  
            #是否再给一次机会？http 1xx 2xx 3xx 4xx  5xx
            if 500 <= res.status_code < 600:
                time.sleep(poll_interval)
                continue
            raise RuntimeError(f"请求MinerU解析接口失败，返回的状态码为{res.status_code}！")
        json_data = res.json()
        if json_data['code'] != 0:
            #大部分都没有给机会 直接打死
            raise RuntimeError(f"MinerU解析接口返回错误信息: {json_data['msg']},错误码为{json_data['code']}")
        #判断解析状态 是否结束
        extract_status = json_data['data']['extract_result'][0] #官方文档有问题，实际是个列表！！
        if extract_status['state']=='done':  
            full_url = extract_status['full_zip_url']
            logger.info(f"pdf解析任务完成，下载zip文件地址：{full_url}，耗时：{time.time() - start_time}秒")
            return full_url
        else:
            time.sleep(poll_interval)

def download_zip(zip_url: str, local_dir_obj: Path, file_name: str) -> str:
    """
    步骤3：下载MinerU解析结果ZIP包并解压，返回解压后的md文件路径
    核心流程：下载ZIP → 解压ZIP → 查找MD文件（按优先级） → 重命名统一为PDF同名
    :param zip_url: MinerU解析结果ZIP包下载地址
    :param local_dir_obj: 解压后的文件保存目录
    :param file_name: 解压后的文件名
    异常：RuntimeError(下载失败)、FileNotFoundError(无MD文件)
    """
    #1.下载ZIP response响应体
    response = requests.get(zip_url)
    if response.status_code != 200 :
        logger.error(f">>> 下载zip文件失败！错误信息: {response.json()}")
        raise RuntimeError("下载zip文件失败！")
    #2.将响应体的zip文件保存到本地 output目录下
    zip_save_path=local_dir_obj / f"{file_name}_result.zip"
    with open(zip_save_path, "wb") as f:
        f.write(response.content)
    logger.info(f">>> 下载zip文件成功，保存路径：{zip_save_path}")
    #3.清空旧目录 图片多可能不会覆盖 但是zip文件肯定会覆盖
    extract_target_dir=local_dir_obj / file_name  #在一个新建的目录下解压
    if extract_target_dir.exists():
        extract_target_dir.rmdir()
    extract_target_dir.mkdir(parents=True, exist_ok=True)
    
    #4.解压ZIP文件到指定目录 内置zipfile模块
    with zipfile.ZipFile(zip_save_path, 'r') as zip_ref:
        zip_ref.extractall(extract_target_dir)
    
    #5.返回解压后的md文件路径
    #解压后的文件名：文件.md full.md （图片的文件夹）
    md_file_list=list(extract_target_dir.glob("*.md"))
    target_md_file=None  #优先级最高的md文件
    if len(md_file_list) == 0:
        raise FileNotFoundError(f"未找到{file_name}.md或full.md文件！")
    
    #有没有文件.md 优先级最高  然后是full.md 考虑大小写问题 而且md_file_list是一个path对象的列表不是字符串
    for md_file in md_file_list:
        if md_file.name==file_name+".md":
            target_md_file=md_file
            break
        elif md_file.name.lower()=="full.md":
            target_md_file=md_file
            break
        else: #实在没有取第一个
            target_md_file=md_file_list[0]
            logger.info(f"未找到{file_name}.md或full.md文件！")
    #统一修改文件名 file_name.md  
    if target_md_file.name!=file_name+".md":
        #rename() 返回新的路径，需要接收它
        target_md_file=target_md_file.rename(target_md_file.with_name(f"{file_name}.md"))
    #最终的md绝对文件路径 并且 转字符串类型
    md_path=str(target_md_file.resolve())
    logger.info(f">>> 解压后的md文件路径：{md_path}")
    return md_path
    
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:

    """
    节点: PDF转Markdown (node_pdf_to_md)
    核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    0. 获取 PDF 路径 和 输出目录 
        进行参数校验 local_dir-》没有则给默认值 pdf_path -》深入校验文件是否真的存在
    1. 调用 MinerU (magic-pdf) 工具。（可以更换其他工具）  给（local_file_path）
    2. 上传文件，将PDF 文件put到签名url, 将 PDF 转换成 Markdown 格式。
    3. 轮巡任务直到任务完成
    4. 下载生成的zip文件，解压文件，将结果保存到 state["md_content"]。

    注意：
    调用第三方API 要考虑容错  加一个异常处理
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] 开始执行！现在的状态是: {state}")
    #add running task 在内存中记录当前正在运行的任务节点 用于前端的sse推送
    add_running_task(state['task_id'],function_name)

    try:
        #校验 返回校验完毕后可以直接使用的路径对象 方法外置  得到的是path对象
        pdf_path_obj,local_dir_obj = validate_state(state)
        #调用 MinerU 工具进行转换 方法外置  得到字符串类型的url路径
        zip_url=upload(pdf_path_obj)    
        #下载zip文件，解压文件 参1 zip的路径 参2解压的文件夹 参3文件名 返回后的md文件的真实路径(字符串)
        md_path=download_zip(zip_url,local_dir_obj,pdf_path_obj.stem)

        state["md_path"] = md_path
        state["local_dir"] = str(local_dir_obj)
        #state["md_content"] = md_path.read_text(encoding="utf-8")
        with open(md_path, "r", encoding="utf-8") as f:
            state["md_content"] = f.read()
    except Exception as e:
        logger.error(f">>> [{function_name}] 使用MinerU工具转换PDF为Markdown失败！错误信息: {e}")
        raise 
    finally:
        logger.info(f">>> [{function_name}] 执行完毕！\n现在的状态是: {format_state(state)}")
        add_done_task(state['task_id'],function_name)
    return state

#测试
if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")
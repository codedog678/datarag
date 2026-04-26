import os
import re
import sys
import base64
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque


# MinIO相关依赖
from minio import Minio
from minio.deleteobjects import DeleteObject

# 【核心改造1：移除原生OpenAI，导入LangChain工具类和多模态消息模块】
from app.clients.minio_utils import get_minio_client
from app.import_process.agent.state import ImportGraphState
from app.utils.format_utils import format_state
from app.utils.task_utils import add_done_task, add_running_task
# LLM客户端工具类（核心复用，替换原生OpenAI调用）
from app.lm.lm_utils import get_llm_client
# LangChain多模态依赖（消息构造+异常捕获）
from langchain.messages import HumanMessage
from langchain_core.exceptions import LangChainException
# 项目配置
from app.conf.minio_config import minio_config
from app.conf.lm_config import lm_config
# 项目日志工具（统一使用）
from app.core.logger import logger
# api访问限速工具
from app.utils.rate_limit_utils import apply_api_rate_limit
# 提示词加载工具
from app.core.load_prompt import load_prompt

# MinIO支持的图片格式集合（小写后缀，统一匹配标准）
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")
'''
    主要目标：将md文件中的图片进行单独处理，方便模型理解图片的含义
    主要动作：图片-》文件服务器-》网络图片地址  
            (上文100)图片（下文100）-》视觉模型-》图片描述
            【图片总结】（网络图片地址）-》state-》md_content=新的内容（包含图片描述）|md_path=新的md文件路径（包含图片描述）
    总结技术：minio 视觉模型
    总结步骤：
    1.state校验获取图片路径
    参数：state  md_path md_content
    返回：a.校验后的md_content b.md路径对象?  c.获取的图片文件夹 images
    2.识别md中使用过的图片 进行下一步（不是所有都要处理）
    参数：images  md_content
    返回：[（图片名，图片地址，（上文，下文））]  [之后处理：key=图片名,value=图片总结]
    3.图片内容总结和处理（多模态模型）
    参数：[（图片名，图片地址，（上文，下文））]   |  md文件的名称
    返回：{图片名:总结....}
    4.上传图片到minio更新md
    参数：minio_clien  {图片名:总结....}   [（图片名，图片地址，（上文，下文））]-minio用  md_content  
    返回：new_md_content 
    state['md_content'] = new_md_content
    5.数据的最终处理和备份
    参数：new_md_content  md_path(旧的）-》xx.md  ->new_xx.md
    返回：新的md文件路径
    state['md_path'] = new_md_path
    return state
'''
# 内置的方法 在步骤里
def is_supported_image(filename: str) -> bool:
    """
    检查文件名是否为支持的图片格式。
    :param filename: 文件名（包含路径）包含后缀
    :return: 是否为支持的图片格式
    """
    return filename.lower().endswith(IMAGE_EXTENSIONS)

#外置的方法 在步骤里
def get_image_context(md_content: str, filename: str,context_len: int=100) -> Tuple[str,str]:
    """
    从Markdown内容中提取图片的上下文（100）。
    :param md_content: Markdown内容字符串
    :param filename: 图片文件名（包含路径）
    :return: 包含图片上下文的元组（上文，下文）
    """
    # ![图片描述](图片地址) 正则提取 eg. ![二大爷](https://example.com/name.jpg)一个图片多个上下文不一样（出现地方不一样
    pattern=re.compile(r"!\[.*?\]\(.*?"+filename+".*?\)")
    results=[] #虽然一个图片可以用在多个地方  但是mineru扫出的图片用uuid命名 即使长得一样也不会重复

    for match in pattern.finditer(md_content):#迭代器
        start,end=match.span()
        #span()返回匹配的字符串的起始索引和结束索引 
        # [start]![二大爷](https://example.com/name.jpg)[end]
        pre_context=md_content[max(0,start-context_len):start]
        post_context=md_content[end:min(len(md_content),end+context_len)]
        results.append((pre_context,post_context))
    if results:
       return results[0]
    else:
       return '',''

#1.校验获取图片路径
def get_content(state: ImportGraphState) -> Tuple[str,Path,str]:
    """
    从state中获取md_content md_path images_dir
    :param state: 包含md_path和_content的ImportGraphState
    :return: 校验后的md_content md_path对象 images_dir对象
    """
    md_file_path = state['md_path']#state里是字符串类型 后面要转成path对象
    if not md_file_path:#变量层面是否有值
        raise ValueError("md_path不能为空")
    md_path_obj = Path(md_file_path)
    if not md_path_obj.exists():#实际层面，是否真的有文件
        raise ValueError(f"md文件不存在: {md_file_path}")

    #检查md_content是否存在,因为可能是md直接来的
    if not state['md_content']:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            state['md_content'] = f.read()
    
    #图片文件夹obj
    #注意：自己传入的md文件也得是images文件夹（待优化）
    images_dir_obj = md_path_obj.parent / 'images'
    return state['md_content'],md_path_obj,images_dir_obj

#2.识别md中使用过的图片 进行下一步（不是所有都要处理）
def scan_images(md_content: str, images_dir_obj: Path) -> List[Tuple[str,str,Tuple[str,str]]]:
    """
    扫描Markdown内容中的图片链接，并且截取图片的上下文（100）。
    :param md_content: Markdown内容字符串
    :param images_dir_obj: 图片文件夹路径对象
    :return: 包含图片名、图片地址、（上文，下文）的元组列表
    """
    #1.创建一个目标集合
    targets=[]
    #2.循环读取images的所有图片 校验是否在md里使用 使用就截取上下文
    for filename in os.listdir(images_dir_obj):
        #检查图片是否可以用
        if not is_supported_image(filename):
            continue
        #读取上下文  外置方法
        context=get_image_context(md_content,filename)
        if not context:
            logger.warning(f"图片{filename}未在md中使用，跳过")
            continue
        targets.append((filename,str(images_dir_obj / filename),context))
    return targets

#3.图片内容总结和处理（多模态模型）
def get_img_summary(targets: List[Tuple[str,str,Tuple[str,str]]],stem:str) -> Dict[str,str]:
    """
    调用多模态模型生成图片描述。
    :param targets:[（图片名，图片地址，（上文，下文））] 
    :param stem: md文件的名称
    :return: 包含图片名、图片描述的字典{图片名.xx:图片描述,图片名.xx:图片描述...}
    """
    #循环每张图片 调用多模态模型生成图片描述
    summaries={}
    request_times=deque() #访问限速队列
    for filename,image_url,content in targets:
        #01.访问限速问题 
        apply_api_rate_limit(request_times,max_requests=9,window_seconds=60)
        #02.像视觉模型发起请求
        #02.1模型对象
        vm_model=get_llm_client(lm_config.lv_model)
        #02.2准备提示词
        prompt=load_prompt("image_summary",root_folder=stem,image_content=content)
        
        with open(image_url, 'rb') as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')  # 图片数据base64编码 b64decode解码
        
        messages = [
            HumanMessage(
                content=[
                    # 文本提示词：携带上下文，限定摘要规则
                    {
                        "type": "text",
                        "text": prompt
                    },
                    # 多模态核心：Base64编码图片数据
                    {
                        "type": "image_url",
                        "image_url": {
                            #正常放网络地址 但是本地图片需要base64编码
                            #jpg->image/jpeg
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            )
        ]
        #02.3调用模型接口 得到图片描述
        response=vm_model.invoke(messages)
        summary=response.content.strip().replace("\n","")
        summaries[filename]=summary
    logger.info(f"图片描述生成完成")
    return summaries

#4.上传图片到minio更新md
def upload_and_replace_images(md_content: str,stem:str,summaries:Dict[str,str],targets: List[Tuple[str,str,Tuple[str,str]]]) -> str:
    """
    上传图片到MinIO，替换Markdown中的图片链接。
    :param md_content: 原md内容
    :param stem: md文件的名称 （用于生成minio的文件夹名）
    :param summaries: 包含图片名、图片描述的字典{图片名.xx:图片描述,图片名.xx:图片描述...}
    :param targets: 包含图片名、图片地址、（上文，下文）的元组列表
    :return: 更新后的md内容
    步骤：
    理解minio中的存储结果：桶/upload-images/文件夹名字/图片名.后缀
    1.删除minio中对应的文件图片
    2.上传图片到minio
    3.合并图片摘要和URL
    4.替换MD内容中的本地图片引用为MinIO URL

    """
    #1.删除minio中对应的文件图片
    #1.1获取要删除的对象
    minio_client=get_minio_client()
    object_list=minio_client.list_objects(minio_config.bucket_name,
                              prefix=f"{minio_config.minio_img_dir}/{stem}/",
                              recursive=True)
    #1.2调用方法进行删除
    #需要DeleteObjects参数批量删除 但现在是还不是这个对象
    delect_object_list=[DeleteObject(obj.object_name) for obj in object_list]
    minio_client.remove_objects(minio_config.bucket_name,delect_object_list)
    #另一种写法：
    #for obj in object_list: 参数类型是string 接受单个对象名字字符串进行删除
    #    minio_client.remove_object(minio_config.bucket_name,obj.object_name)
    logger.info(f"删除{stem}中对应的文件图片，本次删除了{len(delect_object_list)}个文件")
    #2.上传图片到minio
    #记录图片上传结果的字典{图片名.xx:图片url,图片名.xx:图片url...}
    images_url={}
    for filename,image_url,_ in targets:
        try:
            minio_client.fput_object(
                bucket_name= minio_config.bucket_name,
                object_name= f"{minio_config.minio_img_dir}/{stem}/{filename}",
                file_path= image_url,
                content_type= "image/jpeg" )
            #记录图片url:协议+端点+桶名+对象名
            images_url[filename]=f"{minio_config.minio_secure}://{minio_config.endpoint}/{minio_config.bucket_name}/{minio_config.minio_img_dir}/{stem}/{filename}"
            logger.info(f"上传图片{filename}到minio成功，url为：{images_url[filename]}")
        except Exception as e:
            logger.error(f"上传图片{filename}到minio失败：{e}")
            continue
    #3.合并图片摘要和URL
    #summaries: 图片名：描述
    #images_url: 图片名：url
    #汇总： {图片名.xx:图片描述+图片url,图片名.xx:图片描述+图片url...}
    image_infos={}
    for filename,summary in summaries.items():
        if url:=images_url.get(filename):
            image_infos[filename]=(summary,url)
    logger.info(f"图片描述和URL汇总完成：{image_infos}")

    #4.替换MD内容中的本地图片引用为MinIO URL
    # ![xx](图片地址/filename) -》 ![summary](minio图片url) 正则替换
    for filename,(summary,url) in image_infos.items():
        pattern=re.compile(r"!\[.*?\]\("+filename+".*?\)")
        md_content=pattern.sub(f"![{summary}]({url})",md_content)
    logger.info(f"替换MD内容中的本地图片引用为MinIO URL完成")
    return md_content
        
#5.数据的最终处理和备份
def replace_md_file(md_path_obj: Path,new_md_content: str) -> str:
    """
    替换Markdown文件中的图片链接。
    :param md_path_obj: 原始文件路径
    :param new_md_content: 更新后的Markdown内容
    :return: 新的文件路径
    """
    #c:/xx/xxx/xxx/hak180产品安全手册.md->splitext[0]:c:/xx/xxx/xxx/hak180产品安全手册
    #c:/xx/xxx/xxx/hak180产品安全手册_new.md
    new_md_path=os.path.splitext(md_path_obj)[0]+"_new.md"
    with open(new_md_path, 'w', encoding='utf-8') as f:
        f.write(new_md_content)
    logger.info(f"新内容写入已完成，新文件路径：{new_md_path}")
    return new_md_path

def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    未来要实现:
    1. 扫描 Markdown 中的图片链接。
    2. 将图片上传到 MinIO 对象存储。
    3. (可选) 调用多模态模型生成图片描述。
    4. 替换 Markdown 中的图片链接为 MinIO URL。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] 开始执行！现在的状态是: {state}")
    #add running task 在内存中记录当前正在运行的任务节点 用于前端的sse推送
    add_running_task(state['task_id'],function_name)

    md_content,md_path_obj,images_dir_obj=get_content(state)
    #如果没有images 直接返回state
    #如果存在images 则采取下一步进行处理
    if not os.path.exists(images_dir_obj):
        logger.info(f"[{function_name}] 图片文件夹不存在，直接返回状态")
        return state
    targets=scan_images(md_content,images_dir_obj)
    summaries=get_img_summary(targets,md_path_obj.stem)
    #上传图片到minio
    new_md_content=upload_and_replace_images(md_content,md_path_obj.stem,summaries,targets)
    #新的md文件保存替换策略
    new_md_path=replace_md_file(md_path_obj,new_md_content)
    #更新地址和内容
    state['md_path']=new_md_path
    state['md_content']=new_md_content
    logger.info(f">>> [{function_name}] 执行完毕！现在的状态是:{format_state(state)}")
    add_done_task(state['task_id'],function_name) 
    return state

if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
      
        #logger.info(f"本地测试完成 ")
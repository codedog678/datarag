import re
import json
import os
import sys
# 统一类型注解，避免混用any/Any
from typing import List, Dict, Any, Tuple
# LangChain文本分割器（标注核心用途，便于理解）
from langchain_text_splitters import RecursiveCharacterTextSplitter
#langchain的文本切割器！！！

# 项目内部工具/状态/日志导入（保持原有路径）
from app.utils.task_utils import add_done_task, add_running_task
from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger  # 项目统一日志工具，核心替换print

# --- 配置参数 (Configuration) ---
# 单个Chunk最大字符长度：超过则触发二次切分（适配大模型上下文窗口）
DEFAULT_MAX_CONTENT_LENGTH = 2000
# 短Chunk合并阈值：同父标题的短Chunk会被合并，减少碎片化
MIN_CONTENT_LENGTH = 500

#1.参数校验
def get_content(state: ImportGraphState)->Tuple[str,str]:
    md_content = state['md_content']
    if not md_content:
        logger.error("没有有效的md内容")
        raise Exception("没有有效的md内容")
    '''
    windows \r\n
    linux \n
    mac \r
    '''
    md_content = md_content.replace('\r\n', '\n').replace('\r', '\n') #
    file_title = state['file_title']
    return md_content,file_title

#2. 基于 Markdown 标题层级进行递归切分。-》粗粒度切割 语义完善
def splite_by_title(md_content:str,file_title:str)->Tuple[List[str],int,int]:
    """
    基于 Markdown 标题层级进行递归切分。
    输出-》粗粒度切割 语义完善
    :param md_content: 输入的Markdown内容
    :param file_title: 文件标题
    :return: 切分后的章节列表、识别到的有效标题数量、MD原始文本总行数（为后续统计/日志使用）
    [{content,title,line_count}]
    """
    #1.准备前置的工作
    #1.1正则
    '''
    正则匹配Markdown 1-6级标题（核心规则，适配缩进/标准格式）
    ^\s*：行首允许0/多个空格/Tab（兼容缩进的标题）
    #{1,6}：匹配1-6个#（对应MD1-6级标题）
    \s+：#后必须有至少1个空格（区分#是标题还是普通文本）
    .+：标题文字至少1个字符（避免空标题）
    '''
    title_pattern = r'^\s*#{1,6}\s+.+'
    
    #1.2md_content 切割\n
    lines = md_content.split("\n")

    #1.3定义临时存储变量 每次按照标题进行切分 都需要更新
    #current_title=str|current_lines=[] 临时存储|title_count=0 存储了多少块 如果是代码块也得考虑
    current_title = ""  # 当前章节标题
    current_lines = []  # 当前章节的行缓存
    title_count = 0  # 有效标题数量（非代码块内）
    is_code_block = False  # 代码块标记：避免误判代码块内的#为标题
    #1.4 最终存储的列表
    sections = []  # 切分后的章节列表

    #2.循环每行的列表
    for i,line in enumerate(lines):
        #可能有空格去一下空格
         line = line.strip()
    #2.1 判断代码块状态
         if line.startswith("```")or line.startswith("~~~"):
            #进入或退出  看in_code_block  第一次一定进入代码块
            is_code_block = not is_code_block
            current_lines.append(line)
            continue
      #在代码块里的除了''' 行 其他正常收集
    #2.2判断是不是标题 得排除代码块的影响 
      # 是不是第一次的标题 不是第一次就要存储上一次的章节内容
         is_title =(not is_code_block) and re.match(title_pattern, line)
            #2.3是标题怎么处理
         if is_title:
             if current_title:
                 sections.append(
                     {"content":"\n".join(current_lines),
                      "title":current_title,
                      "line_count":len(current_lines),
                      "file_title":file_title}
                 )
             current_title=line
             current_lines=[current_title] # 新章节从标题开始 不是append 应该是覆盖 标题是段落的第一个内容
             title_count+=1
             #2.4不是标题怎么处理
         else:
             current_lines.append(line)
    #2.5最后一个章节的处理
    if current_title:
        sections.append(
            {"content": "\n".join(current_lines),
             "title":current_title,
             "line_count":len(current_lines),
             "file_title":file_title}
        )
    #3.返回结果
    logger.info(f"完成chunks的粗切，识别到的有效标题数量：{title_count}")
    return sections,title_count,len(lines)

#大的切
def split_long_section(section:Dict[str,str],max_content_length:int)->List[Dict[str,str]]:
    #二次切割 返回切割后的列表
    content=section['content']
    if len(content)<=max_content_length:
        return [section]
    #切割
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=max_content_length,
        chunk_overlap=100,
        separator=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
        #优先级：空行(段落)→换行→中文标点→英文标点→空格，最后硬拆 所以有的切割符压根用不上
    )
    #切割
    #title=标题_1 _2... | part 1 2 ...|parent_title=section.title
    sub_sections=[]
    chunks=splitter.split_text(content)
    for idx,chunk in enumerate(chunks,start=1):
        text=chunk.strip()
        title=f"{section['title']}_{idx}"
        parent_title=section['title']
        part=idx
        file_title=section['file_title']
        sub_sections.append({
            "content":text,
            "title":title,
            "parent_title":parent_title,
            "part":part,
            "file_title":file_title,
        })

    return sub_sections

#小的合
def merge_short_sections(sections:List[Dict[str,str]],min_content_length:int)->List[Dict[str,str]]:
    """
    合并过短的段落 语义完整的chunks
    1.content长度小于min_content_length的段落 才进行合并
    2.同一个parent_title的段落 才进行合并(所以切割的时候才要进行保留)
    :param sections: 切分后的章节列表
    :param min_content_length: 最小内容长度（避免过短的Chunk）
    :return: 合并后的章节列表
    """
    #1.准备前置的工作
    merged_sections=[]#合并后的章节列表
    pre_section=None#当前章节

    #2.循环每个section
    for section in sections:
         if pre_section is None:
            pre_section=section
            continue
         #current_section 是第一次（上一块） section是本次（本快）
         is_current_short=len(pre_section['content'])<min_content_length
         is_same_parent=pre_section['parent_title']==section['parent_title']
         #合并条件

         if is_current_short and is_same_parent:#即是短块 又和 本次是同一个父标题
             #合并
             parent_title=pre_section['parent_title']
             next_content=section['content']
             pre_section['content']=pre_section['content']+"\n\n"+next_content
             pre_section['part']=section['part']
             
         else:
             #上一次不是短块  或者  这一次和上一次不是同一块
             merged_sections.append(pre_section)
             pre_section=section#更新 开启下一次循环
    #处理最后一个结果 所以在循环外
    if pre_section is not None:
        merged_sections.append(pre_section)
    return merged_sections


#3. 对过长的段落进行二次切分。-》细粒度切割 大小和重叠合适 大的往小切 小的合并
def refine_chunks(sections:List[Dict[str,str]],max_content_length:int=DEFAULT_MAX_CONTENT_LENGTH,min_content_length:int=MIN_CONTENT_LENGTH)->List[Dict[str,str]]:
    """
    对过长的段落进行二次切分。
    1.超过max_content_length的段落 才进行切分
    2.小于min_content_length的段落 才进行合并
    :param sections: 切分后的章节列表
    :param max_content_length: 最大内容长度（避免过短的Chunk）
    :param min_content_length: 最小内容长度（避免过短的Chunk）
    :return: 切分后的章节列表
    """
    #1.准备前置的工作
    final_sections=[]#处理后的章节列表

    #超过的切
    for section in sections:
        #每个section [content,title,file_title]
        sub_sections=split_long_section(section,max_content_length)
        #切出来的小块好之前是同一个级别 所以直接添加 没改直接加 改了多个一块加
        final_sections.extend(sub_sections)
    #补全part和parent_title
    for section in final_sections:
        section['part'] = section.get('part') or 1
        section['parent_title'] = section.get('parent_title') or section.get('title', '')
    # 小于的合
    final_sections = merge_short_sections(final_sections, min_content_length)
   
    # 返回
    return final_sections

#4. 数据的备份和chunks属性的更改(chunks->state | chunks->本地备份)
def save_chunks_to_local(state: ImportGraphState,chunks:List[Dict[str,str]]):
    """
    备份chunks到本地文件
    :param state: 图状态
    :param chunks: 切分后的章节列表[{}]  其实就是sections
    """
    local_dir=state['local_dir']
    backup_file_path=os.path.join(local_dir,"chunks.json")
    with open(backup_file_path,"w",encoding="utf-8") as f:
        json.dump(chunks,f,ensure_ascii=False,indent=4)
    logger.info(f"备份chunks到本地文件：{backup_file_path}")
    
    
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split) 将长文档切分成小的 Chunks (切片) 以便检索。
    0. 参数校验
    1. 基于 Markdown 标题层级进行递归切分。-》粗粒度切割 语义完善
       特殊场景：没有标题 
    2. 对过长的段落进行二次切分。-》细粒度切割 大小和重叠合适 大的往小切 小的合并
       大小合适 语义完整的chunks
    3. 数据的备份和chunks属性的更改。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] 开始执行！现在的状态是: {state}")
    #add running task 在内存中记录当前正在运行的任务节点 用于前端的sse推送
    add_running_task(state['task_id'],function_name)
    try:
        #1.参数校验
        md_content,file_title=get_content(state)
        #2. 基于 Markdown 标题层级进行递归切分。-》粗粒度切割 语义完善
        #输出：初切后的章节列表、识别到的有效标题数量、MD原始文本总行数（为后续统计/日志使用）
        sections,title_count,lines_count=splite_by_title(md_content,file_title)
        #特殊场景 兜底 没有标题怎么办
        if title_count==0:
            sections=[{
                "content":md_content,
                "title":"没有标题",
                "file_title":file_title
            }]
        #3. 对过长的段落进行二次切分。-》细粒度切割 大小和重叠合适 大的往小切 小的合并
        sections=refine_chunks(sections,DEFAULT_MAX_CONTENT_LENGTH,MIN_CONTENT_LENGTH)
    #4. 数据的备份和chunks属性的更改(chunks->state | chunks->本地备份)
        state['chunks']=sections
        #备份到本地
        save_chunks_to_local(state,sections)

    except Exception as e:
        logger.error(f"节点 [{function_name}] 执行失败！错误信息: {e}")
        raise 
    finally:
        logger.info(f"<<< [{function_name}] 执行完毕！现在的状态是: {state}")
        add_done_task(state['task_id'],function_name,True) #任务结束 前端的sse推送
    return state 

    
if __name__ == '__main__':
    """
    单元测试：单独测试 node_document_split（文档切分节点）
    测试条件：已有处理好的MD文件（已包含图片描述等）
    测试流程：直接读取MD文件 → 执行文档切分 → 查看结果
    """

    from app.utils.path_util import PROJECT_ROOT

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（已处理好的文件，如 hak180产品安全手册_new.md）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册_new.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请确认MD文件已处理完成，或修改上面的 test_md_name 为实际文件名")
    else:
        # 直接读取已处理好的MD文件内容
        with open(test_md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        logger.info(f"本地测试 - 成功读取文件，内容长度：{len(md_content)} 字符")

        # 构造测试状态对象（直接传入 md_content，无需经过 node_md_img）
        test_state = {
            "task_id": "test_task_123456",
            "md_content": md_content,
            "file_title": "hak180产品安全手册",
            "local_dir": os.path.join(PROJECT_ROOT, "output"),
        }

        logger.info(">> 开始运行节点：node_document_split（文档切分）")
        final_state = node_document_split(test_state)

        # 获取并展示结果
        final_chunks = final_state.get("chunks", [])
        logger.info(f"✅ 测试成功：最终生成 {len(final_chunks)} 个 Chunk")

        # 打印每个 chunk 的基本信息
        for i, chunk in enumerate(final_chunks, 1):
            title = chunk.get("title", "无标题")
            content_preview = chunk.get("content", "")[:50] + "..."
            logger.info(f"  Chunk {i}: [{title}] {content_preview}")



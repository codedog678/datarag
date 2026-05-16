import sys
from app.utils.task_utils import add_running_task, add_done_task, set_task_result
from app.utils.sse_utils import push_to_session, SSEEvent
from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from app.clients.mongo_history_utils import save_chat_message
import re

_IMAGE_BLOCK_MARKER = "【图片】"
MAX_CONTEXT_CHARS = 12000   #限制上下文字符数

def check_answer_exist(state: QueryGraphState) -> bool:
  """
  检查state 中是否已经存在answer
  1.获取 answer 和 is_stream
  2.有并且是流式-》push_to_session
  3.有不是流式-》set_task_result
  4.返回True 表示answer已经存在
  5.返回False 表示answer不存在
  """
  answer=state.get("answer")
  is_stream=state.get("is_stream")
  if answer:
    if is_stream:
      push_to_session(state['session_id'], SSEEvent.DELTA, {"delta": answer})
    else:
      set_task_result(state['session_id'], "answer", answer)
    return True
  else:
    return False
  
def load_prompt_answer_output(state: QueryGraphState) -> str:
  '''
  加载模型润色答案的提示词
  '''
  #0.获取state 中的rewritten_query、reranked_docs、history、item_names
  rewritten_query=state.get("rewritten_query") or state.get("original_query")
  reranked_docs=state.get("reranked_docs",[])
  history=state.get("history",[])
  item_names=state.get("item_names",[])
  #1.先处理 chunk 块的内容-》content
  docs=[]
  used_len=0
  for i,doc in enumerate(reranked_docs):     #循环处理reranked_docs
    text=doc.get("text","")
    source=doc.get("source","")
    title=doc.get("title","")
    content=f'[{i}][source={source}][title={title}]\n\n{text}'
    if used_len+len(content)>MAX_CONTEXT_CHARS:
      logger.info(f"本次内容停止追加")
      break
    used_len+=len(content)
    docs.append(content)
  final_content="\n\n".join(docs)
  #2.history 中的内容-》history
  history_str=''
  if history and len(history)>0:
    for i,message in enumerate(history,start=1):
      role=message.get("role","")
      text=message.get("text","")
      if role=="user" and text:
        cur_content=f"用户：{text}\n"
      elif role=="agent" and text:
        cur_content=f"助手：{text}\n"
      used_len+=len(cur_content)
      history_str+=cur_content
      if used_len>MAX_CONTEXT_CHARS:
        logger.info(f"历史对话内容停止追加")
        break
  else:
    history_str='没有历史对话内容'

  #3.提问商品（item_names）-》item_names
  item_names_str=','.join(item_names)
  
  #4.问题-》question
  answer_output_prompt=load_prompt("answer_out",
                       context=final_content,
                       history=history_str,
                       item_names=item_names_str,
                       question=rewritten_query,
                       )
  logger.info(f"加载的提示词：{answer_output_prompt}")
  return answer_output_prompt

def generate_answer(state: QueryGraphState, prompt: str) -> str:
  '''
  调用大模型生成答案
  1.获取大模型客户端
  2.获取流式状态
  3.调用大模型生成答案 sse->stream   set_result->invoke
  4.返回答案
   '''
  #1.获取大模型客户端
  llm_client=get_llm_client()
  
  #2.获取流式状态
  is_stream=state.get("is_stream",False)
  answer=""
  if is_stream:
    for chunk in llm_client.stream(prompt):
      delta=chunk.content  #增量片段
      answer+=delta
      push_to_session(state['session_id'], SSEEvent.DELTA, {"delta": delta})
  else:
    response=llm_client.invoke(prompt)
    content=response.content
    answer=content
    set_task_result(state['session_id'], "answer", content)
  state["answer"]=answer
  return answer
    
def extract_images(state: QueryGraphState):
  '''
  两个位置：
  1.local -》chunk ->text中提起 {text:'!【图片】(图片url)'}
  2.web ->url ->图片 {url:'图片url'}(mcp)
  '''
  images=[]   #这个保留图片先后地址
  set_images=set()   #判断重重复 时间复杂度 O1
  #定义正则
  # ---------------------------------------------------------
    # 正则表达式解释：r'!\[.*?\]\((.*?)\)'
    # 1. !\[   -> 匹配 Markdown 图片语法的开头 "![" (注意 [ 需要转义)
    # 2. .*?   -> 非贪婪匹配图片描述文本 (Alt Text)，即 [] 中间的内容
    # 3. \]    -> 匹配描述文本的结束符 "]"
    # 4. \(    -> 匹配 URL 部分的开始符 "("
    # 5. (.*?) -> 捕获组 (Group 1)：非贪婪匹配括号内的实际 URL 内容
    # 6. \)    -> 匹配 URL 部分的结束符 ")"
    # ( ... ) （不带反斜杠）：这就是 捕获组 。
    # 它的作用是告诉程序：“虽然我匹配了整个 ![...](...) 结构，但我 只要 这括号里的内容”。
    # ---------------------------------------------------------
  md_img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')
  #循环处理切片
  for chunk in state.get("reranked_docs",[]): 
    #url是不是图片
    url=chunk.get("url")
    if url :
      if url.endswith(('.jpg', '.png', '.jpeg', '.gif','.webp')):
        if url not in set_images:
          images.append(url)
          set_images.add(url)
    #text中是否有图片
    text=chunk.get("text","")
    if text:
      matches = md_img_pattern.findall(text)
      for match in matches:
        if match not in set_images:
          images.append(match)
          set_images.add(match)
    logger.info(f"提取到的图片数量：{len(images)}")
  state["image_urls"]=images
  return images

def writer_history(state: QueryGraphState):
  '''
  每次对话存两条：1.用户问的 2.助手回的
  '''
  session_id=state.get("session_id")
  answer=state.get("answer")
  rewritten_query=state.get("rewritten_query") or state.get("original_query")
  item_names=state.get("item_names")

  # 存过了 不用在存了
  # if rewritten_query:
  #   save_chat_message(session_id=session_id, 
  #                     role="user", text=rewritten_query,item_names=item_names)
    
  if answer:
    save_chat_message(session_id=session_id, 
                      role="assistant", text=answer,item_names=item_names,rewritten_query=rewritten_query)
  logger.info(f"写入历史对话记录完成")

    



def node_answer_output(state: QueryGraphState) -> QueryGraphState:
  """
  意图识别那里 传来的answer 进行判断
  1 判断state 中的answer是否已经存在，如果存在直接输出answer中的答案，注意判断是否需要流式输出需要则流式输出
  2 根据state中的问题、重新问题、历史对话、提问商品（item_names）、 重排内容 组织prompt 并调用llm 生成答案answer
  3 阶段三：调用大模型输出答案 注意判断是否需要流式输出需要则流式输出
  4 把答案写入到mongodb的history中 利用utils/mongo_history_utils.py中的save_chat_message方法
  5 做最后一次push操作（主要是为了触发前端图片渲染)
     {
        "answer": "HAK 180 烫金机的操作面板位于...（大模型生成的纯文本）...",
        "status": "completed",
        "image_urls": [
            "http://local-server/images/panel_view.jpg",
            "http://local-server/images/button_detail.jpg"
        ]
      }
      图片最后单独处理 统一返回 也就是提取topklist中的图片url 然后返回给前端渲染
  6 sse 的final event 输出
  """
  logger.info("---node_answer_output (答案生成) 节点开始处理---")
  add_running_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))

  #1. 判断state 中的answer是否已经存在，如果存在直接输出answer中的答案，注意判断是否需要流式输出需要则流式输出
  answer_exist=check_answer_exist(state)
  images_urls=[]
  answer=""
  if not answer_exist:
    #没有 则加载提示词
    prompt=load_prompt_answer_output(state)
    #生成答案
    answer=generate_answer(state, prompt)
    #图片处理
    images_urls=extract_images(state)
  else:
    #answer已存在，从state中取出用于FINAL事件
    answer=state.get("answer","")
    images_urls=extract_images(state)
  #始终发送FINAL事件，确保前端能收到完整响应
  push_to_session(state['session_id'], SSEEvent.FINAL, {"answer": answer, "status": "completed", "image_urls": images_urls})
  
  #添加聊天记录
  writer_history(state)



  add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
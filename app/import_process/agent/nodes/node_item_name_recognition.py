import os
import sys
from typing import List, Dict, Any, Tuple

# 导入Milvus客户端（向量数据库核心操作）、数据类型枚举（定义集合Schema）
from pymilvus import MilvusClient, DataType
# 导入LangChain消息类（标准化大模型对话消息格式）
from langchain_core.messages import SystemMessage, HumanMessage

# 导入自定义模块：
# 1. 流程状态载体：ImportGraphState为LangGraph流程的统一状态管理对象
from app.import_process.agent.state import ImportGraphState
# 2. Milvus工具：获取单例Milvus客户端，实现连接复用
from app.clients.milvus_utils import get_milvus_client
# 3. 大模型工具：获取大模型客户端，统一模型调用入口
from app.lm.lm_utils import get_llm_client
# 4. 向量工具：BGE-M3模型实例、向量生成方法（稠密+稀疏向量）
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
# 5. 稀疏向量工具：归一化处理，保证向量长度为1，提升检索准确性
from app.utils.normalize_sparse_vector import normalize_sparse_vector
# 6. 任务工具：更新任务运行状态，用于任务监控和管理
from app.utils.task_utils import add_done_task, add_running_task
# 7. 日志工具：项目统一日志入口，分级输出（info/warning/error）
from app.core.logger import logger
# 8. 提示词工具：加载本地prompt模板，实现提示词与代码解耦
from app.core.load_prompt import load_prompt

from app.utils.escape_milvus_string_utils import escape_milvus_string


'''
主体识别节点
1.录用文本大模型识别当前的chunks对应的item_name 区分不同的文档
2.采用嵌入模型，将当前的chunks对应的item_name 转换为向量表示存入向量数据库
3.修改state中chunks->title parent_title part content item_name file_name
实现步骤：
1.校验和取值
2.构建上下文 拼接成context
3.调用大模型 识别当前的chunks对应的item_name
4.修改state中chunks->item_name
5.item_name 向量化(稀疏 稠密)
6.将向量数据写入向量数据库
'''

# --- 配置参数 (Configuration) ---
# 大模型识别商品名称的上下文切片数：取前5个切片，避免上下文过长导致大模型输入超限
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 单个切片内容截断长度：防止单切片内容过长，占满大模型上下文
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
CONTEXT_TOTAL_MAX_CHARS = 2500


def get_chunks(state: ImportGraphState) -> Tuple[str,List[Dict[str,Any]]]:
    """
    从state中获取file_title chunks(file_title是做兜底的，如果识别不出来就用标题)
    """
    file_title = state["file_title"]
    chunks = state["chunks"]
    if not chunks:
        raise ValueError("chunks为空，无法识别item_name")
    if not file_title:
        #md_path中获取文件名 多重保险
        file_title = os.path.basename(state["md_path"])
        state["file_title"] = file_title
    return file_title,chunks

def build_context(chunks: List[Dict[str,Any]]) -> str:
    '''
    构建上下文 拼接成context 内容限制：
    1. 取前5个切片
    2. 每个切片内容截断长度：800
    3. 上下文总字符数上限：2500
    截取内容处理：
        切片：{1}，标题{title}，内容{content}\n\n
    '''
    #前置准备
    parts=[]  #存储处理后的切片
    #累加的字符串数量
    total_chars = 0
    #遍历切片
    for i,chunk in enumerate(chunks[:DEFAULT_ITEM_NAME_CHUNK_K],start=1):
        chunk_title = chunk.get("title","")
        chunk_content = chunk.get("content","")
        # # 截取内容处理
        # if len(chunk_content) + total_chars > SINGLE_CHUNK_CONTENT_MAX_LEN:
        #     chunk_content = chunk_content[:SINGLE_CHUNK_CONTENT_MAX_LEN - total_chars]
        # #其实也可以不管 就全部拼接起来 然后最后截取总字符数不超过2500
        data=f"切片：{i}，标题{chunk_title}，内容{chunk_content}\n\n"
        parts.append(data)
        total_chars += len(data)
        if total_chars >= CONTEXT_TOTAL_MAX_CHARS:
            break
        context = "\n\n".join(parts)
        final_context = context[:CONTEXT_TOTAL_MAX_CHARS]
    return final_context

def call_llm(context: str,file_title: str) -> str:
    '''
    调用大模型 识别当前的chunks对应的item_name file_title兜底
    '''
    #1.构建提示词
    human_prompt = load_prompt("item_name_recognition",context=context,file_title=file_title)
    system_prompt = load_prompt("product_recognition_system")
    #2.获取模型对象
    llm_client = get_llm_client(json_mode=False)
    #3.执行调用
    messages=[
        HumanMessage(content=human_prompt),
        SystemMessage(content=system_prompt)
    ]
    response = llm_client.invoke(messages)
    #4.结果判断和兜底
    item_name = response.content
    if not item_name:
        item_name = file_title
    #5.返回识别结果
    return item_name

def update_chunks_and_state(state: ImportGraphState,item_name: str,chunks: List[Dict[str,Any]]) -> None:
    '''
    state["item_name"] = item_name
    chunks->{item_name: item_name}
    '''
    state["item_name"] = item_name
    for chunk in chunks:
        chunk["item_name"] = item_name
    state["chunks"] = chunks

def generate_two_embeddings(item_name: str) -> Tuple[List[float], Dict[str, float]]:
    #客户端已经写好了 封装的嵌入式模型生成向量
    '''
    参数：列表形式的item_name
    返回：稠密向量和稀疏向量
    result={
        "dense": dense_vector,  #[1的稠密向量，2的稠密向量...]
        "sparse": sparse_vector #[1的稀疏向量，2的稀疏向量...] 
        #每个稀疏向量是一个字典，key为列索引，value为向量值
    }
    '''
    result = generate_embeddings([item_name])
    dense_vector = result["dense"][0]
    sparse_vector = result["sparse"][0]
    logger.info(f"API返回的稀疏向量: {result['sparse'][0]}")
    return dense_vector, sparse_vector

def save_to_milvus(dense_vector: List[float], sparse_vector: Dict[str, float], item_name: str,file_title: str) -> None:
    '''
    将item_name的向量数据写入向量数据库
    '''
    #1.获取单例Milvus客户端
    milvus_client = get_milvus_client()
    # 从环境变量读取Milvus核心配置，与MilvusConfig配置类保持一致
    milvus_uri = os.environ.get("MILVUS_URL")
    collection_name = os.environ.get("ITEM_NAME_COLLECTION")
    #2.判断是否存在集合（表），不存在则创建
    if not milvus_client.has_collection(collection_name=collection_name):
        #创建集合
        #01 创建集合对应的列的信息
        schema=milvus_client.create_schema(
            auto_id=False,#自动生成id 主键自增长
            enable_dynamic_field=True,#开启动态字段 允许动态增加
        )
        #02 定义集合的字段信息 主键+file_title+item_name+dense_vector+sparse_vector
        schema.add_field(
            field_name="pk",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        )
        schema.add_field(
            field_name="file_title",
            datatype=DataType.VARCHAR,
            max_length=65535,
        )
        schema.add_field(
            field_name="item_name",
            datatype=DataType.VARCHAR,
            max_length=65535,
        )
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,#官网可查类型
            dim=1024,  #维度是自己的 嵌入模型的输出维度
        )
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR,#稀疏向量是没有维度的
        )
        #03 配置索引 给谁加方便：两个向量 加速近似最近邻搜索（ANN）
        #milvus 向量必须加索引，标量按需加索引 他的核心作用是加速搜索
        #创建一个空的索引参数容器，后续用来 配置各个字段的索引
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector", #给哪个列创建索引
            index_name='dense_vector_index',#索引名称
            index_type='HNSW',#配置查找索引的算法 熟悉：IVF HNSW 两个系列  详细可看milvus官网 索引解释
            metric_type='COSINE',#配置索引的距离度量方式 ip cosine 稠密一般余弦相似度
            params={
                "M":16,
                "efConstruction":200
            },#索引算法参数
        )
        '''
            10000   M=16   efConstruction=200
            50000   M=32   efConstruction=300
            100000   M=64   efConstruction=400
        '''
        index_params.add_index(
            field_name="sparse_vector",
            index_name='sparse_vector_index',
            index_type='SPARSE_INVERTED_INDEX',#稀疏向量索引 只有一种
            metric_type='IP',#配置索引的距离度量方式 ip cosine 稠密一般余弦相似度

        )
        #04 创建集合
        milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )

    #3.先删除之前存在的item_name的向量数据
    #加载和选中集合
    milvus_client.load_collection(collection_name)  
    milvus_client.delete(
        collection_name=collection_name,
        filter=f'item_name == "{item_name}"',
    )
    #4.向集合插入最新的item_name的数据
    item={
        "file_title": file_title,
        "item_name": item_name,
        "dense_vector": dense_vector,
        "sparse_vector": sparse_vector,
    }
    milvus_client.insert(
        collection_name=collection_name,    
        data=[item],
    )
    milvus_client.load_collection(collection_name)
    logger.info(f">>> [ 将item_name: {item_name} 数据写入向量数据库")


def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition)
    为什么叫这个名字: 识别文档核心描述的物品/商品名称 (Item Name)。
    未来要实现:
    1. 取文档前几段内容。
    2. 调用 LLM 识别这篇文档讲的是什么东西 (如: "Fluke 17B+ 万用表")。
    3. 存入 state["item_name"] 用于后续数据幂等性清理。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] 开始执行！现在的状态是: {state}")
    #add running task 在内存中记录当前正在运行的任务节点 用于前端的sse推送
    add_running_task(state['task_id'],function_name)

    try:
        # 1. 校验和取值 从state中获取file_title chunks(file_title是做兜底的，如果识别不出来就用标题)
        file_title,chunks=get_chunks(state)

        # 2. 构建上下文 拼接成context
        context = build_context(chunks)

        # 3. 调用大模型 识别当前的chunks对应的item_name file_title是兜底的
        item_name = call_llm(context,file_title)

        # 4. 修改state中chunks->item_name 
        #期待的chunks结构：[{title parent_title part content item_name（给这部分赋值） file_name}]
        update_chunks_and_state(state,item_name,chunks)

        # 5. item_name 向量化(稀疏 稠密)
        dense_vector, sparse_vector = generate_two_embeddings(item_name)

        # 6. 将向量数据写入向量数据库
        save_to_milvus(dense_vector, sparse_vector, item_name,file_title)

    except Exception as e:
        logger.error(f">>> [{function_name}] 主题识别出错了！错误信息是: {e}")
        raise
    finally:
        logger.info(f">>> [{function_name}] 执行完成！现在的状态是: {state}")
        add_done_task(state['task_id'],function_name)
    return state

# ===================== 本地测试方法（直接运行调试，无需启动LangGraph） =====================
def test_node_item_name_recognition():
    """
    商品名称识别节点本地测试方法
    功能：模拟LangGraph流程输入，独立测试node_item_name_recognition节点全链路逻辑
    适用场景：本地开发、调试、单节点功能验证，无需启动整个LangGraph流程
    测试前准备：
        1. 确保项目环境变量配置完成（MILVUS_URL/ITEM_NAME_COLLECTION等）
        2. 确保大模型、Milvus、BGE-M3服务均可正常访问
        3. 确保prompt模板（item_name_recognition/product_recognition_system）已存在
    使用方法：
        直接运行该函数：if __name__ == "__main__": test_node_item_name_recognition()
    """
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")
    try:
        # 1. 构造模拟的ImportGraphState状态（模拟上游节点产出数据）
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",  # 测试任务ID
            "file_title": "华为Mate60 Pro手机使用说明书",  # 模拟文件标题
            "file_name": "华为Mate60Pro说明书.pdf",  # 模拟原始文件名（兜底用）
            # 模拟文本切片列表（上游切片节点产出，含title/content字段）
            "chunks": [
                {
                    "title": "产品简介",
                    "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
                },
                {
                    "title": "拍照功能",
                    "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
                },
                {
                    "title": "电池参数",
                    "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
                }
            ]
        })

        # 2. 调用商品名称识别核心节点
        result_state = node_item_name_recognition(mock_state)

        # 3. 打印测试结果（调试用）
        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")

        # 4. 验证Milvus存储（可选）
        milvus_client = get_milvus_client()
        collection_name = os.environ.get("ITEM_NAME_COLLECTION")
        if milvus_client and collection_name:
            milvus_client.load_collection(collection_name)
            # 检索测试结果
            item_name = result_state.get('item_name')
            safe_name = escape_milvus_string(item_name)
            res = milvus_client.query(
                collection_name=collection_name,
                filter=f'item_name=="{safe_name}"',
                output_fields=["file_title", "item_name"]
            )
            logger.info(f"Milvus中检索到的数据：{res}")

    except Exception as e:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)


# 测试方法运行入口：直接执行该文件即可触发测试
if __name__ == "__main__":
    # 执行本地测试
    test_node_item_name_recognition()
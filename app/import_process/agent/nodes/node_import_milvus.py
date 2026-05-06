import os
import sys
from typing import List, Dict, Any
# 导入Milvus相关依赖
from pymilvus import DataType
# 导入自定义模块
from app.import_process.agent.state import ImportGraphState
from app.clients.milvus_utils import get_milvus_client
from app.utils.task_utils import add_done_task, add_running_task
from app.core.logger import logger
from app.conf.milvus_config import milvus_config

# 从配置文件读取切片集合名称，与配置解耦，便于环境切换
CHUNKS_COLLECTION_NAME = milvus_config.chunks_collection

#http://milvus-standalone:19530 不是本机电脑单独配置了
#因为在docker-compose.yml中配置了milvus-standalone的别名 所以这里可以直接使用milvus-standalone:19530

def prepare_collections(state: ImportGraphState) :
    '''
    创建chunks对应的集合 之前有写过 粘贴即可
    '''
    milvus_client = get_milvus_client()
    #2.判断是否存在集合（表），不存在则创建
    if not milvus_client.has_collection(collection_name=CHUNKS_COLLECTION_NAME):
        #创建集合
        #01 创建集合对应的列的信息
        schema=milvus_client.create_schema(
            auto_id=False,#自动生成id 主键自增长
            enable_dynamic_field=True,#开启动态字段 允许动态增加
        )
        #02 定义集合的字段信息 
        schema.add_field(
            field_name="chunk_id",
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
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=65535,
        )
        schema.add_field(
            field_name="title",
            datatype=DataType.VARCHAR,
            max_length=65535,
        )
        schema.add_field(
            field_name='parent_title',
            datatype=DataType.VARCHAR,
            max_length=65535,
        )
        schema.add_field(
            field_name='part',
            datatype=DataType.INT8,
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
                "M":32,
                "efConstruction":300
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
            collection_name=CHUNKS_COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
        )

    return milvus_client

def delete_old_data(milvus_client, item_name):
    '''
    删除item_name对应的向量数据
    '''
    milvus_client.load_collection(collection_name=CHUNKS_COLLECTION_NAME)  
    milvus_client.delete(
        collection_name=CHUNKS_COLLECTION_NAME,
        filter=f'item_name == "{item_name}"',
    )
    milvus_client.load_collection(collection_name=CHUNKS_COLLECTION_NAME)
    logger.info(f">>> [ 将item_name: {item_name} 数据删除自向量数据库")

def insert_chunks_data(milvus_client, chunks):
    '''
    批量插入chunks数据
    '''
    
    insert_result = milvus_client.insert(
        collection_name=CHUNKS_COLLECTION_NAME,    
        data=chunks,
    )
    insert_count=insert_result.get("insert_count",0)
    logger.info(f">>> [ 将{insert_count}条数据写入向量数据库 ]")
    #获取回显的ids
    ids=insert_result.get('ids',[])
    if ids and len(ids)==len(chunks):
        for index,chunk in enumerate(chunks):
            chunk['chunk_id']=ids[index]
    return chunks

  

def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入向量库 (node_import_milvus)
    将处理好的向量数据写入 Milvus 数据库。
    未来要实现:
    1. 连接 Milvus。
    2. 根据 item_name 删除旧数据 (幂等性)。
    3. 批量插入新的向量数据。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] 开始执行！现在的状态是: {state}")
    #add running task 在内存中记录当前正在运行的任务节点 用于前端的sse推送
    add_running_task(state['task_id'],function_name)

    try:
        #1.获取输入数据
        chunks = state['chunks']
        if not chunks:
            logger.error("输入数据为空，无法导入向量库")
            raise ValueError("输入数据为空，无法导入向量库")

        #2.没有集合要创建集合 collection（field index collection）
        milvus_client=prepare_collections(state)

        #3.删除旧数据
        delete_old_data(milvus_client, chunks[0]['item_name'])

        #4.插入chunks数据
        with_id_chunks=insert_chunks_data(milvus_client, chunks)

        #5.更新状态
        state['chunks'] = with_id_chunks
    except Exception as e:
        logger.error(f" >>> [{function_name}] 导入向量数据库失败: {e}")
        raise 
    finally:
        logger.info(f">>> [{function_name}] 执行完毕！现在的状态是: {state}")
        add_done_task(state['task_id'],function_name)
    return state

if __name__ == '__main__':
    # --- 单元测试 ---
    # 目的：验证 Milvus 导入节点的完整流程，包括连接、创建集合、清理旧数据和插入新数据。
    import sys
    import os
    from dotenv import load_dotenv

    # 加载环境变量 (自动寻找项目根目录的 .env)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # 构造测试数据
    dim = 1024
    test_state = {
        "task_id": "test_milvus_task",
        "chunks": [
            {
                "content": "Milvus 测试文本 1",
                "title": "测试标题",
                "item_name": "测试项目_Milvus",  # 必须有 item_name，用于幂等清理
                "parent_title":"test.pdf",
                "part":1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            },
            {
                "content": "Milvus 测试文本 2",
                "title": "测试标题2",
                "item_name": "测试项目_Milvus2",  # 必须有 item_name，用于幂等清理
                "parent_title":"test2.pdf",
                "part":1,
                "file_title": "test2.pdf",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
        ]
    }

    print("正在执行 Milvus 导入节点测试...")
    try:
        # 检查必要的环境变量
        if not os.getenv("MILVUS_URL"):
            print("❌ 未设置 MILVUS_URL，无法连接 Milvus")
        elif not os.getenv("CHUNKS_COLLECTION"):
            print("❌ 未设置 CHUNKS_COLLECTION")
        else:
            # 执行节点函数
            result_state = node_import_milvus(test_state)

            # 验证结果
            chunks = result_state.get("chunks", [])
            if chunks and chunks[0].get("chunk_id"):
                print(f"✅ Milvus 导入测试通过，生成 ID: {chunks[0]['chunk_id']}")
            else:
                print("❌ 测试失败：未能获取 chunk_id")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
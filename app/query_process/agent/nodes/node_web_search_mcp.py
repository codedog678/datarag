import sys
import json
import asyncio
from app.utils.task_utils import add_done_task, add_running_task
from app.conf.bailian_mcp_config import mcp_config
from agents.mcp import MCPServerSse, MCPServerStreamableHttp
from app.core.logger import logger

DASHSCOPE_BASE_URL_STREAMABLE = mcp_config.mcp_base_url
API_KEY = mcp_config.api_key

async def mcp_call_streamable(query: str):
    """
    调用mcp streamable接口
    :param query:
    :return:
    """
    #openai mcp官方写法  看官网文档
    search_mcp=MCPServerStreamableHttp(
        name='search_mcp',
        params={
            'url':DASHSCOPE_BASE_URL_STREAMABLE,
            'headers':{
                'Authorization':f'Bearer {API_KEY}'
            },
            'timeout':10,
        },
        max_retry_attempts=3  # 最大重试次数
    )

    try:
        await search_mcp.connect()
        result=await search_mcp.call_tool(
            tool_name='bailian_web_search',
            arguments={
                'query':query,
                'count':5,}
        )
        return result
    
    finally:
        await search_mcp.cleanup()

def node_web_search_mcp(state):
    """
    节点功能，调用外部搜索引擎补充信息
    :param state:
    :return:
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])
    print("---node-web-search-mcp处理---")

    #1.获取问题
    query=state['rewritten_query']

    #2.调用mcp外部引擎
    result=asyncio.run(mcp_call_streamable(query))


    #3.结果处理 并行不直接返回state
    '''
    {
        "isError": false,
        "content": [
            {
            "text": 
                "{\"pages\":
                    [{\"snippet\":\"《哈哈哈哈哈 第六季》综艺在线观看- 全集大陆综艺- 口袋影视 哈哈哈哈哈第六季\",
                    \"hostname\":\"美食\",\"hostlogo\":\"\",
                    \"title\":\"《哈哈哈哈哈 第六季》综艺在线观看- 全集大陆综艺- 口袋影视\",
                    \"url\":\"http://koudaxiang.com/news/1439037.html\"},
                {\"snippet\":\"《哈哈哈哈哈第四季》综艺全集-免费在线观看高清完整版-片库网 哈哈哈哈哈第四季\",\"hostname\":\"\",\"hostlogo\":\"https://pkwdy.com/template/pkwdy/images/favicon.ico\",\"title\":\"《哈哈哈哈哈第四季》综艺全集-免费在线观看高清完整版-片库网\",\"url\":\"http://pkwdy.com/pplayk/1510-1-1.html\"}],\"request_id\":\"f6469ac9-211a-4588-ba92-e5ca447887da\",\"tools\":[],\"status\":0}",
            "type": "text"
            }
        ]
    }
    '''
    #特大错误：result.content 是 属性 （attribute），不是字典的键（key） 不能用[]访问
    #web_document=json.loads(result['content'][0]['text'].get('pages',[]))
    web_document = json.loads(result.content[0].text).get('pages',[])
    logger.info(f"mcp搜索的结果为:{web_document}")

    add_done_task(state["session_id"],sys._getframe().f_code.co_name,state["is_stream"])
    return {'web_search_docs':web_document}
if __name__ == '__main__':
    # 测试代码：单独运行该文件时，验证MCP搜索功能是否正常
    print("\n" + "="*50)
    print(">>> 启动 node_web_search_mcp 本地测试")
    print("="*50)
    
    test_state = {
        "session_id": "test_mcp_session",
        "rewritten_query": "HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部 50 mm–170 mm 的局部区域，应在操作面板上如何设置",
        "is_stream": True
    }

    try:
        # 调用MCP搜索节点函数，执行测试
        result_state = node_web_search_mcp(test_state)

        print("\n" + "="*50)
        print(">>> 测试结果摘要:")
        search_results = result_state.get('web_search_docs', [])
        print(f"搜索结果数量: {len(search_results)}")
        if search_results:
            print("首条结果预览:")
            print(json.dumps(search_results[0], indent=2, ensure_ascii=False))
        else:
            print("未获取到搜索结果")
        print("="*50)
        
    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
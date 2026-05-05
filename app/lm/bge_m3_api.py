"""
BGE-M3 API工具类 - 通过API方式调用BGE-M3模型
替代本地加载，降低本地负载
支持稠密向量（dense）和稀疏向量（sparse）的生成
"""
import os
import requests
from typing import List, Dict, Any
from app.conf.embedding_config import embedding_config
from app.core.logger import logger
from app.utils.rate_limit_utils import apply_api_rate_limit
from collections import deque

# API速率限制队列
_embedding_request_times = deque()


class BGE_M3_API:
    """BGE-M3 API客户端 - 支持稠密和稀疏向量"""
    
    def __init__(self):
        self.api_url = os.getenv("BGE_M3_API_URL", "https://api.siliconflow.cn/v1/embeddings")
        self.api_key = os.getenv("BGE_M3_API_KEY")
        self.model = "bge-m3"  # 模型名称
        
        if not self.api_key:
            raise ValueError("请在.env中配置BGE_M3_API_KEY")
        
    def _parse_sparse_vector(self, sparse_data: Any) -> Dict[int, float]:
        """
        解析稀疏向量数据，统一转换为 {index: weight} 字典格式
        参考 embedding_utils.py 中的稀疏向量处理方式
        
        :param sparse_data: API返回的稀疏向量数据
        :return: 字典格式的稀疏向量 {特征索引: 权重}
        """
        sparse_dict = {}
        
        if sparse_data is None:
            return sparse_dict
            
        if isinstance(sparse_data, dict):
            # 格式1: {"indices": [...], "values": [...]}
            if "indices" in sparse_data and "values" in sparse_data:
                indices = sparse_data["indices"]
                values = sparse_data["values"]
                # 转换为 Python 原生类型（参考 embedding_utils.py 中的 .tolist() 处理）
                for idx, val in zip(indices, values):
                    sparse_dict[int(idx)] = float(val)
            # 格式2: {"token_id": weight, ...} 直接是字典
            else:
                for k, v in sparse_data.items():
                    sparse_dict[int(k)] = float(v)
        elif isinstance(sparse_data, list):
            # 格式3: [[index, weight], ...] 或 [{"index": i, "value": v}, ...]
            for item in sparse_data:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    sparse_dict[int(item[0])] = float(item[1])
                elif isinstance(item, dict) and "index" in item and "value" in item:
                    sparse_dict[int(item["index"])] = float(item["value"])
        
        return sparse_dict
    
    def generate_embeddings(self, texts: List[str]) -> Dict[str, Any]:
        """
        使用API为文本列表生成向量嵌入（支持稠密和稀疏向量）
        参考 embedding_utils.py 的返回格式
        
        :param texts: 要生成嵌入的文本列表
        :return: 字典格式的向量结果，key为dense/sparse，与本地模型格式一致
        """
        if not isinstance(texts, list) or len(texts) == 0:
            raise ValueError("参数texts必须是包含文本的非空列表")
        
        logger.info(f"开始使用BGE-M3 API为{len(texts)}条文本生成向量嵌入")
        
        # 应用API速率限制
        apply_api_rate_limit(
            request_times=_embedding_request_times,
            max_requests=30,
            window_seconds=60
        )
        
        try:
            # 准备API请求数据
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # 构建请求体，请求稠密和稀疏向量
            data = {
                "model": self.model,
                "input": texts,
                "return_dense": True,
                "return_sparse": True
            }
            
            # 发送API请求
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code != 200:
                logger.error(f"BGE-M3 API请求失败: {response.status_code} - {response.text}")
                raise Exception(f"BGE-M3 API请求失败: {response.status_code}")
            
            result = response.json()
            logger.debug(f"BGE-M3 API响应解析完成，共{len(result.get('data', []))}条结果")
            
            # 解析API响应，格式与 embedding_utils.py 保持一致
            dense_embeddings = []
            sparse_embeddings = []
            
            for item in result.get("data", []):
                # 解析稠密向量 - 转为列表格式（参考 embedding_utils.py: emb.tolist()）
                if "embedding" in item and item["embedding"] is not None:
                    dense_embeddings.append(item["embedding"])
                else:
                    # 如果没有稠密向量，添加空列表
                    dense_embeddings.append([])
                
                # 解析稀疏向量 - 转为 {index: weight} 字典格式
                sparse_dict = {}
                if "sparse" in item and item["sparse"] is not None:
                    sparse_dict = self._parse_sparse_vector(item["sparse"])
                sparse_embeddings.append(sparse_dict)
            
            # 验证稠密和稀疏向量数量一致
            if len(dense_embeddings) != len(sparse_embeddings):
                logger.warning(f"稠密向量数量({len(dense_embeddings)})与稀疏向量数量({len(sparse_embeddings)})不一致")
            
            logger.debug(
                f"BGE-M3 API嵌入生成完成，稠密向量：{len(dense_embeddings)}, "
                f"稀疏向量：{len(sparse_embeddings)}"
            )
            
            # 返回与 embedding_utils.py 完全一致的格式
            result = {
                "dense": dense_embeddings,      # 嵌套列表，与输入文本一一对应
                "sparse": sparse_embeddings     # 字典列表，与输入文本一一对应
            }
            
            logger.success(f"{len(texts)}条文本向量生成完成（API模式），格式已适配工业级使用")
            return result
            
        except requests.exceptions.Timeout:
            logger.error("BGE-M3 API请求超时")
            raise Exception("BGE-M3 API请求超时")
        except Exception as e:
            logger.error(f"BGE-M3 API调用失败: {str(e)}")
            raise


# 全局API客户端实例（单例模式，与 embedding_utils.py 风格一致）
_bge_m3_api_client = None


def get_bge_m3_api_client():
    """获取BGE-M3 API客户端单例"""
    global _bge_m3_api_client
    
    if _bge_m3_api_client is None:
        _bge_m3_api_client = BGE_M3_API()
        logger.info("BGE-M3 API客户端初始化成功")
    
    return _bge_m3_api_client


def generate_bge_m3_embeddings_api(texts: List[str]) -> Dict[str, Any]:
    """
    使用API为文本列表生成向量嵌入（对外接口）
    返回格式与 generate_embeddings() 完全一致：
    {
        "dense": [[...], [...], ...],   # 稠密向量列表
        "sparse": [{}, {}, ...]         # 稀疏向量字典列表
    }
    
    :param texts: 要生成嵌入的文本列表
    :return: 包含稠密+稀疏向量的字典格式结果
    """
    client = get_bge_m3_api_client()
    return client.generate_embeddings(texts)
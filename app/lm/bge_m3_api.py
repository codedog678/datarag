"""
BGE-M3 API工具类 - 通过API方式调用BGE-M3模型
替代本地加载，降低本地负载
支持稠密向量（dense）和稀疏向量（sparse）的生成
"""
import os
import requests
import numpy as np
from transformers import AutoTokenizer
from typing import List, Dict, Any
from app.conf.embedding_config import embedding_config
from app.core.logger import logger
from app.utils.rate_limit_utils import apply_api_rate_limit
from collections import deque

# API速率限制队列
_embedding_request_times = deque()

# 分词器单例对象（仅加载分词器，不加载神经网络模型，内存开销极小）
_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer

    model_name = embedding_config.bge_m3_path or "BAAI/bge-m3"
    logger.info(f"开始加载BGE-M3分词器（用于稀疏向量生成）: {model_name}")

    try:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.success("BGE-M3分词器加载成功")
        return _tokenizer
    except Exception as e:
        logger.error(f"BGE-M3分词器加载失败: {e}", exc_info=True)
        raise


class BGE_M3_API:
    """BGE-M3 API客户端 - 支持稠密和稀疏向量"""
    
    def __init__(self):
        self.api_url = os.getenv("BGE_M3_API_URL", "https://api.siliconflow.cn/v1/embeddings")
        self.api_key = os.getenv("BGE_M3_API_KEY")
        self.model = "bge-m3"  # 模型名称
        
        if not self.api_key:
            raise ValueError("请在.env中配置BGE_M3_API_KEY")
        
    def _generate_sparse_from_tokenizer(self, texts: List[str]) -> List[Dict[int, float]]:
        """
        使用BGE-M3分词器在本地生成稀疏向量
        参考 embedding_utils.py 中 CSR 矩阵的解析逻辑，模拟相同输出
        
        :param texts: 文本列表
        :return: 字典列表，每个字典为 {token_id: weight}
        """
        tokenizer = _get_tokenizer()
        processed_sparse = []

        for text in texts:
            encoding = tokenizer(text, add_special_tokens=False)
            token_ids = encoding["input_ids"]
            tokens = tokenizer.convert_ids_to_tokens(token_ids)

            unigram_weights = {}
            for token_id, token in zip(token_ids, tokens):
                if token.startswith('▁') or token.startswith(' '):
                    weight = 1.0
                elif token_id in unigram_weights:
                    weight = unigram_weights[token_id] + 1.0
                else:
                    weight = 1.0
                unigram_weights[token_id] = weight

            values = np.array(list(unigram_weights.values()), dtype=np.float32)
            l2_norm = np.linalg.norm(values)
            if l2_norm > 1e-9:
                normalized_values = (values / l2_norm).tolist()
            else:
                normalized_values = values.tolist()

            sparse_dict = {int(k): v for k, v in zip(unigram_weights.keys(), normalized_values)}
            processed_sparse.append(sparse_dict)

        return processed_sparse
    
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
                "input": texts
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
            
            dense_embeddings = []
            for item in result.get("data", []):
                if "embedding" in item and item["embedding"] is not None:
                    dense_embeddings.append(item["embedding"])
                else:
                    raise Exception("API响应中缺少稠密向量数据")

            logger.debug(f"稠密向量解析完成，共{len(dense_embeddings)}条，开始生成稀疏向量")
            sparse_embeddings = self._generate_sparse_from_tokenizer(texts)

            logger.debug(
                f"BGE-M3向量生成完成，稠密向量：{len(dense_embeddings)}, "
                f"稀疏向量：{len(sparse_embeddings)}"
            )

            result = {
                "dense": dense_embeddings,
                "sparse": sparse_embeddings
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
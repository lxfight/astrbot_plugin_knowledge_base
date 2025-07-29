"""
API重排序客户端实现
支持多种API服务提供商的统一接口
"""

import asyncio
import aiohttp
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import time
from dataclasses import dataclass
import hashlib
from functools import lru_cache

from .base import Document


@dataclass
class RerankRequest:
    """重排序请求数据结构"""
    query: str
    documents: List[Tuple[Document, float]]
    top_k: int = 5


@dataclass
class RerankResponse:
    """重排序响应数据结构"""
    results: List[Tuple[Document, float]]
    provider: str
    cached: bool = False
    error: Optional[str] = None


class BaseAPIClient(ABC):
    """API客户端基类"""
    
    def __init__(self, config):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def rerank(self, request: RerankRequest) -> RerankResponse:
        """执行重排序"""
        pass
    
    @abstractmethod
    def _prepare_payload(self, request: RerankRequest) -> Dict[str, Any]:
        """准备API请求负载"""
        pass
    
    @abstractmethod
    def _parse_response(self, response_data: Dict[str, Any], original_docs: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """解析API响应"""
        pass
    
    async def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """发送HTTP请求"""
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
        
        headers = self._get_headers()
        
        for attempt in range(self.config.max_retries + 1):
            try:
                async with self.session.post(
                    self.config.api_url,
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        logging.warning(f"API请求失败 (状态码: {response.status}): {error_text}")
                        
                        if attempt < self.config.max_retries:
                            await asyncio.sleep(2 ** attempt)  # 指数退避
                        else:
                            raise Exception(f"API请求失败: {response.status} - {error_text}")
                            
            except asyncio.TimeoutError:
                if attempt < self.config.max_retries:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise Exception("API请求超时")
            except Exception as e:
                if attempt < self.config.max_retries:
                    logging.warning(f"API请求异常，重试 {attempt + 1}/{self.config.max_retries}: {e}")
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise e
    
    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }


class CohereClient(BaseAPIClient):
    """Cohere Rerank API客户端"""
    
    def __init__(self, config):
        super().__init__(config)
        if not config.api_url:
            config.api_url = "https://api.cohere.ai/v1/rerank"
    
    def _get_headers(self) -> Dict[str, str]:
        """Cohere专用请求头"""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
    
    def _prepare_payload(self, request: RerankRequest) -> Dict[str, Any]:
        """准备Cohere API请求负载"""
        documents = [doc.text_content for doc, _ in request.documents]
        
        payload = {
            "query": request.query,
            "documents": documents,
            "top_n": min(request.top_k, len(documents)),
            "return_documents": False
        }
        
        if self.config.model_name:
            payload["model"] = self.config.model_name
        
        return payload
    
    def _parse_response(self, response_data: Dict[str, Any], original_docs: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """解析Cohere API响应"""
        results = []
        
        if "results" not in response_data:
            raise ValueError("无效的API响应格式")
        
        # 创建索引映射
        doc_map = {i: doc for i, doc in enumerate(original_docs)}
        
        # 按重排序结果重新组织
        for result in response_data["results"]:
            index = result["index"]
            relevance_score = result["relevance_score"]
            
            if index < len(original_docs):
                doc, original_score = original_docs[index]
                results.append((doc, relevance_score))
        
        return results
    
    async def rerank(self, request: RerankRequest) -> RerankResponse:
        """执行Cohere重排序"""
        try:
            payload = self._prepare_payload(request)
            response_data = await self._make_request(payload)
            reranked_docs = self._parse_response(response_data, request.documents)
            
            return RerankResponse(
                results=reranked_docs,
                provider="cohere",
                cached=False
            )
            
        except Exception as e:
            logging.error(f"Cohere重排序失败: {e}")
            return RerankResponse(
                results=[],
                provider="cohere",
                error=str(e)
            )


class JinaClient(BaseAPIClient):
    """Jina Rerank API客户端"""
    
    def __init__(self, config):
        super().__init__(config)
        if not config.api_url:
            config.api_url = "https://api.jina.ai/v1/rerank"
    
    def _prepare_payload(self, request: RerankRequest) -> Dict[str, Any]:
        """准备Jina API请求负载"""
        documents = [{"text": doc.text_content} for doc, _ in request.documents]
        
        payload = {
            "query": request.query,
            "documents": documents,
            "top_k": min(request.top_k, len(documents))
        }
        
        if self.config.model_name:
            payload["model"] = self.config.model_name
        
        return payload
    
    def _parse_response(self, response_data: Dict[str, Any], original_docs: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """解析Jina API响应"""
        results = []
        
        if "results" not in response_data:
            raise ValueError("无效的API响应格式")
        
        doc_map = {i: doc for i, doc in enumerate(original_docs)}
        
        for result in response_data["results"]:
            index = result.get("index", 0)
            relevance_score = result.get("relevance_score", 0.0)
            
            if index < len(original_docs):
                doc, original_score = original_docs[index]
                results.append((doc, relevance_score))
        
        return results
    
    def _get_headers(self) -> Dict[str, str]:
        """Jina专用请求头"""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
    
    async def rerank(self, request: RerankRequest) -> RerankResponse:
        """执行Jina重排序"""
        try:
            payload = self._prepare_payload(request)
            response_data = await self._make_request(payload)
            reranked_docs = self._parse_response(response_data, request.documents)
            
            return RerankResponse(
                results=reranked_docs,
                provider="jina",
                cached=False
            )
            
        except Exception as e:
            logging.error(f"Jina重排序失败: {e}")
            return RerankResponse(
                results=[],
                provider="jina",
                error=str(e)
            )


class CustomAPIClient(BaseAPIClient):
    """自定义API客户端"""
    
    def _prepare_payload(self, request: RerankRequest) -> Dict[str, Any]:
        """准备自定义API请求负载"""
        return {
            "query": request.query,
            "documents": [doc.text_content for doc, _ in request.documents],
            "top_k": min(request.top_k, len(request.documents))
        }
    
    def _parse_response(self, response_data: Dict[str, Any], original_docs: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """解析自定义API响应"""
        # 假设响应格式: {"results": [{"index": 0, "score": 0.9}, ...]}
        results = []
        
        if "results" not in response_data:
            raise ValueError("无效的API响应格式")
        
        for result in response_data["results"]:
            index = result.get("index", 0)
            score = result.get("score", 0.0)
            
            if index < len(original_docs):
                doc, _ = original_docs[index]
                results.append((doc, score))
        
        return results
    
    async def rerank(self, request: RerankRequest) -> RerankResponse:
        """执行自定义API重排序"""
        try:
            payload = self._prepare_payload(request)
            response_data = await self._make_request(payload)
            reranked_docs = self._parse_response(response_data, request.documents)
            
            return RerankResponse(
                results=reranked_docs,
                provider="custom",
                cached=False
            )
            
        except Exception as e:
            logging.error(f"自定义API重排序失败: {e}")
            return RerankResponse(
                results=[],
                provider="custom",
                error=str(e)
            )


class APIClientFactory:
    """API客户端工厂"""
    
    _clients = {
        "cohere": CohereClient,
        "jina": JinaClient,
        "custom": CustomAPIClient
    }
    
    @classmethod
    def create(cls, provider: str, config) -> BaseAPIClient:
        """创建API客户端"""
        if provider not in cls._clients:
            raise ValueError(f"不支持的API提供商: {provider}")
        
        return cls._clients[provider](config)
    
    @classmethod
    def list_providers(cls) -> list:
        """列出支持的提供商"""
        return list(cls._clients.keys())
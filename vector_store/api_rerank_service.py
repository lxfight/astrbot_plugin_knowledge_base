"""
API重排序服务实现
提供基于API的重排序功能，支持缓存和降级策略
"""

import asyncio
import logging
from typing import List, Tuple, Optional, Dict, Any
import time
from functools import lru_cache

from .base import Document
from .api_rerank_config import APIRerankConfig
from .api_clients import APIClientFactory, RerankRequest, RerankResponse


class APIRerankService:
    """API重排序服务"""
    
    def __init__(self, config: Optional[APIRerankConfig] = None):
        self.config = config or APIRerankConfig()
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._initialized = False
        
    async def initialize(self):
        """初始化服务"""
        if self._initialized:
            return
            
        if not self.config.validate():
            logging.warning("API重排序配置无效，将使用降级策略")
            return
            
        self._initialized = True
        logging.info(f"API重排序服务已初始化，提供商: {self.config.provider}")
    
    def _get_cache_key(self, query: str, documents: List[Tuple[Document, float]], top_k: int) -> str:
        """生成缓存键"""
        # 使用查询文本和文档内容生成哈希
        content = f"{query}_{top_k}_"
        for doc, score in documents:
            content += f"{doc.id}_{score}_"
        
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_cache_valid(self, key: str) -> bool:
        """检查缓存是否有效"""
        if key not in self._cache:
            return False
        
        if not self.config.enable_cache:
            return False
        
        timestamp = self._cache_timestamps.get(key, 0)
        return time.time() - timestamp < self.config.cache_ttl
    
    def _get_cached_result(self, key: str) -> Optional[List[Tuple[Document, float]]]:
        """获取缓存结果"""
        if self._is_cache_valid(key):
            logging.debug(f"使用缓存结果: {key}")
            return self._cache[key]
        return None
    
    def _cache_result(self, key: str, result: List[Tuple[Document, float]]):
        """缓存结果"""
        if not self.config.enable_cache:
            return
            
        # 清理过期缓存
        current_time = time.time()
        expired_keys = [
            k for k, ts in self._cache_timestamps.items()
            if current_time - ts > self.config.cache_ttl
        ]
        for key in expired_keys:
            self._cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
        
        # 限制缓存大小
        if len(self._cache) >= self.config.max_cache_size:
            # 移除最旧的缓存
            oldest_key = min(self._cache_timestamps.keys(), key=lambda k: self._cache_timestamps[k])
            self._cache.pop(oldest_key, None)
            self._cache_timestamps.pop(oldest_key, None)
        
        self._cache[key] = result
        self._cache_timestamps[key] = current_time
    
    async def rerank(
        self, 
        query: str, 
        documents: List[Tuple[Document, float]], 
        top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """执行API重排序"""
        if not documents:
            return []
        
        # 限制文档数量
        documents = documents[:min(len(documents), self.config.max_documents)]
        
        # 检查缓存
        cache_key = self._get_cache_key(query, documents, top_k)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result[:top_k]
        
        # 如果API未初始化或配置无效，使用降级策略
        if not self._initialized or not self.config.validate():
            return await self._fallback_rerank(query, documents, top_k)
        
        try:
            # 创建API客户端
            client = APIClientFactory.create(self.config.provider, self.config)
            
            # 准备请求
            request = RerankRequest(
                query=query,
                documents=documents,
                top_k=top_k
            )
            
            # 执行重排序
            async with client:
                response = await client.rerank(request)
            
            if response.error:
                logging.warning(f"API重排序失败: {response.error}")
                if self.config.fallback_strategy == "simple":
                    return await self._fallback_rerank(query, documents, top_k)
                elif self.config.fallback_strategy == "exception":
                    raise Exception(response.error)
                else:
                    return documents[:top_k]
            
            # 缓存结果
            self._cache_result(cache_key, response.results)
            return response.results[:top_k]
            
        except Exception as e:
            logging.error(f"API重排序异常: {e}")
            if self.config.fallback_strategy == "simple":
                return await self._fallback_rerank(query, documents, top_k)
            elif self.config.fallback_strategy == "exception":
                raise e
            else:
                return documents[:top_k]
    
    async def _fallback_rerank(
        self, 
        query: str, 
        documents: List[Tuple[Document, float]], 
        top_k: int
    ) -> List[Tuple[Document, float]]:
        """降级到简单重排序"""
        logging.info("使用简单重排序作为降级策略")
        
        # 简单重排序：基于原始分数和关键词匹配
        reranked = []
        
        # 简单的关键词匹配分数
        query_words = set(query.lower().split())
        
        for doc, original_score in documents:
            # 计算关键词匹配分数
            doc_words = set(doc.text_content.lower().split())
            keyword_score = len(query_words.intersection(doc_words)) / max(len(query_words), 1)
            
            # 综合分数
            combined_score = 0.7 * original_score + 0.3 * keyword_score
            reranked.append((doc, combined_score))
        
        # 按分数排序
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置信息"""
        return {
            "provider": self.config.provider,
            "initialized": self._initialized,
            "cache_enabled": self.config.enable_cache,
            "cache_size": len(self._cache),
            "fallback_strategy": self.config.fallback_strategy
        }
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        self._cache_timestamps.clear()
        logging.info("API重排序缓存已清空")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计"""
        return {
            "cache_size": len(self._cache),
            "max_cache_size": self.config.max_cache_size,
            "cache_ttl": self.config.cache_ttl
        }


class APIRerankManager:
    """API重排序管理器"""
    
    def __init__(self):
        self.services: Dict[str, APIRerankService] = {}
    
    def create_service(self, name: str, config: APIRerankConfig) -> APIRerankService:
        """创建重排序服务"""
        service = APIRerankService(config)
        self.services[name] = service
        return service
    
    async def initialize_all(self):
        """初始化所有服务"""
        for name, service in self.services.items():
            await service.initialize()
            logging.info(f"初始化重排序服务: {name}")
    
    def get_service(self, name: str) -> Optional[APIRerankService]:
        """获取重排序服务"""
        return self.services.get(name)
    
    def list_services(self) -> List[str]:
        """列出所有服务"""
        return list(self.services.keys())
    
    def remove_service(self, name: str):
        """移除重排序服务"""
        if name in self.services:
            del self.services[name]
            logging.info(f"移除重排序服务: {name}")
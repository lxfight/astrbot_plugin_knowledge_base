"""
重排序服务实现
支持Cross-Encoder模型、API服务和多种重排序策略
"""

import asyncio
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import logging

try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers库未安装，Cross-Encoder功能将不可用")

from .base import Document
from .api_rerank_config import APIRerankConfig
from .api_rerank_service import APIRerankService

@dataclass
class RerankConfig:
    """重排序配置"""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 32
    max_length: int = 512
    use_gpu: bool = False
    top_k: int = 100  # 重排序前保留的候选数量

class RerankService:
    """重排序服务"""
    
    def __init__(self, config: Optional[RerankConfig] = None):
        self.config = config or RerankConfig()
        self.model = None
        self._initialized = False
        
    async def initialize(self):
        """初始化重排序模型"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logging.warning("sentence-transformers库不可用，跳过模型初始化")
            return
            
        if self._initialized:
            return
            
        try:
            self.model = CrossEncoder(
                self.config.model_name,
                max_length=self.config.max_length,
                device='cuda' if self.config.use_gpu else 'cpu'
            )
            self._initialized = True
            logging.info(f"重排序模型 {self.config.model_name} 初始化成功")
        except Exception as e:
            logging.error(f"重排序模型初始化失败: {e}")
            self.model = None
    
    async def rerank(
        self, 
        query: str, 
        documents: List[Tuple[Document, float]], 
        top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """重排序文档"""
        if not documents:
            return []
        
        if not self._initialized or self.model is None:
            # 如果模型不可用，返回原始排序
            logging.warning("重排序模型不可用，返回原始排序")
            return documents[:top_k]
        
        # 限制重排序的文档数量
        candidates = documents[:min(len(documents), self.config.top_k)]
        
        # 准备输入数据
        pairs = [(query, doc.text_content) for doc, _ in candidates]
        
        # 批量预测
        try:
            scores = await self._predict_batch(pairs)
            
            # 重新排序
            reranked = list(zip([doc for doc, _ in candidates], scores))
            reranked.sort(key=lambda x: x[1], reverse=True)
            
            return reranked[:top_k]
            
        except Exception as e:
            logging.error(f"重排序失败: {e}")
            return documents[:top_k]
    
    async def _predict_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """批量预测相似度分数"""
        if not self.model:
            return [0.0] * len(pairs)
        
        # 使用线程池执行同步的模型预测
        loop = asyncio.get_event_loop()
        
        def _predict():
            return self.model.predict(
                pairs, 
                batch_size=self.config.batch_size,
                show_progress_bar=False
            ).tolist()
        
        try:
            scores = await loop.run_in_executor(None, _predict)
            return scores
        except Exception as e:
            logging.error(f"批量预测失败: {e}")
            return [0.0] * len(pairs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if not self._initialized or not self.model:
            return {"available": False}
        
        return {
            "available": True,
            "model_name": self.config.model_name,
            "max_length": self.config.max_length,
            "batch_size": self.config.batch_size,
            "use_gpu": self.config.use_gpu
        }

class SimpleReranker:
    """简单的重排序实现（基于规则）"""
    
    def __init__(self):
        self.weights = {
            'vector_score': 0.5,
            'keyword_score': 0.3,
            'recency_score': 0.2
        }
    
    async def rerank(
        self, 
        query: str, 
        documents: List[Tuple[Document, float]], 
        keyword_scores: Optional[List[Tuple[str, float]]] = None,
        top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """基于规则的重排序"""
        if not documents:
            return []
        
        # 构建关键词分数映射
        keyword_map = dict(keyword_scores) if keyword_scores else {}
        
        # 计算综合分数
        reranked = []
        for doc, vector_score in documents:
            # 关键词分数
            keyword_score = keyword_map.get(doc.id, 0.0)
            
            # 时间分数（基于创建时间）
            recency_score = self._calculate_recency_score(doc.metadata)
            
            # 综合分数
            combined_score = (
                self.weights['vector_score'] * vector_score +
                self.weights['keyword_score'] * keyword_score +
                self.weights['recency_score'] * recency_score
            )
            
            reranked.append((doc, combined_score))
        
        # 排序
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]
    
    def _calculate_recency_score(self, metadata: Dict[str, Any]) -> float:
        """计算时间分数"""
        import time
        
        created_at = metadata.get('created_at', 0)
        if not created_at:
            return 0.5
        
        # 时间衰减：越新的文档分数越高
        days_old = (time.time() - created_at) / (24 * 3600)
        return max(0.1, 1.0 - (days_old / 365))  # 一年后衰减到0.1

class HybridReranker:
    """混合重排序器"""
    
    def __init__(self, config: Optional[RerankConfig] = None):
        self.cross_encoder = RerankService(config)
        self.simple_reranker = SimpleReranker()
        self.use_cross_encoder = SENTENCE_TRANSFORMERS_AVAILABLE
    
    async def initialize(self):
        """初始化混合重排序器"""
        if self.use_cross_encoder:
            await self.cross_encoder.initialize()
    
    async def rerank(
        self,
        query: str,
        documents: List[Tuple[Document, float]],
        keyword_scores: Optional[List[Tuple[str, float]]] = None,
        top_k: int = 5,
        use_cross_encoder: bool = None
    ) -> List[Tuple[Document, float]]:
        """混合重排序"""
        if use_cross_encoder is None:
            use_cross_encoder = self.use_cross_encoder
        
        if use_cross_encoder and self.cross_encoder._initialized:
            # 使用Cross-Encoder重排序
            return await self.cross_encoder.rerank(query, documents, top_k)
        else:
            # 使用简单重排序
            return await self.simple_reranker.rerank(
                query, documents, keyword_scores, top_k
            )
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置信息"""
        return {
            "cross_encoder_available": self.use_cross_encoder,
            "cross_encoder_info": self.cross_encoder.get_model_info() if self.use_cross_encoder else None,
            "simple_reranker_weights": self.simple_reranker.weights
        }

# API重排序服务
class APIReranker:
    """API重排序器"""
    
    def __init__(self, config: Optional[APIRerankConfig] = None):
        self.api_service = APIRerankService(config)
    
    async def initialize(self):
        """初始化API重排序器"""
        await self.api_service.initialize()
    
    async def rerank(
        self,
        query: str,
        documents: List[Tuple[Document, float]],
        top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """使用API服务重排序"""
        return await self.api_service.rerank(query, documents, top_k)
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置信息"""
        return self.api_service.get_config()


# 增强的混合重排序器
class EnhancedHybridReranker:
    """增强的混合重排序器，支持API、Cross-Encoder和简单重排序"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        # 本地Cross-Encoder配置
        self.cross_encoder = RerankService(
            RerankConfig(**config.get("cross_encoder", {}))
        )
        
        # API重排序配置
        self.api_reranker = APIReranker(
            APIRerankConfig(**config.get("api", {}))
        )
        
        # 简单重排序
        self.simple_reranker = SimpleReranker()
        
        # 策略配置
        self.strategy = config.get("strategy", "auto")  # auto, api, cross_encoder, simple
        self.use_cross_encoder = SENTENCE_TRANSFORMERS_AVAILABLE
    
    async def initialize(self):
        """初始化所有重排序器"""
        await self.api_reranker.initialize()
        
        if self.use_cross_encoder:
            await self.cross_encoder.initialize()
    
    async def rerank(
        self,
        query: str,
        documents: List[Tuple[Document, float]],
        keyword_scores: Optional[List[Tuple[str, float]]] = None,
        top_k: int = 5,
        strategy: str = None
    ) -> List[Tuple[Document, float]]:
        """增强的混合重排序"""
        if not documents:
            return []
        
        strategy = strategy or self.strategy
        
        if strategy == "api":
            # 优先使用API重排序
            return await self.api_reranker.rerank(query, documents, top_k)
        
        elif strategy == "cross_encoder" and self.use_cross_encoder:
            # 使用本地Cross-Encoder
            return await self.cross_encoder.rerank(query, documents, top_k)
        
        elif strategy == "simple":
            # 使用简单重排序
            return await self.simple_reranker.rerank(
                query, documents, keyword_scores, top_k
            )
        
        else:  # auto
            # 自动选择最佳策略
            api_config = self.api_reranker.api_service.config
            if api_config.validate():
                # API配置有效，优先使用API
                return await self.api_reranker.rerank(query, documents, top_k)
            elif self.use_cross_encoder and self.cross_encoder._initialized:
                # 使用本地Cross-Encoder
                return await self.cross_encoder.rerank(query, documents, top_k)
            else:
                # 降级到简单重排序
                return await self.simple_reranker.rerank(
                    query, documents, keyword_scores, top_k
                )
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置信息"""
        return {
            "strategy": self.strategy,
            "api_config": self.api_reranker.get_config(),
            "cross_encoder_available": self.use_cross_encoder,
            "cross_encoder_info": self.cross_encoder.get_model_info() if self.use_cross_encoder else None,
            "simple_reranker_weights": self.simple_reranker.weights
        }


# 重排序策略工厂（更新版）
class RerankStrategyFactory:
    """重排序策略工厂（支持API重排序）"""
    
    _strategies = {
        'cross_encoder': RerankService,
        'simple': SimpleReranker,
        'hybrid': HybridReranker,
        'api': APIReranker,
        'enhanced_hybrid': EnhancedHybridReranker
    }
    
    @classmethod
    def create(cls, strategy: str, config: Optional[Any] = None):
        """创建重排序策略"""
        if strategy not in cls._strategies:
            raise ValueError(f"不支持的重排序策略: {strategy}")
        
        return cls._strategies[strategy](config)
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """列出所有支持的策略"""
        return list(cls._strategies.keys())

# 异步重排序包装器
class AsyncRerankWrapper:
    """异步重排序包装器"""
    
    def __init__(self, reranker):
        self.reranker = reranker
    
    async def rerank(self, *args, **kwargs):
        """异步重排序"""
        if hasattr(self.reranker, 'initialize'):
            await self.reranker.initialize()
        
        return await self.reranker.rerank(*args, **kwargs)
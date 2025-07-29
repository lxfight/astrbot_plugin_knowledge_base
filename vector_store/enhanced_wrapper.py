"""
向后兼容包装器
将新的增强型存储无缝集成到现有系统中
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import logging

from .base import VectorDBBase, Document
from .enhanced_faiss_store import EnhancedFaissStore
from .keyword_index import KeywordIndex
from .rerank_service import EnhancedHybridReranker
from .api_rerank_config import APIRerankConfig
from .migration_tool import MigrationTool
from ..utils.embedding import EmbeddingSolutionHelper

class EnhancedVectorStore(VectorDBBase):
    """
    增强型向量存储的向后兼容包装器
    自动处理格式检测和迁移
    """
    
    def __init__(self, embedding_util: EmbeddingSolutionHelper, data_path: str,
                 rerank_config: Optional[Dict[str, Any]] = None):
        super().__init__(embedding_util, data_path)
        
        # 初始化组件
        self.enhanced_store = EnhancedFaissStore(embedding_util, data_path)
        self.migration_tool = MigrationTool(data_path)
        self.reranker = EnhancedHybridReranker(rerank_config)
        
        # 配置
        self.auto_migrate = True
        self.use_enhanced_search = True
        
    async def initialize(self):
        """初始化存储"""
        logging.info("初始化增强型向量存储...")
        
        # 初始化重排序器
        await self.reranker.initialize()
        
        # 自动迁移检测
        if self.auto_migrate:
            old_collections = self.migration_tool.detect_old_format()
            if old_collections:
                logging.info(f"检测到 {len(old_collections)} 个旧格式集合，开始自动迁移")
                results = self.migration_tool.migrate_all(self.embedding_util)
                
                for collection, success in results.items():
                    if success:
                        logging.info(f"集合 {collection} 迁移成功")
                    else:
                        logging.error(f"集合 {collection} 迁移失败")
        
        logging.info("增强型向量存储初始化完成")
    
    async def create_collection(self, collection_name: str):
        """创建集合（总是使用新格式）"""
        await self.enhanced_store.create_collection(collection_name)
    
    async def collection_exists(self, collection_name: str) -> bool:
        """检查集合是否存在"""
        return await self.enhanced_store.collection_exists(collection_name)
    
    async def add_documents(
        self, collection_name: str, documents: List[Document]
    ) -> List[str]:
        """添加文档（总是使用新格式）"""
        return await self.enhanced_store.add_documents(collection_name, documents)
    
    async def search(
        self, collection_name: str, query_text: str, top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """搜索文档（支持混合搜索和重排序）"""
        if not self.use_enhanced_search:
            # 回退到基础搜索
            return await self.enhanced_store.search(collection_name, query_text, top_k)
        
        # 混合搜索
        return await self._hybrid_search(collection_name, query_text, top_k)
    
    async def _hybrid_search(
        self, collection_name: str, query_text: str, top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """混合搜索：向量 + 关键词 + 重排序"""
        try:
            # 1. 向量搜索
            vector_results = await self.enhanced_store.search(
                collection_name, query_text, top_k * 3
            )
            
            if not vector_results:
                return []
            
            # 2. 关键词搜索
            keyword_results = await self._keyword_search(collection_name, query_text, top_k * 3)
            
            # 3. 合并结果
            merged_results = self._merge_results(vector_results, keyword_results)
            
            # 4. 重排序
            reranked_results = await self.reranker.rerank(
                query_text, merged_results, top_k=top_k
            )
            
            return reranked_results
            
        except Exception as e:
            logging.error(f"混合搜索失败: {e}")
            # 回退到基础搜索
            return await self.enhanced_store.search(collection_name, query_text, top_k)
    
    async def _keyword_search(
        self, collection_name: str, query_text: str, limit: int = 100
    ) -> List[Tuple[Document, float]]:
        """关键词搜索"""
        try:
            # 使用关键词索引
            keyword_index = KeywordIndex(
                str(self.data_path / f"{collection_name}.enhanced.db")
            )
            
            keyword_results = keyword_index.search_with_bm25(query_text, limit)
            
            # 转换为Document格式
            documents = []
            for doc_id, score in keyword_results:
                # 这里需要从增强存储获取完整文档
                # 简化实现：使用基础搜索获取文档
                pass
            
            return []
            
        except Exception as e:
            logging.error(f"关键词搜索失败: {e}")
            return []
    
    def _merge_results(
        self, 
        vector_results: List[Tuple[Document, float]], 
        keyword_results: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """合并向量搜索和关键词搜索结果"""
        # 使用加权平均合并分数
        merged = {}
        
        # 向量分数
        for doc, score in vector_results:
            merged[doc.id] = {
                'doc': doc,
                'vector_score': score,
                'keyword_score': 0.0
            }
        
        # 关键词分数
        for doc, score in keyword_results:
            if doc.id in merged:
                merged[doc.id]['keyword_score'] = score
            else:
                merged[doc.id] = {
                    'doc': doc,
                    'vector_score': 0.0,
                    'keyword_score': score
                }
        
        # 计算综合分数
        results = []
        for data in merged.values():
            combined_score = (
                0.7 * data['vector_score'] + 
                0.3 * data['keyword_score']
            )
            results.append((data['doc'], combined_score))
        
        # 排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    async def delete_collection(self, collection_name: str) -> bool:
        """删除集合"""
        return await self.enhanced_store.delete_collection(collection_name)
    
    async def list_collections(self) -> List[str]:
        """列出所有集合"""
        return await self.enhanced_store.list_collections()
    
    async def count_documents(self, collection_name: str) -> int:
        """统计文档数量"""
        return await self.enhanced_store.count_documents(collection_name)
    
    async def close(self):
        """关闭存储"""
        await self.enhanced_store.close()
    
    def get_storage_info(self) -> Dict[str, Any]:
        """获取存储信息"""
        return {
            "type": "enhanced_faiss",
            "features": [
                "faiss_vector_index",
                "sqlite_metadata",
                "keyword_index",
                "rerank_support",
                "migration_support"
            ],
            "format": {
                "metadata": ".enhanced.db",
                "vectors": ".faiss"
            }
        }
    
    def get_migration_status(self) -> Dict[str, Any]:
        """获取迁移状态"""
        return self.migration_tool.get_migration_status()
    
    async def trigger_migration(self, force: bool = False) -> Dict[str, bool]:
        """手动触发迁移"""
        return self.migration_tool.migrate_all(self.embedding_util, force)
    
    def cleanup_old_files(self, collection_name: str) -> bool:
        """清理旧文件"""
        return self.migration_tool.cleanup_old_files(collection_name)

# 工厂函数
def create_enhanced_store(
    embedding_util: EmbeddingSolutionHelper, 
    data_path: str,
    use_enhanced: bool = True
) -> VectorDBBase:
    """
    创建增强型存储实例
    
    Args:
        embedding_util: 嵌入工具
        data_path: 数据路径
        use_enhanced: 是否使用增强功能
    
    Returns:
        VectorDBBase: 向量存储实例
    """
    if use_enhanced:
        return EnhancedVectorStore(embedding_util, data_path)
    else:
        # 回退到旧格式
        from .faiss_store import FaissStore
        return FaissStore(embedding_util, data_path)

# 配置管理（更新版）
class EnhancedStoreConfig:
    """增强存储配置（支持API重排序）"""
    
    def __init__(self):
        self.auto_migrate = True
        self.use_enhanced_search = True
        self.rerank_strategy = "auto"  # auto, api, cross_encoder, simple
        self.rerank_config = {
            "strategy": "auto",
            "api": {
                "provider": "cohere",
                "api_key": "",
                "timeout": 30,
                "max_retries": 3,
                "enable_cache": True,
                "cache_ttl": 3600,
                "fallback_strategy": "simple"
            },
            "cross_encoder": {
                "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "batch_size": 32,
                "max_length": 512,
                "use_gpu": False
            }
        }
        self.migration_config = {
            "backup_before_migrate": True,
            "validate_after_migrate": True,
            "cleanup_old_files": False
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "auto_migrate": self.auto_migrate,
            "use_enhanced_search": self.use_enhanced_search,
            "rerank_strategy": self.rerank_strategy,
            "rerank_config": self.rerank_config,
            "migration_config": self.migration_config
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EnhancedStoreConfig':
        """从字典创建配置"""
        config = cls()
        config.auto_migrate = config_dict.get("auto_migrate", True)
        config.use_enhanced_search = config_dict.get("use_enhanced_search", True)
        config.rerank_strategy = config_dict.get("rerank_strategy", "auto")
        
        if "rerank_config" in config_dict:
            config.rerank_config.update(config_dict["rerank_config"])
        
        if "migration_config" in config_dict:
            config.migration_config.update(config_dict["migration_config"])
        
        return config
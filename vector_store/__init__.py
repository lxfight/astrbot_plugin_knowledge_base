"""
向量存储模块
提供多种向量存储实现和重排序功能
"""

# 基础组件
from .base import VectorDBBase, Document

# 存储实现
from .faiss_store import FaissStore
from .enhanced_faiss_store import EnhancedFaissStore
from .milvus_store import MilvusStore
from .milvus_lite_store import MilvusLiteStore

# 重排序功能
from .rerank_service import (
    RerankService,
    SimpleReranker,
    HybridReranker,
    APIReranker,
    EnhancedHybridReranker,
    RerankStrategyFactory
)

# API重排序
from .api_rerank_config import APIRerankConfig
from .api_rerank_service import APIRerankService
from .api_clients import APIClientFactory

# 配置和示例
from .config_examples import CONFIG_EXAMPLES, get_config_example
from .config_adapter import (
    create_rerank_config_from_astrbot,
    create_reranker_from_astrbot_config,
    get_api_rerank_config_from_astrbot,
    validate_rerank_config,
    get_rerank_config_summary
)

# 增强包装器
from .enhanced_wrapper import EnhancedVectorStore, EnhancedStoreConfig

# 迁移工具
from .migration_tool import MigrationTool

__all__ = [
    # 基础组件
    'VectorDBBase',
    'Document',
    
    # 存储实现
    'FaissStore',
    'EnhancedFaissStore',
    'MilvusStore',
    'MilvusLiteStore',
    
    # 重排序功能
    'RerankService',
    'SimpleReranker',
    'HybridReranker',
    'APIReranker',
    'EnhancedHybridReranker',
    'RerankStrategyFactory',
    
    # API重排序
    'APIRerankConfig',
    'APIRerankService',
    'APIClientFactory',
    
    # 配置和示例
    'CONFIG_EXAMPLES',
    'get_config_example',
    
    # 配置适配器
    'create_rerank_config_from_astrbot',
    'create_reranker_from_astrbot_config',
    'get_api_rerank_config_from_astrbot',
    'validate_rerank_config',
    'get_rerank_config_summary',
    
    # 增强包装器
    'EnhancedVectorStore',
    'EnhancedStoreConfig',
    
    # 迁移工具
    'MigrationTool'
]

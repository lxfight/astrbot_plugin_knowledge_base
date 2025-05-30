from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from ..utils.embedding import EmbeddingSolutionHelper

DEFAULT_BATCH_SIZE = 10  # 默认批处理大小
MAX_RETRIES = 3  # 最大重试次数

@dataclass
class Document:
    text_content: str
    embedding: Optional[List[float]] = None  # 向量数据，添加时生成，查询时不需要
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )  # 例如: {'source': 'file.txt', 'chunk_id': 0}
    id: Optional[str] = None  # 文档在向量数据库中的唯一 ID

@dataclass
class ProcessingBatch:
    documents: List[Document]
    retry_count: int = 0  # 记录当前批次的重试次数


class VectorDBBase(ABC):
    def __init__(self, embedding_util: EmbeddingSolutionHelper, data_path: str):
        self.embedding_util = embedding_util  # EmbeddingUtil 实例
        self.data_path = data_path  # 数据存储路径 (主要用于 Faiss/Milvus Lite)

    @abstractmethod
    async def initialize(self):
        """初始化数据库连接和集合等"""
        pass

    @abstractmethod
    async def add_documents(
        self, collection_name: str, documents: List[Document]
    ) -> List[str]:
        """向指定集合添加文档，返回文档ID列表"""
        pass

    @abstractmethod
    async def search(
        self, collection_name: str, query_text: str, top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """在指定集合中搜索相关文档，返回 (文档, 相似度得分) 列表"""
        pass

    @abstractmethod
    async def create_collection(self, collection_name: str):
        """创建集合"""
        pass

    @abstractmethod
    async def delete_collection(self, collection_name: str) -> bool:
        """删除集合及其所有数据"""
        pass

    @abstractmethod
    async def list_collections(self) -> List[str]:
        """列出所有集合"""
        pass

    @abstractmethod
    async def count_documents(self, collection_name: str) -> int:
        """计算集合中的文档数量"""
        pass

    @abstractmethod
    async def collection_exists(self, collection_name: str) -> bool:
        """检查集合是否存在"""
        pass

    @abstractmethod
    async def close(self):
        """关闭数据库连接"""
        pass

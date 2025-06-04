import os
import asyncio
import json
from typing import List, Dict, Tuple
from .base import (
    VectorDBBase,
    Document,
    ProcessingBatch,
    DEFAULT_BATCH_SIZE,
)
from astrbot.api import logger
from astrbot.core.db.vec_db.faiss_impl import FaissVecDB
from astrbot.core.provider.provider import EmbeddingProvider
from ..utils.embedding import EmbeddingSolutionHelper
from .faiss_store import FaissStore as OldFaissStore


def _check_pickle_file(file_path: str) -> bool:
    """检查文件是否为 Pickle 格式"""
    try:
        with open(file_path, "rb") as f:
            magic = f.read(2)
            return magic == b"\x80\x04"
    except Exception:
        return False


class AstrBotEmbeddingProviderWrapper(EmbeddingProvider):
    """AstrBot Embedding Provider 包装类"""

    def __init__(
        self,
        embedding_util: EmbeddingSolutionHelper,
        collection_name: str,
    ):
        self.embedding_util = embedding_util
        self.collection_name = collection_name

    async def get_embedding(self, text: str) -> List[float]:
        vec = await self.embedding_util.get_embedding_async(text, self.collection_name)
        if not vec:
            raise ValueError(
                "获取向量失败，返回的向量为空或无效。请检查输入文本和配置。"
            )
        return vec

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本的嵌入"""
        vecs = await self.embedding_util.get_embeddings_async(
            texts, self.collection_name
        )
        if not vecs:
            raise ValueError(
                "获取向量失败，返回的向量为空或无效。请检查输入文本和配置。"
            )
        return vecs

    def get_dim(self) -> int:
        return self.embedding_util.get_dimensions(self.collection_name)


class FaissStore(VectorDBBase):
    """对 AstrBot FaissVecDB 的包装类，以适应 KB 的接口规范"""

    def __init__(self, embedding_util: EmbeddingSolutionHelper, data_path: str):
        super().__init__(embedding_util, data_path)
        self.vecdbs: Dict[str, FaissVecDB] = {}
        self._old_faiss_store: OldFaissStore = None
        self._old_collections = {}
        self.embedding_utils: Dict[str, AstrBotEmbeddingProviderWrapper] = {}
        self.filename_map: dict = {} # filename -> collection_name 的映射
        os.makedirs(self.data_path, exist_ok=True)

    async def initialize(self):
        logger.info("初始化 Faiss 存储...")
        await self._load_all_collections()
        logger.info(f"Faiss 存储初始化完成。已加载集合: {list(self.vecdbs.keys())}")

    async def _load_collection(self, collection_name: str):
        logger.info(f"加载 Faiss 集合: {collection_name}")
        true_coll_name = self.embedding_util.user_prefs_handler.get_collection_name_by_file_id(collection_name)
        if not true_coll_name:
            true_coll_name = collection_name
        index_path = os.path.join(self.data_path, f"{collection_name}.index")
        storage_path = os.path.join(self.data_path, f"{collection_name}.db")

        _old_storage_path = os.path.join(self.data_path, f"{collection_name}.docs")
        if _check_pickle_file(storage_path) or os.path.exists(_old_storage_path):
            # old Faiss store format
            self._old_collections[collection_name] = collection_name
            if not self._old_faiss_store:
                self._old_faiss_store = OldFaissStore(
                    self.embedding_util, self.data_path
                )
                await self._old_faiss_store.initialize()
            return

        try:
            self.embedding_utils[collection_name] = AstrBotEmbeddingProviderWrapper(
                embedding_util=self.embedding_util,
                collection_name=collection_name,
            )
            self.vecdbs[collection_name] = FaissVecDB(
                doc_store_path=storage_path,
                index_store_path=index_path,
                embedding_provider=self.embedding_utils[collection_name],
            )
            self.filename_map[collection_name] = true_coll_name # 记录文件名和集合名的映射
            await self.vecdbs[collection_name].initialize()
        except Exception as e:
            logger.error(f"加载知识库集合(FAISS) '{collection_name}' 时出错: {e}")
            return

    async def _load_all_collections(self):
        for filename in os.listdir(self.data_path):
            if filename.endswith(".index"):
                collection_name = filename[: -len(".index")]
                await self._load_collection(collection_name)

    async def create_collection(self, collection_name: str):
        if await self.collection_exists(collection_name):
            logger.info(f"Faiss 集合 '{collection_name}' 已存在。")
            return
        collection_md = (
            self.embedding_util.user_prefs_handler.user_collection_preferences.get(
                "collection_metadata", {}
            ).get(collection_name, {})
        )
        file_id = collection_md.get("file_id", collection_name)
        await self.embedding_util.user_prefs_handler.save_user_preferences()
        index_path = os.path.join(self.data_path, f"{file_id}.index")
        storage_path = os.path.join(self.data_path, f"{file_id}.db")
        self.embedding_utils[collection_name] = AstrBotEmbeddingProviderWrapper(
            embedding_util=self.embedding_util,
            collection_name=collection_name,
        )
        self.vecdbs[collection_name] = FaissVecDB(
            doc_store_path=storage_path,
            index_store_path=index_path,
            embedding_provider=self.embedding_utils[collection_name],
        )
        self.filename_map[file_id] = collection_name  # 记录文件名和集合名的映射
        await self.vecdbs[collection_name].initialize()
        await self.vecdbs[collection_name].embedding_storage.save_index()
        logger.info(f"Faiss 集合 '{collection_name}' 创建成功。")

    async def collection_exists(self, collection_name: str) -> bool:
        return (
            collection_name in self.vecdbs or collection_name in self._old_collections
        )

    async def _batch_process_task(
        self, batch: ProcessingBatch, collection_name: str
    ) -> List[str]:
        """处理单个批次的任务"""
        all_doc_ids = []
        retry_cnt = 3
        for doc in batch.documents:
            while retry_cnt > 0:
                try:
                    id = await self.vecdbs[collection_name].insert(
                        content=doc.text_content,
                        metadata=doc.metadata,
                    )
                    all_doc_ids.append(id)
                    break
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    logger.error(
                        f"向 Faiss 集合 '{collection_name}' 添加文档时发生异常: {e}",
                        stack_info=True,
                    )
                    retry_cnt -= 1
                    if retry_cnt == 0:
                        excerpt = doc.text_content[:100].replace("\n", "")
                        logger.error(f"批次添加失败，文档 {excerpt}... 将被丢弃。")
                    await asyncio.sleep(1)
        return all_doc_ids

    async def add_documents(
        self, collection_name: str, documents: List[Document]
    ) -> List[str]:
        if collection_name in self._old_collections:
            return await self._old_faiss_store.add_documents(collection_name, documents)

        if not await self.collection_exists(collection_name):
            logger.warning(f"Faiss 集合 '{collection_name}' 不存在。将尝试自动创建。")
            await self.create_collection(collection_name)

        all_doc_ids: List[str] = []

        num_batches = 0
        tasks = []
        failed_batches_cnt = 0
        for i in range(0, len(documents), DEFAULT_BATCH_SIZE):
            batch_docs = documents[i : i + DEFAULT_BATCH_SIZE]
            processing_batch = ProcessingBatch(documents=batch_docs)
            tasks.append(self._batch_process_task(processing_batch, collection_name))
            num_batches += 1
        logger.info(
            f"向 Faiss 集合 '{collection_name}' 添加 {len(documents)} 个文档，共分为 {num_batches} 个批次进行处理。"
        )

        try:
            results = await asyncio.gather(*tasks)
            for batch_result in results:
                if not batch_result:
                    failed_batches_cnt += 1
                    continue
                all_doc_ids.extend(batch_result)
        except Exception as e:
            logger.error(f"处理批次时发生异常: {e}")
            return []

        logger.info(
            f"向 Faiss 集合 '{collection_name}' 完成添加操作。总共处理了 {len(documents)} 个原始文档，成功添加 {len(all_doc_ids)} 个文档。"
        )
        logger.info(f"其中 {failed_batches_cnt} 个批次因重试失败被丢弃。")
        return all_doc_ids

    async def search(
        self, collection_name: str, query_text: str, top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        if not await self.collection_exists(collection_name):
            logger.warning(f"Faiss 集合 '{collection_name}' 不存在。")
            return []

        if collection_name in self._old_collections:
            return await self._old_faiss_store.search(
                collection_name, query_text, top_k
            )

        results = await self.vecdbs[collection_name].retrieve(
            query=query_text,
            k=top_k,
        )
        ret = []
        for result in results:
            if result is not None:
                doc = Document(
                    id=result.data["doc_id"],
                    embedding=[],
                    text_content=result.data["text"],
                    metadata=json.loads(result.data["metadata"]),
                )
                ret.append((doc, result.similarity))
        logger.info(
            f"Faiss 集合 '{collection_name}' 搜索完成。查询文本: '{query_text}'，返回 {len(ret)} 个结果。"
        )
        return ret

    async def delete_collection(self, collection_name: str) -> bool:
        if not await self.collection_exists(collection_name):
            logger.info(f"Faiss 集合 '{collection_name}' 不存在，无需删除。")
            return False

        if collection_name in self._old_collections:
            return await self._old_faiss_store.delete_collection(collection_name)

        def _delete_sync():
            self.vecdbs.pop(collection_name, None)  # 从内存中删除集合

            collection_md = (
                self.embedding_util.user_prefs_handler.user_collection_preferences.get(
                    "collection_metadata", {}
                ).get(collection_name, {})
            )
            file_id = collection_md.get("file_id", collection_name)

            index_path = os.path.join(self.data_path, f"{file_id}.index")
            storage_path = os.path.join(self.data_path, f"{file_id}.db")

            try:
                if os.path.exists(index_path):
                    os.remove(index_path)
                if os.path.exists(storage_path):
                    os.remove(storage_path)
                logger.info(f"Faiss 集合 '{collection_name}' 已删除。")
                return True
            except Exception as e:
                logger.error(f"删除 Faiss 集合 '{collection_name}' 文件时出错: {e}")
                return False

        return await asyncio.to_thread(_delete_sync)

    async def list_collections(self) -> List[str]:
        return list(self.filename_map.values()) + list(self._old_collections.keys())

    async def count_documents(self, collection_name: str) -> int:
        if not await self.collection_exists(collection_name):
            return 0
        if collection_name in self._old_collections:
            return await self._old_faiss_store.count_documents(collection_name)
        cnt = await self.vecdbs[collection_name].count_documents()
        return cnt

    async def close(self):
        for collection_name, vecdb in self.vecdbs.items():
            await vecdb.close()
            logger.info(f"Faiss 集合 '{collection_name}' 已关闭。")
        self.vecdbs.clear()
        logger.info("所有 Faiss 集合已关闭。")

        if self._old_faiss_store:
            await self._old_faiss_store.close()

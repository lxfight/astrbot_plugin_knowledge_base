import faiss
import numpy as np
import os
import pickle
import gc  # 添加垃圾回收模块
from typing import List, Dict, Tuple
from .base import (
    VectorDBBase,
    Document,
    ProcessingBatch,
    DEFAULT_BATCH_SIZE,
    MAX_RETRIES,
)
from astrbot.api import logger
from ..utils.embedding import EmbeddingSolutionHelper
import asyncio

# Faiss 是同步库，通过 asyncio.to_thread 包装其操作以适应异步环境


def _check_pickle_file(file_path: str) -> bool:
    """检查文件是否为 Pickle 格式"""
    try:
        with open(file_path, "rb") as f:
            magic = f.read(2)
            return magic == b"\x80\x04"
    except Exception:
        return False


class FaissStore(VectorDBBase):
    def __init__(self, embedding_util: EmbeddingSolutionHelper, data_path: str):
        super().__init__(embedding_util, data_path)
        self.indexes: Dict[str, faiss.Index] = {}
        self.db: Dict[str, List[Document]] = {}  # 存储原始 Document 对象，现在命名为 db
        os.makedirs(self.data_path, exist_ok=True)

    async def initialize(self):
        logger.info("初始化 Faiss 存储...")
        await asyncio.to_thread(self._load_all_collections)
        logger.info(f"Faiss 存储初始化完成。已加载集合: {list(self.indexes.keys())}")

    def _load_collection(self, collection_name: str):
        index_path = os.path.join(self.data_path, f"{collection_name}.index")
        storage_path = os.path.join(
            self.data_path, f"{collection_name}.db"
        )  # 后缀由 .docs 改为 .db

        if not _check_pickle_file(storage_path):
            logger.info(
                f"{collection_name} 对应的存储文件 {storage_path} 不是有效的 Pickle 文件，将跳过加载。"
            )
            return

        if os.path.exists(index_path) and os.path.exists(storage_path):
            try:
                self.indexes[collection_name] = faiss.read_index(index_path)
                with open(storage_path, "rb") as f:
                    self.db[collection_name] = pickle.load(f)
                logger.info(f"成功加载 Faiss 集合: {collection_name}")
            except Exception as e:
                logger.error(f"加载 Faiss 集合 {collection_name} 失败: {e}")
                # 如果加载失败，确保清理不完整的状态
                if collection_name in self.indexes:
                    del self.indexes[collection_name]
                if collection_name in self.db:
                    del self.db[collection_name]
        else:
            logger.info(f"Faiss 集合 {collection_name} 的文件不存在，将不会加载。")

    def _load_all_collections(self):
        for filename in os.listdir(self.data_path):
            if filename.endswith(".index"):
                collection_name = filename[: -len(".index")]
                self._load_collection(collection_name)

    def _save_collection(self, collection_name: str):
        if collection_name in self.indexes and collection_name in self.db:
            index_path = os.path.join(self.data_path, f"{collection_name}.index")
            storage_path = os.path.join(
                self.data_path, f"{collection_name}.db"
            )  # 后缀由 .docs 改为 .db
            try:
                faiss.write_index(self.indexes[collection_name], index_path)
                with open(storage_path, "wb") as f:
                    pickle.dump(self.db[collection_name], f)
                logger.info(f"成功保存 Faiss 集合: {collection_name}")
            except Exception as e:
                logger.error(f"保存 Faiss 集合 {collection_name} 失败: {e}")

    async def create_collection(self, collection_name: str):
        if await self.collection_exists(collection_name):
            logger.info(f"Faiss 集合 '{collection_name}' 已存在。")
            return

        def _create_sync():
            self.indexes[collection_name] = faiss.IndexFlatL2(
                self.embedding_util.get_dimensions(collection_name)
            )  # L2 距离
            self.db[collection_name] = []
            self._save_collection(collection_name)
            logger.info(f"Faiss 集合 '{collection_name}' 创建成功。")

        await asyncio.to_thread(_create_sync)

    async def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.indexes

    async def add_documents(
        self, collection_name: str, documents: List[Document]
    ) -> List[str]:
        if not await self.collection_exists(collection_name):
            logger.warning(f"Faiss 集合 '{collection_name}' 不存在。将尝试自动创建。")
            await self.create_collection(collection_name)

        all_doc_ids: List[str] = []

        # 获取 Faiss 索引和 db 的引用
        faiss_index = self.indexes[collection_name]
        db_storage = self.db[collection_name]

        # 创建一个异步队列来存放待处理的批次
        processing_queue: asyncio.Queue[ProcessingBatch] = asyncio.Queue()

        # 1. 生产者：将所有文档按批次放入队列
        num_batches = 0
        for i in range(0, len(documents), DEFAULT_BATCH_SIZE):
            batch_docs = documents[i : i + DEFAULT_BATCH_SIZE]
            await processing_queue.put(ProcessingBatch(documents=batch_docs))
            num_batches += 1
        logger.info(f"已将 {len(documents)} 份文档分成 {num_batches} 个批次放入队列。")

        # 2. 消费者：从队列中取出批次进行处理
        processed_batches_count = 0
        failed_batches_discarded_count = 0

        while processed_batches_count < num_batches + failed_batches_discarded_count:
            try:
                processing_batch = await asyncio.wait_for(
                    processing_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                if processing_queue.empty():
                    break
                else:
                    continue

            current_docs_in_batch = processing_batch.documents
            current_retry_count = processing_batch.retry_count

            log_prefix = f"[批次 ({processed_batches_count} docs), 重试 {current_retry_count}/{MAX_RETRIES}]"
            logger.debug(f"{log_prefix} 正在处理...")

            try:
                current_batch_texts_to_embed = []
                docs_needing_embedding_in_batch = []

                for doc in current_docs_in_batch:
                    if doc.embedding is None:
                        current_batch_texts_to_embed.append(doc.text_content)
                        docs_needing_embedding_in_batch.append(doc)

                batch_embeddings_generated: List[List[float]] = []
                if current_batch_texts_to_embed:
                    batch_embeddings_generated = (
                        await self.embedding_util.get_embeddings_async(
                            current_batch_texts_to_embed, collection_name
                        )
                    )
                    logger.debug(
                        f"{log_prefix} 成功为 {len(batch_embeddings_generated)} 个文本生成了嵌入。"
                    )

                valid_embeddings_for_batch: List[List[float]] = []
                processed_documents_for_batch: List[Document] = []

                embed_idx = 0
                for doc in current_docs_in_batch:
                    if doc.embedding is None:
                        if (
                            embed_idx < len(batch_embeddings_generated)
                            and batch_embeddings_generated[embed_idx] is not None
                        ):
                            doc.embedding = batch_embeddings_generated[embed_idx]
                            valid_embeddings_for_batch.append(doc.embedding)
                            processed_documents_for_batch.append(doc)
                        else:
                            logger.warning(
                                f"{log_prefix} 未能为文档 '{doc.text_content[:50]}...' 生成 embedding，将跳过。"
                            )
                        embed_idx += 1
                    else:
                        valid_embeddings_for_batch.append(doc.embedding)
                        processed_documents_for_batch.append(doc)

                if not valid_embeddings_for_batch:
                    logger.debug(
                        f"{log_prefix} 没有有效的 embedding 可供添加，跳过此批次。"
                    )
                    processed_batches_count += 1
                    processing_queue.task_done()
                    continue

                def _add_batch_sync(
                    batch_embeds: List[List[float]], batch_proc_docs: List[Document]
                ):
                    nonlocal faiss_index, db_storage

                    new_embeddings_np = np.array(batch_embeds).astype("float32")
                    faiss.normalize_L2(new_embeddings_np)

                    start_id = faiss_index.ntotal
                    faiss_index.add(new_embeddings_np)

                    current_batch_ids = []
                    for j, doc in enumerate(batch_proc_docs):
                        doc_id = str(start_id + j)
                        doc.id = doc_id
                        db_storage.append(doc)
                        current_batch_ids.append(doc_id)
                    return current_batch_ids

                batch_added_ids = await asyncio.to_thread(
                    _add_batch_sync,
                    valid_embeddings_for_batch,
                    processed_documents_for_batch,
                )
                all_doc_ids.extend(batch_added_ids)
                logger.debug(f"{log_prefix} 成功添加了 {len(batch_added_ids)} 个文档。")

                # 内存优化：及时清理批次数据
                del current_batch_texts_to_embed
                del docs_needing_embedding_in_batch
                del batch_embeddings_generated
                del valid_embeddings_for_batch
                del processed_documents_for_batch
                del current_docs_in_batch

                # 每10个批次触发一次垃圾回收
                if processed_batches_count % 10 == 0:
                    gc.collect()
                    logger.debug(f"已处理 {processed_batches_count} 个批次，触发垃圾回收")

                processed_batches_count += 1
                processing_queue.task_done()

            except Exception as e:
                logger.error(f"{log_prefix} 处理失败: {e}")

                if current_retry_count < MAX_RETRIES:
                    processing_batch.retry_count += 1
                    await processing_queue.put(processing_batch)
                    logger.warning(
                        f"{log_prefix} 将批次重新放入队列进行重试 (当前重试次数: {processing_batch.retry_count})。"
                    )
                else:
                    logger.error(
                        f"{log_prefix} 批次达到最大重试次数 ({MAX_RETRIES})，将丢弃此批次。"
                    )
                    failed_batches_discarded_count += 1

                processed_batches_count += 1
                processing_queue.task_done()

            finally:
                del current_batch_texts_to_embed
                del docs_needing_embedding_in_batch
                del batch_embeddings_generated
                del valid_embeddings_for_batch
                del processed_documents_for_batch
                del current_docs_in_batch
                del processing_batch

        await processing_queue.join()

        self._save_collection(collection_name)

        logger.info(
            f"向 Faiss 集合 '{collection_name}' 完成添加操作。总共处理了 {len(documents)} 个原始文档，成功添加 {len(all_doc_ids)} 个文档。"
        )
        logger.info(f"其中 {failed_batches_discarded_count} 个批次因重试失败被丢弃。")
        return all_doc_ids

    async def search(
        self, collection_name: str, query_text: str, top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        if not await self.collection_exists(collection_name):
            logger.warning(f"Faiss 集合 '{collection_name}' 不存在。")
            return []

        query_embedding = await self.embedding_util.get_embedding_async(
            query_text, collection_name
        )
        if query_embedding is None:
            logger.error("无法为查询文本生成 embedding。")
            return []

        def _search_sync():
            index = self.indexes[collection_name]
            storage = self.db[collection_name]

            if index.ntotal == 0:
                logger.info(f"Faiss 集合 '{collection_name}' 为空，无法搜索。")
                return []

            query_embedding_np = np.array([query_embedding]).astype("float32")
            faiss.normalize_L2(query_embedding_np)

            actual_top_k = min(top_k, index.ntotal)
            if actual_top_k == 0:
                return []

            distances, indices = index.search(query_embedding_np, actual_top_k)

            results = []
            for i in range(len(indices[0])):
                doc_index = indices[0][i]
                dist = distances[0][i]
                if 0 <= doc_index < len(storage):
                    similarity_score = 1.0 - (dist / 2.0)
                    results.append((storage[doc_index], float(similarity_score)))
                else:
                    logger.warning(f"搜索结果中的索引 {doc_index} 超出范围。")
            return results

        return await asyncio.to_thread(_search_sync)

    async def delete_collection(self, collection_name: str) -> bool:
        if not await self.collection_exists(collection_name):
            logger.info(f"Faiss 集合 '{collection_name}' 不存在，无需删除。")
            return False

        def _delete_sync():
            del self.indexes[collection_name]
            del self.db[collection_name]

            index_path = os.path.join(self.data_path, f"{collection_name}.index")
            storage_path = os.path.join(self.data_path, f"{collection_name}.db")
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
        return await asyncio.to_thread(lambda: list(self.indexes.keys()))

    async def count_documents(self, collection_name: str) -> int:
        if not await self.collection_exists(collection_name):
            return 0
        return await asyncio.to_thread(lambda: self.indexes[collection_name].ntotal)

    async def close(self):
        logger.info("关闭 Faiss 存储 (实际上是保存所有集合)...")
        logger.info("Faiss 存储已处理关闭请求。")

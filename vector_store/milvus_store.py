# -*- coding: utf-8 -*-
# vector_store/milvus_store.py
from typing import List, Optional, Tuple, Dict, Any
from urllib.parse import urlparse
from .base import (
    VectorDBBase,
    Document,
    ProcessingBatch,
    DEFAULT_BATCH_SIZE,
    MAX_RETRIES,
)
from astrbot.api import logger
from pymilvus import (
    connections,
    utility,
    Collection,
    DataType,
    FieldSchema,
    CollectionSchema,
    Index,
)
from pymilvus.exceptions import ConnectionConfigException, MilvusException
import asyncio

try:
    from pymilvus import LoadState
except ImportError:
    LoadState = None


class MilvusStore(VectorDBBase):
    def __init__(
        self,
        embedding_util,
        dimension: int,
        data_path: str,  # data_path for consistency, not used by remote
        host: str,
        port: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            embedding_util, dimension, data_path
        )  # data_path 传给基类，但此类不用
        self.kwargs = kwargs
        # 可以为加载等待设置默认值，如果kwargs中没有提供
        self.kwargs.setdefault("load_wait_timeout", 60)  # 默认等待60秒
        self.kwargs.setdefault(
            "load_wait_loops", 12
        )  # 默认轮询12次 (配合5秒延时即60秒)
        self.kwargs.setdefault("load_loop_delay", 5)  # 轮询间隔
        # 处理 host 参数，移除协议头
        parsed_uri = urlparse(host)
        if parsed_uri.scheme and parsed_uri.netloc:  # 如果 host 看起来像一个完整的 URL
            self.host = parsed_uri.hostname  # 获取主机名部分 (IP 或域名)
            if self.host is None:  # 万一 hostname 解析为 None
                logger.warning(f"无法从 '{host}' 解析主机名，将尝试直接使用。")
                self.host = host  # 回退
            else:
                logger.info(f"从输入 '{host}' 解析得到主机名: '{self.host}'")
            # 如果原始 host 包含端口，且与传入的 port 参数不同，可能需要决策或警告
            if parsed_uri.port and str(parsed_uri.port) != str(port):
                logger.warning(
                    f"输入 host '{host}' 中包含端口 {parsed_uri.port}，"
                    f"但连接将使用显式参数 port='{port}'。"
                )
        elif "://" in host:  # 简单的检查，以防 urlparse 未正确解析但仍有协议头
            temp_host = host.split("://")[-1]
            temp_host = temp_host.split("/")[0]  # 移除路径
            self.host = temp_host.split(":")[0]  # 移除端口
            logger.info(f"从输入 '{host}' 通过简单分割得到主机名: '{self.host}'")
        else:
            self.host = host  # 假设 host 已经是纯主机名或 IP

        self.port = str(port)
        self.user = user
        self.password = password
        # 更新 alias 生成，确保使用处理后的 self.host，并移除可能存在的协议部分
        clean_host_for_alias = self.host.replace("http://", "").replace("https://", "")
        self.alias = f"milvus_remote_{clean_host_for_alias.replace('.', '_').replace(':', '_')}_{self.port}"

        self.metric_type = kwargs.get("metric_type", "L2")
        self.index_type = kwargs.get("index_type", "AUTOINDEX")  # 推荐使用 AUTOINDEX
        self.index_params_config = kwargs.get(
            "index_params", {"nlist": 128} if self.index_type == "IVF_FLAT" else {}
        )
        self.varchar_max_length = kwargs.get("varchar_max_length", 65535)
        self.pk_max_length = kwargs.get("pk_max_length", 36)
        self.consistency_level_str = kwargs.get(
            "consistency_level", "Strong"
        )  # 存储字符串
        self.kwargs = kwargs  # 存储 kwargs 以便后续使用，例如 auto_create_collection

        self._is_connected = False

    async def _attempt_connect(self):
        """尝试连接到 Milvus 服务。"""
        logger.info(
            f"尝试连接到远程 Milvus 服务: {self.host}:{self.port} (alias: {self.alias})..."
        )
        try:
            # 检查是否已存在同名 alias 的连接，如果存在且参数不同，先移除
            existing_connections = connections.list_connections()
            for (
                conn_alias,
                _,
            ) in existing_connections:  # conn_details_str 通常不直接比较
                if conn_alias == self.alias:
                    logger.warning(
                        f"发现已存在同名连接别名 '{self.alias}'，将尝试断开并重新连接。"
                    )
                    try:
                        connections.disconnect(self.alias)
                    except Exception as dis_e:
                        logger.warning(
                            f"断开旧连接 '{self.alias}' 失败: {dis_e}, 继续尝试连接。"
                        )
                    break

            connections.connect(
                alias=self.alias,
                host=self.host,
                port=self.port,
                user=self.user if self.user else None,
                password=self.password if self.password else None,
                # secure=self.kwargs.get("secure", False), # 从 kwargs 获取 TLS/SSL 配置
                # server_name=self.kwargs.get("server_name", None), # for SNI in TLS
                # timeout=self.kwargs.get("connect_timeout", 10.0) # 连接超时
            )
            self._is_connected = True
            logger.info(
                f"成功连接到远程 Milvus 服务: {self.host}:{self.port} (alias: {self.alias})"
            )
        except ConnectionConfigException as e:  # 捕获特定配置错误
            self._is_connected = False
            logger.error(
                f"连接远程 Milvus 服务 (alias: {self.alias}) 配置错误: {e}",
                exc_info=True,
            )
            raise
        except MilvusException as e:  # 捕获 Milvus 操作错误
            self._is_connected = False
            logger.error(
                f"连接远程 Milvus 服务 (alias: {self.alias}) 时发生 Milvus 错误: {e}",
                exc_info=True,
            )
            raise
        except Exception as e:  # 其他错误
            self._is_connected = False
            logger.error(
                f"连接远程 Milvus 服务 (alias: {self.alias}) 失败 (未知错误): {e}",
                exc_info=True,
            )
            raise

    async def initialize(self):
        if self._is_connected:
            # 可以选择性地ping一下服务器确认连接仍然有效
            try:
                utility.get_server_version(using=self.alias)
                logger.info(
                    f"已连接到远程 Milvus 服务: {self.host}:{self.port} (alias: {self.alias})"
                )
                return
            except Exception as e_ping:
                logger.warning(
                    f"与 Milvus 服务 (alias: {self.alias}) 的连接可能已失效: {e_ping}。尝试重新连接。"
                )
                self._is_connected = False  # 标记为未连接，以便重新尝试

        await self._attempt_connect()

    async def _get_collection_with_retry(
        self, collection_name: str
    ) -> Optional[Collection]:
        if not self._is_connected:
            logger.warning(
                f"获取集合 '{collection_name}' 时 Milvus 服务未连接，尝试重新初始化。"
            )
            try:
                await self.initialize()
            except Exception as e_reconnect:
                logger.error(f"获取集合时重新连接 Milvus 服务失败: {e_reconnect}")
                return None

        if not self._is_connected:  # 如果 initialize 后仍未连接
            logger.error("获取集合时 Milvus 服务仍未连接。")
            return None

        try:
            # 先检查集合是否存在，避免直接构造不存在的 Collection 对象导致异常
            if not await self.collection_exists(
                collection_name
            ):  # collection_exists 内部也会处理连接
                logger.warning(
                    f"尝试获取不存在的远程 Milvus 集合 '{collection_name}'。"
                )
                return None
            return Collection(
                collection_name,
                using=self.alias,
                consistency_level=self.consistency_level_str,
            )
        except MilvusException as e:
            if (
                "collection not found" in str(e).lower()
            ):  # 特别处理集合不存在的 MilvusException
                logger.warning(
                    f"获取远程 Milvus 集合 '{collection_name}' 失败 (MilvusException): {e} - 集合可能不存在。"
                )
                return None
            logger.error(
                f"获取远程 Milvus 集合 '{collection_name}' (alias: {self.alias}) 失败: {e}",
                exc_info=True,
            )
            if (
                "connection refused" in str(e).lower()
                or "failed to connect" in str(e).lower()
            ):
                self._is_connected = False
            return None
        except Exception as e:  # 其他异常
            logger.error(
                f"获取远程 Milvus 集合 '{collection_name}' (alias: {self.alias}) 时发生未知错误: {e}",
                exc_info=True,
            )
            return None

    async def _ensure_index_and_load(self, collection: Collection):
        """辅助函数：确保指定集合的 embedding 字段有索引并已加载"""
        collection_name = collection.name
        try:
            has_embedding_index = False
            if collection.has_index(
                index_name=""
            ):  # 检查默认索引或任何embedding字段的索引
                indexes = collection.indexes
                for idx_obj in indexes:
                    if idx_obj.field_name == "embedding":
                        has_embedding_index = True
                        logger.info(
                            f"集合 '{collection_name}' 的 embedding 字段上已存在索引: {idx_obj.to_dict()}"
                        )
                        break

            if not has_embedding_index:
                logger.warning(
                    f"集合 '{collection_name}' 的 embedding 字段上没有索引。将尝试创建 {self.index_type}。"
                )
                index_obj = Index(
                    collection,
                    field_name="embedding",
                    index_params={
                        "index_type": self.index_type,
                        "metric_type": self.metric_type,
                        "params": self.index_params_config,
                    },
                    using=self.alias
                )
                collection.create_index(field_name="embedding", index_params=index_obj.params)
                logger.info(
                    f"为集合 '{collection_name}' 的 'embedding' 字段创建索引 {self.index_type} 成功 (通过 Index 对象)。"
                )

            # --- 加载逻辑 ---
            if collection.is_empty:
                logger.info(f"远程 Milvus 集合 '{collection_name}' 为空，无需加载。")
                return

            logger.info(f"检查集合 '{collection_name}' 的加载状态...")
            initial_load_state = utility.load_state(collection_name, using=self.alias)
            logger.info(f"集合 '{collection_name}' 当前加载状态: {initial_load_state}")

            is_loaded = False
            if LoadState:  # 如果 LoadState 枚举成功导入
                is_loaded = initial_load_state == LoadState.Loaded
            else:  # 后备：如果 LoadState 枚举不存在，尝试将状态转换为字符串并比较
                is_loaded = str(initial_load_state).upper() == "LOADED"

            if not is_loaded:
                logger.info(
                    f"集合 '{collection_name}' 未加载或加载未完成。尝试调用 load()..."
                )
                try:
                    collection.load(
                        # replica_number=1,
                        # timeout=self.kwargs.get("load_timeout", None)
                    )
                    logger.info(
                        f"已为集合 '{collection_name}' 调用 load()。等待加载完成..."
                    )

                    # 等待加载完成
                    try:
                        if hasattr(utility, "wait_for_loading_complete"):
                            utility.wait_for_loading_complete(
                                collection_name,
                                using=self.alias,
                                timeout=self.kwargs.get("load_wait_timeout", 60),
                            )
                            logger.info(
                                f"utility.wait_for_loading_complete 确认集合 '{collection_name}' 已加载。"
                            )
                        else:
                            import asyncio

                            max_wait_loops = self.kwargs.get("load_wait_loops", 12)
                            loop_delay = self.kwargs.get(
                                "load_loop_delay", 5
                            )  # 从 kwargs 获取
                            for i in range(max_wait_loops):
                                await asyncio.sleep(loop_delay)
                                current_state_obj = utility.load_state(
                                    collection_name, using=self.alias
                                )
                                logger.info(
                                    f"等待加载... 集合 '{collection_name}' 状态: {current_state_obj} (尝试 {i + 1}/{max_wait_loops})"
                                )
                                if LoadState:
                                    if current_state_obj == LoadState.Loaded:
                                        break
                                else:  # 后备字符串比较
                                    if str(current_state_obj).upper() == "LOADED":
                                        break
                            else:
                                logger.warning(
                                    f"等待加载超时，集合 '{collection_name}' 可能未完全加载。"
                                )
                    except Exception as e_wait:
                        logger.warning(
                            f"等待集合 '{collection_name}' 加载完成时出错: {e_wait}。继续执行..."
                        )

                    final_load_state_obj = utility.load_state(
                        collection_name, using=self.alias
                    )
                    final_is_loaded = False
                    if LoadState:
                        final_is_loaded = final_load_state_obj == LoadState.Loaded
                    else:
                        final_is_loaded = str(final_load_state_obj).upper() == "LOADED"

                    if final_is_loaded:
                        logger.info(
                            f"确认：集合 '{collection_name}' 加载成功，状态: {final_load_state_obj}。"
                        )
                    else:
                        logger.warning(
                            f"警告：调用 load() 后，集合 '{collection_name}' 的加载状态为: {final_load_state_obj}。搜索可能失败。"
                        )
                except MilvusException as e_load:
                    logger.error(
                        f"加载集合 '{collection_name}' 失败 (MilvusException): {e_load}",
                        exc_info=True,
                    )
                    if "collection already loaded" in str(e_load).lower():
                        logger.info(f"加载集合 '{collection_name}' 时报告已加载。")
                    else:
                        raise
                except Exception as e_load_general:
                    logger.error(
                        f"加载集合 '{collection_name}' 时发生未知错误: {e_load_general}",
                        exc_info=True,
                    )
                    raise
            else:  # is_loaded is True
                logger.info(f"集合 '{collection_name}' (alias: {self.alias}) 已加载。")

        except MilvusException as e:
            logger.error(
                f"为集合 '{collection_name}' 确保索引和加载状态时发生 Milvus 错误: {e}",
                exc_info=True,
            )
            if (
                "connection refused" in str(e).lower()
                or "failed to connect" in str(e).lower()
            ):
                self._is_connected = False
            raise
        except Exception as e:
            logger.error(
                f"为集合 '{collection_name}' 确保索引和加载状态时发生未知错误: {e}",
                exc_info=True,
            )
            raise

    async def create_collection(self, collection_name: str):
        if not self._is_connected:
            await self.initialize()  # 尝试连接
            if not self._is_connected:
                raise ConnectionError("Milvus 服务未连接，无法创建集合。")

        if await self.collection_exists(collection_name):
            logger.info(
                f"远程 Milvus 集合 '{collection_name}' (alias: {self.alias}) 已存在。将检查索引并加载。"
            )
            collection = await self._get_collection_with_retry(collection_name)
            if collection:
                await self._ensure_index_and_load(collection)
            else:
                logger.error(
                    f"无法获取已存在的集合 '{collection_name}' 进行索引和加载检查。"
                )
            return

        pk_field = FieldSchema(
            name="pk",
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=True,
            max_length=self.pk_max_length,
        )
        vector_field = FieldSchema(
            name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension
        )
        text_content_field = FieldSchema(
            name="text_content",
            dtype=DataType.VARCHAR,
            max_length=self.varchar_max_length,
        )

        schema = CollectionSchema(
            fields=[pk_field, vector_field, text_content_field],
            description=f"Knowledge base collection: {collection_name}",
            enable_dynamic_field=True,
        )

        try:
            collection = Collection(
                collection_name,
                schema=schema,
                using=self.alias,
                consistency_level=self.consistency_level_str,
            )
            logger.info(
                f"远程 Milvus 集合 '{collection_name}' (schema) (alias: {self.alias}) 创建成功。"
            )
            # 索引创建和加载移到 _ensure_index_and_load 中
            await self._ensure_index_and_load(collection)

        except Exception as e:
            logger.error(
                f"创建远程 Milvus 集合 '{collection_name}' (alias: {self.alias}) 或其索引/加载失败: {e}",
                exc_info=True,
            )
            raise

    async def collection_exists(self, collection_name: str) -> bool:
        if not self._is_connected:
            try:
                await self.initialize()
            except Exception:  # 如果 initialize 失败，则认为不存在
                return False
            if not self._is_connected:
                return False  # 如果仍未连接

        try:
            return utility.has_collection(collection_name, using=self.alias)
        except Exception as e:
            if (
                "not connected" in str(e).lower()
                or "connection refused" in str(e).lower()
                or "failed to connect" in str(e).lower()
            ):
                self._is_connected = False
                logger.error(
                    f"检查集合 '{collection_name}' (alias: {self.alias}) 是否存在时连接丢失: {e}"
                )
            else:
                logger.error(
                    f"检查远程 Milvus 集合 '{collection_name}' (alias: {self.alias}) 是否存在时出错: {e}",
                    # exc_info=True # 避免在频繁调用时过多日志
                )
            return False

    async def add_documents(
        self, collection_name: str, documents: List[Document]
    ) -> List[str]:
        if not self._is_connected:
            await self.initialize()
            if not self._is_connected:
                raise ConnectionError("Milvus 服务未连接，无法添加文档。")

        # 1. 获取并准备 Collection 对象 (在批处理循环前执行一次)
        collection = await self._get_collection_with_retry(collection_name)
        if not collection:
            auto_create = self.kwargs.get("auto_create_collection", True)
            if auto_create:
                logger.warning(
                    f"远程 Milvus 集合 '{collection_name}' (alias: {self.alias}) 不存在。将尝试自动创建。"
                )
                await self.create_collection(
                    collection_name
                )  # create_collection 会处理索引和加载
                collection = await self._get_collection_with_retry(collection_name)
                if not collection:
                    logger.error(
                        f"自动创建后仍无法获取远程 Milvus 集合 '{collection_name}'。"
                    )
                    return []
            else:
                logger.error(
                    f"远程 Milvus 集合 '{collection_name}' (alias: {self.alias}) 不存在且未配置自动创建。"
                )
                return []
        else:  # 集合已存在，确保索引和加载
            await self._ensure_index_and_load(collection)

        all_doc_ids: List[str] = []

        # 创建一个异步队列来存放待处理的批次
        processing_queue: asyncio.Queue[ProcessingBatch] = asyncio.Queue()

        # 2. 生产者：将所有文档按批次放入队列
        num_batches = 0
        for i in range(0, len(documents), DEFAULT_BATCH_SIZE):
            batch_docs = documents[i : i + DEFAULT_BATCH_SIZE]
            await processing_queue.put(ProcessingBatch(documents=batch_docs))
            num_batches += 1
        logger.info(f"已将 {len(documents)} 份文档分成 {num_batches} 个批次放入队列。")

        # 3. 消费者：从队列中取出批次进行处理
        processed_batches_count = 0
        failed_batches_discarded_count = 0

        # 这里通过检查 processed_batches_count 和 num_batches 来判断何时退出，
        # 即使有重试，最终也会被计入 processed_batches_count (成功或丢弃)
        while processed_batches_count < num_batches + failed_batches_discarded_count:
            try:
                # 设置超时，避免在队列空时无限等待
                processing_batch = await asyncio.wait_for(
                    processing_queue.get(), timeout=10.0
                )
            except asyncio.TimeoutError:
                if (
                    processing_queue.empty()
                ):  # 如果队列确实空了，并且所有初始批次都已处理或丢弃
                    logger.info(
                        "队列已空，且所有原始批次均已处理或丢弃。退出处理循环。"
                    )
                    break
                else:  # 队列不空但等待超时，可能是因为并发任务或队列任务卡住
                    logger.warning("队列不空，但等待批次超时，继续等待...")
                    continue

            current_docs_in_batch = processing_batch.documents
            current_retry_count = processing_batch.retry_count

            log_prefix = f"[批次 ({len(current_docs_in_batch)} docs), 重试 {current_retry_count}/{MAX_RETRIES}]"
            logger.debug(f"{log_prefix} 正在处理...")

            try:
                current_batch_texts_to_embed = []
                # docs_needing_embedding_in_batch = [] # 此变量在循环中不再直接使用

                # 区分需要生成嵌入的文档和已包含嵌入的文档
                for doc in current_docs_in_batch:
                    if doc.embedding is None:
                        current_batch_texts_to_embed.append(doc.text_content)
                        # docs_needing_embedding_in_batch.append(doc) # 不需要，直接赋值给 doc.embedding

                batch_embeddings_generated: List[List[float]] = []
                if current_batch_texts_to_embed:
                    batch_embeddings_generated = (
                        await self.embedding_util.get_embeddings_async(
                            current_batch_texts_to_embed
                        )
                    )
                    logger.debug(
                        f"{log_prefix} 成功为 {len(batch_embeddings_generated)} 个文本生成了嵌入。"
                    )

                data_to_insert_for_batch: List[Dict[str, Any]] = []

                embed_idx = 0
                for doc in current_docs_in_batch:
                    embedding_to_use = None
                    if doc.embedding is not None:
                        embedding_to_use = doc.embedding
                    elif (
                        embed_idx < len(batch_embeddings_generated)
                        and batch_embeddings_generated[embed_idx] is not None
                    ):
                        embedding_to_use = batch_embeddings_generated[embed_idx]
                        embed_idx += 1

                    if embedding_to_use:
                        entity = {
                            "embedding": embedding_to_use,
                            "text_content": doc.text_content,
                        }
                        # 处理动态元数据字段
                        for key, value in doc.metadata.items():
                            if key not in entity:  # 避免覆盖固定字段
                                if (
                                    isinstance(value, str)
                                    and len(value) > self.varchar_max_length
                                ):
                                    logger.warning(
                                        f"{log_prefix} 元数据字段 '{key}' 的值过长 ({len(value)} > {self.varchar_max_length})，将被截断。"
                                    )
                                    entity[key] = value[: self.varchar_max_length]
                                elif not isinstance(
                                    value, (str, int, float, bool, list)
                                ):
                                    logger.warning(
                                        f"{log_prefix} 元数据字段 '{key}' 的类型 '{type(value)}' 不是常见支持类型，将转换为字符串。"
                                    )
                                    entity[key] = str(value)
                                else:
                                    entity[key] = value
                        data_to_insert_for_batch.append(entity)
                    else:
                        logger.warning(
                            f"{log_prefix} 未能为文档 '{doc.text_content[:50]}...' 获取 embedding，将跳过。"
                        )

                if not data_to_insert_for_batch:
                    logger.info(f"{log_prefix} 没有有效的实体可供插入，跳过此批次。")
                    processed_batches_count += 1
                    processing_queue.task_done()  # 标记此任务已完成
                    continue

                # 将 Milvus insert 操作包装在 asyncio.to_thread 中
                insert_result = await asyncio.to_thread(
                    collection.insert, data_to_insert_for_batch
                )

                # 对于标准 Milvus，flush 很重要以确保数据对搜索可见
                if self.kwargs.get("auto_flush_after_insert", True):
                    logger.info(f"{log_prefix} 正在刷新集合 '{collection_name}'...")
                    await asyncio.to_thread(collection.flush)  # flush 也是同步操作
                    logger.info(f"{log_prefix} 集合 '{collection_name}' 刷新完成。")

                # Milvus 返回的 primary_keys 是 int，转为 str
                batch_added_ids = [str(pk) for pk in insert_result.primary_keys]
                all_doc_ids.extend(batch_added_ids)
                logger.info(
                    f"{log_prefix} 成功向远程 Milvus 集合 '{collection_name}' (alias: {self.alias}) 添加了 {len(batch_added_ids)} 个文档。"
                )

                processed_batches_count += 1
                processing_queue.task_done()  # 标记此任务已完成

            except Exception as e:
                logger.error(
                    f"{log_prefix} 处理失败: {e}", exc_info=True
                )  # 打印完整的堆栈信息

                # 检查连接错误，如果连接断开，设置 _is_connected 为 False
                if (
                    "connection refused" in str(e).lower()
                    or "failed to connect" in str(e).lower()
                    or "rpc error" in str(e).lower()  # 常见的网络或服务错误
                ):
                    self._is_connected = False
                    logger.error(
                        f"{log_prefix} 检测到 Milvus 连接问题，将尝试重新初始化连接。"
                    )
                    # 这里可以考虑在重试前尝试重新初始化连接
                    # await self.initialize() # 如果需要每个批次都重新连接，但通常不推荐

                if current_retry_count < MAX_RETRIES:
                    processing_batch.retry_count += 1
                    await processing_queue.put(
                        processing_batch
                    )  # 将批次重新放回队列尾部
                    logger.warning(
                        f"{log_prefix} 将批次重新放入队列进行重试 (当前重试次数: {processing_batch.retry_count})。"
                    )
                else:
                    logger.error(
                        f"{log_prefix} 批次达到最大重试次数 ({MAX_RETRIES})，将丢弃此批次。"
                    )
                    failed_batches_discarded_count += 1

                processed_batches_count += 1  # 无论成功或失败，原始批次都被“处理”了一次
                processing_queue.task_done()  # 标记此任务已完成 (无论是重试还是丢弃)

            finally:
                # 显式清理本批次可能占用的内存
                del current_batch_texts_to_embed
                del batch_embeddings_generated
                del data_to_insert_for_batch  # 清理 Milvus 插入实体列表
                del current_docs_in_batch  # 清理当前批次文档的引用
                del processing_batch  # 清理批次对象的引用

        logger.info(
            f"向远程 Milvus 集合 '{collection_name}' (alias: {self.alias}) 完成添加操作。"
        )
        logger.info(
            f"总共处理了 {len(documents)} 个原始文档，成功添加 {len(all_doc_ids)} 个文档。"
        )
        logger.info(f"其中有 {failed_batches_discarded_count} 个批次因重试失败被丢弃。")
        return all_doc_ids

    async def search(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 5,
        filter_expr: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        if not self._is_connected:
            await self.initialize()
            if not self._is_connected:
                raise ConnectionError("Milvus 服务未连接，无法搜索。")

        collection = await self._get_collection_with_retry(collection_name)
        if not collection:
            logger.warning(
                f"远程 Milvus 集合 '{collection_name}' (alias: {self.alias}) 不存在或无法获取。"
            )
            return []

        # 确保索引和加载在搜索前完成
        try:
            await self._ensure_index_and_load(collection)  # <--- 关键调用
        except Exception as e_ensure:
            logger.error(
                f"搜索前确保集合 '{collection_name}' 状态失败: {e_ensure}",
                exc_info=True,
            )
            return []  # 如果确保状态失败，则无法继续搜索

        num_docs = await self.count_documents(collection_name)
        if num_docs == 0:
            logger.info(
                f"远程 Milvus 集合 '{collection_name}' (alias: {self.alias}) 为空，无法搜索。"
            )
            return []

        query_embedding = await self.embedding_util.get_embedding_async(query_text)
        if query_embedding is None:
            logger.error("无法为查询文本生成 embedding。")
            return []

        # 运行时搜索参数
        # index_params_config 是创建索引时的参数，search_params 是搜索时的参数
        search_params = {"metric_type": self.metric_type}
        if self.index_type == "IVF_FLAT":
            search_params["params"] = self.kwargs.get(
                "search_params", {"nprobe": 10}
            )  # 从 kwargs 获取搜索参数
        elif self.index_type == "HNSW":
            search_params["params"] = self.kwargs.get("search_params", {"ef": 64})
        elif self.index_type == "AUTOINDEX" or self.index_type == "FLAT":
            search_params["params"] = {}  # AUTOINDEX/FLAT 通常不需要运行时搜索参数
        else:  # 其他索引类型
            search_params["params"] = self.kwargs.get("search_params", {})

        fields_to_request = []
        try:
            coll_schema = collection.schema
            if coll_schema:
                for field in coll_schema.fields:
                    if (
                        field.name not in ["pk", "embedding"]  # 假设主键名为 "pk"
                        and not field.is_primary
                        and field.dtype
                        not in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR]
                    ):
                        fields_to_request.append(field.name)
            # 确保 text_content 在其中
            if "text_content" not in fields_to_request and any(
                f.name == "text_content" for f in coll_schema.fields
            ):
                fields_to_request.append("text_content")

        except Exception as e_desc:
            logger.warning(
                f"获取集合 '{collection_name}' schema 失败: {e_desc}。将默认请求 'text_content'。"
            )
            fields_to_request = ["text_content"]

        if (
            not fields_to_request
            and collection.schema
            and any(f.name == "text_content" for f in collection.schema.fields)
        ):
            fields_to_request.append("text_content")
        elif not fields_to_request:
            logger.warning(
                f"无法确定集合 '{collection_name}' 中的输出字段，搜索可能不返回元数据。"
            )
            # fields_to_request = None # 让 Milvus 返回默认字段 (ID和distance/score)

        try:
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=min(top_k, num_docs) if num_docs > 0 else top_k,
                expr=filter_expr if filter_expr else None,  # None 表示无过滤
                output_fields=fields_to_request if fields_to_request else None,
                consistency_level=self.consistency_level_str,  # 使用字符串形式的一致性级别
            )

            processed_results = []
            if results and results[0]:  # results[0] 对应第一个查询向量的命中结果
                for hit in results[0]:
                    doc_id = str(hit.id)
                    distance = hit.distance
                    similarity_score: float
                    if self.metric_type == "IP":
                        similarity_score = float(distance)
                    else:  # L2
                        similarity_score = 1.0 / (1.0 + max(0.0, float(distance)))

                    entity_data = (
                        hit.entity.to_dict()
                        if hasattr(hit, "entity") and hasattr(hit.entity, "to_dict")
                        else {}
                    )
                    text_content = entity_data.get("text_content", "")

                    metadata = {
                        k: v for k, v in entity_data.items() if k != "text_content"
                    }

                    doc = Document(
                        id=doc_id, text_content=text_content, metadata=metadata
                    )
                    processed_results.append((doc, similarity_score))
            return processed_results
        except Exception as e:
            logger.error(
                f"在远程 Milvus 集合 '{collection_name}' (alias: {self.alias}) 中搜索失败: {e}",
                exc_info=True,
            )
            if (
                "connection refused" in str(e).lower()
                or "failed to connect" in str(e).lower()
            ):
                self._is_connected = False
            return []

    async def delete_collection(self, collection_name: str) -> bool:
        if not self._is_connected:
            await self.initialize()
            if not self._is_connected:
                raise ConnectionError("Milvus 服务未连接，无法删除集合。")

        if not await self.collection_exists(collection_name):
            logger.info(
                f"远程 Milvus 集合 '{collection_name}' (alias: {self.alias}) 不存在，无需删除。"
            )
            return False
        try:
            utility.drop_collection(collection_name, using=self.alias)
            logger.info(
                f"远程 Milvus 集合 '{collection_name}' (alias: {self.alias}) 已删除。"
            )
            return True
        except Exception as e:
            logger.error(
                f"删除远程 Milvus 集合 '{collection_name}' (alias: {self.alias}) 失败: {e}",
                exc_info=True,
            )
            if (
                "connection refused" in str(e).lower()
                or "failed to connect" in str(e).lower()
            ):
                self._is_connected = False
            return False

    async def list_collections(self) -> List[str]:
        if not self._is_connected:
            try:
                await self.initialize()
            except Exception:
                return []
            if not self._is_connected:
                return []

        try:
            return utility.list_collections(using=self.alias)
        except Exception as e:
            if (
                "not connected" in str(e).lower()
                or "connection refused" in str(e).lower()
                or "failed to connect" in str(e).lower()
            ):
                self._is_connected = False
                logger.error(f"列出集合 (alias: {self.alias}) 时连接丢失: {e}")
            else:
                logger.error(
                    f"列出远程 Milvus 集合 (alias: {self.alias}) 失败: {e}",
                    exc_info=True,
                )
            return []

    async def count_documents(self, collection_name: str) -> int:
        if not self._is_connected:
            try:
                await self.initialize()
            except Exception:
                return 0
            if not self._is_connected:
                return 0

        collection = await self._get_collection_with_retry(collection_name)
        if not collection:
            return 0
        try:
            # 确保集合已加载，否则 num_entities 可能不准确或报错
            # await self._ensure_index_and_load(collection) # num_entities 应该在加载后准确

            # Milvus 2.2.x 之后，collection.num_entities 是推荐的
            # 它可能需要集合先被加载 (collection.load() 之后)
            # collection.flush() # 可能影响 num_entities 的准确性，确保数据已提交
            stats = collection.num_entities
            return stats
        except Exception as e_stats:
            logger.warning(
                f"获取集合 '{collection_name}' (alias: {self.alias}) 文档数 (num_entities) 失败: {e_stats}。尝试 query count(*)..."
            )
            try:
                res = collection.query(
                    expr="",  # 或者 "pk != ''" 如果主键可能为空字符串
                    output_fields=["count(*)"],
                    consistency_level=self.consistency_level_str,
                )
                if res and isinstance(res, list) and res[0] and "count(*)" in res[0]:
                    return int(res[0]["count(*)"])
                else:
                    logger.warning(
                        f"count(*) for collection '{collection_name}' (alias: {self.alias}) returned unexpected: {res}. Querying all pks..."
                    )
                    all_pks_res = collection.query(
                        expr="",
                        output_fields=["pk"],  # 假设主键字段名为 'pk'
                        consistency_level=self.consistency_level_str,
                    )
                    return len(all_pks_res) if all_pks_res else 0
            except Exception as query_err:
                logger.error(
                    f"通过 query 获取远程 Milvus 集合 '{collection_name}' (alias: {self.alias}) 文档数也失败: {query_err}",
                    exc_info=True,
                )
                if (
                    "connection refused" in str(query_err).lower()
                    or "failed to connect" in str(query_err).lower()
                ):
                    self._is_connected = False
                return 0

    async def close(self):
        if self._is_connected:  # 检查 self._is_connected 标志
            logger.info(
                f"尝试断开与远程 Milvus 服务 {self.host}:{self.port} (alias: {self.alias}) 的连接..."
            )
            try:
                connections.disconnect(self.alias)
                self._is_connected = False  # 成功断开后更新状态
                logger.info(
                    f"与远程 Milvus 服务 {self.host}:{self.port} (alias: {self.alias}) 的连接已断开。"
                )
            except Exception as e:  # Milvus 可能在 alias 不存在时抛出异常
                logger.error(
                    f"断开远程 Milvus 连接 (alias: {self.alias}) 失败或连接本不存在: {e}",
                    # exc_info=True # 根据需要决定是否在 close 时记录完整 traceback
                )
                self._is_connected = False  # 无论如何，标记为未连接
        else:
            logger.info(
                f"远程 Milvus 服务 (alias: {self.alias}) 连接本就未建立或已断开，无需执行 close。"
            )

import gc  # 添加垃圾回收模块
from typing import List, Optional, Tuple
from astrbot.api import logger
from astrbot.api.star import Context
from ..core.user_prefs_handler import UserPrefsHandler
from openai import (
    AsyncOpenAI,
    APIError,
    APIStatusError,
    APIConnectionError,
    RateLimitError,
)


class EmbeddingUtil:
    def __init__(self, api_url: str, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name

        # 处理 api_url 以适应 AsyncOpenAI 的 base_url 参数
        self.base_url_for_client: Optional[str] = api_url
        if self.base_url_for_client and self.base_url_for_client.endswith(
            "/embeddings"
        ):
            self.base_url_for_client = self.base_url_for_client[: -len("/embeddings")]

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url_for_client,
            timeout=30.0,
        )

    async def get_embedding_async(self, text: str) -> Optional[List[float]]:
        """获取单个文本的 embedding"""
        if not text or not text.strip():
            logger.warning("输入文本为空或仅包含空白，无法获取 embedding。")
            return None
        try:
            response = await self.client.embeddings.create(
                input=text, model=self.model_name
            )
            if (
                response.data
                and len(response.data) > 0
                and hasattr(response.data[0], "embedding")
            ):
                return response.data[0].embedding
            else:
                logger.error(
                    f"获取 Embedding 失败，OpenAI API 响应格式不正确或数据为空: {response}"
                )
                return None
        except APIStatusError as e:
            logger.error(
                f"获取 Embedding API 请求失败 (OpenAI Status Error)，状态码: {e.status_code}, "
                f"类型: {e.type}, 参数: {e.param}, 消息: {e.message}"
                f"响应详情: {e.response.text if e.response else 'N/A'}"
            )
            return None
        except APIConnectionError as e:
            logger.error(f"获取 Embedding API 连接失败 (OpenAI Connection Error): {e}")
            return None
        except RateLimitError as e:
            logger.error(
                f"获取 Embedding API 请求达到速率限制 (OpenAI RateLimit Error): {e}"
            )
            return None
        except APIError as e:
            logger.error(f"获取 Embedding 时发生 OpenAI API 错误: {e}")
            return None
        except Exception as e:
            logger.error(f"获取 Embedding 时发生未知错误: {e}", exc_info=True)
            return None

    async def get_embeddings_async(
        self, texts: List[str]
    ) -> List[Optional[List[float]]]:
        """获取多个文本的 embedding (使用 OpenAI 批量接口)，每 10 条文本分批请求"""
        if not texts:
            logger.info("输入文本列表为空，返回空 embedding 列表。")
            return []

        # 预处理输入：记录有效文本及其原始索引，过滤空或纯空白字符串
        valid_texts_with_indices: List[Tuple[int, str]] = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts_with_indices.append((i, text))
            else:
                logger.warning(
                    f"输入文本列表在索引 {i} 处为空或仅包含空白，将标记为 None。"
                )

        if not valid_texts_with_indices:
            logger.info("输入文本列表所有文本均为空或无效，返回相应数量的 None。")
            return [None] * len(texts)

        # 初始化结果列表，长度与输入文本列表一致，初始值为 None
        final_embeddings: List[Optional[List[float]]] = [None] * len(texts)

        # 设置每批次大小为 10
        batch_size = 10
        # actual_texts_to_embed = [item[1] for item in valid_texts_with_indices]

        # 分批处理文本
        for batch_start in range(0, len(valid_texts_with_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(valid_texts_with_indices))
            batch_indices_texts = valid_texts_with_indices[batch_start:batch_end]
            batch_texts = [text for _, text in batch_indices_texts]

            try:
                response = await self.client.embeddings.create(
                    input=batch_texts, model=self.model_name
                )

                # 验证响应数据
                if response.data and len(response.data) == len(batch_texts):
                    embeddings_for_batch = [item.embedding for item in response.data]
                    for idx, (original_idx, _) in enumerate(batch_indices_texts):
                        final_embeddings[original_idx] = embeddings_for_batch[idx]

                    # 内存优化：及时清理批次数据
                    del embeddings_for_batch
                    del response
                else:
                    logger.error(
                        f"批次 {batch_start // batch_size + 1} 获取 Embeddings 失败："
                        f"API 响应的数据项数量 ({len(response.data) if response.data else 0}) "
                        f"与输入文本数量 ({len(batch_texts)}) 不匹配。"
                    )
                    # 继续处理下一批次，而不是直接返回
                    continue

                # 清理批次临时数据
                del batch_texts
                del batch_indices_texts

                # 每5个批次触发一次垃圾回收
                if (batch_start // batch_size + 1) % 5 == 0:
                    gc.collect()

            except APIStatusError as e:
                logger.error(
                    f"批次 {batch_start // batch_size + 1} 获取 Embeddings API 请求失败 "
                    f"(OpenAI Status Error)，状态码: {e.status_code}, 类型: {e.type}, "
                    f"参数: {e.param}, 消息: {e.message}"
                )
                continue
            except APIConnectionError as e:
                logger.error(
                    f"批次 {batch_start // batch_size + 1} 获取 Embeddings API 连接失败 "
                    f"(OpenAI Connection Error): {e}"
                )
                continue
            except RateLimitError as e:
                logger.error(
                    f"批次 {batch_start // batch_size + 1} 获取 Embeddings API 请求达到速率限制 "
                    f"(OpenAI RateLimit Error): {e}"
                )
                continue
            except APIError as e:
                logger.error(
                    f"批次 {batch_start // batch_size + 1} 获取 Embeddings 时发生 OpenAI API 错误: {e}"
                )
                continue
            except Exception as e:
                logger.error(
                    f"批次 {batch_start // batch_size + 1} 获取 Embeddings 时发生未知错误: {e}",
                    exc_info=True,
                )
                continue

        # 检查是否所有有效文本都未能生成嵌入
        if all(embedding is None for embedding in final_embeddings):
            logger.error("所有批次均未能成功生成嵌入，请检查 API 配置或网络连接。")

        # 最终内存清理
        del valid_texts_with_indices
        gc.collect()

        return final_embeddings

    async def close(self):
        """关闭 OpenAI 客户端"""
        if hasattr(self, "client") and self.client:
            await self.client.close()


class EmbeddingSolutionHelper:
    """适配 AstrBot 的 Embedding 方案专用的帮助类。

    Required:
        - AstrBot Version: >= v3.5.13
    """

    def __init__(
        self,
        curr_embedding_dimensions: int,
        curr_embedding_util: EmbeddingUtil,
        context: Context,
        user_prefs_handler: UserPrefsHandler,
    ):
        self.curr_embedding_dimensions = curr_embedding_dimensions
        self.curr_embedding_util = curr_embedding_util
        self.context = context
        self.user_prefs_handler = user_prefs_handler

    async def _get_embedding_via_astrbot_provider(
        self, text: str | list[str], collection_name: str
    ) -> list[float] | list[list[float]]:
        """通过 AstrBot 提供商获取单个文本的 embedding"""
        from astrbot.core.provider.provider import EmbeddingProvider

        astrbot_embedding_provider_id = (
            self.user_prefs_handler.user_collection_preferences.get(
                "collection_metadata", {}
            )
            .get(collection_name, {})
            .get("embedding_provider_id", None)
        )
        if astrbot_embedding_provider_id:
            provider = self.context.get_provider_by_id(astrbot_embedding_provider_id)
            if provider and isinstance(provider, EmbeddingProvider):
                if isinstance(text, str):
                    return await provider.get_embedding(text)
                elif isinstance(text, list):
                    return await provider.get_embeddings(text)
                else:
                    raise TypeError(
                        f"Unsupported type for text: {type(text)}. Expected str or list[str]."
                    )
        else:
            raise ValueError(
                f"未找到适用于集合 '{collection_name}' 的 AstrBot 嵌入提供商。请检查用户偏好设置或集合元数据。"
            )

    def _get_embedding_dimensions_via_astrbot_provider(
        self, collection_name: str
    ) -> int:
        """通过 AstrBot 提供商获取对应提供商的嵌入模型的维度"""
        from astrbot.core.provider.provider import EmbeddingProvider

        astrbot_embedding_provider_id = (
            self.user_prefs_handler.user_collection_preferences.get(
                "collection_metadata", {}
            )
            .get(collection_name, {})
            .get("embedding_provider_id", None)
        )
        if astrbot_embedding_provider_id:
            provider = self.context.get_provider_by_id(astrbot_embedding_provider_id)
            if provider and isinstance(provider, EmbeddingProvider):
                return provider.get_dim()
        else:
            raise ValueError(
                f"未找到适用于集合 '{collection_name}' 的 AstrBot 嵌入提供商。请检查用户偏好设置或集合元数据。"
            )

    def get_rerank_provider(self, collection_name: str):
        """获取对应集合的重排序提供商"""
        try:
            from astrbot.core.provider.provider import RerankProvider
        except Exception:
            logger.error("无法导入 RerankProvider，请确保 AstrBot 版本 >= v4.0.0。")
            return None
        astrbot_rerank_provider_id = (
            self.user_prefs_handler.user_collection_preferences.get(
                "collection_metadata", {}
            )
            .get(collection_name, {})
            .get("rerank_provider_id", None)
        )
        if astrbot_rerank_provider_id:
            provider = self.context.get_provider_by_id(astrbot_rerank_provider_id)
            if provider and isinstance(provider, RerankProvider):
                return provider
        return None

    async def get_embedding_async(
        self, text: str, collection_name: str
    ) -> Optional[List[float]]:
        """获取单个文本的 embedding，并更新用户偏好设置"""
        astrbot_embedding_provider_id = (
            self.user_prefs_handler.user_collection_preferences.get(
                "collection_metadata", {}
            )
            .get(collection_name, {})
            .get("embedding_provider_id", None)
        )
        if astrbot_embedding_provider_id:
            # we assume that if embedding_provider_id is set, it should be Version >= 3.5.13
            embedding = await self._get_embedding_via_astrbot_provider(
                text, collection_name
            )
        else:
            embedding = await self.curr_embedding_util.get_embedding_async(text)
        return embedding

    async def get_embeddings_async(
        self, texts: List[str], collection_name: str
    ) -> List[Optional[List[float]]]:
        """获取多个文本的 embedding，并更新用户偏好设置"""
        astrbot_embedding_provider_id = (
            self.user_prefs_handler.user_collection_preferences.get(
                "collection_metadata", {}
            )
            .get(collection_name, {})
            .get("embedding_provider_id", None)
        )
        if astrbot_embedding_provider_id:
            # we assume that if embedding_provider_id is set, it should be Version >= 3.5.13
            embeddings = await self._get_embedding_via_astrbot_provider(
                texts, collection_name
            )
        else:
            embeddings = await self.curr_embedding_util.get_embeddings_async(texts)
        return embeddings

    async def close(self):
        """关闭当前的 EmbeddingUtil 客户端"""
        await self.curr_embedding_util.close()

    def get_dimensions(self, collection_name) -> int:
        astrbot_embedding_provider_id = (
            self.user_prefs_handler.user_collection_preferences.get(
                "collection_metadata", {}
            )
            .get(collection_name, {})
            .get("embedding_provider_id", None)
        )
        if astrbot_embedding_provider_id:
            # we assume that if embedding_provider_id is set, it should be Version >= 3.5.13
            return self._get_embedding_dimensions_via_astrbot_provider(collection_name)
        else:
            return self.curr_embedding_dimensions

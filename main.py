import json
import os
import asyncio
import tempfile
import httpx
from typing import Optional, Dict, Union
from urllib.parse import urlparse

from astrbot.api import logger, AstrBotConfig
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.core.utils.session_waiter import (
    session_waiter,
    SessionController,
)
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import StarTools
from .installation import ensure_vector_db_dependencies
from .utils.embedding import EmbeddingUtil
from .utils.text_splitter import TextSplitterUtil
from .utils.file_parser import parse_file_content
from .vector_store.base import VectorDBBase, Document
from .vector_store.faiss_store import FaissStore
from .vector_store.milvus_lite_store import MilvusLiteStore
from .vector_store.milvus_store import MilvusStore


PLUGIN_REGISTER_NAME = "astrbot_plugin_knowledge_base"
# 定义知识库内容标记
KB_START_MARKER = "###KBDATA_START###"
KB_END_MARKER = "###KBDATA_END###"
# 用于 'prepend_prompt' 方式时，在用户原始问题前添加的标记
USER_PROMPT_DELIMITER_IN_HISTORY = "\n\n用户的原始问题是：\n"


@register(
    PLUGIN_REGISTER_NAME,
    "lxfight",
    "一个支持多种向量数据库的知识库插件",
    "0.1.0",
    "https://github.com/lxfight/astrbot_plugin_knowledge_base",
)
class KnowledgeBasePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._initialize()
        self.vector_db: Optional[VectorDBBase] = None
        self.embedding_util: Optional[Union[EmbeddingUtil, Star]] = None
        self.text_splitter: Optional[TextSplitterUtil] = None

        self.plugin_name_for_path = PLUGIN_REGISTER_NAME  # 用于路径创建

        # --- 持久化数据路径计算 ---
        self.persistent_data_root_path = StarTools.get_data_dir(PLUGIN_REGISTER_NAME)
        os.makedirs(self.persistent_data_root_path, exist_ok=True)
        logger.info(f"知识库插件的持久化数据目录: {self.persistent_data_root_path}")

        self.user_prefs_path = os.path.join(
            self.persistent_data_root_path, "user_collection_prefs.json"
        )
        self.user_collection_preferences: Dict[str, str] = {}

        self.init_task = asyncio.create_task(self._initialize_components())

    def _initialize(self):
        """
        初始化知识库插件包
        """
        ensure_vector_db_dependencies(self.config.get("vector_db_type", "faiss"))

    async def _initialize_components(self):
        try:
            logger.info("知识库插件开始初始化...")
            try:
                embedding_plugin = self.context.get_registered_star(
                    "astrbot_plugin_embedding_adapter"
                )
                if embedding_plugin:
                    self.embedding_util = embedding_plugin.star_cls
                    dim = self.embedding_util.get_dim()
                    model_name = self.embedding_util.get_model_name()
                    if dim is not None and model_name is not None:
                        self.config["embedding_dimension"] = dim
                        self.config["embedding_model_name"] = model_name
            except Exception as e:
                logger.warning(f"嵌入服务适配器插件加载失败: {e}", exc_info=True)
                self.embedding_util = None
            if self.embedding_util is None:
                self.embedding_util = EmbeddingUtil(
                    api_url=self.config.get("embedding_api_url"),
                    api_key=self.config.get("embedding_api_key"),
                    model_name=self.config.get("embedding_model_name"),
                )
            logger.info("Embedding 工具初始化完成。")

            self.text_splitter = TextSplitterUtil(
                chunk_size=self.config.get("text_chunk_size"),
                chunk_overlap=self.config.get("text_chunk_overlap"),
            )
            logger.info("文本分割工具初始化完成。")

            db_type = self.config.get("vector_db_type", "faiss")
            dimension = self.config.get("embedding_dimension", 1536)

            if db_type == "faiss":
                # 使用 self.persistent_data_root_path 作为基础
                faiss_subpath = self.config.get("faiss_db_subpath", "faiss_data")
                faiss_full_path = os.path.join(
                    self.persistent_data_root_path, faiss_subpath
                )
                self.vector_db = FaissStore(
                    self.embedding_util, dimension, faiss_full_path
                )
            elif db_type == "milvus_lite":
                milvus_lite_subpath = self.config.get(
                    "milvus_lite_db_subpath", "milvus_lite_data/milvus_lite.db"
                )
                milvus_lite_full_path = os.path.join(
                    self.persistent_data_root_path, milvus_lite_subpath
                )
                os.makedirs(os.path.dirname(milvus_lite_full_path), exist_ok=True)
                self.vector_db = MilvusLiteStore(
                    self.embedding_util, dimension, milvus_lite_full_path
                )
            elif db_type == "milvus":
                self.vector_db = MilvusStore(
                    self.embedding_util,
                    dimension,
                    data_path="",
                    host=self.config.get("milvus_host"),
                    port=self.config.get("milvus_port"),
                    user=self.config.get("milvus_user"),
                    password=self.config.get("milvus_password"),
                )
            else:
                logger.error(f"不支持的向量数据库类型: {db_type}，请检查配置。")
                return

            if self.vector_db:
                await self.vector_db.initialize()
                logger.info(f"向量数据库 '{db_type}' 初始化完成。")

            await self._load_user_preferences()
            logger.info("知识库插件初始化成功。")

        except Exception as e:
            logger.error(f"知识库插件初始化失败: {e}", exc_info=True)
            self.vector_db = None

    async def _ensure_initialized(self) -> bool:
        if self.init_task and not self.init_task.done():
            await self.init_task  # 等待初始化完成
        if not self.vector_db or not self.embedding_util or not self.text_splitter:
            logger.error("知识库插件未正确初始化，请检查日志和配置。")
            return False
        return True

    async def _load_user_preferences(self):
        try:
            if os.path.exists(self.user_prefs_path):
                with open(self.user_prefs_path, "r", encoding="utf-8") as f:
                    self.user_collection_preferences = json.load(f)
                logger.info(f"从 {self.user_prefs_path} 加载了用户知识库偏好。")
            else:
                logger.info(
                    f"用户知识库偏好文件 {self.user_prefs_path} 未找到，将使用默认值。"
                )
        except Exception as e:
            logger.error(f"加载用户知识库偏好失败: {e}")
            self.user_collection_preferences = {}

    async def _save_user_preferences(self):
        try:
            with open(self.user_prefs_path, "w", encoding="utf-8") as f:
                json.dump(
                    self.user_collection_preferences, f, ensure_ascii=False, indent=4
                )
            logger.info(f"用户知识库偏好已保存到 {self.user_prefs_path}。")
        except Exception as e:
            logger.error(f"保存用户知识库偏好失败: {e}")

    def _get_user_default_collection(self, event: AstrMessageEvent) -> str:
        """获取用户/群组的默认知识库，如果没有设置则返回全局默认"""
        user_key = event.unified_msg_origin  # 使用 unified_msg_origin 作为唯一标识
        return self.user_collection_preferences.get(
            user_key, self.config.get("default_collection_name", "general")
        )

    async def _set_user_default_collection(
        self, event: AstrMessageEvent, collection_name: str
    ):
        """设置用户/群组的默认知识库"""
        if not await self.vector_db.collection_exists(collection_name):
            # 如果配置了自动创建，并且集合不存在
            if self.config.get("auto_create_collection", True):
                try:
                    await self.vector_db.create_collection(collection_name)
                    logger.info(f"自动创建知识库 '{collection_name}' 成功。")
                except Exception as e:
                    logger.error(f"自动创建知识库 '{collection_name}' 失败: {e}")
                    yield event.plain_result(
                        f"自动创建知识库 '{collection_name}' 失败: {e}"
                    )
                    return
            else:
                yield event.plain_result(
                    f"知识库 '{collection_name}' 不存在，且未配置自动创建。"
                )
                return

        user_key = event.unified_msg_origin
        self.user_collection_preferences[user_key] = collection_name
        await self._save_user_preferences()
        yield event.plain_result(f"当前会话默认知识库已设置为: {collection_name}")

    # --- 辅助函数：下载文件 ---
    async def _download_file(self, url: str, destination_folder: str) -> Optional[str]:
        """
        异步下载文件到指定文件夹。
        返回下载后的文件路径，如果失败则返回 None。
        """
        try:
            async with httpx.AsyncClient(
                timeout=60.0, follow_redirects=True
            ) as client:  # 增加超时和重定向
                response = await client.get(url)
                response.raise_for_status()  # 检查 HTTP 错误

                # 从 URL 获取文件名，或生成一个
                parsed_url = urlparse(url)
                filename = os.path.basename(parsed_url.path)
                if not filename:  # 如果路径为空，例如 "http://example.com/"
                    # 尝试从 Content-Disposition header 获取文件名
                    content_disposition = response.headers.get("Content-Disposition")
                    if content_disposition:
                        import re

                        match = re.search(r'filename="?([^"]+)"?', content_disposition)
                        if match:
                            filename = match.group(1)
                    if not filename:  # 仍然没有文件名，生成一个
                        filename = (
                            f"downloaded_file_{tempfile._RandomNameSequence().next()}"
                        )

                # 限制文件名，防止路径遍历等问题
                filename = "".join(
                    c for c in filename if c.isalnum() or c in [".", "_", "-"]
                ).strip()
                if not filename:
                    filename = "untitled_download"  # 最终回退

                # 简单的文件大小限制 (例如 50MB)
                max_size = 50 * 1024 * 1024
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > max_size:
                    logger.error(
                        f"文件下载失败：文件过大 ({int(content_length) / (1024 * 1024):.2f} MB > {max_size / (1024 * 1024)} MB)。URL: {url}"
                    )
                    return None

                # 简单的文件类型嗅探或基于扩展名过滤
                _, extension = os.path.splitext(filename)
                allowed_extensions = [".txt", ".md"]  # , ".pdf", ".docx" 等
                if extension.lower() not in allowed_extensions:
                    logger.error(
                        f"文件下载失败：不支持的文件类型 '{extension}'. URL: {url}"
                    )
                    return None

                temp_file_path = os.path.join(destination_folder, filename)

                # 分块写入，处理大文件
                with open(temp_file_path, "wb") as f:
                    downloaded_size = 0
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if (
                            downloaded_size > max_size
                        ):  # 再次检查，以防 Content-Length 未提供或不准确
                            f.close()
                            os.remove(temp_file_path)  # 删除不完整的文件
                            logger.error(
                                f"文件下载失败：文件在下载过程中超出大小限制。URL: {url}"
                            )
                            return None

                logger.info(f"文件已成功下载到: {temp_file_path} 从 URL: {url}")
                return temp_file_path
        except httpx.HTTPStatusError as e:
            logger.error(
                f"文件下载 HTTP 错误: {e.response.status_code} - {e.response.text}. URL: {url}"
            )
            return None
        except Exception as e:
            logger.error(f"文件下载失败: {e}. URL: {url}", exc_info=True)
            return None

    def _clean_contexts_from_kb_content(self, req: ProviderRequest):
        """
        自动删除 req.contexts 里面由知识库补充的历史对话内容。
        使用 ###KBDATA_START### 和 ###KBDATA_END### 标记进行识别。
        """
        if not req.contexts:
            return

        cleaned_contexts = []
        initial_context_count = len(req.contexts)

        for message in req.contexts:
            role = message.get("role")
            content = message.get("content", "")

            # 1. 清理作为 system 消息插入的知识库内容
            # 如果 system 消息包含知识库开始标记，则认为它是知识库注入的消息，直接删除
            if role == "system" and KB_START_MARKER in content:
                logger.debug(
                    f"从历史对话中检测到并删除知识库 system 消息 (通过标记识别): {content[:100]}..."
                )
                continue  # 不将此消息添加到 cleaned_contexts，实现删除

            # 2. 清理作为 user 消息插入的知识库内容 (当使用 prepend_prompt 方法时)
            # 这种情况下，知识库内容会被包裹在标记中，并拼接在用户原始问题之前
            elif role == "user" and KB_START_MARKER in content:
                start_marker_idx = content.find(KB_START_MARKER)
                end_marker_idx = content.find(KB_END_MARKER, start_marker_idx)

                if start_marker_idx != -1 and end_marker_idx != -1:
                    # 知识库内容结束后，后面跟着的是 "用户的原始问题是：" 分隔符和原始用户问题
                    original_prompt_delimiter_idx = content.find(
                        USER_PROMPT_DELIMITER_IN_HISTORY,
                        end_marker_idx + len(KB_END_MARKER),
                    )

                    if original_prompt_delimiter_idx != -1:
                        # 提取原始用户问题部分
                        original_user_prompt = content[
                            original_prompt_delimiter_idx
                            + len(USER_PROMPT_DELIMITER_IN_HISTORY) :
                        ].strip()
                        message["content"] = (
                            original_user_prompt  # 更新消息内容为原始用户问题
                        )
                        cleaned_contexts.append(message)  # 将清理后的消息添加到列表
                        logger.debug(
                            f"从历史对话 user 消息中清理知识库标记和内容，保留原用户问题: {original_user_prompt[:100]}..."
                        )
                    else:
                        # 理论上不会发生，如果原始问题分隔符丢失，则删除该消息
                        logger.warning(
                            f"用户消息中检测到知识库标记但缺少原始用户问题分隔符，删除该消息: {content[:100]}..."
                        )
                        continue
                else:
                    # 如果找到了开始标记但没有找到结束标记，说明消息不完整或格式错误，也删除
                    logger.warning(
                        f"用户消息中检测到知识库起始标记但缺少结束标记，删除该消息: {content[:100]}..."
                    )
                    continue
            else:
                # 保留所有其他消息（例如：助手回复、工具调用结果、以及未被知识库修改的用户/系统消息）
                cleaned_contexts.append(message)

        req.contexts = cleaned_contexts
        if len(req.contexts) < initial_context_count:
            logger.info(
                f"成功从历史对话中删除了 {initial_context_count - len(req.contexts)} 条知识库补充消息。"
            )

    @filter.on_llm_request()
    async def kb_on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """
        在对话时插入知识库内容，如果有默认使用的知识库则先使用默认的，否则在日志中提示未指定知识库，对话中将不会插入知识库内容
        """
        if not await self._ensure_initialized():
            logger.warning("LLM 请求时知识库插件未初始化，跳过知识库增强。")
            return

        # 删除历史记录中的知识库补充的内容
        self._clean_contexts_from_kb_content(req)

        # 1. 获取当前会话的默认知识库
        # 首先检查用户是否通过指令明确禁用了知识库增强
        # (这需要一个额外的配置或用户偏好设置，这里暂时不实现，但可以考虑)
        # 假设有一个配置项: self.config.get("kb_llm_enhancement_enabled_by_default", True)
        # 并且用户可以通过指令临时开关

        default_collection_name = self._get_user_default_collection(event)

        # 检查知识库增强是否对当前集合启用 (可以通过用户偏好或全局配置)
        # 假设有一个用户偏好: self.user_collection_preferences.get(event.unified_msg_origin + "_enhance_enabled", True)
        # 和一个全局配置: self.config.get("enable_kb_for_llm_by_default", True)

        if not default_collection_name:
            logger.debug("未找到当前会话的默认知识库，跳过 LLM 请求增强。")
            return

        if not await self.vector_db.collection_exists(default_collection_name):
            logger.warning(
                f"用户默认知识库 '{default_collection_name}' 不存在，跳过 LLM 请求增强。"
            )
            return

        # 2. 获取配置参数
        # 从配置中获取是否启用知识库增强的全局开关
        enable_kb_enhancement = self.config.get("enable_kb_llm_enhancement", True)
        if not enable_kb_enhancement:
            logger.info("知识库对LLM请求的增强功能已全局禁用。")
            return

        # 获取用于知识库检索的 top_k 数量
        kb_search_top_k = self.config.get("kb_llm_search_top_k", 3)
        # 获取知识库内容插入方式: "prepend_prompt" 或 "system_prompt"
        kb_insertion_method = self.config.get(
            "kb_llm_insertion_method", "prepend_prompt"
        )
        # 获取知识库内容模板
        kb_context_template = self.config.get(
            "kb_llm_context_template",
            "这是相关的知识库信息，请参考这些信息来回答用户的问题：\n{retrieved_contexts}",
        )
        # 最小相关度阈值 (可选)
        min_similarity_score = self.config.get(
            "kb_llm_min_similarity_score", 0.5
        )  # 默认 0.5

        # 3. 从知识库搜索相关内容
        user_query = req.prompt  # 用户当前发送的消息
        if not user_query or not user_query.strip():
            logger.debug("用户查询为空，跳过知识库搜索。")
            return

        try:
            logger.info(
                f"为LLM请求在知识库 '{default_collection_name}' 中搜索: '{user_query[:50]}...' (top_k={kb_search_top_k})"
            )
            search_results = await self.vector_db.search(
                default_collection_name, user_query, top_k=kb_search_top_k
            )
        except Exception as e:
            logger.error(
                f"LLM 请求时从知识库 '{default_collection_name}' 搜索失败: {e}",
                exc_info=True,
            )
            return  # 搜索失败则不增强

        if not search_results:
            logger.info(
                f"在知识库 '{default_collection_name}' 中未找到与查询 '{user_query[:50]}...' 相关的内容。"
            )
            return

        # 4. 筛选和格式化知识库内容
        retrieved_contexts_list = []
        for doc, score in search_results:
            if score >= min_similarity_score:
                # 可以添加文档来源等元数据
                source_info = doc.metadata.get("source", "未知来源")
                context_item = f"- 内容: {doc.text_content} (来源: {source_info}, 相关度: {score:.2f})"
                retrieved_contexts_list.append(context_item)
            else:
                logger.debug(
                    f"文档 '{doc.text_content[:30]}...' 相关度 {score:.2f} 低于阈值 {min_similarity_score}，已忽略。"
                )

        if not retrieved_contexts_list:
            logger.info(
                f"所有检索到的知识库内容相关度均低于阈值 {min_similarity_score}，不进行增强。"
            )
            return

        formatted_contexts = "\n".join(retrieved_contexts_list)
        knowledge_to_insert = kb_context_template.format(
            retrieved_contexts=formatted_contexts
        )
        logger.debug(f"知识库检索出来的东西:{knowledge_to_insert}")
        # TODO 限制插入内容的总长度，避免超出 LLM 的 token 限制,好思想,但是不用
        max_kb_insert_length = self.config.get("kb_llm_max_insert_length", 200000)
        if len(knowledge_to_insert) > max_kb_insert_length:
            logger.warning(
                f"知识库插入内容过长 ({len(knowledge_to_insert)} chars)，将被截断至 {max_kb_insert_length} chars。"
            )
            knowledge_to_insert = (
                knowledge_to_insert[:max_kb_insert_length] + "\n... [内容已截断]"
            )

        # 包裹知识库内容
        knowledge_to_insert = (
            f"{KB_START_MARKER}\n{knowledge_to_insert}\n{KB_END_MARKER}"
        )

        # 5. 将知识库内容插入到 ProviderRequest
        if kb_insertion_method == "system_prompt":
            if req.system_prompt:
                req.system_prompt = f"{knowledge_to_insert}\n\n{req.system_prompt}"
            else:
                req.system_prompt = knowledge_to_insert
            logger.info(
                f"知识库内容已添加到 system_prompt。长度: {len(knowledge_to_insert)}"
            )

        elif kb_insertion_method == "prepend_prompt":
            req.prompt = f"{knowledge_to_insert}\n\n用户的原始问题是：\n{req.prompt}"
            logger.info(
                f"知识库内容已前置到用户 prompt。长度: {len(knowledge_to_insert)}"
            )

        else:  # 默认为 prepend_prompt 或其他自定义方式
            logger.warning(
                f"未知的知识库内容插入方式: {kb_insertion_method}，将默认前置到用户 prompt。"
            )
            req.prompt = f"{knowledge_to_insert}\n\n用户的原始问题是：\n{req.prompt}"

        logger.debug(f"修改后的 ProviderRequest.prompt: {req.prompt[:200]}...")
        if req.system_prompt:
            logger.debug(
                f"修改后的 ProviderRequest.system_prompt: {req.system_prompt[:200]}..."
            )

    # --- 指令组定义 ---
    @filter.command_group("kb", alias={"knowledge", "知识库"})
    def kb_group(self):
        """知识库管理指令集"""
        pass

    @kb_group.command("help", alias={"帮助"})
    async def kb_help(self, event: AstrMessageEvent):
        """显示知识库插件的帮助信息"""
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return

        help_text = """
知识库插件帮助：
/kb add text <内容> [知识库名] - 添加文本到知识库
/kb add file <文件路径或者下载链接> [知识库名] (目前支持.txt, .md)
/kb search <查询内容> [知识库名] [数量] - 搜索知识库
/kb list - 列出所有知识库
/kb current - 查看当前会话默认知识库
/kb use <知识库名> - 设置当前会话默认知识库
/kb create <知识库名> - 创建一个新的知识库
/kb delete <知识库名> - 删除一个知识库及其内容 (危险操作!)
/kb count [知识库名] - 查看知识库中文档数量
/kb help - 显示此帮助信息
        """.strip()
        yield event.plain_result(help_text)

    @kb_group.group("add")
    def kb_add_group(self, event: AstrMessageEvent):
        """添加内容到知识库的子指令组"""
        if not asyncio.create_task(self._ensure_initialized()):
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return
        yield event.plain_result(
            "请使用 `/kb add text <内容> [知识库名称]` 或 `/kb add file <文件路径或者下载地址> [知识库名称]"
        )

    @kb_add_group.command("text")
    async def kb_add_text(
        self,
        event: AstrMessageEvent,
        content: str,
        collection_name: Optional[str] = None,
    ):
        """
        添加文本内容到知识库。
        用法: /kb add text "这是要添加的文本内容" [可选的知识库名称]
        """
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return

        if not content.strip():
            yield event.plain_result("添加的内容不能为空。")
            return

        target_collection = (
            collection_name
            if collection_name
            else self._get_user_default_collection(event)
        )

        if self.config.get(
            "auto_create_collection", True
        ) and not await self.vector_db.collection_exists(target_collection):
            try:
                await self.vector_db.create_collection(target_collection)
                logger.info(f"知识库 '{target_collection}' 不存在，已自动创建。")
                yield event.plain_result(
                    f"知识库 '{target_collection}' 不存在，已自动创建。"
                )
            except Exception as e:
                logger.error(f"自动创建知识库 '{target_collection}' 失败: {e}")
                yield event.plain_result(
                    f"自动创建知识库 '{target_collection}' 失败: {e}"
                )
                return

        chunks = self.text_splitter.split_text(content)
        if not chunks:
            yield event.plain_result("文本分割后无有效内容。")
            return

        documents_to_add = [
            Document(
                text_content=chunk,
                metadata={"source": "direct_text", "user": event.get_sender_name()},
            )
            for chunk in chunks
        ]

        try:
            yield event.plain_result(
                f"正在处理 {len(chunks)} 个文本块并添加到知识库 '{target_collection}'..."
            )
            doc_ids = await self.vector_db.add_documents(
                target_collection, documents_to_add
            )
            if doc_ids:
                yield event.plain_result(
                    f"成功添加 {len(doc_ids)} 条知识到 '{target_collection}'。"
                )
            else:
                yield event.plain_result(
                    f"未能添加任何知识到 '{target_collection}'，请检查日志。"
                )
        except Exception as e:
            logger.error(
                f"添加文本到知识库 '{target_collection}' 失败: {e}", exc_info=True
            )
            yield event.plain_result(f"添加知识失败: {e}")

    @kb_add_group.command("file")
    async def kb_add_file(
        self,
        event: AstrMessageEvent,
        path_or_url: str,
        collection_name: Optional[str] = None,
    ):
        """
        从本地路径 (文件或文件夹) 或 URL 添加文件内容到知识库 (支持 .txt, .md)。
        用法: /kb add file <文件/文件夹路径或URL> [可选的知识库名称]
        """
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return

        if not path_or_url:
            yield event.plain_result("请输入文件/文件夹路径或 URL。")
            return

        target_collection = (
            collection_name
            if collection_name
            else self._get_user_default_collection(event)
        )

        if self.config.get(
            "auto_create_collection", True
        ) and not await self.vector_db.collection_exists(target_collection):
            try:
                await self.vector_db.create_collection(target_collection)
                logger.info(f"知识库 '{target_collection}' 不存在，已自动创建。")
                yield event.plain_result(
                    f"知识库 '{target_collection}' 不存在，已自动创建。"
                )
            except Exception as e_create:
                logger.error(f"自动创建知识库 '{target_collection}' 失败: {e_create}")
                yield event.plain_result(
                    f"自动创建知识库 '{target_collection}' 失败: {e_create}"
                )
                return

        files_to_process_info = []  # 存储 (文件路径, 原始文件名, 是否临时文件)
        is_temp_dir_created = False  # 标记是否创建了临时下载目录
        temp_download_dir_for_cleanup = None

        # 检查是 URL 还是本地路径
        try:
            parsed_uri = urlparse(path_or_url)
            is_url = all(
                [parsed_uri.scheme, parsed_uri.netloc]
            ) and parsed_uri.scheme in ["http", "https"]
        except ValueError:
            is_url = False

        if is_url:
            yield event.plain_result(f"检测到 URL，正在尝试下载: {path_or_url} ...")
            temp_download_dir = os.path.join(
                self.persistent_data_root_path, "temp_downloads"
            )
            os.makedirs(temp_download_dir, exist_ok=True)
            is_temp_dir_created = True
            temp_download_dir_for_cleanup = temp_download_dir

            downloaded_path = await self._download_file(path_or_url, temp_download_dir)
            if downloaded_path:
                files_to_process_info.append(
                    (downloaded_path, os.path.basename(downloaded_path), True)
                )
            else:
                yield event.plain_result(f"无法下载文件: {path_or_url}")
                return
        else:  # 本地路径 (文件或文件夹)
            logger.info(f"用户提供了本地路径: {path_or_url}。将检查是文件还是文件夹。")
            if not os.path.exists(path_or_url):
                yield event.plain_result(f"本地路径无效或不存在: {path_or_url}")
                return

            if os.path.isfile(path_or_url):
                files_to_process_info.append(
                    (path_or_url, os.path.basename(path_or_url), False)
                )
            elif os.path.isdir(path_or_url):
                yield event.plain_result(
                    f"检测到文件夹路径，正在遍历支持的文件: {path_or_url} ..."
                )
                supported_extensions = (".txt", ".md")  # 可配置
                found_files_count = 0
                for root, _, files in os.walk(path_or_url):
                    for filename in files:
                        if filename.lower().endswith(supported_extensions):
                            full_path = os.path.join(root, filename)
                            files_to_process_info.append((full_path, filename, False))
                            found_files_count += 1
                if not files_to_process_info:
                    yield event.plain_result(
                        f"在文件夹 '{path_or_url}' 中未找到支持的文件类型 ({', '.join(supported_extensions)})。"
                    )
                    return
                yield event.plain_result(
                    f"在文件夹中找到 {found_files_count} 个支持的文件，将开始处理。"
                )
            else:
                yield event.plain_result(
                    f"路径 '{path_or_url}' 不是有效的文件或文件夹。"
                )
                return

        if not files_to_process_info:
            yield event.plain_result("未能获取到任何要处理的文件。")
            return

        total_docs_added = 0
        total_chunks_processed = 0
        files_processed_count = 0
        error_files = []

        for file_path, original_filename, is_temp_file in files_to_process_info:
            files_processed_count += 1
            # yield event.plain_result(f"正在处理第 {files_processed_count}/{len(files_to_process_info)} 个文件: '{original_filename}'...")
            logger.debug(
                f"正在处理第 {files_processed_count}/{len(files_to_process_info)} 个文件: '{original_filename}'..."
            )
            content = await parse_file_content(file_path)
            if content is None:
                message = f"无法解析文件 '{original_filename}' 或文件为空，已跳过。"
                yield event.plain_result(message)
                error_files.append(original_filename)
                if is_temp_file:  # 清理单个下载的临时文件
                    try:
                        os.remove(file_path)
                        logger.info(f"已删除临时文件: {file_path}")
                    except OSError as e:
                        logger.error(f"删除临时文件 {file_path} 失败: {e}")
                continue

            chunks = self.text_splitter.split_text(content)
            if not chunks:
                message = f"文件 '{original_filename}' 分割后无有效内容，已跳过。"
                yield event.plain_result(message)
                error_files.append(original_filename)
                if is_temp_file:
                    try:
                        os.remove(file_path)
                        logger.info(f"已删除临时文件: {file_path}")
                    except OSError as e:
                        logger.error(f"删除临时文件 {file_path} 失败: {e}")
                continue

            total_chunks_processed += len(chunks)
            documents_to_add = [
                Document(
                    text_content=chunk,
                    metadata={
                        "source": original_filename,
                        "user": event.get_sender_name(),
                    },
                )
                for chunk in chunks
            ]

            try:
                doc_ids = await self.vector_db.add_documents(
                    target_collection, documents_to_add
                )
                if doc_ids:
                    total_docs_added += len(doc_ids)
                    # yield event.plain_result(f"成功从 '{original_filename}' 添加 {len(doc_ids)} 条知识。") # 避免过多消息
                else:
                    error_files.append(original_filename)
                    # yield event.plain_result(f"未能从 '{original_filename}' 添加任何知识。")
            except Exception as e_add:
                logger.error(
                    f"从文件 '{original_filename}' 添加知识到知识库 '{target_collection}' 失败: {e_add}",
                    exc_info=True,
                )
                yield event.plain_result(
                    f"处理文件 '{original_filename}' 时出错: {e_add}"
                )
                error_files.append(original_filename)
            finally:
                if is_temp_file:  # 清理单个下载的临时文件
                    try:
                        os.remove(file_path)
                        logger.info(f"已删除临时下载文件: {file_path}")
                    except OSError as e_rm:
                        logger.error(f"删除临时文件失败 {file_path}: {e_rm}")

        # 汇总结果
        summary_message = (
            f"文件处理完成。\n总计处理文件数: {len(files_to_process_info)}\n"
        )
        summary_message += f"成功添加知识条目数: {total_docs_added} (来自 {total_chunks_processed} 个文本块)\n"
        if error_files:
            summary_message += f"处理失败或跳过的文件 ({len(error_files)} 个): {', '.join(error_files[:5])}"
            if len(error_files) > 5:
                summary_message += "..."
        else:
            summary_message += "所有文件均成功处理完毕！"

        yield event.plain_result(summary_message)

        # 清理下载用的临时文件夹 (如果创建了且为空)
        if is_temp_dir_created and temp_download_dir_for_cleanup:
            try:
                if not os.listdir(
                    temp_download_dir_for_cleanup
                ):  # 仅当文件夹为空时删除
                    os.rmdir(temp_download_dir_for_cleanup)
                    logger.info(
                        f"已删除空的临时下载目录: {temp_download_dir_for_cleanup}"
                    )
                else:
                    logger.info(
                        f"临时下载目录 {temp_download_dir_for_cleanup} 非空，未删除。可能包含其他未处理文件。"
                    )
            except OSError as e_rmdir:
                logger.error(
                    f"删除临时下载目录 {temp_download_dir_for_cleanup} 失败: {e_rmdir}"
                )

    @kb_group.command("search", alias={"搜索", "find", "查找"})
    async def kb_search(
        self,
        event: AstrMessageEvent,
        query: str,
        collection_name: Optional[str] = None,
        top_k_str: Optional[str] = None,
    ):
        """
        在知识库中搜索内容。
        用法: /kb search "要搜索的内容" [可选的知识库名称] [可选返回数量]
        """
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return

        if not query.strip():
            yield event.plain_result("查询内容不能为空。")
            return

        target_collection = (
            collection_name
            if collection_name
            else self._get_user_default_collection(event)
        )

        if not await self.vector_db.collection_exists(target_collection):
            yield event.plain_result(f"知识库 '{target_collection}' 不存在。")
            return
        top_k = 1  # 默认返回数量
        if top_k_str is not None:  # 如果用户提供了 top_k 参数
            if isinstance(top_k_str, int):
                top_k = top_k_str
            elif isinstance(top_k_str, str):
                if top_k_str.isdigit():
                    try:
                        top_k = int(top_k_str)
                    except ValueError:
                        logger.warning(
                            f"无法将 top_k 参数 '{top_k_str}' 转换为整数，将使用默认值 {top_k}。"
                        )
                        # 可以选择在这里给用户一个提示
                        # yield event.plain_result(f"提示：返回数量参数 '{top_k_param}' 无效，已使用默认值 {top_k}。")
                else:
                    logger.warning(
                        f"top_k 参数 '{top_k_str}' 不是数字，将使用默认值 {top_k}。"
                    )
                    # yield event.plain_result(f"提示：返回数量参数 '{top_k_param}' 无效，已使用默认值 {top_k}。")
            else:  # 其他类型
                logger.warning(
                    f"top_k 参数类型未知 ('{type(top_k_str)}'), 将使用默认值 {top_k}。"
                )
                # yield event.plain_result(f"提示：返回数量参数无效，已使用默认值 {top_k}。")

        top_k = max(1, min(top_k, 30))  # 限制 top_k 范围 (例如，最多返回10条)
        logger.info(
            f"搜索知识库 '{target_collection}'，查询: '{query[:30]}...', top_k: {top_k}"
        )

        try:
            yield event.plain_result(
                f"正在知识库 '{target_collection}' 中搜索 '{query[:30]}...' (最多{top_k}条)..."
            )
            search_results = await self.vector_db.search(
                target_collection, query, top_k=top_k
            )

            if not search_results:
                yield event.plain_result(
                    f"在知识库 '{target_collection}' 中没有找到与 '{query[:30]}...' 相关的内容。"
                )
                return

            response_message = f"知识库 '{target_collection}' 中关于 '{query[:30]}...' 的搜索结果 (相关度从高到低):\n"
            for i, (doc, score) in enumerate(search_results):
                source_info = (
                    f" (来源: {doc.metadata.get('source', '未知')})"
                    if doc.metadata.get("source")
                    else ""
                )
                response_message += f"\n{i + 1}. [相关度: {score:.2f}]{source_info}\n"
                # 限制每个结果的长度，避免消息过长
                content_preview = (
                    doc.text_content[:200] + "..."
                    if len(doc.text_content) > 200
                    else doc.text_content
                )
                response_message += f"   内容: {content_preview}\n"

            # 如果结果很长，考虑分多条消息发送或使用 text_to_image
            if len(response_message) > 1500:  # 阈值可调整
                yield event.plain_result("搜索结果较长，将尝试转为图片发送。")
                img_url = await self.text_to_image(response_message)
                yield event.image_result(img_url)
            else:
                yield event.plain_result(response_message)

        except Exception as e:
            logger.error(f"搜索知识库 '{target_collection}' 失败: {e}", exc_info=True)
            yield event.plain_result(f"搜索失败: {e}")

    @kb_group.command("list", alias={"列表", "showall"})
    async def kb_list_collections(self, event: AstrMessageEvent):
        """列出所有可用的知识库"""
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return

        try:
            collections = await self.vector_db.list_collections()
            if not collections:
                yield event.plain_result("当前没有可用的知识库。")
                return

            response = "可用的知识库列表:\n"
            for col_name in collections:
                count = await self.vector_db.count_documents(col_name)
                response += f"- {col_name} (文档数: {count})\n"
            yield event.plain_result(response.strip())
        except Exception as e:
            logger.error(f"列出知识库失败: {e}", exc_info=True)
            yield event.plain_result(f"列出知识库失败: {e}")

    @kb_group.command("current", alias={"当前"})
    async def kb_current_collection(self, event: AstrMessageEvent):
        """查看当前会话的默认知识库"""
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return
        current_col = self._get_user_default_collection(event)
        yield event.plain_result(f"当前会话默认知识库为: {current_col}")

    @kb_group.command("use", alias={"使用", "set"})
    async def kb_use_collection(self, event: AstrMessageEvent, collection_name: str):
        """设置当前会话的默认知识库"""
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return

        if not collection_name:
            yield event.plain_result(
                "请输入要设置的知识库名称。用法: /kb use <知识库名>"
            )
            return

        # _set_user_default_collection 内部会 yield 消息
        async for msg_result in self._set_user_default_collection(
            event, collection_name
        ):
            yield msg_result

    @kb_group.command("create", alias={"创建"})
    async def kb_create_collection(self, event: AstrMessageEvent, collection_name: str):
        """创建一个新的知识库"""
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return

        if not collection_name:
            yield event.plain_result(
                "请输入要创建的知识库名称。用法: /kb create <知识库名>"
            )
            return

        if await self.vector_db.collection_exists(collection_name):
            yield event.plain_result(f"知识库 '{collection_name}' 已存在。")
            return

        try:
            await self.vector_db.create_collection(collection_name)
            yield event.plain_result(f"知识库 '{collection_name}' 创建成功。")
        except Exception as e:
            logger.error(f"创建知识库 '{collection_name}' 失败: {e}", exc_info=True)
            yield event.plain_result(f"创建知识库 '{collection_name}' 失败: {e}")

    @filter.permission_type(filter.PermissionType.ADMIN)  # 限制管理员使用
    @kb_group.command("delete", alias={"删除"})
    async def kb_delete_collection(self, event: AstrMessageEvent, collection_name: str):
        """删除一个知识库及其所有内容 (危险操作! 仅管理员)。"""
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return

        if not collection_name:
            yield event.plain_result(
                "请输入要删除的知识库名称。用法: /kb delete <知识库名>"
            )
            return

        if not await self.vector_db.collection_exists(collection_name):
            yield event.plain_result(f"知识库 '{collection_name}' 不存在。")
            return

        confirmation_phrase = f"确认删除{collection_name}"
        yield event.plain_result(
            f"警告：你确定要删除知识库 '{collection_name}' 及其所有内容吗？此操作不可恢复！\n"
            f"请在 60 秒内回复 '{confirmation_phrase}' 来执行。"
        )

        @session_waiter(timeout=60, record_history_chains=False)
        async def delete_confirmation_waiter(
            controller: SessionController, confirm_event: AstrMessageEvent
        ):
            user_input = confirm_event.message_str.strip()
            if user_input == confirmation_phrase:
                try:
                    await confirm_event.send(
                        confirm_event.plain_result(
                            f"正在删除知识库 '{collection_name}'..."
                        )
                    )
                    success = await self.vector_db.delete_collection(collection_name)
                    if success:
                        # 如果删除的是某些用户的默认知识库，将其重置为全局默认
                        global_default = self.config.get(
                            "default_collection_name", "general"
                        )
                        updated_prefs = False
                        # 使用 list 进行迭代复制，以便在循环中修改字典
                        for user_key, pref_col in list(
                            self.user_collection_preferences.items()
                        ):
                            if pref_col == collection_name:
                                self.user_collection_preferences[user_key] = (
                                    global_default
                                )
                                updated_prefs = True
                        if updated_prefs:
                            await self._save_user_preferences()
                            logger.info(
                                f"因知识库 '{collection_name}' 被删除，部分用户的默认知识库已重置为 '{global_default}'。"
                            )

                        await confirm_event.send(
                            confirm_event.plain_result(
                                f"知识库 '{collection_name}' 已成功删除。"
                            )
                        )
                    else:
                        await confirm_event.send(
                            confirm_event.plain_result(
                                f"删除知识库 '{collection_name}' 失败，请检查日志。"
                            )
                        )
                except Exception as e_del:
                    logger.error(
                        f"删除知识库 '{collection_name}' 过程中发生错误: {e_del}",
                        exc_info=True,
                    )
                    await confirm_event.send(
                        confirm_event.plain_result(
                            f"删除知识库 '{collection_name}' 失败: {e_del}"
                        )
                    )
                finally:
                    controller.stop()  # 结束会话
            elif user_input.lower() in ["取消", "cancel"]:
                await confirm_event.send(
                    confirm_event.plain_result(
                        f"已取消删除知识库 '{collection_name}'。"
                    )
                )
                controller.stop()
            else:
                # 非确认消息，保持会话，提示用户
                await confirm_event.send(
                    confirm_event.plain_result(
                        f"输入无效。如需删除，请回复 '{confirmation_phrase}'；如需取消，请回复 '取消'。"
                    )
                )
                controller.keep(timeout=60, reset_timeout=True)  # 重置超时

        try:
            await delete_confirmation_waiter(event)  # 启动会话等待器
        except TimeoutError:
            yield event.plain_result(
                f"删除知识库 '{collection_name}' 操作超时，已自动取消。"
            )
        except Exception as e_sess:
            logger.error(f"删除知识库确认会话发生错误: {e_sess}", exc_info=True)
            yield event.plain_result(f"删除确认过程中发生错误: {e_sess}")
        finally:
            # 确保事件停止传播，因为我们已经通过会话控制器处理了回复
            event.stop_event()

    @kb_group.command("count", alias={"数量"})
    async def kb_count_documents(
        self, event: AstrMessageEvent, collection_name: Optional[str] = None
    ):
        """查看指定知识库（或当前默认知识库）中的文档数量"""
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return

        target_collection = (
            collection_name
            if collection_name
            else self._get_user_default_collection(event)
        )

        if not await self.vector_db.collection_exists(target_collection):
            yield event.plain_result(f"知识库 '{target_collection}' 不存在。")
            return

        try:
            count = await self.vector_db.count_documents(target_collection)
            yield event.plain_result(
                f"知识库 '{target_collection}' 中包含 {count} 个文档块。"
            )
        except Exception as e:
            logger.error(
                f"获取知识库 '{target_collection}' 文档数量失败: {e}", exc_info=True
            )
            yield event.plain_result(f"获取文档数量失败: {e}")

    async def terminate(self):
        logger.info("知识库插件正在终止...")
        if hasattr(self, "init_task") and self.init_task and not self.init_task.done():
            logger.info("等待初始化任务完成...")
            try:
                await asyncio.wait_for(self.init_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("初始化任务超时，尝试取消。")
                self.init_task.cancel()
            except Exception as e:
                logger.error(f"等待初始化任务完成时出错: {e}")

        if self.embedding_util and not isinstance(self.embedding_util, Star):
            await self.embedding_util.close()
            logger.info("Embedding 工具已关闭。")
        if self.vector_db:
            await self.vector_db.close()
            logger.info("向量数据库已关闭。")
        await self._save_user_preferences()
        logger.info("知识库插件终止完成。")

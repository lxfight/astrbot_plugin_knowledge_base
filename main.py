# astrbot_plugin_knowledge_base/main.py
import os
import asyncio
from typing import Optional

from astrbot.api import logger, AstrBotConfig
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.core.utils.session_waiter import (
    session_waiter,
    SessionController,
)
from astrbot.core.config.default import VERSION
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import StarTools


from .core import constants
from .utils.installation import ensure_vector_db_dependencies
from .utils.embedding import EmbeddingUtil, EmbeddingSolutionHelper
from .utils.text_splitter import TextSplitterUtil
from .utils.file_parser import FileParser, LLM_Config
from .vector_store.base import VectorDBBase

if VERSION < "3.5.13":
    logger.info("建议升级至 AstrBot v3.5.13 或更高版本。")
    from .vector_store.faiss_store import FaissStore
else:
    from .vector_store.astrbot_faiss_store import FaissStore
from .vector_store.milvus_lite_store import MilvusLiteStore
from .vector_store.milvus_store import MilvusStore
from .web_api import KnowledgeBaseWebAPI
from .core.user_prefs_handler import UserPrefsHandler
from .core.llm_enhancer import clean_contexts_from_kb_content, enhance_request_with_kb
from .commands import (
    general_commands,
    add_commands,
    search_commands,
    manage_commands,
)


@register(
    constants.PLUGIN_REGISTER_NAME,
    "lxfight",
    "一个支持多种向量数据库的知识库插件",
    "0.5.4",
    "https://github.com/lxfight/astrbot_plugin_knowledge_base",
)
class KnowledgeBasePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._initialize_basic_paths()

        self.vector_db: Optional[VectorDBBase] = None
        self.embedding_util: Optional[EmbeddingSolutionHelper] = None
        self.text_splitter: Optional[TextSplitterUtil] = None
        self.file_parser: Optional[FileParser] = None
        self.user_prefs_handler: Optional[UserPrefsHandler] = None

        ensure_vector_db_dependencies(self.config.get("vector_db_type", "faiss"))
        self.init_task = asyncio.create_task(self._initialize_components())

    def _initialize_basic_paths(self):
        self.plugin_name_for_path = constants.PLUGIN_REGISTER_NAME
        self.persistent_data_root_path = StarTools.get_data_dir(
            self.plugin_name_for_path
        )
        os.makedirs(self.persistent_data_root_path, exist_ok=True)
        logger.info(f"知识库插件的持久化数据目录: {self.persistent_data_root_path}")
        self.user_prefs_path = os.path.join(
            self.persistent_data_root_path, "user_collection_prefs.json"
        )

    async def _initialize_components(self):
        try:
            logger.info("知识库插件开始初始化...")
            # User Preferences Handler
            self.user_prefs_handler = UserPrefsHandler(
                self.user_prefs_path, self.vector_db, self.config
            )
            await self.user_prefs_handler.load_user_preferences()

            # Embedding Util
            try:
                embedding_plugin = self.context.get_registered_star(
                    "astrbot_plugin_embedding_adapter"
                )
                if embedding_plugin:
                    embedding_util = embedding_plugin.star_cls
                    dim = embedding_util.get_dim()
                    model_name = embedding_util.get_model_name()
                    self.embedding_util = EmbeddingSolutionHelper(
                        curr_embedding_dimensions=dim,
                        curr_embedding_util=embedding_util,
                        context=self.context,
                        user_prefs_handler=self.user_prefs_handler,
                    )
                    if dim is not None and model_name is not None:
                        self.config["embedding_dimension"] = dim
                        self.config["embedding_model_name"] = model_name
                    logger.info("成功加载并使用 astrbot_plugin_embedding_adapter。")
            except Exception as e:
                logger.warning(f"嵌入服务适配器插件加载失败: {e}", exc_info=True)
                self.embedding_util = None  # Fallback

            if self.embedding_util is None:  # If adapter failed or not found
                embedding_util = EmbeddingUtil(
                    api_url=self.config.get("embedding_api_url"),
                    api_key=self.config.get("embedding_api_key"),
                    model_name=self.config.get("embedding_model_name"),
                )
                self.embedding_util = EmbeddingSolutionHelper(
                    curr_embedding_dimensions=self.config.get(
                        "embedding_dimension", 1024
                    ),
                    curr_embedding_util=embedding_util,
                    context=self.context,
                    user_prefs_handler=self.user_prefs_handler,
                )
            logger.info("Embedding 工具初始化完成。")

            # Text Splitter
            self.text_splitter = TextSplitterUtil(
                chunk_size=self.config.get("text_chunk_size"),
                chunk_overlap=self.config.get("text_chunk_overlap"),
            )
            logger.info("文本分割工具初始化完成。")

            # File Parser
            self.llm_config = LLM_Config(
                context=self.context, status=self.config.get("LLM_model")
            )
            self.file_parser = FileParser(self.llm_config)
            logger.info("文件解析器初始化完成。")

            # Vector DB
            db_type = self.config.get("vector_db_type", "faiss")

            if db_type == "faiss":
                faiss_subpath = self.config.get("faiss_db_subpath", "faiss_data")
                faiss_full_path = os.path.join(
                    self.persistent_data_root_path, faiss_subpath
                )
                self.vector_db = FaissStore(self.embedding_util, faiss_full_path)
            elif db_type == "milvus_lite":
                milvus_lite_subpath = self.config.get(
                    "milvus_lite_db_subpath", "milvus_lite_data/milvus_lite.db"
                )
                milvus_lite_full_path = os.path.join(
                    self.persistent_data_root_path, milvus_lite_subpath
                )
                os.makedirs(os.path.dirname(milvus_lite_full_path), exist_ok=True)
                self.vector_db = MilvusLiteStore(
                    self.embedding_util, milvus_lite_full_path
                )
            elif db_type == "milvus":
                self.vector_db = MilvusStore(
                    self.embedding_util,
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

            self.user_prefs_handler.vector_db = self.vector_db

            # Web API
            try:
                self.web_api = KnowledgeBaseWebAPI(
                    vec_db=self.vector_db,
                    text_splitter=self.text_splitter,
                    astrbot_context=self.context,
                    llm_config=self.llm_config,
                    user_prefs_handler=self.user_prefs_handler,
                    plugin_config=self.config,
                )
            except Exception as e:
                logger.warning(
                    f"知识库 WebAPI 初始化失败，可能导致无法在 WebUI 操作知识库。原因：{e}",
                    exc_info=True,
                )

            logger.info("知识库插件初始化成功。")

        except Exception as e:
            print("出现问题")
            logger.error(f"知识库插件初始化失败: {e}", exc_info=True)
            self.vector_db = None

    async def _ensure_initialized(self) -> bool:
        if self.init_task and not self.init_task.done():
            await self.init_task
        if (
            not self.vector_db
            or not self.embedding_util
            or not self.text_splitter
            or not self.user_prefs_handler
        ):
            logger.error("知识库插件未正确初始化，请检查日志和配置。")
            return False
        return True

    # --- LLM Request Hook ---
    @filter.on_llm_request()
    async def kb_on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        if not await self._ensure_initialized():
            logger.warning("LLM 请求时知识库插件未初始化，跳过知识库增强。")
            return

        clean_contexts_from_kb_content(req)

        await enhance_request_with_kb(
            event, req, self.vector_db, self.user_prefs_handler, self.config
        )

    # --- Command Groups & Commands ---
    @filter.command_group("kb", alias={"knowledge", "知识库"})
    def kb_group(self):
        """知识库管理指令集"""
        pass

    @kb_group.command("help", alias={"帮助"})
    async def kb_help(self, event: AstrMessageEvent):
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return
        async for result in general_commands.handle_kb_help(self, event):
            yield result

    @kb_group.group("add")
    def kb_add_group(self, event: AstrMessageEvent):
        """添加内容到知识库的子指令组"""
        pass

    @kb_add_group.command("text")
    async def kb_add_text(
        self,
        event: AstrMessageEvent,
        content: str,
        collection_name: Optional[str] = None,
    ):
        """添加文本内容到知识库。"""
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return
        async for result in add_commands.handle_add_text(
            self, event, content, collection_name
        ):
            yield result

    @kb_add_group.command("file")
    async def kb_add_file(
        self,
        event: AstrMessageEvent,
        path_or_url: str,
        collection_name: Optional[str] = None,
    ):
        """从本地路径或 URL 添加文件内容到知识库。"""
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return
        async for result in add_commands.handle_add_file(
            self, event, path_or_url, collection_name
        ):
            yield result

    @kb_group.command("search", alias={"搜索", "find", "查找"})
    async def kb_search(
        self,
        event: AstrMessageEvent,
        query: str,
        top_k_str: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        """在知识库中搜索内容。"""
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return
        async for result in search_commands.handle_search(
            self, event, query, collection_name, top_k_str
        ):
            yield result

    @kb_group.command("list", alias={"列表", "showall"})
    async def kb_list_collections(self, event: AstrMessageEvent):
        """列出所有可用的知识库"""
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return
        async for result in manage_commands.handle_list_collections(self, event):
            yield result

    @kb_group.command("current", alias={"当前"})
    async def kb_current_collection(self, event: AstrMessageEvent):
        """查看当前会话的默认知识库"""
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return
        async for result in general_commands.handle_kb_current_collection(self, event):
            yield result

    @kb_group.command("use", alias={"使用", "set"})
    async def kb_use_collection(self, event: AstrMessageEvent, collection_name: str):
        """设置当前会话的默认知识库"""
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return
        async for result in general_commands.handle_kb_use_collection(
            self, event, collection_name
        ):
            yield result

    @kb_group.command("clear_use")
    async def kb_clear_use_collection(self, event: AstrMessageEvent):
        """清除默认使用的知识库，并关闭RAG知识库补充功能"""
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return
        try:
            await self.user_prefs_handler.clear_user_collection_pref(event)
            yield event.plain_result("已清除默认知识库，并关闭RAG知识库补充功能。")
        except Exception as e:
            logger.error(f"清除默认知识库时发生错误: {e}", exc_info=True)
            yield event.plain_result(f"清除默认知识库失败: {e}")

    @kb_group.command("create", alias={"创建"})
    async def kb_create_collection(self, event: AstrMessageEvent, collection_name: str):
        """创建一个新的知识库"""
        if VERSION >= "3.5.13":
            yield event.plain_result("请在 WebUI 中使用知识库创建功能。")
            return
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return
        async for result in manage_commands.handle_create_collection(
            self, event, collection_name
        ):
            yield result

    @filter.permission_type(filter.PermissionType.ADMIN)
    @kb_group.command("delete", alias={"删除"})
    async def kb_delete_collection(
        self,
        event: AstrMessageEvent,
        collection_name: str,
        confirm: Optional[str] = None,
    ):
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

        if event.is_private_chat():
            if confirm != "--confirm":
                yield event.plain_result(
                    f"⚠️ 操作确认 ⚠️\n"
                    f"此操作将永久删除知识库 '{collection_name}' 及其包含的所有数据！此操作无法撤销！\n"
                    f"当前处于私聊环境，指令与群聊中有所不同。\n\n"
                    f"如果您确定要继续，请再次执行命令并添加 `--confirm` 参数:\n"
                    f"`/kb delete {collection_name} --confirm`"
                )
                return
            # 私聊中删除
            await manage_commands.handle_delete_collection_logic(
                self, event, collection_name
            )
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
                # Call the handler logic
                await manage_commands.handle_delete_collection_logic(
                    self, confirm_event, collection_name
                )
                controller.stop()
            elif user_input.lower() in ["取消", "cancel"]:
                await confirm_event.send(
                    confirm_event.plain_result(
                        f"已取消删除知识库 '{collection_name}'。"
                    )
                )
                controller.stop()
            else:
                await confirm_event.send(
                    confirm_event.plain_result(
                        f"输入无效。如需删除，请回复 '{confirmation_phrase}'；如需取消，请回复 '取消'。"
                    )
                )
                controller.keep(timeout=60, reset_timeout=True)

        try:
            await delete_confirmation_waiter(event)
        except TimeoutError:
            yield event.plain_result(
                f"删除知识库 '{collection_name}' 操作超时，已自动取消。"
            )
        except Exception as e_sess:
            logger.error(f"删除知识库确认会话发生错误: {e_sess}", exc_info=True)
            yield event.plain_result(f"删除确认过程中发生错误: {e_sess}")
        finally:
            event.stop_event()

    @kb_group.command("count", alias={"数量"})
    async def kb_count_documents(
        self, event: AstrMessageEvent, collection_name: Optional[str] = None
    ):
        """查看指定知识库的文档数量"""
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return
        async for result in manage_commands.handle_count_documents(
            self, event, collection_name
        ):
            yield result

    @filter.permission_type(filter.PermissionType.ADMIN)
    @kb_group.command("migrate", alias={"迁移"})
    async def kb_faiss_migrate(self, event: AstrMessageEvent):
        """迁移旧的 .docs 文件到新的向量数据库格式"""
        if self.config.get("vector_db_type", "faiss") != "faiss":
            yield event.plain_result(
                "当前配置的向量数据库类型不是 Faiss，迁移操作仅适用于 Faiss 数据库。"
            )
            return
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return
        try:
            data_path = self.persistent_data_root_path
            await manage_commands.handle_migrate_files(self, event, data_path)
            if self.vector_db:
                await self.vector_db.initialize()
            yield event.plain_result(
                "迁移操作已完成。请使用/kb list命令以确认是否成功。"
            )
        except Exception as e:
            logger.error(f"迁移过程中发生错误: {e}", exc_info=True)
            yield event.plain_result(f"迁移失败: {e}")

    # --- Termination ---
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

        if (
            self.embedding_util
            and hasattr(self.embedding_util, "close")
            and not isinstance(self.embedding_util, Star)
        ):
            await self.embedding_util.close()
            logger.info("Embedding 工具已关闭。")

        if self.vector_db:
            await self.vector_db.close()
            logger.info("向量数据库已关闭。")

        if self.user_prefs_handler:
            await self.user_prefs_handler.save_user_preferences()

        logger.info("知识库插件终止完成。")
